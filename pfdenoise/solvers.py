"""solvers.py — 求解器库：经典滤波 + 相场族

分三组
------
1) 经典滤波基线（必须有的对照组）
     identity    不动
     laplacian   拉普拉斯（umbrella）平滑 —— 会收缩，是最朴素的「过度平滑」对照
     bilateral   迭代双边滤波（法向距离 + 空间距离），特征保持的经典基线

2) 相场族（本包的重点，课题组方法的可复现基线）
     ac          Allen–Cahn：曲率驱动平滑，ε 控制界面宽度
     ac_fidelity AC + 保真项 λ(f−φ)：ε 与 λ 对抗 accuracy / volume
     ac_barrier  AC + 边缘指示函数屏障 g=1/(1+|∇(G_σ*f)|^α)：沿论文①的路线
     cahn_hilliard 守恒型：∫φ 守恒，体积自动保持

3) 说明：相场求解器统一走「体素化 + 显式格式」路线。
   关于时间步，这是本包最重要的一条经验（也是最容易踩的坑）：

       PDE 写成  ε²Δφ  时   →   Δt = cfl·h²/ε²      （稳定条件 Δt·ε²·(4 或 6)/h² ≤ 1）
       PDE 写成  Δφ     时   →   Δt = cfl·h²/2

       很多论文印的 Δt = 0.05h² 对应的是**没有** ε² 前缀的写法。
       照抄到 ε²Δφ 上会让界面几百步只动 0.01，现象是「程序在跑但什么都不发生」。
       每个相场求解器都返回它用的 Δt 和 CFL，务必检查。
"""

import numpy as np

from .noise import _NN

__all__ = ["SOLVERS", "get_solver", "identity", "laplacian", "bilateral",
           "phase_field_ac", "phase_field_ac_fidelity", "phase_field_ac_barrier",
           "cahn_hilliard"]


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------
def _voxelize(P, n_vox, lo, hi):
    """把点云体素化成 0/1 占据场。返回 (occ, h)。"""
    h = float(np.mean(hi - lo) / n_vox)
    idx = np.clip(np.floor((P - lo) / h).astype(np.int64), 0, n_vox - 1)
    occ = np.zeros((n_vox, n_vox, n_vox), dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ, h


def _neumann(a):
    return np.pad(a, 1, mode="edge")


def _laplacian(u, h):
    p = _neumann(u)
    return (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
            + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]
            - 6.0 * u) / (h * h)


def _grad_mag(u, h):
    p = _neumann(u)
    gx = (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2 * h)
    gy = (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2 * h)
    gz = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2 * h)
    return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


def _smooth3(u, sigma_cells, passes=2):
    """小规模 3D 高斯平滑（可分离，够用即可）。"""
    if sigma_cells <= 0:
        return u.copy()
    r = max(1, int(np.ceil(2 * sigma_cells)))
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma_cells ** 2))
    k /= k.sum()
    out = u.astype(np.float64)
    for _ in range(passes):
        for ax in range(3):
            out = np.moveaxis(out, ax, -1)
            p = np.pad(out, [(0, 0)] * (out.ndim - 1) + [(r, r)], mode="edge")
            win = np.lib.stride_tricks.sliding_window_view(p, 2 * r + 1, axis=-1)
            out = np.tensordot(win, k, axes=([-1], [0]))
            out = np.moveaxis(out, -1, ax)
    return out


def _extract_surface(phi, h, lo, P, N, min_frac=0.05):
    """把相场 φ 提取回点集。

    做法：取 φ>0 的体素中心，做一次沿 φ 的亚格点重心修正（等值面在格子里的
    重心比格心更准），再把点数重采样回输入规模，这样 Chamfer 类指标才与
    其它方法可比。

    安全阀：如果 φ>0 只剩不到 min_frac 的目标体素，说明相塌缩了（通常是
    体素过粗或 CFL 超限）。这里**直接报错**而不是静默返回输入点——
    静默兜底会让求解器「看起来在跑、指标却和不处理一模一样」，极难排查。
    """
    occ = phi > 0
    if occ.sum() < max(4, min_frac * (occ.size ** (2 / 3))):
        raise RuntimeError(
            f"相场正相塌缩（phi>0 只有 {int(occ.sum())} 个体素，phi 范围 "
            f"[{phi.min():.3f},{phi.max():.3f}]）。"
            f"常见原因：体素过粗（点云只有 1 格厚，扛不住扩散）→ 调大 n_vox；"
            f"或 CFL 过大 → 调小 eps_h。")
    idx = np.argwhere(occ)
    pts = lo + (idx + 0.5) * h
    # 亚格点修正：等值面在 phi 栅格里的重心，相对「voxel 中心集合」的重心做偏移。
    # ⚠ 两边都必须是**索引坐标**。早先写成 (cen - pts.mean(0)/h - 0.5)*h，
    # 而 pts 里含 lo（≈ -1.1），pts.mean(0)/h 并不是索引坐标，
    # 结果平移了约 lo 的量级 → CD 直接变成 1.1（等于整个包围盒），完全错。
    w = np.clip(phi, 0, None)
    s = w.sum()
    if s > 0:
        cc = np.indices(w.shape).reshape(3, -1)
        cen_idx = (w.reshape(-1) @ cc.T) / s                 # 索引坐标下的重心
        mean_idx = (pts.mean(0) - lo.mean(0)) / h - 0.5      # 体素中心集合的索引重心
        pts = pts + (cen_idx - mean_idx) * h
    target = len(P)
    if len(pts) > target:
        sel = np.linspace(0, len(pts) - 1, target).astype(np.int64)
        pts = pts[sel]
    elif len(pts) < target:
        rng = np.random.default_rng(0)
        extra = rng.normal(0, 1e-3, (target - len(pts), 3)) + pts.mean(0)
        pts = np.vstack([pts, extra])
    return pts


# --------------------------------------------------------------------------
# 经典基线
# --------------------------------------------------------------------------
def identity(P, N, **kw):
    return P.copy()


def laplacian(P, N, iters=10, lam=0.5, **kw):
    """拉普拉斯平滑 P ← P + λ(P_nb − P)，nb 为最近邻。

    注意必须取**排除自身**的最近邻：直接查 k=1 会永远查到自己，
    更新量恒为 0，算法看起来在跑但什么都不发生。
    """
    nn = _NN(P)
    out = P.copy()
    for _ in range(iters):
        _, idx = nn.query(out, 2)
        out = out + lam * (out[idx[:, 1]] - out)
    return out


def bilateral(P, N, iters=8, sigma_s=0.02, sigma_n=0.3, nn=None,
              max_points=8000, **kw):
    """迭代双边滤波：空间距离 + 法向夹角双权重，保特征。

    ⚠ 性能：_bilateral_once 是 Python 逐点循环，复杂度约 O(n·27)。
      实测 6000 点约 3 s，20000 点要几分钟。所以超过 max_points 时直接报错
      而不是让人干等——想跑大点云请改用矢量化的相场求解器。
    """
    if len(P) > max_points:
        raise RuntimeError(
            f"bilateral 是 Python 逐点循环，{len(P)} 点会非常慢"
            f"（上限 {max_points}）。要么减小点数，要么改用 pf_* 相场求解器。")
    nn = nn or _NN(P)
    P0, N0 = P.copy(), N.copy()
    out = P.copy()
    curN = N.copy()
    for t in range(iters):
        r = max(2, int(round((2 + 3.5 * t) * sigma_s
                             / max(np.abs(P0).max(), 1e-9) * len(P) ** (1 / 3))))
        out = _bilateral_once(out, curN, r, sigma_s, sigma_n)
        _, idx = nn.query(out, 2)          # 排除自身
        curN = out - out[idx[:, 1]]         # 更新法向（邻域切向）
        nrm = np.linalg.norm(curN, axis=1, keepdims=True)
        curN = curN / np.maximum(nrm, 1e-12)
    return out


def _bilateral_once(P, N, r, sigma_s, sigma_n):
    """用网格哈希做各向异性加权邻域平均（只取 |Δ| 沿法向小且空间近的点）。"""
    cell = max(sigma_s * 1.5, 1e-9)
    lo = P.min(0) - r * cell
    key = np.floor((P - lo) / cell).astype(np.int64)
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, k in enumerate(map(tuple, key)):
        buckets[k].append(i)
    offs = list(np.ndindex(3, 3, 3))
    out = np.empty_like(P)
    wsum = np.empty(len(P))
    for i in range(len(P)):
        kx, ky, kz = key[i]
        acc = np.zeros(3)
        wsum[i] = 0.0
        for ox, oy, oz in offs:
            b = buckets.get((kx + ox - 1, ky + oy - 1, kz + oz - 1))
            if not b:
                continue
            b = np.asarray(b)
            dP = P[b] - P[i]
            ds = np.exp(-np.sum(dP ** 2, axis=1) / (2 * sigma_s ** 2))
            dn = np.abs(dP @ N[i])
            dn = np.exp(-(dn ** 2) / (2 * sigma_n ** 2))
            w = ds * dn
            acc += (w[:, None] * P[b]).sum(0)
            wsum[i] += w.sum()
        out[i] = acc / max(wsum[i], 1e-12)
    return out


def _dilate(occ, iters=1):
    """3D 二值膨胀（26 邻域，零填充）。用移位取或实现，无需 scipy。"""
    if iters <= 0:
        return occ.copy()
    n = occ.shape[0]
    out = occ.copy()
    shifts = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                dst = [slice(None)] * 3
                src = [slice(None)] * 3
                for ax, d in ((0, dx), (1, dy), (2, dz)):
                    if d == 1:
                        dst[ax] = slice(0, n - 1); src[ax] = slice(1, n)
                    elif d == -1:
                        dst[ax] = slice(1, n); src[ax] = slice(0, n - 1)
                shifts.append((tuple(dst), tuple(src)))
    for _ in range(iters):
        new = out.copy()
        for dst, src in shifts:
            new[dst] |= out[src]
        out = new
    return out


def _target_field(occ, h, grow=1, smooth=0.5, w=0.35):
    """把 0/1 占据场变成**宽度可分辨**的相场目标 T ∈ (−1,1)。

    为什么必须这一步（这是本包最容易踩的坑）
    ------------------------------------------
    1 格厚的 ±1 壳扛不住扩散：壳上格点的 6 邻域几乎全是 −1，
    而双阱项在 φ=±1 处恰好取 0（没有回复力），于是 + 相会在几十步内被整个抹光。
    实测 40 步后 φ 的范围是 [−1.00, −0.86]，正相完全消失。

    解法（按代价从低到高）：
      1. 保证 h ≈ 点间距/3，使壳本身就有 3 格厚（由 auto_resolution 负责，最优先）；
      2. 仍然不够厚时再做 **膨胀** grow 格；
      3. 最后用小 σ 高斯 + **较宽的 tanh** 软化，得到宽度 ≈ 2σ 的可分辨界面。

    w 不能太小：w=0.2 而 σ=0.8 时，壳边缘的 S 会被平滑到 0.5 以下，
    tanh 在壳**内部**就变号，正区被掏空（这是调试期真实踩到的坑）。
    """
    o = _dilate(occ, grow) if grow > 0 else occ
    S = _smooth3(o.astype(np.float64), smooth)
    return np.tanh((S - 0.5) / w)


def auto_resolution(P, nn_mult=2.2, n_vox_max=96, verbose=False):
    """**从点间距自动定体素尺寸**，用中位最近邻距离而不是包围盒面积公式。

    为什么必须这样（这是本包调试期最耗时的一块）
    ----------------------------------------------
    体素化相场的成败取决于 h 与点间距的比值。实测（bumpy 曲面，3000 点）：

        h        n_vox   占据体素   6-邻域连通率   Smax(grow=0/1)
        2.0·sp     35      2135       0.90        0.80 / 1.00
        3.0·sp     24      1541       0.99        0.87 / 1.00
        5.0·sp     14       694       1.00        0.92 / 1.00

      · h ≲ 1.5·sp  → 每格约 0.3 个点，壳是**孤立散点**，不连通；
                      膨胀补连通会让正区比原占据大十几倍，几何完全走样。
      · h ≈ 2~3·sp  → 壳连通、Smax 接近 1，目标场才有真正的正相。

    早先版本用包围盒估面积推 h，误差约 5 倍，导致所有相场求解器塌缩。
    而塌缩的症状极具迷惑性：正相被扩散吃光后，_extract_surface 静默返回输入点，
    于是指标和「不做处理」完全一致，看起来像「相场没用」而不是「分辨率不对」。

    ⚠ 硬约束，请务必知道：**n_vox ∝ √n**。
      要更细的网格必须给更多点（n ≈ A/h²）。所以本包把 `resolution_floor = h/2`
      作为指标一起报出来 —— 任何小于它的 CD/P2M 都只是体素量化下限，
      不是求解器的真实精度。这一点在解读表格时不能忽略。
    """
    nn = _NN(P)
    d, _ = nn.query(P, 2)
    sp = float(np.median(d[:, 1])) if len(P) > 2 else float(np.abs(P).max())
    sp = max(sp, 1e-12)
    h = nn_mult * sp
    ext = P.max(0) - P.min(0)
    n_vox = int(np.ceil(np.mean(ext) / h))
    if n_vox > n_vox_max:
        if verbose:
            print(f"  自动分辨率需要 n_vox={n_vox}，截到上限 {n_vox_max}")
        n_vox = n_vox_max
    h = float(np.mean(ext) / n_vox)
    if verbose:
        print(f"  点间距≈{sp:.5f} -> n_vox={n_vox}, h={h:.5f}, "
              f"间距/h={sp/h:.2f}, 体素量化下限≈{h/2:.5f}")
    return n_vox, h


def _prep(P, grow=1, smooth=0.5, w=0.4, nn_mult=2.2, n_vox=None,
          n_vox_max=96, verbose=False):
    """相场求解器公共准备：自动定分辨率 -> 体素化 -> 构造可分辨的目标场 T。"""
    lo, hi = P.min(0) - 1e-6, P.max(0) + 1e-6
    if n_vox is None:
        n_vox, _ = auto_resolution(P, nn_mult, n_vox_max, verbose)
    occ, h = _voxelize(P, n_vox, lo, hi)
    if verbose:
        print(f"  体素: h={h:.5f}, 占据体素={int(occ.sum())}, "
              f"点/体素={len(P)/max(occ.sum(),1):.2f}")
    T = _target_field(occ, h, grow=grow, smooth=smooth, w=w)
    if verbose:
        print(f"  目标场 T: T>0 占 {int((T>0).sum())} / 占据 {int(occ.sum())} "
              f"(ratio={float((T>0).sum())/max(occ.sum(),1):.2f})")
    return occ, h, T


# --------------------------------------------------------------------------
# 相场族
# --------------------------------------------------------------------------
def _run_ac(occ, h, iters, eps, lam, T=None, g=None, R=None, cfl=1.0 / 12.0,
            well=True, track=None):
    """相场通用推进。occ 不参与演化（只为调用方保留位置），目标由 T 给出。"""
    eps2 = eps * eps
    # 时间步：带 eps^2 前缀 => dt = cfl*h^2/eps^2；3D 用 6 邻域，故 cfl 上限 1/6
    dt = cfl * h * h / eps2
    cfl_actual = 6.0 * dt * eps2 / (h * h)
    if cfl_actual > 1.0 + 1e-9:
        raise ValueError(f"CFL={cfl_actual:.2f} > 1，3D 相场需 cfl <= 1/(6*(eps/h)^2)"
                         f"= {1.0/(6.0*(eps/h)**2):.4f}")
    if T is None:
        T = occ.astype(np.float64)
        T = 2.0 * T - 1.0
    phi = T.copy()
    for it in range(iters):
        lap = _laplacian(phi, h)
        r = -phi * (phi ** 2 - 1.0) / 2.0 if well else 0.0
        acc = g if g is not None else 1.0
        Rm = R if R is not None else 1.0
        phi = phi + dt * (eps2 * lap + acc * Rm * (r + lam * (T - phi)))
        if not np.isfinite(phi).all():
            raise RuntimeError(f"第 {it} 步发散：检查 cfl / eps / lam")
        if track is not None:
            track.append((dt, cfl_actual))
    return phi, dt, cfl_actual


def phase_field_ac(P, N, iters=150, eps_h=2.0, lam=0.5, grow=1, smooth=0.5,
                   w=0.4, nn_mult=2.2, n_vox=None, n_vox_max=96,
                   verbose=False, **kw):
    """Allen–Cahn + 弱保真项：曲率驱动平滑，靠 λ 抵抗毛细收缩。

    ⚠ 纯 AC（λ=0）会**腐蚀**有限大小的正相：它是双阱势能的梯度流，
    有限液滴受毛细压力必然收缩到零测度。所以作为去噪器必须给一点保真项。
    保留 lam=0 的能力是为了演示这个物理事实（`--lam-g 0`）。
    """
    occ, h, T = _prep(P, grow, smooth, w, nn_mult, n_vox, n_vox_max, verbose)
    phi, dt, cfl = _run_ac(None, h, iters, eps_h * h, lam, T=T)
    return _extract_surface(phi, h, P.min(0) - 1e-6, P, N), {"dt": dt, "cfl": cfl}


def phase_field_ac_fidelity(P, N, iters=150, eps_h=2.0, lam=0.8, grow=1, smooth=0.5,
                            w=0.4, nn_mult=2.2, n_vox=None,
                            n_vox_max=96, verbose=False, **kw):
    """AC + 保真项 λ(T−φ)。ε 与 λ 对抗 accuracy / volume，这是本包的核心基线。"""
    occ, h, T = _prep(P, grow, smooth, w, nn_mult, n_vox, n_vox_max, verbose)
    phi, dt, cfl = _run_ac(None, h, iters, eps_h * h, lam, T=T)
    return _extract_surface(phi, h, P.min(0) - 1e-6, P, N), {"dt": dt, "cfl": cfl}


def phase_field_ac_barrier(P, N, iters=150, eps_h=2.0, lam=1.0,
                           sigma_h=3.0, alpha=2.0, grow=1, smooth=0.5, w=0.4,
                           nn_mult=2.2, n_vox=None, n_vox_max=96,
                           verbose=False, **kw):
    """AC + 边缘指示函数屏障 g = 1/(1+|∇(G_σ*T)|^α)（论文①路线）。

    g 在平坦区≈1、边界处≈0，像一堵墙把界面钉在数据附近，
    因此比纯 AC 更能保住「贴合度」，但需要更大 λ 才推动得动。
    """
    occ, h, T = _prep(P, grow, smooth, w, nn_mult, n_vox, n_vox_max, verbose)
    S = _smooth3(T, sigma_h)
    g = 1.0 / (1.0 + _grad_mag(S, h) ** alpha)
    R = occ.astype(np.float64)
    phi, dt, cfl = _run_ac(None, h, iters, eps_h * h, lam, T=T, g=g, R=R)
    return _extract_surface(phi, h, P.min(0) - 1e-6, P, N), {"dt": dt, "cfl": cfl}


def cahn_hilliard(P, N, iters=300, eps_h=2.0, lam=0.8, mob=1.0,
                  grow=1, smooth=0.5, w=0.4, nn_mult=2.2,
                  n_vox=None, n_vox_max=96, verbose=False, **kw):
    """Cahn–Hilliard（守恒型）：体积自动保持，代价是欠平滑。

        dφ/dt = -M ∇²( ε²∇²φ − dV/dφ ) + λ(T − φ)
        V(φ) = (φ²−1)²/4  ⇒  dV/dφ = φ(φ²−1)

    CH 满足 d/dt∫φ = 0，所以「保体积」是格式自带的保证，不依赖调参。

    ⚠ 时间步与 AC **不一样**，这是 CH 实现最常见的发散原因：
    化学势 μ 里有 Δφ，于是 μ 外面又套一层 Δ，格式实际含 **四阶**算子
    ε²∇⁴φ。显式格式对它的限制是  Δt·M·ε²·6/h⁴ ≤ 1  ⇒  Δt ≤ h⁴/(6Mε²)，
    比 AC 的 h²/ε² **严格得多**。直接套 AC 的 Δt 会在第 3~5 步就炸掉
    （实测确实如此）。所以这里单独算，并默认用较保守的系数。
    """
    occ, h, T = _prep(P, grow, smooth, w, nn_mult, n_vox, n_vox_max, verbose)
    eps2 = (eps_h * h) ** 2
    # 四阶算子：dt <= h^4 / (6 * mob * eps^2)；再乘安全系数 ch_cfl。
    # ch_cfl 必须小：实测 0.5 会在第 7 步发散，0.05 才稳定。
    # 这也说明为什么组里近年的 CH 论文几乎都改用 BDF2/SAV/ES-SAV 这类隐式或
    # 标量辅助变量格式 —— 纯显式 CH 的刚性实在太难压。
    ch_cfl = 0.05
    dt = ch_cfl * h ** 4 / (6.0 * max(mob, 1e-12) * eps2)
    # 显式报告 CH 的真实扩散数（不是 AC 那个）
    cfl_actual = 6.0 * dt * mob * eps2 / (h ** 4)
    phi = T.copy()
    for it in range(iters):
        mu = eps2 * _laplacian(phi, h) - phi * (phi ** 2 - 1.0)
        lap_mu = _laplacian(mu, h)
        phi = phi + dt * (-mob * lap_mu + lam * (T - phi))
        if not np.isfinite(phi).all():
            raise RuntimeError(f"Cahn-Hilliard 第 {it} 步发散："
                               f"四阶算子对 Δt 极敏感，可减小 eps_h 或增大 n_vox")
    return _extract_surface(phi, h, P.min(0) - 1e-6, P, N), {"dt": dt, "cfl": cfl_actual}


SOLVERS = {
    "none":          identity,
    "laplacian":     laplacian,
    "bilateral":     bilateral,
    "pf_ac":         phase_field_ac,
    "pf_ac_fidelity": phase_field_ac_fidelity,
    "pf_ac_barrier": phase_field_ac_barrier,
    "cahn_hilliard": cahn_hilliard,
}

DESCRIPTIONS = {
    "none": "不做处理（噪声基线）",
    "laplacian": "拉普拉斯平滑，λ=0.5×10 轮。经典收缩基线",
    "bilateral": "迭代双边滤波，空间+法向双权重，特征保持基线（Python 逐点循环，>8000 点会拒绝）",
    "pf_ac": "AC + 弱保真项 λ=0.5，ε=2h。λ 调小可见毛细腐蚀（λ=0 时液滴收缩到消失）",
    "pf_ac_fidelity": "AC + 保真项 λ=0.8，ε=2h。核心基线，accuracy/volume 权衡",
    "pf_ac_barrier": "AC + 边缘指示函数屏障（论文①路线），σ=3h, α=2, λ=1.0",
    "cahn_hilliard": "CH 守恒型，ε=2h, λ=0.8, M=1，ch_cfl=0.05。体积是格式级保证",
}


def get_solver(name):
    if name not in SOLVERS:
        raise ValueError(f"未知求解器 {name!r}，可选：{list(SOLVERS)}")
    return SOLVERS[name]
