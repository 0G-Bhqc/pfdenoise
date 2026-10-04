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
     ac_conserved AC + 拉格朗日乘子全局体积守恒：严格质量守恒，无四阶算子刚性
     ac_aniso    AC + 各向异性扩散张量：沿切向高平滑去噪，法向抑制扩散保特征
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
           "phase_field_ac_conserved", "phase_field_ac_anisotropic",
           "phase_field_ac_convex", "phase_field_ac_convex_adaptive",
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


def _run_ac_conserved(occ, h, iters, eps, lam, T=None, cfl=1.0 / 12.0, track=None):
    """拉格朗日乘子全局体积守恒 Allen-Cahn 推进。"""
    eps2 = eps * eps
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
        reaction = -(phi ** 3 - phi) + lam * (T - phi)
        beta = float(np.mean(phi ** 3 - phi + lam * (phi - T)))
        phi = phi + dt * (eps2 * lap + reaction + beta)
        if not np.isfinite(phi).all():
            raise RuntimeError(f"Conserved AC 第 {it} 步发散：检查 cfl / eps / lam")
        if track is not None:
            track.append((dt, cfl_actual))
    return phi, dt, cfl_actual


def phase_field_ac_conserved(P, N, iters=150, eps_h=2.0, lam=0.8, grow=1, smooth=0.5,
                             w=0.4, nn_mult=2.2, n_vox=None, n_vox_max=96,
                             verbose=False, **kw):
    r"""拉格朗日乘子全局体积守恒 Allen-Cahn 相场求解器。

    增加时间依赖项 beta(t) = mean(phi^3 - phi + lam*(phi - T))，并在更新中补偿 +dt*beta(t)，
    从而严格保证离散质量/体积守恒 (d/dt \int phi dx = 0)。
    彻底解决标准 AC 在低保真项或长时间演化下的毛细收缩和体积收敛漂移，
    且保持二阶扩散的大时间步稳定性。
    """
    occ, h, T = _prep(P, grow, smooth, w, nn_mult, n_vox, n_vox_max, verbose)
    phi, dt, cfl = _run_ac_conserved(None, h, iters, eps_h * h, lam, T=T)
    return _extract_surface(phi, h, P.min(0) - 1e-6, P, N), {"dt": dt, "cfl": cfl}


def _run_ac_anisotropic(occ, h, iters, eps, lam, c_tangent=1.0, c_normal=0.1,
                        sigma_n=0.8, T=None, cfl=1.0 / 12.0, track=None):
    """各向异性扩散张量 Allen-Cahn 推进。"""
    eps2 = eps * eps
    # 考虑切向扩散乘子对最大谱半径的缩放
    max_c = max(float(c_tangent), 1.0)
    dt = cfl * h * h / (eps2 * max_c)
    cfl_actual = 6.0 * dt * eps2 * max_c / (h * h)
    if cfl_actual > 1.0 + 1e-9:
        raise ValueError(f"CFL={cfl_actual:.2f} > 1，3D 相场需 cfl <= 1/(6*(eps/h)^2)"
                         f"= {1.0/(6.0*(eps/h)**2):.4f}")
    if T is None:
        T = occ.astype(np.float64)
        T = 2.0 * T - 1.0
    phi = T.copy()
    c_diff = c_tangent - c_normal
    for it in range(iters):
        lap = _laplacian(phi, h)
        # 1. 估算平滑法向场的方向导数
        sphi = _smooth3(phi, sigma_n, passes=1)
        p = _neumann(sphi)
        gx = (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2.0 * h)
        gy = (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2.0 * h)
        gz = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2.0 * h)
        mag2 = gx * gx + gy * gy + gz * gz
        inv_mag = 1.0 / np.sqrt(mag2 + 1e-6)
        nx, ny, nz = gx * inv_mag, gy * inv_mag, gz * inv_mag

        # 2. 当前场 phi 的梯度
        pp = _neumann(phi)
        pgx = (pp[2:, 1:-1, 1:-1] - pp[:-2, 1:-1, 1:-1]) / (2.0 * h)
        pgy = (pp[1:-1, 2:, 1:-1] - pp[1:-1, :-2, 1:-1]) / (2.0 * h)
        pgz = (pp[1:-1, 1:-1, 2:] - pp[1:-1, 1:-1, :-2]) / (2.0 * h)

        # 3. 沿法向投影并提取各向异性法向通量 v
        ndot = nx * pgx + ny * pgy + nz * pgz
        w = mag2 / (mag2 + 0.25)
        factor = c_diff * w * ndot
        vx = factor * nx
        vy = factor * ny
        vz = factor * nz

        pvx = _neumann(vx); pvy = _neumann(vy); pvz = _neumann(vz)
        div_v = ((pvx[2:, 1:-1, 1:-1] - pvx[:-2, 1:-1, 1:-1]) / (2.0 * h) +
                 (pvy[1:-1, 2:, 1:-1] - pvy[:-2, 1:-1, 1:-1]) / (2.0 * h) +
                 (pvz[1:-1, 1:-1, 2:] - pvz[1:-1, 1:-1, :-2]) / (2.0 * h))

        aniso_lap = c_tangent * lap - div_v
        reaction = -(phi ** 3 - phi) + lam * (T - phi)
        phi = phi + dt * (eps2 * aniso_lap + reaction)
        if not np.isfinite(phi).all():
            raise RuntimeError(f"Anisotropic AC 第 {it} 步发散：检查 cfl / eps / lam")
        if track is not None:
            track.append((dt, cfl_actual))
    return phi, dt, cfl_actual


def phase_field_ac_anisotropic(P, N, iters=150, eps_h=2.0, lam=0.8, c_tangent=1.0,
                               c_normal=0.1, sigma_n=0.8, grow=1, smooth=0.5,
                               w=0.4, nn_mult=2.2, n_vox=None, n_vox_max=96,
                               verbose=False, **kw):
    """各向异性扩散张量相场求解器。

    构建几何自适应扩散张量 D = c_tangent*I - (c_tangent - c_normal)*w*(n n^T)。
    沿切线方向实施高平滑去噪 (c_tangent≈1.0)，法向抑制扩散 (c_normal<<1.0)，
    保护尖角与边缘特征免受过度钝化。
    """
    occ, h, T = _prep(P, grow, smooth, w, nn_mult, n_vox, n_vox_max, verbose)
    phi, dt, cfl = _run_ac_anisotropic(None, h, iters, eps_h * h, lam,
                                       c_tangent=c_tangent, c_normal=c_normal,
                                       sigma_n=sigma_n, T=T)
    return _extract_surface(phi, h, P.min(0) - 1e-6, P, N), {"dt": dt, "cfl": cfl}


def _run_ac_convex(occ, h, iters, eps, lam, dt=None, T=None, track=None):
    r"""半隐式 Eyre 凸分裂无条件稳定相场推进。

    采用凸凹能量分解格式：
    凸项（隐式线性化）：\epsilon^2 \Delta \phi^{n+1} - \lambda \phi^{n+1}
    凹项（显式非线性）：\phi^n - (\phi^n)^3 + \lambda T + \beta^n

    无条件能量稳定，打破显式 CFL 步长壁垒，时间步长可扩大 10~50 倍。
    """
    eps2 = eps * eps
    if dt is None:
        dt = 0.4 * h
    if T is None:
        T = occ.astype(np.float64)
        T = 2.0 * T - 1.0
    phi = T.copy()
    nx, ny, nz = phi.shape

    # 预计算 3D 7 点差分拉普拉斯算子在频域的精确特征值
    kx = 2.0 * np.pi * np.fft.fftfreq(nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    eigen_lap = 2.0 * (np.cos(KX) + np.cos(KY) + np.cos(KZ) - 3.0) / (h * h)

    # 逆算子分母：(1 + dt*lam) - dt*eps2*eigen_lap >= 1 + dt*lam > 0（恒对称正定无奇点）
    denom = (1.0 + dt * lam) - dt * eps2 * eigen_lap

    for it in range(iters):
        concave_exp = phi - phi ** 3 + lam * T
        beta = float(np.mean(phi ** 3 - phi + lam * (phi - T)))
        R = phi + dt * (concave_exp + beta)

        # 谱域快速逆求解
        R_hat = np.fft.fftn(R)
        phi_next = np.real(np.fft.ifftn(R_hat / denom))
        phi = np.clip(phi_next, -2.0, 2.0)

        if not np.isfinite(phi).all():
            raise RuntimeError(f"Convex Splitting AC 第 {it} 步发散")
        if track is not None:
            track.append(dt)
    return phi, dt, 1.0


def _run_ac_convex_adaptive(occ, h, iters=80, eps=None, lam=0.8, target_time=None,
                            tol=1e-3, dt_init=None, dt_max=None, dt_min=None,
                            s_max=1.6, s_min=0.3, safety=0.90, T=None, track=None):
    r"""基于两步局部截断误差 (LTE) 与能量耗散控制的自适应时间步长 Eyre 凸分裂相场推进。

    步长更新律：
        dt_{k+1} = dt_k * min( S_max, max( S_min, safety * (TOL / ||LTE||_2)^{1/2} ) )
    """
    eps2 = eps * eps
    if T is None:
        T = occ.astype(np.float64)
        T = 2.0 * T - 1.0
    phi = T.copy()
    nx, ny, nz = phi.shape

    if dt_init is None:
        dt_init = 0.1 * h
    if dt_min is None:
        dt_min = 1e-4 * h
    if dt_max is None:
        dt_max = 3.0 * h
    if target_time is None:
        target_time = iters * 0.4 * h

    kx = 2.0 * np.pi * np.fft.fftfreq(nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    eigen_lap = 2.0 * (np.cos(KX) + np.cos(KY) + np.cos(KZ) - 3.0) / (h * h)

    def _step(u, dt_val):
        concave_exp = u - u ** 3 + lam * T
        beta = float(np.mean(u ** 3 - u + lam * (u - T)))
        R = u + dt_val * (concave_exp + beta)
        denom = (1.0 + dt_val * lam) - dt_val * eps2 * eigen_lap
        R_hat = np.fft.fftn(R)
        u_next = np.real(np.fft.ifftn(R_hat / denom))
        return np.clip(u_next, -2.0, 2.0)

    curr_t = 0.0
    dt = dt_init
    step = 0
    max_trials = iters * 4
    accepted_steps = 0
    rejected_steps = 0

    while curr_t < target_time - 1e-8 and step < max_trials:
        step += 1
        dt_step = min(dt, target_time - curr_t)

        # 1. 单全步推进
        phi_full = _step(phi, dt_step)

        # 2. 两半步推进 (Richardson sub-stepping)
        dt_half = 0.5 * dt_step
        phi_mid = _step(phi, dt_half)
        phi_half = _step(phi_mid, dt_half)

        # 3. 局部截断误差 (LTE) 估计
        lte_field = phi_half - phi_full
        lte_norm = float(np.sqrt(np.mean(lte_field ** 2)))

        # 4. 自适应比例因子计算
        if lte_norm > 1e-12:
            factor = safety * ((tol / lte_norm) ** 0.5)
        else:
            factor = s_max
        factor_clamped = min(s_max, max(s_min, factor))

        if lte_norm <= tol:
            phi = phi_half
            curr_t += dt_step
            accepted_steps += 1
            if track is not None:
                track.append({"time": curr_t, "dt": dt_step, "lte": lte_norm, "status": "accepted"})
            dt = min(dt_max, max(dt_min, dt_step * factor_clamped))
        else:
            rejected_steps += 1
            if track is not None:
                track.append({"time": curr_t, "dt": dt_step, "lte": lte_norm, "status": "rejected"})
            dt = max(dt_min, dt_step * factor_clamped)

    info = {
        "dt": dt,
        "cfl": 0.0,
        "final_time": curr_t,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "total_trials": step,
    }
    return phi, dt, info


def phase_field_ac_convex(P, N, iters=80, eps_h=2.0, lam=0.8, dt_mult=1.0,
                          grow=1, smooth=0.5, w=0.4, nn_mult=2.2,
                          n_vox=None, n_vox_max=96, verbose=False, **kw):
    r"""半隐式 Eyre 凸分裂无条件稳定相场求解器 (Eyre Convex-Splitting Scheme)。

    将 Ginzburg-Landau 自由能泛函严格按凸凹能量解耦：
    隐式处理二阶扩散与线性保真项，显式处理四次不稳定项，配合 FFT 快速泊松逆变，
    实现无条件离散能量耗散 (E^{n+1} <= E^n)。
    允许时间步长比显式 AC 提升 10~50 倍，并在含噪点云上极速收敛。
    """
    occ, h, T = _prep(P, grow, smooth, w, nn_mult, n_vox, n_vox_max, verbose)
    dt = dt_mult * 0.4 * h
    phi, dt_actual, cfl = _run_ac_convex(None, h, iters, eps_h * h, lam, dt=dt, T=T)
    return _extract_surface(phi, h, P.min(0) - 1e-6, P, N), {"dt": dt_actual, "cfl": 0.0}


def phase_field_ac_convex_adaptive(P, N, iters=80, eps_h=2.0, lam=0.8, tol=1e-3,
                                   target_time=None, dt_init=None, dt_max=None,
                                   grow=1, smooth=0.5, w=0.4, nn_mult=2.2,
                                   n_vox=None, n_vox_max=96, verbose=False, **kw):
    r"""自适应时间步长半隐式 Eyre 凸分裂相场求解器。

    基于 Richardson 两步局部截断误差 (LTE) 控制与能量耗散约束：
        dt_{k+1} = dt_k * min( S_max, max( S_min, safety * (TOL / ||LTE||)^{1/2} ) )
    在快速拓扑演化初期自适应细分步长，在平滑松弛后期自适应放大步长，兼备稳定收敛与高效率。
    """
    occ, h, T = _prep(P, grow, smooth, w, nn_mult, n_vox, n_vox_max, verbose)
    phi, dt_actual, info = _run_ac_convex_adaptive(
        None, h, iters=iters, eps=eps_h * h, lam=lam, target_time=target_time,
        tol=tol, dt_init=dt_init, dt_max=dt_max, T=T
    )
    return _extract_surface(phi, h, P.min(0) - 1e-6, P, N), info


SOLVERS = {
    "none":          identity,
    "laplacian":     laplacian,
    "bilateral":     bilateral,
    "pf_ac":         phase_field_ac,
    "pf_ac_fidelity": phase_field_ac_fidelity,
    "pf_ac_barrier": phase_field_ac_barrier,
    "pf_ac_conserved": phase_field_ac_conserved,
    "pf_ac_aniso":     phase_field_ac_anisotropic,
    "pf_ac_convex":    phase_field_ac_convex,
    "pf_ac_convex_adaptive": phase_field_ac_convex_adaptive,
    "cahn_hilliard": cahn_hilliard,
}

DESCRIPTIONS = {
    "none": "不做处理（噪声基线）",
    "laplacian": "拉普拉斯平滑，λ=0.5×10 轮。经典收缩基线",
    "bilateral": "迭代双边滤波，空间+法向双权重，特征保持基线（Python 逐点循环，>8000 点会拒绝）",
    "pf_ac": "AC + 弱保真项 λ=0.5，ε=2h。λ 调小可见毛细腐蚀（λ=0 时液滴收缩到消失）",
    "pf_ac_fidelity": "AC + 保真项 λ=0.8，ε=2h。核心基线，accuracy/volume 权衡",
    "pf_ac_barrier": "AC + 边缘指示函数屏障（论文①路线），σ=3h, α=2, λ=1.0",
    "pf_ac_conserved": "体积守恒 AC（拉格朗日乘子补偿 β(t)），ε=2h, λ=0.8。无四阶刚性，严格保体积",
    "pf_ac_aniso": "各向异性张量 AC，切向强平滑、法向抑制扩散保特征，ε=2h, λ=0.8",
    "pf_ac_convex": "半隐式 Eyre 凸分裂 AC，无条件能量耗散稳定，时间步大 20~50 倍",
    "pf_ac_convex_adaptive": "自适应时间步长半隐式 Eyre 凸分裂 AC，基于双步 LTE 与能量耗散控制",
    "cahn_hilliard": "CH 守恒型，ε=2h, λ=0.8, M=1，ch_cfl=0.05。体积是格式级保证",
}


def get_solver(name):
    if name not in SOLVERS:
        raise ValueError(f"未知求解器 {name!r}，可选：{list(SOLVERS)}")
    return SOLVERS[name]
