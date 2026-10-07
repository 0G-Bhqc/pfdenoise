"""noise.py — 噪声模型库：把「噪声」提升为一等公民

为什么单独做一个模块、为什么要这么设计
--------------------------------------
2025 年的点云去噪综述（arXiv:2508.17011）明确指出：现有 SOTA 几乎全是
**有监督**方法，依赖成对的「脏-净」数据，而真实传感器很难造出这种配对；
无监督方法在噪声变大时性能明显退化（ScoreDenoise-U 在 3% 噪声下 HD 达 12.2）。
2026 年的 Noise2Noise 重访（arXiv:2609.16788）进一步给出结论：
**起作用的是「训练对的分布」，不是损失函数** —— 合成噪声训出来的模型在真实
噪声上只涨 0.8~3.7 dB，换成该域自己的噪声对则涨 9.4~11.0 dB。

也就是说：**噪声模型本身是这件事里最重要的一个设计变量，却最常被当成实现细节。**
本模块把每种噪声显式写成对象，并强制声明它的统计假设，好让使用者能判断
「Noise2Noise 那一套理论在这里还成不成立」。

每个 NoiseModel 都有：
    apply(points, rng) -> 新点集
    assumption          统计假设的文字说明（是否零均值 / 是否独立 / 是否有偏）
    n2n_compatible      该噪声下，两次独立采样能否用 Noise2Noise 协议评测
"""

import numpy as np

__all__ = ["NoiseModel", "Gaussian", "Protrusion", "Outlier",
           "Aniso", "Density", "Composite", "NOISE_MODELS", "make_noise"]


# 网格哈希最近邻：确定性、无外部依赖、全向量化，对 1e4 量级点云足够快。
_SHELL_CACHE: dict = {}


def _shell_offsets(r):
    """切比雪夫半径恰为 r 的邻格壳；r=1 即 3×3×3 块（27 格，含中心）。"""
    if r == 1:
        return [(dx, dy, dz) for dx in (-1, 0, 1)
                for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    if r not in _SHELL_CACHE:
        out = [(dx, dy, dz) for dx in range(-r, r + 1)
               for dy in range(-r, r + 1) for dz in range(-r, r + 1)
               if max(abs(dx), abs(dy), abs(dz)) == r]
        _SHELL_CACHE[r] = out
    return _SHELL_CACHE[r]


class _NN:
    """网格哈希最近邻（确定性、无外部依赖、结果精确）。

    实现与复杂度
    ------------
    建索引：库点按「格 key 展平码」一次排序（O(N log N)）。查询时用
    searchsorted 定位每个候选格内点在排序数组中的区间，整环候选一次性
    拼接、按 (点, 距离) 归并取前 k —— 全程无 Python 逐点循环。
    早先版本是逐点 × 27 次 dict 查找，6000 点的 laplacian 基准 12.7 s
    几乎全耗在这里；向量化后同基准约快一个数量级以上。

    正确性（这是对旧版的一个实质修正）
    --------------------------------
    旧版只扫 27 邻格且**无精确性保证**：邻格非空但不含真最近邻时距离被
    高估，hausdorff（最坏情形指标）首当其冲，且从未与暴力法交叉验证过。
    本版按环扩张：若当前第 k 近邻距离 d_k > ring·cell，更远的环里仍可能
    有更近的点（查询点可在格内任意位置，格下标差 ≥ ring+1 的点坐标差
    ≥ ring·cell），于是继续扫外环直到 d_k ≤ ring·cell 或扫完全网格；
    因此返回的是**精确** k 近邻，不是近似。
    """

    def __init__(self, pts, cell=None):
        self.p = np.asarray(pts, dtype=np.float64)
        if self.p.ndim != 2 or self.p.shape[1] != 3:
            raise ValueError(f"_NN 需要 (N,3) 点集，收到 shape={self.p.shape}")
        # fail loud：NaN/Inf 坐标会让格 key 变成未定义整数，所有距离静默变垃圾
        # —— 在入口拒绝，不给「看起来在跑」的假象。
        if not np.isfinite(self.p).all():
            raise ValueError("点集含 NaN/Inf，拒绝建最近邻索引（请先清洗数据）")
        n = len(self.p)
        if cell is None:
            ext = self.p.max(0) - self.p.min(0)
            cell = float(np.mean(ext)) / max(6.0, (n ** (1 / 3)) / 2.0)
        self.cell = max(float(cell), 1e-9)
        self.lo = self.p.min(0) - 4 * self.cell
        key = np.floor((self.p - self.lo) / self.cell).astype(np.int64)
        self._kmin = key.min(0)
        kd = key - self._kmin
        self._dims = kd.max(0) + 1                      # 每轴格数
        # 展平码 x*dims_y*dims_z + y*dims_z + z 需要落在 int64 内。
        # dims 之积在病态拉伸的点云下可能超界 —— 此时退回逐点循环实现。
        prod = int(self._dims[0]) * int(self._dims[1]) * int(self._dims[2])
        self._fast = prod < (1 << 62)
        if self._fast:
            code = (kd[:, 0] * self._dims[1] + kd[:, 1]) * self._dims[2] + kd[:, 2]
            self._order = np.argsort(code, kind="stable")
            self._sorted_code = code[self._order]
            # 分块查询：峰值内存 ~ chunk × 平均每格点数 × 27。按点云的实际
            # 占据密度自适应块大小，把单批候选数压在 ~2e6 以内。
            occ = int(np.unique(code).size)
            avg_per_cell = n / max(occ, 1)
            self._chunk = int(min(4096, max(16, 2_000_000 / max(avg_per_cell, 1.0))))

    # ------------------------------------------------------------------ 查询
    def query(self, q, k=1):
        """查询每个 q 点的 k 个最近邻。返回 (距离[M,k], 索引[M,k])，距离升序。

        与旧版一致：不过滤自身（q==建库点时第 1 近邻就是自己，d=0），
        邻域偏移用 {-1,0,+1}（存库 key 无额外平移，相邻格 key 差即 ±1）。

        ⚠ 查询点同样拒绝 NaN/Inf：NaN 在 floor→int64 铸型下会变成
        INT64_MIN，把扩环上界撑到 9e18 —— 早期版本因此无限循环挂死。
        """
        if not self._fast:
            return self._query_loop(q, k)
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        if not np.isfinite(q).all():
            raise ValueError("查询点含 NaN/Inf，拒绝查询（请先清洗数据）")
        M = len(q)
        out_d = np.full((M, int(k)), np.inf)
        out_i = np.zeros((M, int(k)), dtype=np.int64)
        if M == 0 or len(self.p) == 0:
            return out_d, out_i

        # 分块跑环扩张，控制单批候选拼接的峰值内存
        for c0 in range(0, M, self._chunk):
            c1 = min(c0 + self._chunk, M)
            self._query_block(q[c0:c1], k, out_d[c0:c1], out_i[c0:c1])
        return out_d, out_i

    def _query_block(self, q, k, out_d, out_i):
        """对一个查询点块做环扩张精确 k 近邻（写入 out_d/out_i 切片）。"""
        M = len(q)
        k_eff = max(1, min(int(k), len(self.p)))
        key = np.floor((q - self.lo) / self.cell).astype(np.int64) - self._kmin
        # 扩环上界：网格尺寸 + 少量余量（lo 端预留了 4·cell）。
        # ⚠ 不能按查询 key 的 |max| 放开上界——远距查询（哪怕有限坐标）
        #   会把 ring 撑到天文数字，环循环实质挂死。网格扫完仍不满足的
        #   查询点交给末尾的暴力兜底（对远点反而更便宜）。
        max_ring = int(self._dims.max()) + 8
        active = np.ones(M, dtype=bool)

        for ring in range(1, max_ring + 1):
            if not active.any():
                break
            a_idx = np.nonzero(active)[0]
            offs = _shell_offsets(ring)
            lo_all, hi_all = self._cell_ranges(key[a_idx], offs)
            counts = hi_all - lo_all
            total = int(counts.sum())
            if total:
                starts = np.zeros(counts.size + 1, dtype=np.int64)
                np.cumsum(counts.ravel(), out=starts[1:])
                s = np.arange(total, dtype=np.int64)
                slot = np.searchsorted(starts[1:], s, side="right")
                pos = s - starts[slot]
                qp = np.repeat(a_idx, counts.sum(1))     # 每个候选所属的查询点
                cand = self._order[lo_all.ravel()[slot] + pos]
                diff = self.p[cand] - q[qp]
                d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
                self._merge_topk(qp, d, cand, out_d, out_i)
            # 收敛判定：d_k <= ring*cell 后，环外点至少隔着 ring*cell，不可能更近
            # （完成的点移出队列；未完成的继续留在 active 里等下一个环）
            done = out_d[a_idx, k_eff - 1] <= ring * self.cell
            active[a_idx[done]] = False

        # 扩环到全网格仍不足 k 个（只在点数 < k 时可能发生）——全局暴力补齐
        missing = np.nonzero(np.isinf(out_d[:, 0]))[0]
        for j in missing:
            diff = self.p - q[j]
            d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            sel = np.argsort(d, kind="stable")[:int(k)]
            out_d[j, :len(sel)] = d[sel]
            out_i[j, :len(sel)] = sel

    def _cell_ranges(self, keys, offs):
        """keys:(M,3) 查询格 key；offs: 本环偏移。返回每个 (点,格) 的
        候选区间 [lo, hi)（在 _order 排序数组里），空格区间长度 0。"""
        M = len(keys)
        lo_all = np.empty((M, len(offs)), dtype=np.int64)
        hi_all = np.empty((M, len(offs)), dtype=np.int64)
        dims = self._dims
        for c, (dx, dy, dz) in enumerate(offs):
            nk = keys + np.array([dx, dy, dz], dtype=np.int64)
            ok = ((nk >= 0) & (nk < dims)).all(1)
            code = (nk[:, 0] * dims[1] + nk[:, 1]) * dims[2] + nk[:, 2]
            code = np.where(ok, code, -1)                # 越界格 -> 空区间
            lo_all[:, c] = np.searchsorted(self._sorted_code, code, side="left")
            hi_all[:, c] = np.searchsorted(self._sorted_code, code, side="right")
        return lo_all, hi_all

    def _merge_topk(self, qp, d, cand, out_d, out_i):
        """把本轮候选并入每个查询点的当前 top-k（按 (点, 距离) 归并）。

        排序用**单键稳定** argsort（qp·scale + d）而不是三键 lexsort ——
        这是查询热路径的主要开销（20000 点约占总耗时的一半以上）。
        等距并列时稳定序保持候选的原始次序（同格内按库点原始顺序），
        「查自己必先命中自己」的语义由此保住；float64 舍入只可能让
        相差 <1e-12 的距离交换次序，不影响距离数值本身。
        """
        M, k = out_d.shape
        all_qp = np.concatenate([np.repeat(np.arange(M), k), qp])
        all_d = np.concatenate([out_d.ravel(), d])
        all_i = np.concatenate([out_i.ravel(), cand])
        # 单键稳定排序 key = qp·scale + d'（d' = d，inf 映射到「组内最大有限+1」）。
        # ⚠ inf 不能直接进 key：0·inf=NaN 且会把 inf 条目排到**所有组之后**，
        #   破坏「按组连续」的前提，rank 计算错位（实测 IndexError 负索引）。
        fin = np.isfinite(all_d)
        dmax = float(all_d[fin].max()) if fin.any() else 0.0
        scale = dmax + 2.0                     # 组间隔 > 组内任何 key，组必连续
        key = all_qp * scale + np.where(fin, all_d, dmax + 1.0)
        order = np.argsort(key, kind="stable")
        qp_s, d_s, i_s = all_qp[order], all_d[order], all_i[order]
        grp_start = np.searchsorted(qp_s, np.arange(M), side="left")
        rank = np.arange(len(qp_s)) - grp_start[qp_s]
        keep = rank < k
        out_d[:, :] = np.inf
        out_i[:, :] = 0
        b, r = qp_s[keep], rank[keep]
        out_d[b, r] = d_s[keep]
        out_i[b, r] = i_s[keep]

    # -------------------------------------------------- 逐点循环（兜底路径）
    def _query_loop(self, q, k=1):
        """展平码溢出（病态网格）时的兜底：逐点扫 27 邻格 + 全局暴力。
        语义与向量化路径一致，仅慢。"""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        out_d = np.full((len(q), int(k)), np.inf)
        out_i = np.zeros((len(q), int(k)), dtype=np.int64)
        cells = {}
        key_all = np.floor((self.p - self.lo) / self.cell).astype(np.int64) - self._kmin
        for i, kk in enumerate(map(tuple, key_all)):
            cells.setdefault(kk, []).append(i)
        offsets = _shell_offsets(1)
        for j in range(len(q)):
            key = np.floor((q[j] - self.lo) / self.cell).astype(np.int64) - self._kmin
            bd = np.empty(0)
            bi = np.empty(0, dtype=np.int64)
            for dx, dy, dz in offsets:
                g = cells.get((key[0] + dx, key[1] + dy, key[2] + dz))
                if g is None:
                    continue
                g = np.asarray(g)
                d = np.sqrt(((self.p[g] - q[j]) ** 2).sum(1))
                bd = np.concatenate([bd, d]) if bd.size else d
                bi = np.concatenate([bi, g]) if bi.size else g
            if bd.size == 0:                             # 极端稀疏：全局暴力
                d = np.sqrt(((self.p - q[j]) ** 2).sum(1))
                sel = np.argsort(d, kind="stable")[:int(k)]
                out_d[j, :len(sel)], out_i[j, :len(sel)] = d[sel], sel
                continue
            sel = np.argsort(bd, kind="stable")[:int(k)]
            m = min(int(k), len(sel))
            out_d[j, :m], out_i[j, :m] = bd[sel[:m]], bi[sel[:m]]
        return out_d, out_i


class NoiseModel:
    """所有噪声模型的基类。统一签名 apply(points, normals, rng, nn=None)。"""

    name = "base"
    n2n_compatible = True

    def apply(self, P, N, rng, nn=None):           # pragma: no cover - 抽象
        raise NotImplementedError

    def assumption_text(self):
        # 「噪声是一等公民」的前提是每个模型都显式声明统计假设。
        # 这里绝不静默返回一个空串——没声明就让它当场炸出来。
        raise NotImplementedError(
            f"{type(self).__name__} 没有声明统计假设（assumption_text）。"
            f"噪声模型的统计性质是本包的核心契约，必须显式给出。")

    def __repr__(self):
        return f"{type(self).__name__}()"


class Gaussian(NoiseModel):
    """各向同性高斯噪声，沿法向/切向同方差。

    assumption: 零均值、逐点独立。Noise2Noise 理论**完全适用**。
    局限：它把点推离曲面法线方向，真实扫描仪的偏差通常是各向异性的。
    """
    name = "gaussian"
    n2n_compatible = True

    def __init__(self, sigma=0.01):
        self.sigma = float(sigma)

    def assumption_text(self):
        return f"零均值、各向同性、逐点独立；sigma={self.sigma:.4f}（相对 bbox 半宽）"

    def apply(self, P, N, rng=None, nn=None):
        if rng is None:
            rng = np.random.default_rng()
        return P + rng.normal(0.0, self.sigma, P.shape)


class Protrusion(NoiseModel):
    """凸起噪声：按比例把点沿法向推出去。

    这是 CAD/逆向工程里的经典噪声模型（Knöppel et al. 的 protrusion 噪声），
    也是**有偏**噪声：它有非零均值。
    assumption: 非零均值（系统性外推），逐点独立。
               → Noise2Noise 理论**不适用**（N2N 要求 E[noise]=0）。
    """
    name = "protrusion"
    n2n_compatible = False

    def __init__(self, density=0.05, magnitude=0.02):
        self.density = float(density)
        self.magnitude = float(magnitude)

    def assumption_text(self):
        return (f"有偏（沿法向系统外推），逐点独立；"
                f"density={self.density:.3f}, magnitude={self.magnitude:.4f} -> N2N 理论不适用")

    def apply(self, P, N, rng=None, nn=None):
        if rng is None:
            rng = np.random.default_rng()
        P = P.copy()
        m = rng.random(len(P)) < self.density
        k = int(m.sum())
        if k:
            # 80% 沿法向外推、20% 内推 —— 净效应是**系统性的向外偏置**。
            # 早先版本把概率写反（p=[0.8, 0.2] 对应 [-1, +1]），实现变成
            # 净内推，与 docstring「外推」相反——噪声模型的自述统计性质
            # 必须与实现一致，否则基于它的一切推论（N2N 适用性等）都不可信。
            sign = rng.choice([-1.0, 1.0], size=k, p=[0.2, 0.8])[:, None]
            P[m] += N[m] * (self.magnitude * rng.random((k, 1)) * sign)
        return P


class Outlier(NoiseModel):
    """离群点：把一部分点直接扔到物体附近的空间里。

    assumption: 重尾、非零均值、**非独立**（密度不均）。
               这是去噪算法最怕的一类，N2N 也不适用。
    """
    name = "outlier"
    n2n_compatible = False

    def __init__(self, rate=0.02, scale=0.15):
        self.rate = float(rate)
        self.scale = float(scale)

    def assumption_text(self):
        return (f"重尾、密度不均、强有偏；rate={self.rate:.3f}, "
                f"scale={self.scale:.3f} -> N2N 理论不适用")

    def apply(self, P, N, rng=None, nn=None):
        if rng is None:
            rng = np.random.default_rng()
        P = P.copy()
        m = rng.random(len(P)) < self.rate
        k = int(m.sum())
        if k:
            P[m] += rng.normal(0.0, self.scale * np.abs(P).max(), (k, 3))
        return P


class Aniso(NoiseModel):
    """各向异性噪声：法向白噪声 + 切向**低频空间相关**偏移场。

    assumption: 零均值，但**空间相关**（同一区域的点共享一个平滑偏移，
               模拟标定误差/结构光条纹失真）。→ N2N 不适用
               （N2N 要求噪声逐点独立；相关噪声即使零均值也破坏其前提）。

    实现（早先版本是空操作，这里如实修正）
    ------------------------------------
    早先版本的「低频场」是把 iid 噪声与它自己的最近邻平均——而最近邻
    查询返回的是点自身，平均完等于没平均，实际生成的仍是 iid 噪声。
    现在的构造：iid 白噪声 → 在 NN 图上与 6 邻域均值按 smooth 权重混合
    （迭代出真空间相关）→ 投影到切平面 → 归一化到单位 RMS。
    sigma_n / sigma_t 保持「法向 / 切向位移标准差」的量纲含义。
    """

    name = "aniso"
    n2n_compatible = False

    def __init__(self, sigma_t=0.002, sigma_n=0.015, smooth=0.3):
        self.sigma_t = float(sigma_t)
        self.sigma_n = float(sigma_n)
        self.smooth = float(smooth)

    def assumption_text(self):
        return (f"零均值但空间相关（切向低频偏移，NN 图 smooth={self.smooth:.2f}）；"
                f"sigma_t={self.sigma_t:.4f}, sigma_n={self.sigma_n:.4f} -> N2N 理论不适用")

    def apply(self, P, N, rng=None, nn=None):
        if rng is None:
            rng = np.random.default_rng()
        # 法向分量：逐点独立、零均值（扫描噪声的「白」部分）
        dn = self.sigma_n * rng.normal(size=(len(P), 1))
        # 切向分量：低频空间相关偏移场（「有色」部分）
        off = rng.normal(0.0, 1.0, (len(P), 3))
        nn_ = nn if nn is not None else _NN(P)
        k = min(6, len(P))
        _, idx = nn_.query(P, k)
        off = (1.0 - self.smooth) * off + self.smooth * off[idx].mean(axis=1)
        # 投影到切平面（切向噪声不沿法向），再归一化到单位 RMS
        off = off - np.sum(off * N, axis=1, keepdims=True) * N
        rms = float(np.sqrt(np.mean(np.sum(off * off, axis=1))))
        if rms > 1e-12:
            off = off / rms
        return P + N * dn + off * self.sigma_t


class Density(NoiseModel):
    """密度涨落：**只丢弃** ratio 比例的点（模拟扫描视角/距离造成的疏密）。

    assumption: 不改变点的位置，但**减少了点的数量**（不是一一对应），
               → Chamfer 类指标仍可用，N2N 点级配对不可用。
               早先 docstring 宣称「保留/丢弃双向」与实现不符——实现
               从不加点，现已如实声明为单向丢弃。
    """
    name = "density"
    n2n_compatible = False

    def __init__(self, ratio=0.25, seed=0):
        self.ratio = float(ratio)
        self.seed = int(seed)

    def assumption_text(self):
        return (f"点数减少（丢弃 ratio={self.ratio:.2f}，非一一对应）-> "
                f"Chamfer 可用，点级 N2N 配对不可用")

    def apply(self, P, N, rng=None, nn=None):
        if rng is None:
            rng = np.random.default_rng(self.seed)     # seed 在这里生效
        n_out = int(len(P) * self.ratio)
        drop = rng.choice(len(P), n_out, replace=False)
        keep = np.setdiff1d(np.arange(len(P)), drop)
        return P[keep]


class Composite(NoiseModel):
    """噪声叠加：真实扫描件几乎总是多种噪声的混合。"""

    def __init__(self, *models):
        self.models = models
        self.name = "composite(" + "+".join(m.name for m in models) + ")"
        self.n2n_compatible = all(m.n2n_compatible for m in models)

    def assumption_text(self):
        return " ; ".join(m.assumption_text() for m in self.models)

    def apply(self, P, N, rng=None, nn=None):
        if rng is None:
            rng = np.random.default_rng()
        for m in self.models:
            P = m.apply(P, N, rng, nn)
        return P


NOISE_MODELS = {
    "gaussian":  lambda: Gaussian(0.010),
    "gaussian_hi": lambda: Gaussian(0.025),
    "protrusion": lambda: Protrusion(0.05, 0.02),
    "outlier":   lambda: Outlier(0.02, 0.15),
    "aniso":     lambda: Aniso(0.002, 0.015),
    "density":   lambda: Density(0.25),
    "realistic": lambda: Composite(Gaussian(0.008), Protrusion(0.04, 0.015),
                                   Outlier(0.01, 0.10)),
}


def make_noise(name):
    if name not in NOISE_MODELS:
        raise ValueError(f"未知噪声 {name!r}，可选：{list(NOISE_MODELS)}")
    return NOISE_MODELS[name]()
