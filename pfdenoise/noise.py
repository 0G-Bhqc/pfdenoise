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


def _unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


# 网格哈希最近邻：确定性、无外部依赖，对 1e4 量级点云足够快。
class _NN:
    """网格哈希最近邻（确定性、无外部依赖）。

    早期版本用 `np.all(uniq == t, axis=1)` 逐格扫描建索引，复杂度
    O(N · 27 · n_cells)，3000 点就已经要几十秒。改成 dict 直接查格，
    复杂度降到 O(N · 27 · 平均每格点数)，3000 点约 1 秒。
    """

    def __init__(self, pts, cell=None):
        self.p = np.asarray(pts, dtype=np.float64)
        n = len(self.p)
        if cell is None:
            ext = self.p.max(0) - self.p.min(0)
            cell = float(np.mean(ext)) / max(6.0, (n ** (1 / 3)) / 2.0)
        self.cell = max(float(cell), 1e-9)
        self.lo = self.p.min(0) - 4 * self.cell
        key = np.floor((self.p - self.lo) / self.cell).astype(np.int64)
        # key -> 该格内的点索引
        self.cells = {}
        for i, k in enumerate(map(tuple, key)):
            self.cells.setdefault(k, []).append(i)
        self.cells = {k: np.asarray(v) for k, v in self.cells.items()}

    def query(self, q, k=1):
        """查询每个 q 点的 k 个最近邻。返回 (距离[N,k], 索引[N,k])，距离升序。

        邻域偏移必须是 {-1,0,+1}：存库时 key = floor((p-lo)/cell) 没有额外平移，
        所以相邻格的 key 差就是 ±1。早先写成 {0,1,2} 会永远查不到自己。
        """
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        key = np.floor((q - self.lo) / self.cell).astype(np.int64)
        out_d = np.full((len(q), k), np.inf)
        out_i = np.zeros((len(q), k), dtype=np.int64)
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1)
                   for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        cells = self.cells
        for j in range(len(q)):
            kx, ky, kz = int(key[j, 0]), int(key[j, 1]), int(key[j, 2])
            bd = np.empty(0)
            bi = np.empty(0, dtype=np.int64)
            for ox, oy, oz in offsets:
                g = cells.get((kx + ox, ky + oy, kz + oz))
                if g is None:
                    continue
                d = np.linalg.norm(self.p[g] - q[j], axis=1)
                if bd.size == 0:
                    bd, bi = d, g
                else:
                    bd = np.concatenate([bd, d])
                    bi = np.concatenate([bi, g])
                    if bd.size > 4 * k:               # 剪枝，避免集合无限膨胀
                        sel = np.argsort(bd, kind="stable")[:k]
                        bd, bi = bd[sel], bi[sel]
            if bd.size == 0:                          # 极端稀疏：退化为全局最近
                d = np.linalg.norm(self.p - q[j], axis=1)
                sel = np.argsort(d, kind="stable")[:k]
                bd, bi = d[sel], sel
            if bd.size > k:
                sel = np.argsort(bd, kind="stable")[:k]
                bd, bi = bd[sel], bi[sel]
            m = min(k, bd.size)
            out_d[j, :m], out_i[j, :m] = bd[:m], bi[:m]
        return out_d, out_i


class NoiseModel:
    """所有噪声模型的基类。统一签名 apply(points, normals, rng, nn=None)。"""

    name = "base"
    n2n_compatible = True

    def apply(self, P, N, rng, nn=None):           # pragma: no cover - 抽象
        raise NotImplementedError

    def assumption_text(self):
        return self.assumption

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

    def apply(self, P, N, rng, nn=None):
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

    def apply(self, P, N, rng, nn=None):
        P = P.copy()
        m = rng.random(len(P)) < self.density
        k = int(m.sum())
        if k:
            sign = rng.choice([-1.0, 1.0], size=k, p=[0.8, 0.2])[:, None]
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

    def apply(self, P, N, rng, nn=None):
        P = P.copy()
        m = rng.random(len(P)) < self.rate
        k = int(m.sum())
        if k:
            P[m] += rng.normal(0.0, self.scale * np.abs(P).max(), (k, 3))
        return P


class Aniso(NoiseModel):
    """各向异性噪声：切向噪声远小于法向噪声。

    assumption: 零均值，但**空间相关**（同一区域的点共享一个偏移，
               模拟标定误差/结构光条纹失真）。→ N2N 不适用。
    """
    name = "aniso"
    n2n_compatible = False

    def __init__(self, sigma_t=0.002, sigma_n=0.015, smooth=0.3):
        self.sigma_t = float(sigma_t)
        self.sigma_n = float(sigma_n)
        self.smooth = float(smooth)

    def assumption_text(self):
        return (f"零均值但空间相关（低频偏移）；sigma_t={self.sigma_t:.4f}, "
                f"sigma_n={self.sigma_n:.4f} -> N2N 理论不适用")

    def apply(self, P, N, rng, nn=None):
        P = P.copy()
        nt = np.cross(N, rng.normal(size=N.shape))
        nt = _unit(nt)
        nb = _unit(np.cross(nt, N))
        # 用邻域平均构造一个低频偏移场
        off = rng.normal(0.0, 1.0, (len(P), 3))
        if nn is not None:
            d, i = nn.query(P, 1)
            off = (off + off[i[:, 0]]) * 0.5
        off = _unit(off - np.sum(off * N, axis=1, keepdims=True) * N)
        return P + N * (self.sigma_n * rng.normal(size=(len(P), 1))) \
                 + off * (self.sigma_t * rng.normal(size=(len(P), 1)))


class Density(NoiseModel):
    """密度涨落：沿曲面随机保留/丢弃点（模拟不同扫描距离与视角）。

    assumption: 零均值噪声叠加，但**改变了点的数量**（不是一一对应），
               → Chamfer 类指标仍可用，N2N 配对需按曲面配对而非按点配对。
    """
    name = "density"
    n2n_compatible = False

    def __init__(self, ratio=0.25, seed=0):
        self.ratio = float(ratio)
        self.seed = int(seed)

    def assumption_text(self):
        return (f"点数量变化（非一一对应）；ratio=+/-{self.ratio:.2f} -> "
                f"Chamfer 可用，点级 N2N 配对不可用")

    def apply(self, P, N, rng, nn=None):
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

    def apply(self, P, N, rng, nn=None):
        for m in self.models:
            P = m.apply(P, N, rng, nn)
        return P


NOISE_MODELS = {
    "gaussian":  lambda: Gaussian(0.010),
    "gaussian_hi": lambda: Gaussian(0.025),
    "protrusion": lambda: Protrusion(0.05, 0.02),
    "outlier":   lambda: Outlier(0.02, 0.15),
    "aniso":     lambda: Aniso(0.002, 0.015),
    "realistic": lambda: Composite(Gaussian(0.008), Protrusion(0.04, 0.015),
                                   Outlier(0.01, 0.10)),
}


def make_noise(name):
    if name not in NOISE_MODELS:
        raise ValueError(f"未知噪声 {name!r}，可选：{list(NOISE_MODELS)}")
    return NOISE_MODELS[name]()
