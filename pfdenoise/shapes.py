"""shapes.py — 解析参考曲面（无外部资产依赖）

每个曲面提供三样东西：
  points(N, 3)   均匀采样的干净点
  normals(N, 3)  解析法向（去噪评测里 P2M / 法向误差都需要它）
  enclosed_volume()  被曲面包住的体积（用来量化「保体积」，经典文献的卖点、
                    深度学习论文通常不报的一项）

刻意用解析曲面而不是 .ply 资产，原因有三：
  1. 干净真值精确已知，不需要额外的网格文件；
  2. 自包含，pip install 之后立刻能跑，不需要下载数据；
  3. 解析法向让「法向保持」这类指标没有离散化误差，可以干净地归因到求解器。
"""

import numpy as np

__all__ = ["make_shape", "SHAPES"]


def _sample_sphere(rng, n, R=1.0):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * R, d.copy(), lambda: 4.0 / 3.0 * np.pi * R ** 3


def _sample_torus(rng, n, R=1.0, r=0.38):
    # 面积元 dA = R(R + r cos v) dv du
    u = rng.uniform(0, 2 * np.pi, n)
    while True:
        v = rng.uniform(0, 2 * np.pi, n)
        w = rng.uniform(0, 1, n)
        keep = w <= (R + r * np.cos(v)) / (R + r)
        u[~keep], v[~keep], w[~keep] = (rng.uniform(0, 2 * np.pi, int((~keep).sum())),
                                         rng.uniform(0, 2 * np.pi, int((~keep).sum())),
                                         rng.uniform(0, 1, int((~keep).sum())))
        if keep.all():
            break
    P = np.stack([(R + r * np.cos(v)) * np.cos(u),
                  (R + r * np.cos(v)) * np.sin(u),
                  r * np.sin(v)], axis=1)
    Nv = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)], axis=1)
    return P, Nv, lambda: 2 * np.pi ** 2 * R * r ** 2


def _sample_superquadric(rng, n, a=1.0, b=0.8, c=0.7, e1=0.6, e2=0.6):
    # |x/a|^e1 + |y/b|^e2 + |z/c|^e2 = 1 的椭球型超二次曲面（可调「方-ness」，
    # 用来测试求解器在平坦大面 + 锐利棱边处的行为）
    e1v, e2v = 2.0 / e1, 2.0 / e2
    while True:
        P = rng.uniform(-1, 1, (n, 3)) * np.array([a, b, c])
        v = (np.abs(P[:, 0] / a) ** e1v + np.abs(P[:, 1] / b) ** e2v
             + np.abs(P[:, 2] / c) ** e2v)
        keep = v <= 1.0
        if keep.all():
            break
        P[~keep] = rng.uniform(-1, 1, ((~keep).sum(), 3)) * np.array([a, b, c])
    f = lambda P: (np.abs(P[:, 0] / a) ** (e1v - 1) / a
                   + np.abs(P[:, 1] / b) ** (e2v - 1) / b
                   + np.abs(P[:, 2] / c) ** (e2v - 1) / c)
    Nv = f(P)
    Nv /= np.linalg.norm(Nv, axis=1, keepdims=True)
    return P, Nv, lambda: float("nan")     # 超二次曲面无解析体积


def _sample_bumpy(rng, n, R=1.0, amp=0.12, k=7):
    """半径带多频起伏的球（类 Stanford bunny 的「非平凡拓扑 + 特征」替身）。

    刻意做成高频起伏 + 整体光滑：这是去噪算法最容易「过度平滑」的工况，
    能把 accuracy / structure-preservation 的权衡暴露出来。
    """
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    # 半径扰动：低频大形 + 高频细节
    low = (0.6 * np.sin(3 * np.arctan2(y, x)) * np.cos(2 * z)
           + 0.4 * np.sin(4 * np.arctan2(z, x) + 0.7))
    hi = (0.25 * np.sin(k * x + 1.1) * np.sin(k * y + 0.4) * np.cos(k * z))
    rad = R * (1.0 + amp * (low + hi))
    P = d * rad[:, None]
    # 数值法向（该曲面的解析梯度太啰嗦，且我们只要真值精度到 1e-3 量级）
    eps = 1e-4
    f = lambda Q: (np.linalg.norm(Q, axis=1) - R * (1.0 + amp * (
        0.6 * np.sin(3 * np.arctan2(Q[:, 1], Q[:, 0])) * np.cos(2 * Q[:, 2])
        + 0.4 * np.sin(4 * np.arctan2(Q[:, 2], Q[:, 0]) + 0.7)
        + 0.25 * np.sin(k * Q[:, 0] + 1.1) * np.sin(k * Q[:, 1] + 0.4)
        * np.cos(k * Q[:, 2]))))
    Nv = np.stack([(f(P + eps * np.eye(3)[j]) - f(P - eps * np.eye(3)[j])) / (2 * eps)
                   for j in range(3)], axis=1)
    Nv /= np.linalg.norm(Nv, axis=1, keepdims=True)
    return P, Nv, lambda: float("nan")


SHAPES = {
    "sphere":  _sample_sphere,
    "torus":   _sample_torus,
    "superquadric": _sample_superquadric,
    "bumpy":   _sample_bumpy,
}


def make_shape(name="bumpy", n=4000, seed=0):
    """返回 (points, normals, volume_fn, bbox_scale)。"""
    if name not in SHAPES:
        raise ValueError(f"未知曲面 {name!r}，可选：{list(SHAPES)}")
    rng = np.random.default_rng(seed)
    P, N, vfn = SHAPES[name](rng, n)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    scale = float(np.abs(P).max())          # 用 bbox 半宽做尺度的归一化基准
    return P, N, vfn, scale
