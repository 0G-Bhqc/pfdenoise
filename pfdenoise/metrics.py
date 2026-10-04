"""metrics.py — 评测指标（含无标签协议）

分两类
------
有标签（需要干净真值）:
    chamfer        单向 Chamfer 距离
    p2m            Point-to-Mesh 距离（用密集参考点采样代理，见下）
    hausdorff      单向最大距离，捕捉最坏情况
    normal_err     法向平均夹角，衡量「特征/结构保持」
    area_drift     面积/体积相对真值的漂移 —— **经典文献的卖点，DL 论文普遍不报**
    p2m_p95        P2M 的 95 分位，比均值更能反映「大部分点贴住了没有」

无标签（拿不到干净真值时用）:
    n2n_consistency  两次独立噪声观测各自去噪后，结果之间的一致性。
        依据 Noise2Noise：若去噪器是无偏估计，两次结果都指向同一个干净信号，
        它们的差异就是「不确定度」的上界，可以在完全没有真值时用来选模型/选超参。
    self_consistency 单次观测反复施加去噪的稳定性（固定点距离），
        反映「算法是否收敛到一个稳定不动点」。

关于 P2M 的诚实说明
-----------------
文献里的 Point-to-Mesh 需要真值三角网格的点到三角形距离。本包不引入网格资产，
而是用**高密度重采样**（默认 8×）作为代理：P2M_proxy = 点到密集参考点云的距离。
它在高密度极限下收敛到真 P2M，但数值会略偏小（网格离散误差的相反方向）。
所有 baseline 都用同一个代理，所以**方法之间的相对比较是公平的**；
要报绝对数值时请注明用了代理。
"""

import numpy as np

from .noise import _NN

__all__ = ["chamfer", "hausdorff", "estimate_normals", "normal_error",
           "extent_stats", "radius_drift", "p2m", "thickness",
           "n2n_consistency", "self_consistency", "evaluate", "evaluate_n2n"]


def _nn_dist(query, ref, nn=None):
    nn = nn or _NN(ref)
    d, _ = nn.query(np.asarray(query, dtype=np.float64), 1)
    return d[:, 0]


def chamfer(P, Q, nn=None):
    """双向 Chamfer 距离的均值形式。越小越好。"""
    d1 = _nn_dist(P, Q, nn)
    d2 = _nn_dist(Q, P)
    return float(0.5 * (d1.mean() + d2.mean()))


def hausdorff(P, Q):
    """双向 Hausdorff 距离：max( sup_{p∈P} d(p,Q), sup_{q∈Q} d(q,P) )。

    两个方向必须各自在**被查询的那一侧**建索引：
      d(p,Q) 用 Q 的索引，d(q,P) 用 P 的索引。
    早先版本把同一个索引复用到两个方向，第二个方向实际上算成了
    「Q 内部点到 Q 内部点的距离」，恒接近 0 —— 单向 bug 很难从数值上察觉。
    这里不接收外部 nn，各自建索引（评测里这点开销可接受，换来正确性）。
    """
    d1 = _nn_dist(P, Q)          # NN 建于 Q
    d2 = _nn_dist(Q, P)          # NN 建于 P
    return float(max(d1.max(), d2.max()))


def p2m(P, ref, nn=None, pct=None):
    """点到参考曲面的距离。pct=None 取均值，否则取该分位数。"""
    d = _nn_dist(P, ref, nn)
    return float(d.mean() if pct is None else np.percentile(d, pct))


def estimate_normals(P, k=12, nn=None):
    """用 k-NN 局部 PCA 估计法向（最小特征值对应的特征向量）。

    去噪论文里的「法向误差」标准定义就是「估计法向 vs 真值法向的夹角」，
    所以必须真的去估计法向，不能拿位置向量凑。本函数是纯 numpy 的实现。
    """
    P = np.asarray(P, dtype=np.float64)
    nn = nn or _NN(P)
    _, idx = nn.query(P, k)
    nb = P[idx]                                     # [N,k,3]
    nb = nb - nb.mean(axis=1, keepdims=True)
    # 3x3 协方差用 einsum 一次算出来，再批量取最小特征向量
    C = np.einsum('nki,nkj->nij', nb, nb) / k
    w, V = np.linalg.eigh(C)                        # 特征值升序
    n = V[:, :, 0]
    # 统一朝向：用质心指向该点的方向作为参考，避免符号翻转
    ref = P - P.mean(0)
    flip = np.sum(n * ref, axis=1) < 0
    n[flip] *= -1
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n


def normal_error(P_est, P_ref, N_ref, k=12, nn_ref=None):
    """估计法向与真值法向的平均夹角（度）。越小说明结构/特征保持得越好。

    这是相场类方法的卖点指标：平滑噪声的同时保住尖锐特征。
    注意它只对**有真值法向**的解析曲面成立；真实扫描需要人工标定法向。
    """
    nn_ref = nn_ref or _NN(P_ref)
    _, i = nn_ref.query(np.asarray(P_est, dtype=np.float64), 1)
    n_ref = N_ref[i[:, 0]]
    n_est = estimate_normals(P_est, k=k)
    d = np.sum(n_est * n_ref, axis=1)
    return float(np.degrees(np.arccos(np.clip(d, -1, 1))).mean())


def extent_stats(P):
    """相对质心的半径统计：mean / std / p95。

    关于「保体积」指标的诚实说明
    --------------------------
    点云**没有**三角剖分就算不出真正的封闭体积（需要凸包/三角化库）。
    本包刻意不引入 scipy/skl，所以这里报的是**尺度代理**而非真体积：
        mean_radius  质心到点的平均距离 —— 对 sphere/torus/bumpy 这类
                     相对质心「星形」的曲面，它与体积单调相关，漂移百分比可解释。
        radius_drift 去噪前后 mean_radius 的相对变化（%）。
    它的失效场景要清楚：非星形曲面（例如 U 形、多个分离分量）下这个量会失真。
    需要真体积时请自行接 scipy.spatial.ConvexHull 或 trimesh，本函数留了接口位置。
    """
    P = np.asarray(P, dtype=np.float64)
    c = P.mean(0)
    r = np.linalg.norm(P - c, axis=1)
    return {"mean_radius": float(r.mean()),
            "std_radius": float(r.std()),
            "p95_radius": float(np.percentile(r, 95)),
            "centroid": c}


def radius_drift(P_est, P_ref):
    """相对参考曲面的尺度漂移（%）。正值=膨胀，负值=收缩。

    经典相场文献用它来论证「保体积」；深度学习去噪论文通常不报这一项。
    """
    r0 = extent_stats(P_ref)["mean_radius"]
    r1 = extent_stats(P_est)["mean_radius"]
    return float(100.0 * (r1 - r0) / max(abs(r0), 1e-12))


def thickness(P_est, P_ref, nn=None):
    """去噪点到参考曲面的平均距离（越小越贴合）。

    这是「过度平滑 vs 残留噪声」最直观的单一指标：
    过度平滑会让点塌到曲面上（thickness↓但 p2m_p95 可能变差），
    欠去噪则 thickness↑。配合 p2m_p95 一起看能区分这两种失效。
    """
    return p2m(P_est, P_ref, nn)


def n2n_consistency(A, B, nn=None):
    """无标签质量代理：两次独立噪声观测去噪后的一致性（越小通常越好）。

    注意这是**代理**不是真值：一致性高可能意味着「两个结果都停在同一个错的解上」。
    所以它要和有标签指标一起看，或用于选超参时的稳定性排序。
    """
    nn = nn or _NN(B)
    d = _nn_dist(A, B, nn)
    return float(d.mean())


def self_consistency(P, fn, n=2, **kw):
    """固定点距离：连续施加去噪 f，量 ‖f(f(P)) − f(P)‖。

    相场类模型的不动点性质使这个量有明确含义：
    越小说明迭代越接近不动点（即算法已收敛），否则说明还在「动」。
    """
    cur = fn(P, **kw)
    for _ in range(n - 1):
        cur = fn(cur, **kw)
    return float(chamfer(P, cur))


def evaluate(P_est, P_ref, N_ref, ref_dense=None, res_floor=None):
    """有标签全指标。ref_dense 可选；res_floor 是体素量化下限（若有则标注）。

    res_floor 很重要：体素化相场的几何精度被 h/2 卡死，任何小于它的
    CD/P2M 只是在**测分辨率**，不是求解器的真实能力。at_floor 标志就是
    提醒读者「这个数字已经触底了，别再拿来比方法」。
    """
    nn_ref = _NN(P_ref)
    out = {
        "chamfer": chamfer(P_est, P_ref, nn_ref),
        "p2m": p2m(P_est, ref_dense if ref_dense is not None else P_ref),
        "p2m_p95": p2m(P_est, ref_dense if ref_dense is not None else P_ref, pct=95),
        "hausdorff": hausdorff(P_est, P_ref),
        "normal_err": normal_error(P_est, P_ref, N_ref, nn_ref=nn_ref),
        "radius_drift": radius_drift(P_est, P_ref),
    }
    if res_floor:
        out["res_floor"] = res_floor
        out["at_floor"] = bool(out["chamfer"] <= 1.5 * res_floor)
    return out


def evaluate_n2n(P_a, P_b):
    """无标签协议：输入两份「同一曲面的独立噪声观测」，输出可用指标。"""
    return {"raw_consistency": n2n_consistency(P_a, P_b)}
