"""benchmark.py — 基准编排：跑表、出表、无标签协议

三件事
------
run_benchmark()  在「曲面 × 噪声 × 求解器」上跑一遍，出一张可直接贴进论文的表
run_n2n()        无标签协议：只给两份独立噪声观测，不给真值
make_table()     把结果整理成 markdown / CSV
"""

import csv
import io
import time

import numpy as np

from . import metrics as M
from . import solvers as S
from .noise import _NN, make_noise
from .shapes import make_shape

__all__ = ["run_benchmark", "run_n2n", "make_table", "DEFAULT_SOLVERS"]

DEFAULT_SOLVERS = ["none", "laplacian", "bilateral",
                   "pf_ac", "pf_ac_fidelity", "pf_ac_barrier", "cahn_hilliard"]

# 越小越好的指标
LOWER_BETTER = {"chamfer", "p2m", "p2m_p95", "hausdorff",
                "normal_err", "radius_drift", "raw_consistency"}
# 相对真值的绝对值，漂移类指标要看 |·|
ABS_METRICS = {"radius_drift"}


def _obs(P_ref, N_ref, noise, seed, shape, shape_seed, dense_mult=4):
    """生成一次噪声观测 + 密集参考采样。"""
    rng = np.random.default_rng(seed)
    nn = _NN(P_ref)
    Pn = noise.apply(P_ref, N_ref, rng, nn)
    P_dense, N_dense, _, _ = make_shape(shape, n=int(len(P_ref) * dense_mult),
                                        seed=shape_seed + 991)
    return Pn, P_dense, N_dense


def run_benchmark(shape="bumpy", n=6000, noise="realistic", seed=0,
                  solver_names=None, solver_kwargs=None, dense_mult=4,
                  verbose=True):
    """跑一遍有标签基准，返回 rows（list[dict]）。

    每个 row: shape, noise, solver, <各指标>, seconds
    """
    solver_names = solver_names or DEFAULT_SOLVERS
    solver_kwargs = solver_kwargs or {}

    P_ref, N_ref, vfn, scale = make_shape(shape, n, seed)
    noise_model = make_noise(noise)
    Pn, P_dense, N_dense = _obs(P_ref, N_ref, noise_model, seed + 1, shape, seed,
                                dense_mult)

    if verbose:
        print(f"曲面={shape}  n={n}  噪声={noise}")
        print(f"  噪声假设: {noise_model.assumption_text()}")
        print(f"  N2N 理论适用: {'是' if noise_model.n2n_compatible else '否'}")
        print(f"  bbox 半宽={scale:.3f}\n")

    rows = []
    for name in solver_names:
        fn = S.get_solver(name)
        kw = solver_kwargs.get(name, {})
        t0 = time.perf_counter()
        try:
            out = fn(Pn, N_ref, **kw)
            if isinstance(out, tuple):
                P_est, info = out
            else:
                P_est, info = out, {}
        except Exception as e:                       # noqa: BLE001
            rows.append(dict(shape=shape, noise=noise, solver=name,
                             error=f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"  {name:<18} FAILED: {e}")
            continue
        dt_s = time.perf_counter() - t0
        # 体素化求解器要报分辨率下限：CD/P2M 触到 h/2 就是在测分辨率而非精度
        res_floor = None
        if name.startswith("pf_") or name == "cahn_hilliard":
            try:
                _, h = S.auto_resolution(Pn)
                res_floor = 0.5 * h
            except Exception:                          # noqa: BLE001
                res_floor = None
        m = M.evaluate(P_est, P_ref, N_ref, ref_dense=P_dense, res_floor=res_floor)
        m.update(shape=shape, noise=noise, solver=name, seconds=dt_s,
                 n_out=len(P_est))
        if "cfl" in info:
            m["cfl"] = info["cfl"]
        rows.append(m)
        if verbose:
            print(f"  {name:<18} CD={m['chamfer']:.4f}  P2M={m['p2m']:.4f}  "
                  f"HD={m['hausdorff']:.4f}  nErr={m['normal_err']:.2f}deg  "
                  f"rDrift={m['radius_drift']:+.1f}%  ({dt_s:.2f}s)")
    return rows


def run_n2n(shape="bumpy", n=6000, noise="gaussian", seed=0,
            solver_names=None, solver_kwargs=None, verbose=True):
    """无标签协议：两份独立噪声观测，不给真值，看结果一致性。

    这是本包相对「纯监督评测」的核心增量：
    真实传感器往往拿不到成对脏-净数据，而 N2N 类方法的前提是
    **同一曲面的两次独立零均值噪声观测**。本函数把这个前提显式化。
    """
    solver_names = solver_names or DEFAULT_SOLVERS
    solver_kwargs = solver_kwargs or {}

    P_ref, N_ref, _, _ = make_shape(shape, n, seed)
    nm = make_noise(noise)
    nn = _NN(P_ref)
    rng = np.random.default_rng(seed + 1)
    A = nm.apply(P_ref, N_ref, rng, nn)
    rng = np.random.default_rng(seed + 2)             # 独立第二次观测
    B = nm.apply(P_ref, N_ref, rng, nn)

    if verbose:
        print(f"无标签协议  曲面={shape}  噪声={noise}")
        print(f"  噪声假设: {nm.assumption_text()}")
        print(f"  N2N 理论适用: {'是' if nm.n2n_compatible else '否 —— 一致性数值不可按 N2N 解读'}")
        print(f"  原始两观测一致性 = {M.n2n_consistency(A, B):.4f}（应作为下界参考）\n")

    rows = []
    for name in solver_names:
        fn = S.get_solver(name)
        kw = solver_kwargs.get(name, {})
        try:
            oa = fn(A, N_ref, **kw)
            ob = fn(B, N_ref, **kw)
            Pa = oa[0] if isinstance(oa, tuple) else oa
            Pb = ob[0] if isinstance(ob, tuple) else ob
            cons = M.n2n_consistency(Pa, Pb)
        except Exception as e:                       # noqa: BLE001
            rows.append(dict(solver=name, error=f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"  {name:<18} FAILED: {e}")
            continue
        rows.append(dict(solver=name, consistency=cons))
        if verbose:
            print(f"  {name:<18} 去噪后一致性 = {cons:.4f}")
    return rows


COLUMNS = [("solver", "solver", "str"), ("chamfer", "CD↓", "num"),
           ("p2m", "P2M↓", "num"), ("p2m_p95", "P2M95↓", "num"),
           ("hausdorff", "HD↓", "num"), ("normal_err", "nErr↓°", "num"),
           ("radius_drift", "rDrift%", "num"), ("at_floor", "触底", "flag"),
           ("seconds", "sec", "num")]


def make_table(rows, fmt="markdown", title=None):
    """把 rows 变成 markdown 或 csv 文本。"""
    cols = [c for c in COLUMNS if any(c[0] in r for r in rows)]
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([c[1] for c in cols])
        for r in rows:
            w.writerow([r.get(c[0], "") for c in cols])
        return buf.getvalue()

    out = []
    if title:
        out.append(f"**{title}**\n")
    out.append("| " + " | ".join(c[1] for c in cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        cells = []
        for key, label, kind in cols:
            v = r.get(key, "")
            if kind == "num" and v != "":
                sign = f"{v:+.2f}" if key in ABS_METRICS else f"{v:.4f}"
                cells.append(sign)
            elif kind == "flag":
                cells.append("⚠" if v else "")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
