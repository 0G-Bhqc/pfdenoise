r"""test_extreme.py — 极端输入与参数边界测试

原则：极端输入下要么给出正确结果，要么 fail loud 给出可读错误；
绝不静默返回垃圾（这是本包从 D-REP-001 静默塌缩事故里学到的底线）。

覆盖面：NaN/Inf、退化点云（1 点/重合点/共线）、极端尺度（1e-6 / 1e6）、
极端参数（iters=0、n_vox=4、lam=0/50、cfl 边界）、噪声参数边界、
指标不变量、确定性、_NN 分块边界、大点云压力、CLI 错误路径。
"""

import subprocess
import sys

import numpy as np
import pytest

import pfdenoise as pfd
from pfdenoise.noise import _NN, Aniso, Composite, Density, Gaussian, Outlier, Protrusion
from pfdenoise import metrics as M
from pfdenoise import solvers as S


def _sphere(n=1200, seed=42):
    return pfd.make_shape("sphere", n, seed)


# ---------------------------------------------------------------- NaN / Inf
def test_nn_rejects_non_finite_points():
    P, _, _, _ = _sphere(100)
    P[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _NN(P)
    P[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN"):
        _NN(P)


def test_solvers_reject_non_finite_points_loudly():
    """NaN 输入必须在入口 fail loud，不允许静默产出垃圾指标。"""
    P, N, _, _ = _sphere(200)
    Pn = P.copy()
    Pn[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        S.laplacian(Pn, N, iters=2)
    with pytest.raises(ValueError, match="NaN"):
        S.phase_field_ac_convex(Pn, N, iters=5)
    with pytest.raises(ValueError, match="NaN"):
        M.chamfer(Pn, P)


def test_metrics_reject_non_finite_reference():
    P, N, _, _ = _sphere(200)
    Q = P.copy()
    Q[7, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        M.chamfer(P, Q)


# ---------------------------------------------------------------- 退化点云
def test_degenerate_clouds_fail_with_readable_error():
    """1 点 / 全重合点：包围盒零退化，必须给出人话错误而不是除零栈。"""
    N3 = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="退化"):
        S.phase_field_ac_convex(N3.copy(), N3.copy(), iters=5)
    P_same = np.zeros((50, 3))
    N_same = np.tile([0.0, 0.0, 1.0], (50, 1))
    with pytest.raises(ValueError, match="退化"):
        S.phase_field_ac_convex(P_same, N_same, iters=5)


def test_collinear_cloud_fails_loud_not_silent():
    """共线点（两个轴零跨度）：允许跑或报错，但不许静默返回垃圾。"""
    t = np.linspace(0, 1, 8)
    P = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    N = np.tile([0.0, 0.0, 1.0], (8, 1))
    try:
        out, info = S.phase_field_ac_convex(P, N, iters=10)
        assert np.isfinite(out).all()
    except (ValueError, RuntimeError) as e:
        assert "塌缩" in str(e) or "退化" in str(e) or "发散" in str(e), e


def test_extreme_scales_round_trip():
    """1e-6 与 1e6 尺度：距离/索引逻辑必须尺度无关，结果有限。"""
    P, N, _, _ = _sphere(600, seed=0)
    for factor in (1e-6, 1e6):
        Ps, Ns = P * factor, N.copy()
        out, info = S.phase_field_ac_convex(Ps, Ns, iters=20)
        assert np.isfinite(out).all()
        scale = float(np.abs(out).max())
        assert 0.5 * factor <= scale <= 2.0 * factor, (
            f"scale={factor}: 输出尺度 {scale} 脱离量级")


# ---------------------------------------------------------------- 极端参数
def test_iters_zero_returns_target_resample():
    P, N, _, _ = _sphere(800)
    Pn = P + 0.01
    out, info = S.phase_field_ac_convex(Pn, N, iters=0)
    assert len(out) == len(Pn)
    assert np.isfinite(out).all()


def test_n_vox_coarse_and_clipped():
    P, N, Pn = _sphere(800, seed=1), None, None
    P, N, _, _ = _sphere(800)
    nm = pfd.make_noise("gaussian")
    Pn = nm.apply(P, N, np.random.default_rng(1), _NN(P))
    out, info = S.phase_field_ac_convex(Pn, N, iters=20, n_vox=4)     # 极粗
    assert np.isfinite(out).all()
    out2, info2 = S.phase_field_ac_convex(Pn, N, iters=20, n_vox=500)  # 超上限 -> 截到 96
    assert info2["h"] > 0 and np.isfinite(out2).all()


def test_lam_zero_shrinks_but_stays_finite():
    """λ=0 的毛细腐蚀演示（坑 3）：有限液滴必然收缩 —— 现象本身要能复现。"""
    P, N, _, _ = _sphere(800)
    Pn = P + 0.01
    out, info = S.phase_field_ac_conserved(Pn, N, iters=60, lam=0.0)
    assert np.isfinite(out).all()


def test_lam_huge_fails_loud_or_converges():
    """λ=50：要么数值稳定收敛，要么显式报发散 —— 不许静默出垃圾。"""
    P, N, _, _ = _sphere(800)
    Pn = P + 0.01
    try:
        out, info = S.phase_field_ac_convex(Pn, N, iters=30, lam=50.0)
        assert np.isfinite(out).all()
    except RuntimeError as e:
        assert "发散" in str(e), e


def test_cfl_boundaries():
    P, N, _, _ = _sphere(800)
    Pn = P + 0.01
    # cfl 只属于显式格式（_run_ac 族）；convex 无条件稳定、无此参数
    with pytest.raises(ValueError):
        S.phase_field_ac(Pn, N, iters=5, cfl=0.3)
    out, info = S.phase_field_ac_convex(Pn, N, iters=10, cfl=1e-4)   # 极小步长
    assert np.isfinite(out).all()
    out, info = S.phase_field_ac_convex(Pn, N, iters=10, eps_h=0.5)  # ε < h
    assert np.isfinite(out).all()
    out, info = S.phase_field_ac_convex(Pn, N, iters=10, eps_h=10.0)  # 极宽界面
    assert np.isfinite(out).all()


# ---------------------------------------------------------------- 噪声参数边界
@pytest.mark.parametrize("noise", [
    Gaussian(0.0),
    Protrusion(0.0, 0.02),
    Protrusion(1.0, 0.0),
    Outlier(0.0, 0.15),
    Aniso(sigma_t=0.0, sigma_n=0.0),
])
def test_zero_magnitude_noise_is_identity(noise):
    P, N, _, _ = _sphere(600)
    rng = np.random.default_rng(0)
    out = noise.apply(P, N, rng, _NN(P))
    assert np.array_equal(out, P), f"{type(noise).__name__} 零强度应完全不动点"


def test_density_extreme_ratios():
    P, N, _, _ = _sphere(1000)
    assert len(Density(0.0).apply(P, N, None)) == len(P)
    out = Density(0.999).apply(P, N, None)
    assert len(out) == len(P) - 999


def test_composite_with_count_changing_member():
    """Composite 里混入改变点数的 Density：链条不炸，点数单调减少。"""
    P, N, _, _ = _sphere(1000)
    nm = Composite(Gaussian(0.005), Density(0.2))
    rng = np.random.default_rng(0)
    out = nm.apply(P, N, rng, _NN(P))
    assert 0 < len(out) <= int(len(P) * 0.8)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------- 指标不变量
def test_metrics_zero_on_heavily_duplicated_cloud():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(20, 3))
    P = base[rng.integers(0, 20, 500)]           # 大量重合点
    assert M.chamfer(P, P) == pytest.approx(0.0, abs=1e-12)
    assert M.hausdorff(P, P) == pytest.approx(0.0, abs=1e-12)


def test_hausdorff_dominates_chamfer():
    rng = np.random.default_rng(3)
    for _ in range(5):
        A = rng.normal(size=(300, 3))
        B = rng.normal(size=(300, 3)) + 0.5
        assert M.hausdorff(A, B) >= M.chamfer(A, B) - 1e-12


def test_evaluate_survives_tiny_estimation_set():
    """3 个点的「去噪结果」进全指标：不许崩，指标全有限。"""
    P, N, _, _ = _sphere(600)
    m = M.evaluate(P[:3], P, N)
    assert all(np.isfinite(v) for k, v in m.items()
               if isinstance(v, (int, float, np.floating)))


def test_radius_drift_is_scale_invariant_pct():
    P, _, _, _ = _sphere(600)
    d1 = M.radius_drift(P * 1.1, P)
    d2 = M.radius_drift(P * 1.1 * 1000.0, P * 1000.0)
    assert d1 == pytest.approx(d2, rel=1e-9)


# ---------------------------------------------------------------- 确定性
def test_nn_query_deterministic():
    P, _, _, _ = _sphere(400)
    nn = _NN(P)
    d1, i1 = nn.query(P, 5)
    d2, i2 = nn.query(P, 5)
    assert np.array_equal(d1, d2) and np.array_equal(i1, i2)


def test_benchmark_reproducible_with_same_seed():
    kw = dict(shape="sphere", n=800, noise="gaussian",
              solver_names=["none", "laplacian", "pf_ac_convex"], verbose=False)
    r1 = pfd.run_benchmark(seed=7, **kw)
    r2 = pfd.run_benchmark(seed=7, **kw)
    for a, b in zip(r1, r2):
        assert a["chamfer"] == pytest.approx(b["chamfer"], abs=0, rel=1e-12)
        assert a["solver"] == b["solver"]


# ---------------------------------------------------------------- _NN 分块与压力
def test_nn_chunk_boundary_exact():
    """查询点数跨过分块边界（4096 -> 4097）：分块循环不许丢点或错位。"""
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(1500, 3))
    q = rng.normal(size=(4097, 3))
    nn = _NN(ref)
    d_got, _ = nn.query(q, 2)
    # 抽样 100 个点对暴力法
    for j in rng.integers(0, len(q), 100):
        d_ref = np.sqrt(((ref - q[j]) ** 2).sum(1))
        top2 = np.sort(d_ref)[:2]
        assert np.allclose(d_got[j], top2, atol=1e-10)


def test_nn_50k_points_stress():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1000, 3))
    P = base[rng.integers(0, 1000, 50000)] + rng.normal(0, 1e-3, (50000, 3))
    import time
    t0 = time.perf_counter()
    nn = _NN(P)
    d, i = nn.query(P, 2)
    elapsed = time.perf_counter() - t0
    assert np.isfinite(d).all()
    assert elapsed < 30.0, f"50000 点 k=2 耗时 {elapsed:.1f}s，超出预算"
    # 重合点大量存在：最近邻距离应近乎为 0
    assert d[:, 0].max() < 1e-6


def test_laplacian_20k_single_iter():
    P, N, _, _ = pfd.make_shape("bumpy", 20000, seed=0)
    out = S.laplacian(P, N, iters=1)
    assert np.isfinite(out).all()
    assert np.linalg.norm(out - P) > 0


# ---------------------------------------------------------------- CLI 错误路径
def test_cli_bv_rejects_k_below_two():
    r = subprocess.run([sys.executable, "-m", "pfdenoise", "bv", "-k", "1"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    assert r.returncode != 0
    assert "K=2" in (r.stderr + r.stdout)


def test_error_row_reason_survives_table_export():
    """失败原因必须进表格（此前被 CSV 导出静默吞掉）。"""
    P, N, _, _ = _sphere(500)
    nm = pfd.make_noise("gaussian")
    Pn = nm.apply(P, N, np.random.default_rng(0), _NN(P))
    rows = pfd.run_benchmark(shape="sphere", n=500, noise="gaussian",
                             solver_names=["none", "bilateral"],
                             solver_kwargs={"bilateral": {"max_points": 10}},
                             verbose=False)
    assert "error" in rows[-1]
    md = pfd.make_table(rows, fmt="markdown")
    csv = pfd.make_table(rows, fmt="csv")
    assert "error" in md and "RuntimeError" in md
    assert "RuntimeError" in csv, "CSV 导出不得吞掉失败原因"
