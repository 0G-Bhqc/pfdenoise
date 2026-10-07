r"""test_regression_fixes.py — 2026-10 正确性专项回归测试

守住本轮修复涉及的六类真实缺陷：
  1. _NN 的精确性（与暴力 O(N²) 交叉验证，含孤立点/云外查询）
  2. torus / superquadric 采样死循环与法向几何（符号、朝向、方向不变夹角）
  3. 噪声模型「实现与自述统计性质一致」（本包的核心契约）
  4. CH 零均值保真项的格式级质量守恒
  5. CFL / ch_cfl 守卫在开跑前拒绝不稳定时间步
  6. bilateral 守卫、各向异性体素化、self_consistency 元组拆包
"""

import numpy as np
import pytest

import pfdenoise as pfd
from pfdenoise.noise import _NN, Aniso, Density, NoiseModel, Protrusion, make_noise
from pfdenoise.solvers import _run_ac, _run_ch, _voxelize, bilateral


def _brute_knn(P, Q, k):
    """暴力 k 近邻（O(N²)），作为 _NN 的金标准。"""
    d = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(d, idx, axis=1)


# ---------------------------------------------------------------- 最近邻精确性
@pytest.mark.parametrize("k", [1, 2, 5, 12])
def test_nn_matches_bruteforce_on_adversarial_cloud(k):
    rng = np.random.default_rng(7)
    P = rng.normal(size=(300, 3))
    P[0] = P[1]              # 重复点
    P[2] += 100.0            # 孤立点：触发跨格扩环查找
    nn = _NN(P)
    d_got, _ = nn.query(P, k)
    d_ref = _brute_knn(P, P, k)
    assert np.allclose(d_got[:, :k], d_ref[:, :k], atol=1e-10)


def test_nn_exact_for_query_outside_cloud():
    P, _, _, _ = pfd.make_shape("torus", 600, seed=3)
    nn = _NN(P)
    Q = np.vstack([P[:100], P[:20] + np.array([5.0, 0, 0])])
    d_got, _ = nn.query(Q, 3)
    d_ref = _brute_knn(Q, P, 3)
    assert np.allclose(d_got, d_ref, atol=1e-10)


# ---------------------------------------------------------------- 曲面法向几何
@pytest.mark.parametrize("shape", ["sphere", "torus", "superquadric", "bumpy"])
def test_all_shapes_sample_and_unit_normals(shape):
    """torus/superquadric 曾是死循环/法向错误的重灾区，必须全部可采样。"""
    P, N, _, _ = pfd.make_shape(shape, 1500, seed=0)
    assert len(P) == 1500
    assert np.allclose(np.linalg.norm(N, axis=1), 1.0, atol=1e-9)


def test_torus_volume_fn_matches_analytic():
    """torus 的体积函数应与解析值 2π²Rr² 一致（顺带守住采样修复）。"""
    P, N, vfn, _ = pfd.make_shape("torus", 1000, seed=0)
    assert vfn() == pytest.approx(2 * np.pi ** 2 * 1.0 * 0.38 ** 2, rel=1e-9)


def test_superquadric_normals_point_outward():
    """回归：早先用 |x|^(e-1) 求梯度丢了符号，负象限法向全部镜像。"""
    P, N, _, _ = pfd.make_shape("superquadric", 2000, seed=0)
    outward_ratio = float((np.sum(N * P, axis=1) > 0).mean())
    assert outward_ratio > 0.98, f"外向法向占比过低: {outward_ratio:.3f}"


def test_normal_error_is_sign_invariant():
    """回归：法向夹角必须方向不变（|n·n|）。

    早先版本不取绝对值，而「质心指向外」的统一朝向启发式在 torus 内环
    会系统性翻转 —— 估计得很好的法向被判成 ~90° 误差。
    """
    P, N, _, _ = pfd.make_shape("torus", 1200, seed=1)
    e0 = pfd.metrics.normal_error(P, P, N)
    assert e0 < 15.0, f"干净 torus 的法向误差异常: {e0:.2f}°"
    N_flipped = N.copy()
    N_flipped[:600] *= -1              # 人为翻转一半法向的符号
    e1 = pfd.metrics.normal_error(P, P, N_flipped)
    assert abs(e1 - e0) < 1.0, f"方向不变性被破坏: {e0:.2f}° vs {e1:.2f}°"


# ---------------------------------------------------------------- 噪声模型性质
def test_gaussian_zero_mean():
    P, N, _, _ = pfd.make_shape("sphere", 6000, seed=0)
    disp = make_noise("gaussian").apply(P, N, np.random.default_rng(0)) - P
    assert abs(disp.mean()) < 0.002


def test_protrusion_is_net_outward():
    """回归：早先概率写反（p=[0.8,0.2] 对 80% 向内推），与「系统外推」声明相反。"""
    P, N, _, _ = pfd.make_shape("sphere", 6000, seed=0)
    dn = np.sum((Protrusion(0.4, 0.05).apply(P, N, np.random.default_rng(0)) - P) * N,
                axis=1)
    moved = np.abs(dn) > 1e-12
    assert moved.mean() > 0.3
    assert dn[moved].mean() > 0, "净法向位移应为正（外推偏置），实测向内"


def test_aniso_offsets_tangential_and_spatially_correlated():
    """回归：早先「低频场」是把噪声与它自己的最近邻（自身）平均 —— 空操作。"""
    P, N, _, _ = pfd.make_shape("sphere", 4000, seed=0)
    rng = np.random.default_rng(0)
    Pa = Aniso(sigma_t=0.02, sigma_n=0.0, smooth=0.9).apply(P, N, rng, _NN(P))
    off = Pa - P
    assert np.abs(np.sum(off * N, axis=1)).max() < 1e-10, "纯切向配置不应有法向分量"
    _, nb = _NN(P).query(P, 3)
    n_off = np.linalg.norm(off, axis=1)
    cos_nb = np.sum(off * off[nb[:, 1]], axis=1) / (n_off * n_off[nb[:, 1]] + 1e-12)
    perm = rng.permutation(len(P))
    cos_rd = np.sum(off * off[perm], axis=1) / (n_off * n_off[perm] + 1e-12)
    assert cos_nb.mean() > cos_rd.mean() + 0.2, (
        f"邻域偏移方向相似度应显著高于随机对: {cos_nb.mean():.3f} vs {cos_rd.mean():.3f}")


def test_density_drops_points_and_respects_seed():
    P, N, _, _ = pfd.make_shape("sphere", 2000, seed=0)
    a = Density(0.25, seed=5).apply(P, N, None)
    b = Density(0.25, seed=5).apply(P, N, None)
    assert len(a) == 1500
    assert np.array_equal(a, b), "相同 seed 应可复现"


def test_density_registered_and_declares_assumption():
    m = make_noise("density")
    assert m.name == "density"
    assert "丢弃" in m.assumption_text()


def test_base_noise_model_demands_assumption():
    """「噪声是一等公民」的前提是假设显式 —— 基类不许静默返回空串。"""
    with pytest.raises(NotImplementedError):
        NoiseModel().assumption_text()


# ---------------------------------------------------------------- CH 质量守恒
def test_ch_mass_conservation_format_level():
    """保真项零均值化后 d/dt∫φ=0 应逐格式精确成立。

    回归：早先把 λ(T−φ) 裸加在 ∇² 外面，∫φ 随演化漂移 ——
    「保体积是格式级保证」的说法对那份代码在数学上不成立。
    """
    n = 16
    h = 0.1
    T = np.full((n, n, n), -1.0)
    T[4:12, 4:12, 4:12] = 1.0
    phi, dt, cfl = _run_ch(None, h, 40, 2.0 * h, 0.8, T=T)
    assert abs(phi.sum() - T.sum()) < 1e-8, (
        f"质量漂移 {abs(phi.sum() - T.sum()):.3g} 超过机器精度级")


# ---------------------------------------------------------------- CFL 守卫
def test_run_ac_rejects_unstable_cfl():
    """回归：早先报错把 dt 上界错标成 cfl 上界，且 cfl 参数未暴露。"""
    h = 0.1
    T = np.full((10, 10, 10), -0.5)
    with pytest.raises(ValueError, match="cfl"):
        _run_ac(None, h, 5, 2.0 * h, 0.8, T=T, cfl=0.3)


def test_ch_rejects_unsafe_ch_cfl():
    h = 0.1
    T = np.full((10, 10, 10), -0.5)
    with pytest.raises(ValueError, match="ch_cfl"):
        _run_ch(None, h, 5, 2.0 * h, 0.8, T=T, ch_cfl=1.2)


# ---------------------------------------------------------------- 杂项回归
def test_bilateral_runs_and_guard_raises():
    P, N, _, _ = pfd.make_shape("sphere", 500, seed=0)
    out = bilateral(P, N, iters=2)
    assert out.shape == P.shape
    assert np.isfinite(out).all()
    with pytest.raises(RuntimeError, match="bilateral"):
        bilateral(P, N, iters=1, max_points=10)


def test_voxelize_long_axis_not_clipped():
    """回归：早先用平均轴长定 h，长轴坐标越界被 clip 压进最后一层体素。"""
    rng = np.random.default_rng(0)
    P = rng.uniform([0, 0, 0], [2.0, 2.0, 0.5], size=(2000, 3))   # 扁平盒
    n_vox = 20
    lo, hi = P.min(0) - 1e-6, P.max(0) + 1e-6
    occ, h = _voxelize(P, n_vox, lo, hi)
    assert h >= (hi - lo).max() / n_vox - 1e-12
    idx = np.floor((P - lo) / h).astype(np.int64)   # clip 之前就不应越界
    assert idx.max() < n_vox


def test_self_consistency_handles_tuple_returning_solvers():
    """回归：pf 求解器返回 (points, info)，不拆包时该函数直接崩。"""
    P, _, _, _ = pfd.make_shape("sphere", 400, seed=0)

    def fake_fn(X, **kw):
        return X + 0.01, {"dt": 0.1}

    val = pfd.metrics.self_consistency(P, fake_fn, n=3)
    assert np.isfinite(val) and val > 0
    with pytest.raises(ValueError):
        pfd.metrics.self_consistency(P, fake_fn, n=1)


def test_extract_surface_no_shift_on_anisotropic_bbox():
    """回归（D-REP-003 变体）：亚格点重心校验必须逐轴用 lo，不能用 lo.mean(0)。

    立方包围盒下标量均值无症状，但 torus 这类扁平包围盒（lo 的 z 分量与
    三轴均值差 0.66 个世界单位）会让整个重建平移出曲面。
    """
    from pfdenoise.solvers import _extract_surface
    rng = np.random.default_rng(0)
    # 各向异性包围盒内放一个已知位置的 blob
    lo = np.array([-1.4, -1.4, -0.4])
    h = 0.05
    grid = (56, 56, 16)
    phi = np.full(grid, -1.0)
    phi[20:36, 20:36, 4:12] = 1.0                      # 索引重心 (27.5, 27.5, 7.5)
    blob_center_world = lo + (np.array([27.5, 27.5, 7.5])) * h
    P_probe = blob_center_world + rng.normal(0, 0.02, (100, 3))
    out = _extract_surface(phi, h, lo, P_probe, np.zeros_like(P_probe))
    assert np.linalg.norm(out.mean(0) - blob_center_world) < 3 * h, (
        f"提取重心偏离 blob 中心过多: {out.mean(0)} vs {blob_center_world}")


def test_benchmark_on_previously_untested_shapes():
    """torus / superquadric 此前从未进过基准 —— 给它们端到端覆盖。"""
    rows = pfd.run_benchmark(shape="torus", n=1200, noise="gaussian",
                             solver_names=["none", "pf_ac_convex"], verbose=False)
    assert all("error" not in r for r in rows), rows
    rows2 = pfd.run_benchmark(shape="superquadric", n=1200, noise="gaussian",
                              solver_names=["none", "laplacian"], verbose=False)
    assert all("error" not in r for r in rows2), rows2
