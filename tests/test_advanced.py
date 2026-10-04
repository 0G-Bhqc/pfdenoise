"""进阶相场求解器、偏差-方差分解协议与 3D 数据 I/O 测试套件。"""

import numpy as np
import pytest

import pfdenoise as pfd
from pfdenoise.noise import _NN


def _make_sphere(n=1200):
    return pfd.make_shape("sphere", n, seed=42)


# --------------------------------------------------------------------------
# 求解器测试
# --------------------------------------------------------------------------
def test_conserved_ac_runs_and_preserves_finite_values():
    P, N, _, _ = _make_sphere(1000)
    nm = pfd.make_noise("gaussian")
    Pn = nm.apply(P, N, np.random.default_rng(1), _NN(P))
    P_est, info = pfd.phase_field_ac_conserved(Pn, N, iters=40)
    assert len(P_est) == len(P)
    assert np.isfinite(P_est).all()
    assert "dt" in info and info["dt"] > 0


def test_aniso_ac_runs():
    P, N, _, _ = _make_sphere(1000)
    nm = pfd.make_noise("gaussian")
    Pn = nm.apply(P, N, np.random.default_rng(2), _NN(P))
    P_est, info = pfd.phase_field_ac_anisotropic(Pn, N, iters=40)
    assert len(P_est) == len(P)
    assert np.isfinite(P_est).all()
    assert "dt" in info


def test_convex_ac_runs():
    P, N, _, _ = _make_sphere(1000)
    nm = pfd.make_noise("gaussian")
    Pn = nm.apply(P, N, np.random.default_rng(3), _NN(P))
    P_est, info = pfd.phase_field_ac_convex(Pn, N, iters=30)
    assert len(P_est) == len(P)
    assert np.isfinite(P_est).all()


def test_convex_adaptive_ac_runs():
    P, N, _, _ = _make_sphere(1000)
    nm = pfd.make_noise("gaussian")
    Pn = nm.apply(P, N, np.random.default_rng(4), _NN(P))
    P_est, info = pfd.phase_field_ac_convex_adaptive(Pn, N, iters=30)
    assert len(P_est) == len(P)
    assert np.isfinite(P_est).all()
    assert "accepted_steps" in info


# --------------------------------------------------------------------------
# 评测指标与无标签协议测试
# --------------------------------------------------------------------------
def test_bias_variance_mathematical_properties():
    P_ref, _, _, _ = _make_sphere(800)
    rng = np.random.default_rng(123)
    # 模拟 3 组带有独立白噪声的预测结果
    preds = [P_ref + rng.normal(0, 0.02, P_ref.shape) for _ in range(3)]
    res = pfd.metrics.bias_variance_decomposition(preds, ref=P_ref)

    assert res["variance"] > 0
    assert res["std_dev"] == pytest.approx(np.sqrt(res["variance"]))
    assert res["consistency"] > 0
    assert "bias_chamfer" in res
    assert "total_mse" in res
    # 检查 MSE = Bias^2 + Var
    expected_mse = res["bias_sq"] + res["variance"]
    assert res["total_mse"] == pytest.approx(expected_mse, rel=1e-6)


def test_volume_drift_and_thickness():
    P, N, _, _ = _make_sphere(1000)
    # 对同一曲面，漂移应接近 0，厚度应接近 0
    vdrift = pfd.metrics.volume_drift(P, P)
    thick = pfd.metrics.thickness(P, P)
    assert abs(vdrift) < 1e-4
    assert thick == pytest.approx(0.0, abs=1e-8)

    # 整体膨胀 10% 尺度，体积应增加约 33.1% (1.1^3 - 1)
    P_expanded = P * 1.1
    vdrift_exp = pfd.metrics.volume_drift(P_expanded, P)
    assert vdrift_exp > 20.0  # 显著正漂移


# --------------------------------------------------------------------------
# 数据 I/O 与端到端管线测试
# --------------------------------------------------------------------------
def test_ply_io_roundtrip(tmp_path):
    P, N, _, _ = _make_sphere(500)
    ply_file = tmp_path / "test_mesh.ply"
    pfd.mesh_io.save_ply(ply_file, P, N)
    assert ply_file.is_file()

    P_loaded, N_loaded, faces = pfd.mesh_io.load_ply(ply_file)
    assert len(P_loaded) == len(P)
    assert np.allclose(P, P_loaded, atol=1e-5)
    assert N_loaded is not None and np.allclose(N, N_loaded, atol=1e-5)


def test_obj_io_roundtrip(tmp_path):
    P, N, _, _ = _make_sphere(500)
    obj_file = tmp_path / "test_mesh.obj"
    pfd.mesh_io.save_obj(obj_file, P, N)
    assert obj_file.is_file()

    P_loaded, N_loaded, _ = pfd.mesh_io.load_obj(obj_file)
    assert len(P_loaded) == len(P)
    assert np.allclose(P, P_loaded, atol=1e-5)


def test_normalize_cloud_roundtrip():
    P = np.array([[10.0, 20.0, 30.0], [12.0, 25.0, 34.0], [8.0, 15.0, 26.0]])
    P_norm, centroid, scale = pfd.mesh_io.normalize_cloud(P, target_scale=1.0)
    assert np.allclose(P_norm.mean(axis=0), [0, 0, 0], atol=1e-12)
    assert np.max(np.ptp(P_norm, axis=0)) == pytest.approx(2.0, abs=1e-9)

    P_restored = (P_norm / scale) + centroid
    assert np.allclose(P, P_restored, atol=1e-12)


def test_denoise_file_pipeline(tmp_path):
    P, N, _, _ = _make_sphere(600)
    in_file = tmp_path / "in.ply"
    out_file = tmp_path / "out.ply"
    pfd.mesh_io.save_ply(in_file, P, N)

    res = pfd.mesh_io.denoise_file(in_file, out_file, solver="laplacian", iters=2)
    assert out_file.is_file()
    assert res["input_points"] == 600
    assert res["output_points"] == 600

    P_res, _, _ = pfd.mesh_io.load_point_cloud(out_file)
    assert len(P_res) == 600
    assert np.isfinite(P_res).all()
