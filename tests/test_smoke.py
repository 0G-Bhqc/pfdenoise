"""冒烟 + 不变量测试。

这些测试主要守住三件在开发期真实踩过的事：
  1. 最近邻不能把自己当邻居（否则拉普拉斯平滑静默失效）
  2. 相场塌缩必须报错，不能静默返回输入点
  3. 指标要满足基本不变量（恒等输入误差为 0、保体积方法漂移≈0）
"""

import numpy as np
import pytest

import pfdenoise as pfd
from pfdenoise.noise import _NN


def _cloud(n=1500, seed=0, shape="bumpy"):
    return pfd.make_shape(shape, n, seed)


# ---------------------------------------------------------------- 最近邻
def test_nn_finds_self_first():
    P, _, _, _ = _cloud()
    nn = _NN(P)
    d, i = nn.query(P, 1)
    assert np.allclose(d[:, 0], 0.0, atol=1e-12), "最近邻应是自己"
    assert np.array_equal(i[:, 0], np.arange(len(P)))


def test_nn_excludes_self_when_k2():
    P, _, _, _ = _cloud()
    nn = _NN(P)
    d, i = nn.query(P, 2)
    assert d[:, 1].min() > 0, "第 2 近邻不应是自己"
    assert np.all(i[:, 0] != i[:, 1])


def test_nn_k12_covers_more_than_k1():
    P, _, _, _ = _cloud(800)
    nn = _NN(P)
    d1, _ = nn.query(P, 1)
    d12, _ = nn.query(P, 12)
    assert d12[:, -1].mean() > d1[:, 0].mean()


# ---------------------------------------------------------------- 指标不变量
def test_metrics_zero_on_identical_cloud():
    P, N, _, _ = _cloud()
    m = pfd.metrics.evaluate(P, P, N, ref_dense=P)
    assert m["chamfer"] == pytest.approx(0.0, abs=1e-9)
    assert m["hausdorff"] == pytest.approx(0.0, abs=1e-9)
    assert m["radius_drift"] == pytest.approx(0.0, abs=1e-6)


def test_estimate_normals_recovers_analytic_normals():
    P, N, _, _ = _cloud(2000, shape="sphere")
    est = pfd.metrics.estimate_normals(P, k=16)
    ang = np.degrees(np.arccos(np.clip(np.sum(est * N, axis=1), -1, 1)))
    assert ang.mean() < 5.0, f"球面法向估计误差应很小，实测 {ang.mean():.2f}°"


def test_hausdorff_is_two_sided():
    P, _, _, _ = _cloud(600)
    Q = P.copy()
    Q[0] += 0.5                                  # 只挪一个点
    # 单向会漏掉这个点，双向必须抓到
    assert pfd.metrics.hausdorff(P, Q) > 0.3


# ---------------------------------------------------------------- 噪声
@pytest.mark.parametrize("name", list(pfd.NOISE_MODELS))
def test_noise_models_declare_assumptions(name):
    m = pfd.make_noise(name)
    assert isinstance(m.assumption_text(), str) and len(m.assumption_text()) > 10
    assert isinstance(m.n2n_compatible, bool)


def test_protrusion_is_biased_and_flagged():
    m = pfd.make_noise("protrusion")
    assert m.n2n_compatible is False
    assert "有偏" in m.assumption_text()


def test_gaussian_is_unbiased_and_flagged():
    m = pfd.make_noise("gaussian")
    assert m.n2n_compatible is True


# ---------------------------------------------------------------- 求解器
def test_identity_returns_input():
    P, N, _, _ = _cloud(400)
    out = pfd.solvers.identity(P, N)
    assert np.allclose(out, P)


def test_laplacian_actually_moves_points():
    """回归测试：早期版本查最近邻时查到自己，更新量恒为 0。"""
    P, N, _, _ = _cloud(400)
    out = pfd.solvers.laplacian(P, N, iters=5)
    assert np.linalg.norm(out - P) > 1e-6, "拉普拉斯平滑必须真的移动点"


def test_extract_surface_raises_when_phase_collapses():
    """回归测试：正相塌缩时必须抛错，不能静默返回输入点。

    这里直接测守卫本身，而不是靠「调一个很粗的 n_vox 去撞」。
    早期用 n_vox=8 试过，反而不会塌（粗网格下整个域变成一个实心团），
    真正触发塌缩的是**过细**的网格 + 稀疏壳，所以那个测试前提本身就是错的。
    """
    from pfdenoise.solvers import _extract_surface
    P, N, _, _ = _cloud(400)
    lo = P.min(0) - 1e-6
    dead = np.full((16, 16, 16), -0.9)          # 正相已消失
    with pytest.raises(RuntimeError, match="塌缩"):
        _extract_surface(dead, 0.1, lo, P, N)


def test_extract_surface_returns_input_count():
    from pfdenoise.solvers import _extract_surface
    P, N, _, _ = _cloud(400)
    lo = P.min(0) - 1e-6
    phi = np.full((24, 24, 24), -1.0)
    phi[8:16, 8:16, 8:16] = 1.0
    out = _extract_surface(phi, 0.1, lo, P, N)
    assert len(out) == len(P)


def test_cahn_hilliard_does_not_diverge():
    P, N, _, _ = _cloud(3000)
    nm = pfd.make_noise("gaussian")
    Pn = nm.apply(P, N, np.random.default_rng(1), _NN(P))
    out, info = pfd.solvers.cahn_hilliard(Pn, N, iters=60)
    assert np.isfinite(out).all()
    assert info["cfl"] <= 1.0 + 1e-9


# ---------------------------------------------------------------- 端到端
def test_benchmark_runs_and_returns_rows():
    rows = pfd.run_benchmark(shape="sphere", n=1200, noise="gaussian",
                            solver_names=["none", "laplacian"], verbose=False)
    assert len(rows) == 2
    for r in rows:
        assert "error" not in r, r.get("error")
        assert np.isfinite(r["chamfer"])


def test_n2n_protocol_reports_consistency():
    rows = pfd.run_n2n(shape="sphere", n=1200, noise="gaussian",
                       solver_names=["none"], verbose=False)
    assert len(rows) == 1
    assert np.isfinite(rows[0]["consistency"])


def test_make_table_markdown_and_csv():
    rows = pfd.run_benchmark(shape="sphere", n=1000, noise="gaussian",
                            solver_names=["none"], verbose=False)
    md = pfd.make_table(rows, fmt="markdown")
    csv = pfd.make_table(rows, fmt="csv")
    assert "| solver |" in md
    # CSV 用的是表头显示名（CD↓ 等），不是内部字段名
    assert "solver" in csv.splitlines()[0] and "CD" in csv.splitlines()[0]


def test_cli_subcommands():
    """验证命令行入口在当前平台上可跑，守住输出编码与子命令派发不退化。"""
    import subprocess
    import sys
    for sub in ["list", "noise"]:
        r = subprocess.run([sys.executable, "-m", "pfdenoise", sub],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        assert r.returncode == 0, f"pfdenoise {sub} 失败：{r.stderr}"
        assert r.stdout.strip(), f"pfdenoise {sub} 没有任何输出"
