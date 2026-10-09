r"""test_advanced_solvers.py — 针对体积守恒与各向异性相场求解器的单元测试

验证重点：
1. 注册不变量：solvers.py 字典 SOLVERS、DESCRIPTIONS、__all__ 完整性
2. 离散质量守恒：beta(t) 补偿确保 \sum phi 零漂移（数值精度级守恒）
3. 毛细收缩抵抗：零保真/低保真下相场不塌缩
4. 各向异性张量稳定性：切向平滑与法向抑制、CFL 约束与输出有限性
5. 端到端基准流程与 N2N 协议兼容性
"""

import numpy as np

import pfdenoise as pfd
from pfdenoise.solvers import (
    DESCRIPTIONS,
    SOLVERS,
    _prep,
    _run_ac_anisotropic,
    _run_ac_conserved,
    get_solver,
    phase_field_ac_anisotropic,
    phase_field_ac_conserved,
    phase_field_ac_convex,
    phase_field_ac_convex_adaptive,
)


def _make_cloud(n=1200, shape="sphere", noise="gaussian", seed=42):
    P, N, _, _ = pfd.make_shape(shape, n, seed)
    nm = pfd.make_noise(noise)
    Pn = nm.apply(P, N, np.random.default_rng(seed + 1), pfd.noise._NN(P))
    return P, N, Pn


# ---------------------------------------------------------------- 注册与接口
def test_solvers_registration():
    """验证两个新相场求解器正确注册到字典、描述与 __all__。"""
    assert "pf_ac_conserved" in SOLVERS
    assert "pf_ac_aniso" in SOLVERS
    assert "pf_ac_convex" in SOLVERS
    assert "pf_ac_convex_adaptive" in SOLVERS
    assert SOLVERS["pf_ac_conserved"] is phase_field_ac_conserved
    assert SOLVERS["pf_ac_aniso"] is phase_field_ac_anisotropic
    assert SOLVERS["pf_ac_convex"] is phase_field_ac_convex
    assert SOLVERS["pf_ac_convex_adaptive"] is phase_field_ac_convex_adaptive

    assert "pf_ac_conserved" in DESCRIPTIONS
    assert "pf_ac_aniso" in DESCRIPTIONS
    assert "pf_ac_convex" in DESCRIPTIONS
    assert "pf_ac_convex_adaptive" in DESCRIPTIONS
    assert len(DESCRIPTIONS["pf_ac_conserved"]) > 10
    assert len(DESCRIPTIONS["pf_ac_aniso"]) > 10
    assert len(DESCRIPTIONS["pf_ac_convex"]) > 10
    assert len(DESCRIPTIONS["pf_ac_convex_adaptive"]) > 10

    assert get_solver("pf_ac_conserved") is phase_field_ac_conserved
    assert get_solver("pf_ac_aniso") is phase_field_ac_anisotropic
    assert get_solver("pf_ac_convex") is phase_field_ac_convex
    assert get_solver("pf_ac_convex_adaptive") is phase_field_ac_convex_adaptive


# ---------------------------------------------------------------- 体积守恒 Allen-Cahn
def test_conserved_ac_discrete_mass_conservation():
    """验证 beta(t) 拉格朗日乘子在离散时间步推进中严格保证质量守恒（机器精度级）。"""
    h = 0.05
    eps = 2.0 * h
    n_vox = 20
    # 构造一个含正负相的立方测试场
    phi0 = np.full((n_vox, n_vox, n_vox), -1.0)
    phi0[6:14, 6:14, 6:14] = 1.0
    mass_init = float(phi0.sum())

    phi_end, dt, cfl = _run_ac_conserved(None, h, iters=50, eps=eps, lam=0.5, T=phi0)
    mass_end = float(phi_end.sum())

    # 机器精度级浮点守恒
    assert abs(mass_end - mass_init) < 1e-10, (
        f"离散质量守恒偏差过大：|Δm| = {abs(mass_end - mass_init)}"
    )
    assert cfl <= 1.0


def test_conserved_ac_end_to_end_sphere():
    """端到端验证 phase_field_ac_conserved 在含噪球面上稳定输出并保体积。"""
    P, N, Pn = _make_cloud(1200, shape="sphere", noise="gaussian")
    out, info = phase_field_ac_conserved(Pn, N, iters=80, lam=0.8, n_vox_max=32)

    assert len(out) == len(Pn), "重采样点数必须与输入点数一致"
    assert np.isfinite(out).all(), "输出点集坐标必须全为有限实数"
    assert info["cfl"] <= 1.0

    # 评估指标
    m = pfd.metrics.evaluate(out, P, N)
    assert m["chamfer"] < 0.25, f"CD 过大: {m['chamfer']}"
    assert abs(m["radius_drift"]) < 5.0, f"半径漂移过大: {m['radius_drift']}%"


def test_conserved_ac_resists_zero_fidelity_capillary_collapse():
    """对比测试：在 λ=0（纯曲率流无保真项）下，体积守恒 AC 不会塌缩。"""
    P, N, _, _ = pfd.make_shape("sphere", 1000, seed=1)
    occ, h, T = _prep(P, n_vox_max=24)

    # 运行 60 步 λ=0 的体积守恒 AC
    phi, dt, cfl = _run_ac_conserved(None, h, iters=60, eps=2.0 * h, lam=0.0, T=T)
    # 正相依然存活且质量严格守恒
    assert (phi > 0).sum() > 0, "体积守恒 AC 正相不应在曲率流下完全泯灭"
    assert abs(phi.sum() - T.sum()) < 1e-10, "λ=0 时全局积分质量必须严格守恒"


# ---------------------------------------------------------------- 各向异性扩散 Allen-Cahn
def test_anisotropic_ac_numerical_stability():
    """验证各向异性扩散相场求解器在多尺度网格上的数值稳定性与 CFL 条件。"""
    h = 0.05
    eps = 2.0 * h
    n_vox = 20
    phi0 = np.full((n_vox, n_vox, n_vox), -1.0)
    phi0[5:15, 5:15, 5:15] = 1.0

    phi_end, dt, cfl = _run_ac_anisotropic(
        None, h, iters=40, eps=eps, lam=0.8, c_tangent=1.0, c_normal=0.1, T=phi0
    )
    assert np.isfinite(phi_end).all(), "各向异性相场数值发散"
    assert cfl <= 1.0


def test_anisotropic_ac_end_to_end_bumpy():
    """端到端验证 phase_field_ac_anisotropic 在含噪凹凸曲面上的平滑去噪能力。"""
    P, N, Pn = _make_cloud(1200, shape="bumpy", noise="gaussian")
    out, info = phase_field_ac_anisotropic(
        Pn, N, iters=60, lam=0.8, c_tangent=1.0, c_normal=0.1, n_vox_max=32
    )

    assert len(out) == len(Pn)
    assert np.isfinite(out).all()
    assert info["cfl"] <= 1.0

    m = pfd.metrics.evaluate(out, P, N)
    assert m["chamfer"] < 0.35, f"CD 超标: {m['chamfer']}"
    assert m["normal_err"] < 50.0, f"法向估计误差过大: {m['normal_err']} deg"


# ---------------------------------------------------------------- 半隐式 Eyre 凸分裂 AC
def test_convex_ac_large_dt_stability():
    """验证半隐式 Eyre 凸分裂在大时间步长（超出显式 CFL 限制）下的无条件数值稳定性。"""
    P, N, Pn = _make_cloud(1000, shape="sphere", noise="gaussian")
    # 显式 CFL 要求 dt ~ 0.015，这里强制放大时间步乘子 dt_mult=2.5
    out, info = pfd.solvers.phase_field_ac_convex(
        Pn, N, iters=40, eps_h=2.0, lam=0.8, dt_mult=2.5, n_vox_max=24
    )
    assert len(out) == len(Pn)
    assert np.isfinite(out).all(), "大时间步下凸分裂求解器不应发散"
    m = pfd.metrics.evaluate(out, P, N)
    assert m["chamfer"] < 0.25, f"CD 过大: {m['chamfer']}"


def test_convex_ac_energy_dissipation():
    """验证 Eyre 凸分裂的离散自由能单调非递增性质 E^{n+1} <= E^n。"""
    n_vox = 16
    h = 0.1
    eps = 2.0 * h
    lam = 0.8
    T = np.full((n_vox, n_vox, n_vox), -1.0)
    T[4:12, 4:12, 4:12] = 1.0

    def compute_energy(phi):
        p = pfd.solvers._neumann(phi)
        gx = (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2.0 * h)
        gy = (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2.0 * h)
        gz = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2.0 * h)
        grad2 = gx ** 2 + gy ** 2 + gz ** 2
        e_grad = 0.5 * (eps ** 2) * np.sum(grad2)
        e_pot = 0.25 * np.sum((phi ** 2 - 1.0) ** 2)
        e_fid = 0.5 * lam * np.sum((phi - T) ** 2)
        return float(e_grad + e_pot + e_fid)

    e_init = compute_energy(T)
    phi_end, _, _ = pfd.solvers._run_ac_convex(
        None, h, iters=30, eps=eps, lam=lam, dt=0.05, T=T
    )
    e_end = compute_energy(phi_end)

    assert e_end <= e_init + 1e-4, f"能量非递增违反: E_init={e_init:.4f}, E_end={e_end:.4f}"


def test_convex_ac_adaptive_end_to_end():
    """验证自适应时间步长 Eyre 求解器在含噪点云上的端到端去噪与自适应步长记录。"""
    P, N, Pn = _make_cloud(1000, shape="sphere", noise="gaussian")
    out, info = phase_field_ac_convex_adaptive(
        Pn, N, iters=30, tol=1e-3, eps_h=2.0, lam=0.8, n_vox_max=24
    )
    assert len(out) == len(Pn)
    assert np.isfinite(out).all()
    assert info["accepted_steps"] > 0
    assert info["total_trials"] >= info["accepted_steps"]
    m = pfd.metrics.evaluate(out, P, N)
    assert m["chamfer"] < 0.25, f"自适应 Eyre 去噪 CD 过大: {m['chamfer']}"


def test_convex_ac_adaptive_lte_order():
    """验证局部截断误差 (LTE) 满足 O(dt^2) 二阶局部收敛率。"""
    h = 0.05
    nx = 16
    eps = 2.0 * h
    lam = 0.8
    eps2 = eps * eps
    phi0 = np.full((nx, nx, nx), -1.0)
    phi0[4:12, 4:12, 4:12] = 1.0

    kx = 2.0 * np.pi * np.fft.fftfreq(nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(nx)
    kz = 2.0 * np.pi * np.fft.fftfreq(nx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    eigen_lap = 2.0 * (np.cos(KX) + np.cos(KY) + np.cos(KZ) - 3.0) / (h * h)

    def _step(u, dt_val):
        concave_exp = u - u ** 3 + lam * phi0
        beta = float(np.mean(u ** 3 - u + lam * (u - phi0)))
        R = u + dt_val * (concave_exp + beta)
        denom = (1.0 + dt_val * lam) - dt_val * eps2 * eigen_lap
        R_hat = np.fft.fftn(R)
        return np.real(np.fft.ifftn(R_hat / denom))

    dt1 = 0.01
    dt2 = 0.02

    # dt1
    f1 = _step(phi0, dt1)
    h1 = _step(_step(phi0, 0.5 * dt1), 0.5 * dt1)
    lte1 = float(np.sqrt(np.mean((h1 - f1) ** 2)))

    # dt2
    f2 = _step(phi0, dt2)
    h2 = _step(_step(phi0, 0.5 * dt2), 0.5 * dt2)
    lte2 = float(np.sqrt(np.mean((h2 - f2) ** 2)))

    # log2(lte2 / lte1) 应接近 2.0
    rate = np.log(lte2 / lte1) / np.log(dt2 / dt1)
    assert 1.4 <= rate <= 2.2, f"LTE 阶数不符合预期: rate={rate:.2f}"


# ---------------------------------------------------------------- 基准与 N2N 协同
def test_new_solvers_in_benchmark():
    """验证新求解器可通过 run_benchmark 正常运行并输出完整指标字段。"""
    rows = pfd.run_benchmark(
        shape="sphere",
        n=1000,
        noise="gaussian",
        solver_names=[
            "pf_ac_conserved", "pf_ac_aniso", "pf_ac_convex", "pf_ac_convex_adaptive"
        ],
        solver_kwargs={"pf_ac_conserved": {"iters": 40, "n_vox_max": 24},
                       "pf_ac_aniso": {"iters": 40, "n_vox_max": 24},
                       "pf_ac_convex": {"iters": 30, "n_vox_max": 24},
                       "pf_ac_convex_adaptive": {"iters": 25, "n_vox_max": 24}},
        verbose=False,
    )
    assert len(rows) == 4
    for r in rows:
        assert "error" not in r, f"求解器运行报错: {r.get('error')}"
        assert np.isfinite(r["chamfer"])
        assert np.isfinite(r["p2m"])
        assert np.isfinite(r["radius_drift"])
        assert "cfl" in r


def test_new_solvers_in_n2n_protocol():
    """验证新求解器支持无标签 N2N 一致性评测协议。"""
    rows = pfd.run_n2n(
        shape="sphere",
        n=1000,
        noise="gaussian",
        solver_names=[
            "pf_ac_conserved", "pf_ac_aniso", "pf_ac_convex", "pf_ac_convex_adaptive"
        ],
        solver_kwargs={"pf_ac_conserved": {"iters": 30, "n_vox_max": 24},
                       "pf_ac_aniso": {"iters": 30, "n_vox_max": 24},
                       "pf_ac_convex": {"iters": 25, "n_vox_max": 24},
                       "pf_ac_convex_adaptive": {"iters": 20, "n_vox_max": 24}},
        verbose=False,
    )
    assert len(rows) == 4
    for r in rows:
        assert "error" not in r
        assert np.isfinite(r["consistency"])


# ---------------------------------------------------------------- 零等值面投影
def test_project_to_zero_level():
    """验证高阶零等值面正交投影（Level-Set Newton 投影）的数学收敛性与鲁棒性。"""
    from pfdenoise.solvers import (
        _extract_surface,
        _trilinear_interp,
        project_to_zero_level,
    )

    # 1. 空点云边界条件
    empty = project_to_zero_level(np.empty((0, 3)), np.zeros((10, 10, 10)), 0.1, np.zeros(3))
    assert len(empty) == 0

    # 2. 构造解析球体相场 phi = tanh((R - r) / (sqrt(2)*eps))
    n_vox = 32
    h = 2.4 / n_vox
    lo = np.array([-1.2, -1.2, -1.2])
    coords_1d = lo[0] + (np.arange(n_vox) + 0.5) * h
    X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    R = 0.8
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    eps = 2.0 * h
    phi = np.tanh((R - r) / (np.sqrt(2.0) * eps))

    # 3. 构造偏离 R=0.8 的测试点集（偏离 +0.05）
    rng = np.random.default_rng(42)
    dirs = rng.normal(size=(50, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts_perturbed = dirs * (R + 0.05)

    phi_before = _trilinear_interp(phi, h, lo, pts_perturbed)
    # 投影前点不在零等值面上，phi 有显著非零值
    assert np.all(np.abs(phi_before) > 0.05)

    # 4. 执行投影
    pts_proj = project_to_zero_level(pts_perturbed, phi, h, lo, max_iter=2)
    phi_after = _trilinear_interp(phi, h, lo, pts_proj)

    # 投影后 phi 绝对值显著下降，向零等值面严格收敛
    assert np.mean(np.abs(phi_after)) < np.mean(np.abs(phi_before)) * 0.3
    # 半径误差显著减小
    r_err_before = np.mean(np.abs(np.linalg.norm(pts_perturbed, axis=1) - R))
    r_err_after = np.mean(np.abs(np.linalg.norm(pts_proj, axis=1) - R))
    assert r_err_after < r_err_before

    # 5. 测试 _extract_surface 在开启 project_grad=True 时的鲁棒性
    P, N, _ = _make_cloud(n=300, shape="sphere")
    extracted = _extract_surface(phi, h, lo, P, N, project_grad=True)
    assert len(extracted) == len(P)
    assert np.isfinite(extracted).all()


