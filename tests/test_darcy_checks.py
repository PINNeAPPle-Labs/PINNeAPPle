"""Physics checks of 2-D Darcy fields (pinneapple_physics.darcy_checks), against the reference finite-volume solver."""
import numpy as np
import pytest

from pinneapple_physics import darcy_checks as dc
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob


def _energy(e, a):
    return np.sqrt(np.sum(e * dc.darcy_operator_apply(e, a), axis=(-2, -1)) / e.shape[-1] ** 2)


@pytest.fixture(scope="module")
def solved():
    n = 64
    a = ob.threshold_coefficient(ob.periodic_grf(4, (n, n), alpha=2.0, tau=3.0, seed=7))
    u = np.stack([ob.darcy_2d(ai) for ai in a])
    return a, u


def test_operator_is_the_solver_matrix(solved):
    a, u = solved
    r = dc.darcy_residual(u, a, 1.0)
    assert np.abs(r).max() < 1e-9                       # the solver's own solution has zero residual
    v = np.random.default_rng(0).standard_normal(a.shape)
    w = dc.darcy_operator_apply(v, a)
    assert np.allclose(np.sum(v * w), np.sum(w * v))
    assert (np.sum(v * w, axis=(1, 2)) > 0).all()       # symmetric positive definite
    assert dc.darcy_residual(u[0], a[0]).shape == (64, 64)


def test_solver_solution_passes_every_physics_check(solved):
    a, u = solved
    checks = dc.darcy_checks(a, u, 1.0)
    assert {c.name for c in checks if c.passed} == {c.name for c in checks}
    assert dc.residual_error_bound(u, a).max() < 1e-10
    assert dc.boundary_flux_balance(u, a)["imbalance"].max() < 1e-10
    rep, _ = dc.darcy_trust_report(a, u, 1.0, kinds=("physics",))
    assert rep.decisao == "APPROVED"


def test_error_bound_is_a_guaranteed_upper_bound(solved):
    a, u = solved
    rng = np.random.default_rng(1)
    x = (np.arange(64) + 0.5) / 64
    for v in (u * (1 + 0.01 * rng.standard_normal(u.shape)),
              u + 0.05 * u.max() * np.sin(np.pi * x)[None, :, None] * np.sin(2 * np.pi * x)[None, None, :],
              0.9 * u):
        true = _energy(u - v, a) / _energy(v, a)
        bound = dc.residual_error_bound(v, a)
        assert (bound >= true * (1 - 1e-9)).all()
        assert (bound <= 10 * true).all()                # and not vacuous


def test_each_check_catches_its_own_violation(solved):
    a, u = solved
    by = lambda checks: {c.name.split(" ")[0]: c for c in checks}   # noqa: E731
    c = by(dc.darcy_checks(a, 1.1 * u))
    assert not c["flux"].passed and c["flux"].value == pytest.approx(0.1, rel=1e-6)
    c = by(dc.darcy_checks(a, u + 0.1 * u.max()))
    assert not c["boundary"].passed and not c["PDE"].passed
    c = by(dc.darcy_checks(a, u - 0.05 * u.max()))
    assert not c["positivity"].passed
    nan = u.copy()
    nan[0, 3, 3] = np.nan
    c = dc.darcy_checks(a, nan)
    assert c[0].name == "finite values" and c[0].passed is False


def test_reference_and_envelope(solved):
    a, u = solved
    env = dc.CoefficientEnvelope.fit(ob.threshold_coefficient(ob.periodic_grf(200, (64, 64), alpha=2.0, tau=3.0,
                                                                               seed=3)))
    assert env.a_min == 3.0 and env.a_max == 12.0
    assert env.contains(a).mean() >= 0.75
    wide = ob.threshold_coefficient(ob.periodic_grf(4, (64, 64), alpha=2.0, tau=3.0, seed=8), high=20.0, low=1.0)
    assert not env.contains(wide).any()
    assert any("max a" in s for s in env.violations(wide)[0])
    assert dc.CoefficientEnvelope.from_dict(env.to_dict()) == env
    checks = dc.darcy_checks(wide, u, reference=u, envelope=env)
    kinds = {c.kind: c for c in checks}
    assert kinds["reference"].passed and kinds["applicability"].passed is False
    rep, _ = dc.darcy_trust_report(wide, u, envelope=env)
    assert rep.decisao == "REJECT"
    phys, _ = dc.darcy_trust_report(wide, u, envelope=env, kinds=("physics",))
    assert phys.decisao == "REJECT"                  # u is not the solution for these coefficients


def test_coefficient_stats_length_scale():
    short = ob.threshold_coefficient(ob.periodic_grf(8, (32, 32), alpha=2.0, tau=9.0, seed=1))
    long = ob.threshold_coefficient(ob.periodic_grf(8, (32, 32), alpha=2.0, tau=3.0, seed=1))
    assert dc.coefficient_stats(short)["interface_density"].mean() > dc.coefficient_stats(long)[
        "interface_density"].mean()


# ------------------------------------------------------------------------------- node-based training data
def _nodal_case(n=24, cross_denominator=None):
    dx = 1.0 / (n + 1)
    x = np.arange(n + 1) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = np.where(np.sin(5 * X) * np.cos(3 * Y) + 0.3 * np.sin(7 * Y) > 0, 2.0, 0.5)
    u = dc.solve_nodal_darcy(k, 1.0, dx, cross_denominator=cross_denominator)
    return k, u, dx


def test_nodal_residual_is_the_solved_stencil():
    k, u, dx = _nodal_case()
    assert np.abs(dc.nodal_darcy_residual(u, k, 1.0, dx, cropped=0)).max() < 1e-9
    assert dc.nodal_darcy_residual(u[:-1, :-1], k[:-1, :-1], 1.0, dx, cropped=1).shape == (23, 23)
    assert np.abs(dc.nodal_darcy_residual(u[:-1, :-1], k[:-1, :-1], 1.0, dx, cropped=1)).max() < 1e-9
    assert dc.nodal_boundary_values(u, cropped=0).shape == (4, 25)


def test_nodal_batch_check_flags_a_mis_scaled_stencil_and_duplicate_samples():
    k, u, dx = _nodal_case()
    kb, ub, _ = _nodal_case(cross_denominator=2 * dx)            # grad k . grad u divided by 2 dx, not 4 dx^2
    good = dc.check_nodal_darcy_batch(np.stack([k[:-1, :-1], k[:-1, :-1].T]),
                                      np.stack([u[:-1, :-1], u[:-1, :-1].T]), 1.0, dx)
    assert good["passed"], good
    bad = dc.check_nodal_darcy_batch(np.stack([kb[:-1, :-1]] * 3), np.stack([ub[:-1, :-1]] * 3), 1.0, dx,
                                     alt_cross_denominators={"2 dx": 2 * dx})
    c = bad["checks"]
    assert not c["residual"]["passed"] and c["residual"]["median"] > 0.1
    assert not c["flux"]["passed"]
    d = c["distinct_coefficients"]
    assert d["n_distinct"] == 1 and d["exact_copies"] == 3 and not d["passed"]
    assert bad["residual_of_other_stencils"]["2 dx"] < 1e-9           # the data solve that other stencil
    assert bad["direct_solve_of_the_documented_equation"]["checks"]["residual"]["passed"]


def test_save_npz(tmp_path):
    x = np.random.default_rng(0).standard_normal((5, 8, 8))
    p = dc.save_darcy_npz(tmp_path / "r" / "f.npz", np.full((8, 8), 3.0), x[0])
    with np.load(p) as z:
        assert np.allclose(z["x"], (np.arange(8) + 0.5) / 8) and z["u"].dtype == np.float32


def test_shifted_darcy_sets_and_block_average():
    from pinneapple_simulation.numerical_solvers.operator_datasets import block_average, darcy_set
    a, u = darcy_set(2, 5, n=16, high=20.0, low=1.0)
    assert set(np.unique(a)) == {1.0, 20.0} and u.shape == (2, 16, 16) and (u > 0).all()
    # cell averages of a 4x finer solve: close to the 16 x 16 finite-volume solution of the same coefficient
    assert dc.residual_error_bound(u, a).max() < 0.2
    a2, _ = darcy_set(2, 5, n=16, tau=9.0)
    assert set(np.unique(a2)) == {3.0, 12.0}
    assert dc.coefficient_stats(a2)["interface_density"].mean() > dc.coefficient_stats(
        darcy_set(2, 5, n=16)[0])["interface_density"].mean()
    x = np.arange(16.0).reshape(4, 4)
    assert np.array_equal(block_average(x, 2), [[2.5, 4.5], [10.5, 12.5]])
    with pytest.raises(ValueError):
        block_average(np.ones((5, 5)), 2)


def test_calibrated_tolerances_flag_drift_without_labels(solved):
    a, u = solved
    rng = np.random.default_rng(2)
    noisy = lambda x, s: x * (1 + s * rng.standard_normal(x.shape))    # noqa: E731  a "model" with 1 % noise
    cal = dc.calibrate_tolerances(a, noisy(u, 0.01))
    assert set(cal) == set(dc.physics_metrics(a, u)) and cal["energy_bound"] > 0
    same = dc.darcy_checks(a, noisy(u, 0.01), tolerances=cal, physics_kind="drift")
    worse = dc.darcy_checks(a, noisy(u, 0.05), tolerances=cal, physics_kind="drift")
    by = lambda cs: {c.name.split(" ")[0]: c for c in cs}             # noqa: E731
    assert {c.kind for c in same} == {"drift"}
    assert by(same)["PDE"].value < 1.5 * cal["energy_bound"] and not by(worse)["PDE"].passed
    rep = dc.trust_report(worse + dc.darcy_checks(a, u), model="m")
    assert rep.decisao == "REJECT" and any("[drift]" in c.name for c in rep.checks)
    assert dc.trust_report(dc.darcy_checks(a, u), kinds=("physics",)).decisao == "APPROVED"


def test_l2_bound_and_use_profiles(solved):
    a, u = solved
    rng = np.random.default_rng(4)
    for v in (u * (1 + 0.02 * rng.standard_normal(u.shape)), 0.95 * u):
        true = np.linalg.norm((v - u).reshape(len(u), -1), axis=1) / np.linalg.norm(v.reshape(len(u), -1), axis=1)
        assert (dc.residual_l2_bound(v, a) >= true * (1 - 1e-9)).all()      # guaranteed
    assert set(dc.PROFILES) == {"fields", "fluxes"}
    for p in dc.PROFILES.values():
        assert set(p.limits) == set(p.reasons) and all(p.reasons.values())
        assert all(p.limits[k] == dc.DEFAULT_TOLERANCES[k] for k in p.limits)   # the library's fixed tolerances
    ok = dc.profile_verdicts(a, u)
    assert all(v["decision"] == "APPROVED" and v["ratio"] < 1 for v in ok.values())
    bad = dc.profile_verdicts(a, 1.1 * u)
    assert bad["fluxes"]["decision"] == "REJECT" and bad["fluxes"]["worst"] == "flux"
    assert bad["fluxes"]["metrics"]["flux"]["ratio"] == pytest.approx(0.1 / 0.02, rel=1e-6)
    checks = dc.sample_checks(a[0], u[0])
    assert [c["key"] for c in checks] == ["energy_bound", "l2_bound", "flux", "bc", "positivity"]
    assert dc.sample_verdict(checks) == "PASS" and dc.sample_verdict(dc.sample_checks(a[0], 1.1 * u[0])) == "FAIL"
    env = dc.CoefficientEnvelope.fit(a)
    assert dc.CoefficientEnvelope.from_check_config(env.to_check_config()) == env
    assert "envelope = {" in dc.check_config_entry("x", "x.npz", envelope=env, tolerances={"flux": 0.1})


def test_where_it_errs_estimates_follow_the_true_error(solved):
    a, u = solved
    n = u.shape[-1]
    x = (np.arange(n) + 0.5) / n
    # an error confined to the walls, and one spread over the interior
    wall = np.zeros((n, n))
    wall[:2, :] = wall[-2:, :] = wall[:, :2] = wall[:, -2:] = 1.0
    bump = np.sin(np.pi * x)[:, None] * np.sin(np.pi * x)[None, :]
    for pert, worse in ((0.2 * wall * u, "wall"), (0.05 * bump * u.max(), "interior")):
        v = u + pert
        e = dc.estimated_error(v, a)
        true = u - v
        assert all(np.corrcoef(e[i].ravel(), true[i].ravel())[0, 1] > 0.6 for i in range(len(u)))
        reg = dc.regional_errors(v, a)
        assert reg[worse].mean() > reg[{"wall": "interior", "interior": "wall"}[worse]].mean()
    assert np.abs(dc.estimated_error(u, a)).max() < 1e-10 and dc.estimated_flux_error(u, a).max() < 1e-10
    # flux of a smooth field against the analytic a grad u
    X, Y = np.meshgrid(x, x, indexing="ij")
    q = dc.flux_magnitude(X * (1 - X) * Y * (1 - Y), np.full((n, n), 2.0))
    exact = 2 * np.hypot((1 - 2 * X) * Y * (1 - Y), X * (1 - X) * (1 - 2 * Y))
    assert np.abs(q - exact).max() < 0.02 * exact.max()
    m = dc.region_masks(n)
    assert m["wall"].sum() == n * n - (n - 4) ** 2 and not (m["wall"] & m["interior"]).any()


def test_use_verdicts(solved):
    a, u = solved
    env = dc.CoefficientEnvelope.fit(a, q=0.0)              # covers every one of the 4 samples
    ok = dc.use_verdicts(a[0], u[0], envelope=env)
    assert set(ok) == set(dc.USES) | {"inputs"} and all(v["ok"] for v in ok.values())
    v = u[0].copy()
    v[:2, :] *= 1.5                                         # wrong next to one wall only
    bad = dc.use_verdicts(a[0], v)
    assert bad["interior field"]["ok"] and not bad["walls and fluxes"]["ok"] and ">" in bad["walls and fluxes"]["reason"]
    wide = dc.use_verdicts(np.where(a[0] > 5, 20.0, 1.0), u[0], envelope=env)
    assert not wide["inputs"]["ok"]
