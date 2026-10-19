"""Verification of pinneapple_design.topology: adjoint gradients against finite differences, MMA against
SLSQP on an analytic problem, and convergence of the classic benchmarks (MBB beam, cantilever, heat
conduction, Stokes double pipe, reluctance rotor). Tolerances come from runs of this code (quoted)."""
import math

import numpy as np
import pytest

from pinneapple_design.topology import (
    MMA,
    Compliance2D,
    DensityFilter,
    DesignChain,
    StokesDissipation2D,
    SymmetryMap,
    ThermalCompliance2D,
    finite_difference_check,
    optimize,
)


def _chain_fun(problem, beta, r_min=1.5):
    chain = DesignChain(problem.n, DensityFilter.grid(problem.grid.nelx, problem.grid.nely, r_min), beta=beta)

    def fun(z):
        f, df = problem.evaluate(chain.forward(z))
        return f, chain.backward(df)

    return fun


@pytest.mark.parametrize("make", [lambda: Compliance2D.mbb(30, 10), lambda: ThermalCompliance2D(20, 20),
                                  lambda: StokesDissipation2D.double_pipe(nely=12)])
@pytest.mark.parametrize("beta", [0.0, 4.0])
def test_sensitivities_through_filter_and_projection_match_finite_differences(make, beta):
    problem = make()
    rng = np.random.default_rng(0)
    x = rng.uniform(0.2, 0.9, problem.n)
    fun = _chain_fun(problem, beta)
    _, g = fun(x)
    idx = np.concatenate([np.argsort(-np.abs(g))[:4], rng.choice(problem.n, 4, replace=False)])
    errs = finite_difference_check(fun, x, idx, h=1e-5)
    assert errs[:4].max() < 1e-5  # entries with the largest sensitivities
    assert errs.max() < 1e-4  # random entries (FD noise grows where the sensitivity is small)


def test_mma_matches_slsqp_on_svanberg_cantilever_problem():
    """min 0.0624 sum(x)  s.t.  61/x1^3 + 37/x2^3 + 19/x3^3 + 7/x4^3 + 1/x5^3 <= 1, 1 <= x <= 10
    (the five-variable test problem of Svanberg 1987); MMA( against the reference solution of SLSQP."""
    from scipy.optimize import minimize

    c1, a = 0.0624, np.array([61.0, 37.0, 19.0, 7.0, 1.0])

    def f(x):
        return c1 * x.sum()

    def g(x):
        return float((a / x ** 3).sum() - 1)

    ref = minimize(f, np.full(5, 5.0), jac=lambda x: np.full(5, c1), method="SLSQP", bounds=[(1, 10)] * 5,
                   constraints=[{"type": "ineq", "fun": lambda x: -g(x), "jac": lambda x: 3 * a / x ** 4}],
                   options={"ftol": 1e-14, "maxiter": 500})
    mma = MMA(5, 1, xmin=1.0, xmax=10.0)
    x = np.full(5, 5.0)
    for _ in range(60):
        x = mma.update(x, f(x), np.full(5, c1), np.array([g(x)]), np.array([-3 * a / x ** 4]))
    assert f(x) == pytest.approx(ref.fun, rel=1e-8)  # both 1.33995636
    assert np.allclose(x, ref.x, rtol=1e-6)
    assert g(x) < 1e-8


@pytest.fixture(scope="module")
def mbb_runs():
    p = Compliance2D.mbb(60, 20)
    c_uniform = p.evaluate(np.full(p.n, 0.5))[0]
    oc = optimize(p, 0.5, DensityFilter.grid(60, 20, 2.4), "oc", max_iter=300)
    mma = optimize(p, 0.5, DensityFilter.grid(60, 20, 2.4), "mma", max_iter=300)
    return c_uniform, oc, mma


def test_mbb_beam_oc_and_mma_reach_the_same_compliance(mbb_runs):
    c_uniform, oc, mma = mbb_runs
    for r in (oc, mma):
        assert r.history["volume"][-1] == pytest.approx(0.5, abs=1e-3)
        assert r.objective < 0.5 * c_uniform  # measured: 233.7 (OC), 233.5 (MMA) vs 498 uniform
    assert oc.objective == pytest.approx(mma.objective, rel=0.02)


def test_mbb_beam_is_a_truss_attached_to_load_and_supports(mbb_runs):
    _, oc, _ = mbb_runs
    x = oc.x_phys.reshape(60, 20)  # [i (x), j (y)]
    assert x[0, -1] > 0.9  # material under the load (top-left)
    assert x[-1, 0] > 0.9  # material on the roller (bottom-right)
    assert x[:, -1].mean() > 0.7 and x[:, 0].mean() > 0.5  # top and bottom chords


def test_projection_continuation_gives_a_crisp_cantilever():
    p = Compliance2D.cantilever(60, 30)
    r = optimize(p, 0.4, DensityFilter.grid(60, 30, 2.0), "mma", max_iter=200, beta0=1.0, beta_max=16.0,
                 beta_every=30)
    gray = ((r.x_phys > 0.05) & (r.x_phys < 0.95)).mean()
    assert gray < 0.08
    assert np.mean(r.x_phys) == pytest.approx(0.4, abs=5e-3)


def test_thermal_compliance_optimization_builds_conducting_paths():
    p = ThermalCompliance2D(40, 40)
    c0 = p.evaluate(np.full(p.n, 0.3))[0]
    r = optimize(p, 0.3, DensityFilter.grid(40, 40, 1.5), "mma", max_iter=80)
    assert r.objective < 0.25 * c0
    x = r.x_phys.reshape(40, 40)
    assert x[0, 18:22].mean() > 0.9  # material at the sink


def _double_pipe(length, nely=30):
    """Continuation on the Brinkman interpolation parameter q (0.01 then 0.1)."""
    x = None
    for q in (0.01, 0.1):
        p = StokesDissipation2D.double_pipe(nely=nely, length=length, q=q)
        r = optimize(p, 1.0 / 3.0, DensityFilter.grid(p.grid.nelx, p.grid.nely, 1.5), "mma", max_iter=300,
                     move=0.1, tol=1e-3, x0=x)
        x = r.x
    return p, r


def test_stokes_double_pipe_two_pipes_in_square_domain_one_channel_in_long_domain():
    """Qualitative result of Borrvall & Petersson (2003) for the double pipe: two separate pipes when
    the domain is 1 x 1, one merged channel when it is 1.5 x 1."""
    for length, merged in ((1.0, False), (1.5, True)):
        p, r = _double_pipe(length)
        uniform = p.evaluate(np.full(p.n, 1.0 / 3.0))[0]
        assert np.mean(r.x_phys) <= 1.0 / 3.0 + 2e-3
        assert r.objective < 0.25 * uniform
        x = r.x_phys.reshape(p.grid.nelx, p.grid.nely)
        mid = x[p.grid.nelx // 2]
        assert x[0, 6:9].mean() > 0.8 and x[0, 21:24].mean() > 0.8  # fluid at both inlets
        if merged:
            assert mid[13:17].mean() > 0.8 and mid[6:9].mean() < 0.2
        else:
            assert mid[13:17].mean() < 0.2 and mid[6:9].mean() > 0.8


def test_reluctance_rotor_optimization_increases_torque():
    from pinneapple_design.topology.magnetic import RotorTorqueProblem
    from pinneapple_simulation.electromagnetics import LinearMaterial
    from pinneapple_simulation.electromagnetics.machine import SPMMachine

    m = SPMMachine(n_theta=120, slots=12, magnet_thickness=0.0, rotor_core_inner=0.01, rotor_core_layers=5,
                   iron=LinearMaterial(1000.0))
    prob = RotorTorqueProblem(m, thetas=np.radians([0.0, 10.0]), I_peak=60.0, gamma=math.radians(60.0))
    sym = SymmetryMap.rotational(prob.centroids, 4, tol=1e-9)
    filt = DensityFilter(prob.centroids, 0.003, prob.volumes)
    chain = DesignChain(prob.n, filt, sym, beta=2.0)
    rng = np.random.default_rng(3)
    x = rng.uniform(0.3, 0.8, sym.n_var)

    def fun(z):
        f, df = prob.evaluate(chain.forward(z))
        return f, chain.backward(df)

    _, g = fun(x)
    idx = np.argsort(-np.abs(g))[:4]
    assert finite_difference_check(fun, x, idx, h=1e-5).max() < 1e-5
    f_solid = prob.evaluate(np.ones(prob.n))[0]
    r = optimize(prob, 0.5, filt, "mma", max_iter=25, symmetry=sym, objective_scale=1.0, move=0.2)
    assert abs(f_solid) < 0.05  # a round, solid rotor makes (almost) no reluctance torque
    assert r.objective < -5.0  # mean torque > 5 N m after 25 iterations


def test_ripple_objective_gradient():
    from pinneapple_design.topology.magnetic import RotorTorqueProblem
    from pinneapple_simulation.electromagnetics import LinearMaterial
    from pinneapple_simulation.electromagnetics.machine import SPMMachine

    m = SPMMachine(n_theta=120, slots=12, magnet_thickness=0.0, rotor_core_inner=0.01, rotor_core_layers=4,
                   iron=LinearMaterial(1000.0))
    prob = RotorTorqueProblem(m, thetas=np.radians([0.0, 5.0, 10.0]), I_peak=60.0, gamma=1.0,
                              objective="ripple")
    rng = np.random.default_rng(5)
    x = rng.uniform(0.3, 0.9, prob.n)
    _, g = prob.evaluate(x)
    idx = np.argsort(-np.abs(g))[:4]
    assert finite_difference_check(prob.evaluate, x, idx, h=1e-5).max() < 1e-4
    con = prob.mean_torque_constraint(1.0)
    _, gc = con(x)
    errs = finite_difference_check(con, x, np.argsort(-np.abs(gc))[:3], h=1e-5)
    assert errs.max() < 1e-5
