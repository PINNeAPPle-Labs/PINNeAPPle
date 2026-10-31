"""Gated X-TFC (pinneapple_simulation.numerical_solvers.gated_xtfc)."""
import math

import numpy as np

from pinneapple_simulation.numerical_solvers.gated_xtfc import (
    _features,
    _operator_rows,
    convection_diffusion,
    fit_inverse,
    gated_layout,
    log_evidence,
    search_gates_nested,
    solve_forward,
    twin_boundary_layer,
)


def test_layout_matches_the_paper_formulas():
    lay = gated_layout([0.9], [50.0], 1e-3, 100, n_colloc=40)
    assert lay.centers.size == 200 and lay.collocation.size == 80
    assert np.isclose(lay.centers[0], 0.0) and np.isclose(lay.centers[100], 0.9)          # endpoint grids
    assert np.isclose(lay.centers[1] - lay.centers[0], 0.9 / 100)
    assert np.isclose(lay.widths[0], 1.5 * 0.9 / 100, rtol=1e-6)                           # sigma_L = k d_L far left
    eb = max(50.0 * 0.1 / 100, 5e-3)                                                      # eps_b = max(geo, 5 nu)
    assert np.isclose(lay.eps_b[0], eb)
    s = 1 / (1 + np.exp(-(lay.centers - 0.9) / eb))
    assert np.allclose(lay.widths, (1 - s) * 1.5 * 0.9 / 100 + s * 1.5 * 0.1 / 100)
    two = gated_layout([0.1, 0.9], [50.0, 50.0], 1e-3, 50)
    assert two.centers.size == 150 and len(two.eps_b) == 2


def test_feature_derivatives_and_exact_boundary_values():
    lay = gated_layout([0.7], [30.0], 1e-2, 20)
    x = np.array([0.13, 0.5, 0.71, 0.95])
    h = 1e-6
    psi, d1, d2 = _features(x, lay)
    p1, _, _ = _features(x + h, lay)
    m1, _, _ = _features(x - h, lay)
    assert np.allclose(d1, (p1 - m1) / (2 * h), atol=1e-5)
    _, a1, _ = _features(x + h, lay)
    _, b1, _ = _features(x - h, lay)
    assert np.allclose(d2, (a1 - b1) / (2 * h), rtol=1e-4, atol=1e-3)
    ends, _, _ = _features(np.array([0.0, 1.0]), lay)
    assert np.allclose(ends, 0.0, atol=1e-15)


def test_operator_rows_vanish_on_the_exact_solution_space():
    """R c + r0 is the strong residual: check it against a direct evaluation for random c."""
    p = twin_boundary_layer(0.05)
    lay = gated_layout([0.2, 0.8], [40.0, 40.0], 0.05, 15)
    x = np.linspace(0.01, 0.99, 9)
    c = np.random.default_rng(0).normal(size=lay.centers.size)
    R, r0 = _operator_rows(p, x, lay)
    psi, d1, d2 = _features(x, lay)
    u, du, d2u = 1 + psi @ c, d1 @ c, d2 @ c
    assert np.allclose(R @ c + r0, -0.05 * d2u + 2 * (2 * x - 1) * du + 4 * u)


def test_forward_solve_is_accurate_on_a_mild_layer_and_exact_at_the_boundary():
    p = convection_diffusion(0.05)
    sol = solve_forward(p, gated_layout([0.8], [50.0], 0.05, 150))
    x = np.linspace(0, 1, 1001)
    assert np.abs(sol.predict(x) - p.exact(x)).max() < 1e-3
    assert sol.predict(np.array([0.0]))[0] == 0.0 and abs(sol.predict(np.array([1.0]))[0] - 1.0) < 1e-14


def test_nested_search_returns_a_point_inside_the_box():
    p = convection_diffusion(0.05)
    out = search_gates_nested(p, (np.arange(200) + 0.5) / 200, n_centers=60, xs_tol=1e-2, eps_tol=5.0)
    assert 0.80 <= out["xs"] <= 0.999 and 10 <= out["eps_scale"] <= 100 and out["n_evaluations"] > 4


def test_log_evidence_matches_the_closed_form():
    rng = np.random.default_rng(1)
    Phi = rng.normal(size=(30, 6))
    y = Phi @ rng.normal(size=6) + rng.normal(size=30)
    out = log_evidence(Phi, y, max_iter=0)          # eta stays at eta0
    eta = out["eta"]
    A = eta * np.eye(6) + Phi.T @ Phi
    m = np.linalg.solve(A, Phi.T @ y)
    ref = 3 * math.log(eta) - 0.5 * (np.sum((y - Phi @ m) ** 2) + eta * m @ m) - 0.5 * np.linalg.slogdet(A)[1] \
        - 15 * math.log(2 * math.pi)
    assert np.isclose(out["log_evidence"], ref) and np.allclose(out["mean"], m)


def test_inverse_recovers_a_mild_viscosity():
    nu = 0.1
    p = convection_diffusion(nu)
    x = 1 - (1 - np.linspace(0, 1, 40)) ** 3
    y = p.exact(x) + np.random.default_rng(0).normal(0, 1e-2, x.size)
    fit = fit_inverse(convection_diffusion, x, y, sigma=1e-2, beta_pde=1e2, n_centers=60, n_colloc=80,
                      n_eval=20, seed=0)
    assert abs(fit.nu - nu) / nu < 0.5
    assert np.all(fit.predictive_std(np.array([0.3, 0.6])) >= 0)


def test_bo_search_with_two_splits_runs_and_keeps_order():
    from pinneapple_simulation.numerical_solvers.gated_xtfc import search_gates_bo
    p = twin_boundary_layer(0.05)
    out = search_gates_bo(p, (np.arange(100) + 0.5) / 100, split_bounds=[(0.01, 0.3), (0.7, 0.99)],
                          n_centers=30, n_eval=8, seed=0)
    assert out["splits"][0] < out["splits"][1] and len(out["history"]) == 8
    x = np.linspace(0, 1, 201)
    assert np.abs(out["sol"].predict(x) - p.exact(x)).max() < 0.1
