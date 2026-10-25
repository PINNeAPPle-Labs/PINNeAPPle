"""General X-TFC solver (pinneapple_simulation.numerical_solvers.xtfc_general) and the X-TFC paper problems."""
import numpy as np
import pytest
import torch
from numpy.polynomial import polynomial as P

from pinneapple_physics.paper_problems_xtfc import XTFC_CASES, p6_rhs
from pinneapple_simulation.numerical_solvers.xtfc_general import (
    Constraint,
    XTFCProblem,
    activation_derivative,
    grid,
    solve_xtfc,
    switching_functions,
)


@pytest.mark.parametrize("name", ["tanh", "logistic"])
def test_activation_derivatives_match_finite_differences(name):
    z = np.linspace(-3, 3, 13)
    h = 1e-4
    for n in range(1, 5):
        fd = (activation_derivative(name, z + h, n - 1) - activation_derivative(name, z - h, n - 1)) / (2 * h)
        assert np.allclose(activation_derivative(name, z, n), fd, atol=1e-6)


def test_switching_functions_are_the_paper_vectors():
    # Dirichlet at 0 and 1: v(x) = {1 - x, x}; Dirichlet at 0, Neumann at 1: v(y) = {1, y}
    S = switching_functions([Constraint(0, 0.0), Constraint(0, 1.0)])
    assert np.allclose(S, [[1, -1], [0, 1]])
    S = switching_functions([Constraint(0, 0.0), Constraint(0, 1.0, 1)])
    assert np.allclose(S, [[1, 0], [0, 1]])
    S = switching_functions([Constraint(0, 0.0), Constraint(0, 2.0)])     # on [0, 2]: (2 - x)/2, x/2
    assert np.allclose(P.polyval(1.5, S[0]), 0.25) and np.allclose(P.polyval(1.5, S[1]), 0.75)


def test_constraints_hold_for_any_beta():
    case = XTFC_CASES["pde2"]
    r = solve_xtfc(case.problem, grid(case.problem.bounds, 6), n_neurons=20, seed=3)
    r.beta["f"] = np.random.default_rng(0).normal(size=20) * 10        # not a solution
    s = np.linspace(0, 1, 9)
    zero = np.zeros_like(s)
    assert np.allclose(r.predict(np.c_[zero, s]), 0, atol=1e-12)               # f(0, y) = 0
    assert np.allclose(r.predict(np.c_[zero + 1, s]), 0, atol=1e-12)           # f(1, y) = 0
    assert np.allclose(r.predict(np.c_[s, zero]), 0, atol=1e-12)               # f(x, 0) = 0
    fy = r.predict(np.c_[s, zero + 1], alpha=(0, 1))
    assert np.allclose(fy, 2 * np.sin(np.pi * s), atol=1e-10)                  # f_y(x, 1) = 2 sin(pi x)


def test_derivatives_of_the_expression_match_finite_differences():
    case = XTFC_CASES["pde5"]
    r = solve_xtfc(case.problem, grid(case.problem.bounds, 4), n_neurons=30, seed=1)
    r.beta["f"] = np.random.default_rng(1).normal(size=30)
    X = np.array([[0.7, 0.3, 0.4], [1.2, 0.8, 0.9]])
    h = 1e-5
    for d, alpha in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        e = np.eye(3)[d] * h
        fd = (r.predict(X + e) - r.predict(X - e)) / (2 * h)
        assert np.allclose(r.predict(X, alpha=alpha), fd, rtol=1e-6, atol=1e-8)


def test_linear_problem_recovers_a_polynomial_exactly():
    # u'' = 2 on [0, 1], u(0) = 0, u(1) = 1: u = x^2; the constrained expression only has to fit a constant
    prob = XTFCProblem([(0.0, 1.0)], {"u": [Constraint(0, 0.0), Constraint(0, 1.0)]}, {"u": lambda X: X[:, 0]},
                       lambda X, D: D[("u", (2,))] - 2.0, [("u", (2,))], linear=True)
    r = solve_xtfc(prob, grid(prob.bounds, 20), n_neurons=15, seed=0)
    x = np.linspace(0, 1, 50)[:, None]
    assert np.abs(r.predict(x) - x[:, 0] ** 2).max() < 1e-8


def test_map_to_unit_changes_only_the_feature_inputs():
    case = XTFC_CASES["pde4"]
    a = solve_xtfc(case.problem, grid(case.problem.bounds, 10), n_neurons=40, seed=0)
    b = solve_xtfc(case.problem, grid(case.problem.bounds, 10), n_neurons=40, seed=0, map_to_unit=True)
    x = np.array([[0.0, 0.3], [1.0, 0.7], [0.5, 0.0]])
    assert np.allclose(a.predict(x), case.exact_np(x), atol=1e-12) and np.allclose(b.predict(x), case.exact_np(x),
                                                                                    atol=1e-12)


@pytest.mark.parametrize("name", sorted(XTFC_CASES))
def test_exact_solutions_satisfy_their_equations(name):
    """Every case's exact solution zeroes its residual (checks the transcription of each equation)."""
    case = XTFC_CASES[name]
    p = case.problem
    rng = np.random.default_rng(0)
    lo, hi = np.array(p.bounds).T
    X = torch.tensor(lo + (hi - lo) * rng.random((50, p.ndim)), dtype=torch.float64)
    D = {}
    for f, alpha in p.derivatives:
        Xr = X.clone().requires_grad_(True)
        out = case.exact[f](Xr)
        for d, n in enumerate(alpha):
            for _ in range(n):
                out = torch.autograd.grad(out.sum(), Xr, create_graph=True)[0][:, d]
        D[(f, alpha)] = out.detach()
    R = p.residual(X, D)
    for r in ([R] if isinstance(R, torch.Tensor) else R):
        assert torch.max(torch.abs(r)) < 1e-10


def test_problem6_printed_rhs_is_the_operator_on_the_exact_solution():
    X = torch.rand(40, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1)).requires_grad_(True)
    z = XTFC_CASES["pde6"].exact["z"](X)
    g = torch.autograd.grad(z.sum(), X)[0]
    assert torch.allclose(g[:, 2] * g[:, 0] + g[:, 1], p6_rhs(X.detach()), atol=1e-12)


def test_heat_problem_reaches_the_papers_order_at_its_settings():
    """Problem 4 at the paper's settings (196 tanh neurons, 30x30 grid, unif(-1, 1)); one seed, a smoke bound."""
    case = XTFC_CASES["pde4"]
    r = solve_xtfc(case.problem, grid(case.problem.bounds, case.n_train), n_neurons=case.n_neurons,
                   activation=case.activation, weight_range=case.weight_range, seed=0)
    X = np.random.default_rng(5).random((500, 2))
    assert np.abs(r.predict(X) - case.exact_np(X)).max() < 1e-5
