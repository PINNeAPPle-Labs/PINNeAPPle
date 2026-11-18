"""Integral equations of the second kind in the PDE compiler (``pinn_solver/compiler/integral_kinds.py``).

Every case has a closed-form solution, and each test pairs "the exact solution has the residual the discretisation
predicts" with "a wrong solution does not". The measured orders are part of the contract: the composite trapezoid
with an analytic self cell is second order for a bounded kernel, and first order for a logarithmic one, because the
two cells adjacent to the singularity carry an ``O(h)`` truncation error. That is a property of the rule, not of the
implementation, and it is asserted so that a change of scheme has to update the numbers deliberately.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pinneapple_physics.pinn_solver.compiler.integral_kinds import (
    INTEGRAL_KINDS,
    constant_kernel,
    helmholtz2_diag,
    helmholtz2_kernel,
    integral_residuals,
    log_diag,
    log_kernel,
    nystrom_matrix,
    trapezoid_weights,
)

E = math.e


# -------------------------------------------------------------------------------------------------- helpers
def spec_of(kind, fields, params, bounds=(-0.0, 1.0)):
    from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec

    return ProblemSpec(name=kind, dim=1, coords=("x",), fields=fields,
                       pde=PDETermSpec(kind=kind, fields=fields, coords=("x",), params=params),
                       conditions=(), domain_bounds={"x": bounds})


class Exact(torch.nn.Module):
    """A model that ignores its weights and returns the closed-form field (the compiler only calls it)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X):
        return self.fn(X)


def residual(spec, fn, x, ctx=None):
    """Root-mean-square PDE residual at the nodes ``x`` (the compiler aggregates the mean of the squares)."""
    from pinneapple_physics.pinn_solver.compiler.compile import compile_problem

    loss = compile_problem(spec)(Exact(fn), None, {
        "x_col": torch.as_tensor(np.asarray(x, float), dtype=torch.float64).reshape(-1, 1),
        "ctx": ctx or {},
    })
    return float(loss["pde"].detach()) ** 0.5


def free_term_log(X, ctx):
    """``f = 1 - (x ln x + (1 - x) ln(1 - x) - 1)``, i.e. ``u = 1`` solves ``u = f + int_0^1 ln|x - s| u ds``."""
    x = X[:, 0]
    left = np.where(x > 0.0, x * np.log(np.clip(x, 1e-300, None)), 0.0)
    right = np.where(x < 1.0, (1.0 - x) * np.log(np.clip(1.0 - x, 1e-300, None)), 0.0)
    return 1.0 - (left + right - 1.0)


def order(errors, ratio=2.0):
    """Observed order from a sequence of errors on grids refined by ``ratio`` at each step."""
    return float(np.mean(np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(ratio)))


# -------------------------------------------------------------------------------------------------- quadrature
def test_trapezoid_weights_tile_the_domain():
    x = np.linspace(0.0, 1.0, 33)
    w = trapezoid_weights(x)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert w[0] == pytest.approx(0.5 * (x[1] - x[0]))
    assert w[-1] == pytest.approx(0.5 * (x[-1] - x[-2]))
    assert np.allclose(w[1:-1], 0.5 * (x[2:] - x[:-2]))    # interior: half of the two neighbouring spacings


def test_trapezoid_weights_refuse_degenerate_node_sets():
    with pytest.raises(ValueError, match="at least 3"):
        trapezoid_weights(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="distinct"):
        trapezoid_weights(np.array([0.0, 0.5, 0.5, 1.0]))


def test_bounded_kernel_rule_integrates_a_constant_exactly():
    x = np.linspace(0.0, 1.0, 41)
    a, d = nystrom_matrix(x, constant_kernel)
    assert float((a @ np.ones(x.size))[20] + d[20]) == pytest.approx(1.0, abs=1e-14)


def test_log_diagonal_matches_a_numerical_cell_integral():
    """``log_diag(w)`` is the exact ``int_{-w/2}^{w/2} ln|s| ds``; check it against a quadrature that is exact to
    machine precision away from the origin and analytic in the two half panels."""
    from scipy.integrate import quad

    for w in (0.1, 0.02, 0.005):
        numeric = quad(lambda s: math.log(abs(s)), 0.0, 0.5 * w)[0] * 2.0
        assert float(log_diag(np.array([w]), {})[0]) == pytest.approx(numeric, rel=1e-12)


def test_helmholtz_diagonal_matches_the_small_argument_expansion_of_the_kernel():
    """The expansion-based self cell is the ``k w -> 0`` limit of the true cell integral of ``(j/4) H_0^(2)``."""
    from scipy.integrate import quad
    from scipy.special import hankel2

    k, w = 1.0, 0.004
    numeric = 2.0 * quad(lambda s: (0.25j * hankel2(0, k * s)).real, 0.0, 0.5 * w, limit=400)[0] \
        + 2.0j * quad(lambda s: (0.25j * hankel2(0, k * s)).imag, 0.0, 0.5 * w, limit=400)[0]
    assert complex(helmholtz2_diag(np.array([w]), {"k": k})[0]) == pytest.approx(numeric, rel=2e-3)
    with pytest.raises(ValueError, match="small-argument"):
        helmholtz2_diag(np.array([2.0]), {"k": k})


def test_the_kernel_must_be_finite_on_the_diagonal():
    def divergent(X, S, ctx):
        with np.errstate(divide="ignore"):
            return 1.0 / np.linalg.norm(X[:, None, :] - S[None, :, :], axis=-1)

    with pytest.raises(ValueError, match="non-finite"):
        nystrom_matrix(np.linspace(0, 1, 21), divergent)


# -------------------------------------------------------------------------------------------------- Fredholm
def test_fredholm_bounded_kernel_residual_is_second_order_on_the_closed_form_solution():
    """``u = exp(x)`` solves ``u = f + int_0^1 u ds`` with ``f = exp(x) - (e - 1)``; the residual is the trapezoid's
    own error, so it must halve by four on every refinement."""
    spec = spec_of("fredholm2", ("u",), {"lambda": 1.0})
    ctx = {"kernel_fn": constant_kernel, "f_fn": lambda X, c: np.exp(X[:, 0]) - (E - 1.0)}
    errs = [residual(spec, lambda X: torch.exp(X[:, 0:1]), np.linspace(0, 1, n), ctx)
            for n in (21, 41, 81, 161)]
    assert errs[0] < 1e-3 and order(errs) == pytest.approx(2.0, abs=0.2)


def test_fredholm_bounded_kernel_residual_does_not_vanish_for_a_wrong_solution():
    spec = spec_of("fredholm2", ("u",), {"lambda": 1.0})
    ctx = {"kernel_fn": constant_kernel, "f_fn": lambda X, c: np.exp(X[:, 0]) - (E - 1.0)}
    errs = [residual(spec, lambda X: torch.exp(0.9 * X[:, 0:1]), np.linspace(0, 1, n), ctx)
            for n in (41, 81, 161)]
    assert errs[-1] > 1e-3 and order(errs) == pytest.approx(0.0, abs=0.2)


def test_direct_nystrom_solve_reaches_the_closed_form_away_from_the_eigenvalue():
    """``lambda = 0.5`` keeps ``I - lambda K`` well conditioned and the direct solve must land on ``exp(x)``."""
    x = np.linspace(0.0, 1.0, 81)
    a, d = nystrom_matrix(x, constant_kernel)
    f = np.exp(x) - 0.5 * (E - 1.0)
    u = np.linalg.solve(np.eye(x.size) - 0.5 * (a + np.diag(d)), f)
    assert float(np.max(np.abs(u - np.exp(x)))) < 1e-4


def test_lambda_one_is_the_eigenvalue_of_the_unit_kernel():
    """The Fredholm alternative, asserted as a number: at ``lambda = 1`` the unit kernel makes the system singular,
    yet the exact solution still has a vanishing residual -- the two facts are different and both matter."""
    x = np.linspace(0.0, 1.0, 81)
    a, d = nystrom_matrix(x, constant_kernel)
    assert np.linalg.cond(np.eye(x.size) - (a + np.diag(d))) > 1e10
    spec = spec_of("fredholm2", ("u",), {"lambda": 1.0})
    ctx = {"kernel_fn": constant_kernel, "f_fn": lambda X, c: np.exp(X[:, 0]) - (E - 1.0)}
    assert residual(spec, lambda X: torch.exp(X[:, 0:1]), x, ctx) < 1e-3   # the quadrature error, not zero


# -------------------------------------------------------------------------------------------------- singular
def test_log_kernel_residual_converges_at_first_order_with_the_analytic_diagonal():
    """``u = 1`` solves ``u = f + int_0^1 ln|x-s| u ds``. The rule is first order: the two cells next to the
    singularity are integrated by the trapezoid and carry an ``O(h)`` error."""
    spec = spec_of("fredholm2", ("u",), {"lambda": 1.0, "singular": True})
    ctx = {"kernel_fn": log_kernel, "diag_fn": log_diag, "f_fn": free_term_log}
    errs = [residual(spec, lambda X: torch.ones_like(X[:, 0:1]), np.linspace(0, 1, n), ctx)
            for n in (41, 81, 161, 321)]
    assert order(errs) == pytest.approx(1.0, abs=0.15)


def test_removing_the_analytic_diagonal_costs_more_than_an_order_of_magnitude():
    """Without the analytic self cell the diagonal contribution is simply dropped, so the rule degrades to
    ``O(h |ln h|)``: it still converges, but it is roughly thirty times worse at the same resolution. Measured, not
    assumed -- the naive guess that it would stop converging is wrong."""
    with_diag = spec_of("fredholm2", ("u",), {"lambda": 1.0, "singular": True})
    without = spec_of("fredholm2", ("u",), {"lambda": 1.0})
    good = {"kernel_fn": log_kernel, "diag_fn": log_diag, "f_fn": free_term_log}
    bad = {"kernel_fn": log_kernel, "f_fn": free_term_log}
    ones = lambda X: torch.ones_like(X[:, 0:1])                                # noqa: E731
    at_41 = [residual(without, ones, np.linspace(0, 1, 41), bad),
             residual(without, ones, np.linspace(0, 1, 321), bad)]
    assert residual(with_diag, ones, np.linspace(0, 1, 41), good) < at_41[0] / 10.0
    assert order(at_41, ratio=8.0) < 1.1                     # h |ln h|: no better than first order, and no worse


def test_a_singular_kernel_without_a_diagonal_is_refused_rather_than_approximated():
    spec = spec_of("fredholm2", ("u",), {"lambda": 1.0, "singular": True})
    with pytest.raises(ValueError, match="singular"):
        residual(spec, lambda X: torch.ones_like(X[:, 0:1]), np.linspace(0, 1, 41),
                 {"kernel_fn": log_kernel, "f_fn": free_term_log})


# -------------------------------------------------------------------------------------------------- Volterra
def test_volterra_matrix_is_causal():
    x = np.linspace(0.0, 1.0, 21)
    a, d = nystrom_matrix(x, constant_kernel, volterra=True)
    assert np.allclose(np.triu(a, 0), 0.0)                                     # nothing from the future
    assert d[0] == 0.0                                                         # node 0 integrates over [x0, x0]
    assert np.allclose(d[1:], 0.5 * (x[1:] - x[:-1]))                           # half a cell everywhere else


def test_volterra_residual_and_solve_are_second_order():
    """``u = exp(x)`` is the closed form of ``u(x) = 1 + int_0^x u ds``: residual and solve are second order."""
    spec = spec_of("volterra2", ("u",), {"lambda": 1.0})
    ctx = {"kernel_fn": constant_kernel, "f_fn": lambda X, c: np.ones(len(X))}
    errs = [residual(spec, lambda X: torch.exp(X[:, 0:1]), np.linspace(0, 1, n), ctx)
            for n in (21, 41, 81, 161)]
    assert order(errs) == pytest.approx(2.0, abs=0.2)
    x = np.linspace(0.0, 1.0, 81)
    a, d = nystrom_matrix(x, constant_kernel, volterra=True)
    u = np.linalg.solve(np.eye(x.size) - (a + np.diag(d)), np.ones(x.size))
    assert float(np.max(np.abs(u - np.exp(x)))) < 1e-4


def test_including_the_future_changes_the_volterra_solution_by_two_orders_of_magnitude():
    """The causal mask is not a detail. Solved with the correct half cell but with the future allowed to contribute
    -- the Fredholm matrix, which already has a zero diagonal -- the "solution" of the same equation leaves the
    domain of the exact one by two orders of magnitude, so a wrong mask is not a small error."""
    x = np.linspace(0.0, 1.0, 81)
    a_ok, d = nystrom_matrix(x, constant_kernel, volterra=True)      # tril(-1) plus the half self cell
    a_bad, _ = nystrom_matrix(x, constant_kernel)                    # every j != i, past and future
    u_ok = np.linalg.solve(np.eye(x.size) - (a_ok + np.diag(d)), np.ones(x.size))
    u_bad = np.linalg.solve(np.eye(x.size) - (a_bad + np.diag(d)), np.ones(x.size))
    assert float(np.max(np.abs(u_ok - np.exp(x)))) < 1e-4
    assert float(np.max(np.abs(u_bad - np.exp(x)))) > 10.0
    assert float(np.max(np.abs(u_bad - u_ok))) > 10.0


# -------------------------------------------------------------------------------------------------- plumbing
def test_complex_kernel_needs_two_fields_and_says_so():
    from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec

    spec = ProblemSpec(name="fredholm2", dim=1, coords=("x",), fields=("u",),
                       pde=PDETermSpec(kind="fredholm2", fields=("u",), coords=("x",), params={"lambda": 1.0}),
                       conditions=(), domain_bounds={"x": (0.0, 1.0)})
    from pinneapple_physics.pinn_solver.compiler.compile import compile_problem

    with pytest.raises(ValueError, match="complex"):
        compile_problem(spec)(Exact(lambda X: torch.zeros(X.shape[0], 1)), None, {
            "x_col": torch.as_tensor(np.linspace(0, 1, 41), dtype=torch.float64).reshape(-1, 1),
            "ctx": {"kernel_fn": helmholtz2_kernel, "diag_fn": helmholtz2_diag},
        })


def test_a_complex_kernel_with_two_fields_gives_two_residuals():
    kind, fields = "fredholm2", ("ur", "ui")
    assert kind in INTEGRAL_KINDS
    x = np.linspace(0.0, 1.0, 41)
    a, d = nystrom_matrix(x, helmholtz2_kernel, {"k": 1.0}, diag_fn=helmholtz2_diag)
    assert np.iscomplexobj(a) and np.iscomplexobj(d)
    u = np.linspace(0.0, 1.0, 41).reshape(-1, 1)
    res = integral_residuals(kind, fields={"ur": torch.as_tensor(u), "ui": torch.as_tensor(u)},
                             xcol=torch.as_tensor(x).reshape(-1, 1), coords=("x",), params={"lambda": 0.5},
                             ctx={"kernel_fn": helmholtz2_kernel, "diag_fn": helmholtz2_diag, "k": 1.0},
                             model=None, field_names=fields, split_fields=None)
    assert len(res) == 2 and all(r.shape == (41, 1) for r in res)


def test_the_batch_may_arrive_in_any_order():
    """The nodes are sorted internally; the residual comes back in the caller's order."""
    from pinneapple_physics.pinn_solver.compiler.compile import compile_problem

    spec = spec_of("fredholm2", ("u",), {"lambda": 1.0})
    ctx = {"kernel_fn": constant_kernel, "f_fn": lambda X, c: np.exp(X[:, 0]) - (E - 1.0)}
    fn = Exact(lambda X: torch.exp(X[:, 0:1]))
    model = compile_problem(spec)
    x = np.linspace(0.0, 1.0, 41)
    straight = model(fn, None, {"x_col": torch.as_tensor(x).reshape(-1, 1), "ctx": ctx})["pde"]
    perm = np.random.default_rng(0).permutation(x.size)
    shuffled = model(fn, None, {"x_col": torch.as_tensor(x[perm]).reshape(-1, 1), "ctx": ctx})["pde"]
    assert float(shuffled.detach()) == pytest.approx(float(straight.detach()), rel=1e-12)


def test_training_the_residual_reduces_it_by_two_orders_of_magnitude():
    """A smoke test of the road the training loop takes, not an accuracy test: the residual the compiler produces is
    differentiable and its minimisation moves, which is the only thing that can break here without breaking the
    tests above."""
    torch.manual_seed(0)
    x = np.linspace(0.0, 1.0, 41)
    a, d = nystrom_matrix(x, constant_kernel)
    at, dt = torch.as_tensor(a), torch.as_tensor(d)[:, None]
    nodes = torch.as_tensor(x).reshape(-1, 1)
    lam = 0.5
    f = torch.as_tensor(np.exp(x) - lam * (E - 1.0)).reshape(-1, 1)
    net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).double()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        u = net(nodes)
        loss = ((u - f - lam * (at @ u + dt * u)) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] / 100.0
