"""Integral-equation presets: each has a closed-form solution (``pp.solve(problem, "analytic")``, registered from
:mod:`pinneapple_physics.closed_form.integral_equations`), so any method can be scored against the exact answer.

``integral_fredholm_unit_exp``       Fredholm, second kind, ``K = 1``, ``u = exp(x)``
``integral_fredholm_separable_exp``  Fredholm, second kind, degenerate ``K = x s`` (rank one), ``u = exp(x)``
``integral_volterra_unit_exp``       Volterra, second kind, ``K = 1``, ``u = exp(x)``
``integral_log_kernel_linear``       Fredholm, second kind, weakly singular ``K = ln|x - s|``, ``u = a + b x``

The kernel and the free term travel in ``pde.params`` as callables, which is how the compiler passes material maps and
sources to every other kind; nothing has to be injected into the batch.

Two things a caller has to know. The collocation points **are** the Nyström nodes, so a resampled collocation set
turns the quadrature error into noise -- for a convergence study, hand the solver a fixed grid and raise ``n_col`` with
it. And ``lambda`` must stay away from the eigenvalue of the kernel (``lambda = 1`` for ``K = 1``): there the system is
singular and the Fredholm alternative applies, so the exact solution exists but is not unique and no comparison means
anything. The presets default to ``lambda = 0.5`` for that reason.

Non-dimensional units, unit domain, one spatial coordinate.
"""
from __future__ import annotations

import math

from ..scales import ScaleSpec
from ..spec import PDETermSpec, ProblemSpec
from .registry import register_preset

_ATKINSON = ("Atkinson, The Numerical Solution of Integral Equations of the Second Kind, "
             "Cambridge University Press, 1997, ch. 3")


def _spec(kind: str, name: str, *, params: dict, u_range: tuple[float, float], references: tuple[str, ...],
          n_col: int) -> ProblemSpec:
    return ProblemSpec(
        name=name, dim=1, coords=("x",), fields=("u",),
        pde=PDETermSpec(kind=kind, fields=("u",), coords=("x",), params=params,
                        meta={"analytic": name, "physics": "integral_equations"}),
        conditions=(),
        sample_defaults={"n_col": n_col, "n_bc": 0, "n_ic": 0, "n_data": 0},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0),
        field_ranges={"u": u_range},
        references=references,
        domain_bounds={"x": (0.0, 1.0)},
    )


@register_preset("integral_fredholm_unit_exp")
def integral_fredholm_unit_exp(lam: float = 0.5, n_col: int = 81) -> ProblemSpec:
    """``u(x) = f(x) + lam int_0^1 u ds`` with ``K = 1`` and ``u = exp(x)``: the model has to recover a curve, not a
    constant, and the operator is full rank."""
    from pinneapple_physics.closed_form.integral_equations import fredholm_unit_exp_source
    from pinneapple_physics.pinn_solver.compiler.integral_kinds import constant_kernel

    params = {"lambda": lam, "kernel_fn": constant_kernel,
              "f_fn": lambda X, ctx: fredholm_unit_exp_source(X[:, 0], lam)}
    return _spec("fredholm2", "integral_fredholm_unit_exp", params=params,
                 u_range=(1.0, math.e), references=(_ATKINSON,), n_col=n_col)


@register_preset("integral_fredholm_separable_exp")
def integral_fredholm_separable_exp(lam: float = 0.5, n_col: int = 81) -> ProblemSpec:
    """``u(x) = f(x) + lam int_0^1 x s u(s) ds`` with ``u = exp(x)``: a rank-one kernel, so the solution is the free
    term plus a single constant times ``x`` -- the cheapest case in which a wrong quadrature still shows."""
    from pinneapple_physics.closed_form.integral_equations import fredholm_separable_exp_source
    from pinneapple_physics.pinn_solver.compiler.integral_kinds import product_kernel

    params = {"lambda": lam, "kernel_fn": product_kernel,
              "f_fn": lambda X, ctx: fredholm_separable_exp_source(X[:, 0], lam)}
    return _spec("fredholm2", "integral_fredholm_separable_exp", params=params,
                 u_range=(1.0, math.e), references=(_ATKINSON,), n_col=n_col)


@register_preset("integral_volterra_unit_exp")
def integral_volterra_unit_exp(n_col: int = 81) -> ProblemSpec:
    """``u(x) = 1 + int_0^x u ds`` with ``u = exp(x)``: causal, so the residual at ``x`` may not depend on the future
    and a wrong mask is visible in the solution."""
    from pinneapple_physics.closed_form.integral_equations import volterra_unit_exp_source
    from pinneapple_physics.pinn_solver.compiler.integral_kinds import constant_kernel

    params = {"lambda": 1.0, "kernel_fn": constant_kernel,
              "f_fn": lambda X, ctx: volterra_unit_exp_source(X[:, 0])}
    return _spec("volterra2", "integral_volterra_unit_exp", params=params,
                 u_range=(1.0, math.e), references=(_ATKINSON,), n_col=n_col)


@register_preset("integral_log_kernel_linear")
def integral_log_kernel_linear(lam: float = 0.5, a: float = 1.0, b: float = 0.5, n_col: int = 81) -> ProblemSpec:
    """``u(x) = f(x) + lam int_0^1 ln|x - s| u(s) ds`` with ``u = a + b x``: the weakly singular case, which only
    converges if the self cell is integrated analytically (``diag_fn``), and the preset that exercises that path."""
    from pinneapple_physics.closed_form.integral_equations import log_kernel_linear_source
    from pinneapple_physics.pinn_solver.compiler.integral_kinds import log_diag, log_kernel

    params = {"lambda": lam, "a": a, "b": b, "singular": True, "kernel_fn": log_kernel, "diag_fn": log_diag,
              "f_fn": lambda X, ctx: log_kernel_linear_source(X[:, 0], lam, a, b)}
    lo, hi = min(a, a + b), max(a, a + b)
    return _spec("fredholm2", "integral_log_kernel_linear", params=params,
                 u_range=(lo, hi),
                 references=(_ATKINSON, "Delves & Mohamed, Computational Methods for Integral Equations, "
                                        "Cambridge University Press, 1985, ch. 6"), n_col=n_col)
