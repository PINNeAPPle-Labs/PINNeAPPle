"""Robin (and radiative-Robin) boundary condition support in FDMSolver.

These tests validate against a manufactured solution: since the interior
PDE here is source-free Laplace/heat, T(x, y) = A + B*x satisfies it
exactly regardless of B, so pinning the centerline and the two transverse
(y) edges to that exact profile via Dirichlet isolates the Robin edge as
the only unknown -- letting a real analytical target be checked without
needing a full 2D closed-form solution.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from pinneapple_physics.pde_environment.conditions import DirichletBC, RobinBC
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_simulation.numerical_solvers.fdm import FDMSolver, _apply_robin_2d, _build_2d_grid

L = 1.0
T0 = 300.0
H_CONV = 25.0
K_COND = 10.0
T_GAS = 800.0


def _manufactured_edges(B_star, nx=41, ny=9):
    def analytical(xx):
        return T0 + B_star * xx

    def on_edge(pts, edges):
        m = np.zeros(len(pts), dtype=bool)
        for e in edges:
            if e == "x0":
                m |= pts[:, 0] <= 1e-9
            if e == "x1":
                m |= pts[:, 0] >= L - 1e-9
            if e == "y0":
                m |= pts[:, 1] <= 1e-9
            if e == "y1":
                m |= pts[:, 1] >= 1.0 - 1e-9
        return m

    conditions = (
        DirichletBC("left", fields=("T",), selector=lambda pts, ctx: on_edge(pts, ["x0"]), value_fn=lambda pts, ctx: analytical(pts[:, 0])),
        DirichletBC("bottom", fields=("T",), selector=lambda pts, ctx: on_edge(pts, ["y0"]), value_fn=lambda pts, ctx: analytical(pts[:, 0])),
        DirichletBC("top", fields=("T",), selector=lambda pts, ctx: on_edge(pts, ["y1"]), value_fn=lambda pts, ctx: analytical(pts[:, 0])),
        RobinBC(name="surface", selector=lambda pts, ctx: on_edge(pts, ["x1"]), value_fn=lambda pts, ctx: np.full(len(pts), H_CONV * T_GAS)),
    )
    return conditions, analytical


def test_robin_poisson_matches_analytical_linear_profile():
    """-k dT/dx|_L = h(T_gas - T_s), steady 2D Poisson: T(x)=A+B*x with
    B = h(T_gas-A)/(hL-k) is the exact solution -- FDMSolver's Robin
    handling should reproduce it (via `_apply_robin_2d`, block-coupled
    with SOR every outer pass in `_poisson`)."""
    B_star = H_CONV * (T_GAS - T0) / (H_CONV * L - K_COND)
    conditions, analytical = _manufactured_edges(B_star)

    spec = ProblemSpec(
        name="steady_robin_1d", dim=2, coords=("x", "y"), fields=("T",),
        pde=PDETermSpec(kind="poisson", fields=("T",), coords=("x", "y"), params={"source": 0.0}),
        conditions=conditions, domain_bounds={"x": (0.0, L), "y": (0.0, 1.0)},
    )
    solver = FDMSolver(nx=41, ny=9)
    out = solver.solve_from_spec(spec, ctx_extra={"robin_coeffs": {"surface": {"a": H_CONV, "b": -K_COND}}})

    u = out.result.numpy()
    x = out.extras["coords"]["x"]
    j_mid = u.shape[1] // 2
    err = np.abs(u[:, j_mid] - analytical(x))
    assert err.max() < 1.0, f"max abs error {err.max()} vs analytical linear profile"


def test_robin_heat_transient_relaxes_to_same_steady_state():
    """The transient ADI path (_heat, with Robin embedded in each sweep's
    tridiagonal system -- see _robin_edge_specs/_adi_heat_2d) should relax,
    given enough time, to the exact same steady linear profile the
    elliptic solver finds."""
    B_star = H_CONV * (T_GAS - T0) / (H_CONV * L - K_COND)
    conditions, analytical = _manufactured_edges(B_star)

    spec = ProblemSpec(
        name="transient_robin_1d", dim=2, coords=("x", "y", "t"), fields=("T",),
        pde=PDETermSpec(kind="heat", fields=("T",), coords=("x", "y", "t"), params={"alpha": 1.0}),
        conditions=conditions, domain_bounds={"x": (0.0, L), "y": (0.0, 1.0), "t": (0.0, 5.0)},
    )
    solver = FDMSolver(nx=41, ny=9, nt=4000, theta=1.0)
    out = solver.solve_from_spec(spec, ctx_extra={"robin_coeffs": {"surface": {"a": H_CONV, "b": -K_COND}}})

    u = out.result.numpy()
    x = out.extras["coords"]["x"]
    j_mid = u.shape[1] // 2
    err = np.abs(u[:, j_mid] - analytical(x))
    assert err.max() < 5.0, f"max abs error {err.max()} vs analytical steady state"


def test_radiative_robin_node_solve_matches_independent_nonlinear_root():
    """Unit-level check of the nonlinear (radiative) Robin node solve in
    isolation: `_apply_robin_2d`'s Picard iteration, given the correct
    neighbor value, must reproduce the root of
    a*Ts - k*B + eps*sigma*(Ts^4 - t_env^4) = g
    found independently here (brentq, no FDM code involved) -- this is
    the mathematical core of the radiative extension.

    (Full elliptic/transient solves with strong radiative coupling in a
    stiff configuration are not guaranteed to converge to this same root
    from a cold start with the current block-coupled iteration -- see the
    module docstring and docs note in the PR; this test isolates the part
    that is fully validated: the per-node nonlinear solve itself.)
    """
    eps, sigma = 0.05, 5.670374419e-8

    def residual_Ts(Ts):
        B = (Ts - T0) / L
        return H_CONV * Ts - K_COND * B + eps * sigma * (Ts**4 - T_GAS**4) - H_CONV * T_GAS

    Ts_star = brentq(residual_Ts, T_GAS, 1133.34)
    B_star = (Ts_star - T0) / L

    nx, ny = 41, 9
    x, y, _, _, dx, dy = _build_2d_grid(0, L, nx, 0, 1, ny)
    u = np.zeros((nx, ny))
    for i in range(nx):
        u[i, :] = T0 + B_star * x[i]
    u[-1, :] = 500.0  # perturb the boundary away from the root

    conditions = (RobinBC(name="surface", selector=lambda pts, ctx: pts[:, 0] >= L - 1e-9, value_fn=lambda pts, ctx: np.full(len(pts), H_CONV * T_GAS)),)
    ctx = {"robin_coeffs": {"surface": {"a": H_CONV, "b": -K_COND, "radiative": {"epsilon": eps, "sigma": sigma, "t_env": T_GAS}}}}

    out = _apply_robin_2d(u.copy(), x, y, conditions, ctx, dx, dy, radiative_iters=20)
    assert abs(out[-1, 4] - Ts_star) < 1e-6
