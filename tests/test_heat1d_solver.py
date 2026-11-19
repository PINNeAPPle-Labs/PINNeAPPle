"""1-D transient heat conduction, checked against its closed forms and against the spec it refuses to misread.

The scheme is Crank-Nicolson, and the three solutions below are the ones it reproduces: a straight line between
the two held ends (exact), a sine mode decaying as ``exp(-alpha pi^2 t / L^2) sin(pi x / L)`` (exact up to the
central-difference decay rate), and the steady parabola ``q x (L - x) / (2 alpha)`` of a constant source (exact:
a second difference of a quadratic is exact). The tolerance each test carries is the error the run should leave
behind at that time, not a slackened bound: the transient error decays as ``exp(-alpha pi^2 t / L^2)``, so three
diffusion times leave about ``1e-13`` of it, and ``1e-6`` is thirty orders of margin over that.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_physics.pde_environment.conditions import DirichletBC, InitialCondition, NeumannBC
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_simulation.numerical_solvers.heat1d import Heat1D, Heat1DSolver

ALPHA = 1.0
LENGTH = 1.0


def _explicit_step(nx: int, alpha: float = ALPHA, length: float = LENGTH) -> float:
    """``0.4 dx^2 / alpha``: the step ``Heat1D`` defaults to (the explicit bound), so the transient is resolved."""
    return 0.4 * (length / (nx - 1)) ** 2 / alpha


# --------------------------------------------------------------------------- the scheme, against closed forms


def test_a_linear_profile_is_a_steady_state_to_machine_precision():
    # the discrete Laplacian of a straight line is zero and the end rows are its exact values, so Crank-Nicolson
    # has nothing to move: what is left is rounding, accumulated over the steps
    nx = 51
    line = np.linspace(0.0, 1.0, nx)
    scheme = Heat1D(alpha=ALPHA, length=LENGTH, nx=nx, T_left=0.0, T_right=1.0, T_init=line, nt=200)
    field = scheme.solve()
    assert np.max(np.abs(field - line)) < 1e-12          # measured 1.1e-14


def test_the_profile_relaxes_to_the_straight_line_between_the_two_ends():
    nx = 21
    dt = _explicit_step(nx)
    nt = int(round(3.0 * LENGTH**2 / ALPHA / dt))        # three diffusion times: the transient is gone
    scheme = Heat1D(alpha=ALPHA, length=LENGTH, nx=nx, dt=dt, nt=nt, T_left=0.0, T_right=1.0, T_init=0.0)
    field = scheme.solve()
    assert np.max(np.abs(field - scheme.x / LENGTH)) < 1e-6   # measured 8.5e-14 at t = 3 L^2 / alpha
    assert field[0] == 0.0 and field[-1] == 1.0               # the ends stay where they were held


def test_a_sine_mode_decays_at_the_analytic_rate():
    # T(x, 0) = sin(pi x / L) with both ends at zero is an eigenvector of the discrete Laplacian too, so the shape
    # stays exactly sinusoidal and the only error is the decay rate: the discrete eigenvalue is
    # (4 / dx^2) sin^2(pi dx / 2) instead of pi^2, which is 8.2e-5 low at nx = 101 -- 4.1e-5 after the 0.49 turns
    # of exp(-alpha pi^2 t / L^2) this run makes
    nx, dt, t_end = 101, 1.0e-4, 0.05
    x = np.linspace(0.0, LENGTH, nx)
    scheme = Heat1D(alpha=ALPHA, length=LENGTH, nx=nx, dt=dt, nt=int(round(t_end / dt)),
                    T_left=0.0, T_right=0.0, T_init=np.sin(np.pi * x / LENGTH))
    field = scheme.solve()
    analytic = np.exp(-ALPHA * np.pi**2 * t_end / LENGTH**2) * np.sin(np.pi * x / LENGTH)
    mid = nx // 2
    assert abs(field[mid] - analytic[mid]) / analytic[mid] < 1e-3    # measured 4.1e-5
    assert np.max(np.abs(field - analytic)) < 1e-3                   # measured 2.5e-5 (absolute, peak 0.61)
    assert np.max(np.abs(field / field.max() - np.sin(np.pi * x / LENGTH))) < 1e-12   # measured 1.2e-15
    assert field[mid] == pytest.approx(0.6105227, rel=1e-3)          # exp(-pi^2 * 0.05)


def test_a_constant_source_leaves_the_analytic_parabola():
    # alpha T'' = -q with T = 0 at both ends: T = q x (L - x) / (2 alpha). A constant source enters Crank-Nicolson
    # at the middle of the step, so this steady state is exact and only the transient from a cold start is left.
    nx, q = 21, 1.0
    dt = _explicit_step(nx)
    scheme = Heat1D(alpha=ALPHA, length=LENGTH, nx=nx, dt=dt, nt=int(round(3.0 / ALPHA / dt)),
                    T_left=0.0, T_right=0.0, T_init=0.0, source=q)
    field = scheme.solve()
    parabola = q * scheme.x * (LENGTH - scheme.x) / (2.0 * ALPHA)
    assert np.max(np.abs(field - parabola)) < 1e-4                    # measured 1.7e-14
    assert field.max() == pytest.approx(q * LENGTH**2 / (8.0 * ALPHA), abs=1e-4)   # the peak, at x = L / 2


def test_a_source_that_depends_on_space_is_honoured():
    # q(x) = 1 everywhere is the parabola above; the callable form is what a distributed source uses, and it must
    # reach the same answer
    nx = 21
    dt = _explicit_step(nx)
    scheme = Heat1D(alpha=ALPHA, length=LENGTH, nx=nx, dt=dt, nt=int(round(3.0 / dt)),
                    T_left=0.0, T_right=0.0, T_init=0.0, source=lambda x, t: np.ones_like(x))
    field = scheme.solve()
    parabola = scheme.x * (LENGTH - scheme.x) / (2.0 * ALPHA)
    assert np.max(np.abs(field - parabola)) < 1e-4                    # measured 1.7e-14


def test_a_scheme_the_parameters_rule_out_is_refused():
    with pytest.raises(ValueError, match="alpha"):
        Heat1D(alpha=0.0)
    with pytest.raises(ValueError, match="nx"):
        Heat1D(nx=2)
    with pytest.raises(ValueError, match="profile"):
        Heat1D(nx=5, T_init=np.zeros(4))


# ------------------------------------------------------------------------------------- the ProblemSpec face


def _spec(kind: str = "heat_equation", dim: int = 1, params: dict | None = None, conditions: tuple = (),
          bounds: dict | None = None, coords: tuple = ("x", "t")) -> ProblemSpec:
    return ProblemSpec(
        name="heat1d_case", dim=dim, coords=coords, fields=("T",),
        pde=PDETermSpec(kind=kind, fields=("T",), coords=coords,
                        params=params if params is not None else {"alpha": 0.01, "nx": 81, "nt": 40,
                                                                  "T_init": 20.0}),
        conditions=conditions,
        domain_bounds=bounds if bounds is not None else {"x": (0.0, 2.0), "t": (0.0, 1.0)},
    )


def _held(position: float, value: float, name: str) -> DirichletBC:
    """A Dirichlet condition on the end at ``position`` (selected by a callable on x), held at ``value``."""
    return DirichletBC(
        name, ("T",), "callable",
        (lambda p: lambda X, ctx: np.isclose(X[:, 0], p))(position),
        (lambda v: lambda X, ctx: np.full((X.shape[0], 1), v, dtype=np.float32))(value), 10.0)


def test_the_spec_face_holds_the_ends_and_fills_between_them():
    specification = _spec(conditions=(_held(0.0, 10.0, "cold_end"), _held(2.0, 30.0, "hot_end")))
    out = Heat1DSolver().solve_from_spec(specification)
    field = out.result.numpy()
    assert field.shape == (81,)
    assert field[0] == pytest.approx(10.0) and field[-1] == pytest.approx(30.0)
    # the maximum principle: the start (20) and both ends lie in [10, 30], so the interior does too
    assert float(field.min()) >= 10.0 - 1e-5 and float(field.max()) <= 30.0 + 1e-5
    assert np.all(np.diff(field) >= -1e-5)               # and nothing turns around between the two ends
    assert field[1] < field[-2]                          # the cold end is at x = 0
    assert out.extras["method"] == "heat1d_fdm"
    assert out.extras["alpha"] == 0.01
    assert out.extras["dt"] == pytest.approx(_explicit_step(81, alpha=0.01, length=2.0))   # the default step
    assert out.extras["nt"] == 40
    assert out.extras["T_left"] == 10.0 and out.extras["T_right"] == 30.0
    assert out.extras["x"][0] == 0.0 and out.extras["x"][-1] == 2.0 and len(out.extras["x"]) == 81
    assert float(out.losses["max_T"]) == pytest.approx(30.0)


def test_the_spec_face_reads_the_diffusivity_grid_and_start_from_the_spec():
    # k / (rho cp) instead of alpha, a given step and number of steps, and a uniform start from the parameters
    specification = _spec(params={"k": 500.0, "rho": 1000.0, "cp": 0.5, "nx": 41, "dt": 0.05, "nt": 20,
                                  "T_init": 5.0},
                          conditions=(_held(0.0, 0.0, "left"), _held(2.0, 0.0, "right")))
    out = Heat1DSolver().solve_from_spec(specification)
    assert out.extras["alpha"] == pytest.approx(500.0 / (1000.0 * 0.5))     # 1.0: k / (rho cp), not a default
    assert out.extras["dt"] == 0.05 and out.extras["nt"] == 20
    field = out.result.numpy()
    assert field.shape == (41,)
    assert field[0] == 0.0 and field[-1] == 0.0
    assert 0.0 < float(field.max()) < 5.0                # it cools towards the zero ends, from a start of 5
    assert float(field.max()) == float(field[20])        # and it is coldest at the ends, warmest in the middle


def test_an_end_no_condition_covers_is_held_at_zero():
    # documented behaviour rather than a guess: a spec with no boundary condition is a bar with both ends at zero
    out = Heat1DSolver().solve_from_spec(_spec(params={"alpha": 0.01, "nx": 21, "nt": 3, "T_init": 1.0},
                                               conditions=()))
    assert out.extras["T_left"] == 0.0 and out.extras["T_right"] == 0.0
    field = out.result.numpy()
    assert field[0] == 0.0 and field[-1] == 0.0
    assert 0.0 < float(field.max()) < 1.0                # the middle is still warm after three steps
    assert float(field[10]) == float(field.max())


def test_the_initial_condition_is_read_when_it_is_a_constant():
    start = InitialCondition(name="T_start", fields=("T",), selector_type="callable",
                             selector=lambda X, ctx: np.isclose(X[:, 1], 0.0),
                             value_fn=lambda X, ctx: np.full((X.shape[0], 1), 5.0, dtype=np.float32),
                             weight=10.0)
    out = Heat1DSolver().solve_from_spec(_spec(params={"alpha": 0.01, "nx": 21, "nt": 0}, conditions=(start,)))
    field = out.result.numpy()
    assert field[1] == pytest.approx(5.0) and field[-2] == pytest.approx(5.0)   # nothing has moved yet
    assert field[0] == 0.0 and field[-1] == 0.0                                 # the uncovered ends are zero


# ------------------------------------------------------------------------------------------------- refusals


def test_a_kind_the_solver_does_not_solve_is_refused_by_name():
    with pytest.raises(ValueError, match="wave"):
        Heat1DSolver().solve_from_spec(_spec(kind="wave", conditions=(_held(0.0, 0.0, "a"), _held(2.0, 0.0, "b"))))


def test_a_two_dimensional_spec_is_refused():
    with pytest.raises(ValueError, match="1-D"):
        Heat1DSolver().solve_from_spec(_spec(dim=2, conditions=(_held(0.0, 0.0, "a"), _held(2.0, 0.0, "b"))))


def test_an_end_value_that_moves_in_time_is_refused_by_condition_name():
    ramp = DirichletBC(
        "ramped_end", ("T",), "callable", lambda X, ctx: np.isclose(X[:, 0], 0.0),
        lambda X, ctx: (10.0 + X[:, 1:2]).astype(np.float32), 10.0)
    with pytest.raises(ValueError, match="ramped_end"):
        Heat1DSolver().solve_from_spec(_spec(conditions=(ramp, _held(2.0, 30.0, "hot_end"))))


def test_a_boundary_the_solver_does_not_have_is_refused():
    flux = NeumannBC("flux_end", ("T",), "callable", lambda X, ctx: np.isclose(X[:, 0], 2.0),
                     lambda X, ctx: np.ones((X.shape[0], 1), dtype=np.float32))
    with pytest.raises(ValueError, match="neumann"):
        Heat1DSolver().solve_from_spec(_spec(conditions=(_held(0.0, 10.0, "cold_end"), flux)))


def test_an_initial_condition_that_varies_in_space_is_refused_by_name():
    profile = InitialCondition(name="T_profile", fields=("T",), selector_type="callable",
                               selector=lambda X, ctx: np.isclose(X[:, 1], 0.0),
                               value_fn=lambda X, ctx: np.sin(np.pi * X[:, :1] / 2.0).astype(np.float32),
                               weight=10.0)
    with pytest.raises(ValueError, match="T_profile"):
        Heat1DSolver().solve_from_spec(_spec(params={"alpha": 0.01, "nx": 21, "nt": 5},
                                             conditions=(profile, _held(0.0, 0.0, "a"), _held(2.0, 0.0, "b"))))


def test_a_missing_diffusivity_is_refused():
    with pytest.raises(ValueError, match="alpha"):
        Heat1DSolver().solve_from_spec(_spec(params={"nx": 21, "nt": 5},
                                             conditions=(_held(0.0, 0.0, "a"), _held(2.0, 0.0, "b"))))
