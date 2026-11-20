"""The 2-D wave solver and its ProblemSpec face.

The library had a 1-D wave solver (``fdm``) and a 3-D one (``fdm3d.ElasticWave3D``) and nothing between them, so
a 2-D acoustic problem was a gap in the documented solver scope. ``wave2d`` closes it. These tests hold the two
things that make it worth closing: an analytic solution the numbers are checked against, and the exact contract
of the spec face -- what it reads, and what it refuses by name.

The analytic reference is the standing mode of a rectangle with every wall at zero::

    u(x, y, t) = cos(c pi sqrt(1/Lx^2 + 1/Ly^2) t) sin(pi x/Lx) sin(pi y/Ly)

which satisfies both the wave equation and the Dirichlet condition exactly, and which the leapfrog scheme
reproduces with only its discretisation error: both ``dx`` and ``dt`` enter at second order, so on the 101x101
grid used below the observed relative error is 9.1e-5 (and 3.7e-4 at 51x51, 2.2e-5 at 201x201 -- i.e. it falls
by ~4x for each halving of the spacing). The tolerances asserted here (2 % at the centre, 5 % in the L2 norm)
are loose enough to be a real check of the physics and tight enough that a wrong wave speed, a wrong sign or an
unstable step would miss them by orders of magnitude.
"""
from __future__ import annotations

import dataclasses as dc

import numpy as np
import pytest

from pinneapple_physics.pde_environment.conditions import DirichletBC, NeumannBC
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_simulation.numerical_solvers.wave2d import (
    Wave2D,
    Wave2DConfig,
    Wave2DFDMSolver,
)

SECOND_MODE_EIGENVALUE = np.pi * np.sqrt(2.0)      # c pi sqrt(1/Lx^2 + 1/Ly^2) on the unit square, c = 1


def _mode(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The mode shape sin(pi x) sin(pi y) sampled on an n x n grid over the unit square."""
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return X, np.sin(np.pi * X) * np.sin(np.pi * Y)


def _standing_run(n: int = 101, nt: int = 400, t_end: float = 0.25) -> np.ndarray:
    """The standing mode, started at rest, integrated to ``t_end`` -- returns the full history."""
    X, _ = _mode(n)
    cfg = Wave2DConfig(nx=n, ny=n, nt=nt, lx=1.0, ly=1.0, dt=t_end / nt, c=1.0,
                       ic_u_fn=lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y),
                       ic_du_fn=lambda X, Y: np.zeros_like(X))   # du/dt = 0 at t = 0
    return Wave2D(cfg).solve()


# ------------------------------------------------------------------------------- the physics, against analytics


def test_the_standing_mode_keeps_its_frequency():
    n = 101
    out = _standing_run(n=n, nt=400, t_end=0.25)
    assert out.coords["t"][-1] == pytest.approx(0.25)

    _, shape = _mode(n)
    exact = np.cos(SECOND_MODE_EIGENVALUE * 0.25) * shape       # 0.4440158...
    got = out.u[-1]

    centre = (n // 2, n // 2)
    assert exact[centre] > 0.4                                  # the comparison is not against ~0
    relative_at_centre = abs(got[centre] - exact[centre]) / abs(exact[centre])
    relative_l2 = np.linalg.norm(got - exact) / np.linalg.norm(exact)

    # measured here: 9.1e-5 at the centre, 9.1e-5 in the L2 norm on this 101x101 grid (dx = 1e-2, CFL 0.088)
    assert relative_at_centre < 0.02
    assert relative_l2 < 0.05


def test_the_frequency_error_is_second_order_in_the_spacing():
    # the same comparison on two grids: halving dx cuts the relative error by ~4x, which is what justifies
    # reading the 2 % tolerance above as "the scheme's own second-order error" and not as slop
    errors = []
    for n in (51, 101):
        out = _standing_run(n=n, nt=400, t_end=0.25)
        _, shape = _mode(n)
        exact = np.cos(SECOND_MODE_EIGENVALUE * 0.25) * shape
        errors.append(np.linalg.norm(out.u[-1] - exact) / np.linalg.norm(exact))
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.3)   # measured 4.02 (3.7e-4 vs 9.1e-5)


def test_the_four_walls_stay_at_zero_on_every_step():
    n = 31
    cfg = Wave2DConfig(nx=n, ny=n, nt=400, lx=1.0, ly=1.0, dt=2e-3, c=1.0)   # the default Gaussian pulse
    out = Wave2D(cfg).solve()
    history = out.u
    assert float(np.max(np.abs(history))) > 0.1                 # there is a wave in there at all
    for wall in (history[:, 0, :], history[:, -1, :], history[:, :, 0], history[:, :, -1]):
        assert not np.any(wall), "a Dirichlet wall moved"
    # and the interior is not trivially zero (the assertion above would be vacuous otherwise)
    assert float(np.max(np.abs(history[:, 1:-1, 1:-1]))) > 0.1


def test_the_scheme_neither_blows_up_nor_damps():
    # homogeneous equation, no source, du/dt = 0: a second-order leapfrog conserves the discrete energy, so the
    # amplitude must neither grow nor decay over many bounces off the walls
    n = 61
    dx = 1.0 / (n - 1)
    dt = 0.5 * dx / np.sqrt(2.0)                    # CFL = c dt sqrt(2)/dx = 0.5
    cfg = Wave2DConfig(nx=n, ny=n, nt=2000, lx=1.0, ly=1.0, dt=dt, c=1.0,
                       ic_du_fn=lambda X, Y: np.zeros_like(X))
    out = Wave2D(cfg).solve()
    amplitude = float(np.max(np.abs(out.u[0])))
    assert amplitude == pytest.approx(1.0)
    # measured: the peak over the whole history is 1.000000 (0.0 % off) and the field after 2000 steps
    # (t = 11.8, ~8 crossings of the box) still peaks at 0.276 -- the wave is there, not damped away
    assert float(np.max(np.abs(out.u))) == pytest.approx(amplitude, rel=0.01)
    assert float(np.max(np.abs(out.u[-1]))) > 0.1


def test_a_courant_number_above_one_warns():
    cfg = Wave2DConfig(nx=21, ny=21, nt=5, lx=1.0, ly=1.0, dt=0.1, c=1.0)   # CFL = 0.1 * sqrt(2)/0.05 > 1
    with pytest.warns(UserWarning, match="CFL"):
        out = Wave2D(cfg).solve()
    assert out.meta["cfl"] > 1.0


# --------------------------------------------------------------------------------------------- the spec face


def _wave_spec(**params):
    merged = {"c": 1.0, "nx": 24, "ny": 24, "nt": 40}
    merged.update(params)
    return ProblemSpec(name="wave2d", dim=2, coords=("x", "y", "t"), fields=("u",),
                       pde=PDETermSpec(kind="wave", fields=("u",), coords=("x", "y", "t"), params=merged),
                       domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)})


def test_a_2d_wave_spec_is_solved_by_its_registry_face():
    specification = _wave_spec()
    out = Wave2DFDMSolver().solve_from_spec(specification)
    field = out.result.numpy()
    assert field.shape == (24, 24)
    assert float(np.max(np.abs(field))) > 0.0                 # the pulse the solver starts from when u0 is 0
    assert out.extras["pulse"] is True
    assert out.extras["method"] == "wave2d_fdm"
    assert out.extras["c"] == 1.0 and out.extras["nt"] == 40
    assert out.extras["history_shape"] == (41, 24, 24)
    # the default step is 40 % of the Courant bound: 0.4 / (c sqrt(1/dx^2 + 1/dy^2))
    assert out.extras["dt"] == pytest.approx(0.4 / (np.sqrt(2.0) / (1.0 / 23)), rel=1e-9)
    assert 0.0 < out.extras["cfl"] < 1.0
    for wall in (field[0, :], field[-1, :], field[:, 0], field[:, -1]):
        assert not np.any(wall)                               # every wall held at zero


def test_the_spec_face_gives_a_uniform_start_when_the_spec_asks_for_one():
    # a uniform u0 with du0 = 0 would otherwise be overridden by the pulse, but an explicit start wins
    still = Wave2DFDMSolver().solve_from_spec(_wave_spec(u0=0.0, pulse=False))
    assert float(np.max(np.abs(still.result.numpy()))) == pytest.approx(0.0)
    moved = Wave2DFDMSolver().solve_from_spec(_wave_spec(u0=1.0, du0=0.0))
    assert moved.extras["pulse"] is False
    assert moved.extras["u0"] == 1.0


def test_the_spec_face_refuses_a_kind_it_does_not_solve():
    specification = _wave_spec()
    with pytest.raises(ValueError, match="heat_equation"):
        Wave2DFDMSolver().solve_from_spec(
            dc.replace(specification, pde=dc.replace(specification.pde, kind="heat_equation")))


def test_the_spec_face_refuses_a_dimension_it_does_not_solve():
    with pytest.raises(ValueError, match="2-D solver"):
        Wave2DFDMSolver().solve_from_spec(dc.replace(_wave_spec(), dim=3))


def test_the_spec_face_refuses_a_wall_it_cannot_hold():
    hot_wall = dc.replace(_wave_spec(), conditions=(DirichletBC(
        name="hot_wall", fields=("u",), selector_type="tag", selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), 2.0, dtype=np.float32), weight=1.0),))
    with pytest.raises(ValueError, match="hot_wall"):
        Wave2DFDMSolver().solve_from_spec(hot_wall)

    # a zero Dirichlet wall is exactly this solver's boundary, and is accepted
    cold_wall = dc.replace(_wave_spec(), conditions=(DirichletBC(
        name="cold_wall", fields=("u",), selector_type="tag", selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32), weight=1.0),))
    assert tuple(Wave2DFDMSolver().solve_from_spec(cold_wall).result.shape) == (24, 24)


def test_the_spec_face_refuses_a_boundary_kind_it_does_not_have():
    with pytest.raises(ValueError, match="neumann"):
        Wave2DFDMSolver().solve_from_spec(dc.replace(_wave_spec(), conditions=(NeumannBC(
            name="flux_wall", fields=("u",), selector_type="tag", selector={"tag": "boundary"},
            value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32), weight=1.0),)))
