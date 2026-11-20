"""The 3-D Burgers solver, held against things that are known exactly.

Four independent checks, in the order the module's docstring argues them:

1. a field whose only non-zero component is constant along its own axis has ``u.grad(u) = 0`` identically, so it
   must decay as the heat kernel does -- ``u_x = exp(-nu t) sin y`` on ``[0, 2 pi]^3``, asserted at 1e-12;
2. the pseudo-spectral advection kernel reproduces ``-u u_x = -sin x cos x = -sin(2x)/2`` for ``u = (sin x, 0, 0)``;
3. a field that varies only along ``x`` is the 1-D Burgers equation, and the 3-D solver's answer is compared with
   the Cole-Hopf exact solution of that 1-D problem (and with the library's own 1-D ``fdm`` Burgers solver, whose
   first-order upwind error is the limit of that second comparison);
4. the energy of a smooth random field decreases at every step and ``max|u|`` never grows (the component-wise
   maximum principle), plus the spec face: what it returns, and the specs it refuses by name.

Every tolerance below is the measured value in the comment beside it with margin, not a tuning knob.
"""
from __future__ import annotations

import dataclasses as dc

import numpy as np
import pytest
import torch

from pinneapple_physics.pde_environment.conditions import DirichletBC, InitialCondition, NeumannBC
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_simulation.numerical_solvers.burgers3d import (
    Burgers3D,
    Burgers3DSpectralSolver,
    BurgersConfig3D,
    burgers_cole_hopf_1d,
)
from pinneapple_simulation.numerical_solvers.fdm import FDMSolver
from pinneapple_simulation.numerical_solvers.registry import SolverRegistry

TWO_PI = 2 * np.pi


def _single_mode(X, Y, Z):
    """``u = (sin y, 0, 0)``: the one non-zero component is constant along its own axis, so ``u.grad(u)`` vanishes."""
    return np.stack([np.sin(Y), np.zeros_like(X), np.zeros_like(X)])


def _along_x(X, Y, Z):
    """``u = (sin x, 0, 0)``: the same shape of field along ``x``, where it does advect itself."""
    return np.stack([np.sin(X), np.zeros_like(X), np.zeros_like(X)])


def _burgers_spec(**params):
    """A hand-built ``ProblemSpec`` for the 3-D Burgers equation on the periodic box ``[0, 2 pi]^3``."""
    merged = {"nu": 0.05, "n": 12, "nt": 5}
    merged.update(params)
    coords = ("x", "y", "z", "t")
    return ProblemSpec(
        name="burgers3d", dim=3, coords=coords, fields=("u",),
        pde=PDETermSpec(kind="burgers", fields=("u",), coords=coords, params=merged),
        domain_bounds={"x": (0.0, TWO_PI), "y": (0.0, TWO_PI), "z": (0.0, TWO_PI)},
    )


# ------------------------------------------------------------------------------------------- 1. viscous decay


def test_a_mode_that_does_not_advect_itself_decays_at_the_viscous_rate():
    nu, nt, dt = 0.1, 50, 0.01
    system = Burgers3D(BurgersConfig3D(nu=nu, shape=(8, 32, 8), length=TWO_PI, dt=dt, nt=nt,
                                       ic=_single_mode, save_every=nt))
    u0 = system.initial_field()
    # the premise of the closed form: for this field the non-linear term is exactly zero
    assert float(np.max(np.abs(system.nonlinear(np.fft.fftn(u0, axes=(1, 2, 3)), 0.0)))) == 0.0

    out = system.solve()
    assert out.u.shape == (2, 3, 8, 32, 8)
    assert out.meta["t_end"] == pytest.approx(nt * dt)
    assert out.meta["dealiased"] is True

    _, Y, _ = system.grid
    exact = np.exp(-nu * out.meta["t_end"]) * np.sin(Y)
    assert float(np.max(np.abs(out.u[-1][0] - exact))) < 1e-12      # measured 1.1e-15
    assert float(np.max(np.abs(out.u[-1][1:]))) == 0.0              # u_y and u_z stay exactly zero
    # the energy of a single decaying mode falls as exp(-2 nu t) -- the same number, seen through the norm
    energy = out.meta["energy"]
    assert float(energy[-1] / energy[0]) == pytest.approx(np.exp(-2 * nu * nt * dt), rel=1e-12)


def test_the_advection_term_of_a_one_component_field_is_minus_half_the_double_angle():
    system = Burgers3D(BurgersConfig3D(nu=0.1, shape=(16, 8, 8), length=TWO_PI, ic=_along_x))
    X, _, _ = system.grid
    conv = np.fft.ifftn(system.nonlinear(np.fft.fftn(system.initial_field(), axes=(1, 2, 3)), 0.0),
                        axes=(1, 2, 3)).real
    # u.grad(u) = sin x cos x = sin(2x)/2 in the x component, and nothing in the other two
    assert float(np.max(np.abs(conv[0] + 0.5 * np.sin(2 * X)))) < 1e-13   # measured 4.4e-16
    assert float(np.max(np.abs(conv[1:]))) == 0.0


# -------------------------------------------------------------------------------------- 2. 1-D reduction


def _burgers_1d_spec(nu: float, t_end: float) -> ProblemSpec:
    """A 1-D Burgers ``ProblemSpec`` for :class:`FDMSolver`, starting from the same ``sin x``."""
    initial = InitialCondition(name="u0", fields=("u",),
                               value_fn=lambda X, ctx: np.sin(X[:, :1]).astype(np.float32))
    return ProblemSpec(
        name="burgers_1d", dim=1, coords=("x", "t"), fields=("u",),
        pde=PDETermSpec(kind="burgers", fields=("u",), coords=("x", "t"), params={"nu": nu}),
        conditions=(initial,), domain_bounds={"x": (0.0, TWO_PI), "t": (0.0, t_end)},
    )


def test_a_field_that_varies_only_along_x_is_one_dimensional_burgers():
    nu, t_end = 0.1, 0.5
    system = Burgers3D(BurgersConfig3D(nu=nu, shape=(96, 8, 8), length=TWO_PI, ic=_along_x, t_end=t_end))
    out = system.solve()
    x = out.coords["x"]
    u = out.u[-1][0][:, 0, 0]

    exact = burgers_cole_hopf_1d(np.sin(x), TWO_PI, nu, t_end)
    assert float(np.max(np.abs(u - exact))) < 5e-7                 # measured 2.5e-8 at n = 96
    assert float(np.linalg.norm(u - exact) / np.linalg.norm(exact)) < 5e-8   # measured 1.5e-8
    assert float(np.max(np.abs(out.u[-1][1:]))) == 0.0              # the transverse components stay zero

    # and against the library's own 1-D Burgers solver (FTCS + upwind): their difference is that solver's own
    # first-order error against the exact answer, so this comparison can only be as tight as ~1e-3
    one_d = FDMSolver(nx=512, ny=2).solve_from_spec(_burgers_1d_spec(nu, t_end))
    x_fdm = np.linspace(0.0, TWO_PI, 512)
    u_fdm = np.asarray(one_d.result.numpy()[:, -1], dtype=np.float64)
    ours = np.interp(x_fdm, x, u)
    interior = slice(1, -1)                                        # x_fdm includes both endpoints, x does not
    rel = np.linalg.norm(ours[interior] - u_fdm[interior]) / np.linalg.norm(u_fdm[interior])
    assert rel < 1e-2                                              # measured 4.0e-3
    exact_fdm = burgers_cole_hopf_1d(np.sin(x_fdm), TWO_PI, nu, t_end)
    fdm_error = np.linalg.norm(u_fdm[interior] - exact_fdm[interior]) / np.linalg.norm(u_fdm[interior])
    assert fdm_error < 1e-2                                        # measured 3.1e-3: the 1-D solver's own error
    assert rel < 2 * fdm_error                                     # unrefined grids, so it is the same order


# ------------------------------------------------------------------------------ 3. the energy never grows


def test_the_energy_of_a_smooth_random_field_decreases_at_every_step():
    shape, nu = (16, 16, 16), 0.05
    system = Burgers3D(BurgersConfig3D(nu=nu, shape=shape, length=TWO_PI))
    rng = np.random.default_rng(0)
    spectrum = (rng.standard_normal((3,) + shape) + 1j * rng.standard_normal((3,) + shape)) * system.mask
    u0 = np.real(np.fft.ifftn(spectrum, axes=(1, 2, 3)))
    u0 *= 0.5 / float(np.max(np.abs(u0)))

    out = Burgers3D(BurgersConfig3D(nu=nu, shape=shape, length=TWO_PI, ic=u0, dt=0.01, nt=100,
                                    save_every=1)).solve()
    energy = out.meta["energy"]
    assert energy.size == 101
    assert np.all(np.diff(energy) < 0.0)          # measured: every one of the 100 steps negative, 6.44 -> 1.42
    # the diffusion term can only remove energy, and each component obeys the maximum principle
    assert float(np.max(energy)) == pytest.approx(float(energy[0]))
    assert float(np.max(np.abs(out.u))) <= float(np.max(np.abs(u0))) + 1e-12


# ------------------------------------------------------------------------------------- 4. the spec face


def test_a_burgers_spec_comes_back_as_the_three_component_field():
    out = Burgers3DSpectralSolver().solve_from_spec(_burgers_spec())
    field = out.result
    assert tuple(field.shape) == (3, 12, 12, 12)
    assert field.dtype == torch.float32
    assert float(field.abs().max()) > 0.9                      # the Taylor-Green vortex, evolved
    assert set(out.losses) == {"max_abs_u"}

    extras = out.extras
    assert extras["method"] == "burgers3d_spectral"
    assert extras["shape"] == (12, 12, 12) and extras["nu"] == pytest.approx(0.05)
    assert extras["dealiased"] is True
    assert 0.0 < extras["cfl"] <= 0.5 + 1e-9
    assert extras["t_end"] == pytest.approx(extras["nt"] * extras["dt"])
    assert extras["history_shape"] == (2, 3, 12, 12, 12)
    assert set(extras["coords"]) == {"x", "y", "z", "t"}
    # the reported energy is the kinetic energy of the field that was returned
    volume = (TWO_PI / 12) ** 3
    assert extras["energy"] == pytest.approx(0.5 * float((field.double() ** 2).sum()) * volume, rel=1e-5)
    # a spec handed to the registry's build goes to solve_from_spec (SolverBase.solve dispatches on ProblemSpec)
    via_registry = SolverRegistry.build("burgers3d_spectral").solve(_burgers_spec())
    assert torch.allclose(via_registry.result, field, atol=1e-6)


def test_the_inviscid_equation_and_a_foreign_kind_are_refused_by_name():
    solver = Burgers3DSpectralSolver
    base = _burgers_spec()
    with pytest.raises(ValueError, match="inviscid"):
        solver().solve_from_spec(dc.replace(base, pde=dc.replace(base.pde, kind="inviscid_burgers")))
    with pytest.raises(ValueError, match="inviscid"):
        solver().solve_from_spec(_burgers_spec(nu=0.0))
    with pytest.raises(ValueError, match="kind is 'laplace'"):
        solver().solve_from_spec(dc.replace(base, pde=dc.replace(base.pde, kind="laplace")))
    with pytest.raises(ValueError, match="params\\['nu'\\]"):
        solver().solve_from_spec(dc.replace(base, pde=dc.replace(base.pde, params={"n": 12})))
    with pytest.raises(ValueError, match="three-dimensional"):
        solver().solve_from_spec(dc.replace(base, dim=2))


def test_a_wall_on_a_periodic_box_is_refused_by_name():
    base = _burgers_spec()
    for condition, kind in ((DirichletBC({"u": 0.0}), "dirichlet"), (NeumannBC({"u": 1.0}), "neumann")):
        spec = dc.replace(base, conditions=(condition,))
        with pytest.raises(ValueError, match=f"{kind} '{condition.name}'"):
            Burgers3DSpectralSolver().solve_from_spec(spec)


def test_the_field_is_the_three_component_vector_and_a_scalar_start_is_refused():
    base = _burgers_spec()
    scalar = InitialCondition(name="u0", fields=("u",), value_fn=lambda X, ctx: np.sin(X[:, :1]))
    with pytest.raises(ValueError, match="three-component"):
        Burgers3DSpectralSolver().solve_from_spec(dc.replace(base, conditions=(scalar,)))

    with pytest.raises(ValueError, match="must be \\(3, 12, 12, 12\\)"):
        Burgers3DSpectralSolver().solve_from_spec(_burgers_spec(ic=lambda X, Y, Z: np.sin(X)))

    vector = InitialCondition(
        name="u0", fields=("u",),
        value_fn=lambda X, ctx: np.stack([np.sin(X[:, 1]), np.zeros(len(X)), np.zeros(len(X))], axis=1))
    out = Burgers3DSpectralSolver().solve_from_spec(dc.replace(base, conditions=(vector,)))
    assert tuple(out.result.shape) == (3, 12, 12, 12)
    assert out.extras["ic"] == "spec_initial_condition"
    # the spec's own starting field is the one used: it agrees with the y-sine decayed by the run's time
    y = TWO_PI * np.arange(12) / 12
    expected = np.exp(-0.05 * out.extras["t_end"]) * np.sin(y)
    assert float(np.max(np.abs(out.result[0, 0, :, 0].numpy() - expected))) < 1e-5   # float32 round-trip


def test_the_advective_courant_number_warns_and_then_refuses():
    # 12 points on 2 pi: 1/dx + 1/dy + 1/dz = 5.73 at max|u| = 1, so dt = 0.1 is cfl 0.57 and dt = 0.5 is cfl 2.9
    with pytest.warns(UserWarning, match="Courant"):
        Burgers3DSpectralSolver().solve_from_spec(_burgers_spec(nt=2, dt=0.1))
    with pytest.raises(ValueError, match="Courant"):
        Burgers3DSpectralSolver().solve_from_spec(_burgers_spec(nt=2, dt=0.5))
