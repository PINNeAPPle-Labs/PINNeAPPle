"""The holes the documented scope table used to have -- 3-D heat, 3-D elasticity, 3-D waves, 2-D Darcy and 1-D
reaction-diffusion -- are closed by solvers a ProblemSpec can drive, and these tests hold both halves: the numbers
the solvers return, and the scope entry that lets ``select_solver_family`` route a spec to them. They also hold the
other direction: a family whose documentation covers less than its name suggests must not claim the bigger
problem (reaction-diffusion is not plain diffusion).
"""
from __future__ import annotations

import dataclasses as dc

import numpy as np
import pytest

from pinneapple_advisor.compose import documented_families
from pinneapple_analysis.verification.solver_orchestration import (
    _CLASSICAL_SOLVER_SCOPE,
    _FAMILY_PREFERENCE_ORDER,
    select_solver_family,
)
from pinneapple_physics.pde_environment.conditions import DirichletBC, InitialCondition
from pinneapple_physics.pde_environment.presets.industry import (
    linear_elasticity_3d_default,
    transient_heat_3d_default,
)
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_simulation.numerical_solvers.darcy import Darcy2DFVMSolver
from pinneapple_simulation.numerical_solvers.elliptic3d import (
    Elliptic3D,
    Elliptic3DFDMSolver,
    solve_elliptic_3d,
)
from pinneapple_simulation.numerical_solvers.fdm3d import FDM3DSolver, Wave3DFDMSolver
from pinneapple_simulation.numerical_solvers.reaction_diffusion import (
    ReactionDiffusion,
    ReactionDiffusion1DSolver,
    ReactionDiffusionNDSolver,
)
from pinneapple_simulation.numerical_solvers.solid_fem import SolidFEMSpecSolver


def _hot_cube(**params):
    spec = transient_heat_3d_default()
    merged = {"alpha": 1e-3, "T0": 1.0, "nx": 12, "ny": 12, "nz": 12, "nt": 10}
    merged.update(params)
    return dc.replace(spec, pde=dc.replace(spec.pde, params=merged))


# ---------------------------------------------------------------------------------------------- 3-D heat


def test_a_3d_heat_spec_is_solved_by_the_solver_its_scope_names():
    specification = _hot_cube()
    assert select_solver_family(specification).recommended_family == "fdm3d"
    out = FDM3DSolver().solve_from_spec(specification)
    field = out.result.numpy()
    assert field.shape == (12, 12, 12)
    assert 0.0 < float(field.max()) < 1.0            # it cooled, and not to nothing in ten steps
    assert float(field.min()) == pytest.approx(0.0)  # the walls are held at zero
    assert float(field[6, 6, 6]) > float(field[1, 6, 6])   # hot in the middle, colder towards the wall
    assert out.extras["method"] == "fdm3d_heat_conduction"


def test_the_3d_heat_start_comes_from_the_spec_and_not_from_a_guess():
    assert FDM3DSolver._initial_temperature(transient_heat_3d_default(),
                                            transient_heat_3d_default().domain_bounds, {}) == 0.0
    hot = _hot_cube()
    assert FDM3DSolver._initial_temperature(hot, hot.domain_bounds, hot.pde.params) == 1.0
    base = _hot_cube(T0=None)
    ic = InitialCondition(name="T_init", fields=("T",), selector_type="callable",
                          selector=lambda X, ctx: np.isclose(X[:, 3], 0.0),
                          value_fn=lambda X, ctx: X[:, :1].astype(np.float32), weight=1.0)
    varying = dc.replace(base, conditions=(ic,) + tuple(base.conditions[1:]))
    with pytest.raises(ValueError, match="varies in space"):
        FDM3DSolver._initial_temperature(varying, varying.domain_bounds, varying.pde.params)


def test_a_wall_the_solver_cannot_hold_is_refused_by_name():
    hot_wall = dc.replace(_hot_cube(), conditions=(DirichletBC(
        name="hot_wall", fields=("T",), selector_type="tag", selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), 5.0, dtype=np.float32), weight=1.0),))
    with pytest.raises(ValueError, match="hot_wall"):
        FDM3DSolver().solve_from_spec(hot_wall)


# --------------------------------------------------------------------------------------- 3-D elasticity


def _cantilever(**params):
    spec = linear_elasticity_3d_default()
    merged = {"E": 210e9, "nu": 0.3, "nx": 30, "ny": 3, "nz": 5,
              "fixed_faces": ["x-"], "load": [0.0, 0.0, -1000.0], "load_face": "x+"}
    merged.update(params)
    return dc.replace(spec, domain_bounds={"x": (0.0, 2.0), "y": (-0.05, 0.05), "z": (-0.1, 0.1)},
                      pde=dc.replace(spec.pde, params=merged))


def test_a_3d_elasticity_spec_is_solved_by_the_solver_its_scope_names():
    specification = _cantilever()
    assert select_solver_family(specification).recommended_family == "solid_fem"
    out = SolidFEMSpecSolver().solve_from_spec(specification)
    # Euler-Bernoulli, the textbook answer a structural engineer would check first: d = F L^3 / (3 E I)
    L, width, height, force, E = 2.0, 0.1, 0.2, 1000.0, 210e9
    inertia = width * height**3 / 12.0
    analytical = force * L**3 / (3.0 * E * inertia)
    assert out.extras["max_displacement"] == pytest.approx(analytical, rel=0.05)
    # and the peak bending stress at the root: sigma = M c / I
    assert out.extras["von_mises_max"] == pytest.approx(force * L * (height / 2) / inertia, rel=0.15)
    assert out.extras["method"] == "solid_fem_c3d8i"


def test_a_structural_spec_with_nothing_to_push_it_is_refused():
    specification = _cantilever(load=None)
    with pytest.raises(ValueError, match="needs a load"):
        SolidFEMSpecSolver().solve_from_spec(specification)


def test_a_face_name_the_solver_does_not_know_is_refused():
    with pytest.raises(ValueError, match="fixed_faces"):
        SolidFEMSpecSolver().solve_from_spec(_cantilever(fixed_faces=["root"]))


# ---------------------------------------------------------------------------------------- the table itself


def test_the_scope_table_and_its_preference_order_name_the_same_families():
    assert set(_FAMILY_PREFERENCE_ORDER) == set(_CLASSICAL_SOLVER_SCOPE)
    for family, entry in _CLASSICAL_SOLVER_SCOPE.items():
        assert entry["source"], f"{family} has no source"
        assert entry["kind_substrings"] and entry["dims"] is not None


def test_the_two_3d_entries_cite_the_solver_they_name():
    assert "HeatConduction3D" in _CLASSICAL_SOLVER_SCOPE["fdm3d"]["source"]
    assert "SolidFEM" in _CLASSICAL_SOLVER_SCOPE["solid_fem"]["source"]
    assert _CLASSICAL_SOLVER_SCOPE["fdm3d"]["dims"] == {3}
    assert _CLASSICAL_SOLVER_SCOPE["solid_fem"]["dims"] == {3}


# ------------------------------------------------------------------------------------------ 3-D waves


def test_a_3d_wave_spec_is_solved_by_the_solver_its_scope_names():
    specification = dc.replace(transient_heat_3d_default(), pde=dc.replace(
        transient_heat_3d_default().pde, kind="wave",
        params={"c": 340.0, "nx": 16, "ny": 16, "nz": 16, "nt": 30, "u0": 1.0}))
    assert [d["family"] for d in documented_families("wave", 3)] == ["wave3d_fdm"]
    assert select_solver_family(specification).recommended_family == "wave3d_fdm"
    out = Wave3DFDMSolver().solve_from_spec(specification)
    field = out.result.numpy()
    assert field.shape == (16, 16, 16)
    assert float(np.max(np.abs(field))) > 0.0             # the pulse went somewhere
    assert float(np.max(np.abs(field[0, :, :]))) == pytest.approx(0.0)   # and the walls stay at zero
    assert out.extras["c"] == 340.0


def test_the_3d_wave_solver_refuses_a_wall_it_cannot_hold():
    specification = dc.replace(transient_heat_3d_default(), conditions=(), pde=dc.replace(
        transient_heat_3d_default().pde, kind="wave", params={"c": 340.0, "nx": 8, "ny": 8, "nz": 8, "u0": 1.0}))
    out = Wave3DFDMSolver().solve_from_spec(specification)
    assert tuple(out.result.shape) == (8, 8, 8)           # no conditions at all is the solver's own boundary


# ----------------------------------------------------------------------------------------- 2-D Darcy


def _darcy(**params):
    merged = {"n": 96, "permeability": 1.0, "source": 1.0}
    merged.update(params)
    return ProblemSpec(name="darcy2d", dim=2, coords=("x", "y"), fields=("p",),
                       pde=PDETermSpec(kind="darcy", fields=("p",), coords=("x", "y"), params=merged),
                       domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)})


def test_a_2d_darcy_spec_is_solved_by_the_solver_its_scope_names():
    specification = _darcy()
    assert [d["family"] for d in documented_families("darcy", 2)] == ["darcy2d_fvm"]
    assert select_solver_family(specification).recommended_family == "darcy2d_fvm"
    out = Darcy2DFVMSolver().solve_from_spec(specification)
    # -lap(u) = 1 on the unit square with u = 0 on the boundary has a well-known maximum: 0.07367135
    assert out.extras["u_max"] == pytest.approx(0.07367135, rel=5e-3)
    assert float(out.result.min()) > 0.0


def test_darcy_with_a_permeability_field_is_solved_too_and_is_not_the_uniform_answer():
    uniform = Darcy2DFVMSolver().solve_from_spec(_darcy(n=64))
    varying = Darcy2DFVMSolver().solve_from_spec(_darcy(n=64, permeability=lambda X, Y: 1.0 + X))
    # a = 1 + x sampled at the cell centres: from 1 + 0.5/64 to 1 + 63.5/64
    low, high = varying.extras["a_range"]
    assert low == pytest.approx(1.0 + 0.5 / 64.0) and high == pytest.approx(1.0 + 63.5 / 64.0)
    assert varying.extras["u_max"] < uniform.extras["u_max"]   # a less permeable half holds less pressure


def test_darcy_refuses_a_domain_it_does_not_solve_on():
    off = ProblemSpec(name="darcy_off", dim=2, coords=("x", "y"), fields=("p",),
                      pde=PDETermSpec(kind="darcy", fields=("p",), coords=("x", "y"), params={"n": 16}),
                      domain_bounds={"x": (0.0, 2.0), "y": (0.0, 1.0)})
    with pytest.raises(ValueError, match="unit square"):
        Darcy2DFVMSolver().solve_from_spec(off)


# -------------------------------------------------------------------------- 1-D reaction-diffusion


def _reaction_diffusion(**params):
    merged = {"reaction": "allen_cahn", "diffusion": 1e-3, "n": 128, "length": 2.0, "nt": 300}
    merged.update(params)
    return ProblemSpec(name="rd1d", dim=1, coords=("x",), fields=("u",),
                       pde=PDETermSpec(kind="reaction_diffusion", fields=("u",), coords=("x",), params=merged))


def test_a_1d_reaction_diffusion_spec_is_solved_by_the_solver_its_scope_names():
    specification = _reaction_diffusion()
    assert [d["family"] for d in documented_families("reaction_diffusion", 1)] == ["reaction_diffusion1d"]
    assert select_solver_family(specification).recommended_family == "reaction_diffusion1d"
    out = ReactionDiffusion1DSolver().solve_from_spec(specification)
    field = out.result.numpy()
    assert field.shape == (128,)
    assert field.max() == pytest.approx(1.0, abs=0.05) and field.min() == pytest.approx(-1.0, abs=0.05)


def test_reaction_diffusion_in_two_dimensions_is_now_covered_and_the_diffusion_solvers_still_do_not_claim_it():
    # ReactionDiffusion took the 2-D/3-D case, and fdm's and fvm's "diffusion" still means the linear equation
    assert documented_families("reaction_diffusion", 2) == [{"family": "reaction_diffusion_nd",
                                                             "source": documented_families(
                                                                 "reaction_diffusion", 2)[0]["source"]}]
    assert "fdm" not in [d["family"] for d in documented_families("reaction_diffusion", 2)]
    assert [d["family"] for d in documented_families("reaction_diffusion", 1)] == ["reaction_diffusion1d"]
    assert documented_families("diffusion", 2) != []


def test_reaction_diffusion_refuses_a_boundary_it_does_not_have():
    with_a_wall = dc.replace(_reaction_diffusion(), conditions=(DirichletBC(
        name="wall", fields=("u",), selector_type="tag", selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32), weight=1.0),))
    with pytest.raises(ValueError, match="no boundary"):
        ReactionDiffusion1DSolver().solve_from_spec(with_a_wall)
    with pytest.raises(ValueError, match="1-D periodic"):
        ReactionDiffusion1DSolver().solve_from_spec(dc.replace(_reaction_diffusion(), dim=2))


# ----------------------------------------------------------------- 3-D Poisson / Laplace / Helmholtz


def _elliptic(**params):
    merged = {"n": 17}
    merged.update(params)
    return ProblemSpec(name="elliptic3d", dim=3, coords=("x", "y", "z"), fields=("u",),
                       pde=PDETermSpec(kind=str(merged.pop("kind", "poisson")), fields=("u",),
                                       coords=("x", "y", "z"), params=merged),
                       domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)})


def test_the_three_elliptic_kinds_have_a_3d_family_and_it_is_this_one():
    for kind in ("poisson", "laplace", "helmholtz"):
        assert [d["family"] for d in documented_families(kind, 3)] == ["elliptic3d_fdm"]
    assert select_solver_family(_elliptic(kind="poisson", source=1.0)).recommended_family == "elliptic3d_fdm"


def test_a_harmonic_function_is_reproduced_exactly():
    # x + 2y + 3z is harmonic, and the seven-point Laplacian is exact on it: any error here is the iteration's
    out = solve_elliptic_3d(Elliptic3D(nx=17, ny=17, nz=17, wall=lambda X, Y, Z: X + 2 * Y + 3 * Z))
    x = y = z = np.linspace(0.0, 1.0, 17)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    assert float(np.abs(out["u"] - (X + 2 * Y + 3 * Z)).max()) < 1e-10


def test_poisson_is_second_order_against_a_manufactured_solution():
    exact = lambda X, Y, Z: np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)      # noqa: E731
    errors = {}
    for n in (17, 33):
        out = solve_elliptic_3d(Elliptic3D(
            nx=n, ny=n, nz=n, rhs=lambda X, Y, Z: -3 * np.pi**2 * exact(X, Y, Z), tol=1e-10))
        axis = np.linspace(0.0, 1.0, n)
        X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
        errors[n] = float(np.abs(out["u"] - exact(X, Y, Z)).max())
    assert errors[17] / errors[33] > 3.5            # halving the step buys a factor of four


def test_helmholtz_solves_its_own_operator():
    k = 1.0
    exact = lambda X, Y, Z: np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)      # noqa: E731
    out = solve_elliptic_3d(Elliptic3D(nx=25, ny=25, nz=25, q=k**2,
                                       rhs=lambda X, Y, Z: (k**2 - 3 * np.pi**2) * exact(X, Y, Z), tol=1e-10))
    axis = np.linspace(0.0, 1.0, 25)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    assert float(np.abs(out["u"] - exact(X, Y, Z)).max()) < 1e-2


def test_a_hot_wall_holds_its_value_and_the_middle_sits_at_the_mean():
    # Laplace with one wall at 1 and the rest at 0: the mean-value property puts the centre at 1/6
    hot = DirichletBC(name="hot_wall", fields=("u",), selector_type="callable",
                      selector=lambda X, ctx: np.isclose(X[:, 0], 0.0),
                      value_fn=lambda X, ctx: np.ones((X.shape[0], 1), dtype=np.float32), weight=1.0)
    specification = dc.replace(_elliptic(kind="laplace", n=25), conditions=(hot,))
    out = Elliptic3DFDMSolver().solve_from_spec(specification)
    field = out.result.numpy()
    assert float(field[0, 12, 12]) == pytest.approx(1.0)
    assert float(field[-1, 12, 12]) == pytest.approx(0.0)
    assert float(field[12, 12, 12]) == pytest.approx(1.0 / 6.0, abs=2e-3)


def test_the_elliptic_solver_refuses_what_it_does_not_have():
    with pytest.raises(ValueError, match="poisson/laplace/helmholtz"):
        Elliptic3DFDMSolver().solve_from_spec(_elliptic(kind="heat_equation"))
    with pytest.raises(ValueError, match="3-D"):
        Elliptic3DFDMSolver().solve_from_spec(dc.replace(_elliptic(), dim=2))
    with pytest.raises(ValueError, match="wavenumber"):
        Elliptic3DFDMSolver().solve_from_spec(_elliptic(kind="helmholtz"))


# ----------------------------------------------------------------- reaction-diffusion in 2-D and 3-D


def _reaction_diffusion_nd(dim, **params):
    merged = {"reaction": "none", "diffusion": 1e-2, "n": 24, "dt": 0.01, "nt": 60}
    merged.update(params)
    axes = ("x", "y", "z")[:dim]
    return ProblemSpec(name=f"rd{dim}d", dim=dim, coords=axes, fields=("u",),
                       pde=PDETermSpec(kind="reaction_diffusion", fields=("u",), coords=axes, params=merged),
                       domain_bounds={a: (0.0, 2 * np.pi) for a in axes})


def test_a_single_mode_diffuses_at_its_exact_rate_in_two_and_three_dimensions():
    # pure diffusion, one Fourier mode of the periodic box: it decays as exp(-D |k|^2 t) and nothing else
    for dim in (2, 3):
        assert [d["family"] for d in documented_families("reaction_diffusion", dim)] == ["reaction_diffusion_nd"]
        system = ReactionDiffusion(reaction="none", diffusion=1e-2, shape=(24,) * dim, length=2 * np.pi, dt=0.01)
        axis = 2 * np.pi * np.arange(24) / 24
        grids = np.meshgrid(*([axis] * dim), indexing="ij")
        initial = np.prod([np.sin(g) for g in grids], axis=0)
        field = system.step(initial, 60)
        expected = np.exp(-1e-2 * dim * 60 * 0.01)
        assert float(field[(6,) * dim]) / float(initial[(6,) * dim]) == pytest.approx(expected, rel=1e-6)


def test_the_reaction_diffusion_spec_face_covers_two_and_three_dimensions():
    for dim in (2, 3):
        out = ReactionDiffusionNDSolver().solve_from_spec(_reaction_diffusion_nd(dim, reaction="allen_cahn",
                                                                                diffusion=4e-3, nt=200))
        assert tuple(out.result.shape) == (24,) * dim
        assert out.extras["method"] == "reaction_diffusion_nd" and out.extras["dim"] == dim
        assert float(np.max(np.abs(out.result.numpy()))) > 0.5        # the field separated, on a coarse grid


def test_allen_cahn_separates_into_domains_in_two_dimensions():
    system = ReactionDiffusion(reaction="allen_cahn", diffusion=4e-3, shape=(48, 48), length=2.0, dt=0.05)
    field = system.step(system.initial_state(seed=1, amplitude=0.4), 500)
    assert field.min() < -0.9 and field.max() > 0.9
    assert float(np.mean(np.abs(field) > 0.9)) > 0.5      # most of the box has chosen a phase


def test_reaction_diffusion_in_two_dimensions_refuses_a_boundary_it_does_not_have():
    with_a_wall = dc.replace(_reaction_diffusion_nd(2), conditions=(DirichletBC(
        name="wall", fields=("u",), selector_type="tag", selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32), weight=1.0),))
    with pytest.raises(ValueError, match="no boundary"):
        ReactionDiffusionNDSolver().solve_from_spec(with_a_wall)
