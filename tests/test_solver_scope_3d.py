"""The two 3-D gaps the documented scope table used to have -- heat conduction and linear elasticity -- are
closed by solvers a ProblemSpec can drive, and these tests hold both halves: the numbers the solvers return, and
the scope entry that lets ``select_solver_family`` route a 3-D spec to them.
"""
from __future__ import annotations

import dataclasses as dc

import numpy as np
import pytest

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
from pinneapple_simulation.numerical_solvers.fdm3d import FDM3DSolver
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
