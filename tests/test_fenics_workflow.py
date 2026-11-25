"""``external_solvers.fenics.solver``: the config→problem-spec translation, a solve that returns its points, and the
"what do I install" answer when FEniCS is absent (#2477).

The wrapper was dead for a long time -- it imported ``FEniCSBridge``, a class that does not exist (the bridge's class is
``FEnicsBridge``) and passed it keyword arguments its constructor does not take -- and nobody noticed because the import
was lazy. These tests fail on the first call, which is where a wrapper should fail if its translation is wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_simulation.external_solvers.fenics import (
    FEniCSConfig,
    FEniCSUnavailable,
    FEniCSWorkflow,
    problem_spec,
    solve_and_package,
)


def _fenics_ready() -> bool:
    from pinneapple_simulation.numerical_solvers import fenics_bridge

    return bool(fenics_bridge._FENICS_AVAILABLE)      # noqa: SLF001 - the bridge's own probe


needs_fenics = pytest.mark.skipif(not _fenics_ready(), reason="DOLFINx or legacy FEniCS is not installed here")


def _conduction_config(nx: int = 8) -> FEniCSConfig:
    return FEniCSConfig(pde="heat_equation_steady",
                        domain={"type": "rectangle", "x": [0.0, 1.0], "y": [0.0, 0.5], "nx": nx, "ny": nx // 2},
                        bcs=[{"side": "left", "value": 0.0}, {"side": "right", "value": 100.0}],
                        params={"k": 1.0, "f": 0.0})


def test_problem_spec_names_what_the_bridge_reads():
    """The translation, on its own: the bridge reads ``kind``, ``domain_bounds``, ``conditions``, ``parameters``."""
    spec = problem_spec(_conduction_config(nx=6))
    assert spec.kind == "heat_equation_steady"
    assert spec.domain_bounds == (0.0, 0.0, 1.0, 0.5)
    assert spec.mesh_nx == 6 and spec.mesh_ny == 3
    assert spec.parameters == {"k": 1.0, "f": 0.0}
    sides = {name: cond.side for name, cond in spec.conditions.items()}
    assert sorted(sides.values()) == ["left", "right"]
    assert {cond.value for cond in spec.conditions.values()} == {0.0, 100.0}


def test_problem_spec_keeps_a_named_boundary_and_an_extra_argument():
    cfg = FEniCSConfig(pde="linear_elasticity_plane_stress", domain={"x": [0, 1], "y": [0, 1]},
                       bcs=[{"name": "clamp", "side": "left", "value": [0.0, 0.0], "type": "dirichlet"}],
                       params={"E": 1e6, "nu": 0.3})
    spec = problem_spec(cfg)
    assert list(spec.conditions) == ["clamp"]
    assert spec.conditions["clamp"].value == [0.0, 0.0]
    assert spec.parameters["E"] == 1e6 and spec.domain_bounds == (0.0, 0.0, 1.0, 1.0)


def test_solve_and_package_says_what_to_install_when_fenics_is_missing(monkeypatch):
    """The old wrapper re-raised an ImportError worded like a missing optional dependency for a bug that installing
    FEniCS would not fix. Now the message is true, and it names the command."""
    from pinneapple_simulation.numerical_solvers import fenics_bridge

    monkeypatch.setattr(fenics_bridge, "_FENICS_AVAILABLE", False)
    with pytest.raises(FEniCSUnavailable) as excinfo:
        solve_and_package(_conduction_config())
    assert "fenics-dolfinx" in str(excinfo.value)


@needs_fenics
def test_solve_and_package_returns_the_field_with_its_points():
    """A linear profile (no source) is in the P1 space, so the answer is exact -- in the values *and* in the points
    that say where they are, which is what makes the result checkable at all."""
    cfg = _conduction_config(nx=8)
    sample = solve_and_package(cfg)
    assert "u" in sample.state and sample.domain["coords"]
    assert "ux" not in sample.state and "uy" not in sample.state      # a scalar solution has no components
    x = np.asarray(sample.domain["coords"]["x"])
    values = np.asarray(sample.state["u"]).ravel()
    assert values.size == x.size
    assert np.abs(values - 100.0 * x).max() < 1e-3             # T_right = 100, so the exact profile is 100 x
    assert sample.provenance["source"] == "fenics" and sample.provenance["pde"] == "heat_equation_steady"


@needs_fenics
def test_the_workflow_solves_and_sweeps_the_same_config():
    workflow = FEniCSWorkflow(_conduction_config(nx=6))
    single = workflow.solve()
    assert float(np.asarray(single.state["u"]).ravel().max()) == pytest.approx(100.0, abs=1e-3)
    sweep = workflow.sweep({"k": [1.0, 2.0]})
    assert len(sweep) == 2
    for sample in sweep:
        field = np.asarray(sample.state["u"]).ravel()
        assert np.abs(field - 100.0 * np.asarray(sample.domain["coords"]["x"])).max() < 1e-3
