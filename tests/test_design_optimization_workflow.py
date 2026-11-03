"""The design-optimisation workflow (#1188): a gmsh-parametrized cantilever solved by plane-stress FEM, optimised for
minimum mass under a stress and a deflection limit, re-verified on refined meshes, compared with the beam-theory
optimum, with its STEP read back; the generic path on a closed-form tool; negative controls."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("scipy")

from pinneapple_design.design_optimizer.cantilever import (  # noqa: E402
    export_step,
    tapered_cantilever,
    tapered_cantilever_beam,
)
from pinneapple_design.design_optimizer.workflow import design_optimization, richardson  # noqa: E402


def _status(r):
    return {c["code"]: c["status"] for c in r["checks"]}


def test_richardson_recovers_the_order_and_the_limit():
    vals = [1 + (0.1 / 2 ** k) ** 2 for k in range(3)]          # f(h) = 1 + h^2, h halved
    rr = richardson(vals)
    assert rr["order"] == pytest.approx(2.0, abs=1e-9)
    assert rr["value"] == pytest.approx(1.0, abs=1e-12)
    assert richardson([5.0, 5.0, 5.0])["gci"] == 0.0            # a mesh-independent output (the mass)
    assert richardson([1.0, 1.2, 1.1])["monotone"] is False     # oscillating: the finest, with its last change


def test_fem_converges_to_beam_theory():
    beam = tapered_cantilever_beam(0.12, 0.04)
    fem = [tapered_cantilever(0.12, 0.04, n_through=n) for n in (8, 16, 32)]
    s = richardson([f["stress_max_Pa"] for f in fem])
    d = richardson([f["tip_deflection_m"] for f in fem])
    # measured: 0.07 % and 0.35 % (sigma_xx vs the inclined-surface stress, plane stress vs beam: stated)
    assert abs(s["value"] / beam["stress_max_Pa"] - 1) < 0.005
    assert abs(d["value"] / beam["tip_deflection_m"] - 1) < 0.01
    assert 1.8 < s["order"] < 2.2 and 1.8 < d["order"] < 2.2
    assert min(f["sicn_min"] for f in fem) > 0.5
    assert fem[0]["mass_kg"] == pytest.approx(beam["mass_kg"], rel=1e-12)


def test_step_export_reads_back_the_design_volume(tmp_path):
    e = export_step(0.1, 0.05, str(tmp_path), thickness=0.03)
    assert e["n_solids"] == 1
    assert e["volume_read_m3"] == pytest.approx(0.03 * 1.0 * 0.075, rel=1e-9)


def test_cantilever_minimum_mass_matches_the_beam_theory_optimum(tmp_path):
    r = design_optimization(budget=60, out_dir=str(tmp_path))
    st = _status(r)
    assert r["verdict"] == "pass", [c for c in r["checks"] if c["status"] == "fail"]
    for code in ("optimizer.converged", "mesh.quality", "mesh.convergence", "constraints.stress_max_Pa",
                 "constraints.tip_deflection_m", "optimality.kkt", "reference.known_optimum", "export.geometry"):
        assert st[code] == "pass", code
    # the known optimum: beam theory, 7.850 kg (width at its lower bound, stress limit active); measured 0.04 %
    assert r["reference"]["objective"] == pytest.approx(7.85, rel=1e-4)
    assert abs(r["verified"]["mass_kg"]["value"] / r["reference"]["objective"] - 1) < 0.005
    assert r["design"]["thickness"] == pytest.approx(0.01, rel=1e-3)
    assert "thickness at lower bound" in r["sensitivity"]["active"]
    assert "stress_max_Pa" in r["sensitivity"]["active"]
    # the search mesh under-predicts the stress by about 7 %: the correction is what makes the optimum feasible
    first = r["history"][0]
    assert first["verified"]["stress_max_Pa"] > 150e6 * 1.05
    assert r["verified"]["stress_max_Pa"]["value"] <= 150e6 * 1.005
    # nominal optimum on an active limit: about half of the toleranced parts violate it (stated, not hidden)
    assert st["robustness.tolerances"] == "warn"
    assert 0.3 < r["robustness"]["p_feasible"] < 0.8
    assert r["export"]["path"].endswith("optimum.step")


def test_robust_back_off_reaches_the_target_at_a_small_mass_cost():
    r = design_optimization(budget=60, robust=True, reference=None)
    assert r["robust"] is not None
    assert r["robustness"]["p_feasible"] >= 0.9
    assert 0 < r["robust"]["objective_cost"] < 0.02                # measured: +0.82 %
    assert _status(r)["optimality.kkt"] == "pass"


def test_generic_tool_closed_form_optimum_and_multiplier():
    def tool(x, y):
        return {"f": (x - 1) ** 2 + (y - 2) ** 2, "g": x + y}
    r = design_optimization(tool=tool, bounds={"x": [0, 3], "y": [0, 3]}, objective="f",
                            constraints=[{"name": "g", "upper": 2.0}])
    assert r["design"]["x"] == pytest.approx(0.5, abs=2e-3) and r["design"]["y"] == pytest.approx(1.5, abs=2e-3)
    assert r["sensitivity"]["multipliers"]["g"] == pytest.approx(1.0, rel=0.02)   # d f* / d limit = -1
    assert r["verdict"] == "pass"


def test_negative_controls():
    # infeasible: no design satisfies the limit
    def tool(x, y):
        return {"f": x + y, "g": x + y}
    r = design_optimization(tool=tool, bounds={"x": [1, 2], "y": [1, 2]}, objective="f",
                            constraints=[{"name": "g", "upper": 1.0}], budget=20)
    assert r["verdict"] == "fail" and _status(r)["optimizer.feasible"] == "fail"

    # a reference model with the wrong load: the known-optimum check must fail
    def wrong(**kw):
        return tapered_cantilever_beam(**kw, load=6000.0)
    r = design_optimization(budget=40, reference=wrong, robust=False,
                            fidelity={"param": "n_through", "coarse": 4, "levels": [8, 16]})
    assert _status(r)["reference.known_optimum"] == "fail"


def test_surrogate_path_when_forced_still_verifies_on_the_tool():
    r = design_optimization(budget=40, force_surrogate=True,
                            fidelity={"param": "n_through", "coarse": 4, "levels": [8, 16, 32]})
    assert r["method"].startswith("surrogate-assisted")
    assert _status(r)["reference.known_optimum"] == "pass"
    assert _status(r)["constraints.stress_max_Pa"] == "pass"


def test_several_objectives_give_verified_pareto_picks():
    r = design_optimization(objective=[{"name": "mass_kg"}, {"name": "tip_deflection_m"}],
                            constraints=[{"name": "stress_max_Pa", "upper": 150e6}], budget=60, n_starts=1,
                            fidelity={"param": "n_through", "coarse": 4, "levels": [8, 16]})
    assert r["method"] == "nsga2" and len(r["pareto"]) >= 2
    masses = [p["verification"]["outputs"]["mass_kg"]["value"] for p in r["pareto"]]
    defl = [p["verification"]["outputs"]["tip_deflection_m"]["value"] for p in r["pareto"]]
    order = np.argsort(masses)
    assert all(np.diff(np.asarray(defl)[order]) < 0)             # lighter is more flexible along the front
    assert all(math.isfinite(m) for m in masses)


def test_flow_node_and_capability():
    from pinneapple_flow.planner.extensions import match
    from pinneapple_flow.registry import REGISTRY
    assert REGISTRY.get("design.optimization") is not None
    for text in ("otimização de projeto de uma viga em balanço com massa mínima e limite de tensão",
                 "design optimization of a bracket geometry for minimum mass"):
        ids = [c.id for _, c, _ in match(text)]
        assert ids and ids[0] == "design.optimization", (text, ids)
