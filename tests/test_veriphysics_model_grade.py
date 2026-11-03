"""VeriPhysics grade of an external model (#1187): a correct model is approved; each corruption fails the check it
breaks; float32 models and models known only by predictions are graded at their own precision."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_physics.symbolic_pde.text import parse_system
from pinneapple_veriphysics.model_grade import grade_model, grade_predictions

A = 0.1
SYS = parse_system(["u_t = alpha*lap(u)"], [{"kind": "initial", "on": "t_min", "equation": "u = sin(pi*x)"},
                                          {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}],
                   coords=["x", "t"], fields=["u"], coefficients={"alpha": A})
BOX = {"x": (0.0, 1.0), "t": (0.0, 1.0)}


def exact(X, a=A):
    return (np.exp(-a * np.pi ** 2 * X[:, 1]) * np.sin(np.pi * X[:, 0]))[:, None]


XR = np.random.default_rng(1).random((300, 2))
REF = (XR, exact(XR))


def _st(r):
    return {c["code"]: c["status"] for c in r["checks"]}


def test_the_exact_solution_is_approved():
    r = grade_model(exact, SYS, BOX, reference=REF, validity=BOX)
    assert r["decision"] == "APPROVED"
    assert r["checks"][0]["value"][0]["rel_rms"] < 1e-6


@pytest.mark.parametrize("name, model, broken", [
    ("wrong diffusivity", lambda X: exact(X, 0.2), "grade.pde_residual"),
    ("boundary offset", lambda X: exact(X) + 0.05, "grade.conditions"),
    ("shifted input", lambda X: exact(X + np.array([0.05, 0.0])), "grade.conditions"),
])
def test_each_corruption_fails_the_check_it_breaks(name, model, broken):
    r = grade_model(model, SYS, BOX, reference=REF, validity=BOX)
    assert r["decision"] == "REJECT" and _st(r)[broken] == "fail" and _st(r)["grade.reference"] == "fail"


def test_a_shift_keeps_the_pde_and_breaks_only_the_conditions():
    st = _st(grade_model(lambda X: exact(X + np.array([0.05, 0.0])), SYS, BOX))
    assert st["grade.pde_residual"] == "pass" and st["grade.conditions"] == "fail"


def test_a_float32_model_is_graded_at_its_precision():
    r = grade_model(lambda X: exact(X).astype(np.float32).astype(np.float64), SYS, BOX)
    assert _st(r)["grade.pde_residual"] == "pass" and "0.00492" in r["method"]


def test_outside_the_declared_validity_is_flagged():
    r = grade_model(exact, SYS, BOX, validity={"x": (0.0, 0.5), "t": (0.0, 1.0)})
    assert _st(r)["grade.applicability"] == "fail"


def test_predictions_on_a_grid_separate_good_from_wrong():
    g = np.linspace(0, 1, 41)
    P = np.array(np.meshgrid(g, g, indexing="ij")).reshape(2, -1).T
    good = grade_predictions(P, exact(P), SYS, BOX)
    bad = grade_predictions(P, exact(P, 0.2), SYS, BOX)
    assert _st(good)["grade.pde_residual"] == "pass" and _st(bad)["grade.pde_residual"] == "fail"
    assert bad["checks"][0]["value"][0]["rel_rms"] > 100 * good["checks"][0]["value"][0]["rel_rms"]


def test_flow_and_planner():
    from pinneapple_flow.planner.extensions import match
    from pinneapple_flow.registry import REGISTRY
    assert "veriphysics.grade" in REGISTRY.types()
    assert match("verificar meu modelo externo treinado no physicsnemo com veriphysics")[0][1].id == \
        "veriphysics.grade"
