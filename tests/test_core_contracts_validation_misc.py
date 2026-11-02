"""Core coverage (#1129, toward 1.0) for the validation helpers not yet called by name: rollout invariants and
spectral drift, the analytical and solver comparisons, the PINN problem description, the physics-learning
verifier's checks, the gallery's conclusion, the trust card of a field and the small value objects.

References:
* mass and energy of a known field; a field that is exactly the reference has zero spectral bias;
* a model equal to the analytical solution has zero error, and one scaled by 1.1 has relative L2 error exactly 0.1;
* autograd derivatives of a polynomial; the reference model of the physics-learning case passes every check;
* an amplifying stepper is caught by the rollout invariants (gallery conclusion).
"""
import json
import warnings

import numpy as np
import pytest

from pinneapple_analysis.trust import field_card as FC
from pinneapple_analysis.validation import analytical as AN
from pinneapple_analysis.validation import core as CO
from pinneapple_analysis.validation import field_metrics as FM
from pinneapple_analysis.validation import gallery as GA
from pinneapple_analysis.validation import metamorphic as MM
from pinneapple_analysis.validation import physics_learning as PL
from pinneapple_analysis.validation import physics_learning_cases as PLC
from pinneapple_analysis.validation import pinn as PI
from pinneapple_analysis.validation import report_artifact as RA
from pinneapple_analysis.validation import rollout as RO
from pinneapple_analysis.validation import schema as SC
from pinneapple_analysis.validation import validate as VA
from pinneapple_analysis.validation import validator as VR

X = np.linspace(0, 2 * np.pi, 32, endpoint=False)


# ------------------------------------------------------------------------------------------------ rollout
def test_invariant_series_and_spectral_drift():
    u = np.array([(1 + 0.1 * k) * (1.0 + np.sin(X))[:, None] * np.ones(8)[None, :] for k in range(4)])
    s = RO.invariant_series(u, 2)
    assert np.allclose(s["mass"], [1.0 + 0.1 * k for k in range(4)])                       # mean of 1 + sin
    assert np.allclose(s["energy"], [0.5 * (1 + 0.1 * k) ** 2 * 1.5 for k in range(4)])  # 1/2 mean (1 + sin)^2
    ref = np.array([np.sin(3 * X)[:, None] * np.cos(5 * X)[None, :] + np.sin(X)[:, None] for _ in range(3)])
    d = RO.spectral_drift(ref, 2, ref)
    assert np.allclose(d["hf_bias"], 0.0, atol=1e-12) and d["band"][0] >= 1                # exact copy: no bias
    with pytest.raises(ValueError):
        RO.spectral_drift(ref, 2)
    rep = RO.RolloutReport(checks=[{"id": "a", "status": "pass"}, {"id": "b", "status": "warn"}])
    assert rep.passed and rep["b"]["status"] == "warn"
    assert not RO.RolloutReport(checks=[{"id": "a", "status": "fail"}]).passed


# ------------------------------------------------------------------------------------------------ analytical
def test_compare_to_analytical_and_to_a_solver():
    exact = lambda P: np.sin(P[:, :1]) * P[:, 1:2]                                        # noqa: E731
    r0 = AN.compare_to_analytical(exact, exact, ["x", "y"], {"x": (0, 1), "y": (0, 1)}, n_points=500)
    assert r0["rel_l2"] == 0.0 and r0["max_error"] == 0.0 and r0["n_points"] == 500
    r1 = AN.compare_to_analytical(lambda P: 1.1 * exact(P), exact, ["x", "y"], {"x": (0, 1), "y": (0, 1)},
                                  n_points=500)
    assert r1["rel_l2"] == pytest.approx(0.1, rel=1e-6)                                   # 10 % scaling (float32)
    torch = pytest.importorskip("torch")

    class Twice(torch.nn.Module):
        def forward(self, x):
            return torch.cat([2 * x[:, :1], 3 * x[:, 1:2]], dim=1)
    xs, ys = np.linspace(0, 1, 7), np.linspace(1, 2, 7)
    sv = AN.validate_against_solver(Twice(), {"x": xs, "y": ys, "u": 2 * xs, "v": 3 * ys}, ["x", "y"], ["u", "v"])
    flat = json.dumps(sv)
    assert "rel_l2" in flat or "rmse" in flat
    vals = [v for k, v in _walk(sv) if k in ("rel_l2", "rmse", "max_error")]
    assert vals and max(vals) < 1e-6                                                       # solver data reproduced


def _walk(d, out=None):
    out = [] if out is None else out
    for k, v in (d.items() if isinstance(d, dict) else []):
        if isinstance(v, dict):
            _walk(v, out)
        elif isinstance(v, (int, float)):
            out.append((k, float(v)))
    return out


# ------------------------------------------------------------------------------------------------ PINN problem
def test_pinn_problem_mesh_uniform_and_derivatives():
    torch = pytest.importorskip("torch")
    p = PI.PINNProblem(name="p", bounds=[(0.0, 1.0), (-1.0, 1.0)], residual=lambda u, X: u)
    assert p.dim == 2
    M, g = p.mesh((3, 5))
    assert g == (3, 5) and M.shape == (15, 2) and M[:, 1].min() == -1.0 and M[:, 0].max() == 1.0
    U = p.uniform(1000, np.random.default_rng(0))
    assert U[:, 0].min() >= 0 and U[:, 1].min() >= -1 and U[:, 1].max() <= 1
    Xt = torch.rand(10, 2, requires_grad=True)
    u = Xt[:, :1] ** 3 * Xt[:, 1:]
    assert torch.allclose(PI.d(u, Xt, 0, order=2), 6 * Xt[:, :1] * Xt[:, 1:])              # exact
    assert torch.allclose(PI.d(u, Xt, 1), Xt[:, :1] ** 3)


# ------------------------------------------------------------------------------------------------ physics learning
@pytest.fixture(scope="module")
def case():
    return PLC.Section()


def test_the_reference_model_passes_every_physics_learning_check(case):
    full = PLC.Model("reference, every parameter", lambda X, th: case.evaluate(th, X),
                     inputs=("P", "x_s", "k_tim", "h", "R_c", "k_die", "k_lid", "sigma"))
    Xp = PL.random_points(case, 16, seed=0)
    assert Xp.shape == (16, 2) and (Xp[:, 0] <= PLC.WIDTH).all() and (Xp[:, 1] <= case.height).all()
    s = PL.structural_checks(full, case, {}, Xp)
    assert s and all(f.passed for f in s)
    iv = PL.intervention_checks(full, case, {}, Xp)
    assert iv and all(f.passed for f in iv)                                                # same response, exactly
    f, missing = PL.sufficiency(full, case, {}, Xp)
    assert f.passed and missing == []


def test_training_distribution_ties_cooling_to_power():
    th = PLC.training_distribution(20, np.random.default_rng(0))
    assert all(t["h"] == pytest.approx(1e2 * t["P"]) for t in th)                          # one product line
    free = PLC.training_distribution(20, np.random.default_rng(0), correlated=False)
    assert not all(t["h"] == pytest.approx(1e2 * t["P"]) for t in free)


def test_surrogate_model_and_the_detection_study(case):
    pytest.importorskip("sklearn")
    sm = PLC.surrogate_model(case, n_train=12)
    th = PLC.training_distribution(1, np.random.default_rng(5))[0]
    Xp = PL.random_points(case, 30, seed=1)
    ref = case.evaluate({**PLC.DEFAULT, **th}, Xp)
    pred = sm(Xp, {**PLC.DEFAULT, **th})
    assert np.linalg.norm(pred - ref) / np.linalg.norm(ref) < 0.05                         # on its training law
    d = PL.detection_study(severities=(1.0,), budgets=(4,))
    assert d and json.dumps(d, default=float)


# ------------------------------------------------------------------------------------------------ gallery
def test_gallery_conclusion_catches_an_amplifying_stepper(tmp_path):
    import pinneapple as pp
    Y = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    U = np.array([np.sin(X - 0.1 * k)[:, None] * np.cos(Y)[None, :] for k in range(30)])

    def good(u):
        return U[min(int(np.argmin([np.abs(u - r).sum() for r in U])) + 1, len(U) - 1)]

    def bad(u):
        return 1.05 * good(u)
    problem = {"rollout": {"ndim": 2, "dt": 1.0, "periodic": True, "steps": len(U) - 1}, "name": "travelling wave"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = pp.validate(good, problem, data={"u0": U[0], "reference": U}, level="standard")
        b = pp.validate(bad, problem, data={"u0": U[0], "reference": U}, level="standard")
    res = GA.conclude("t", g, b, expect="rollout.invariants", family="rom", physics="incompressible-flow",
                      out_dir=tmp_path)
    assert res.caught == ["rollout.invariants"] and all((tmp_path / f).is_file() for f in
                                                         ("report.json", "trust_card.json", "report.sarif"))
    with pytest.raises(AssertionError):
        GA.conclude("t", g, g, expect="rollout.invariants", family="rom", physics="incompressible-flow",
                    out_dir=tmp_path)
    with pytest.raises(ValueError):
        GA.conclude("t", g, b, expect="x", family="nope", physics="incompressible-flow", out_dir=tmp_path)
    assert RA.as_dict(str(tmp_path / "report.json")) == json.loads((tmp_path / "report.json").read_text())
    assert RA.as_dict({"a": 1}) == {"a": 1}


# ------------------------------------------------------------------------------------------------ small objects
def test_field_trust_card_and_value_objects():
    ok = FC.FieldCheck("residual", "physics", True, 1e-4, 1e-3)
    bad = FC.FieldCheck("bc", "physics", False, 0.2, 0.01, critical=True, detail="inlet")
    na = FC.FieldCheck("ood", "generalization", None, detail="no data")
    card = FC.FieldTrustCard([ok, bad, na], decision="REJECT", reasons=["bc"], name="demo")
    assert card.failed == [bad] and card.to_dict()["checks"][1]["critical"]
    assert bad.line() == "[FAIL] bc: 0.2 (limit 0.01) -- inlet" and na.line().startswith("[N/A ] ood")
    assert "| bc (critical) | physics | FAIL |" in card.to_markdown()
    c = CO.CheckResult("mass", True, 1e-9, 1e-6, "conserved")
    assert str(c).startswith("[PASS] mass")
    r = FM.FieldErrorReport(1.0, 0.1, 2.0, 0.5, 0.2, 0.1, 0.2, 0.3, (4, 8))
    assert r.to_dict()["bands"] == (4, 8) and np.isnan(r.to_dict()["pearson"])
    spec = SC.CheckSpec("rollout.finite", "rollout", "finite rollout")
    assert spec.critical and spec.better == "lower" and spec.strength == 3
    with pytest.raises(NotImplementedError):
        MM.Relation("r").violations(lambda x: x, {})


def test_field_model_adapters_and_the_quick_validator():
    torch = pytest.importorskip("torch")
    f = VA.as_field_model(lambda a: 2 * a)
    assert np.allclose(f(np.ones((2, 4, 4))), 2.0)

    class Triple(torch.nn.Module):
        def forward(self, x):
            return 3 * x
    assert np.allclose(VA.as_field_model(Triple())(np.ones((2, 4, 4))), 3.0)
    rep = VR.validate_model(torch.nn.Linear(2, 1), None, coord_names=["x", "y"],
                            domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    assert type(rep).__name__ == "ValidationReport" and rep.checks == []                  # no equations given
