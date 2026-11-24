"""``run_study``: baseline -> DOE -> surrogate -> NSGA-II -> verification, on the three example studies and on toy
evaluators that exercise one mechanism each.

The numbers asserted here are the ones the runs measure: the baseline outputs, the held-out error of the chosen
surrogate family, how many designs were re-measured, and the reasons attached to each rejected design. Nothing is
asserted about a design the study did not evaluate.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pinneapple_analysis.design_study import (
    Base,
    Budget,
    DesignStudy,
    Evaluator,
    Limit,
    Objective,
    Quantity,
    Variable,
    run_study,
)
from pinneapple_analysis.design_study.examples import (
    example,
    heatsink_model,
    heatsink_study,
    polar_model,
    polar_study,
    wing_study,
    wing_vlm,
)


def toy(kind="optimize", budget=None, constraints=(), objectives=None):
    spec = DesignStudy.create(
        kind=kind, title="toy bowl",
        base=Base(kind="geometry", ref="pp.geometries:box",
                  variables=[Variable("a", -1.0, 1.0, unit="1"), Variable("b", 0.0, 1.0, unit="1")],
                  baseline={"a": 0.0, "b": 0.5}),
        evaluator=Evaluator(tool="toy.bowl", inputs={"a": "a", "b": "b"},
                            outputs=[Quantity("f", "1"), Quantity("g", "s")]),
        objectives=list(objectives or [Objective("f", "min")]), constraints=list(constraints),
        budget=budget or Budget(max_evaluations=10, n_doe=5, n_verify=2, seed=1))
    return spec


def bowl(p):
    """A convex bowl in ``a`` and a linear ``g``: the front is the ``a`` that minimises ``g * (1 + f)``."""
    a, b = float(p["a"]), float(p["b"])
    return {"f": (a - 0.25) ** 2 + 0.5 * b, "g": 1.0 + b}


# ---------------------------------------------------------------------------------------------- an example, end to end
def test_the_wing_study_measures_and_signs_a_front():
    run = run_study(wing_study(), wing_vlm)
    assert run.status == "done" and run.trust["decision"] == "APPROVED" and run.approved
    assert run.spec.evaluator.tool == "vlm.wing_planform"

    # the baseline, exactly as the model returns it at AR 7, taper 0.5, alpha 5 deg, alpha0 -2 deg
    e = 0.95 - 0.6 * (0.5 - 0.45) ** 2
    cl = (2 * math.pi * 7.0 / 9.0) * math.radians(7.0)
    assert run.baseline["cl"] == pytest.approx(cl, rel=1e-12) == pytest.approx(0.5971, abs=5e-4)
    assert run.baseline["cd"] == pytest.approx(0.008 + cl ** 2 / (math.pi * e * 7.0), rel=1e-12)
    assert run.baseline["mass_kg"] == pytest.approx(12 * 30 * math.sqrt(math.sqrt(7 * 30) / 10), rel=1e-12)

    # every output the study predicts was fitted out of sample, well inside the 10 % target
    assert set(run.surrogate["outputs"]) == {"cl", "cd", "mass_kg"}
    assert all(v["nrmse"] <= 0.10 for v in run.surrogate["outputs"].values())
    assert run.trust["checks"][0]["name"] == "surrogate held-out nRMSE of cd <= 0.1"

    # the limits as written, resolved against the measured baseline
    resolved = {c["output"]: c for c in run.constraints}
    assert resolved["cl"]["lower"] == 0.55 and not resolved["cl"]["relative"]
    assert resolved["mass_kg"]["upper"] == pytest.approx(1.05 * run.baseline["mass_kg"])
    assert resolved["mass_kg"]["relative"]

    # the answer: measured designs first, and the pair of objectives is respected
    measured = [d for d in run.pareto if d["verified"]]
    assert measured, "the front carried no re-measured design"
    for d in measured:
        assert d["outputs"]["cl"] >= 0.55
        assert d["outputs"]["mass_kg"] <= 1.05 * run.baseline["mass_kg"] + 1e-9
        assert d["predicted"] is None
    # a predicted design can only beat a measured one by less than the surrogate's own held-out error: the front
    # is denser on the surrogate, and that is why the verified members are the answer
    nrmse = run.surrogate["outputs"]
    for p in [d for d in run.pareto if d["predicted"] is not None]:
        best = min(m["objectives"]["cd"] for m in measured)
        assert p["objectives"]["cd"] >= best * (1 - 3 * nrmse["cd"]["nrmse"])

    # the rejected designs carry the limit they violate, with the numbers
    sources = {r["source"] for r in run.rejected}
    assert "constraint" in sources
    violation = next(r for r in run.rejected if "violated by" in r["reasons"][0])
    assert violation["params"] and "lift coefficient >= 0.55" in violation["reasons"][0] \
        or "mass <=" in violation["reasons"][0]
    # the domain says where the surrogate had evidence
    assert run.domain["n_simulated"] >= 3 and run.domain["spacing"] > 0
    assert all(d["inside"] for d in run.domain["verified"])

    # the budget is respected and reported
    assert run.budget["used"] == len(run.rows) <= wing_study().budget.max_evaluations
    assert run.budget["planned"]["doe"] == 10 and run.budget["planned"]["verify"] == 3
    assert [r["tag"] for r in run.rows].count("baseline") == 1


def test_the_heatsink_study_stays_inside_the_pressure_budget_it_was_given():
    spec = heatsink_study()
    run = run_study(spec, heatsink_model)
    assert run.status == "done" and run.trust["decision"] == "APPROVED"
    # 1 baseline + 32 DOE + 3 verifications, all inside the 38 evaluations the budget allows
    assert len(run.rows) == 36 and run.budget["used"] == 36
    assert run.baseline["r_th_k_w"] == pytest.approx(0.9991, abs=5e-4)
    assert run.baseline["pressure_drop_pa"] == pytest.approx(1.2264, abs=5e-4)

    verified = [r for r in run.rows if r["tag"] == "verification"]
    assert verified and all(r["ok"] for r in verified)
    assert all(r["outputs"]["pressure_drop_pa"] <= 10.0 for r in verified)
    # the prediction was recorded before the run, and the two agree inside the tolerance
    assert all(r["predicted"] is not None for r in verified)
    worst = max(abs(r["predicted"]["pressure_drop_pa"] - r["outputs"]["pressure_drop_pa"])
                / r["outputs"]["pressure_drop_pa"] for r in verified)
    assert worst < 0.10
    # the tightening of the limits the surrogate searched against is stated, with the number it used
    assert any("held-out error" in n for n in run.notes)
    assert any(r["source"] == "domain" for r in run.rejected)


def test_the_polar_is_a_surrogate_study_that_measures_and_fits_without_optimising():
    spec = polar_study()
    run = run_study(spec, polar_model)
    assert run.status == "done" and run.trust["decision"] == "APPROVED"
    assert run.optimizer == {}                                     # no NSGA-II ran
    assert all(d["predicted"] is None for d in run.pareto)         # the front is the measured one
    assert not run.pareto or all(d["verified"] is False for d in run.pareto)
    assert all(d["outputs"]["cl"] > 0 and d["outputs"]["cd"] > 0 for d in run.pareto)
    assert set(run.surrogate["outputs"]) == {"cl", "cd"}
    assert all(v["nrmse"] <= 0.10 for v in run.surrogate["outputs"].values())
    assert any(c["status"] == "not_run" and "does not optimise" in c["reason"] for c in run.trust["checks"])
    assert len(run.rows) == 31                                     # 1 baseline + 30 DOE, no verification


def test_the_run_is_reproducible_and_serialises():
    a = run_study(toy(), bowl)
    b = run_study(toy(), bowl)
    drop = lambda s: {k: v for k, v in s.items() if k != "wall_s"}  # noqa: E731
    assert drop(a.summary()) == drop(b.summary())
    assert [d["params"] for d in a.pareto] == [d["params"] for d in b.pareto]
    text = json.dumps(a.to_dict())                                 # JSON-ready, no numpy scalars left over
    assert json.loads(text)["summary"]["status"] == a.status
    page = a.to_markdown()
    assert page.startswith("# Design study: toy bowl") and "## Pareto front" in page
    assert "## Trust checks" in page


# ---------------------------------------------------------------------------------------------- the evaluator's shapes
class _Outcome:
    def __init__(self, outputs, ok=True, message="", cost_s=0.0):
        self.outputs, self.ok, self.message, self.cost_s, self.fields = outputs, ok, message, cost_s, None


class _Simulator:
    """The autolab Simulator shape, without importing it: check() then run(params, workdir)."""

    def __init__(self):
        self.workdirs = []

    def check(self, params):
        return ["a = 0.0 is the wall of the fixture"] if params["a"] == 0.0 else []

    def run(self, params, workdir):
        self.workdirs.append(workdir)
        return _Outcome(bowl(params), True, "", 0.01)


def test_a_simulator_shaped_evaluator_is_checked_before_it_is_run():
    """The baseline is at a = 0, which this simulator refuses *before* running: nothing is executed for it."""
    sim = _Simulator()
    refused_run = run_study(toy(), sim)
    assert refused_run.status == "refused" and sim.workdirs == []
    assert "wall of the fixture" in refused_run.reasons[0]

    class Later(_Simulator):
        def check(self, params):
            return [f"b = {params['b']:.3f} is below the fixture"] if params["b"] < 0.1 else []

    later = Later()
    run = run_study(toy(), later)
    assert later.workdirs and all(w for w in later.workdirs)
    dropped = [r for r in run.rows if not r["ok"]]
    assert dropped and all("below the fixture" in r["message"] for r in dropped)
    assert any(r["source"] == "evaluator" for r in run.rejected)


def test_an_evaluator_that_raises_or_returns_nonsense_is_a_rejection_not_a_crash():
    def raises(p):
        if p["a"] > 0.5:
            raise RuntimeError(f"the solver diverged at a = {p['a']:.3f}")
        return bowl(p)

    run = run_study(toy(), raises)
    bad = [r for r in run.rejected if r["source"] == "evaluator"]
    assert bad and "RuntimeError: the solver diverged" in bad[0]["reasons"][0]

    run = run_study(toy(), lambda p: {"f": float("nan"), "g": 1.0})
    assert run.status == "refused"                      # not even the baseline could be evaluated
    assert "non-finite outputs" in run.reasons[0]

    run = run_study(toy(), lambda p: {"f": 1.0})        # a missing output is named, not filled in
    assert run.status == "refused" and "did not return ['g']" in run.reasons[0]


def test_a_progress_callback_sees_every_step():
    events = []
    two = [Objective("f", "min"), Objective("g", "min")]
    run_study(toy(objectives=two), bowl, progress=events.append)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "baseline" and "doe" in kinds and "surrogate" in kinds and "optimize" in kinds
    assert kinds.count("verification") == 2
    assert all({"done", "planned", "design"} <= set(e) for e in events)


# ---------------------------------------------------------------------------------------------- the verdict
def test_a_surrogate_that_cannot_reach_the_target_is_not_signed():
    spec = toy()
    spec = DesignStudy.create(kind=spec.kind, title=spec.title, base=spec.base, evaluator=spec.evaluator,
                              objectives=list(spec.objectives), budget=spec.budget,
                              verifier={"surrogate_target": 0.001, "min_verified": 1, "folds": 3})
    run = run_study(spec, lambda p: {"f": math.exp(20 * p["a"]) + p["b"], "g": bowl(p)["g"]})
    assert run.status == "inconclusive" and run.trust["decision"] in ("REVIEW", "REJECT")
    assert any("held-out nRMSE of f" in r for r in run.reasons)


def test_a_limit_on_an_output_the_tool_does_not_declare_never_reaches_the_run():
    with pytest.raises(ValueError, match="does not declare"):
        toy(constraints=[Limit("missing_output", upper_frac_of_baseline=1.0)])
    with pytest.raises(ValueError, match="does not declare"):
        toy(constraints=[Limit("g", upper=1.0), Limit("nope", lower=0.0)])


def test_no_design_is_called_feasible_against_a_limit_nobody_measured():
    """The lift floor is above every measured lift, so nothing can be feasible and the front is empty."""
    spec = toy(constraints=[Limit("g", lower=10.0)])
    run = run_study(spec, bowl)
    assert run.baseline["g"] < 10.0
    assert not [d for d in run.pareto if d["verified"]]
    assert all(d["outputs"]["g"] < 10.0 for d in run.pareto if d["predicted"] is None)


def test_the_examples_are_reachable_by_name():
    spec, evaluator = example("wing")
    assert spec.kind == "optimize" and evaluator is wing_vlm
    with pytest.raises(ValueError, match="unknown example"):
        example("fuselage")


def test_a_measured_design_outside_the_domain_is_never_produced_by_the_optimizer():
    run = run_study(toy(), bowl)
    for d in run.pareto:
        assert -1.0 <= d["params"]["a"] <= 1.0 and 0.0 <= d["params"]["b"] <= 1.0
    assert np.isfinite([d["objectives"]["f"] for d in run.pareto]).all()
