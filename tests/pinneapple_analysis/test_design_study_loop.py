"""The budgeted loop of #79: the stop rule as data, and a study that spends its reference budget in rounds.

Two levels are tested separately, because they fail differently. :func:`.loop.decide` is a pure function over the
rounds recorded so far, so its four numbers are tested with the numbers typed in — no surrogate, no solver, no
clock. The loop itself is tested on the beam example, whose evaluator is the finite-difference Euler-Bernoulli
solver of ``pinneapple_simulation``: a counting wrapper proves how many reference evaluations were made, and the
run's own ``loop`` block proves which number stopped it.

Every assertion below is a number the run measured; nothing is asserted about a design the study did not evaluate.
"""
from __future__ import annotations

import json

import pytest

from pinneapple_analysis.design_study import StopRule, decide, run_study
from pinneapple_analysis.design_study.examples import beam_fdm, beam_study, polar_study

# ---------------------------------------------------------------------------------------------- the rule, as data
def test_the_rule_refuses_numbers_that_do_not_mean_anything():
    assert StopRule().to_dict() == {"min_improvement": 0.01, "patience": 2, "max_rounds": 6, "per_round": 1,
                                    "decision_tol": 0.02}
    assert StopRule.from_dict(StopRule(patience=3).to_dict()) == StopRule(patience=3)
    for bad in ({"min_improvement": -0.1}, {"patience": 0}, {"max_rounds": 0}, {"per_round": 0},
                {"decision_tol": 1.0}):
        with pytest.raises(ValueError):
            StopRule.from_dict(bad)
    with pytest.raises(ValueError, match="unknown field"):
        StopRule.from_dict({"patience": 2, "tolerence": 0.1})       # a typo is refused, not dropped


ROUND = {"round": 1, "evaluations": 12, "measured": 1, "best": {"f": 1.0}, "improvement": None, "regret": None}


def _rounds(**over):
    return [dict(ROUND, round=i + 1, **over) for i in range(3)]


def test_a_spent_budget_stops_the_loop_before_anything_else_is_looked_at():
    d = decide(StopRule(patience=2), _rounds(improvement=0.5), budget_left=0)
    assert d["stop"] and d["reason"] == "budget" and d["evaluations"] == 12 and d["budget_left"] == 0
    assert "spent" in d["text"]


def test_a_round_that_measured_nothing_stops_the_loop_because_it_would_repeat_itself():
    d = decide(StopRule(patience=2), [dict(ROUND, measured=0)], budget_left=5)
    assert d["stop"] and d["reason"] == "nothing_new" and d["value"] == 0.0


def test_a_decision_grade_surrogate_stops_it_after_patience_rounds():
    rule = StopRule(patience=2, decision_tol=0.02)
    rounds = _rounds(improvement=0.5, regret=0.005)
    d = decide(rule, rounds, budget_left=5)
    assert d["stop"] and d["reason"] == "decision"
    assert d["value"] == pytest.approx(0.005) and d["limit"] == pytest.approx(0.02)
    assert decide(rule, rounds[:1], budget_left=5)["stop"] is False      # one round is not patience


def test_a_regret_above_the_tolerance_is_not_decision_grade():
    rounds = _rounds(improvement=0.5, regret=0.5)
    d = decide(StopRule(patience=2), rounds, budget_left=5)
    assert not d["stop"] and d["reason"] == "continue"


def test_an_objective_that_stopped_moving_stops_it_and_says_by_how_much():
    d = decide(StopRule(patience=2, min_improvement=0.01), _rounds(improvement=0.001), budget_left=5)
    assert d["stop"] and d["reason"] == "no_improvement"
    assert d["value"] == pytest.approx(0.001) and d["limit"] == pytest.approx(0.01)


def test_a_round_with_no_previous_incumbent_is_not_counted_as_no_improvement():
    rounds = [dict(ROUND, improvement=None), dict(ROUND, round=2, improvement=None)]
    assert decide(StopRule(patience=2), rounds, budget_left=5)["reason"] == "continue"


def test_the_round_cap_stops_it_with_the_budget_still_left():
    d = decide(StopRule(max_rounds=3), _rounds(improvement=0.5), budget_left=7)
    assert d["stop"] and d["reason"] == "max_rounds" and d["budget_left"] == 7


# ---------------------------------------------------------------------------------------------- the loop, on a solver
class Counting:
    """The beam solver behind a counter: the loop's cost is the number of calls it makes, and nothing else."""

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, params):
        self.calls += 1
        return self.fn(params)


def _beam(**budget):
    """The beam example with a cheaper optimiser, for the tests that run the loop more than once."""
    spec = beam_study()
    b = spec.budget.to_dict()
    b.update(budget)
    return type(spec)(kind=spec.kind, title=spec.title, base=spec.base, evaluator=spec.evaluator,
                      objectives=spec.objectives, constraints=spec.constraints, verifier=spec.verifier,
                      model=spec.model, statement=spec.statement, metadata=spec.metadata,
                      budget={"max_evaluations": 26, "n_doe": 20, "n_verify": 2, "surrogate_evaluations": 150,
                              "seed": 0} | budget)


def test_the_beam_is_designed_against_a_solver_and_its_answer_is_re_measured():
    solve = Counting(beam_fdm)
    run = run_study(beam_study(), solve)
    assert run.status == "done" and run.trust["decision"] == "APPROVED", run.reasons
    assert run.spec.evaluator.fidelity == "reference"            # the FDM solve, not a closed-form formula

    # the baseline is the solver's own answer at b = 50 mm, h = 80 mm: 2700 kg/m^3 over 2 m of section
    assert run.baseline["mass_kg"] == pytest.approx(2700.0 * 0.050 * 0.080 * 2.0, rel=1e-12)
    assert run.baseline["deflection_mm"] == pytest.approx(27.1746, abs=5e-3)
    assert run.baseline["sigma_vm_mpa"] == pytest.approx(75.0405, abs=5e-3)

    # one reference evaluation per row: the count is the table, not a claim
    assert solve.calls == len(run.rows) == run.budget["used"]
    assert len(run.rows) <= beam_study().budget.max_evaluations

    # the mass is fitted exactly by the response surface (it is rho A L) and the deflection to well inside 10 %
    assert run.surrogate["outputs"]["mass_kg"]["nrmse"] < 1e-4
    assert run.surrogate["outputs"]["deflection_mm"]["nrmse"] <= 0.10

    # the answer: designs the solver confirmed, inside both limits, and lighter than the baseline
    verified = [d for d in run.pareto if d["verified"]]
    assert verified
    for d in verified:
        assert d["predicted"] is None
    best_mass = min(d["outputs"]["mass_kg"] for d in verified)
    assert best_mass < run.baseline["mass_kg"]
    for r in run.rows:
        if r["tag"] == "verification":
            assert r["predicted"] is not None                       # recorded before the solver ran
            assert r["outputs"]["sigma_vm_mpa"] <= 150.0
            assert r["outputs"]["deflection_mm"] <= 30.0
    assert run.stop["reason"] == "single_pass" and run.loop["rule"] is None


def test_the_loop_spends_exactly_the_budget_it_was_given_and_stops_on_it():
    spec = _beam(max_evaluations=26)
    solve = Counting(beam_fdm)
    # patience and improvement are out of the way: only the cost can stop this loop
    run = run_study(spec, solve, stop=StopRule(min_improvement=0.0, patience=99, max_rounds=99))
    assert run.stop["reason"] == "budget" and run.stop["budget_left"] == 0
    assert solve.calls == spec.budget.max_evaluations == len(run.rows) == 26
    assert run.stop["evaluations"] == 26

    rounds = run.loop["rounds"]
    assert len(rounds) > 1, "the loop ran one round: it did not spend the budget it was given"
    assert [r["evaluations"] for r in rounds] == sorted({r["evaluations"] for r in rounds})
    # every round after the first measured a design the surrogate had not been trained on
    assert all(r["measured"] == 1 for r in rounds)
    # and the reference agreed with the recommendation: the regret is the decision layer's number, per round
    assert all(r["regret"] is not None for r in rounds[1:])

    # the front's mass came down from the baseline as the evidence grew
    best = [r["best"]["mass_kg"] for r in rounds]
    assert best[-1] < best[0] < run.baseline["mass_kg"]


def test_the_loop_stops_when_the_objective_stops_moving_instead_of_spending_the_rest():
    spec = _beam(max_evaluations=26)
    solve = Counting(beam_fdm)
    # nothing short of a 50 % gain a round counts as progress: this loop gives up early and says so
    run = run_study(spec, solve, stop=StopRule(min_improvement=0.5, patience=2, max_rounds=99))
    assert run.stop["reason"] == "no_improvement"
    assert run.stop["value"] < 0.5
    assert solve.calls < spec.budget.max_evaluations          # the budget it did not spend
    assert run.stop["budget_left"] == spec.budget.max_evaluations - solve.calls
    assert not run.notes or any("stopped on no_improvement" in n for n in run.notes)


def test_each_round_is_a_design_the_surrogate_recommended_and_the_solver_confirmed():
    spec = _beam(max_evaluations=30)
    run = run_study(spec, beam_fdm, stop=StopRule(min_improvement=0.0, patience=99, max_rounds=3))
    rounds = run.loop["rounds"]
    assert len(rounds) == 3                                    # the cap, not the budget
    assert run.stop["reason"] == "max_rounds"
    assert run.stop["budget_left"] == spec.budget.max_evaluations - len(run.rows) > 0
    verified = [r for r in run.rows if r["tag"] == "verification"]
    assert len(verified) == len(rounds)                        # one design measured per round
    for r in verified:
        assert r["predicted_std"] and all(v == v for v in r["predicted_std"].values())
        # the prediction is a number the report can compare with the measurement
        assert abs(r["predicted"]["mass_kg"] - r["outputs"]["mass_kg"]) / r["outputs"]["mass_kg"] < 0.05
    # the reported surrogate is the one that recommended the last round's design: it was fitted on every measurement
    # made *before* that round, which is what makes its prediction out of sample
    assert run.surrogate["n_train"] == len(run.measured) - 1


def test_the_rounds_and_the_stop_are_in_the_report_and_serialise():
    spec = _beam(max_evaluations=24)
    run = run_study(spec, beam_fdm, stop=StopRule(min_improvement=0.0, patience=99, max_rounds=2))
    doc = json.loads(json.dumps(run.to_dict()))
    assert doc["loop"]["rule"]["max_rounds"] == 2
    assert [r["round"] for r in doc["loop"]["rounds"]] == [1, 2]
    assert doc["summary"]["rounds"] == 2 and doc["summary"]["stop"] == run.stop["reason"]
    page = run.to_markdown()
    assert "## Budgeted loop" in page
    assert f"stopped on **{run.stop['reason']}**" in page
    assert page.count("| round |") == 1


def test_a_surrogate_study_has_one_round_because_it_has_no_recommendation_to_measure():
    spec = polar_study()
    run = run_study(spec, lambda p: {"cl": 0.5, "cd": 0.02}, stop=StopRule(max_rounds=9))
    assert run.stop["reason"] == "single_pass" and "does not optimise" in run.stop["text"]
    assert len(run.loop["rounds"]) == 1
    assert run.loop["rule"]["max_rounds"] == 9                 # the rule is reported as given, not silently changed


def test_a_rule_that_does_not_parse_stops_the_call_before_the_evaluator_is_touched():
    solve = Counting(beam_fdm)
    with pytest.raises(ValueError, match="unknown field"):
        run_study(beam_study(), solve, stop={"patience": 2, "rounds": 4})
    assert solve.calls == 0
