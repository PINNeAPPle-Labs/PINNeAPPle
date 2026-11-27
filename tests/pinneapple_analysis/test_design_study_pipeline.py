"""The design pipeline of #33: the problem in, a validated design and a report out.

The pipeline is the part that was still manual — deciding which of the designs the run measured *is* the answer, on a
rule a reader can redo, and writing the two files a project keeps. The tests read that decision as a field: the rule,
the number of candidates it beat, the design, its objective, and the fact that the reference solver ran it.

The beam is the case, because its evaluator is a numerical solver: ``answer["verified"]`` is then a claim about a
solve, not about a formula. The run is configured small (three rounds, a cheaper optimiser) so the whole file stays a
few seconds.
"""
from __future__ import annotations

import json
import os

import pytest

from pinneapple_analysis.design_study import (
    Base,
    Budget,
    DesignPipeline,
    Evaluator,
    Objective,
    Quantity,
    StopRule,
    Variable,
    run_study,
    select,
)
from pinneapple_analysis.design_study.examples import beam_fdm, beam_study


def beam(**budget):
    """The beam example with a cheaper optimiser and room for a few rounds."""
    s = beam_study()
    b = {"max_evaluations": 26, "n_doe": 20, "n_verify": 2, "surrogate_evaluations": 150, "seed": 0}
    b.update(budget)
    return type(s)(kind=s.kind, title=s.title, base=s.base, evaluator=s.evaluator, objectives=s.objectives,
                   constraints=s.constraints, verifier=s.verifier, model=s.model, statement=s.statement,
                   metadata=s.metadata, budget=b)


LOOP = StopRule(min_improvement=0.0, patience=99, max_rounds=3)


@pytest.fixture(scope="module")
def beam_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipeline")
    pipeline = DesignPipeline(beam(), beam_fdm, stop=LOOP, out_dir=d)
    return pipeline.run()


# ---------------------------------------------------------------------------------------------- the answer
def test_the_pipeline_answers_with_a_design_the_solver_confirmed(beam_run):
    r = beam_run
    assert r.approved and r.decision == "APPROVED"
    a = r.answer
    assert a and a["verified"] is True
    assert a["evaluator"]["fidelity"] == "reference"            # a solve, not a formula
    # the design satisfies every limit the study wrote, and it is lighter than the baseline it started from
    outputs = a["outputs"]
    assert outputs["sigma_vm_mpa"] <= 150.0
    assert outputs["deflection_mm"] <= 30.0
    assert outputs["mass_kg"] < r.study.baseline["mass_kg"]
    # and the number in "value" is the objective the rule used, taken from the same row
    assert a["primary"] == r.study.spec.objectives[0].output == "mass_kg"
    assert a["value"] == pytest.approx(outputs["mass_kg"])
    assert a["objectives"]["mass_kg"] == pytest.approx(outputs["mass_kg"])


def test_the_rule_that_chose_the_design_is_a_field_with_the_numbers_it_used(beam_run):
    a = beam_run.answer
    assert "best measured design on the Pareto front" in a["rule"]
    assert "satisfy every limit" in a["rule"]
    assert a["n_candidates"] >= 1 and a["n_measured_on_front"] >= a["n_candidates"]
    assert a["design"]["b_mm"] == pytest.approx(30.0, abs=1e-9)  # the thinnest, tallest-tolerated section


def test_the_answer_is_the_best_measured_member_of_the_front_and_not_a_prediction(beam_run):
    study = beam_run.study
    measured = [d for d in study.pareto if d["verified"] and d["predicted"] is None]
    expected = min(d["objectives"]["mass_kg"] for d in measured)
    assert beam_run.answer["value"] == pytest.approx(expected)
    # the design it points at is a row the reference ran, not one the front only predicts
    assert beam_run.answer["design"] in [d["params"] for d in measured]
    predicted = [d["params"] for d in study.pareto if d["predicted"] is not None]
    assert predicted, "the fixture has no predicted member: the assertion above would be vacuous"


def test_the_two_files_are_written_and_say_the_same_thing(beam_run, tmp_path):
    r = beam_run
    assert os.path.basename(r.report_path) == "report.md" and os.path.basename(r.study_path) == "study.json"
    assert os.path.isfile(r.report_path) and os.path.isfile(r.study_path)
    page = open(r.report_path, encoding="utf-8").read()
    assert page.startswith("# Design pipeline: ")
    for section in ("**The problem.**", "**Decision.** APPROVED", "## The final design", "## How it was chosen",
                    "## Trust checks"):
        assert section in page, section
    for k, v in r.answer["design"].items():                    # the design's numbers are in the page, not a summary
        assert f"| {k} | {v:.4g} |" in page
    doc = json.loads(open(r.study_path, encoding="utf-8").read())
    assert doc["approved"] is True and doc["decision"] == "APPROVED"
    assert doc["answer"]["design"] == r.answer["design"]
    assert doc["study"]["summary"]["status"] == "done"
    assert doc["study"]["loop"]["stop"]["reason"] == "max_rounds"


def test_the_report_prints_the_question_above_the_answer(tmp_path):
    s = beam()
    s = type(s)(kind=s.kind, title=s.title, base=s.base, evaluator=s.evaluator, objectives=s.objectives,
                constraints=s.constraints, verifier=s.verifier, model=s.model, metadata=s.metadata,
                budget=s.budget, statement="Quanto pesa a menor secao que aguenta a carga?")
    r = DesignPipeline(s, beam_fdm, stop=StopRule(max_rounds=1, patience=99, min_improvement=0.0),
                       out_dir=tmp_path).run()
    page = open(r.report_path, encoding="utf-8").read()
    assert "**The problem.** Quanto pesa a menor secao que aguenta a carga?" in page


# ---------------------------------------------------------------------------------------------- from the problem
def test_from_problem_builds_the_study_from_the_sentence_and_runs_it(tmp_path):
    p = DesignPipeline.from_problem(
        "Find the smallest section that keeps the deflection under 30 mm for a 2 m cantilever.",
        base=Base(kind="geometry", ref="pp.geometries:cantilever_box",
                  variables=[Variable("b_mm", 30.0, 80.0, unit="mm"), Variable("h_mm", 50.0, 120.0, unit="mm")],
                  baseline={"b_mm": 50.0, "h_mm": 80.0}),
        evaluator=Evaluator(tool="beam.euler_bernoulli_fdm", fidelity="reference",
                            inputs={"b_mm": "b_mm", "h_mm": "h_mm"},
                            conditions={"length_m": 2.0, "load_n_per_m": 2000.0, "E_gpa": 69.0,
                                        "rho_kg_m3": 2700.0, "nx": 200},
                            outputs=[Quantity("mass_kg", "kg"), Quantity("deflection_mm", "mm"),
                                     Quantity("sigma_vm_mpa", "MPa")]),
        objectives=[Objective("mass_kg", "min")],
        budget=Budget(max_evaluations=16, n_doe=12, n_verify=1, surrogate_evaluations=150, seed=0),
        stop=StopRule(max_rounds=1, patience=99, min_improvement=0.0), out_dir=tmp_path)
    assert p.spec.title.startswith("Find the smallest section")
    assert p.spec.statement.startswith("Find the smallest section")
    r = p.run(beam_fdm)
    assert r.answer["verified"] and os.path.isfile(r.report_path)

    with pytest.raises(ValueError, match="statement"):
        DesignPipeline.from_problem("   ", base=p.spec.base, evaluator=p.spec.evaluator,
                                    objectives=list(p.spec.objectives))


def test_a_pipeline_without_an_evaluator_says_so_before_running_anything():
    with pytest.raises(ValueError, match="needs the evaluator"):
        DesignPipeline(beam(), None).run()


# ---------------------------------------------------------------------------------------------- when there is no answer
def test_no_measured_design_means_no_answer_and_the_reason(tmp_path):
    """A study whose limits nothing can satisfy measures designs and validates none of them as an answer."""
    from pinneapple_analysis.design_study import Limit

    s = beam()
    s = type(s)(kind=s.kind, title=s.title, base=s.base, evaluator=s.evaluator, objectives=s.objectives,
                model=s.model, metadata=s.metadata, budget=s.budget, statement=s.statement,
                constraints=[Limit("deflection_mm", upper=0.01)])          # nothing deflects less than 10 microns
    r = DesignPipeline(s, beam_fdm, stop=StopRule(max_rounds=1, patience=99, min_improvement=0.0),
                       out_dir=tmp_path).run()
    assert r.answer is None or r.answer["design"] is None
    assert r.answer["verified"] is False and r.answer["why"]
    assert not r.approved
    page = open(r.report_path, encoding="utf-8").read()
    assert "## No validated design" in page


def test_select_says_why_when_the_run_measured_nothing():
    """A refused study has no front at all: the selection returns None and the caller's reason names the refusal."""
    spec = beam()
    run = run_study(spec, lambda p: {"ok": False, "message": "the solver refuses every design in this fixture"})
    assert run.status == "refused" and select(run) is None
