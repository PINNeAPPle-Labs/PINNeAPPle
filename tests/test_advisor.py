"""The pipeline advisor: what it reads, what it decides, and that the diagram is the plan.

Nothing here needs a model or a network: the language-model path is exercised with a stub callable, and the
offline path is the default. The one test that imports the library's recommendation modules is the tool catalog
check -- it is what keeps a suggestion from naming a function that does not exist.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from pinneapple_advisor import advise, answer, brief, next_questions, questions
from pinneapple_advisor import capabilities as cap
from pinneapple_advisor.brief import Brief, BriefError
from pinneapple_advisor.compose import build
from pinneapple_advisor.estimate import CostModel

# ---------------------------------------------------------------------------------------------------- reading text

def test_rules_read_what_the_sentence_states():
    b = brief("transient heat in a 3D bracket, I have the STEP but no mesh, 200 simulations would be fine", llm=False)
    assert b.values["dimension"] == 3
    assert b.values["time_dependent"] is True
    assert b.values["geometry_present"] is True
    assert b.values["geometry_formats"] == ["step"]
    assert b.values.get("mesh_present") is None        # "no mesh" is not a ready mesh
    assert b.values["n_simulations_present"] == 200
    assert b.values["physics_family"] == "heat_equation"


def test_portuguese_reading_and_percent_target():
    b = brief("escoamento laminar 2D em um canal, erro de 2 % aceitável, tenho a malha", llm=False)
    assert b.values["dimension"] == 2
    assert b.values["physics_family"] == "navier_stokes"
    assert b.values["mesh_present"] is True
    assert b.values["accuracy_target"] == pytest.approx(0.02)


def test_unknowns_are_the_open_questions_in_asking_order():
    b = brief("Laplace in a 2D plate", llm=False)
    ids = [q["id"] for q in next_questions(b)]
    assert ids[0] == "time_dependent"                  # physics_family and dimension are already known
    assert "dimension" not in ids
    assert ids.index("geometry_present") < ids.index("budget_hours")
    assert {q["id"] for q in questions()} >= {"physics_family", "dimension", "dataset_present", "goal"}


def test_an_answer_is_read_against_its_closed_set():
    assert answer("dimension", "3 dimensões", llm=False) == "3"
    assert answer("physics_family", "condução de calor", llm=False) == "heat_equation"
    assert answer("geometry_present", "sim, tenho o STEP", llm=False) is True
    assert answer("mesh_present", "não", llm=False) is False
    assert answer("accuracy_target", "1,5 %", llm=False) == pytest.approx(0.015)
    assert answer("approx_cells", "120k células", llm=False) == 120000


def test_a_model_fills_only_what_the_rules_left_open_and_is_validated():
    calls = []

    def fake_call(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps({"fields": {"dimension": 4, "goal": "inverse", "nonlinear": "maybe",
                                      "invented_field": 1},
                           "notes": ["the bracket is aluminium"]})

    b = brief("heat in a bracket, the material is aluminium", llm=fake_call)
    assert calls, "the model should have been asked"
    assert "\"dimension\"" in calls[0]
    assert b.values.get("dimension") is None           # 4 is not a dimension: refused, not passed on
    assert b.values["goal"] == "inverse"               # the rules read no goal, so the model fills it
    refused = {r["field"] for r in b.rejected}
    assert {"dimension", "nonlinear", "invented_field"} <= refused
    assert b.notes == ["the bracket is aluminium"]


def test_a_model_that_is_down_costs_a_note_not_the_advice():
    def broken(prompt, **kwargs):
        raise ConnectionError("ollama is not running")

    b = brief("Poisson in a 2D plate", llm=broken)
    assert b.values["physics_family"] == "poisson"
    assert any("did not answer" in n for n in b.notes)


def test_bad_answers_are_refused_with_the_reason():
    with pytest.raises(BriefError):
        Brief().set("dimension", 5)
    with pytest.raises(BriefError):
        Brief().set("goal", "make it fast")
    assert Brief().try_set("cores", "many") is False


# ---------------------------------------------------------------------------------------------------- the plan

def test_three_d_with_cad_gives_geometry_solver_and_mesh():
    plan = advise("transient heat in a 3D bracket, I have the STEP", answers={"mesh_present": False}, llm=False)
    ids = [s.id for s in plan.stages]
    assert ids[:4] == ["physics", "geometry", "mesh", "solver"]
    assert plan.stage("geometry").status == "have"        # the CAD is the user's
    assert plan.stage("mesh").status == "generate"        # the mesh is not
    assert plan.stage("mesh").depends_on == ["geometry"]


def test_two_d_analytic_needs_no_geometry_and_no_mesh():
    plan = advise("Laplace in a rectangular plate", answers={"dimension": 2}, llm=False)
    ids = [s.id for s in plan.stages]
    assert "geometry" not in ids and "mesh" not in ids
    assert plan.stage("solver").depends_on == ["physics"]


def test_a_dataset_in_hand_removes_the_campaign_and_keeps_the_training():
    plan = advise("linear elasticity on a 2D beam", answers={"dataset_present": True, "n_simulations_present": 300},
                  llm=False)
    dataset = plan.stage("dataset")
    assert dataset.status == "have"
    assert all(s.id != "dataset" or s.estimate is None for s in plan.stages if s.id == "dataset")
    assert plan.stage("training").status == "generate"
    assert plan.stage("training").depends_on == ["dataset"]


def test_cheap_direct_study_asks_for_no_surrogate():
    plan = advise("Laplace in a plate", answers={"approx_cells": 2000, "solve_seconds_measured": 0.4,
                                                 "evaluations_needed": 5}, llm=False)
    assert all(s.id not in ("dataset", "training") for s in plan.stages)
    decision = {d.topic: d.choice for d in plan.decisions}
    assert decision["architecture"] == "none"
    assert plan.total_estimate().confidence == "measured"


def test_expensive_study_builds_a_dataset_and_trains():
    plan = advise("transient heat in a 2D plate", answers={"evaluations_needed": 500, "parameter_varies": True,
                                                           "output_kind": "grid_field", "n_parameters": 2},
                  llm=False)
    ids = [s.id for s in plan.stages]
    assert ids.index("dataset") < ids.index("training")
    assert plan.stage("dataset").status == "generate"
    decision = {d.topic: d.choice for d in plan.decisions}
    assert decision["architecture"] in ("fno", "deeponet", "pino", "mesh_graph_net", "pinn", "xpinn")
    assert plan.provenance["n_samples"] >= 50


def test_the_architecture_follows_the_nature_of_the_problem():
    fixed = advise("steady heat in one fixed 2D plate", answers={"evaluations_needed": 400}, llm=False)
    family = advise("heat in plates whose geometry varies across the family", answers={
        "evaluations_needed": 400, "geometry_varies": True, "output_kind": "mesh_field"}, llm=False)
    assert fixed.provenance["architecture"], "an architecture is always named"
    assert family.provenance["architecture"][0] in ("mesh_graph_net", "fno", "deeponet", "pino", "pinn", "xpinn")
    inverse = advise("recover the conductivity from measurements", answers={"goal": "inverse"}, llm=False)
    assert "inverse" in [s.id for s in inverse.stages]
    assert inverse.stage("inverse").depends_on == ["solver"]


def test_the_default_optimisation_adds_the_search_stage():
    plan = advise("optimise the thickness of a 2D fin",
                  answers={"goal": "optimization", "evaluations_needed": 400}, llm=False)
    assert "optimization" in [s.id for s in plan.stages]
    assert plan.stage("optimization").status == "generate"


def test_every_stage_answers_one_of_the_ten_questions_and_is_checked():
    plan = advise("convective heat transfer in a 3D duct", answers={"turbulent": True}, llm=False)
    for stage in plan.stages:
        assert stage.answers and all(1 <= a <= 10 for a in stage.answers), stage.id
        assert stage.checks or stage.kind in ("geometry",), stage.id
        assert stage.tools or stage.status == "skip", stage.id


def test_the_dependency_graph_is_acyclic_and_points_at_real_stages():
    plan = advise("transient heat in a 3D bracket, 300 simulations", answers={"evaluations_needed": 300},
                  llm=False)
    ids = {s.id for s in plan.stages}
    assert all(dep in ids for s in plan.stages for dep in s.depends_on)
    order = [s.id for s in plan.stages]
    assert all(order.index(dep) < order.index(s.id) for s in plan.stages for dep in s.depends_on)


# ---------------------------------------------------------------------------------------------------- the cost

def test_the_cost_model_shows_its_formula_and_grows_the_right_way():
    model = CostModel()
    small = model.solve(family="heat_equation", dim=2, cells=10_000)
    large = model.solve(family="heat_equation", dim=3, cells=500_000)
    assert large.seconds_per_evaluation > small.seconds_per_evaluation
    transient = model.solve(family="heat_equation", dim=2, cells=10_000, time_dependent=True)
    assert transient.seconds_per_evaluation > small.seconds_per_evaluation
    assert "cells" in small.basis and small.confidence == "order of magnitude"
    assert small.assumptions, "an estimate without its assumptions is a number without a source"
    assert small.cost_usd is None                     # a local machine costs no equipment money by default


def test_a_measured_solve_replaces_the_formula():
    est = CostModel().solve(family="heat_equation", dim=3, measured_solve_seconds=42.0, n_evaluations=10)
    assert est.confidence == "measured"
    assert est.seconds_per_evaluation == 42.0
    assert est.total_seconds == 420.0


def test_calibration_reproduces_the_measurement():
    model = CostModel()
    before = model.solve(family="navier_stokes", dim=3, cells=100_000).seconds_per_evaluation
    calibrated = model.calibrate(before * 3, cells=100_000, family="navier_stokes")
    after = calibrated.solve(family="navier_stokes", dim=3, cells=100_000).seconds_per_evaluation
    assert after == pytest.approx(before * 3, rel=1e-6)


def test_training_cost_scales_with_samples_and_falls_on_a_gpu():
    model = CostModel()
    small = model.training("fno", n_samples=50)
    large = model.training("fno", n_samples=200)
    assert large.total_seconds > small.total_seconds
    assert model.training("fno", n_samples=50, gpu=True).total_seconds < small.total_seconds
    assert model.training("pinn", n_samples=0).total_seconds > 0        # physics-only: no dataset needed


# ---------------------------------------------------------------------------------------------------- the outputs

def test_the_diagram_is_a_rendering_of_the_plan():
    plan = advise("transient heat in a 3D bracket with a CAD", llm=False)
    mermaid = plan.diagram("mermaid")
    assert mermaid.startswith("flowchart TD")
    for stage in plan.stages:
        assert f'{stage.id}["' in mermaid
        assert f":::{stage.status}" in mermaid
    for stage in plan.stages:
        for dep in stage.depends_on:
            assert f"  {dep} --> {stage.id}" in mermaid
    ET.fromstring(plan.diagram("svg"))
    assert plan.diagram("dot").startswith("digraph")
    page = plan.diagram("html")
    assert page.startswith("<!doctype html>") and "<svg" in page and page.count("<section") == len(plan.stages)


def test_the_json_is_the_contract():
    plan = advise("Laplace in a plate", llm=False)
    data = json.loads(plan.json())
    assert data["schema"] == "pinneapple.advisor/1"
    assert set(data) >= {"problem", "brief", "stages", "edges", "decisions", "open_questions", "warnings",
                         "estimate", "provenance"}
    stage = data["stages"][0]
    assert set(stage) >= {"id", "kind", "title", "why", "status", "tools", "checks", "views", "depends_on",
                          "estimate", "answers"}
    assert isinstance(stage["why"], dict) and {"pt", "en"} <= set(stage["why"])
    json.dumps(data)                                   # nothing unserialisable got in


def test_write_puts_the_plan_and_the_diagrams_on_disk(tmp_path):
    plan = advise("Laplace in a plate", llm=False)
    written = plan.write(str(tmp_path), diagrams=("mermaid", "svg", "html", "markdown"))
    names = sorted(p.rsplit("/", 1)[-1] for p in written)
    assert names == ["pipeline.html", "pipeline.md", "pipeline.mermaid", "pipeline.svg", "plan.json"]
    assert json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))["schema"] == "pinneapple.advisor/1"


# ---------------------------------------------------------------------------------------------------- the catalog

def test_every_tool_a_plan_cites_exists_in_the_library():
    plans = [
        advise("transient heat in a 3D bracket", answers={"evaluations_needed": 300, "goal": "inverse"}, llm=False),
        advise("optimise a 2D fin", answers={"goal": "optimization"}, llm=False),
        advise("linear elasticity with a mesh in hand", answers={"mesh_present": True, "dataset_present": True,
                                                                 "n_simulations_present": 100}, llm=False),
    ]
    tools = [t for plan in plans for s in plan.stages for t in s.tools]
    assert tools
    problems = cap.verify_tools(tools)
    assert problems == [], problems


# ---------------------------------------------------------------------------------------------------- the CLI

def test_the_cli_writes_the_files(tmp_path):
    from pinneapple_advisor.cli import main

    code = main(["--text", "transient heat in a 3D bracket", "--offline", "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "plan.json").exists()
    assert (tmp_path / "pipeline.svg").exists()


def test_the_cli_lists_the_questions(capsys):
    from pinneapple_advisor.cli import main

    assert main(["--questions"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {q["id"] for q in out} >= {"dimension", "goal", "dataset_present"}


def test_build_is_the_same_plan_as_advise():
    text = "conduction in a 2D plate with a heat source"
    a = advise(text, llm=False).to_dict()
    b = build(brief(text, llm=False)).plan.to_dict()
    assert [s["id"] for s in a["stages"]] == [s["id"] for s in b["stages"]]


# ---------------------------------------------------------------------------------- the scenarios, end to end

SCENARIOS = {
    "3d_cad_surrogate": ("transient heat in a 3-D bracket, I have the STEP but no mesh",
                         {"evaluations_needed": 300},
                         ["physics", "geometry", "mesh", "solver", "dataset", "training", "verification",
                          "visualization"]),
    "2d_analytic_direct": ("Laplace in a rectangular plate with 20k cells",
                           {"approx_cells": 20_000},
                           ["physics", "solver", "verification", "visualization"]),
    "dataset_in_hand": ("linear elasticity on a 2D beam, I have a dataset with 300 simulations",
                        {"dataset_present": True, "n_simulations_present": 300},
                        ["physics", "solver", "dataset", "training", "verification", "visualization"]),
    "inverse": ("recover the thermal conductivity from temperature measurements in a 2D cylinder",
                {"goal": "inverse"},
                ["physics", "solver", "inverse", "verification", "visualization"]),
    "optimise": ("optimise the thickness of a 2D fin for the lowest mass",
                 {"goal": "optimization", "evaluations_needed": 400},
                 ["physics", "solver", "dataset", "training", "optimization", "verification", "visualization"]),
    "mesh_in_hand": ("turbulent flow in a 3-D duct, I have the mesh",
                     {"mesh_present": True, "turbulent": True},
                     ["physics", "mesh", "solver", "verification", "visualization"]),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_each_scenario_gives_the_stages_it_needs(name):
    text, answers, expected = SCENARIOS[name]
    plan = advise(text, answers=answers, llm=False)
    assert [s.id for s in plan.stages] == expected
    # every plan is a DAG whose edges point at stages that exist, and every stage says what it costs or why not
    order = {s.id: i for i, s in enumerate(plan.stages)}
    for stage in plan.stages:
        assert all(dep in order and order[dep] < order[stage.id] for dep in stage.depends_on), stage.id
        assert stage.answers and all(1 <= a <= 10 for a in stage.answers), stage.id
        assert stage.tools, stage.id
        if stage.status == "generate":
            assert stage.checks, stage.id
    # the campaign estimate covers every stage that still costs work, and says what it is based on
    total = plan.total_estimate()
    assert total.total_seconds >= 0.0
    assert total.basis and total.confidence in ("measured", "order of magnitude")


def test_the_diagram_of_every_scenario_is_drawable():
    for name, (text, answers, _expected) in SCENARIOS.items():
        plan = advise(text, answers=answers, llm=False)
        for kind in ("mermaid", "dot", "svg", "html", "markdown"):
            out = plan.diagram(kind)
            assert out.strip(), (name, kind)
        root = ET.fromstring(plan.diagram("svg"))
        boxes = root.findall(".//{http://www.w3.org/2000/svg}rect")
        assert len(boxes) >= len(plan.stages), name


def test_the_training_waits_for_the_dataset_the_plan_builds():
    """A PINN that is fed the simulations the plan generates must depend on that stage, or the drawing lies."""
    plan = advise("transient heat in a 3-D bracket, I have the STEP but no mesh",
                  answers={"evaluations_needed": 300}, llm=False)
    assert plan.stage("dataset").status == "generate"
    assert plan.stage("training").depends_on == ["dataset"]
    assert plan.stage("solver").depends_on == ["physics", "mesh"]
    # and the stage that runs the solver says it is one run, because the campaign is the dataset stage
    assert "one run to calibrate" in plan.stage("solver").estimate.basis


def test_a_narrow_plan_is_still_wide_enough_for_its_own_header():
    """One stage per row gives a narrow drawing; the subtitle must not run off the edge of it."""
    from pinneapple_advisor import render

    plan = advise("optimise the thickness of a 2D fin", answers={"goal": "optimization"}, llm=False)
    svg = plan.diagram("svg")
    width = float(svg.split('width="', 1)[1].split('"', 1)[0])
    assert width >= render._MIN_W
    longest = max(len(_text_of(plan.problem, "pt")), len(render._estimate_line(plan, "pt")))
    assert longest <= render._fit_chars(width - 2 * render._MARGIN, 11.5) + 200        # elided, not clipped


def _text_of(value, lang):
    from pinneapple_advisor.schema import text_of
    return text_of(value, lang)


def test_the_flow_node_writes_the_plan_and_its_drawings(tmp_path):
    """The same plan through the flow runtime: the node writes the four files and the report node reads them."""
    import pinneapple_flow as pf
    from pinneapple_flow.nodes_ext.advisor import build_flow

    flow = build_flow("transient heat in a 3-D bracket, I have the STEP but no mesh", language="pt",
                      answers={"evaluations_needed": 300}).flow()
    result = pf.run(flow, workdir=tmp_path)
    folder = tmp_path / "engine" / "runs" / result.run_id / "advise"
    assert (folder / "plan.json").is_file()
    assert (folder / "pipeline.svg").is_file()
    assert (folder / "pipeline.html").is_file()
    assert (folder / "pipeline.mmd").is_file()
    plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == "pinneapple.advisor/1"
    assert [s["id"] for s in plan["stages"]][:3] == ["physics", "geometry", "mesh"]
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    metrics = report["metrics"]
    assert "campaign.hours" in metrics and "choice.solver" in metrics
