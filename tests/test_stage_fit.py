"""The stage map joined to the planner: a proposal now says which workflow stages it spans and which are thin.

The point of the join is the answer it gives about the *library*, not about the flow: a planned inverse problem runs
its inverse step on PINNeAPPle's own code, because exactly one external tool is wired at that stage and it is an
equation-discovery package. These tests pin that the join happens, that the thin stage is the one the map says, and
that a missing catalogue degrades to a stated reason rather than to an empty map.
"""
from __future__ import annotations

import pytest

from pinneapple_flow import stage_fit
from pinneapple_flow.planner.heuristic import plan_heuristic


def _have_map() -> bool:
    try:
        import pinneapple_catalog.stages  # noqa: F401
        return True
    except Exception:
        return False


needs_map = pytest.mark.skipif(not _have_map(), reason="pinneapple_catalog.stages is not installed")


# ---------------------------------------------------------------------------------------------- node -> stage
def test_a_curated_node_belongs_to_the_palette_section_it_is_shown_under():
    assert stage_fit.node_stage("geometry.domain") == "problem"
    assert stage_fit.node_stage("inverse.calibrate") == "inverse"
    assert stage_fit.node_stage("export.report") == "export"


def test_an_unregistered_type_has_no_stage_rather_than_a_guess():
    assert stage_fit.node_stage("not.a.node.type") is None


def test_the_stages_of_a_flow_come_back_in_workflow_order():
    nodes = [{"id": "r", "type": "export.report"}, {"id": "g", "type": "geometry.domain"},
             {"id": "c", "type": "inverse.calibrate"}, {"id": "x", "type": "not.a.node.type"}]
    assert list(stage_fit.flow_stages(nodes)) == ["problem", "inverse", "export"]


# ---------------------------------------------------------------------------------------------- the join
@needs_map
def test_the_inverse_stage_is_thin_and_says_why():
    result = stage_fit.fit([{"id": "c", "type": "inverse.calibrate"}])
    assert [s["id"] for s in result["stages"]] == ["inverse"]
    assert result["stages"][0]["wired"] == 1
    thin = result["thin"]
    assert [t["id"] for t in thin] == ["inverse"], "the map's thinnest stage stopped being reported as thin"
    assert thin[0]["gap"], "a thin stage with no gap sentence tells a reader nothing to do"
    assert thin[0]["wired_tools"] == ["pysindy"], "the one wired inverse tool changed without the map being updated"


@needs_map
def test_a_stage_with_plenty_of_wired_tools_is_not_flagged():
    result = stage_fit.fit([{"id": "s", "type": "solver.classical"}])
    assert result["stages"][0]["wired"] > stage_fit.THIN
    assert result["thin"] == []


@needs_map
def test_the_datasets_a_stage_offers_are_split_by_whether_this_library_can_read_them():
    result = stage_fit.fit([{"id": "s", "type": "solver.classical"}, {"id": "p", "type": "geometry.domain"}])
    problem = next(s for s in result["stages"] if s["id"] == "problem")
    assert problem["datasets"], "the problem stage offers no dataset at all"
    assert set(problem["readable_datasets"]) <= set(problem["datasets"])


def test_a_missing_catalogue_is_reported_as_a_reason_not_as_an_empty_map(monkeypatch):
    monkeypatch.setattr(stage_fit, "_stage_catalog", lambda: (None, "ImportError: no pinneapple_catalog"))
    result = stage_fit.fit([{"id": "c", "type": "inverse.calibrate"}])
    assert result["available"] is False
    assert "pinneapple_catalog" in result["reason"]
    assert result["stages"], "the stages the flow spans are known without the catalogue"
    assert "not checked" in stage_fit.fit_message(result)


# ---------------------------------------------------------------------------------------------- the planner
@needs_map
def test_a_planned_inverse_flow_carries_the_stage_map_and_records_the_thin_stage():
    plan = plan_heuristic("identify the conductivity of a rod from measurements")
    proposal = plan["proposals"][0]
    assert "stage_map" in proposal, "the planner stopped annotating proposals with the stage map"
    assert proposal["flow"]["meta"]["stage_map"]["stages"], "the annotation did not travel with the flow"
    inverse = next(s for s in proposal["stage_map"]["stages"] if s["id"] == "inverse")
    assert inverse["nodes"], "the inverse stage is listed without naming the nodes that sit there"

    decisions = [d for d in proposal["flow"]["meta"]["decisions"] if d["decided_by"]["id"] == "stage.coverage"]
    assert decisions, "a thin stage was not recorded as a decision, so it lives only in the proposal dict"
    assert "inverse" in decisions[0]["what"]
    assert decisions[0]["evidence"] and decisions[0]["uncited_reason"]


@needs_map
def test_a_plan_whose_stages_are_all_well_covered_records_no_coverage_decision():
    plan = plan_heuristic("solve the heat equation on a rod and compare finite differences with a PINN")
    for proposal in plan["proposals"]:
        thin = proposal["stage_map"]["thin"]
        recorded = [d for d in proposal["flow"]["meta"]["decisions"] if d["decided_by"]["id"] == "stage.coverage"]
        assert bool(recorded) == bool(thin), (proposal["title"], thin)


def test_the_rule_states_that_it_has_no_literature_reference():
    from pinneapple_flow.rules import RULES

    rule = RULES["stage.coverage"]
    assert rule.uncited, "a rule with neither a citation nor a reason fails the decisions gate"
    assert not rule.cites
