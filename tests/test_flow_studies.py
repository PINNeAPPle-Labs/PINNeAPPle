"""Kinds of study: a goal is classified into the study it asks for, and each kind has its own steps (not every study
is a design optimisation). The words, roles and checks are data in pinneapple_flow.studies."""
import pytest

from pinneapple_flow.studies import (
    STUDY_KINDS,
    STUDY_ROLES,
    classify_study,
    role_checks,
    roles_for_kind,
)

GOALS = [
    ("minimize the mass of a bracket under a 2 kN load", "design_optimization"),
    ("estimate the thermal conductivity of the wall from these temperature measurements", "inverse"),
    ("identificar a rigidez da viga a partir das medições", "inverse"),
    ("how sensitive is the peak stress to the tolerances", "uq_sensitivity"),
    ("a digital twin of the pump fed by live sensor data", "twin"),
    ("gêmeo digital da bomba com dados ao vivo", "twin"),
    ("forecast next week electricity demand from the time series", "forecast"),
    ("design a PID controller to stabilize the drone altitude", "control"),
    ("fluid-structure interaction of a flexible flap", "coupled"),
    ("fit a curve to the lab data of the spring test", "data_analysis"),
    ("a fast surrogate of the CFD for real-time use", "surrogate"),
    ("qual é a temperatura máxima da placa", "simulation"),
]


@pytest.mark.parametrize("goal,kind", GOALS)
def test_goals_are_classified_into_their_study(goal, kind):
    m = classify_study(goal)
    assert m[0].kind == kind and m[0].words


def test_nothing_recognised_is_a_simulation_and_evidence_breaks_ties():
    assert classify_study("hello")[0].kind == "simulation"
    assert classify_study("the plate", has_objective=True)[0].kind == "design_optimization"


def test_each_kind_has_its_own_steps():
    assert roles_for_kind("design_optimization") == ["geometry", "physics", "doe", "simulate", "dataset", "surrogate",
                                                     "optimize", "verify"]
    assert roles_for_kind("inverse", spatial=True, has_data=True) == ["geometry", "mesh", "dataset", "physics",
                                                                      "simulate", "identify", "uq"]
    assert roles_for_kind("control", spatial=True) == ["physics", "linearize", "control", "closed_loop", "verify"]
    assert roles_for_kind("forecast") == ["dataset", "analyze", "forecast"]   # forecast_step verifies itself
    assert roles_for_kind("surrogate", spatial=True, has_geometry=True, has_data=True) == [
        "mesh", "physics", "dataset", "surrogate", "verify"]                    # data replaces the DOE campaign
    assert roles_for_kind("simulation", is_step=True, has_geometry=True, spatial=True)[:2] == ["cad", "mesh"]


def test_every_role_is_defined_and_verification_means_something_per_kind():
    for k in STUDY_KINDS.values():
        assert set(k.roles) <= set(STUDY_ROLES) and k.roles[-1] == "verify" and k.verify_checks
        assert k.title and k.title_pt and k.words
    for r in STUDY_ROLES.values():
        assert r.title and r.title_pt and r.query and r.checks
    assert role_checks("forecast", "verify")[0][0] == "verify.beats_baseline"
    assert role_checks("inverse", "identify")[0][0] == "identify.fits_data"
    # a kind verified inside a step: that step carries the kind's checks, and its tool emits them
    assert [c for c, _ in role_checks("forecast", "forecast")] == ["forecast.horizon", "verify.beats_baseline"]
    assert role_checks("inverse", "identify")[-1][0] == "verify.held_out"
    for k in STUDY_KINDS.values():
        if k.verify_in:
            assert k.verify_in in k.roles and STUDY_ROLES[k.verify_in].tool


def test_every_step_tool_named_by_a_role_exists():
    import importlib
    named = {r.id: r.tool for r in STUDY_ROLES.values() if r.tool}
    assert set(named) == {"identify", "sensitivity", "uq", "analyze", "forecast"}
    for tid in named.values():
        mod, fn = tid.split(":")
        assert callable(getattr(importlib.import_module(mod), fn)), tid
