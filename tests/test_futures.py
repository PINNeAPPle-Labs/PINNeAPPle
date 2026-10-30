"""pinneapple_futures (pp.futures): the epistemic schema, the What If Engine, the plausibility framework and the
three demonstrators -- each of which must pass its own comparisons and checks, and each of which must say what it
did not assess."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import pinneapple as pp
import pinneapple_futures as F
from pinneapple_futures import plausibility, schema
from pinneapple_futures.labs import exoplanet, programmable_matter
from pinneapple_futures.labs.mars_habitat import composite_wall, resistance_network

EXPERIMENTS = ["futures.exoplanet_laboratory", "futures.mars_habitat", "futures.programmable_matter"]


# ------------------------------------------------------------------------------------------------ schema
def test_plausibility_classes_are_ordered_and_the_weakest_link_wins():
    assert schema.weakest(["established", "extrapolated"]) == "extrapolated"
    assert schema.weakest(["hypothetical", "fictional"]) == "fictional"
    assert schema.weakest(["established"]) == "established"
    assert schema.weakest([]) == "fictional"           # nothing supports it
    with pytest.raises(ValueError):
        schema.weakest(["plausible"])


def test_a_number_without_provenance_or_source_is_refused():
    with pytest.raises(ValueError):
        F.Input("k", 1.0, "W/m/K", "guess", "a hunch")
    with pytest.raises(ValueError):
        F.Input("k", 1.0, "W/m/K", "design_input", "")
    with pytest.raises(ValueError):
        F.Input("k", 1.0, "", "design_input", "a choice")
    ok = F.Input("k", 1.0, "-", "design_input", "a choice")
    assert ok.to_dict()["provenance"] == "design_input"


def test_a_hypothesis_needs_a_falsifier_and_an_unestablished_claim_needs_a_source():
    with pytest.raises(ValueError):
        F.Hypothesis("the sky is green")
    with pytest.raises(ValueError):
        F.Claim("cities can float", "F = ma", "hypothetical", "")
    assert F.Hypothesis("x", "y", "a measurement that would contradict it").plausibility == "extrapolated"


def test_scenario_conversions_are_conversions_not_a_model():
    sc = F.Scenario("super", gravity=2.5 * schema.EARTH_GRAVITY, ambient_temperature=288.0)
    d = sc.derived()
    assert d["gravity_si"].value == pytest.approx(24.5166, abs=1e-4)
    assert d["relative_weight"].value == pytest.approx(2.5)
    assert d["temperature_c"].value == pytest.approx(14.85, abs=1e-9)
    assert sc.with_gravity(9.80665).gravity == 9.80665
    assert sc.gravity == 2.5 * schema.EARTH_GRAVITY      # with_gravity returns a copy
    with pytest.raises(ValueError):
        F.Scenario("bad", gravity=0.0)
    with pytest.raises(ValueError):
        F.Scenario("bad", ambient_temperature=-1.0)


def test_a_proposal_knows_what_it_still_needs():
    p = F.Proposal("t", "s", F.Scenario("w"))
    assert not p.complete and "formalization (no equation)" in p.missing()
    with pytest.raises(ValueError):
        p.run()
    full = exoplanet.super_earth()
    assert full.complete and full.missing() == []
    assert full.classification() == "extrapolated"        # the scenario's speculative change, not "established"
    assert "speculation" in full.stages() and "assessment" in full.stages()


def test_classification_reads_anything_with_a_class():
    claim = F.Claim("steady conduction", "d/dx[k dT/dx] = 0")
    hyp = F.Hypothesis("a material exists", "materials", "it cannot be made", plausibility="hypothetical")
    assert F.classification(claim, hyp) == "hypothetical"
    assert F.classification("established", claim) == "established"
    with pytest.raises(TypeError):
        F.classification(object())


# ---------------------------------------------------------------------------------------------- plausibility
def test_an_unassessed_axis_is_not_a_zero_and_not_a_one():
    a = F.assess(_stub_result([]), epistemic="established")
    assert a.axes["technological_feasibility"].score is None
    assert not a.axes["technological_feasibility"].assessed
    assert "technological_feasibility" in a.missing()
    assert a.physics_verdict == "NOT ASSESSED"
    assert a.engineering_verdict == "NOT ASSESSED"
    assert "cannot establish it" in str(a)


def test_each_axis_counts_only_its_own_evidence():
    checks = [{"name": "residual", "passed": True, "source": "analytic"},
              {"name": "mesh", "passed": True, "source": "convergence"},
              {"name": "balance", "passed": False, "source": "conservation"},
              {"name": "unused", "passed": True, "source": "physics-not-an-axis"}]
    a = F.assess(_stub_result(checks), epistemic="extrapolated")
    assert a.axes["mathematical_consistency"].passed == 1
    assert a.axes["numerical_stability"].passed == 1
    assert a.axes["physical_plausibility"].passed == 0
    assert a.physics_verdict == "CONTRADICTED"
    assert a.blockers() == ["physical_plausibility"]


def test_the_engineering_verdict_is_independent_of_the_physics():
    checks = [{"name": "residual", "passed": True, "source": "analytic"},
              {"name": "can it be built", "passed": False, "source": "engineering"}]
    a = F.assess(_stub_result(checks), epistemic="hypothetical")
    assert a.physics_verdict == "SUPPORTED"
    assert a.engineering_verdict == "NOT FEASIBLE"
    assert "class hypothetical" in a.headline()


def test_axis_scores_are_validated():
    with pytest.raises(ValueError):
        F.AxisScore("nonsense")
    with pytest.raises(ValueError):
        F.AxisScore("numerical_stability", passed=3, total=2)
    assert F.AxisScore("numerical_stability", 1, 4).score == 0.25


def _stub_result(checks):
    class Stub:
        comparisons: list = []

    s = Stub()
    s.comparisons = []
    s.checks = checks
    return s


# ------------------------------------------------------------------------------------------------ what if
def test_the_engine_reads_the_question_it_advertises():
    b = F.ask("What if Earth's gravity were 2.5 times stronger?")
    assert b.understood and b.answerable
    assert b.parameter == "gravity"
    assert b.scenario.gravity == pytest.approx(2.5 * 9.80665)
    assert b.changes["gravity"].value == pytest.approx(24.5166, abs=1e-4)
    assert "futures.exoplanet_laboratory" in b.experiments
    laws = " ".join(b.models)
    assert "R T / (M g)" in laws
    assert any("does not appear" in effect for _, _, effect in b.dependencies())
    stages = b.stages()
    assert len(stages) == 8 and "scenario" in " ".join(stages)


def test_the_engine_reads_portuguese_and_other_parameters():
    assert F.ask("e se a gravidade fosse 2,5 vezes maior?").scenario.gravity == pytest.approx(2.5 * 9.80665)
    p = F.ask("what if the pressure were 5 kPa")
    assert p.scenario.ambient_pressure == pytest.approx(5000.0)
    assert p.experiments == ("futures.mars_habitat",)
    s = F.ask("what if the insolation were 2000 W/m^2")
    assert s.params["insolation"] == pytest.approx(2000.0)
    assert "futures.exoplanet_laboratory" in s.experiments


def test_a_question_it_cannot_read_says_so_and_refuses_to_run():
    b = F.ask("What if cats had wings?")
    assert not b.understood and not b.answerable
    assert any("not understood" in w for w in b.warnings)
    with pytest.raises(ValueError):
        b.run()
    with pytest.raises(ValueError):
        F.ask("   ")


def test_the_scenario_of_a_question_starts_from_the_world_that_will_run_it():
    b = F.ask("what if the pressure were 5 kPa")
    mars = F.get("futures.mars_habitat").world()
    assert b.scenario.gravity == pytest.approx(mars.gravity)              # not reset to Earth's
    assert b.scenario.ambient_temperature == pytest.approx(mars.ambient_temperature)


def test_a_question_runs_the_experiment_and_keeps_the_answer_checkable():
    res = F.ask("What if Earth's gravity were 2.5 times stronger?").run(quick=True)
    assert res.passed and res.decision == "APPROVED"
    earth = exoplanet.scale_height(288.15, 9.80665)
    assert res.values["scale_height"]["value"] == pytest.approx(earth / 2.5, rel=1e-9)
    assert res.proposal is not None and res.proposal["scenario"]["gravity"] == pytest.approx(24.5166, abs=1e-3)


# ------------------------------------------------------------------------------------------------ experiments
def test_every_experiment_is_registered_and_declared():
    assert F.names() == EXPERIMENTS
    for name in EXPERIMENTS:
        card = F.get(name).card()
        assert card["level"] in ("academic", "research")
        assert card["dims"] and card["area"]
        assert card["scenario"] is not None
        assert card["references"], f"{name} cites nothing"


def test_a_futures_experiment_is_a_science_experiment():
    from pinneapple_science.core import Experiment, Result

    exp = F.get("futures.exoplanet_laboratory")
    assert isinstance(exp, Experiment)
    assert exp.result_cls is F.SpeculativeResult and issubclass(exp.result_cls, Result)


def test_the_flow_nodes_list_every_experiment():
    from pinneapple_flow.nodes import futures as nodes

    assert nodes.EXPERIMENTS == F.names()
    for dim in ("0D", "1D"):
        assert dim in nodes.DIMS


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_experiment_passes_its_own_checks(name):
    for dim in F.get(name).dims:
        res = F.get(name).run(dim, quick=True)
        assert res.passed, f"{name} [{dim}] failed: {[c.quantity for c in res.comparisons if not c.passed]} " \
                           f"{[c['name'] for c in res.checks if c['passed'] is False]}"
        assert res.decision == "APPROVED"
        assert res.assessment is not None
        assert res.assessment.physics_verdict == "SUPPORTED"
        assert res.assessment.engineering_verdict == "NOT ASSESSED"


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_result_never_claims_what_it_did_not_assess(name):
    res = F.run(name, quick=True)
    a = res.assessment
    assert a.axes["technological_feasibility"].score is None
    assert a.axes["technological_feasibility"].evidence == ()
    text = json.dumps(res.to_dict())
    assert "pinneapple.futures.result/1" in text
    assert res.to_dict()["assessment"]["engineering_verdict"] == "NOT ASSESSED"
    assert "technological_feasibility" in res.to_dict()["assessment"]["not_assessed"]
    # only the experiment that compares against a measurement may claim one
    bench = [c for c in res.checks if c["source"] == "bench"]
    assert a.axes["experimental_validation"].assessed == bool(bench)


def test_only_a_measurement_earns_the_experimental_axis():
    res = F.run("futures.exoplanet_laboratory", quick=True)
    bench = [c["name"] for c in res.checks if c["source"] == "bench"]
    assert len(bench) == 1 and "ISO 2533" in bench[0]
    assert F.run("futures.mars_habitat", quick=True).assessment.axes["experimental_validation"].score is None


def test_the_epistemic_class_of_a_result_comes_from_its_world():
    assert F.run("futures.programmable_matter", quick=True).classification == "hypothetical"
    assert F.run("futures.mars_habitat", quick=True).classification == "extrapolated"
    res = exoplanet.super_earth().run(quick=True)
    assert res.classification == "extrapolated"


def test_experiment_parameters_are_validated_and_unknown_ones_refused():
    exp = F.get("futures.exoplanet_laboratory")
    with pytest.raises(TypeError):
        exp.run(wrong=1)
    with pytest.raises(ValueError):
        exp.run("5D")
    with pytest.raises(ValueError):
        exoplanet.equilibrium_temperature(1361.0, 1.2)
    with pytest.raises(ValueError):
        exoplanet.scale_height(-1.0, 9.8)


# ------------------------------------------------------------------------------------------------ the labs
def test_the_exoplanet_is_not_hotter_just_because_it_is_heavier():
    light = F.run("futures.exoplanet_laboratory", quick=True, gravity=9.80665)
    heavy = F.run("futures.exoplanet_laboratory", quick=True, gravity=2.5 * 9.80665)
    assert heavy.values["T_eq_redistributed"]["value"] == pytest.approx(light.values["T_eq_redistributed"]["value"])
    assert heavy.values["scale_height"]["value"] == pytest.approx(light.values["scale_height"]["value"] / 2.5)


def test_the_latitudinal_solution_reproduces_its_legendre_solution():
    res = F.run("futures.exoplanet_laboratory", dim="1D", quick=True)
    x = res.arrays["profile"]["x"]
    t = res.arrays["profile"]["T"]
    exact = res.arrays["profile"]["T_exact"]
    assert np.max(np.abs(t - exact)) / np.ptp(exact) < 1e-2
    assert t[0] < t[len(t) // 2]                     # pole colder than equator: the transport damps the contrast


def test_the_mars_wall_matches_its_resistance_network_on_any_mesh():
    layers = [(1.0, 0.10), (0.10, 0.015), (0.01, 0.50)]
    r = resistance_network(layers, 8.0, 3.0)
    for nodes, grade in ((11, 1.0), (41, 1.0), (19, 1.7), (57, 0.6)):
        w = composite_wall(layers, 294.15, 210.0, 8.0, 3.0, nodes, grade)
        assert w["q"] == pytest.approx((294.15 - 210.0) / r, rel=1e-10)
        assert np.ptp(w["faces"]) / abs(w["q"]) < 1e-9


def test_the_programmable_material_stops_regulating_where_it_stops_conducting():
    k0, beta, t_hot, t_cold = 5.0, 2e-3, 400.0, 300.0
    q = programmable_matter.variable_k_flux(k0, beta, t_hot, t_cold)
    assert q == pytest.approx(k0 * ((t_hot - t_cold) + 0.5 * beta * (t_hot**2 - t_cold**2)))
    res = F.run("futures.programmable_matter", quick=True)
    assert res.values["sensitivity_with_a_falling_conductivity"]["value"] == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError):
        programmable_matter.switch_k_flux(2.0, 8.0, 200.0, 400.0, 300.0)


# ------------------------------------------------------------------------------------------------ atlas
def test_the_atlas_points_only_to_code_that_exists():
    assert F.validate() == []


def test_the_atlas_covers_the_eight_directions_and_renders():
    at = F.atlas()
    ids = {a.id for a in at.areas}
    assert ids == {"speculative_physics", "future_engineering", "artificial_life", "planetary_systems",
                   "advanced_materials", "post_biological_systems", "computational_worldbuilding",
                   "civilization_scale_engineering"}
    assert set(at.counts()) == {"available", "partial", "planned"}
    assert "| Direction | Subarea | Status | Covered by |" in at.to_markdown()


def test_the_atlas_validation_catches_a_bad_pointer_and_an_uncovered_experiment():
    at = F.atlas()
    at.areas[0].subareas[0]["covered_by"].append({"kind": "module", "module": "pinneapple_futures.no_such"})
    problems = F.validate(at)
    assert any("no_such" in p for p in problems)

    at2 = F.atlas()
    for area in at2.areas:
        for sub in area.subareas:
            sub["covered_by"] = [p for p in sub["covered_by"] if p.get("kind") != "experiment"]
            if sub["status"] != "planned" and not sub["covered_by"]:
                sub["status"] = "planned"
    problems2 = F.validate(at2)
    assert any("nobody" in p or "no direction" in p for p in problems2)


# ------------------------------------------------------------------------------------------------ report
def test_publishing_writes_the_page_the_result_and_the_limits(tmp_path):
    from pinneapple_futures.report import publish

    info = publish("futures.mars_habitat", root=str(tmp_path), quick=True)
    assert info["passed"] and info["decision"] == "APPROVED"
    page = open(info["page"], encoding="utf-8").read()
    for needed in ("## What this does not show", "## Evidence and verdict", "## Against the references",
                   "NASA NSSDC Mars Fact Sheet", "not assessed"):
        assert needed in page
    assert os.path.isfile(info["result"])


def test_the_catalogue_and_the_cli(tmp_path):
    from pinneapple_futures.__main__ import main

    assert main(["atlas", "--check"]) == 0
    assert main(["list"]) == 0
    assert main(["ask", "What if Earth's gravity were 2.5 times stronger?", "--json"]) == 0
    assert main(["ask", "What if cats had wings?"]) == 1
    assert main(["run", "futures.exoplanet_laboratory", "--quick"]) == 0
    assert main(["publish", "all", "--out", str(tmp_path), "--quick"]) == 0
    assert os.path.isfile(os.path.join(str(tmp_path), "Catalogue.md"))
    assert os.path.isfile(os.path.join(str(tmp_path), "ATLAS.md"))


# ------------------------------------------------------------------------------------------------ front door
def test_the_lab_is_reachable_from_the_pp_namespace():
    assert pp.futures.ask is F.ask
    assert "futures.exoplanet_laboratory" in pp.futures.names()
    assert pp.futures.validate() == []
