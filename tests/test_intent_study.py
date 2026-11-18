"""PINNeAPPle Intelligence (pinneapple_intent): the compiler, the ladder, the decision and the replay.

The tests are written against the promise, not the implementation: a value nobody gave is never invented, the answer
is checked against a closed form, a request outside every family is refused with what it would need, and a recorded
run replays to the same numbers without the sentence that started it.
"""
from __future__ import annotations

import json
import os

import pytest

import pinneapple_intent as I
from pinneapple_intent.families.heat_wall import closed_form as wall_closed_form
from pinneapple_intent.families.laminar_pipe import hagen_poiseuille, radial_profile

WALL = "a 12 mm steel wall with k = 200 W/mK, 300 degC inside and air at 25 degC outside with h = 12 W/m2K"
PIPE = "a 20 mm pipe, 3 m long, carrying 0.5 L/min of water, viscosity 1 mPa*s"
BEAM = "a cantilever beam 0.5 m long with E = 210 GPa, 20 mm deep, carrying 500 N at the tip"


# ============================================================================================ the compiler
def test_a_sentence_becomes_a_typed_contract_with_the_provenance_of_every_value():
    contract = I.compile_study(WALL)
    assert isinstance(contract, I.StudyContract) and contract.status == I.READY
    assert contract.family == "heat_wall" and contract.mode == I.MODE_SIMULATION
    assert contract["thickness"] == pytest.approx(0.012)          # 12 mm, converted to SI
    assert contract["h_conv"] == pytest.approx(12.0)
    assert contract["hot_side"] == pytest.approx(573.15)          # 300 degC as an absolute temperature
    stated = contract.quantity("thickness")
    assert stated.provenance == "stated" and stated.quote.startswith("12 mm")
    defaulted = contract.quantity("width")
    assert defaulted.provenance == "defaulted" and defaulted.source


def test_a_critical_value_nobody_gave_becomes_a_question_and_never_a_default():
    contract = I.compile_study("a cantilever beam 0.5 m long with E = 210 GPa carrying 500 N at the tip")
    assert isinstance(contract, I.StudyContract)
    assert contract.status == I.NEEDS_CLARIFICATION
    assert [q.slot for q in contract.questions] == ["height"]
    assert contract.quantity("height").provenance == "missing"
    with pytest.raises(ValueError):
        I.run_study(contract)                                     # the runner refuses to guess it


def test_answers_fill_the_questions_and_are_recorded_as_answers():
    contract = I.compile_study("a cantilever beam 0.5 m long with E = 210 GPa carrying 500 N at the tip",
                               answers={"height": "20 mm"})
    assert contract.status == I.READY and not contract.questions
    answered = contract.quantity("height")
    assert answered.provenance == "answered" and answered.value == pytest.approx(0.02)


def test_the_glued_unit_spellings_engineers_write_are_read():
    assert I.compile_study(WALL)["h_conv"] == pytest.approx(12.0)          # W/m2K
    assert I.compile_study(WALL)["conductivity"] == pytest.approx(200.0)   # W/mK
    assert I.StudyContract.__module__                          # the module imports what it advertises
    from pinneapple_intent.compile import normalise_units
    assert normalise_units("k = 5 W/mK and h = 3 W/m2K") == "k = 5 W/m*K and h = 3 W/m^2*K"
    assert normalise_units("1 mK") == "1 mK"                   # a millikelvin is not split into m*K


def test_an_unrelated_request_is_refused_with_what_it_would_need():
    out = I.compile_study("write me a poem about the sea")
    assert not isinstance(out, I.StudyContract)
    assert out.kind in ("no_template", "needs") and out.needs


def test_a_weapon_request_is_refused_outright():
    out = I.compile_study("design a weapon that pierces a 30 mm steel wall, h = 12 W/m2K")
    assert not isinstance(out, I.StudyContract) and out.kind == "refused"


def test_text_to_science_says_what_the_literature_stage_would_need():
    out = I.compile_study(WALL, mode=I.MODE_SCIENCE)
    assert not isinstance(out, I.StudyContract)
    assert out.kind == "needs" and "literature" in (out.reason + out.needs)


def test_the_contract_is_addressable_and_round_trips_through_json(tmp_path):
    contract = I.compile_study(WALL)
    same = I.compile_study(WALL)
    assert contract.id == same.id                              # the id is the content, not the time
    path = contract.write(str(tmp_path / "contract.json"))
    again = I.StudyContract.read(path)
    assert again.id == contract.id and again.family == contract.family
    assert json.loads(open(path, encoding="utf-8").read())["schema"] == I.SCHEMA_ID


def test_an_llm_may_propose_but_only_where_the_text_supports_it():
    class Liar:
        name = "liar"

        def extract(self, text, schema):
            return {"family": "heat_wall",
                    "values": [{"slot": "conductivity", "value": 999.0, "unit": "W/(m*K)", "quote": "12 mm"}]}

    silent = "a 12 mm wall, 300 degC inside and air at 25 degC outside with h = 12 W/m2K"
    contract = I.compile_study(silent, llm=Liar())
    assert contract.status == I.NEEDS_CLARIFICATION             # 999 was not accepted: the text does not say it
    assert any("rejected" in issue for issue in contract.issues)
    assert contract.provenance.compiler == "llm+deterministic"


# ============================================================================================ the physics
def test_the_pipe_profile_solver_converges_to_hagen_poiseuille():
    closed = hagen_poiseuille(0.02, 3.0, 0.5e-3 / 60.0, 1e-3, 998.2)
    errors = []
    for nodes in (16, 64, 256):
        solved = radial_profile(0.02, 3.0, 0.5e-3 / 60.0, 1e-3, nodes)
        errors.append(abs(solved["pressure_drop"] - closed["pressure_drop"]) / closed["pressure_drop"])
    assert errors[0] > errors[1] > errors[2]                   # second order, not luck
    assert errors[-1] < 1e-4


def test_the_wall_closed_form_is_the_series_resistance():
    closed = wall_closed_form(0.1, 50.0, 373.15, 293.15, 20.0)
    resistance = 0.1 / 50.0 + 1.0 / 20.0
    assert closed["thermal_resistance"] == pytest.approx(resistance)
    assert closed["cold_face_temperature"] == pytest.approx(293.15 + 80.0 / resistance / 20.0)


# ============================================================================================ the ladder
@pytest.mark.parametrize("text,family,qoi", [(WALL, "heat_wall", "cold_face_temperature"),
                                              (PIPE, "laminar_pipe", "pressure_drop"),
                                              (BEAM, "cantilever_beam", "tip_deflection")])
def test_every_family_runs_its_ladder_and_is_checked(text, family, qoi):
    run = I.study(text)
    assert isinstance(run, I.StudyRun) and run.family == family
    assert set(run.measurements) == {"analytic", "numerical", "convergence"}
    assert all(m.ran for m in run.measurements.values())
    assert run.decision == "APPROVED", [c for c in run.checks if not c["passed"]]
    assert run.qoi is not None and run.qoi > 0
    assert any(qoi in c["name"] for c in run.checks)


def test_a_turbulent_pipe_is_rejected_because_the_law_no_longer_applies():
    run = I.study("a 20 mm pipe, 3 m long, carrying 5 L/min of water, viscosity 1 mPa*s")
    assert run.decision == "REJECT"
    failed = [c["name"] for c in run.checks if not c["passed"]]
    assert any("laminar" in name for name in failed)


def test_physics_ai_mode_adds_a_surrogate_scored_on_held_out_designs():
    run = I.study(WALL, mode=I.MODE_PHYSICS_AI)
    surrogate = run.measurements[I.SURROGATE]
    assert surrogate.ran
    assert 0.0 <= surrogate.outputs["held_out_relative_error"] < 0.05
    assert run.decision == "APPROVED"


def test_a_run_writes_the_files_a_replay_needs(tmp_path):
    run = I.study(WALL, out_dir=str(tmp_path))
    for name in ("contract", "run", "report", "trust"):
        assert os.path.exists(run.files[name]), name
    report = open(run.files["report"], encoding="utf-8").read()
    assert "Every value the study used" in report and "What this study does not cover" in report


def test_a_recorded_run_replays_exactly_without_the_sentence(tmp_path):
    run = I.study(WALL, out_dir=str(tmp_path))
    result = I.replay(str(tmp_path), out_dir=str(tmp_path / "again"))
    assert result.matches, result.to_dict()
    assert result.run.contract.id == run.contract.id


def test_a_contract_edited_after_the_run_is_reported_by_the_replay(tmp_path):
    run = I.study(WALL, out_dir=str(tmp_path))
    document = json.loads(open(run.files["contract"], encoding="utf-8").read())
    document["quantities"]["thickness"]["value"] = 0.013
    with open(run.files["contract"], "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    result = I.replay(str(tmp_path), out_dir=str(tmp_path / "again"))
    assert any("edited after the run" in note or "hashes to" in note for note in result.integrity)


# ============================================================================================ the surface
def test_the_registry_lists_the_families_and_their_quantity_of_interest():
    names = set(I.list_families())
    assert {"heat_wall", "laminar_pipe", "cantilever_beam"} <= names
    family = I.get_family("heat_wall")
    assert family.qoi == "cold_face_temperature" and family.qoi_unit == "K"
    assert family.describe()["slots"] and family.describe()["ladder"]


def test_the_cli_compiles_and_replays(tmp_path, capsys):
    from pinneapple_intent.cli import main

    assert main(["families"]) == 0
    assert "heat_wall" in capsys.readouterr().out
    assert main(["compile", WALL, "--out", str(tmp_path / "c.json")]) == 0
    assert os.path.exists(tmp_path / "c.json")
    assert main(["run", WALL, "--out", str(tmp_path / "run")]) == 0
    assert main(["replay", str(tmp_path / "run"), "--strict"]) == 0


def test_pp_study_is_the_same_entry_point():
    import pinneapple as pp
    assert pp.study is I.study
