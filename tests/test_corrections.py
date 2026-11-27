"""A correction recorded once is applied to the next design, and the report names the rule (#173).

The failure these tests are written against is the quiet one: an engineer corrects a generated design, the correction
stays in the chat, and the next campaign generates the same design again. So the correction is stored as a rule with
its reviewer and date, and a design is checked against every stored rule that applies to it — with a status that says
``contradicts`` when the design repeats the mistake, and ``not_run`` (never ``supports``) when the design does not
carry the number the rule reads.
"""
from __future__ import annotations

import json

import pytest

from pinneapple_analysis.trust import Correction, CorrectionStore, TrustReport, with_corrections
from pinneapple_analysis.trust.corrections import CorrectionError, correction_checks
from pinneapple_analysis.trust.trust_report import CONTRADICTS, NOT_RUN, SUPPORTS, Check


def _rule(**kw):
    base = dict(rule_id="R-001", condition="thickness_mm >= 2.0", case="bracket*",
                statement="a 1.5 mm web cracked in test 42; nothing under 2 mm", reviewer="A. Ribeiro",
                date="2026-11-20", source="design D-17")
    base.update(kw)
    return Correction(**base)


# ---------------------------------------------------------------------------------------- the rule as a check
def test_a_design_that_repeats_the_mistake_contradicts_the_rule_and_the_check_names_it():
    chk = _rule().check({"thickness_mm": 1.5})
    assert chk.status == CONTRADICTS
    assert chk.name == "correction R-001"
    assert chk.critical and chk.limit is None
    assert "R-001" not in chk.reason and "1.5 mm web cracked" in chk.reason   # the sentence travels with the failure
    assert "A. Ribeiro" in chk.source and "2026-11-20" in chk.source


def test_a_design_that_respects_the_rule_supports_it():
    chk = _rule().check({"thickness_mm": 2.5})
    assert chk.status == SUPPORTS and chk.reason == ""


def test_a_design_without_the_column_is_not_run_and_never_a_pass():
    chk = _rule().check({"mass_kg": 3.0})
    assert chk.status == NOT_RUN
    assert "thickness_mm" in chk.reason


def test_a_missing_value_is_not_a_pass_either():
    chk = _rule().check({"thickness_mm": float("nan")})
    assert chk.status == NOT_RUN and "missing number" in chk.reason


def test_a_rule_only_applies_to_the_case_it_was_recorded_for():
    rule = _rule()
    assert rule.applies_to("bracket-A") and rule.applies_to("bracket")
    assert not rule.applies_to("airfoil-NACA0012")
    assert _rule(case="*").applies_to("anything")


def test_a_rule_without_a_reviewer_or_with_a_bad_date_is_refused():
    with pytest.raises(CorrectionError, match="reviewer"):
        _rule(reviewer="  ")
    with pytest.raises(CorrectionError, match="ISO 8601"):
        _rule(date="20/11/2026")


def test_a_rule_whose_condition_cannot_be_parsed_fails_at_recording_time_not_at_the_next_design():
    with pytest.raises(CorrectionError, match="checkable expression"):
        _rule(condition="thickness_mm >>> 2")


# ---------------------------------------------------------------------------------------- the store
def test_a_recorded_rule_reads_back_identical_and_the_chain_verifies(tmp_path):
    store = CorrectionStore(tmp_path / "corrections.jsonl")
    rule = store.save(_rule())
    store.save(_rule(rule_id="R-002", case="airfoil*", condition="cd <= 0.012"))
    assert store.verify() == []
    back = store.rules()[0]
    assert back == rule and back.id == rule.id
    assert [c.rule_id for c in store.rules(case="bracket-A")] == ["R-001"]
    assert [c.rule_id for c in store.rules(case="airfoil")] == ["R-002"]


def test_saving_the_same_rule_twice_is_idempotent(tmp_path):
    store = CorrectionStore(tmp_path / "corrections.jsonl")
    store.save(_rule())
    store.save(_rule())
    assert len(store) == 1


def test_an_edited_rule_is_detected_by_the_chain(tmp_path):
    path = tmp_path / "corrections.jsonl"
    store = CorrectionStore(path)
    store.save(_rule())
    store.save(_rule(rule_id="R-002", condition="mass_kg <= 50"))
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["condition"] = "thickness_mm >= 0.5"                 # the rule quietly loosened after the fact
    path.write_text("\n".join([json.dumps(row), lines[1]]) + "\n", encoding="utf-8")
    problems = store.verify()
    assert problems and "altered after saving" in problems[0]


# ---------------------------------------------------------------------------------------- applied to a new design
def test_a_correction_recorded_once_is_applied_to_a_new_design_automatically(tmp_path):
    """The acceptance: record the rule, then build the trust report of a design that never heard of it."""
    store = CorrectionStore(tmp_path / "corrections.jsonl")
    store.save(_rule())
    report = TrustReport([Check("accuracy", SUPPORTS)], model="bracket-surrogate")
    updated = with_corrections(report, store.rules(), {"thickness_mm": 1.5, "mass_kg": 3.2}, case="bracket-B")
    failed = [c for c in updated.checks if c.status == CONTRADICTS]
    assert [c.name for c in failed] == ["correction R-001"]
    assert updated.decisao == "REJECT"                       # a critical rule that a design contradicts rejects
    assert updated.details["corrections"]["failed"] == ["correction R-001"]


def test_the_report_for_a_rule_it_could_not_check_says_so_and_does_not_claim_a_pass(tmp_path):
    store = CorrectionStore(tmp_path / "corrections.jsonl")
    store.save(_rule())
    updated = with_corrections(TrustReport([Check("accuracy", SUPPORTS)]), store.rules(), {"mass_kg": 3.2},
                               case="bracket-B")
    assert [c.status for c in updated.checks if c.name.startswith("correction")] == [NOT_RUN]
    assert updated.details["corrections"]["not_run"] == ["correction R-001"]
    assert updated.decisao == "REVIEW"                       # coverage drops, and nothing pretends otherwise


def test_a_rule_for_another_case_is_not_in_the_report(tmp_path):
    store = CorrectionStore(tmp_path / "corrections.jsonl")
    store.save(_rule())
    updated = with_corrections(TrustReport([Check("accuracy", SUPPORTS)]), store.rules(),
                               {"thickness_mm": 1.5}, case="airfoil-001")
    assert updated.checks == [Check("accuracy", SUPPORTS)]
    assert updated.details["corrections"]["applied"] == []


def test_correction_checks_of_several_rules_keep_the_order_they_were_recorded_in():
    rules = [_rule(rule_id="R-010", condition="mass_kg <= 50"), _rule(rule_id="R-002")]
    checks = correction_checks(rules, {"mass_kg": 60.0, "thickness_mm": 3.0}, case="bracket-A")
    assert [(c.name, c.status) for c in checks] == [("correction R-010", CONTRADICTS),
                                                    ("correction R-002", SUPPORTS)]
