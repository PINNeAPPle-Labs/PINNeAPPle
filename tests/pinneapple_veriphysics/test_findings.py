"""Findings (pinneapple_veriphysics.findings): four questions per signal, deterministic text, no invented numbers."""
from __future__ import annotations

import pytest

from pinneapple_veriphysics.findings import RULES, build_findings, sort_findings
from pinneapple_veriphysics.usecase import PROFILE_PRESETS
from pinneapple_veriphysics.validation_case import (ApplicabilityFacts, CheckRecord, ExpertReview,
                                                    ReferenceComparison, UncertaintyReport)


def _build(**kw):
    base = dict(profile=PROFILE_PRESETS["industrial_screening"].with_changes(decision="screen designs",
                                                                            target_error_pct=5.0),
                checks=(), references=(), uncertainty=None, applicability=None, provenance_problems=(), reviews=())
    return build_findings(**{**base, **kw})


def _by_rule(findings, rule):
    return [f for f in findings if f.rule == rule]


def test_a_failed_comparison_quotes_the_number_and_the_tolerance_it_was_measured_against():
    f = _by_rule(_build(references=[ReferenceComparison("ref1", "analytic solution", "T at the wall", 6.2,
                                                        source="exact")]), "VP-ERR-OVER")[0]
    assert f.status == "fail" and f.observed == 6.2 and f.threshold == 5.0
    assert "6.2 % error, tolerance 5 %" in f.what_happened
    text = f.render()
    for question in ("What happened:", "Why it matters:", "What is still unknown:", "Recommended next steps:"):
        assert question in text
    assert "ref1" in text and f.finding_id.startswith("VP-")


def test_a_tolerance_set_on_the_comparison_wins_over_the_profile_default():
    f = _by_rule(_build(references=[ReferenceComparison("ref1", "benchmark", "drag", 3.0, tolerance_pct=2.0)]),
                 "VP-ERR-OVER")[0]
    assert f.threshold == 2.0 and f.observed == 3.0


def test_a_check_that_did_not_run_is_never_a_pass():
    findings = _build(checks=[CheckRecord("boundary conditions", "physics", None, critical=True,
                                          message="no Dirichlet data given")])
    assert not any(f.status == "pass" for f in findings)
    not_run = [f for f in findings if f.status == "not run"]
    assert any("boundary conditions" in f.what_happened for f in not_run)
    # a non-mandatory check that did not run stays in the checks table without becoming a finding
    optional = _build(checks=[CheckRecord("maximum principle", "physics", None)])
    assert not any(f.rule == "VP-CHECK-NOT-RUN" for f in optional)


def test_a_mandatory_check_missing_from_the_record_is_reported():
    profile = PROFILE_PRESETS["industrial_screening"].with_changes(required_checks=("conservation",))
    findings = _build(profile=profile, checks=[CheckRecord("PDE residual", "verification", True, 1e-3, 1e-2)])
    assert any(f.rule == "VP-CHECK-NOT-RUN" and "conservation" in f.what_happened for f in findings)


def test_extrapolation_and_uncalibrated_uncertainty_are_warnings_with_their_evidence():
    findings = _build(
        applicability=ApplicabilityFacts(checked=True, requested_condition="95 C", outside_variables=("T_inlet",)),
        uncertainty=UncertaintyReport("ensemble spread", spread_pct=1.7, limitation="no reference data"))
    ex = _by_rule(findings, "VP-EXTRAPOLATION")[0]
    assert ex.status == "warning" and ex.evidence == ("T_inlet",) and "95 C" in ex.what_happened
    uq = _by_rule(findings, "VP-UQ-UNCALIBRATED")[0]
    assert "not calibrated" in uq.what_happened and "1.7" in uq.what_happened


def test_an_absent_uncertainty_is_a_gap_not_a_zero():
    findings = _build()
    assert _by_rule(findings, "VP-UQ-MISSING")[0].status == "not run"


def test_reviews_map_to_distinct_findings_and_keep_the_justification():
    approved = ExpertReview.record("eng. A", "approved with conditions", "within tolerance in the verified range",
                                   scope={"uses": ["screening at 90 C"]})
    f = _by_rule(_build(reviews=[approved]), "VP-REVIEW-APPROVED")[0]
    assert f.status == "pass" and "eng. A" in f.what_happened and "screening at 90 C" in f.what_happened
    rejected = ExpertReview.record("eng. B", "rejected", "reference itself not converged")
    assert _by_rule(_build(reviews=[rejected]), "VP-REVIEW-REJECTED")[0].status == "fail"
    more = ExpertReview.record("eng. B", "more evidence requested", "run the reference on a finer mesh")
    assert _by_rule(_build(reviews=[more]), "VP-REVIEW-MORE")[0].status == "warning"


def test_findings_are_numbered_in_severity_order_and_are_deterministic():
    refs = [ReferenceComparison("ref1", "analytic solution", "T", 40.0), ReferenceComparison("ref2", "benchmark", "T", 1.0)]
    a, b = _build(references=refs), _build(references=refs)
    assert [f.to_dict() for f in a] == [f.to_dict() for f in b]
    statuses = [f.status for f in a]
    assert statuses == [s for s in statuses if s in ("fail", "warning", "not run", "pass")]
    assert statuses.index("fail") == 0 and statuses[-1] in ("pass", "not run")
    assert [f.finding_id for f in a] == [f"VP-{i:03d}" for i in range(1, len(a) + 1)]


def test_every_rule_has_the_four_answers_and_an_unknown_is_never_empty():
    for key, rule in RULES.items():
        assert rule.why.strip() and rule.unknown.strip() and rule.actions, key


def test_an_unrecognised_rule_cannot_fire():
    from pinneapple_veriphysics.findings import _rule
    with pytest.raises(KeyError, match="unknown finding rule"):
        _rule("VP-NOT-A-RULE")


def test_sort_findings_is_stable_for_equal_severity():
    findings = _build(references=[ReferenceComparison("a", "analytic solution", "T", 99.0)])
    assert [f.finding_id for f in sort_findings(findings)] == [f.finding_id for f in findings]
