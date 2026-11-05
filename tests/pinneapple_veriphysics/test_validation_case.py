"""Validation case (pinneapple_veriphysics.validation_case): status from evidence + review, and an append-only history."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_veriphysics.findings import FAIL
from pinneapple_veriphysics.ledger import EvidenceLedger, sha256_bytes
from pinneapple_veriphysics.usecase import PROFILE_PRESETS, ValidationProfile
from pinneapple_veriphysics.validation_case import (VALIDATED, FAILED, REVIEW_REQUIRED, UNVALIDATED,
                                                    ApplicabilityFacts, CheckRecord, ExpertReview, ModelRef,
                                                    ProblemRef, ProcessControls, Provenance, ReferenceComparison,
                                                    UncertaintyReport, ValidationCase, ValidationRegistry,
                                                    applicability_facts_from_map, checks_from_lab_report,
                                                    checks_from_ledger, checks_from_trust_card)

PROFILE = PROFILE_PRESETS["industrial_screening"].with_changes(
    primary_metric="relative error of T at the wall", target_error_pct=5.0, decision="screen candidate designs",
    critical_quantities=("T at the wall",))


def _case(profile: ValidationProfile | None = None, **kw) -> ValidationCase:
    return ValidationCase.open(profile=profile or PROFILE, model=ModelRef("pinn_heat2d", "v3", checkpoint_sha256="a" * 64),
                               problem=ProblemRef("steady heat 2-D", pde="poisson", geometry="unit square"),
                               requested_use="screening at 95 C", actor="analyst", **kw)


def _verified(case: ValidationCase) -> ValidationCase:
    """A case whose automated side is complete: checks, an independent reference inside tolerance, domain, UQ."""
    case = case.with_checks([CheckRecord("PDE residual", "verification", True, 3e-3, 1e-2),
                             CheckRecord("boundary conditions", "physics", True, 1e-6, 1e-5),
                             CheckRecord("conservation (net flux = source)", "physics", True, 2e-3, 1e-2)])
    case = case.with_reference(ReferenceComparison("ref01", "analytic solution", "T at the wall", 1.2,
                                                  source="exact solution"))
    case = case.with_applicability(ApplicabilityFacts(checked=True, requested_condition="screening at 95 C",
                                                      verified_ranges=(("T_inlet", 25.0, 80.0),)))
    case = case.with_uncertainty(UncertaintyReport("ensemble spread", spread_pct=1.7, limitation="uncalibrated"))
    case = case.with_provenance(Provenance(tool="pinneapple check", tool_version="1.2", python="3.11.9",
                                          platform="darwin", code_sha256="b" * 64, data_sha256="c" * 64, seeds=(0, 1, 2),
                                          tolerances=(("residual", 1e-2),)))
    return case


# --------------------------------------------------------------------------- status policy
def test_a_case_with_no_evidence_is_unvalidated_not_approved():
    case = _case()
    a = case.assess()
    assert a.status == UNVALIDATED and "no evidence" in a.reason
    # and it never reads as validated because nothing failed
    assert not any(f.status == FAIL for f in a.findings)


def test_a_missing_independent_reference_blocks_validation_even_when_the_checks_pass():
    case = _case().with_checks([CheckRecord("PDE residual", "verification", True, 1e-3, 1e-2)])
    a = case.assess()
    assert a.status == UNVALIDATED and "independent reference" in a.reason


def test_complete_evidence_still_needs_a_human_decision():
    a = _verified(_case()).assess()
    assert a.status == REVIEW_REQUIRED and "no expert decision" in a.reason
    assert a.maturity.level("industrial") >= 4


def test_an_error_above_tolerance_fails_and_a_reviewer_cannot_override_it():
    case = _verified(_case()).with_reference(
        ReferenceComparison("ref02", "experimental data", "T at the wall", 12.0, source="bench"))
    case = case.with_review(ExpertReview.record("eng. A", "approved", "looks fine to me", scope={"uses": []}))
    a = case.assess()
    assert a.status == FAILED and a.status_source == "technical"
    assert "ref02" in a.reason or "Error above" in a.reason


def test_an_approval_outside_the_requested_use_does_not_validate_it():
    case = _verified(_case()).with_review(ExpertReview.record("eng. A", "approved", "reviewed the evidence",
                                                             scope={"uses": ["screening at 90 C"]}))
    a = case.assess()
    assert a.status == REVIEW_REQUIRED and "does not cover the requested use" in a.reason


def test_a_full_loop_reaches_validated_with_the_scope_recorded():
    case = _verified(_case()).with_review(
        ExpertReview.record("eng. B", "approved with conditions", "error inside tolerance in the verified envelope",
                            role="lead", scope={"uses": ["screening at 95 C"], "temperature": "25-80 C"}))
    a = case.assess()
    assert a.status == VALIDATED and a.status_source == "authorization"
    assert a.approved_scope == {"uses": ["screening at 95 C"], "temperature": "25-80 C"}
    assert "eng. B" in a.reason


def test_a_rejection_is_recorded_as_an_authorization_decision_not_a_technical_failure():
    case = _verified(_case()).with_review(ExpertReview.record("eng. C", "rejected", "the reference is not converged"))
    a = case.assess()
    assert a.status == FAILED and a.status_source == "authorization" and "not a technical failure" in a.reason


def test_a_mandatory_check_that_did_not_run_keeps_the_case_unvalidated():
    case = _verified(_case()).with_checks([CheckRecord("PDE residual", "verification", True, 1e-3, 1e-2),
                                           CheckRecord("boundary conditions", "physics", None, critical=True,
                                                       message="no data")])
    a = case.assess()
    assert a.status == UNVALIDATED and "mandatory check" in a.reason


def test_an_expert_decision_needs_a_justification():
    with pytest.raises(ValueError, match="justification"):
        ExpertReview(reviewer="eng. A", decision="approved", justification="  ")
    with pytest.raises(ValueError, match="decision"):
        ExpertReview(reviewer="eng. A", decision="approved-ish", justification="because")


# --------------------------------------------------------------------------- append-only history
def test_every_change_appends_a_revision_and_the_history_verifies():
    case = _verified(_case()).with_review(ExpertReview.record("eng. A", "approved", "checked", scope={}))
    assert [r.revision for r in case.history] == list(range(1, len(case.history) + 1))
    assert case.history_problems() == []
    assert case.history[-1].status == VALIDATED
    assert case.history[0].action == "case opened" and case.history[0].status == UNVALIDATED


def test_an_edited_revision_is_detected():
    case = _verified(_case())
    bad = case.history[2]
    tampered = type(bad)(**{**bad.to_dict(), "snapshot": {**bad.snapshot, "requested_use": "screening at 200 C"}})
    broken = type(case)(**{**{f: getattr(case, f) for f in case.__dataclass_fields__},
                           "history": case.history[:2] + (tampered,) + case.history[3:]})
    problems = broken.history_problems()
    assert problems and "altered after it was sealed" in problems[0]


def test_evidence_is_never_overwritten():
    case = _verified(_case())
    with pytest.raises(ValueError, match="already in this case"):
        case.with_reference(ReferenceComparison("ref01", "benchmark", "T", 0.1))


def test_previous_reviews_stay_in_the_history_while_the_latest_decides():
    first = ExpertReview.record("eng. A", "more evidence requested", "run a finer reference")
    second = ExpertReview.record("eng. B", "approved", "evidence reviewed", scope={"uses": ["screening at 95 C"]})
    case = _verified(_case()).with_review(first)
    assert case.assess().status == REVIEW_REQUIRED
    case = case.with_review(second)
    assert case.assess().status == VALIDATED and len(case.reviews) == 2
    assert any("more evidence requested" in r.action for r in case.history)


# --------------------------------------------------------------------------- serialisation
def test_a_case_round_trips_with_the_same_status_and_history():
    case = _verified(_case()).with_review(ExpertReview.record("eng. A", "approved", "checked", scope={}))
    back = ValidationCase.from_dict(case.to_dict())
    assert back.assess().status == case.assess().status == VALIDATED
    assert back.history_problems() == [] and len(back.history) == len(case.history)
    assert back.to_dict()["assessment"]["status"] == VALIDATED


def test_the_model_version_must_be_identifiable():
    assert ModelRef("m").problems()
    assert not ModelRef("m", "v1", configuration=(("epochs", 200),)).problems()
    assert "version" in ModelRef("m", checkpoint_sha256="a" * 64).problems()[0]


# --------------------------------------------------------------------------- extrapolation from a map
def test_a_query_outside_the_verified_range_is_detected():
    amap = {"envelope": [{"name": "x", "kind": "coordinate", "verified_range": [0.0, 1.0], "verified_point": None,
                          "extrapolation_supported": False, "note": ""},
                         {"name": "T_inlet", "kind": "parameter", "verified_range": None, "verified_point": 25.0,
                          "extrapolation_supported": False, "note": ""}],
            "checklist": [], "verified": [], "inferred": [], "unsupported": []}
    inside = applicability_facts_from_map(amap, {"x": 0.5, "T_inlet": 25.0})
    assert inside.checked and inside.outside_variables == () and inside.within_domain
    out = applicability_facts_from_map(amap, {"x": 1.5, "T_inlet": 95.0}, requested_condition="95 C")
    assert set(out.outside_variables) == {"x", "T_inlet"} and not out.within_domain
    assert out.verified_ranges == (("x", 0.0, 1.0),) and out.requested_condition == "95 C"


def test_a_measured_extrapolation_margin_widens_the_range_it_actually_supports():
    amap = {"envelope": [{"name": "x", "kind": "coordinate", "verified_range": [0.0, 1.0], "verified_point": None,
                          "extrapolation_supported": True,
                          "note": "Residual sampled uniformly over this range; measured to hold up to +20% beyond it."}]}
    facts = applicability_facts_from_map(amap, {"x": 1.1})
    assert facts.supported_margin == pytest.approx(0.20) and facts.outside_variables == ()
    assert applicability_facts_from_map(amap, {"x": 1.3}).outside_variables == ("x",)


# --------------------------------------------------------------------------- converters from real tools
def test_a_real_trust_card_becomes_check_records():
    x = y = np.linspace(0, 1, 25)
    X, Y = np.meshgrid(x, y, indexing="ij")
    T = np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)
    import pinneapple as pp
    card = pp.trust_card({"x": x, "y": y}, T, pde="laplace",
                         dirichlet={"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": lambda P: np.sin(np.pi * P[:, 0])},
                         reference=T, train_bounds={"x": (0, 1), "y": (0, 1)})
    records = checks_from_trust_card(card)
    assert {c.category for c in records} <= {"verification", "physics", "validation", "applicability", "other"}
    assert any(c.check_id == "PDE residual" and c.passed for c in records)
    assert any(c.category == "applicability" for c in records)
    case = _case().with_checks(records)
    assert case.assess().findings


def test_a_lab_report_and_a_ledger_become_check_records():
    lab = {"outcomes": [{"rule": "PC401", "target": "T.npz", "subject": "T", "passed": True, "value": 0.003,
                         "threshold": 0.05, "message": "residual ok", "severity": "error", "category": "verification",
                         "check_id": "PC401/T.npz/T"}]}
    assert checks_from_lab_report(lab)[0].passed is True
    led = EvidenceLedger()
    led.record("residual", passed=True, inputs_sha256=sha256_bytes(b"x"), outputs={"value": 1e-3},
               started_at="2026-01-01T00:00:00Z", finished_at="2026-01-01T00:00:01Z", category="verification")
    led.record("conservation", passed=None, inputs_sha256=sha256_bytes(b"x"), outputs={},
               started_at="2026-01-01T00:00:00Z", finished_at="2026-01-01T00:00:01Z", category="physics")
    records = {c.check_id: c for c in checks_from_ledger(led)}
    assert records["residual"].passed is True and records["residual"].value == 1e-3


# --------------------------------------------------------------------------- registry
def test_the_registry_keeps_cases_and_turns_gaps_into_tasks():
    reg = ValidationRegistry()
    reg.add(_verified(_case()).with_review(ExpertReview.record("eng. A", "approved", "checked")))
    other = ValidationCase.open(profile=PROFILE, model=ModelRef("pinn_heat2d", "v3", checkpoint_sha256="a" * 64),
                                problem=ProblemRef("transient heat 2-D", pde="heat"), requested_use="screening at 95 C")
    reg.add(other.with_reference(ReferenceComparison("ref02", "benchmark", "T at the wall", 30.0)))
    assert len(reg.cases) == 2 and reg.by_status(FAILED) and reg.by_status(VALIDATED)
    assert len(reg.by_model("pinn_heat2d")) == 2
    tasks = reg.tasks()
    assert any(t["source"].startswith("maturity requirement") for t in tasks)
    assert any(t["source"].startswith("finding ") for t in tasks) and any(t["priority"] == "high" for t in tasks)
    rows = reg.evidence_rows()
    assert {r["case_id"] for r in rows} == set(reg.cases)


def test_a_case_id_is_stable_for_the_same_model_and_problem():
    a, b = _case(), _case()
    assert a.case_id == b.case_id
    c = ValidationCase.open(profile=PROFILE, model=ModelRef("pinn_heat2d", "v3", checkpoint_sha256="a" * 64),
                            problem=ProblemRef("another problem"))
    assert c.case_id != a.case_id


def test_the_registry_round_trips_through_a_file(tmp_path):
    reg = ValidationRegistry()
    reg.add(_verified(_case()))
    sha = reg.save(str(tmp_path / "registry.json"))
    back = ValidationRegistry.load(str(tmp_path / "registry.json"))
    assert set(back.cases) == set(reg.cases) and len(sha) == 64
    assert back.get(list(reg.cases)[0]).assess().status == REVIEW_REQUIRED


def test_controls_document_the_operational_requirements():
    controls = ProcessControls(owner="team", monitoring="residual on every run", fallback="run the solver")
    assert controls.documented
    assert not ProcessControls(monitoring="x").documented
