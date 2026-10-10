"""Applicability map: pure classification of already-computed facts -- no
training, no pinneapple imports needed for the map itself."""
from pinneapple_veriphysics.applicability import (
    FAIL, INFERRED, NOT_RUN, PASS, build_applicability_map,
)
from pinneapple_veriphysics.decision import DecisionRecord
from pinneapple_veriphysics.evidence_report import render_evidence_report_pdf

CHECKS = [{"name": "pde_residual", "passed": True, "detail": "res=1e-3", "value": 1e-3, "threshold": 1e-2}]


def _map(**kw):
    base = dict(
        architecture="vanilla_pinn", coords=["x", "t"], fields=["u"],
        domain_bounds={"x": (0.0, 1.0), "t": (0.0, 1.0)}, physical_parameters={"nu": 0.01},
        guardrail_checks=CHECKS, confidence_components=[], final_loss=1e-3,
        loss_history=[1.0] * 50 + [1e-3] * 50, epochs=100, n_collocation=64,
    )
    base.update(kw)
    return build_applicability_map(**base)


def _item(m, key):
    return next(i for i in m.checklist if i.key == key)


def test_all_eight_items_present_and_absent_checks_are_not_run():
    m = _map()
    assert [i.key for i in m.checklist] == [
        "surrogate", "error", "convergence", "generalization", "variables", "weights", "uq", "physics_laws"]
    assert _item(m, "uq").status == NOT_RUN
    assert _item(m, "generalization").status == NOT_RUN
    assert _item(m, "physics_laws").status == NOT_RUN and _item(m, "physics_laws").tier == INFERRED
    assert _item(m, "error").status == PASS
    assert any("nu other than 0.01" in u for u in m.unsupported)


def test_unconverged_training_and_failed_conservation_are_reported_as_fail():
    m = _map(loss_history=[1.0 - i * 0.005 for i in range(100)],
             guardrail_checks=CHECKS + [{"name": "conservation_mass_continuity", "passed": False, "detail": "div!=0"}])
    assert _item(m, "convergence").status == FAIL
    assert _item(m, "physics_laws").status == FAIL


def test_uq_and_heldout_become_verified_only_when_supplied():
    m = _map(confidence_components=[{"name": "uq_calibration", "score": 0.8, "source_summary": "coverage 0.9"}],
             calibration={"ece": 0.05, "coverage": 0.91, "target_coverage": 0.9},
             heldout_error={"metric": "rel_L2", "value": 0.02, "threshold": 0.05, "description": "unseen nu=0.02"})
    assert _item(m, "uq").status == PASS
    assert _item(m, "generalization").status == PASS


def test_record_renders_and_pdf_builds_from_dict_form():
    rec = DecisionRecord(
        run_id="r", problem_description="p", recommended_strategy="s", reasoning="r",
        trust_score=50.0, trust_coverage=0.5, trustworthy=True, applicability=_map().to_dict())
    text = rec.render()
    assert "Applicability map" in text and "[NOT_RUN] Uncertainty quantification?" in text
    assert render_evidence_report_pdf(rec).startswith(b"%PDF")


def test_dimensional_check_alone_does_not_count_as_conservation():
    m = _map(guardrail_checks=CHECKS + [{"name": "dimensional_analysis", "passed": True, "detail": "ok"}])
    assert _item(m, "physics_laws").status == NOT_RUN


def test_overconservative_uq_is_not_reported_as_calibrated():
    m = _map(confidence_components=[{"name": "uq_calibration", "score": 0.76, "source_summary": "ECE=0.24"}],
             calibration={"ece": 0.24, "coverage": 1.0, "target_coverage": 0.9})
    assert _item(m, "uq").status == FAIL and "OVER-conservative" in _item(m, "uq").summary


def test_noisy_but_plateaued_loss_counts_as_converged():
    import random
    random.seed(0)
    hist = [1.0 / (1 + i) for i in range(200)] + [1e-3 * (1 + 0.3 * random.random()) for i in range(800)]
    assert _item(_map(loss_history=hist), "convergence").status == PASS
