"""Validation dossier (pinneapple_veriphysics.dossier): the case as the document a reviewer signs."""
from __future__ import annotations

from pinneapple_veriphysics.dossier import DISCLAIMER, ValidationDossier
from pinneapple_veriphysics.usecase import PROFILE_PRESETS
from pinneapple_veriphysics.validation_case import (CheckVerdict, ExpertReview, ModelVersion, ProblemRef, RunProvenance,
                                                    ReferenceComparison, ValidationCase)

PROFILE = PROFILE_PRESETS["academic_cfd"].with_changes(decision="report the comparison", target_error_pct=2.0)


def _case() -> ValidationCase:
    case = ValidationCase.open(profile=PROFILE, model=ModelVersion("fno_darcy", "v1", checkpoint_sha256="a" * 64),
                               problem=ProblemRef("Darcy 2-D", pde="laplace", geometry="unit square"),
                               requested_use="benchmark comparison")
    case = case.with_checks([CheckVerdict("PDE residual", "verification", True, 2e-3, 1e-2),
                             CheckVerdict("boundary conditions", "physics", False, 4e-2, 1e-3, "walls move")])
    case = case.with_reference(ReferenceComparison("ref01", "numerical reference", "flux at the outlet", 3.4,
                                                  source="OpenFOAM, 1e6 cells", artefact_sha256="d" * 64))
    case = case.with_provenance(RunProvenance(tool="pinneapple check", tool_version="1.2", data_sha256="c" * 64,
                                          seeds=(0,)))
    return case.with_review(ExpertReview.record("eng. A", "approved", "only for the fixed geometry",
                                                scope={"uses": ["benchmark comparison"]}))


def test_the_dossier_says_it_is_not_a_certification():
    d = ValidationDossier.from_case(_case())
    for text in (d.render_text(), d.render_markdown(), d.render_html()):
        assert "Not a certification" in text
    assert "Not a certification" in DISCLAIMER and "does not extend beyond that scope" in DISCLAIMER


def test_the_text_has_every_component_of_the_loop():
    text = ValidationDossier.from_case(_case()).render_text()
    for section in ("Declared use", "Findings", "Checks recorded", "Reference evidence", "Applicability",
                    "Uncertainty", "Maturity", "Expert decisions", "Limitations", "RunProvenance", "Revision history"):
        assert f"== {section} ==" in text
    assert "3.4 %" in text and "tolerance 2 %" in text           # the measured error against the declared criterion
    assert "boundary conditions" in text and "walls move" in text  # a failing check is listed with its message
    assert "What happened:" in text and "What is still unknown:" in text


def test_a_failed_case_is_printed_as_failed_with_its_reason():
    d = ValidationDossier.from_case(_case())
    assert d.assessment.status == "failed"
    text = d.render_text()
    assert "Status: FAILED" in text and d.assessment.reason.split(":")[0] in text


def test_the_html_is_self_contained_and_escapes_content():
    case = _case()
    hostile = case.with_limitations(["<script>alert(1)</script>"])
    body = ValidationDossier.from_case(hostile).render_html()
    assert "<script>" not in body and "&lt;script&gt;" in body
    assert body.startswith("<!doctype html>") and "http://" not in body and "https://" not in body


def test_the_dossier_can_be_written_and_read_back(tmp_path):
    d = ValidationDossier.from_case(_case())
    for fmt, suffix in (("html", ".html"), ("md", ".md"), ("txt", ".txt")):
        p = d.write(tmp_path / f"case{suffix}", fmt=fmt)
        assert p.read_text(encoding="utf-8").strip()


def test_the_dict_form_carries_the_history_and_the_tasks():
    row = ValidationDossier.from_case(_case()).to_dict()
    assert row["schema"] == "pinneapple-validation-dossier/v1"
    assert row["history"] and row["improvement_tasks"]
    assert row["assessment"]["status"] == "failed" and row["profile"]["objective"] == "academic"
    assert {t["priority"] for t in row["improvement_tasks"]} <= {"high", "medium", "low"}
