"""Stored validation reports of the zoo models (#706): one ``pp.validate`` report per model, kept with the package,
shown in the model's Hub card, and compared with a fresh report before a release.

``build(name)`` writes ``validation_reports/<name>.json``; ``stored(name)`` reads it; ``check_release(name)`` validates
the model as it is now and diffs the result against the stored report: a check that passed and no longer passes is
a regression and blocks the release (``scripts/zoo/validation_reports.py check``, run by the zoo publishing
workflow before any upload).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["REPORT_DIR", "build", "stored", "check_release", "card_section"]

REPORT_DIR = Path(__file__).resolve().parent / "validation_reports"
LEVEL = "standard"


def _report(model):
    import warnings

    from pinneapple_analysis.validation.validate import validate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return validate(model, level=LEVEL, seed=0)


def build(name: str):
    """Validate the zoo model ``name`` and store its report."""
    from pinneapple_hub import zoo
    rep = _report(zoo.load(name))
    REPORT_DIR.mkdir(exist_ok=True)
    rep.to_json(REPORT_DIR / f"{name}.json")
    return rep


def stored(name: str) -> dict[str, Any] | None:
    p = REPORT_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def check_release(name: str, model=None):
    """``(diff, fresh report)`` of the model as it is now (or ``model``, e.g. a new version) against its stored
    report; ``diff.regressed`` blocks the release. Without a stored report the diff is against an empty one."""
    from pinneapple_analysis.validation.report_artifact import diff
    from pinneapple_hub import zoo
    fresh = _report(model if model is not None else zoo.load(name))
    old = stored(name) or {"model": f"{name} (no stored report)", "decision": "pass", "checks": []}
    return diff(old, fresh), fresh


def card_section(name: str) -> str:
    """The markdown block of the stored report for the model's Hub card ("" when none is stored)."""
    from pinneapple_analysis.validation.report_artifact import model_card_section
    rep = stored(name)
    return model_card_section(rep) if rep else ""
