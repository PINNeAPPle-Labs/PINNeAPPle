"""Fail-closed assurance policy and reproducible evidence gate.

A passing subset of checks must never be presented as proof of physical validity.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

from .pipeline_assurance import Check, Status, STAGES


@dataclass(frozen=True)
class RequiredCheck:
    stage: str
    name: str
    critical: bool = True


def evaluate_gate(checks: Sequence[Check], required: Sequence[RequiredCheck],
                  *, empirical_validation_required: bool = False,
                  empirical_validation_present: bool = False) -> dict:
    """Evaluate declared required checks; missing, duplicate, and inconclusive fail closed."""
    if not required:
        return {"decision": "blocked", "reason": "No required checks configured",
                "missing": [], "failed": [], "inconclusive": [], "duplicate": [],
                "limitations": ["No evidence of physical validity established."]}
    keys = [(c.stage, c.name) for c in checks]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    evidence = {(c.stage, c.name): c for c in checks}
    missing = []
    failed = []
    inconclusive = []
    for requirement in required:
        if requirement.stage not in STAGES:
            raise ValueError(f"Unknown stage: {requirement.stage}")
        key = (requirement.stage, requirement.name)
        if key not in evidence:
            missing.append(key)
        elif evidence[key].status == Status.FAIL:
            failed.append(key)
        elif evidence[key].status != Status.PASS:
            inconclusive.append(key)
    empirical_missing = empirical_validation_required and not empirical_validation_present
    blocked = bool(missing or failed or inconclusive or duplicates or empirical_missing)
    return {
        "decision": "blocked" if blocked else "eligible_for_review",
        "missing": missing, "failed": failed, "inconclusive": inconclusive,
        "duplicate": duplicates, "empirical_validation_missing": empirical_missing,
        "review_required": True,
        "limitations": ["Eligible for review is not engineering approval or empirical validation."],
    }
