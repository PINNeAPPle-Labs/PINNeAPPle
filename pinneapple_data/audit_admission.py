"""Strict admission of existing dataset audit cards.

Audit cards must be produced by a trusted auditor for the exact dataset
being consumed. This helper does not authenticate or bind cards to bytes.
"""
from collections.abc import Mapping, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetAdmission:
    approved: bool
    reasons: tuple[str, ...]


def assess_audit(card: Mapping, *, required_checks: Iterable[str]) -> DatasetAdmission:
    """Require explicit passes, including physics and split checks as applicable.

    The caller defines the mandatory checks for the intended training use.
    A failed check anywhere in the card is a veto, even if not mandatory.
    """
    required = tuple(required_checks)
    if not required or any(not isinstance(c, str) or not c.strip() for c in required):
        raise ValueError("Explicit mandatory check IDs are required")
    reasons = []
    if card.get("schema") != "dataset-audit/v1":
        reasons.append("unsupported audit schema")
    if card.get("decision") != "pass":
        reasons.append("audit decision is not pass")
    checks = card.get("checks")
    if not isinstance(checks, list):
        return DatasetAdmission(False, tuple(reasons + ["missing audit checks"]))
    by_id = {}
    for check in checks:
        if not isinstance(check, Mapping) or not isinstance(check.get("id"), str):
            reasons.append("malformed check")
            continue
        key = check["id"]
        if key in by_id:
            reasons.append(f"{key}: duplicate check")
        by_id[key] = check.get("status")
        if check.get("status") == "fail":
            reasons.append(f"{key}: failed")
    for key in required:
        if by_id.get(key) != "pass":
            reasons.append(f"{key}: mandatory check did not pass")
    return DatasetAdmission(not reasons, tuple(reasons))
