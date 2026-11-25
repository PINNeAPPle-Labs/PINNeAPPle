"""Bridge solver/VeriPhysics check outputs to engineering evidence.

Only explicitly successful checks can verify a requirement. Missing or malformed
results fail closed; the adapter does not execute solvers or claim certification.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .engineering_trace import EngineeringTrace, Evidence


def record_check_result(
    trace: EngineeringTrace,
    *,
    requirement_id: str,
    evidence_id: str,
    artifact_uri: str,
    result: Mapping[str, Any],
) -> Evidence:
    """Record a check result with explicit success, units and provenance.

    Expected input: {"check": str, "passed": bool, "value": number?,
                     "unit": str?}. A missing or non-boolean passed field
    is rejected, never silently converted to truthy.
    """
    if not isinstance(result, Mapping):
        raise TypeError("Check result must be a mapping")
    passed = result.get("passed")
    if type(passed) is not bool:
        raise ValueError("Check result requires an explicit boolean 'passed'")
    check = result.get("check")
    if not isinstance(check, str) or not check.strip():
        raise ValueError("Check result requires a non-empty 'check'")
    value = result.get("value")
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Measured value must be numeric")
        if not math.isfinite(value):
            raise ValueError("Measured value must be finite")
    unit = result.get("unit")
    if unit is not None and not isinstance(unit, str):
        raise ValueError("Unit must be a string")
    evidence = Evidence(
        id=evidence_id,
        requirement_id=requirement_id,
        artifact_uri=artifact_uri,
        check=check,
        passed=passed,
        measured_value=value,
        unit=unit,
    )
    trace.add_evidence(evidence)
    return evidence


def release_gate(trace: EngineeringTrace) -> dict[str, Any]:
    """Summarize a fail-closed release gate for all requirements."""
    statuses = {key: trace.status(key) for key in trace.requirements}
    return {
        "approved": bool(statuses) and all(v == "verified" for v in statuses.values()),
        "statuses": statuses,
        "unverified": [k for k, v in statuses.items() if v == "unverified"],
        "failed": [k for k, v in statuses.items() if v == "failed"],
    }
