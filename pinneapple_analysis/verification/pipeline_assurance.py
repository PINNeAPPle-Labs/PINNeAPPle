"""Evidence-first scientific assurance for pipeline artifacts.

Pure Python, deliberately independent of PINNeAPPle runtime. Every check has an
explicit applicability predicate and an evidence state; missing checks never pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any, Mapping


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    NOT_RUN = "not_run"
    NOT_APPLICABLE = "not_applicable"


STAGES = (
    "specification", "geometry", "mesh", "boundary_conditions", "solver",
    "field_extraction", "dataset", "training", "inverse", "export",
    "serving", "digital_twin", "engineering_decision",
)


@dataclass(frozen=True)
class Check:
    name: str
    stage: str
    status: Status
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    assumptions: tuple[str, ...] = ()
    tolerance: float | None = None
    suggested_action: str | None = None


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    stage: str
    parents: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    content_sha256: str | None = None


def fingerprint(payload: Any) -> str:
    return sha256(json.dumps(payload, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def numeric_close(name: str, stage: str, actual: float, expected: float,
                  *, atol: float = 1e-8, rtol: float = 1e-5,
                  assumptions: tuple[str, ...] = ()) -> Check:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    if atol < 0 or rtol < 0:
        raise ValueError("Tolerances must be nonnegative")
    if not all(math.isfinite(float(x)) for x in (actual, expected)):
        return Check(name, stage, Status.INCONCLUSIVE, reason="Nonfinite measurement")
    error = abs(actual - expected)
    threshold = atol + rtol * abs(expected)
    return Check(name, stage, Status.PASS if error <= threshold else Status.FAIL,
                 evidence={"actual": actual, "expected": expected, "absolute_error": error,
                           "threshold": threshold}, assumptions=assumptions, tolerance=threshold)


def conservation(name: str, stage: str, incoming: float, outgoing: float,
                 *, accumulation: float = 0.0, sources: float = 0.0,
                 atol: float = 1e-8, rtol: float = 1e-5) -> Check:
    # incoming + sources = outgoing + accumulation
    return numeric_close(name, stage, incoming + sources, outgoing + accumulation,
                         atol=atol, rtol=rtol,
                         assumptions=("Consistent signs, units and control volume",))


def metamorphic(name: str, stage: str, original: list[float], transformed: list[float],
                expected: list[float], *, atol: float = 1e-8, rtol: float = 1e-5,
                assumptions: tuple[str, ...]) -> Check:
    if not assumptions:
        return Check(name, stage, Status.INCONCLUSIVE,
                     reason="Transformation applicability assumptions are required")
    if len(transformed) != len(expected) or len(original) == 0:
        return Check(name, stage, Status.INCONCLUSIVE, reason="Incompatible or empty vectors")
    if not transformed:
        return Check(name, stage, Status.INCONCLUSIVE, reason="Empty transformed output")
    checks = [numeric_close(name, stage, a, b, atol=atol, rtol=rtol)
              for a, b in zip(transformed, expected)]
    if any(c.status == Status.INCONCLUSIVE for c in checks):
        status = Status.INCONCLUSIVE
    else:
        status = Status.FAIL if any(c.status == Status.FAIL for c in checks) else Status.PASS
    return Check(name, stage, status, assumptions=assumptions,
                 evidence={"samples": len(checks),
                           "max_error": max((c.evidence.get("absolute_error", 0) for c in checks), default=0),
                           "failures": sum(c.status == Status.FAIL for c in checks)})


def assess(artifacts: list[Artifact], checks: list[Check]) -> dict[str, Any]:
    ids = [a.artifact_id for a in artifacts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate artifact IDs")
    by_id = {a.artifact_id: a for a in artifacts}
    for a in artifacts:
        if a.stage not in STAGES:
            raise ValueError(f"Unknown stage: {a.stage}")
        if any(p not in by_id for p in a.parents):
            raise ValueError(f"Missing parent for {a.artifact_id}")
    for c in checks:
        if c.stage not in STAGES:
            raise ValueError(f"Unknown check stage: {c.stage}")
    visited: set[str] = set()
    active: set[str] = set()
    def visit(node: str) -> None:
        if node in active:
            raise ValueError("Cycle in artifact lineage")
        if node in visited:
            return
        active.add(node)
        for parent in by_id[node].parents:
            visit(parent)
        active.remove(node)
        visited.add(node)
    for node in ids:
        visit(node)
    counts = {s.value: sum(c.status == s for c in checks) for s in Status}
    executed = counts["pass"] + counts["fail"] + counts["inconclusive"]
    coverage = executed / len(checks) if checks else 0.0
    if counts["fail"]:
        verdict = "fail"
    elif counts["inconclusive"] or counts["not_run"] or not executed:
        verdict = "inconclusive"
    else:
        verdict = "pass"
    return {"verdict": verdict, "coverage": coverage, "counts": counts,
            "first_failing_stage": next((s for s in STAGES if any(
                c.stage == s and c.status == Status.FAIL for c in checks)), None),
            "checks": [asdict(c) for c in checks],
            "artifacts": [asdict(a) for a in artifacts],
            "limitations": ["Passing internal checks does not establish empirical validity."]}


FAILURE_SIGNATURES = {
    "scale_factor": ("units", "normalization", "field_extraction"),
    "boundary_localized": ("boundary_conditions", "mesh", "training"),
    "temporal_growth": ("time_integration", "distribution_shift", "sensor_sync"),
    "geometry_ood": ("geometry", "training_distribution", "topology"),
    "interface_localized": ("material_interface", "contact", "field_mapping"),
    "export_disagreement": ("export", "runtime", "preprocessing"),
    "inverse_nonunique": ("identifiability", "sensor_placement", "noise"),
    "conservation_violation": ("boundary_conditions", "solver", "field_extraction"),
}


def diagnose(signatures: list[str], checks: list[Check]) -> dict[str, Any]:
    hypotheses = sorted({h for signature in signatures
                         for h in FAILURE_SIGNATURES.get(signature, ())})
    return {"hypotheses": hypotheses,
            "unknown_signatures": [s for s in signatures if s not in FAILURE_SIGNATURES],
            "confirmed_violations": [c.name for c in checks if c.status == Status.FAIL],
            "next_step": "Run controlled discriminating experiments before applying corrections.",
            "auto_repair_authorized": False}
