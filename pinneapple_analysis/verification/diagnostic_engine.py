"""Evidence-grounded failure localization and controlled remediation planning.

A causal hypothesis is never promoted to a confirmed cause merely from
correlation or a failure signature. All decisions retain missing evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .pipeline_assurance import Artifact, Check, Status, STAGES, FAILURE_SIGNATURES, assess


@dataclass(frozen=True)
class Hypothesis:
    cause: str
    supporting_checks: tuple[str, ...] = ()
    contradicting_checks: tuple[str, ...] = ()
    discriminating_tests: tuple[str, ...] = ()
    status: str = "candidate"


@dataclass(frozen=True)
class RepairProposal:
    action: str
    stage: str
    risk: str
    requires_approval: bool
    validation_checks: tuple[str, ...]
    permitted_to_execute: bool = False


_TESTS = {
    "units": ("compare raw and transformed magnitudes", "verify dimensional conversions"),
    "normalization": ("roundtrip normalize/inverse", "compare training vs serving preprocess"),
    "field_extraction": ("compare source field and exported field at identical coordinates",),
    "boundary_conditions": ("inspect CAD face to patch mapping", "evaluate independent boundary residual"),
    "mesh": ("refine spatial mesh and compare quantity of interest",),
    "training": ("re-evaluate residuals on independent samples", "hold out geometry families"),
    "time_integration": ("time-step refinement study",),
    "distribution_shift": ("evaluate out-of-distribution envelope",),
    "sensor_sync": ("compare sensor timestamps and lag",),
    "geometry": ("verify topology, dimensions and rigid transformation equivariance",),
    "training_distribution": ("compare geometric train/test support",),
    "topology": ("hold out topology families",),
    "material_interface": ("verify flux and traction jump conditions",),
    "contact": ("verify gap, complementarity and contact traction",),
    "field_mapping": ("compare coordinate and component transformations",),
    "export": ("compare original framework with exported runtime",),
    "runtime": ("compare runtime versions and precision",),
    "preprocessing": ("roundtrip input schema and preprocessing",),
    "identifiability": ("evaluate Jacobian rank and profile likelihood",),
    "sensor_placement": ("evaluate sensitivity of independent sensor configurations",),
    "noise": ("repeat synthetic recovery with controlled noise",),
    "solver": ("evaluate conservation on independent control volumes",),
}


def localize(artifacts: Sequence[Artifact], checks: Sequence[Check],
             signatures: Sequence[str] = ()) -> dict[str, Any]:
    """Locate observed violations; report candidate origins, never assert root cause."""
    result = assess(list(artifacts), list(checks))
    failing = [c for c in checks if c.status == Status.FAIL]
    by_stage = {stage: [c.name for c in failing if c.stage == stage] for stage in STAGES}
    suspected = sorted({cause for signature in signatures
                        for cause in FAILURE_SIGNATURES.get(signature, ())})
    hypotheses = [Hypothesis(cause=cause, supporting_checks=tuple(
        c.name for c in failing if cause in c.name.lower()),
        discriminating_tests=_TESTS.get(cause, ("design a controlled independent experiment",)))
        for cause in suspected]
    unknown = [s for s in signatures if s not in FAILURE_SIGNATURES]
    return {
        "verdict": result["verdict"],
        "first_observed_failure_stage": result["first_failing_stage"],
        "violations_by_stage": {k: v for k, v in by_stage.items() if v},
        "candidate_causes": [asdict(h) for h in hypotheses],
        "unrecognized_signatures": unknown,
        "root_cause_confirmed": False,
        "missing_evidence": [c.name for c in checks if c.status in (Status.NOT_RUN, Status.INCONCLUSIVE)],
        "limitations": [
            "First observed failure is not necessarily the stage where the error originated.",
            "Failure signatures suggest tests; they do not prove causality.",
        ],
    }


def propose_repairs(checks: Sequence[Check]) -> list[RepairProposal]:
    """No automatic changes to equations, materials, BCs, geometry or safety."""
    proposals: list[RepairProposal] = []
    for c in checks:
        if c.status != Status.FAIL:
            continue
        action = c.suggested_action or "Investigate and reproduce the failure"
        proposals.append(RepairProposal(
            action=action, stage=c.stage, risk="requires_review",
            requires_approval=True,
            validation_checks=(c.name, "independent_verification", "downstream_regression"),
        ))
    return proposals


def compare_intervention(before: Sequence[Check], after: Sequence[Check],
                         *, changed_artifacts: Sequence[str]) -> dict[str, Any]:
    """A/B comparison: improvement is not proof of a physical root cause."""
    baseline = {(c.stage, c.name): c for c in before}
    updated = {(c.stage, c.name): c for c in after}
    shared = baseline.keys() & updated.keys()
    fixed = sorted(f"{stage}:{name}" for stage, name in shared
                   if baseline[stage, name].status == Status.FAIL
                   and updated[stage, name].status == Status.PASS)
    regressions = sorted(f"{stage}:{name}" for stage, name in shared
                         if baseline[stage, name].status == Status.PASS
                         and updated[stage, name].status != Status.PASS)
    missing = sorted(f"{stage}:{name}" for stage, name in baseline.keys() - updated.keys())
    return {"changed_artifacts": list(changed_artifacts),
            "fixed_checks": fixed, "regressions": regressions,
            "missing_rechecks": missing,
            "eligible_for_approval": bool(fixed) and not regressions and not missing,
            "causal_claim": "Not established without controlled intervention and independent evidence"}
