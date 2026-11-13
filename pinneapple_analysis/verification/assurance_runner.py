"""Execute deterministic assurance checks for an artifact DAG.

This is an opt-in runner for declared artifacts; it does not silently
claim that a solver or physical model has been independently validated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .artifact_contracts import check_artifact_metadata, check_lineage_compatibility
from .assurance_policy import RequiredCheck, evaluate_gate
from .diagnostic_engine import localize
from .pipeline_assurance import Artifact, Check, Status, assess


@dataclass(frozen=True)
class PipelineReport:
    assessment: dict
    decision_gate: dict
    diagnosis: dict
    executed_checks: tuple[dict, ...]
    stage_summary: dict


def run_artifact_assurance(
    artifacts: Sequence[Artifact],
    required: Sequence[RequiredCheck],
    *,
    additional_checks: Sequence[Check] = (),
    signatures: Sequence[str] = (),
    empirical_validation_required: bool = False,
    empirical_validation_present: bool = False,
) -> PipelineReport:
    """Run declared metadata and lineage checks, then fail-closed decision gate.

    A report is only about the checks actually executed. Required checks
    for solver/physics validation remain missing until supplied by a real
    independent verifier.
    """
    # Validate the complete DAG before producing any findings.
    assess(list(artifacts), [])
    by_id = {a.artifact_id: a for a in artifacts}
    checks = list(additional_checks)
    for artifact in artifacts:
        checks.extend(check_artifact_metadata(artifact))
        for parent_id in artifact.parents:
            checks.extend(check_lineage_compatibility(by_id[parent_id], artifact))
    # Preserve evidence of duplicate checks rather than overwriting it.
    assessment = assess(list(artifacts), checks)
    gate = evaluate_gate(
        checks, required,
        empirical_validation_required=empirical_validation_required,
        empirical_validation_present=empirical_validation_present,
    )
    diagnosis = localize(artifacts, checks, signatures)
    summary = {
        stage: {status.value: sum(c.stage == stage and c.status == status for c in checks)
                for status in Status}
        for stage in sorted({a.stage for a in artifacts} | {c.stage for c in checks})
    }
    return PipelineReport(
        assessment=assessment, decision_gate=gate, diagnosis=diagnosis,
        executed_checks=tuple(asdict(c) for c in checks),
        stage_summary=summary,
    )
