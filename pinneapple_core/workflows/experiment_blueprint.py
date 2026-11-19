"""Reusable, reviewable industrial/research workflow blueprints.

This module builds DAG specifications only; it never executes generated code.
Every executable adapter must be explicitly registered by the host application.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pinneapple_core.workflows.dag import Step, Workflow

Domain = Literal["industrial", "research"]


@dataclass(frozen=True)
class ExperimentRequest:
    objective: str
    domain: Domain
    source: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    evidence_required: bool = True

    def validate(self) -> None:
        if self.domain not in ("industrial", "research"):
            raise ValueError("domain must be industrial or research")
        if not self.objective.strip():
            raise ValueError("objective is required")
        if not isinstance(self.constraints, dict):
            raise ValueError("constraints must be a mapping")


def build_experiment_workflow(request: ExperimentRequest) -> Workflow:
    """Build an auditable execution plan, with no unsafe implicit execution.

    The 'review' nodes are intentionally declarative: the API must map them
    to authorized adapters and require human approval before real-world action.
    """
    request.validate()
    phases = (
        ("scope", "Acquire requirements and establish units, boundaries and provenance"),
        ("design", "Specify physical model, baseline, metrics and experimental design"),
        ("data", "Acquire or generate licensed data and validate schema"),
        ("simulate", "Execute reference simulation or experimental baseline"),
        ("model", "Train or calibrate candidate surrogate"),
        ("verify", "Assess numerical, physical, statistical and applicability evidence"),
        ("optimize", "Evaluate constrained candidates without operating equipment"),
        ("report", "Produce reproducible evidence dossier and limitations"),
    )
    steps = [
        Step(
            id=name,
            kind="review",
            params={
                "instruction": instruction,
                "objective": request.objective,
                "source": request.source,
                "constraints": request.constraints,
                "requires_approval": name in {"simulate", "optimize"},
            },
            needs=[phases[i - 1][0]] if i else [],
        )
        for i, (name, instruction) in enumerate(phases)
    ]
    if request.evidence_required:
        steps[-1].params["requires_verified_evidence"] = True
    return Workflow(
        name=f"{request.domain}_experiment",
        steps=steps,
        fail_fast=True,
        metadata={"domain": request.domain, "objective": request.objective,
                  "evidence_required": request.evidence_required,
                  "blueprint_version": "1.0", "execution_mode": "plan_only"},
    )
