"""Minimal, dependency-free engineering requirements and evidence traceability.

This module models *engineering* requirements, not Python runtime dependencies.
It deliberately never marks a requirement verified without passing evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Requirement:
    id: str
    statement: str
    source: str
    unit: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.statement.strip() or not self.source.strip():
            raise ValueError("Requirement id, statement and source must be non-empty")
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ValueError("lower_bound cannot exceed upper_bound")
        if (self.lower_bound is not None or self.upper_bound is not None) and not self.unit:
            raise ValueError("Numeric bounds require an explicit unit")


@dataclass(frozen=True)
class Evidence:
    id: str
    requirement_id: str
    artifact_uri: str
    check: str
    passed: bool
    measured_value: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if not all((self.id.strip(), self.requirement_id.strip(), self.artifact_uri.strip(), self.check.strip())):
            raise ValueError("Evidence identifiers, artifact and check must be non-empty")


@dataclass
class EngineeringTrace:
    requirements: dict[str, Requirement] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)

    def add_requirement(self, requirement: Requirement) -> None:
        if requirement.id in self.requirements:
            raise ValueError(f"Duplicate requirement: {requirement.id}")
        self.requirements[requirement.id] = requirement

    def add_evidence(self, item: Evidence) -> None:
        if item.id in self.evidence:
            raise ValueError(f"Duplicate evidence: {item.id}")
        if item.requirement_id not in self.requirements:
            raise ValueError(f"Unknown requirement: {item.requirement_id}")
        req = self.requirements[item.requirement_id]
        if item.measured_value is not None:
            if req.unit != item.unit:
                raise ValueError(f"Unit mismatch for {req.id}: expected {req.unit!r}, got {item.unit!r}")
            if req.lower_bound is not None and item.measured_value < req.lower_bound and item.passed:
                raise ValueError("Evidence cannot pass below the lower bound")
            if req.upper_bound is not None and item.measured_value > req.upper_bound and item.passed:
                raise ValueError("Evidence cannot pass above the upper bound")
        self.evidence[item.id] = item

    def status(self, requirement_id: str) -> str:
        if requirement_id not in self.requirements:
            raise KeyError(requirement_id)
        items = [e for e in self.evidence.values() if e.requirement_id == requirement_id]
        if not items:
            return "unverified"
        return "failed" if any(not e.passed for e in items) else "verified"

    def report(self) -> dict[str, Any]:
        return {
            "requirements": [
                {
                    "id": r.id,
                    "statement": r.statement,
                    "source": r.source,
                    "unit": r.unit,
                    "lower_bound": r.lower_bound,
                    "upper_bound": r.upper_bound,
                    "status": self.status(r.id),
                    "evidence_ids": [e.id for e in self.evidence.values() if e.requirement_id == r.id],
                }
                for r in self.requirements.values()
            ],
            "evidence": [vars(e).copy() for e in self.evidence.values()],
        }
