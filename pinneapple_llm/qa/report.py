"""Reproducible quality report -- the machine-readable output of every run.

Two design commitments, both from the same failure mode (a pipeline that
says PASS because nothing ran):

1. There is no boolean "ok" anywhere. Every step carries one of
   :class:`Status`, and ``NOT_RUN``/``NOT_APPLICABLE`` are never folded
   into ``PASS``. A consumer that only wants a yes/no has to state which
   steps it requires.
2. The report is self-contained: it carries the spec it was given, the
   hashes of the artifacts it saw, the versions of the software, and the
   exact command that reproduces the run. Reading it does not require the
   conversation that produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = "pinneapple-llm-quality/1"


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"          # could not be attempted (missing input/permission)
    NOT_RUN = "NOT_RUN"          # applicable, but not executed in this run
    NOT_APPLICABLE = "NOT_APPLICABLE"  # genuinely does not apply to this spec

    @property
    def is_positive(self) -> bool:
        return self is Status.PASS


SEVERITIES = ("error", "warning", "info")


@dataclass
class Finding:
    """One observed fact, with the evidence that produced it."""

    id: str
    status: Status
    message: str
    severity: str = "error"
    evidence: Dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    limitations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")

    @property
    def blocks(self) -> bool:
        return self.severity == "error" and self.status is not Status.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "status": self.status.value, "severity": self.severity,
            "message": self.message, "evidence": _jsonable(self.evidence),
            "justification": self.justification, "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Finding":
        return cls(
            id=d["id"], status=Status(d["status"]), message=d["message"],
            severity=d.get("severity", "error"), evidence=dict(d.get("evidence", {})),
            justification=d.get("justification", ""), limitations=list(d.get("limitations", [])),
        )


@dataclass
class StepOutcome:
    """The result of one pipeline stage."""

    step: str
    status: Status
    findings: List[Finding] = field(default_factory=list)
    duration_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def failure_count(self) -> int:
        return sum(1 for f in self.findings if f.blocks)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning" and f.status is not Status.PASS)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step, "status": self.status.value,
            "findings": [f.to_dict() for f in self.findings],
            "duration_s": round(float(self.duration_s), 6),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepOutcome":
        return cls(
            step=d["step"], status=Status(d["status"]),
            findings=[Finding.from_dict(f) for f in d.get("findings", [])],
            duration_s=float(d.get("duration_s", 0.0)), metadata=dict(d.get("metadata", {})),
        )


@dataclass
class ArtifactRecord:
    name: str
    kind: str
    path: Optional[str] = None
    sha256: str = ""
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, name: str, path: str, kind: str = "", **metadata: Any) -> "ArtifactRecord":
        with open(path, "rb") as fh:
            blob = fh.read()
        return cls(name=name, kind=kind or os.path.splitext(path)[1].lstrip("."),
                   path=path, sha256=hashlib.sha256(blob).hexdigest(), size_bytes=len(blob),
                   metadata=metadata)

    @classmethod
    def from_value(cls, name: str, value: Any, kind: str = "value", **metadata: Any) -> "ArtifactRecord":
        blob = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return cls(name=name, kind=kind, path=None, sha256=hashlib.sha256(blob).hexdigest(),
                   size_bytes=len(blob), metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "path": self.path, "sha256": self.sha256,
            "size_bytes": self.size_bytes, "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactRecord":
        return cls(name=d["name"], kind=d.get("kind", ""), path=d.get("path"),
                   sha256=d.get("sha256", ""), size_bytes=int(d.get("size_bytes", 0)),
                   metadata=dict(d.get("metadata", {})))


@dataclass
class QualityReport:
    """Everything one pipeline run observed, and what it is allowed to claim."""

    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    spec_id: str = ""
    spec_hash: str = ""
    criteria_hash: str = ""
    provider: str = ""
    model: str = ""
    steps: List[StepOutcome] = field(default_factory=list)
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    quality_level: str = "Q0"
    level_reasons: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    operational_evidence: Optional[Dict[str, Any]] = None
    reproduce: str = ""

    # -- construction -----------------------------------------------------
    def add_step(self, outcome: StepOutcome) -> StepOutcome:
        self.steps.append(outcome)
        return outcome

    def record(self, step: str, status: Status, findings: Optional[Iterable[Finding]] = None,
               duration_s: float = 0.0, **metadata: Any) -> StepOutcome:
        return self.add_step(StepOutcome(step=step, status=status,
                                         findings=list(findings or []),
                                         duration_s=duration_s, metadata=dict(metadata)))

    def artifact(self, record: ArtifactRecord) -> ArtifactRecord:
        self.artifacts.append(record)
        return record

    # -- queries ----------------------------------------------------------
    def step(self, name: str) -> Optional[StepOutcome]:
        for s in self.steps:
            if s.step == name:
                return s
        return None

    def status_of(self, name: str) -> Status:
        s = self.step(name)
        return Status.NOT_RUN if s is None else s.status

    def findings(self, *, status: Optional[Status] = None) -> List[Finding]:
        out = [f for s in self.steps for f in s.findings]
        return out if status is None else [f for f in out if f.status is status]

    @property
    def blocking_findings(self) -> List[Finding]:
        return [f for f in self.findings() if f.blocks]

    @property
    def overall_status(self) -> Status:
        if any(s.status is Status.FAIL for s in self.steps):
            return Status.FAIL
        if any(s.status is Status.BLOCKED for s in self.steps):
            return Status.BLOCKED
        if not self.steps:
            return Status.NOT_RUN
        if any(s.status is Status.NOT_RUN for s in self.steps):
            return Status.NOT_RUN
        if all(s.status in (Status.PASS, Status.NOT_APPLICABLE) for s in self.steps):
            return Status.PASS
        return Status.FAIL

    def summary(self) -> str:
        lines = [
            f"Quality report {self.run_id} ({self.created_at})",
            f"  level: {self.quality_level}   overall: {self.overall_status.value}",
            f"  spec: {self.spec_id or '(none)'}  hash={self.spec_hash[:12]}",
        ]
        for s in self.steps:
            lines.append(f"  [{s.status.value:<14}] {s.step} "
                         f"({len(s.findings)} finding(s), {s.failure_count} blocking)")
        if self.level_reasons:
            lines.append("  level rationale:")
            lines.extend(f"    - {r}" for r in self.level_reasons)
        if self.missing_information:
            lines.append("  missing information: " + ", ".join(self.missing_information))
        return "\n".join(lines)

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "criteria_hash": self.criteria_hash,
            "provider": self.provider,
            "model": self.model,
            "overall_status": self.overall_status.value,
            "quality_level": self.quality_level,
            "level_reasons": list(self.level_reasons),
            "steps": [s.to_dict() for s in self.steps],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "limitations": list(self.limitations),
            "assumptions": list(self.assumptions),
            "missing_information": list(self.missing_information),
            "environment": _jsonable(self.environment),
            "operational_evidence": _jsonable(self.operational_evidence),
            "reproduce": self.reproduce,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QualityReport":
        return cls(
            run_id=d.get("run_id", ""), created_at=d.get("created_at", ""),
            spec_id=d.get("spec_id", ""), spec_hash=d.get("spec_hash", ""),
            criteria_hash=d.get("criteria_hash", ""), provider=d.get("provider", ""),
            model=d.get("model", ""),
            steps=[StepOutcome.from_dict(s) for s in d.get("steps", [])],
            artifacts=[ArtifactRecord.from_dict(a) for a in d.get("artifacts", [])],
            quality_level=d.get("quality_level", "Q0"),
            level_reasons=list(d.get("level_reasons", [])),
            limitations=list(d.get("limitations", [])),
            assumptions=list(d.get("assumptions", [])),
            missing_information=list(d.get("missing_information", [])),
            environment=dict(d.get("environment", {})),
            operational_evidence=d.get("operational_evidence"),
            reproduce=d.get("reproduce", ""),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
        return path

    @classmethod
    def read(cls, path: str) -> "QualityReport":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def environment_snapshot() -> Dict[str, Any]:
    """Versions and platform, so a report can be traced to an environment."""
    versions: Dict[str, str] = {}
    for package in ("pinneapple", "numpy", "torch", "pinneapple_twin3d"):
        try:
            from importlib.metadata import version
            versions[package] = version(package)
        except Exception:
            versions[package] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "tolist"):  # numpy scalar/array
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
