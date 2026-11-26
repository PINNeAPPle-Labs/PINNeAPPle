"""What the pipeline produces and how it is reported.

A run is a list of :class:`StageRecord` (what each stage did, how long it took, what it wrote), a list of
:class:`VerificationResult` (one per VeriPhysics gate) and one :class:`PipelineResult` that gathers the artefacts.
Nothing here decides anything: the gates do (:mod:`pinneapple_pipeline.verification`) and the pipeline applies the
``fail_policy``.

``VerificationResult.status`` is one of five values, and the last two matter as much as the others — a gate that
could not be evaluated is never reported as one that passed:

===========================  ==========================================================================
``PASS``                     the requirement holds, measured
``WARN``                     usable within the stated limitation
``FAIL``                     the requirement does not hold
``NOT_EVALUATED``            the evidence needed was not produced (say which, in ``limitations``)
``NOT_APPLICABLE``           the gate does not apply to this kind of run
===========================  ==========================================================================
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = ["PASS", "WARN", "FAIL", "NOT_EVALUATED", "NOT_APPLICABLE", "GateStatus", "VerificationResult",
           "StageRecord", "PipelineResult", "RunStage", "STAGE_ORDER"]

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
NOT_EVALUATED, NOT_APPLICABLE = "NOT_EVALUATED", "NOT_APPLICABLE"
GateStatus = (PASS, WARN, FAIL, NOT_EVALUATED, NOT_APPLICABLE)

#: The stages of a run, in order. Skipped stages stay in the list with ``status="skipped"`` and a reason, so a
#: report always says what was not done.
STAGE_ORDER = ("problem", "data", "preprocess", "model", "training", "export", "verification")


@dataclass
class VerificationResult:
    """One gate's outcome, with the numbers it was decided on and what to do about it."""

    gate: str
    status: str = NOT_EVALUATED
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in GateStatus:
            raise ValueError(f"gate {self.gate!r}: status {self.status!r} is not one of {GateStatus}")

    @property
    def passed(self) -> bool:
        """True only for a measured pass: an unevaluated gate is not a pass."""
        return self.status == PASS

    def fail(self, why: str, *, recommendation: str = "") -> VerificationResult:
        self.status = FAIL
        self.limitations.append(why)
        if recommendation:
            self.recommendations.append(recommendation)
        return self

    def warn(self, why: str, *, recommendation: str = "") -> VerificationResult:
        if self.status != FAIL:
            self.status = WARN
        self.limitations.append(why)
        if recommendation:
            self.recommendations.append(recommendation)
        return self

    def note(self, evidence: str) -> VerificationResult:
        self.evidence.append(evidence)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"gate": self.gate, "status": self.status, "metrics": _jsonable(self.metrics),
                "thresholds": _jsonable(self.thresholds), "evidence": list(self.evidence),
                "limitations": list(self.limitations), "recommendations": list(self.recommendations)}

    def to_markdown(self) -> str:
        head = f"- **{self.gate}: {self.status}**"
        if self.metrics:
            head += " — " + ", ".join(f"{k}={_num(v)}" for k, v in self.metrics.items()
                                      if isinstance(v, (int, float)))
        lines = [head]
        lines += [f"  - {e}" for e in self.evidence]
        lines += [f"  - limitation: {x}" for x in self.limitations]
        lines += [f"  - next: {x}" for x in self.recommendations]
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationResult:
        """A gate result read back from ``report.json`` (``pp.SurrogatePipeline.from_run``)."""
        return cls(gate=str(data.get("gate", "")), status=str(data.get("status", NOT_EVALUATED)),
                   metrics=dict(data.get("metrics") or {}), thresholds=dict(data.get("thresholds") or {}),
                   evidence=list(data.get("evidence") or []), limitations=list(data.get("limitations") or []),
                   recommendations=list(data.get("recommendations") or []))


@dataclass
class StageRecord:
    """What one stage did: its status, wall time, the artefacts it produced and anything it could not do."""

    stage: str
    status: str = "pending"          # pending | running | done | skipped | failed
    seconds: float = 0.0
    artifacts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "status": self.status, "seconds": round(self.seconds, 3),
                "artifacts": {k: _jsonable(v) for k, v in self.artifacts.items()}, "notes": list(self.notes),
                "error": self.error}


@dataclass
class PipelineResult:
    """Everything a run produced. ``verification`` holds one :class:`VerificationResult` per gate that ran."""

    run_id: str
    name: str
    status: str = "pending"          # completed | rejected | failed
    problem: Any = None
    dataset: Any = None
    model: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    verification: list[VerificationResult] = field(default_factory=list)
    validated_domain: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    stages: list[StageRecord] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # ------------------------------------------------------------------ queries
    def gate(self, name: str) -> VerificationResult | None:
        return next((g for g in self.verification if g.gate == name), None)

    @property
    def failed_gates(self) -> list[VerificationResult]:
        return [g for g in self.verification if g.status == FAIL]

    @property
    def unevaluated_gates(self) -> list[VerificationResult]:
        return [g for g in self.verification if g.status == NOT_EVALUATED]

    @property
    def passed(self) -> bool:
        """The run is usable: no gate failed, and the run was not rejected by the failure policy."""
        return self.status == "completed" and not self.failed_gates

    def summary(self) -> str:
        counts = {s: sum(g.status == s for g in self.verification) for s in GateStatus}
        parts = [f"{v} {k.lower().replace('_', ' ')}" for k, v in counts.items() if v]
        return f"{self.name}: {self.status} ({', '.join(parts) or 'no gate ran'})"

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "name": self.name, "status": self.status, "created": self.created,
            "execution_time": round(self.execution_time, 3),
            "problem": _jsonable(self.problem), "metrics": _jsonable(self.metrics),
            "benchmarks": _jsonable(self.benchmarks), "validated_domain": _jsonable(self.validated_domain),
            "uncertainty": _jsonable(self.uncertainty),
            "verification": [g.to_dict() for g in self.verification],
            "stages": [s.to_dict() for s in self.stages],
            "artifacts": {k: _jsonable(v) for k, v in self.artifacts.items()},
            "recommendations": list(self.recommendations), "limitations": list(self.limitations),
        }

    def to_json(self, path: str | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=1, ensure_ascii=False)
        if path:
            from pathlib import Path
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineResult:
        """A run read back from ``report.json``: the gates, the metrics and the domain, without re-running."""
        return cls(
            run_id=str(data.get("run_id", "")), name=str(data.get("name", "")),
            status=str(data.get("status", "pending")), problem=data.get("problem"),
            metrics=dict(data.get("metrics") or {}), benchmarks=dict(data.get("benchmarks") or {}),
            verification=[VerificationResult.from_dict(g) for g in data.get("verification") or []],
            validated_domain=dict(data.get("validated_domain") or {}),
            uncertainty=dict(data.get("uncertainty") or {}),
            stages=[StageRecord(stage=str(s.get("stage", "")), status=str(s.get("status", "pending")),
                                seconds=float(s.get("seconds") or 0.0), artifacts=dict(s.get("artifacts") or {}),
                                notes=list(s.get("notes") or []), error=str(s.get("error", "")))
                    for s in data.get("stages") or []],
            artifacts=dict(data.get("artifacts") or {}),
            recommendations=list(data.get("recommendations") or []),
            limitations=list(data.get("limitations") or []),
            execution_time=float(data.get("execution_time") or 0.0), created=str(data.get("created", "")))

    def to_markdown(self) -> str:
        """The run as the page an engineer files: what ran, what it measured, where it stops being valid."""
        out = [f"# Surrogate pipeline: {self.name}", "",
               f"Run `{self.run_id}`, created {self.created}, {self.execution_time:.1f} s. "
               f"Status: **{self.status}**.", ""]
        if self.artifacts:
            out += ["## Artefacts", ""]
            out += [f"- `{k}`: {v}" for k, v in self.artifacts.items()]
            out += [""]
        out += ["## Stages", "", "| stage | status | seconds | what it wrote |", "|---|---|---|---|"]
        for s in self.stages:
            wrote = ", ".join(f"{k}={_short(v)}" for k, v in s.artifacts.items() if not _empty(v))
            out.append(f"| {s.stage} | {s.status} | {s.seconds:.2f} | {wrote or s.error or ''} |")
        out += ["", "## Verification (VeriPhysics)", ""]
        out += [g.to_markdown() for g in self.verification]
        scalar = {k: v for k, v in _flatten(self.metrics).items() if isinstance(v, (int, float, bool))}
        if scalar:
            out += ["", "## Measured metrics", "",
                    "_The full record, including the per-candidate comparison, is in `model_card.json`._", "",
                    "| metric | value |", "|---|---|"]
            out += [f"| {k} | {_num(v)} |" for k, v in scalar.items()]
        if self.limitations:
            out += ["", "## Limitations", ""] + [f"- {x}" for x in self.limitations]
        if self.recommendations:
            out += ["", "## Recommendations", ""] + [f"- {x}" for x in self.recommendations]
        if self.unevaluated_gates:
            out += ["", "## Not evaluated", ""]
            out += [f"- {g.gate}: {'; '.join(g.limitations) or 'no evidence produced'}"
                    for g in self.unevaluated_gates]
        return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------- helpers
def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _empty(v: Any) -> bool:
    return v is None or (isinstance(v, (list, tuple, dict, str)) and len(v) == 0)


def _short(v: Any) -> str:
    """An artefact path shortened to its last two components (the run folder is the same for every row)."""
    if isinstance(v, str) and ("/" in v or "\\" in v):
        parts = v.replace("\\", "/").split("/")
        return "/".join(parts[-2:])
    return _num(v)


def _num(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}" if v == v and abs(v) < 1e6 else f"{v:.3e}"
    return str(v)


def _jsonable(v: Any) -> Any:
    """A JSON-safe copy: numpy scalars/arrays, paths and dataclasses become plain values."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:                                        # pragma: no cover
            return str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "to_dict"):
        try:
            return _jsonable(v.to_dict())
        except Exception:                                        # pragma: no cover
            return str(v)
    return str(v)


#: Alias kept for readability in type hints (a stage name from :data:`STAGE_ORDER`).
RunStage = str
