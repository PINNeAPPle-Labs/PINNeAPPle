"""Validator protocol, shared context and registry.

A validator answers one question about one artifact and returns evidence
for its answer. Two conventions are load-bearing:

* A validator that does not apply returns ``NOT_APPLICABLE`` **with a
  reason**, never ``PASS``. A pipeline that counted inapplicable checks as
  passes would award Q2 to an empty artifact.
* A validator that would apply but was not given the data it needs returns
  ``BLOCKED`` (or ``NOT_RUN`` if the caller disabled it) -- also never
  ``PASS``. The distinction the spec asks for between "not applicable",
  "not run" and "passed" survives all the way into the report.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..report import Finding, Status
from ..spec import PhysicalSpecification

Mesh = Tuple[np.ndarray, np.ndarray]


@dataclass
class SensorReading:
    """One sensor: where, what unit, and the series over the artifact's time axis."""

    id: str
    position: Tuple[float, float, float]
    unit: str
    quantity: str
    series: np.ndarray
    holdout: bool = False
    position_unit: str = "m"


@dataclass
class ValidationContext:
    """Everything a validator may look at. Absent data means "not supplied"."""

    spec: PhysicalSpecification
    meshes: Dict[str, Mesh] = field(default_factory=dict)
    fields: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> (V,) or (T, V)
    field_units: Dict[str, str] = field(default_factory=dict)
    times: Optional[np.ndarray] = None
    time_unit: str = "s"
    sensors: List[SensorReading] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    model: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def field_array(self, name: str) -> Optional[np.ndarray]:
        value = self.fields.get(name)
        return None if value is None else np.asarray(value)


@dataclass
class ValidatorResult:
    name: str
    status: Status
    findings: List[Finding] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is Status.PASS

    def to_findings(self) -> List[Finding]:
        """Findings plus a summary finding carrying the validator's own status."""
        out = list(self.findings)
        if not out or not any(f.status is self.status for f in out):
            out.append(Finding(id=f"validator.{self.name}", status=self.status,
                               message=_summary_message(self), severity=_severity_for(self.status),
                               evidence=dict(self.evidence), limitations=list(self.limitations)))
        return out


def _severity_for(status: Status) -> str:
    if status is Status.PASS:
        return "info"
    if status in (Status.NOT_APPLICABLE, Status.NOT_RUN):
        return "warning"
    return "error"


def _summary_message(result: ValidatorResult) -> str:
    problems = [f.message for f in result.findings if f.blocks]
    if problems:
        return f"{result.name}: {len(problems)} problem(s): " + "; ".join(problems[:3])
    if result.status is Status.NOT_APPLICABLE:
        return str(result.evidence.get("reason", f"{result.name}: not applicable"))
    return f"{result.name}: passed"


Validator = Callable[[ValidationContext], ValidatorResult]

_REGISTRY: Dict[str, Validator] = {}


def register(name: str) -> Callable[[Validator], Validator]:
    def decorator(fn: Validator) -> Validator:
        if name in _REGISTRY:
            raise ValueError(f"validator '{name}' is already registered")
        _REGISTRY[name] = fn
        return fn
    return decorator


def available() -> List[str]:
    return sorted(_REGISTRY)


def get(name: str) -> Validator:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown validator '{name}'; registered: {available()}") from None


def run(name: str, ctx: ValidationContext) -> ValidatorResult:
    started = time.monotonic()
    result = get(name)(ctx)
    result.duration_s = time.monotonic() - started
    if result.name != name:
        result.name = name
    return result


def not_applicable(name: str, reason: str, **evidence: Any) -> ValidatorResult:
    return ValidatorResult(
        name=name, status=Status.NOT_APPLICABLE,
        evidence={"reason": reason, **evidence},
        findings=[Finding(id=f"validator.{name}.not_applicable", status=Status.NOT_APPLICABLE,
                          severity="warning", message=reason, evidence=evidence)],
    )


def blocked(name: str, reason: str, **evidence: Any) -> ValidatorResult:
    return ValidatorResult(
        name=name, status=Status.BLOCKED,
        evidence={"reason": reason, **evidence},
        findings=[Finding(id=f"validator.{name}.blocked", status=Status.BLOCKED,
                          severity="error", message=reason, evidence=evidence,
                          justification="the check applies but lacks the data or permission to run")],
    )


def _load_builtin_validators() -> None:
    """Import the built-in validators for their registration side effects."""
    from . import geometry, numeric, physics, sensors, temporal, units  # noqa: F401
    try:  # optional: only needed when a trained model is being gated
        from . import guardrail as _guardrail  # noqa: F401
    except Exception:  # pragma: no cover - torch/pinneapple_physics absent
        pass


_load_builtin_validators()
