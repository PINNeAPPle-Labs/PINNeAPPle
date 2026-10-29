"""Stage G: structured diagnosis and bounded automatic repair.

The loop is deliberately conservative about what "fixed" means:

* every attempt re-runs **all** validators, not just the one that failed;
* a candidate is accepted only when every validator the caller declared
  required is ``PASS`` -- compiling is not acceptance;
* the specification's criteria hash is snapshotted before the loop and
  re-checked after every attempt, so a "repair" that lowers a threshold,
  drops an invariant or changes reference data is rejected;
* an identical diff twice in a row means the loop is stuck and stops
  early instead of burning attempts;
* every attempt keeps its diagnosis, diff and findings, so a failed loop
  produces an actionable report rather than a shrug.
"""
from __future__ import annotations

import difflib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .report import Finding, Status
from .validators.base import ValidatorResult

DEFAULT_MAX_ATTEMPTS = 3


@dataclass
class Diagnosis:
    """Why an attempt was rejected, in the form a repair step can consume."""

    failed_steps: List[str]
    findings: List[Finding]
    confirmed_cause: str = ""
    probable_cause: str = ""
    constraints_to_preserve: List[str] = field(default_factory=list)
    tests_that_must_pass: List[str] = field(default_factory=list)

    def as_prompt(self) -> str:
        lines = ["The artifact was rejected."]
        for f in self.findings:
            lines.append(f"- [{f.id}] {f.status.value}: {f.message}")
            if f.evidence:
                lines.append(f"  evidence: {json.dumps(_jsonable(f.evidence), default=str)[:600]}")
        if self.confirmed_cause:
            lines.append(f"Confirmed cause: {self.confirmed_cause}")
        if self.probable_cause:
            lines.append(f"Probable cause (unconfirmed): {self.probable_cause}")
        if self.constraints_to_preserve:
            lines.append("Constraints the fix must preserve: " + "; ".join(self.constraints_to_preserve))
        if self.tests_that_must_pass:
            lines.append("Validators that must pass afterwards: " + ", ".join(self.tests_that_must_pass))
        lines.append("Do not relax thresholds, change reference data or remove checks.")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_steps": list(self.failed_steps),
            "findings": [f.to_dict() for f in self.findings],
            "confirmed_cause": self.confirmed_cause,
            "probable_cause": self.probable_cause,
            "constraints_to_preserve": list(self.constraints_to_preserve),
            "tests_that_must_pass": list(self.tests_that_must_pass),
        }


@dataclass
class RepairAttempt:
    attempt: int
    accepted: bool
    status: Status
    diagnosis: Optional[Diagnosis] = None
    diff: str = ""
    notes: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt, "accepted": self.accepted, "status": self.status.value,
            "diagnosis": None if self.diagnosis is None else self.diagnosis.to_dict(),
            "diff": self.diff, "notes": self.notes, "duration_s": round(self.duration_s, 4),
        }


@dataclass
class RepairOutcome:
    accepted: bool
    artifact: Any
    attempts: List[RepairAttempt] = field(default_factory=list)
    results: Dict[str, ValidatorResult] = field(default_factory=dict)
    stopped_because: str = ""

    @property
    def final_status(self) -> Status:
        return Status.PASS if self.accepted else Status.FAIL

    def summary(self) -> str:
        lines = [f"Repair {'ACCEPTED' if self.accepted else 'REJECTED'} after "
                 f"{len(self.attempts)} attempt(s): {self.stopped_because}"]
        for a in self.attempts:
            lines.append(f"  attempt {a.attempt}: {a.status.value} "
                         f"({a.notes or 'validated'})")
        return "\n".join(lines)


Evaluate = Callable[[Any], Dict[str, ValidatorResult]]
Repair = Callable[[Any, Diagnosis, int], Any]


def diagnose(results: Dict[str, ValidatorResult], *,
             required: Sequence[str] = ()) -> Diagnosis:
    """Turn failing validator results into one structured diagnosis."""
    failed = {name: r for name, r in results.items() if r.status is Status.FAIL}
    blocked = {name: r for name, r in results.items() if r.status is Status.BLOCKED}
    missing = [name for name in required if name not in results or results[name].status is not Status.PASS]

    findings = [f for r in list(failed.values()) + list(blocked.values())
                for f in r.findings if f.blocks]

    steps = sorted(set(failed) | set(blocked) | set(m for m in missing if m not in results))
    confirmed = " ".join(f.message for f in findings[:2]) if findings else ""
    probable = ""
    if blocked and not failed:
        probable = "the check could not run because its inputs were absent, not because the artifact was shown wrong"

    return Diagnosis(
        failed_steps=steps,
        findings=findings,
        confirmed_cause=confirmed,
        probable_cause=probable,
        constraints_to_preserve=[
            "acceptance criteria and invariants unchanged",
            "reference data unchanged",
            "no validator removed or weakened",
        ],
        tests_that_must_pass=sorted(set(required) | set(results)),
    )


class RepairLoop:
    """Bounded generate -> validate -> diagnose -> repair loop."""

    def __init__(self, *, evaluate: Evaluate, repair: Repair,
                 spec=None, required: Sequence[str] = (),
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 on_attempt: Optional[Callable[[RepairAttempt], None]] = None):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if max_attempts > 10:
            raise ValueError(
                "max_attempts above 10 is refused: an unbounded repair loop is how a pipeline "
                "quietly turns into a random search")
        self.evaluate = evaluate
        self.repair = repair
        self.spec = spec
        self.required = tuple(required)
        self.max_attempts = max_attempts
        self.on_attempt = on_attempt

    def run(self, artifact: Any) -> RepairOutcome:
        attempts: List[RepairAttempt] = []
        criteria_before = self._criteria_hash()
        previous_diff = ""
        results: Dict[str, ValidatorResult] = {}
        current = artifact

        for attempt_number in range(1, self.max_attempts + 1):
            started = time.monotonic()
            results = self.evaluate(current)
            status = _aggregate(results, self.required)
            if status is Status.PASS:
                record = RepairAttempt(attempt_number, True, status, notes="validated without changes"
                                       if attempt_number == 1 else "accepted after repair",
                                       duration_s=time.monotonic() - started)
                attempts.append(record)
                self._notify(record)
                return RepairOutcome(True, current, attempts, results,
                                     "all required validators passed")

            diagnosis = diagnose(results, required=self.required)
            if self._criteria_hash() != criteria_before:
                record = RepairAttempt(attempt_number, False, status, diagnosis,
                                       notes="criteria changed -- rejected",
                                       duration_s=time.monotonic() - started)
                attempts.append(record)
                self._notify(record)
                return RepairOutcome(False, current, attempts, results,
                                     "the acceptance criteria changed during repair; a result cannot "
                                     "be approved by relaxing what it must satisfy")

            if attempt_number == self.max_attempts:
                record = RepairAttempt(attempt_number, False, status, diagnosis,
                                       notes="attempt limit reached",
                                       duration_s=time.monotonic() - started)
                attempts.append(record)
                self._notify(record)
                break

            try:
                candidate = self.repair(current, diagnosis, attempt_number)
            except Exception as exc:
                record = RepairAttempt(attempt_number, False, status, diagnosis,
                                       notes=f"repair raised {type(exc).__name__}: {exc}",
                                       duration_s=time.monotonic() - started)
                attempts.append(record)
                self._notify(record)
                return RepairOutcome(False, current, attempts, results,
                                     f"the repair step itself failed: {exc}")

            diff = _diff(current, candidate)
            if not diff:
                record = RepairAttempt(attempt_number, False, status, diagnosis,
                                       diff="", notes="repair produced an identical artifact",
                                       duration_s=time.monotonic() - started)
                attempts.append(record)
                self._notify(record)
                return RepairOutcome(False, current, attempts, results,
                                     "the repair step made no change")
            if diff == previous_diff:
                record = RepairAttempt(attempt_number, False, status, diagnosis,
                                       diff=diff, notes="repeated the previous diff",
                                       duration_s=time.monotonic() - started)
                attempts.append(record)
                self._notify(record)
                return RepairOutcome(False, current, attempts, results,
                                     "the repair step repeated an identical change (oscillation)")
            previous_diff = diff
            record = RepairAttempt(attempt_number, False, status, diagnosis, diff=diff,
                                   notes="rejected, repaired artifact submitted for revalidation",
                                   duration_s=time.monotonic() - started)
            attempts.append(record)
            self._notify(record)
            current = candidate

        return RepairOutcome(False, current, attempts, results,
                             f"still failing after {len(attempts)} attempt(s)")

    # -- helpers ----------------------------------------------------------
    def _criteria_hash(self) -> str:
        if self.spec is None or not hasattr(self.spec, "criteria_hash"):
            return ""
        return self.spec.criteria_hash()

    def _notify(self, record: RepairAttempt) -> None:
        if self.on_attempt is not None:
            self.on_attempt(record)


def _aggregate(results: Dict[str, ValidatorResult], required: Sequence[str]) -> Status:
    if not results:
        return Status.BLOCKED
    for name in required:
        result = results.get(name)
        if result is None or result.status is not Status.PASS:
            return Status.FAIL if result is not None and result.status is Status.FAIL else Status.BLOCKED
    hard = [r.status for r in results.values()
            if r.status in (Status.FAIL, Status.BLOCKED, Status.NOT_RUN)]
    if Status.FAIL in hard:
        return Status.FAIL
    if Status.BLOCKED in hard:
        return Status.BLOCKED
    if Status.NOT_RUN in hard:
        return Status.NOT_RUN
    return Status.PASS


def _diff(before: Any, after: Any) -> str:
    a = _render(before).splitlines(keepends=True)
    b = _render(after).splitlines(keepends=True)
    if a == b:
        return ""
    return "".join(difflib.unified_diff(a, b, fromfile="artifact", tofile="repaired", n=2))


def _render(artifact: Any) -> str:
    if isinstance(artifact, str):
        return artifact
    if isinstance(artifact, bytes):
        return artifact.decode("utf-8", "replace")
    try:
        return json.dumps(artifact, indent=2, sort_keys=True, default=str)
    except TypeError:
        return repr(artifact)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
