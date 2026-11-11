"""Before a package is submitted: did it evaluate the whole split, does it reach every entity, and where is it
extrapolating?

FluidsBench ranks accuracy on a fixed support and trusts the submitter for everything else. That leaves three questions
a participant should answer about their own result, and that this module answers from records rather than from prose:

* **scope** — were the cases scored exactly the official split's cases, all of them?
* **integrity** — does the support the result was scored on hash to what the specification publishes, and does every
  relative L2 agree with its own sufficient statistics?
* **applicability** — is the model inside the range it was trained on for each case, or is it extrapolating (the
  ``aoa_extrapolation`` split exists precisely to ask this)?

Each check is written to a :class:`~pinneapple_veriphysics.ledger.EvidenceLedger`, so a check is ``passed`` only when an
intact record shows it ran, with the digest of its inputs — the rule the rest of PINNeAPPle's trust layer follows. A
check that did not run stays ``not run``; it never becomes a neutral pass.

    from pinneapple_arena.fluidsbench import assess_submission

    trust = assess_submission(scores, release=release, split_case_ids=ids, contract=contract,
                              train_bounds={"aoa": (0.0, 15.0)}, case_conditions={"case-01": {"aoa": 18.5}})
    print(trust)                    # decision: review — 3 cases above the training angle of attack
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contract import DatasetContract
from .metrics import REL_L2, SupportScore, relative_l2_from_statistics
from .support import SupportRelease

__all__ = ["SUBMIT", "REVIEW", "HOLD", "CheckResult", "SubmissionTrust", "assess_submission"]

SUBMIT, REVIEW, HOLD = "submit", "review", "hold"


@dataclass(frozen=True)
class CheckResult:
    """One check: what it measured, and whether it passed — ``None`` when it did not run at all."""
    name: str
    passed: bool | None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ran(self) -> bool:
        return self.passed is not None

    def line(self) -> str:
        verdict = "not run" if not self.ran else ("PASS" if self.passed else "FAIL")
        measured = ", ".join(f"{k}={v}" for k, v in self.detail.items())
        return f"[{verdict}] {self.name}" + (f" — {measured}" if measured else "")


@dataclass(frozen=True)
class SubmissionTrust:
    """The three checks, the ledger they were recorded in, and the decision they support."""
    decision: str
    checks: tuple[CheckResult, ...]
    reasons: tuple[str, ...] = ()
    ledger: Any = None

    @property
    def passed(self) -> bool:
        return self.decision == SUBMIT

    def __str__(self) -> str:
        lines = [f"submission trust: {self.decision.upper()}"]
        lines += [f"  {check.line()}" for check in self.checks]
        lines += [f"  - {reason}" for reason in self.reasons]
        return "\n".join(lines)


def _coverage_check(scores: Sequence[SupportScore], required: Mapping[str, Any]) -> CheckResult:
    count_fraction = float(required.get("count_fraction", 1.0))
    weight_fraction = float(required.get("weight_fraction", 1.0))
    worst = min((s.coverage.get("coverage_fraction", 1.0) for s in scores), default=0.0)
    worst_weight = min((s.coverage.get("weight_coverage_fraction", 1.0) for s in scores), default=0.0)
    unmapped = sum(int(s.coverage.get("unmapped_count", 0)) for s in scores)
    extrapolated = sum(int(s.coverage.get("extrapolated_count", 0)) for s in scores)
    passed = (worst + 1e-12 >= count_fraction and worst_weight + 1e-12 >= weight_fraction
              and unmapped == 0 and extrapolated == 0)
    return CheckResult("complete entity coverage", passed,
                       {"min_count_fraction": round(worst, 6), "min_weight_fraction": round(worst_weight, 6),
                        "unmapped": unmapped, "extrapolated": extrapolated, "cases": len(scores)})


def _statistics_check(scores: Sequence[SupportScore]) -> CheckResult:
    worst = 0.0
    checked = 0
    for score in scores:
        for metric_id, statistics in score.statistics.items():
            if statistics.reduction != REL_L2:
                continue
            expected = relative_l2_from_statistics([statistics])
            declared = float(score.metric_values.get(metric_id, math.nan))
            if not math.isfinite(declared):
                return CheckResult("relative L2 agrees with its sufficient statistics", False,
                                   {"case_id": score.case_id, "metric_id": metric_id})
            worst = max(worst, abs(declared - expected) / max(abs(expected), 1e-12))
            checked += 1
    return CheckResult("relative L2 agrees with its sufficient statistics", worst <= 1e-9,
                       {"metrics_checked": checked, "max_relative_difference": float(f"{worst:.3g}")})


def _scope_check(scores: Sequence[SupportScore], expected: Sequence[str] | None) -> CheckResult:
    if expected is None:
        return CheckResult("the whole official split was scored", None, {"status": "no split cases were given"})
    scored = {score.case_id for score in scores}
    missing = sorted(set(map(str, expected)) - scored)
    extra = sorted(scored - set(map(str, expected)))
    return CheckResult("the whole official split was scored", not missing and not extra,
                       {"scored": len(scored), "official": len(set(expected)), "missing": len(missing),
                        "not_in_split": len(extra)})


def _applicability_check(case_conditions: Mapping[str, Mapping[str, float]] | None,
                         train_bounds: Mapping[str, tuple[float, float]] | None) -> CheckResult:
    if case_conditions is None or train_bounds is None:
        return CheckResult("every case is inside the training range", None,
                           {"status": "train_bounds and case_conditions are needed"})
    outside: dict[str, list[str]] = {}
    for case_id, conditions in case_conditions.items():
        for name, value in conditions.items():
            bounds = train_bounds.get(name)
            if bounds is None:
                continue
            low, high = float(bounds[0]), float(bounds[1])
            if float(value) < low or float(value) > high:
                outside.setdefault(case_id, []).append(f"{name}={value:g} outside [{low:g}, {high:g}]")
    return CheckResult("every case is inside the training range", not outside,
                       {"cases_outside": len(outside), "cases_checked": len(case_conditions),
                        "worst": next(iter(outside.values()), [])[:1]})


def assess_submission(scores: Sequence[SupportScore], *, release: SupportRelease,
                      contract: DatasetContract | None = None, split_case_ids: Sequence[str] | None = None,
                      train_bounds: Mapping[str, tuple[float, float]] | None = None,
                      case_conditions: Mapping[str, Mapping[str, float]] | None = None,
                      ledger: Any = None) -> SubmissionTrust:
    """Run the four checks, record them in the ledger, and turn them into ``submit`` / ``review`` / ``hold``.

    ``hold``    a hard check failed: coverage, statistics, scope, or the support digest.
    ``review``  the hard checks passed but something is unknown: an applicability check that did not run, or cases
                outside the training range (which is exactly what an extrapolation split measures — worth submitting,
                worth saying).
    ``submit``  everything ran and passed."""
    if ledger is None:
        from pinneapple_veriphysics.ledger import EvidenceLedger

        ledger = EvidenceLedger()

    required = _required_coverage(release, scores)
    checks: list[CheckResult] = []

    digest_ok = release.verified
    if contract is not None and contract.support_manifest_sha256 and release.manifest_sha256:
        digest_ok = digest_ok and contract.support_manifest_sha256 == release.manifest_sha256
    checks.append(CheckResult("support release digests verify", bool(digest_ok),
                              {"release_id": release.release_id,
                               "expected": contract.support_manifest_sha256[:16] if contract else "",
                               "actual": release.manifest_sha256[:16]}))
    checks.append(_coverage_check(scores, required))
    checks.append(_statistics_check(scores))
    checks.append(_scope_check(scores, split_case_ids))
    checks.append(_applicability_check(case_conditions, train_bounds))

    for check in checks:
        _record(ledger, check)

    hard = [c for c in checks[:4] if c.passed is False]
    not_run = [c for c in checks[:4] if c.passed is None]
    applicability = checks[4]
    reasons: list[str] = []
    if hard or not_run:
        decision = HOLD
        reasons += [f"{c.name}: {dict(c.detail)}" for c in hard + not_run]
    elif not applicability.ran:
        decision = REVIEW
        reasons.append(f"applicability did not run: {dict(applicability.detail)}")
    elif not applicability.passed:
        decision = REVIEW
        reasons.append(f"cases outside the training range: {dict(applicability.detail)}")
    else:
        decision = SUBMIT
    return SubmissionTrust(decision=decision, checks=tuple(checks), reasons=tuple(reasons), ledger=ledger)


def _required_coverage(release: SupportRelease, scores: Sequence[SupportScore]) -> Mapping[str, Any]:
    """The strictest ``required_coverage`` among the supports these scores were produced on."""
    required: dict[str, float] = {"count_fraction": 1.0, "weight_fraction": 1.0, "unmapped_count": 0.0}
    for support_id in {score.support_id for score in scores}:
        definition = release.definitions.get(support_id)
        if definition is None:
            continue
        payload = dict(definition.required_coverage or {})
        required["count_fraction"] = max(required["count_fraction"], float(payload.get("count_fraction", 1.0)))
        required["weight_fraction"] = max(required["weight_fraction"], float(payload.get("weight_fraction", 1.0)))
        required["unmapped_count"] = max(required["unmapped_count"], float(payload.get("unmapped_count", 0)))
    return required


def _record(ledger: Any, check: CheckResult) -> None:
    from pinneapple_veriphysics.ledger import sha256_json

    with ledger.run(f"fluidsbench/{check.name}", inputs_sha256=sha256_json(dict(check.detail)),
                    tool="pinneapple_arena.fluidsbench.trust") as record:
        record.outputs = dict(check.detail)
        record.passed = None if check.passed is None else bool(check.passed)
