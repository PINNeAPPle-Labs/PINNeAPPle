"""Physical invariants declared by the specification (stage E, part 3).

Every invariant in the specification is evaluated here against the
artifact's own arrays. The invariants are declarative (``kind`` +
``params``) -- this module never evaluates generated Python, which is the
whole reason :mod:`pinneapple_llm.qa.spec` refuses expression strings.

An invariant whose target the artifact does not provide is ``BLOCKED``,
not skipped: "the constraint we said mattered was never checked" is
exactly the state that must not look like success.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..report import Finding, Status
from ..spec import Invariant
from ..units import UnitError, are_compatible
from .base import ValidationContext, ValidatorResult, not_applicable, register

_NAME = "physics"


@register(_NAME)
def check_physics(ctx: ValidationContext) -> ValidatorResult:
    spec = ctx.spec
    if not spec.invariants:
        return not_applicable(
            _NAME, "the specification declares no physical invariants, so none were checked")

    findings: List[Finding] = []
    evidence = {"evaluated": [], "not_evaluated": []}
    any_fail = False
    any_blocked = False

    for inv in spec.invariants:
        result = _evaluate(inv, ctx)
        findings.append(result)
        if result.status is Status.PASS:
            evidence["evaluated"].append(inv.name)
        elif result.status is Status.FAIL:
            evidence["evaluated"].append(inv.name)
            if inv.severity == "error":
                any_fail = True
        else:
            evidence["not_evaluated"].append(inv.name)
            if inv.kind != "unit_compatible":  # unit kind is handled by the units validator too
                any_blocked = True

    if any_fail:
        status = Status.FAIL
    elif any_blocked:
        status = Status.BLOCKED
    else:
        status = Status.PASS
        findings.append(Finding(id="validator.physics", status=Status.PASS, severity="info",
                                message=f"{len(evidence['evaluated'])} invariant(s) satisfied"))
    return ValidatorResult(
        name=_NAME, status=status, findings=findings, evidence=evidence,
        limitations=["an invariant constrains the artifact; it does not show the model represents "
                     "the real system, only that it respects the constraints we could state"],
    )


def _evaluate(inv: Invariant, ctx: ValidationContext) -> Finding:
    if inv.kind == "unit_compatible":
        return _unit_compatible(inv, ctx)
    arr = ctx.fields.get(inv.target)
    if arr is None:
        return Finding(
            id=f"invariant.{inv.name}", status=Status.BLOCKED, severity="error",
            message=f"invariant '{inv.name}' targets '{inv.target}', which the artifact does not provide",
            evidence={"invariant": inv.name, "target": inv.target,
                      "available": sorted(ctx.fields)},
            justification="a constraint that was never evaluated cannot count as satisfied",
        )
    values = np.asarray(arr)
    try:
        return _KINDS[inv.kind](inv, values, ctx)
    except Exception as exc:  # a malformed invariant must not crash the pipeline
        return Finding(
            id=f"invariant.{inv.name}", status=Status.BLOCKED, severity="error",
            message=f"invariant '{inv.name}' could not be evaluated: {type(exc).__name__}: {exc}",
            evidence={"invariant": inv.name, "kind": inv.kind, "error": str(exc)},
        )


def _finite_values(values: np.ndarray) -> np.ndarray:
    flat = values.ravel()
    return flat[np.isfinite(flat)]


def _result(inv: Invariant, ok: bool, message: str, **evidence) -> Finding:
    return Finding(
        id=f"invariant.{inv.name}",
        status=Status.PASS if ok else Status.FAIL,
        severity="info" if ok else inv.severity,
        message=message, evidence={"invariant": inv.name, "kind": inv.kind, **evidence},
        justification=inv.description,
    )


def _nonnegative(inv, values, ctx):
    finite = _finite_values(values)
    worst = float(finite.min()) if finite.size else 0.0
    return _result(inv, worst >= 0.0, f"minimum = {worst:g} (must be >= 0)", minimum=worst)


def _max(inv, values, ctx):
    limit = float(inv.params["limit"])
    finite = _finite_values(values)
    worst = float(finite.max()) if finite.size else float("-inf")
    return _result(inv, worst <= limit, f"maximum = {worst:g} (limit {limit:g})",
                   maximum=worst, limit=limit)


def _min(inv, values, ctx):
    limit = float(inv.params["limit"])
    finite = _finite_values(values)
    worst = float(finite.min()) if finite.size else float("inf")
    return _result(inv, worst >= limit, f"minimum = {worst:g} (limit {limit:g})",
                   minimum=worst, limit=limit)


def _range(inv, values, ctx):
    low, high = float(inv.params["low"]), float(inv.params["high"])
    finite = _finite_values(values)
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (float("nan"),) * 2
    ok = bool(finite.size) and lo >= low and hi <= high
    return _result(inv, ok, f"observed range [{lo:g}, {hi:g}] within [{low:g}, {high:g}]",
                   observed=[lo, hi], allowed=[low, high])


def _monotonic_nondecreasing(inv, values, ctx):
    arr = np.asarray(values, dtype=float)
    axis = 0 if arr.ndim > 1 else 0
    finite = np.where(np.isfinite(arr), arr, np.nan)
    if finite.ndim == 1:
        finite = finite.reshape(1, -1)
    diffs = np.diff(np.nan_to_num(finite, nan=0.0), axis=axis)
    violations = int(np.sum(diffs < -1e-12))
    return _result(inv, violations == 0,
                   f"{violations} decreasing step(s) along axis {axis}",
                   violations=violations)


def _conservation(inv, values, ctx):
    arr = np.asarray(values, dtype=float)
    axis = int(inv.params.get("axis", 1))
    rtol = float(inv.params.get("rtol", 1e-3))
    if arr.ndim <= axis:
        return _result(inv, False, f"field has {arr.ndim} dimension(s); conservation needs axis {axis}",
                       shape=list(arr.shape))
    totals = np.nansum(arr, axis=axis)
    if totals.size < 2:
        return _result(inv, False, "conservation needs at least two time steps to compare",
                       totals=totals.tolist())
    spread = float(np.nanmax(totals) - np.nanmin(totals))
    scale = float(np.nanmean(np.abs(totals))) or 1.0
    rel = spread / scale
    return _result(inv, rel <= rtol,
                   f"relative drift of the conserved total = {rel:.3e} (rtol {rtol:g})",
                   relative_drift=rel, rtol=rtol,
                   totals=[float(t) for t in totals[:20]])


def _finite(inv, values, ctx):
    arr = np.asarray(values)
    bad = int(arr.size - np.isfinite(arr).sum())
    return _result(inv, bad == 0, f"{bad} non-finite value(s)", non_finite=bad)


def _unit_compatible(inv, ctx: ValidationContext) -> Finding:
    expected = inv.params.get("unit", "")
    actual = ctx.field_units.get(inv.target)
    if actual is None:
        return Finding(
            id=f"invariant.{inv.name}", status=Status.BLOCKED, severity="error",
            message=f"invariant '{inv.name}': field '{inv.target}' declares no unit",
            evidence={"invariant": inv.name, "target": inv.target},
        )
    try:
        ok = are_compatible(actual, expected)
    except UnitError as exc:
        return Finding(id=f"invariant.{inv.name}", status=Status.BLOCKED, severity="error",
                       message=f"invariant '{inv.name}': {exc}",
                       evidence={"target": inv.target, "actual": actual, "expected": expected})
    return _result(inv, ok, f"'{inv.target}' is in '{actual}', expected compatibility with '{expected}'",
                   actual=actual, expected=expected)


_KINDS = {
    "nonnegative": _nonnegative,
    "max": _max,
    "min": _min,
    "range": _range,
    "monotonic_nondecreasing": _monotonic_nondecreasing,
    "conservation": _conservation,
    "finite": _finite,
}
