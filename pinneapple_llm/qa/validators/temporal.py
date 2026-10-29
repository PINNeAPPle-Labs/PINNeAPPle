"""Temporal consistency: ordering, sample counts, validity window.

A transient artifact that reports 12 timestamps and 10 time steps of
field data is wrong in a way no syntax check notices, and that a viewer
will happily render as if nothing were missing.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..report import Finding, Status
from ..units import convert
from .base import ValidationContext, ValidatorResult, not_applicable, register

_NAME = "temporal"


@register(_NAME)
def check_temporal(ctx: ValidationContext) -> ValidatorResult:
    if ctx.times is None:
        return not_applicable(_NAME, "artifact declares no time axis (steady-state artifact)")

    times = np.asarray(ctx.times, dtype=float)
    findings: List[Finding] = []
    evidence = {"n_times": int(times.size), "time_unit": ctx.time_unit}

    if times.ndim != 1:
        findings.append(_fail("temporal.ndim", f"time axis has {times.ndim} dimensions; expected 1",
                              {"shape": list(times.shape)}))
        return ValidatorResult(name=_NAME, status=Status.FAIL, findings=findings, evidence=evidence)
    if times.size == 0:
        return not_applicable(_NAME, "artifact declares an empty time axis")
    if not np.isfinite(times).all():
        findings.append(_fail("temporal.non_finite",
                              f"{int((~np.isfinite(times)).sum())} non-finite timestamp(s)",
                              {"times": times.tolist()[:20]}))
        return ValidatorResult(name=_NAME, status=Status.FAIL, findings=findings, evidence=evidence)

    diffs = np.diff(times)
    if np.any(diffs <= 0):
        bad = int(np.sum(diffs <= 0))
        findings.append(_fail(
            "temporal.monotonic",
            f"time axis is not strictly increasing ({bad} non-increasing step(s))",
            {"times": times.tolist()[:20], "steps": diffs.tolist()[:20]}))

    evidence["start"] = float(times[0])
    evidence["end"] = float(times[-1])
    if times.size > 1:
        evidence["median_step"] = float(np.median(diffs[np.isfinite(diffs)])) if diffs.size else None

    # fields must agree with the time axis
    for name, arr in ctx.fields.items():
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[0] != times.size:
            findings.append(_fail(
                "temporal.field_steps",
                f"field '{name}' has {a.shape[0]} time step(s); the time axis has {times.size}",
                {"field": name, "field_steps": int(a.shape[0]), "times": int(times.size)}))

    # validity window declared by the specification
    validity = ctx.spec.temporal_validity
    if validity is not None:
        lo, hi = validity
        try:
            lo_si = convert(lo.value, lo.unit, ctx.time_unit)
            hi_si = convert(hi.value, hi.unit, ctx.time_unit)
        except Exception as exc:
            findings.append(Finding(
                id="temporal.validity_units", status=Status.BLOCKED, severity="error",
                message=f"cannot compare the time axis with the declared validity window: {exc}",
                evidence={"window": [lo.to_dict(), hi.to_dict()], "time_unit": ctx.time_unit}))
        else:
            outside = int(np.sum((times < lo_si) | (times > hi_si)))
            evidence["validity_window"] = [lo_si, hi_si]
            if outside:
                findings.append(_fail(
                    "temporal.outside_validity",
                    f"{outside} timestamp(s) fall outside the declared validity window "
                    f"[{lo_si:g}, {hi_si:g}] {ctx.time_unit}",
                    {"outside": outside, "window": [lo_si, hi_si]}))

    if any(f.blocks for f in findings):
        status = Status.FAIL
    else:
        status = Status.PASS
        findings.append(Finding(id="validator.temporal", status=Status.PASS, severity="info",
                                message=f"{times.size} timestamps strictly increasing and consistent"))
    return ValidatorResult(
        name=_NAME, status=status, findings=findings, evidence=evidence,
        limitations=["monotonic timestamps do not establish that the time evolution is physical"],
    )


def _fail(fid: str, message: str, evidence) -> Finding:
    return Finding(id=fid, status=Status.FAIL, severity="error", message=message, evidence=dict(evidence))
