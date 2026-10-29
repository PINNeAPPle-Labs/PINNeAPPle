"""Sensor / observation consistency.

Checks that a reading is attached to a place that exists, a time that
exists and a quantity the artifact actually models -- and that readings
used to calibrate are separated from readings reserved for validation.
That last one is a warning, not a pass: a twin with no holdout data can be
internally consistent and still have no evidential value.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..report import Finding, Status
from ..units import convert
from .base import ValidationContext, ValidatorResult, not_applicable, register

_NAME = "sensors"


@register(_NAME)
def check_sensors(ctx: ValidationContext) -> ValidatorResult:
    if not ctx.sensors:
        return not_applicable(_NAME, "artifact declares no sensor readings")

    findings: List[Finding] = []
    evidence = {"n_sensors": len(ctx.sensors), "holdout": 0, "checked": []}
    bounds = _union_bounds(ctx)

    for sensor in ctx.sensors:
        evidence["checked"].append(sensor.id)
        series = np.asarray(sensor.series, dtype=float)
        if series.ndim != 1:
            findings.append(_fail("sensors.series_shape",
                                  f"sensor '{sensor.id}': series must be 1-D, got {series.shape}",
                                  {"sensor": sensor.id}))
            continue
        if ctx.times is not None and series.size != len(ctx.times):
            findings.append(_fail(
                "sensors.alignment",
                f"sensor '{sensor.id}': {series.size} reading(s) for {len(ctx.times)} time step(s)",
                {"sensor": sensor.id, "readings": int(series.size), "times": int(len(ctx.times))}))
        if not np.isfinite(series).all():
            findings.append(_fail("sensors.non_finite",
                                  f"sensor '{sensor.id}': {int((~np.isfinite(series)).sum())} "
                                  "non-finite reading(s)",
                                  {"sensor": sensor.id, "values": series.tolist()[:20]}))
        pos = np.asarray(sensor.position, dtype=float)
        if pos.shape != (3,):
            findings.append(_fail("sensors.position", f"sensor '{sensor.id}': position must be a 3-vector",
                                  {"sensor": sensor.id, "position": list(sensor.position)}))
        elif not np.isfinite(pos).all():
            findings.append(_fail("sensors.position_finite",
                                  f"sensor '{sensor.id}': non-finite position", {"sensor": sensor.id}))
        elif bounds is not None:
            try:
                pos_m = np.array([convert(float(c), sensor.position_unit, "m") for c in pos])
            except Exception as exc:
                findings.append(Finding(
                    id="sensors.position_unit", status=Status.BLOCKED, severity="error",
                    message=f"sensor '{sensor.id}': cannot convert its position to metres: {exc}",
                    evidence={"sensor": sensor.id, "unit": sensor.position_unit}))
            else:
                lo, hi = bounds
                slack = 1e-6 + 0.01 * float(np.linalg.norm(hi - lo))
                outside = bool(np.any(pos_m < lo - slack) or np.any(pos_m > hi + slack))
                if outside:
                    findings.append(_fail(
                        "sensors.outside_geometry",
                        f"sensor '{sensor.id}' at {pos_m.round(6).tolist()} m lies outside the "
                        f"artifact's bounds [{lo.round(6).tolist()}, {hi.round(6).tolist()}]",
                        {"sensor": sensor.id, "position_m": pos_m.tolist(),
                         "bounds": [lo.tolist(), hi.tolist()]}))
        if sensor.holdout:
            evidence["holdout"] += 1
        if sensor.quantity and ctx.fields and sensor.quantity not in ctx.fields:
            findings.append(Finding(
                id="sensors.unmodelled_quantity", status=Status.FAIL, severity="warning",
                message=f"sensor '{sensor.id}' measures '{sensor.quantity}', which the artifact "
                        "does not model as a field",
                evidence={"sensor": sensor.id, "quantity": sensor.quantity,
                          "fields": sorted(ctx.fields)},
                justification="a reading that maps to no modelled quantity cannot be compared to "
                              "the model output"))

    if evidence["holdout"] == 0 and len(ctx.sensors) > 1:
        findings.append(Finding(
            id="sensors.no_holdout", status=Status.NOT_RUN, severity="warning",
            message="no sensor is marked as holdout: calibration and validation sets are not separated",
            evidence={"n_sensors": len(ctx.sensors)},
            justification="metrics computed on the data used to fit cannot detect overfitting"))

    if any(f.blocks for f in findings):
        status = Status.FAIL
    else:
        status = Status.PASS
        findings.append(Finding(id="validator.sensors", status=Status.PASS, severity="info",
                                message=f"{len(ctx.sensors)} sensor(s) aligned and inside the geometry"))
    return ValidatorResult(
        name=_NAME, status=status, findings=findings, evidence=evidence,
        limitations=["sensor consistency does not validate the sensor data itself; calibration, "
                     "mounting and measurement uncertainty are out of scope here"],
    )


def _union_bounds(ctx: ValidationContext):
    if not ctx.meshes:
        return None
    lows, highs = [], []
    for vertices, _faces in ctx.meshes.values():
        v = np.asarray(vertices, dtype=float)
        if v.size:
            lows.append(v.min(0))
            highs.append(v.max(0))
    if not lows:
        return None
    return np.min(lows, axis=0), np.max(highs, axis=0)


def _fail(fid: str, message: str, evidence) -> Finding:
    return Finding(id=fid, status=Status.FAIL, severity="error", message=message, evidence=dict(evidence))
