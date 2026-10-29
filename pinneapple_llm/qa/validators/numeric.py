"""Numeric integrity: finite values, shapes, degenerate fields.

The shape check is the one the specification calls out explicitly: a field
whose time axis does not match the number of timestamps, or whose value
count does not match the mesh it is attached to, is wrong even when the
code that produced it compiles and runs.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..report import Finding, Status
from .base import ValidationContext, ValidatorResult, not_applicable, register

_NAME = "numeric"


@register(_NAME)
def check_numeric(ctx: ValidationContext) -> ValidatorResult:
    if not ctx.fields:
        return not_applicable(_NAME, "artifact supplied no numeric fields")

    findings: List[Finding] = []
    evidence = {"fields": {}, "checked": 0}
    n_times = len(ctx.times) if ctx.times is not None else None
    vertex_counts = {name: len(v) for name, (v, _f) in ctx.meshes.items()}

    for name, raw in ctx.fields.items():
        arr = np.asarray(raw)
        info = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        evidence["checked"] += 1
        if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            findings.append(_fail("numeric.dtype", f"field '{name}' is not numeric (dtype {arr.dtype})",
                                  {"field": name, "dtype": str(arr.dtype)}))
            continue
        if arr.ndim not in (1, 2):
            findings.append(_fail("numeric.ndim",
                                  f"field '{name}' has {arr.ndim} dimensions; expected (V,) or (T, V)",
                                  {"field": name, "shape": list(arr.shape)}))
            continue
        finite = np.isfinite(arr)
        n_bad = int(arr.size - finite.sum())
        info["non_finite"] = n_bad
        if n_bad:
            where = np.argwhere(~finite)[:5].tolist()
            findings.append(_fail(
                "numeric.non_finite",
                f"field '{name}' has {n_bad} non-finite value(s) (NaN/inf) out of {arr.size}",
                {"field": name, "count": n_bad, "examples": where}))
        if ctx.times is not None:
            if arr.ndim == 2 and arr.shape[0] != n_times:
                findings.append(_fail(
                    "numeric.time_axis",
                    f"field '{name}' has {arr.shape[0]} time step(s) but the artifact has {n_times}",
                    {"field": name, "field_times": arr.shape[0], "artifact_times": n_times}))
            elif arr.ndim == 1 and n_times not in (None, 1):
                findings.append(Finding(
                    id="numeric.time_axis", status=Status.FAIL, severity="error",
                    message=f"field '{name}' is steady (shape (V,)) but the artifact declares "
                            f"{n_times} time steps",
                    evidence={"field": name, "shape": list(arr.shape), "artifact_times": n_times},
                    justification="a steady field cannot represent a transient twin's per-time values"))
        target = _field_vertex_count(name, vertex_counts)
        if target is not None:
            width = arr.shape[-1]
            if width != target:
                findings.append(_fail(
                    "numeric.vertex_axis",
                    f"field '{name}' has {width} value(s) per step but its mesh has {target} vertices",
                    {"field": name, "values": width, "vertices": target}))
        if np.all(finite) and arr.size and np.nanmax(arr) == np.nanmin(arr):
            findings.append(Finding(
                id="numeric.degenerate", status=Status.FAIL, severity="warning",
                message=f"field '{name}' is constant at {float(np.nanmax(arr)):g} -- a physically "
                        "meaningful field is rarely constant over a whole mesh",
                evidence={"field": name, "value": float(np.nanmax(arr))}))
        evidence["fields"][name] = info

    if any(f.blocks for f in findings):
        status = Status.FAIL
    else:
        status = Status.PASS
        findings.append(Finding(id="validator.numeric", status=Status.PASS, severity="info",
                                message=f"{evidence['checked']} field(s) finite with consistent shapes",
                                evidence=evidence))
    return ValidatorResult(
        name=_NAME, status=status, findings=findings, evidence=evidence,
        limitations=["finite and shaped correctly is not the same as physically right"],
    )


def _field_vertex_count(name: str, vertex_counts: dict):
    if name in vertex_counts:
        return vertex_counts[name]
    for part, count in vertex_counts.items():
        if name.startswith(part + ".") or name.startswith(part + ":"):
            return count
    if len(vertex_counts) == 1:
        # a bundle with exactly one mesh gives the field nowhere else to belong
        return next(iter(vertex_counts.values()))
    return None


def _fail(fid: str, message: str, evidence) -> Finding:
    return Finding(id=fid, status=Status.FAIL, severity="error", message=message, evidence=dict(evidence))
