"""Acceptance criteria on a finished run, computed from the produced files with PINNeAPPle's checks.

=================  ==========================================================================================
kind               what is computed
=================  ==========================================================================================
``mass_balance``   volume flux through every boundary patch of the OpenFOAM result (``pinneapple_check``'s
                   finite-volume reader and :func:`pinneapple_check.fv.fv_mass_balance`, from ``phi`` when written);
                   passes when |inlet + outlet| / |inlet| <= ``tol``
``inflow``         the volume flow through the inlet patch equals ``velocity`` x the patch area of the mesh within
                   ``tol`` (the boundary condition the case asked for was actually applied)
``pressure_drop``  area-weighted mean static pressure on the inlet minus the outlet, times ``density`` for
                   incompressible solvers (kinematic p); passes when ``min_pa`` <= dp <= ``max_pa``
``converged``      the named step ran and its "solver converged" and "solver finished" checks held
``artifact``       a file or folder of the run exists and is not empty
``physics_lint``   ``pinneapple check`` on the case (conservation, divergence, boundary conditions); passes when it
                   reports no error
=================  ==========================================================================================

Each criterion gives one :class:`~pinneapple_tools.automation.record.Check` with the measured value and the
threshold; a criterion whose inputs are missing fails with the reason, it is never skipped silently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .record import Check

__all__ = ["patch_fluxes", "patch_mean", "patch_area", "mass_balance", "inflow", "pressure_drop",
           "evaluate_criterion"]


def _fv(case: Path, fields: list[str]):
    from pinneapple_check.fields import load_openfoam
    from pinneapple_check.fv import FV
    fd = load_openfoam(Path(case), time="latest", only=fields)
    return fd, FV(fd)


def patch_fluxes(case: str | Path) -> dict[str, Any]:
    """Volume flux [m³/s] out of every boundary patch of the latest time, plus net and total (sum of |flux|)."""
    from pinneapple_check.fv import fv_mass_balance
    fd, fv = _fv(Path(case), ["U", "phi"])
    b = fv_mass_balance(fv)
    if b is None:
        raise ValueError(f"{case}: no U or phi at the latest time")
    return {"time": fd.time, "patches": b["patches"], "net": b["net"], "total": b["total"], "source": b["source"],
            "cells": int(fv.nc)}


def patch_mean(case: str | Path, field: str, patches: list[str]) -> dict[str, float]:
    """Area-weighted mean of a scalar field on each named patch at the latest time."""
    fd, fv = _fv(Path(case), [field])
    if field not in fd.fields:
        raise ValueError(f"{case}: no field {field} at the latest time")
    vals = fv.bvals(field, fd.fields[field])
    area = fv.Smag[fv.ni:]
    out = {}
    for name in patches:
        sl = fv.patch_slice(name)
        if sl is None:
            raise ValueError(f"{case}: no patch {name}; patches are {[p['name'] for p in fd.patches]}")
        out[name] = float(np.sum(vals[sl] * area[sl]) / np.sum(area[sl]))
    return out


def patch_area(case: str | Path, patch: str) -> float:
    """Area [m²] of a boundary patch of the mesh."""
    fd, fv = _fv(Path(case), ["U"])
    sl = fv.patch_slice(patch)
    if sl is None:
        raise ValueError(f"{case}: no patch {patch}; patches are {[p['name'] for p in fd.patches]}")
    return float(fv.Smag[fv.ni:][sl].sum())


def inflow(case: str | Path, inlet: str, velocity: float, tol: float = 0.01) -> Check:
    """Volume flow through ``inlet`` against ``velocity`` [m/s] x its area."""
    f = patch_fluxes(case)
    q = -f["patches"].get(inlet, float("nan"))
    want = velocity * patch_area(case, inlet)
    rel = abs(q - want) / abs(want) if want else float("inf")
    return Check("inflow equals the prescribed velocity x inlet area", bool(rel <= tol),
                 f"{q:.6g} m³/s through {inlet}, prescribed {velocity:g} m/s x {want / velocity:.6g} m² = {want:.6g} "
                 f"m³/s; difference {rel:.3g} ({'<=' if rel <= tol else '>'} {tol:g})", value=rel, threshold=tol,
                 source="pinneapple_check.fv.fv_mass_balance")


def mass_balance(case: str | Path, inlet: str, outlet: str, tol: float = 0.01) -> Check:
    f = patch_fluxes(case)
    q_in, q_out = f["patches"].get(inlet), f["patches"].get(outlet)
    if q_in is None or q_out is None:
        return Check("mass balance", False, f"patches {inlet!r}/{outlet!r} not found in {sorted(f['patches'])}",
                     source="pinneapple_check.fv.fv_mass_balance")
    if not q_in < 0:                                                 # outward flux: inflow is negative
        return Check("mass balance (outlet vs inlet)", False, f"no inflow through {inlet!r} (flux {q_in:.3g} m³/s)",
                     value=q_in, source="pinneapple_check.fv.fv_mass_balance")
    rel = abs(q_in + q_out) / abs(q_in)
    others = {k: v for k, v in f["patches"].items() if k not in (inlet, outlet)}
    detail = (f"inflow {-q_in:.6g} m³/s, outflow {q_out:.6g} m³/s, imbalance {rel:.3g} "
              f"({'<=' if rel <= tol else '>'} {tol:g}); flux through the other patches "
              + ", ".join(f"{k} {v:.2g}" for k, v in others.items()) + f"; from {f['source']} at t = {f['time']:g}")
    return Check("mass balance (outlet vs inlet)", bool(rel <= tol), detail, value=rel, threshold=tol,
                 source="pinneapple_check.fv.fv_mass_balance")


def pressure_drop(case: str | Path, inlet: str, outlet: str, density: float | None = None,
                  min_pa: float | None = None, max_pa: float | None = None) -> tuple[Check, float]:
    m = patch_mean(case, "p", [inlet, outlet])
    dp = (m[inlet] - m[outlet]) * (density if density else 1.0)
    unit = "Pa" if density else "m²/s² (kinematic)"
    lo = -np.inf if min_pa is None else min_pa
    hi = np.inf if max_pa is None else max_pa
    ok = bool(lo <= dp <= hi)
    detail = (f"mean p on {inlet} {m[inlet]:.6g}, on {outlet} {m[outlet]:.6g} (kinematic); dp = {dp:.4g} {unit}, "
              f"required [{lo:g}, {hi:g}] {unit}")
    return Check("pressure drop inlet to outlet", ok, detail, value=dp, threshold=[min_pa, max_pa],
                 source="pinneapple_check.fv (area-weighted boundary values)"), dp


def evaluate_criterion(kind: str, params: dict[str, Any], run_dir: Path, previous: dict[str, Any]) -> Check:
    """One criterion of a workflow (see the module docstring). ``previous`` maps step ids to their records."""
    try:
        if kind == "mass_balance":
            return mass_balance(params["case"], params.get("inlet", "inlet"), params.get("outlet", "outlet"),
                                float(params.get("tol", 0.01)))
        if kind == "inflow":
            return inflow(params["case"], params.get("inlet", "inlet"), float(params["velocity"]),
                          float(params.get("tol", 0.01)))
        if kind == "pressure_drop":
            return pressure_drop(params["case"], params.get("inlet", "inlet"), params.get("outlet", "outlet"),
                                 params.get("density"), params.get("min_pa"), params.get("max_pa"))[0]
        if kind == "converged":
            rec = previous.get(params["step"])
            if rec is None or rec.state != "passed":
                return Check("solver converged", False, f"step {params['step']} did not pass")
            got = {c.name: c for c in rec.checks}
            need = [got.get("solver finished"), got.get("solver converged")]
            ok = all(c is not None and c.passed for c in need)
            return Check("solver converged", ok, "; ".join(c.detail for c in need if c), value=need[1].value
                         if need[1] else None, threshold=need[1].threshold if need[1] else None,
                         source=f"step {params['step']}")
        if kind == "artifact":
            p = Path(params["path"])
            p = p if p.is_absolute() else run_dir / p
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else \
                (p.stat().st_size if p.exists() else 0)
            return Check(f"artifact {p.name}", size > 0, f"{size} bytes" if size else "missing or empty", value=size)
        if kind == "physics_lint":
            from pinneapple_check import run
            report = run([str(params["case"])])
            errs = [o for o in report.errors]
            return Check("pinneapple check", not errs,
                         f"{len(report.outcomes)} checks, {len(errs)} errors" +
                         (": " + "; ".join(o.message for o in errs[:3]) if errs else ""),
                         value=len(errs), threshold=0, source="pinneapple_check.run")
    except Exception as e:                                           # a criterion that cannot be computed fails
        return Check(params.get("title") or kind, False, f"could not be evaluated: {type(e).__name__}: {e}")
    return Check(kind, False, f"unknown criterion kind {kind!r}")
