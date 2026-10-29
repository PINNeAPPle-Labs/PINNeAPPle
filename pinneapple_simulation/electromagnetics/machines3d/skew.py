"""Skew study: cogging of a skewed surface-PM machine in 3-D against multi-slice 2-D on the same mesh.

For each skew (step skew with ``K`` segments or a continuous skew, represented by one rotor offset per stack
layer) the 3-D model gives the cogging torque with the axial flux between segments and, when the model has end
regions, the end leakage; the multi-slice model is the mean of straight one-layer models at the same rotor
offsets, so the two differ only by what multi-slice neglects. The skew is symmetric about the mid-stack plane, so
the cogging is odd in the rotor angle and angles over half a cogging period suffice.

The closed-form reference for a harmonic of order ``h`` of the cogging period ``lam``: continuous skew ``s``
multiplies it by ``sin(h pi s / lam) / (h pi s / lam)``; ``K`` steps over a total skew ``s`` (step angle
``s / (K - 1)``) by :func:`~pinneapple_simulation.electromagnetics.machines.robustness.step_skew_factor`.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import numpy as np

from .spm import SPM3D, continuous_skew, step_skew

__all__ = ["skew_function", "slice_offsets", "skew_study", "continuous_factor"]


def continuous_factor(h, total: float, period: float):
    x = np.asarray(h, float) * math.pi * total / period
    return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))


def skew_function(kind: str, total: float, length: float) -> Callable:
    """``"step<K>"`` (e.g. ``"step2"``) or ``"continuous"``."""
    if kind.startswith("step"):
        return step_skew(total, int(kind[4:]), length)
    if kind == "continuous":
        return continuous_skew(total, length)
    raise ValueError(f"unknown skew kind {kind!r}")


def slice_offsets(model: SPM3D, kind: str, total: float) -> np.ndarray:
    """Rotor offsets of the multi-slice model: one per stack layer of ``model`` (rounded to its angular step), so
    its slices hold exactly the angular content of the 3-D model's layers."""
    f = skew_function(kind, total, model.machine.length)
    zc = model._zc[model._in_stack]
    return np.round(np.asarray(f(zc), float) / model.step) * model.step


def skew_study(model: SPM3D, totals: Sequence[float], kinds: Sequence[str] = ("step2", "step3", "continuous"),
               thetas: Sequence[float] | None = None, slice_model: SPM3D | None = None,
               progress: Callable[[str], None] | None = None) -> dict:
    """Cogging against total skew (radians) for each kind, 3-D and multi-slice. ``thetas`` default: half a
    cogging period in angular steps of the model. Returns ``{"thetas", "period", "rows": [...]}`` with, per row,
    the kind, the total skew, both cogging curves, their peak-to-peak (twice the largest magnitude: the curves are
    odd) and seconds."""
    m = model.machine
    period = 2 * math.pi / math.lcm(m.slots, 2 * m.pole_pairs)
    if thetas is None:
        n = int(round(period / 2 / model.step))
        thetas = np.arange(n + 1) * model.step
    thetas = np.asarray(thetas, float)
    sm = slice_model or model.slice_model()
    rows = []
    for kind in kinds:
        for total in totals:
            if total == 0 and rows and any(r["total"] == 0 for r in rows):
                continue
            t0 = time.perf_counter()
            f = skew_function(kind, total, m.length)
            c3 = model.cogging(thetas, f)
            t1 = time.perf_counter()
            off = slice_offsets(model, kind, total)
            ms = model.multislice(thetas, off, slice_model=sm)
            row = {"kind": kind if total else "straight", "total": float(total),
                   "offsets_deg": np.degrees(off).round(6).tolist(),
                   "cogging_3d": c3.tolist(), "cogging_ms": ms.tolist(),
                   "pkpk_3d": float(2 * np.abs(c3).max()), "pkpk_ms": float(2 * np.abs(ms).max()),
                   "seconds_3d": t1 - t0, "seconds_ms": time.perf_counter() - t1}
            rows.append(row)
            if progress:
                progress(f"{row['kind']} {math.degrees(total):.2f} deg: pk-pk 3-D {row['pkpk_3d']:.4f}, "
                         f"multi-slice {row['pkpk_ms']:.4f} N m ({row['seconds_3d']:.0f} s)")
    return {"thetas": thetas.tolist(), "period": period, "rows": rows}
