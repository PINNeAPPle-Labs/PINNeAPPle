"""Load study: the torque of a skewed surface-PM machine in 3-D against multi-slice 2-D on the same mesh.

The skew study of :mod:`.skew` is at no load, where the only torque is cogging. This one drives the machine: the
armature source of :meth:`~.spm.SPM3D.current_density` carries the instantaneous phase currents of a synchronous
drive, so each rotor angle has a mean torque and a ripple on top of it. For each skew the 3-D model gives the whole
torque curve; the multi-slice model is the mean of straight one-layer models at the same rotor offsets, each with
the same instantaneous currents, so the two differ only by what multi-slice neglects: the axial flux between
segments and the end leakage.

What is measured, per skew, is the mean torque and the ripple ``(max - min) / mean`` of the two models. The mean is
what a 2-D model is usually trusted for; the ripple at load is where the skew acts (it cancels the slot harmonics
of the armature reaction as it cancels cogging), and it is what the multi-slice approximation has to get right.

Approximations, stated: iron is linear (``mu_r_iron``); the winding is stranded, straight and uniform over the
slot, with no end turns in the mesh; the rotor is skewed and the stator is not; the currents are sinusoidal and
synchronous. What the 3-D model therefore adds over 2-D is the axial flux path and the end regions, and nothing
else.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence

import numpy as np

from .skew import skew_function, slice_offsets
from .spm import SPM3D

__all__ = ["ripple", "load_study"]


def ripple(torque) -> tuple[float, float, float]:
    """``(mean, peak-to-peak, ripple)`` of a torque curve, ``ripple = (max - min) / mean`` (a share of the mean)."""
    T = np.asarray(torque, float)
    mean = float(T.mean())
    pkpk = float(T.max() - T.min())
    return mean, pkpk, pkpk / mean if mean else float("nan")


def load_study(model: SPM3D, totals: Sequence[float], kinds: Sequence[str] = ("step2", "step3", "continuous"),
               thetas: Sequence[float] | None = None, currents_of: Callable | None = None,
               slice_model: SPM3D | None = None, progress: Callable[[str], None] | None = None,
               checkpoint=None) -> dict:
    """Torque against total skew (radians) for each kind, 3-D and multi-slice, at the currents
    ``currents_of(theta)`` (A). ``thetas`` default: one cogging period in angular steps of the model (end point
    excluded). Returns ``{"thetas", "period", "rows": [...]}`` with, per row, the kind, the total skew, both torque
    curves and their ``mean`` / ``pkpk`` / ``ripple``, the relative difference of the means, and seconds.
    ``checkpoint``: a JSON-lines file; each finished row is appended, and rows already there for the same angles and
    currents are reused, so an interrupted study resumes where it stopped."""
    m = model.machine
    if currents_of is None:
        raise ValueError("load_study needs currents_of(theta): the phase currents of the drive at each rotor angle")
    period = 2 * math.pi / math.lcm(m.slots, 2 * m.pole_pairs)
    steps = int(round(period / model.step))
    if steps % 2:
        raise ValueError(f"a cogging period of {steps} angular steps is not a symmetry of the polar mesh (its "
                         "triangulation repeats every two steps): use an even number of steps per period")
    if thetas is None:
        thetas = np.arange(steps) * model.step
    thetas = np.asarray(thetas, float)
    sm = slice_model or model.slice_model()
    tag = _currents_tag(currents_of, thetas)
    done = _load_checkpoint(checkpoint, thetas, tag)
    rows = []
    for kind in kinds:
        for total in totals:
            if total == 0 and rows and any(r["total"] == 0 for r in rows):
                continue
            key = (kind if total else "straight", round(float(total), 12))
            if key in done:
                rows.append(done[key])
                if progress:
                    progress(f"{key[0]} {math.degrees(total):.2f} deg: from the checkpoint")
                continue
            t0 = time.perf_counter()
            f = skew_function(kind, total, m.length)
            t3 = model.torque_series(thetas, f, currents_of=currents_of)
            t1 = time.perf_counter()
            off = slice_offsets(model, kind, total)
            ms = model.multislice(thetas, off, slice_model=sm, currents_of=currents_of)
            mean3, pk3, rip3 = ripple(t3)
            meanm, pkm, ripm = ripple(ms)
            row = {"kind": kind if total else "straight", "total": float(total),
                   "offsets_deg": np.degrees(off).round(6).tolist(),
                   "torque_3d": t3.tolist(), "torque_ms": ms.tolist(),
                   "mean_3d": mean3, "pkpk_3d": pk3, "ripple_3d": rip3,
                   "mean_ms": meanm, "pkpk_ms": pkm, "ripple_ms": ripm,
                   "d_mean": (meanm - mean3) / mean3 if mean3 else float("nan"),
                   "d_pkpk": (pkm - pk3) / pk3 if pk3 else float("nan"),
                   "seconds_3d": t1 - t0, "seconds_ms": time.perf_counter() - t1,
                   "currents_tag": tag, "thetas": thetas.tolist()}
            rows.append(row)
            if checkpoint is not None:
                with open(checkpoint, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
            if progress:
                progress(f"{row['kind']} {math.degrees(total):.2f} deg: mean {mean3:.3f} N m "
                         f"(multi-slice {100 * row['d_mean']:+.3f} %), ripple {100 * rip3:.2f} % "
                         f"(multi-slice {100 * ripm:.2f} %) [{row['seconds_3d']:.0f} s]")
    return {"thetas": thetas.tolist(), "period": period, "rows": rows}


def _currents_tag(currents_of: Callable, thetas: np.ndarray) -> list[float]:
    """The phase currents at the studied angles, as a fingerprint of the drive for the checkpoint: two runs at
    different currents must not reuse each other's rows."""
    return [round(float(c), 6) for th in thetas[:1] for c in np.atleast_1d(currents_of(float(th)))]


def _load_checkpoint(path, thetas: np.ndarray, tag: list[float]) -> dict:
    """Rows of a checkpoint file computed at the same angles and the same currents, keyed by (kind, total)."""
    from pathlib import Path

    if path is None or not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (len(r.get("thetas", [])) == len(thetas) and np.allclose(r["thetas"], thetas)
                and r.get("currents_tag") == tag):
            out[(r["kind"], round(float(r["total"]), 12))] = r
    return out
