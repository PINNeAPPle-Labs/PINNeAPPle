"""Uncertainty evidence for lab runs: the checks that answer the trust card's "where may the result not be
reliable?" with numbers instead of declared limitations.

* ``coverage_check``   -- a learned model's predictive interval (mean +- z sigma) must hold its nominal fraction of
                          unseen data: 90 % intervals that contain 60 % of the truth are not uncertainty, they are
                          optimism. Kind "uncertainty".
* ``gci``              -- grid convergence index (Roache 1994; Celik et al., J. Fluids Eng. 130, 2008) of a solver
                          result from two or three meshes: the discretisation error band of the fine-mesh value.
* ``gci_check``        -- records the GCI band as a metric and checks it is below a tolerance. Kind "uncertainty".
* ``ensemble_spread``  -- relative spread of a quantity over repeated runs (seeds, members).

    from pinneapple_lab.uq import coverage_check, gci_check
    coverage_check(ctx, "displacement", y_true, mean, std, level=0.9)
    gci_check(ctx, "Kt", [kt_coarse, kt_fine], [h_coarse, h_fine], order=2, max_band=0.05)
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

_Z = {0.5: 0.674, 0.68: 0.994, 0.8: 1.282, 0.9: 1.645, 0.95: 1.960, 0.99: 2.576}


def coverage(y_true, mean, std, level: float = 0.9) -> float:
    """Fraction of ``y_true`` inside the central ``level`` Gaussian interval of (mean, std)."""
    z = _Z.get(level) or float(np.sqrt(2) * _erfinv(level))
    y, m, s = (np.asarray(a, float).ravel() for a in (y_true, mean, std))
    return float(np.mean(np.abs(y - m) <= z * np.maximum(s, 1e-300)))


def _erfinv(x: float) -> float:
    a = 0.147
    ln = math.log(1 - x * x)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)


def coverage_check(ctx, name: str, y_true, mean, std, level: float = 0.9, tol: float = 0.07) -> float:
    """Record the empirical coverage of the ``level`` interval on unseen data and check |coverage - level| <= tol
    (calibrated: neither over- nor under-confident)."""
    c = coverage(y_true, mean, std, level)
    ctx.metric(f"{name}_coverage_{int(100 * level)}", c)
    ctx.check(f"{name}_uncertainty_calibrated", value=c, min=level - tol, max=min(1.0, level + tol),
              detail=f"fraction of unseen values inside the predicted {int(100 * level)} % interval", kind="uncertainty")
    return c


def gci(values: Sequence[float], h: Sequence[float], order: float | None = None, safety: float | None = None) -> dict:
    """Grid convergence index of the finest value. ``values`` and ``h`` (representative cell sizes) from coarse to
    fine. With three meshes the observed order is computed (Celik et al. 2008); with two, ``order`` is the formal
    order of the scheme and the safety factor is 3 (1.25 with three meshes)."""
    f = [float(v) for v in values]
    h = [float(x) for x in h]
    if len(f) == 3:
        r21, r32 = h[1] / h[2], h[0] / h[1]
        e21, e32 = f[2] - f[1], f[1] - f[0]
        p = order or 2.0
        if e21 != 0 and e32 != 0:
            s = np.sign(e32 / e21)
            for _ in range(50):                              # fixed point for the observed order
                q = math.log((r21 ** p - s) / (r32 ** p - s))
                p = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        fs = safety or 1.25
        r, e = r21, e21
    elif len(f) == 2:
        p = order or 2.0
        fs = safety or 3.0
        r, e = h[0] / h[1], f[1] - f[0]
    else:
        raise ValueError("gci needs two or three meshes")
    rel = abs(e / f[-1]) if f[-1] else float("inf")
    band = fs * rel / (r ** p - 1)
    f_ext = f[-1] + e / (r ** p - 1)
    return {"gci": band, "order": p, "extrapolated": f_ext, "ratio": r, "safety": fs}


def gci_check(ctx, name: str, values: Sequence[float], h: Sequence[float], order: float | None = None,
              max_band: float = 0.05) -> dict:
    """Record the GCI band (relative) and the Richardson-extrapolated value; check the band is below ``max_band``."""
    g = gci(values, h, order)
    ctx.metric(f"{name}_gci", g["gci"])
    ctx.metric(f"{name}_extrapolated", g["extrapolated"])
    ctx.check(f"{name}_discretisation_uncertainty", value=g["gci"], max=max_band,
              detail=f"grid convergence index of the fine-mesh value ({len(values)} meshes, order {g['order']:.2f}, "
                     f"safety {g['safety']:g})", kind="uncertainty")
    return g


def ensemble_spread(values: Sequence[float]) -> float:
    """Relative standard deviation of repeated results."""
    v = np.asarray(values, float)
    return float(v.std(ddof=1) / abs(v.mean())) if len(v) > 1 and v.mean() else float("nan")
