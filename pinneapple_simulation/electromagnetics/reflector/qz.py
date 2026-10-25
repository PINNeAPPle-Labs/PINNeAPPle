"""Quiet-zone metrics of a collimated field (a compact range, a plane-wave synthesiser, a near-field probe scan).

:func:`qz_metrics` takes the complex co-polar field on points of a plane transverse to the range axis and returns,
over a disk of the quiet-zone diameter:

* ``amp_pp_dB`` / ``phase_pp_deg`` -- peak-to-peak amplitude (dB) and phase (deg) variation;
* ``taper_dB`` -- the range of a quadratic surface fitted to the amplitude in dB (the smooth illumination taper);
* ``ripple_pp_dB`` -- peak-to-peak residual about that quadratic (the ripple, mostly edge diffraction);
* ``phase_ripple_pp_deg`` -- the same for the phase (deg) about a fitted plane plus quadratic;
* ``usable_diameter`` -- the largest centred disk where the amplitude stays within ``+-amp_tol`` dB and the phase
  within ``+-phase_tol`` deg (peak-to-peak 2 amp_tol and 2 phase_tol), searched over the points given (give a
  region larger than the quiet zone to find it when it is larger);
* ``xpol_dB`` -- the largest cross-polar to co-polar ratio in the disk, when the cross-polar field is given.
"""
from __future__ import annotations

import numpy as np

__all__ = ["qz_metrics", "disk_grid"]


def disk_grid(diameter: float, step: float, centre=(0.0, 0.0)) -> np.ndarray:
    """Points (M, 2) of a square grid of ``step`` inside a disk of ``diameter``."""
    n = int(np.floor(diameter / 2 / step))
    g = np.arange(-n, n + 1) * step
    X, Y = np.meshgrid(g, g)
    m = X ** 2 + Y ** 2 <= (diameter / 2) ** 2 * (1 + 1e-9)
    return np.c_[X[m] + centre[0], Y[m] + centre[1]]


def _fit_resid(x, y, v, linear_only=False):
    cols = [np.ones_like(x), x, y] + ([] if linear_only else [x * x, x * y, y * y])
    A = np.stack(cols, 1)
    c = np.linalg.lstsq(A, v, rcond=None)[0]
    fit = A @ c
    return fit, v - fit


def qz_metrics(xy, E_co, diameter: float, *, centre=(0.0, 0.0), E_x=None, amp_tol: float = 0.5,
               phase_tol: float = 5.0) -> dict[str, float]:
    """Metrics of the co-polar field ``E_co`` (M,) complex at points ``xy`` (M, 2); see the module docstring."""
    xy = np.asarray(xy, float) - np.asarray(centre, float)
    E = np.asarray(E_co)
    r = np.hypot(xy[:, 0], xy[:, 1])
    c = int(np.argmin(r))
    A = 20 * np.log10(np.abs(E) / np.abs(E[c]))
    P = np.rad2deg(np.angle(E / E[c]))
    inside = r <= diameter / 2 * (1 + 1e-9)
    x, y = xy[inside, 0], xy[inside, 1]
    fitA, resA = _fit_resid(x, y, A[inside])
    fitP, resP = _fit_resid(x, y, P[inside])
    out = {"diameter": float(diameter), "n_points": int(inside.sum()),
           "amp_pp_dB": float(np.ptp(A[inside])), "phase_pp_deg": float(np.ptp(P[inside])),
           "taper_dB": float(np.ptp(fitA)), "ripple_pp_dB": float(np.ptp(resA)),
           "phase_ripple_pp_deg": float(np.ptp(resP))}
    order = np.argsort(r)
    rs, As, Ps = r[order], A[order], P[order]
    amax = np.maximum.accumulate(As) - np.minimum.accumulate(As)
    pmax = np.maximum.accumulate(Ps) - np.minimum.accumulate(Ps)
    ok = (amax <= 2 * amp_tol) & (pmax <= 2 * phase_tol)
    bad = np.flatnonzero(~ok)
    # the disk is usable up to (not including) the first point that breaks a tolerance
    out["usable_diameter"] = float(2 * (rs[bad[0]] if len(bad) else rs[-1]))
    out["usable_limited_by_region"] = bool(len(bad) == 0)
    if E_x is not None:
        X = np.abs(np.asarray(E_x)[inside]) / np.maximum(np.abs(E[inside]), 1e-300)
        out["xpol_dB"] = float(20 * np.log10(max(X.max(), 1e-15)))
    return out
