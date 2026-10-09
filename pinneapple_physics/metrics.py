"""``pp.metrics``: the error metrics used everywhere in the library, with one convention.

Convention (applies to every function here):

* ``pred`` and ``true`` are arrays of the same shape ``(N,)`` or ``(N, F)``: N points, F fields.
  NumPy arrays, PyTorch tensors and nested lists are accepted; computation is in float64.
* For ``(N, F)`` inputs a function returns one value per field (shape ``(F,)``); use ``per_field`` to get a
  ``{field_name: value}`` dict. Fields are never mixed: a relative error of a velocity and of a pressure with
  different scales would be meaningless.
* Relative errors divide by the norm of ``true`` **for that field**. If that norm is zero the relative error is
  undefined and ``nan`` is returned (never a silently huge or zero number).

    >>> from pinneapple_physics import metrics
    >>> metrics.relative_l2([1.0, 2.0], [1.0, 2.0])
    0.0
    >>> metrics.per_field(metrics.rmse, pred, true, fields=("u", "p"))
    {'u': 0.012, 'p': 0.31}
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

__all__ = ["l2", "relative_l2", "rmse", "mae", "max_abs", "relative_linf", "r2", "per_field", "summary", "pooled",
           "METRICS"]


def _as2d(pred, true):
    def conv(a):
        try:
            import torch
            if isinstance(a, torch.Tensor):
                return a.detach().cpu().numpy().astype(np.float64)
        except ImportError:  # pragma: no cover
            pass
        return np.asarray(a, dtype=np.float64)

    p, t = conv(pred), conv(true)
    if p.shape != t.shape:
        raise ValueError(f"pred and true must have the same shape, got {p.shape} and {t.shape}")
    if p.ndim not in (1, 2):
        raise ValueError(f"expected shape (N,) or (N, F), got {p.shape}")
    if p.shape[0] == 0:
        raise ValueError("empty arrays: no points to compare")
    if p.ndim == 1:
        return p[:, None], t[:, None], True
    return p, t, False


def _out(v: np.ndarray, squeeze: bool):
    return float(v[0]) if squeeze else v


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def l2(pred, true):
    """Euclidean norm of the error, per field: ``||pred - true||_2``."""
    p, t, s = _as2d(pred, true)
    return _out(np.linalg.norm(p - t, axis=0), s)


def relative_l2(pred, true):
    """``||pred - true||_2 / ||true||_2`` per field (``nan`` where ``||true|| = 0``)."""
    p, t, s = _as2d(pred, true)
    return _out(_safe_div(np.linalg.norm(p - t, axis=0), np.linalg.norm(t, axis=0)), s)


def rmse(pred, true):
    """Root mean square error per field."""
    p, t, s = _as2d(pred, true)
    return _out(np.sqrt(np.mean((p - t) ** 2, axis=0)), s)


def mae(pred, true):
    """Mean absolute error per field."""
    p, t, s = _as2d(pred, true)
    return _out(np.mean(np.abs(p - t), axis=0), s)


def max_abs(pred, true):
    """Largest absolute error per field (the L-infinity norm of the error)."""
    p, t, s = _as2d(pred, true)
    return _out(np.max(np.abs(p - t), axis=0), s)


def relative_linf(pred, true):
    """``max|pred - true| / max|true|`` per field (``nan`` where ``max|true| = 0``)."""
    p, t, s = _as2d(pred, true)
    return _out(_safe_div(np.max(np.abs(p - t), axis=0), np.max(np.abs(t), axis=0)), s)


def r2(pred, true):
    """Coefficient of determination per field (``nan`` when ``true`` is constant)."""
    p, t, s = _as2d(pred, true)
    ss_res = np.sum((p - t) ** 2, axis=0)
    ss_tot = np.sum((t - t.mean(axis=0)) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), np.nan)
    return _out(v, s)


METRICS: Dict[str, Callable] = {"relative_l2": relative_l2, "rmse": rmse, "mae": mae, "max_abs": max_abs,
                                "relative_linf": relative_linf, "r2": r2, "l2": l2}


def per_field(metric: Callable, pred, true, fields: Sequence[str]) -> Dict[str, float]:
    """``{field: value}`` for an ``(N, F)`` comparison."""
    v = np.atleast_1d(metric(pred, true))
    if len(v) != len(fields):
        raise ValueError(f"{len(fields)} field names for {len(v)} fields")
    return {f: float(x) for f, x in zip(fields, v)}


def summary(pred, true, fields: Sequence[str], names: Sequence[str] = ("relative_l2", "rmse", "max_abs")
            ) -> Dict[str, Dict[str, float]]:
    """``{metric: {field: value}}`` for several metrics at once (the shape benchmark tables use)."""
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise KeyError(f"unknown metric(s) {unknown}; available: {sorted(METRICS)}")
    return {n: per_field(METRICS[n], pred, true, fields) for n in names}


def pooled(pred, true) -> Dict[str, float]:
    """One number per metric over every entry, all fields together: ``relative_l2``, ``rmse``, ``mse``, ``max_abs``,
    ``r2``. For leaderboards that rank models by a single value; use the per-field functions for anything reported
    per quantity. When the shapes differ, a 1-D side gets a trailing axis and the two are broadcast, as the benchmark
    suite always did."""
    p, t = _np(pred), _np(true)
    if p.shape != t.shape:
        p = p[:, None] if p.ndim == 1 else p
        t = t[:, None] if t.ndim == 1 else t
        p, t = np.broadcast_arrays(p, t)
    p, t = p.ravel(), t.ravel()
    m = {k: float(f(p, t)) for k, f in (("relative_l2", relative_l2), ("rmse", rmse), ("max_abs", max_abs),
                                         ("r2", r2))}
    m["mse"] = m["rmse"] ** 2
    return m


def _np(a) -> np.ndarray:
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().astype(np.float64)
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(a, dtype=np.float64)
