"""Extremes extraction: block maxima, peaks over a threshold with runs declustering, the mean-excess (mean
residual life) curve, threshold-stability estimates and an automatic threshold choice."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .models import fit_gpd

__all__ = ["block_maxima", "decluster_runs", "mean_excess", "threshold_stability", "choose_threshold"]


def block_maxima(x, blocks, *, min_fraction: float = 0.0) -> dict[str, np.ndarray]:
    """Maximum of ``x`` in each block label of ``blocks`` (for example the year of each day). A block with fewer
    than ``min_fraction`` of the largest block's valid count, or with no finite value, is dropped."""
    x = np.asarray(x, float).ravel()
    blocks = np.asarray(blocks).ravel()
    if len(blocks) != len(x):
        raise ValueError("one block label per observation")
    labels = np.unique(blocks)
    counts = np.array([np.sum(np.isfinite(x[blocks == b])) for b in labels])
    keep = (counts > 0) & (counts >= min_fraction * counts.max())
    labels = labels[keep]
    maxima = np.array([np.nanmax(x[blocks == b]) for b in labels])
    return {"block": labels, "maximum": maxima, "count": counts[keep]}


def decluster_runs(x, threshold: float, run: int = 1) -> dict[str, Any]:
    """Runs declustering: exceedances separated by fewer than ``run`` consecutive non-exceedances form one
    cluster, and each cluster is represented by its maximum. The extremal index estimate is
    ``theta = n_clusters / n_exceedances`` (Coles 2001, §5.3.2)."""
    x = np.asarray(x, float).ravel()
    idx = np.nonzero(x > threshold)[0]
    if len(idx) == 0:
        return {"peaks": np.zeros(0), "index": np.zeros(0, int), "n_exceed": 0, "n_clusters": 0,
                "extremal_index": float("nan")}
    starts = np.concatenate([[0], np.nonzero(np.diff(idx) > run)[0] + 1])
    ends = np.concatenate([starts[1:], [len(idx)]])
    peak_idx = np.array([idx[a:b][np.argmax(x[idx[a:b]])] for a, b in zip(starts, ends, strict=True)])
    return {"peaks": x[peak_idx], "index": peak_idx, "n_exceed": int(len(idx)), "n_clusters": int(len(peak_idx)),
            "extremal_index": float(len(peak_idx) / len(idx))}


def mean_excess(x, thresholds, level: float = 0.95) -> dict[str, np.ndarray]:
    """Mean excess ``e(u) = E[X - u | X > u]`` with a normal interval: linear in ``u`` above a threshold where
    the GPD holds, with slope ``xi / (1 - xi)``."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    z = stats.norm.ppf(0.5 + level / 2)
    e, lo, hi, n = [], [], [], []
    for u in thresholds:
        y = x[x > u] - u
        n.append(len(y))
        if len(y) < 2:
            e.append(np.nan), lo.append(np.nan), hi.append(np.nan)
            continue
        m, s = y.mean(), y.std(ddof=1) / np.sqrt(len(y))
        e.append(m), lo.append(m - z * s), hi.append(m + z * s)
    return {"threshold": np.asarray(thresholds, float), "mean_excess": np.array(e), "lower": np.array(lo),
            "upper": np.array(hi), "n_exceed": np.array(n)}


def threshold_stability(x, thresholds, *, run: int = 1, min_exceed: int = 20) -> list[dict[str, float]]:
    """GPD shape and modified scale ``sigma* = sigma - xi u`` fitted to the declustered peaks above each
    threshold, with their delta-method standard errors: both are constant above a valid threshold."""
    x = np.asarray(x, float)
    good = np.isfinite(x)
    out = []
    for u in thresholds:
        d = decluster_runs(np.where(good, x, -np.inf), u, run)
        if d["n_clusters"] < min_exceed:
            continue
        try:
            f = fit_gpd(d["peaks"], u, n_obs=int(good.sum()))
        except ValueError:
            continue
        g = np.array([1.0, -u])                          # d sigma* / d (sigma, xi)
        v = float(g @ f.cov @ g) if np.all(np.isfinite(f.cov)) else np.nan
        out.append({"threshold": float(u), "xi": f.xi, "xi_se": float(np.sqrt(f.cov[1, 1])),
                    "modified_scale": f.modified_scale, "modified_scale_se": float(np.sqrt(v)) if v > 0 else np.nan,
                    "n_clusters": d["n_clusters"], "extremal_index": d["extremal_index"]})
    return out


def choose_threshold(x, *, quantiles=None, run: int = 1, min_exceed: int = 30, level: float = 0.95
                     ) -> dict[str, Any]:
    """The lowest candidate threshold above which the GPD shape is stable: every shape estimate at a higher
    candidate lies inside this candidate's ``level`` interval (the stability-plot rule of Coles 2001, §4.3.4,
    made automatic). Candidates are quantiles of the positive finite values (default 0.80 to 0.995)."""
    x = np.asarray(x, float)
    v = x[np.isfinite(x) & (x > 0)]
    if quantiles is None:
        quantiles = np.linspace(0.80, 0.995, 40)
    cand = np.unique(np.quantile(v, quantiles))
    rows = threshold_stability(x, cand, run=run, min_exceed=min_exceed)
    if not rows:
        raise ValueError("no candidate threshold leaves enough exceedances for a GPD fit")
    z = stats.norm.ppf(0.5 + level / 2)
    chosen = rows[-1]
    for i, r in enumerate(rows):
        lo, hi = r["xi"] - z * r["xi_se"], r["xi"] + z * r["xi_se"]
        if all(lo <= q["xi"] <= hi for q in rows[i + 1:]):
            chosen = r
            break
    me = mean_excess(x, [r["threshold"] for r in rows])
    return {"threshold": chosen["threshold"], "rule": "lowest threshold whose shape interval holds every higher "
                                                      "threshold's shape estimate", "stability": rows,
            "mean_excess": {k: v.tolist() for k, v in me.items()}}
