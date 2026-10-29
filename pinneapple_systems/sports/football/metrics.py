"""Scores for probabilistic binary predictions: log-loss, Brier, ROC AUC, calibration (reliability bins, ECE) and
bootstrap intervals of score differences between two models on the same events."""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["log_loss", "brier", "auc", "reliability", "ece", "score", "paired_bootstrap"]

_EPS = 1e-6


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def auc(p, y) -> float:
    """ROC AUC by the rank-sum (Mann-Whitney) formula, ties averaged."""
    from scipy.stats import rankdata
    p, y = np.asarray(p, float), np.asarray(y, int)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(p, y, bins: int = 10, strategy: str = "quantile") -> dict[str, np.ndarray]:
    """Mean predicted probability, observed frequency and count per bin (``quantile`` or ``uniform`` bins)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    mp, fo, n = [], [], []
    for b in range(len(edges) - 1):
        s = idx == b
        if s.any():
            mp.append(p[s].mean())
            fo.append(y[s].mean())
            n.append(s.sum())
    return {"pred": np.array(mp), "obs": np.array(fo), "n": np.array(n)}


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error with quantile bins."""
    r = reliability(p, y, bins)
    return float(np.sum(r["n"] * np.abs(r["pred"] - r["obs"])) / max(r["n"].sum(), 1))


def score(p, y) -> dict[str, float]:
    """All scores of one model."""
    return {"log_loss": log_loss(p, y), "brier": brier(p, y), "auc": auc(p, y), "ece": ece(p, y),
            "n": int(len(y)), "base_rate": float(np.mean(y))}


def paired_bootstrap(p_a, p_b, y, metric: str = "log_loss", n: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Difference ``metric(a) - metric(b)`` on the same events with a 95 % bootstrap interval (negative: a better
    for losses)."""
    f = {"log_loss": log_loss, "brier": brier, "auc": auc}[metric]
    p_a, p_b, y = (np.asarray(a, float) for a in (p_a, p_b, y))
    rng = np.random.default_rng(seed)
    d0 = f(p_a, y) - f(p_b, y)
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        ds.append(f(p_a[i], y[i]) - f(p_b[i], y[i]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {"diff": float(d0), "ci95": [float(lo), float(hi)], "metric": metric}
