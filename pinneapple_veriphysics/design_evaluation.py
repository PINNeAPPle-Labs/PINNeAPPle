"""Reference-blind evaluation and statistical analysis for design selection.

This module is solver-agnostic: callers supply frozen candidate indices, surrogate
predictions, and independently computed reference values. No reference solver is
called during selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ArmMetrics:
    selected: int
    promised: int
    confirmed: int
    confirmation_rate: float | None
    best_actual_gain: float | None
    mean_actual_gain: float | None

    def to_dict(self):
        return vars(self).copy()


def evaluate_arm(indices, predictions, reference_values, *,
                 predicted_baseline, reference_baseline, threshold):
    """Evaluate an already-frozen selection against aligned independent truth.

    reference_values must contain exactly one result per selected candidate,
    in the same order. This function never selects or optimizes candidates.
    """
    ids = np.asarray(indices, dtype=int)
    pred = np.asarray(predictions, dtype=float)
    truth = np.asarray(reference_values, dtype=float)
    if ids.ndim != 1 or pred.ndim != 1 or truth.shape != ids.shape:
        raise ValueError("indices/predictions/reference shape mismatch")
    if len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= len(pred)):
        raise ValueError("duplicate or invalid candidate index")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ValueError("nonfinite values")
    if not all(np.isfinite(v) for v in (predicted_baseline, reference_baseline, threshold)):
        raise ValueError("nonfinite baseline/threshold")
    if threshold < 0:
        raise ValueError("negative improvement threshold")
    if not len(ids):
        return ArmMetrics(0, 0, 0, None, None, None)
    promised = pred[ids] - predicted_baseline >= threshold
    actual = truth - reference_baseline
    confirmed = int(np.count_nonzero(promised & (actual >= threshold)))
    n_promised = int(np.count_nonzero(promised))
    return ArmMetrics(len(ids), n_promised, confirmed,
                      confirmed / n_promised if n_promised else None,
                      float(np.max(actual)), float(np.mean(actual)))


def paired_bootstrap(baseline: Sequence[float], screened: Sequence[float], *,
                     draws: int = 10000, seed: int = 1729, confidence: float = .95):
    """Paired seed-level bootstrap for screened-minus-baseline differences.

    Missing observations must be handled by the caller, retaining paired seeds.
    The interval is descriptive and does not prove a causal or physical gain.
    """
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(screened, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("paired vectors required")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("nonfinite paired observations")
    if draws < 100 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    delta = b - a
    rng = np.random.default_rng(seed)
    samples = np.mean(delta[rng.integers(0, len(delta), size=(draws, len(delta)))], axis=1)
    alpha = (1 - confidence) / 2
    return {"paired_seeds": len(delta), "mean_difference": float(np.mean(delta)),
            "confidence": confidence,
            "interval": [float(v) for v in np.quantile(samples, [alpha, 1 - alpha])],
            "bootstrap_draws": draws, "seed": seed}


def summarize_paired_runs(rows: Sequence[Mapping], metric="confirmation_rate", **kwargs):
    """Only complete paired seeds contribute; explicitly report exclusions."""
    key_a = "baseline_" + metric
    key_b = "screened_" + metric
    valid = [r for r in rows if r.get(key_a) is not None and r.get(key_b) is not None]
    excluded = [r.get("seed") for r in rows if r not in valid]
    if not valid:
        return {"paired_seeds": 0, "excluded_seeds": excluded, "result": None}
    result = paired_bootstrap([r[key_a] for r in valid],
                              [r[key_b] for r in valid], **kwargs)
    return {"paired_seeds": len(valid), "excluded_seeds": excluded, "result": result}
