"""Neutral tools for reproducing published numbers: one evaluator for the error metrics papers report, seed
statistics, and the status rule fixed before any run (examples/paper_reproductions).

Every metric takes ``pred`` and ``ref`` on the same points (flattened); none of them knows which method produced
``pred``, so all methods in a comparison are scored by the same code.

Metric definitions (each cites where the definition comes from, never a paper's numbers):

- ``rel_l2``  ``||pred - ref||_2 / ||ref||_2`` (Raissi et al. 2019, Lu et al. 2021 and most PINN papers);
  identical to PINNsFormer's "rRMSE" (``sqrt(sum (u - u_hat)^2 / sum u^2)``, official notebook
  ``demo/convection/convection_pinnsformer.ipynb``, cell 6, variable ``rl2``).
- ``rmae``    ``sum |pred - ref| / sum |ref|`` (PINNsFormer's "rMAE", same cell, variable ``rl1``).
- ``max_abs`` / ``mean_abs``  maximum and mean absolute error (X-TFC tables report absolute errors).
- ``mse``     mean squared error (MeshGraphNets / GNS rollout RMSE is ``sqrt(mse)``, see ``rmse``).

The status rule (``status``) is the one in the reproduction protocol: "reproduced" when our median is inside the
paper's reported spread or, with no spread, within a factor ``2`` of the paper's error (same order of magnitude
when the paper's error is below ``1e-6``); "partially reproduced" when the qualitative ordering against the
paper's baseline holds but the number is outside that band; "not reproduced" otherwise. Reduced-scale runs and
runs that were not attempted carry their own labels and are never upgraded.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["rel_l2", "rmae", "max_abs", "mean_abs", "mse", "rmse", "METRICS", "evaluate", "SeedStats",
           "seed_stats", "status", "STATUSES"]


def _pair(pred, ref) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    if p.shape != r.shape:
        raise ValueError(f"pred and ref must have the same number of values, got {p.size} and {r.size}")
    if not (np.all(np.isfinite(r))):
        raise ValueError("ref has non-finite values")
    return p, r


def rel_l2(pred, ref) -> float:
    """``||pred - ref||_2 / ||ref||_2`` (also PINNsFormer's rRMSE). ``inf`` when ``pred`` is not finite."""
    p, r = _pair(pred, ref)
    if not np.all(np.isfinite(p)):
        return math.inf
    return float(np.linalg.norm(p - r) / np.linalg.norm(r))


def rmae(pred, ref) -> float:
    """``sum |pred - ref| / sum |ref|`` (PINNsFormer's rMAE)."""
    p, r = _pair(pred, ref)
    if not np.all(np.isfinite(p)):
        return math.inf
    return float(np.sum(np.abs(p - r)) / np.sum(np.abs(r)))


def max_abs(pred, ref) -> float:
    p, r = _pair(pred, ref)
    return float(np.max(np.abs(p - r))) if np.all(np.isfinite(p)) else math.inf


def mean_abs(pred, ref) -> float:
    p, r = _pair(pred, ref)
    return float(np.mean(np.abs(p - r))) if np.all(np.isfinite(p)) else math.inf


def mse(pred, ref) -> float:
    p, r = _pair(pred, ref)
    return float(np.mean((p - r) ** 2)) if np.all(np.isfinite(p)) else math.inf


def rmse(pred, ref) -> float:
    return math.sqrt(mse(pred, ref))


METRICS = {"rel_l2": rel_l2, "rrmse": rel_l2, "rmae": rmae, "max_abs": max_abs, "mean_abs": mean_abs,
           "mse": mse, "rmse": rmse}


def evaluate(pred, ref, metrics: Iterable[str] = ("rel_l2", "rmae", "max_abs")) -> dict[str, float]:
    """The named metrics of ``pred`` against ``ref`` (keys of :data:`METRICS`)."""
    return {m: METRICS[m](pred, ref) for m in metrics}


@dataclass(frozen=True)
class SeedStats:
    """Every seed's value with the median and the min-max; no best-seed selection anywhere."""
    values: tuple[float, ...]
    median: float
    min: float
    max: float
    n_failed: int

    def as_dict(self) -> dict:
        return {"values": list(self.values), "median": self.median, "min": self.min, "max": self.max,
                "n_seeds": len(self.values), "n_failed": self.n_failed}


def seed_stats(values: Sequence[float]) -> SeedStats:
    """Median and min-max over seeds. A diverged seed (``inf`` / ``nan``) counts as ``inf`` in the median, so a
    failure raises the median instead of being dropped."""
    v = [math.inf if (x is None or not math.isfinite(float(x))) else float(x) for x in values]
    if not v:
        raise ValueError("no seeds")
    return SeedStats(tuple(v), float(np.median(v)), float(min(v)), float(max(v)),
                     sum(1 for x in v if math.isinf(x)))


STATUSES = ("reproduced", "partially reproduced", "not reproduced", "reduced-scale", "not attempted")


def status(ours_median: float, paper: float, *, spread: tuple[float, float] | None = None,
           ordering_holds: bool | None = None, reduced_scale: bool = False, factor: float = 2.0) -> str:
    """Status of one target by the protocol's rule (see the module docstring).

    ``spread`` is the paper's own (low, high) for the number when it reports one. ``ordering_holds`` says whether
    the qualitative ordering the paper claims (e.g. method A beats baseline B) holds in our runs; ``None`` when the
    target has no ordering, in which case "partially reproduced" is not available. ``reduced_scale`` runs always
    return "reduced-scale" (the band is still worth reporting next to it, but it is not a reproduction).
    """
    if reduced_scale:
        return "reduced-scale"
    if not math.isfinite(ours_median):
        return "partially reproduced" if ordering_holds else "not reproduced"
    if spread is not None:
        lo, hi = spread
        inside = lo <= ours_median <= hi
    elif abs(paper) < 1e-6:
        inside = ours_median > 0 and abs(math.log10(ours_median) - math.log10(abs(paper))) < 1.0 \
            if paper != 0 else ours_median < 1e-6
    else:
        inside = abs(paper) / factor <= ours_median <= abs(paper) * factor
    if inside:
        return "reproduced"
    return "partially reproduced" if ordering_holds else "not reproduced"
