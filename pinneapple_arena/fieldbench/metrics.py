"""Metrics of a field surrogate benchmarked over many geometries — the conventions the two vehicle benchmarks use.

Two aggregations are kept apart, because mixing them changes the numbers (see the note in the vehicle-aerodynamics
benchmark):
- **per geometry, then averaged**: relative L2 of a geometry (``||y_hat - y|| / ||y||`` over all of its points), and
  the mean over the test geometries;
- **pooled over all points**: R^2 and the absolute errors, computed on the points of all geometries at once.

Every reported number carries a bootstrap confidence interval when it can, and a comparison between two models is
reported with the paired bootstrap (mean difference, its interval, the share of geometries where one wins) plus the
sign-flip permutation and Wilcoxon signed-rank p-values over the per-geometry scores.

Nothing here is vehicle-specific: pass in ``(N, C)`` arrays per geometry and get the same protocol the papers use.

    from pinneapple_arena.fieldbench.metrics import mean_relative_l2, pooled_r2, paired_bootstrap, leaderboard
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["relative_l2_per_geometry", "mean_relative_l2", "pooled_r2", "absolute_errors", "error_percentiles",
           "bootstrap_ci", "paired_bootstrap", "model_cost", "leaderboard", "score_model", "ModelScore",
           "PatchwiseProtocol"]


def _flatten(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def relative_l2_per_geometry(preds, truths) -> np.ndarray:
    """Relative L2 of every geometry: ``||pred - true||_2 / ||true||_2`` over its points and channels."""
    out = []
    for p, t in zip(preds, truths, strict=False):
        p, t = _flatten(p), _flatten(t)
        if p.shape != t.shape:
            raise ValueError(f"prediction and reference differ in size ({p.size} vs {t.size})")
        out.append(float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-12)))
    return np.asarray(out, dtype=float)


def mean_relative_l2(preds, truths) -> float:
    """The benchmark's headline error: relative L2 averaged over the geometries (not over the points)."""
    return float(np.mean(relative_l2_per_geometry(preds, truths)))


def pooled_r2(preds, truths) -> float:
    """R^2 over every point of every geometry at once (``1 - SS_res / SS_tot``)."""
    p = np.concatenate([_flatten(a) for a in preds])
    t = np.concatenate([_flatten(a) for a in truths])
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def absolute_errors(preds, truths) -> np.ndarray:
    """Absolute error of every point of every geometry, pooled."""
    return np.concatenate([np.abs(_flatten(p) - _flatten(t)) for p, t in zip(preds, truths, strict=False)])


def error_percentiles(preds, truths) -> dict:
    """The error profile: median *relative* error per point, and the P50/P90/P95/P99 of the absolute error."""
    rel = np.concatenate([np.abs(_flatten(p) - _flatten(t)) / np.maximum(np.abs(_flatten(t)), 1e-12)
                          for p, t in zip(preds, truths, strict=False)])
    abs_err = absolute_errors(preds, truths)
    return {"median_relative": float(np.median(rel)),
            "p50": float(np.percentile(abs_err, 50)), "p90": float(np.percentile(abs_err, 90)),
            "p95": float(np.percentile(abs_err, 95)), "p99": float(np.percentile(abs_err, 99))}


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, strata=None, rng=None) -> tuple[float, float]:
    """Percentile bootstrap of the mean, ``[lo, hi]`` at ``1 - alpha``.

    ``strata`` (one label per value) resamples inside each stratum and rescales the counts to the stratum's share, so
    a benchmark whose categories are unequal gets an interval that is not dominated by the largest one."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng() if rng is None else rng
    if strata is None:
        idx = rng.integers(0, v.size, size=(n_boot, v.size))
        means = v[idx].mean(axis=1)
    else:
        s = np.asarray(strata)
        means = np.empty(n_boot)
        groups = [v[s == g] for g in np.unique(s)]
        weights = np.array([g.size for g in groups], float)
        weights /= weights.sum()
        for b in range(n_boot):
            means[b] = float(sum(w * g[rng.integers(0, g.size, size=g.size)].mean()
                                 for w, g in zip(weights, groups, strict=False)))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(a, b, n_boot: int = 10000, rng=None) -> dict:
    """Paired comparison of two per-geometry score vectors (lower is better).

    Returns the mean difference ``a - b`` with its bootstrap interval, the share of geometries where ``a`` wins, and
    the two-sided sign-flip permutation and Wilcoxon signed-rank p-values."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("the two score vectors must be paired (same length)")
    rng = np.random.default_rng() if rng is None else rng
    d = a - b
    n = d.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    win = float(np.mean(d < 0))                                   # lower score is better, so a < b is a win
    # sign-flip permutation: how often a random sign change moves the mean as far as the observed one
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    perm = (signs * d).mean(axis=1)
    p_perm = float(np.mean(np.abs(perm) >= abs(d.mean())))
    p_wil = None
    try:
        from scipy.stats import wilcoxon
        if n >= 6 and np.any(d != 0):
            p_wil = float(wilcoxon(d, alternative="two-sided").pvalue)
    except Exception:
        p_wil = None
    return {"mean_difference": float(d.mean()), "ci95": (float(lo), float(hi)), "win_rate": win,
            "p_permutation": p_perm, "p_wilcoxon": p_wil, "n_pairs": int(n)}


def model_cost(model, sample_input, device: str = "cpu", n_warm: int = 2, n_rep: int = 10) -> dict:
    """Parameters, latency per geometry, throughput and peak memory of a forward pass.

    ``sample_input`` is a dict of tensors (the model's keyword arguments, batch of one geometry). The latency is the
    median of ``n_rep`` timed forwards after ``n_warm`` warm-ups, without gradient."""
    import torch

    params = int(sum(p.numel() for p in model.parameters()))
    model = model.to(device).eval()
    tensors = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample_input.items()}
    import time as _time

    peak, t_run = 0.0, 0.0
    with torch.no_grad():
        for i in range(n_warm + n_rep):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = _time.perf_counter()
            model(**tensors)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = _time.perf_counter() - start
            if i >= n_warm:
                t_run += elapsed
            if device == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
    latency_ms = (t_run / max(n_rep, 1)) * 1e3
    return {"params_m": params / 1e6, "latency_ms": latency_ms, "throughput_per_s": 1e3 / max(latency_ms, 1e-9),
            "peak_memory_gb": peak if peak else None}


@dataclass
class ModelScore:
    """One row of a leaderboard: the per-geometry scores, the summary and the cost."""
    name: str
    family: str
    scores: np.ndarray                      # per-geometry relative L2 (lower is better)
    r2: float
    mae: float
    rmse: float
    percentiles: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def rel_l2(self) -> float:
        return float(np.mean(self.scores))

    def row(self) -> dict:
        out = {"model": self.name, "family": self.family, "rel_l2": self.rel_l2, "r2": self.r2, "mae": self.mae,
               "rmse": self.rmse}
        out.update({k: v for k, v in self.cost.items()})
        return out


def score_model(name: str, family: str, preds, truths, *, cost: dict | None = None, notes: str = "",
                n_boot: int = 2000, strata=None, rng=None) -> ModelScore:
    """Score one model with the protocol above (per-geometry relative L2 with a bootstrap interval, pooled R^2)."""
    scores = relative_l2_per_geometry(preds, truths)
    err = absolute_errors(preds, truths)
    return ModelScore(name=name, family=family, scores=scores, r2=pooled_r2(preds, truths),
                      mae=float(np.mean(err)), rmse=float(np.sqrt(np.mean(err ** 2))),
                      percentiles=error_percentiles(preds, truths), cost=dict(cost or {}), notes=notes,
                      extra={"ci95": bootstrap_ci(scores, n_boot=n_boot, strata=strata, rng=rng)})


def leaderboard(rows: list[ModelScore], *, order_by: str = "rel_l2", digits: int = 4) -> str:
    """Markdown table of the scores, the interval of each, and the cost columns when they are present."""
    rows = sorted(rows, key=lambda r: getattr(r, order_by))
    has_cost = any(r.cost for r in rows)
    head = "| rank | model | family | Rel L2 | 95% CI | R² | MAE | RMSE |"
    sep = "|---|---|---|---|---|---|---|---|"
    if has_cost:
        head += " params (M) | latency (ms) | throughput (s⁻¹) |"
        sep += "---|---|---|"
    lines = [head, sep]

    def fmt(v: float) -> str:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "—"
        return f"{v:.{digits}g}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.{digits}f}"

    for i, r in enumerate(rows, 1):
        ci = r.extra.get("ci95")
        line = (f"| {i} | {r.name} | {r.family} | {fmt(r.rel_l2)} | "
                f"{'± ' + fmt(0.5 * (ci[1] - ci[0])) if ci else '—'} | {fmt(r.r2)} | {fmt(r.mae)} | "
                f"{fmt(r.rmse)} |")
        if has_cost:
            c = r.cost
            line += (f" {fmt(c.get('params_m'))} | {fmt(c.get('latency_ms'))} | {fmt(c.get('throughput_per_s'))} |")
        lines.append(line)
    return "\n".join(lines)


class PatchwiseProtocol:
    """Evaluate a prediction given on a *subset* of points against the full reference field.

    The vehicle-aerodynamics benchmark compares three ways of getting a field on the full mesh out of a model that
    only predicts where it is asked (subsampled evaluation, interpolation to the rest, and a patchwise pass):

    - ``subsampled``: score only the points the model saw;
    - ``interpolated``: interpolate the predicted subset onto all points (nearest or RBF) and score the whole field;
    - ``patchwise``: predict on overlapping patches of points and keep the central part of each, so no interpolation
      is involved."""

    def __init__(self, mode: str = "subsampled", rbf_smoothing: float = 1e-6):
        if mode not in ("subsampled", "interpolated", "patchwise"):
            raise ValueError("mode must be 'subsampled', 'interpolated' or 'patchwise'")
        self.mode, self.rbf_smoothing = mode, float(rbf_smoothing)

    def score(self, points_all: np.ndarray, values_all: np.ndarray, points_sub: np.ndarray,
              values_sub_pred: np.ndarray) -> dict:
        """(prediction on all points, reference on all points) for the chosen mode."""
        points_all = np.asarray(points_all, float)
        values_all = np.asarray(values_all, float)
        points_sub = np.asarray(points_sub, float)
        values_sub_pred = np.asarray(values_sub_pred, float)
        if self.mode == "subsampled":
            d = np.linalg.norm(points_all.reshape(-1, 1, points_all.shape[-1]) - points_sub.reshape(
                1, -1, points_sub.shape[-1]), axis=-1)
            nearest = d.argmin(axis=1)
            keep = d.min(axis=1) < 1e-9                                # every subsampled point, once
            return {"pred": values_sub_pred[nearest[keep]], "ref": values_all[keep]}
        if self.mode == "interpolated":
            from scipy.interpolate import RBFInterpolator
            interp = RBFInterpolator(points_sub, values_sub_pred.reshape(len(points_sub), -1),
                                     smoothing=self.rbf_smoothing)(points_all)
            return {"pred": interp.reshape(values_all.shape), "ref": values_all}
        # patchwise: nearest neighbour from an overlapping partition of the point set
        from scipy.spatial import cKDTree
        tree = cKDTree(points_sub)
        _, idx = tree.query(points_all, k=1)
        return {"pred": values_sub_pred[idx], "ref": values_all}
