"""The one evaluator of the cross-framework benchmark, and the statistics over seeds.

Every runner exports plain arrays (predictions on the shared test points, on the residual stencil, the identified
parameter of an inverse run, the operator predictions on the test split); this module turns them into metrics. A
framework's own reported error is never used.

Metrics (float64):

* ``rel_l2``: ``||u_pred - u_ref||_2 / ||u_ref||_2`` over the test points (:func:`pinneapple_physics.metrics.relative_l2`).
* ``max_abs``: ``max |u_pred - u_ref|`` over the test points.
* ``residual_rms``: root mean square of the PDE residual at fresh interior points, from second-order central
  differences of the exported stencil values (:func:`.problems.residual_from_stencil`); the same formula and step
  for every framework, so it measures the trained function, not the framework's autograd.
* inverse runs: ``nu_rel_err = |nu_hat - nu| / nu`` and the field ``rel_l2``.
* operator runs: mean per-sample relative L2 over the test split (the metric of Li et al. 2021).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from pinneapple_physics import metrics as _metrics

from .problems import PINNProblem, residual_from_stencil, residual_points, test_points

__all__ = ["evaluate_pinn", "evaluate_inverse", "evaluate_operator", "per_sample_rel_l2", "seed_stats",
           "paired_ratio", "sign_test_p"]


def _finite(a: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(a)))


def evaluate_pinn(problem: PINNProblem, u_test: np.ndarray, u_stencil: np.ndarray | None = None,
                  *, reference: np.ndarray | None = None) -> dict[str, float]:
    """Metrics of a forward run from its exported predictions.

    ``u_test``: predictions at :func:`.problems.test_points` (N or N x 1). ``u_stencil``: predictions at
    ``stencil(problem, residual_points(problem)).reshape(-1, 2)`` (5 M values). ``reference``: the reference at the
    test points (computed here when omitted; pass it to avoid recomputing it for every run).
    """
    X = test_points(problem)
    ref = problem.exact(X) if reference is None else np.asarray(reference, dtype=np.float64).ravel()
    u = np.asarray(u_test, dtype=np.float64).ravel()
    if u.shape != ref.shape:
        raise ValueError(f"u_test has {u.size} values, the test grid has {ref.size} points")
    out: dict[str, float] = {"n_test": int(u.size)}
    if not _finite(u):
        out.update(rel_l2=float("nan"), max_abs=float("nan"), residual_rms=float("nan"), finite=False)
        return out
    out["rel_l2"] = float(_metrics.relative_l2(u[:, None], ref[:, None])[0])
    out["max_abs"] = float(np.max(np.abs(u - ref)))
    out["finite"] = True
    if u_stencil is not None:
        Xr = residual_points(problem)
        r = residual_from_stencil(problem, Xr, u_stencil)
        out["residual_rms"] = float(np.sqrt(np.mean(r ** 2))) if _finite(r) else float("nan")
        out["n_residual"] = int(len(Xr))
    return out


def evaluate_inverse(nu_hat: float, nu_true: float, u_pred: np.ndarray, u_ref: np.ndarray) -> dict[str, float]:
    """Inverse run: relative error of the identified coefficient and the field error on the test points."""
    u = np.asarray(u_pred, dtype=np.float64).ravel()
    ref = np.asarray(u_ref, dtype=np.float64).ravel()
    finite = bool(np.isfinite(nu_hat)) and _finite(u)
    return {"nu_hat": float(nu_hat), "nu_rel_err": float(abs(nu_hat - nu_true) / abs(nu_true)) if finite else float("nan"),
            "rel_l2": float(_metrics.relative_l2(u[:, None], ref[:, None])[0]) if finite else float("nan"),
            "finite": finite}


def per_sample_rel_l2(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-sample relative L2 over all axes but the first (operator learning)."""
    p = np.asarray(pred, dtype=np.float64).reshape(len(pred), -1)
    t = np.asarray(ref, dtype=np.float64).reshape(len(ref), -1)
    return np.linalg.norm(p - t, axis=1) / np.linalg.norm(t, axis=1)


def evaluate_operator(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Operator run: mean, median and max of the per-sample relative L2 on the test split."""
    e = per_sample_rel_l2(pred, ref)
    if not _finite(e):
        return {"rel_l2_mean": float("nan"), "rel_l2_median": float("nan"), "rel_l2_max": float("nan"),
                "n_test": int(len(e)), "finite": False}
    return {"rel_l2_mean": float(e.mean()), "rel_l2_median": float(np.median(e)), "rel_l2_max": float(e.max()),
            "n_test": int(len(e)), "finite": True}


# --------------------------------------------------------------------------- statistics over seeds
def seed_stats(values: Iterable[float]) -> dict[str, object]:
    """Median, min, max and n of the finite values; ``n_failed`` counts NaN / inf (failures are not dropped silently:
    they are counted, and the median is over the finite values only)."""
    v = np.asarray(list(values), dtype=np.float64)
    ok = v[np.isfinite(v)]
    if ok.size == 0:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan"), "n": int(v.size),
                "n_failed": int(v.size)}
    return {"median": float(np.median(ok)), "min": float(ok.min()), "max": float(ok.max()), "n": int(v.size),
            "n_failed": int(v.size - ok.size)}


def paired_ratio(a: Mapping[int, float], b: Mapping[int, float]) -> dict[str, object]:
    """Per-seed ratio ``a / b`` over the seeds both have (finite): median, min, max, and how many seeds ``a < b``."""
    seeds = sorted(s for s in set(a) & set(b) if np.isfinite(a[s]) and np.isfinite(b[s]) and b[s] != 0)
    r = np.array([a[s] / b[s] for s in seeds], dtype=np.float64)
    wins = int(sum(a[s] < b[s] for s in seeds))
    if r.size == 0:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0, "a_better": 0,
                "sign_test_p": float("nan")}
    return {"median": float(np.median(r)), "min": float(r.min()), "max": float(r.max()), "n": int(r.size),
            "a_better": wins, "sign_test_p": sign_test_p(wins, int(r.size))}


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact sign test p-value for ``k`` successes in ``n`` paired trials (ties excluded beforehand).
    With n = 5 the smallest attainable p is 0.0625: five seeds cannot show significance at 0.05 by this test."""
    if n == 0:
        return float("nan")
    k = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def summarize(rows: Sequence[Mapping[str, object]], keys: Sequence[str], by: Sequence[str]) -> list[dict[str, object]]:
    """Group ``rows`` by the ``by`` fields and give :func:`seed_stats` of each metric in ``keys``."""
    groups: dict[tuple, list[Mapping[str, object]]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(b) for b in by), []).append(r)
    out = []
    for g, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row: dict[str, object] = dict(zip(by, g, strict=False))
        for k in keys:
            row[k] = seed_stats(float(r.get(k, float("nan"))) if r.get(k) is not None else float("nan") for r in rs)
        out.append(row)
    return out


__all__ += ["summarize"]
