"""Does a design that passes at its nominal dimensions still pass with its manufacturing tolerances? Monte Carlo
margin study against an upper limit, with checks on the study itself.

``margin_study(fn, nominal, tolerances, limit)`` draws ``n`` designs within the tolerances
(:func:`pinneapple_analysis.uncertainty.tolerance.propagate`, uniform by default: with no process-capability data
there is no basis for a narrower normal), evaluates the model on each and reports the distribution (mean, standard
deviation, min, max, the nearest-rank 95th percentile, the fraction under the limit with its Wilson interval) and a
verdict on the p95: ``FRAGILE`` (p95 above the limit), ``MARGINAL`` (p95 within ``marginal_fraction`` of the limit)
or ``ROBUST``.

Six checks of the result, each PASS / WARNING / FAIL with its number:

1. every sample is finite;
2. p95 stability: an independent batch of the same size gives a p95 within half the p95 margin (at least
   ``min_drift``) of this one;
3. p95 with fewer samples: an independent batch of ``n_small`` gives a p95 within the same band;
4. the hottest sample drawn against the limit;
5. corner stack-up (:func:`corner_stackup`): every tolerance at an end of its band at once, a deterministic method
   independent of the sampling (exact worst case for a model monotone in each input);
6. p95 against the limit (the verdict).

The checks are also given as a :class:`pinneapple_analysis.trust.TrustReport` (FAIL of 4-6 is critical; WARNING is a
non-critical contradiction), so the decision is APPROVED / REVIEW / REJECT by the library's rules.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .tolerance import propagate, wilson_interval

__all__ = ["corner_stackup", "summarize_samples", "margin_study"]


def corner_stackup(fn: Callable[[np.ndarray], np.ndarray], nominal: Sequence[float],
                   tolerances: Sequence[float]) -> dict[str, Any]:
    """Evaluate ``fn`` at the 2^d corners of the tolerance box ``nominal +- tolerances``."""
    x0 = np.asarray(nominal, float)
    tol = np.broadcast_to(np.asarray(tolerances, float), x0.shape)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=x0.size)))
    X = x0 + signs * tol
    y = np.asarray(fn(X), float).reshape(-1)
    i = int(np.argmax(y))
    return {"corners": X, "values": y, "worst_index": i, "worst_value": float(y[i]), "worst_corner": X[i].tolist(),
            "best_value": float(y.min())}


def summarize_samples(values: Sequence[float], limit: float, marginal_fraction: float = 0.05) -> dict[str, Any]:
    """Distribution statistics and the p95 verdict against an upper ``limit``."""
    y = np.sort(np.asarray(values, float))
    n = y.size
    p95 = float(y[min(math.ceil(0.95 * n) - 1, n - 1)])
    k = int(np.sum(y <= limit))
    margin = (limit - p95) / limit
    verdict = "FRAGILE" if p95 > limit else "MARGINAL" if margin < marginal_fraction else "ROBUST"
    return {"n_samples": n, "mean": float(y.mean()), "std": float(y.std()), "min": float(y[0]),
            "max": float(y[-1]), "p95": p95, "safe_fraction": k / n, "safe_fraction_ci": wilson_interval(k, n),
            "margin_at_p95": limit - p95, "verdict": verdict}


def _draw(fn, x0, tol, n, dist, seed):
    return np.sort(propagate(fn, x0, tol, n=n, dist=dist, seed=seed).outputs[:, 0])


def margin_study(fn: Callable[[np.ndarray], np.ndarray], nominal: Sequence[float], tolerances: Sequence[float],
                 limit: float, *, n: int = 200, n_small: int = 50, dist: str = "uniform", seed: int | None = None,
                 check_seed: int | None = None, marginal_fraction: float = 0.05, min_drift: float = 0.3,
                 unit: str = "") -> dict[str, Any]:
    """Monte Carlo margin study of ``fn`` (``(m, d)`` inputs -> ``(m,)``) against the upper ``limit``.

    ``seed`` draws the main batch, ``check_seed`` the two independent batches of checks 2 and 3 (``None``: fresh).
    Returns ``nominal_value``, ``samples`` (sorted), ``summary``, ``checks``, ``overall_verdict``, ``corner`` and
    ``trust`` (``TrustReport.to_dict()``)."""
    from pinneapple_analysis.trust import Check, TrustReport

    x0 = np.asarray(nominal, float)
    tol = np.broadcast_to(np.asarray(tolerances, float), x0.shape)
    u = f" {unit}" if unit else ""
    nominal_value = float(np.asarray(fn(x0[None, :]), float).reshape(-1)[0])
    y = _draw(fn, x0, tol, n, dist, seed)
    s = summarize_samples(y, limit, marginal_fraction)
    band = max(abs(s["margin_at_p95"]) * 0.5, min_drift)
    checks: list[dict[str, Any]] = []

    def add(name, verdict, value, lim, detail, critical=False):
        checks.append({"name": name, "verdict": verdict, "value": value, "limit": lim, "detail": detail,
                       "critical": critical})

    finite = bool(np.all(np.isfinite(y)))
    add("sample_physical_validity", "PASS" if finite else "FAIL", float(np.sum(~np.isfinite(y))), 0.0,
        f"{n} sampled designs, {int(np.sum(~np.isfinite(y)))} non-finite results")
    p_ind = summarize_samples(_draw(fn, x0, tol, n, dist, check_seed), limit)["p95"]
    drift = abs(p_ind - s["p95"])
    add("p95_estimate_stability", "PASS" if drift < band else "WARNING", drift, band,
        f"independent {n}-sample batch: p95 {p_ind:.2f}{u} vs {s['p95']:.2f}{u} ({drift:.2f}{u} apart, band "
        f"{band:.2f}{u})")
    p_small = summarize_samples(_draw(fn, x0, tol, n_small, dist, None if check_seed is None else check_seed + 1),
                                limit)["p95"]
    drift_s = abs(p_small - s["p95"])
    add("p95_convergence_with_sample_size", "PASS" if drift_s < band else "WARNING", drift_s, band,
        f"independent {n_small}-sample batch: p95 {p_small:.2f}{u} vs {s['p95']:.2f}{u} ({drift_s:.2f}{u} apart)")
    add("single_worst_sample_vs_limit", "FAIL" if s["max"] > limit else "PASS", s["max"], limit,
        f"hottest sample drawn {s['max']:.2f}{u} vs limit {limit:g}{u}", critical=True)
    corner = corner_stackup(fn, x0, tol)
    cw = corner["worst_value"]
    cv = "FAIL" if cw > limit else "WARNING" if (limit - cw) < marginal_fraction * limit else "PASS"
    add("deterministic_worst_case_stackup", cv, cw, limit,
        f"all tolerances at the worst corner: {cw:.2f}{u}, {limit - cw:.2f}{u} from the limit (Monte Carlo p95 "
        f"{s['p95']:.2f}{u})", critical=True)
    lv = {"FRAGILE": "FAIL", "MARGINAL": "WARNING", "ROBUST": "PASS"}[s["verdict"]]
    add("p95_vs_safety_limit", lv, s["p95"], limit,
        f"p95 {s['p95']:.2f}{u}, {s['margin_at_p95']:.2f}{u} under the {limit:g}{u} limit: {s['verdict']}",
        critical=True)
    overall = "FAIL" if any(c["verdict"] == "FAIL" for c in checks) else (
        "WARNING" if any(c["verdict"] == "WARNING" for c in checks) else "PASS")
    trust = TrustReport([Check(c["name"], "supports" if c["verdict"] == "PASS" else "contradicts",
                               critical=c["critical"] and c["verdict"] == "FAIL", value=c["value"], limit=c["limit"],
                               reason=c["detail"]) for c in checks], model="tolerance margin study")
    return {"nominal_value": nominal_value, "limit": limit, "samples": y.tolist(), "summary": s, "checks": checks,
            "overall_verdict": overall, "corner": {k: v for k, v in corner.items() if k != "corners"} |
            {"corners": corner["corners"].tolist(), "values": corner["values"].tolist()},
            "trust": trust.to_dict(), "decision": trust.decisao,
            "scope": "Covers sampling validity, two independent checks of the p95 estimate, the worst sample, a "
                     "deterministic corner stack-up and the limit. Does not cover model error, transients or "
                     "measurement."}
