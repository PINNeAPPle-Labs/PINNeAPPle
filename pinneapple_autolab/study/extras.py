"""What a validated surrogate buys next: global sensitivity, multi-objective design and where to simulate next.

    S = sobol_on_surrogate(predict, bounds, ("CD", "CL"), n=4096)       # first-order and total Sobol indices
    pareto = pareto_on_surrogate(predict, bounds, objectives, fixed={"speed": 40.0}, max_half_width={"CD": 0.02})
    picks = acquire(predict_std, bounds, X_done, k=8)                    # largest predicted uncertainty, spread out

``predict(U) -> {output: (mean, half_width)}`` takes unit-cube inputs (columns in ``bounds`` order) and returns,
per output, the mean and the calibrated 90 % half width (a :class:`~pinneapple_autolab.study.benchmark.Board`'s
predictors). Every number these return is a surrogate prediction until a simulation confirms it: the caller runs
the reference solver on the Pareto picks and on the acquired points and reports both.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["sobol_on_surrogate", "pareto_on_surrogate", "acquire", "random_points"]

Predict = Callable[[np.ndarray], Mapping[str, tuple[np.ndarray, np.ndarray]]]


def sobol_on_surrogate(predict: Predict, bounds: Mapping[str, tuple[float, float]], outputs: Sequence[str], *,
                       n: int = 4096, seed: int = 0) -> dict[str, Any]:
    """Saltelli estimates (:func:`pinneapple_analysis.sensitivity.sobol.sobol_indices`) of the surrogate means over
    uniform inputs in ``bounds``, with bootstrap 95 % intervals."""
    from pinneapple_analysis.sensitivity.sobol import sobol_indices
    names = list(bounds)
    lo = np.array([bounds[k][0] for k in names])
    hi = np.array([bounds[k][1] for k in names])

    def fn(X):
        P = predict((np.asarray(X) - lo) / (hi - lo))
        return np.stack([P[o][0] for o in outputs], 1)
    si = sobol_indices(fn, {k: tuple(bounds[k]) for k in names}, n=n, seed=seed, outputs=list(outputs))
    out = {"inputs": names, "outputs": list(outputs), "n_base": n, "n_evaluations": int(n * (len(names) + 2))}
    for j, o in enumerate(outputs):
        out[o] = {"S1": np.asarray(si.S1)[j].tolist(), "ST": np.asarray(si.ST)[j].tolist(),
                  "S1_ci": np.asarray(si.S1_ci)[:, j].tolist(), "ST_ci": np.asarray(si.ST_ci)[:, j].tolist()}
    return out


def pareto_on_surrogate(predict: Predict, bounds: Mapping[str, tuple[float, float]],
                        objectives: Sequence[tuple[str, str]], *, fixed: Mapping[str, float] | None = None,
                        max_half_width: Mapping[str, float] | None = None, budget: int = 3000, seed: int = 0,
                        n_picks: int = 3) -> dict[str, Any]:
    """NSGA-II (:func:`pinneapple_design.design_optimizer.methods.run`) on the surrogate means over the free inputs
    (those not in ``fixed``); ``objectives``: ``[(output, "min" | "max"), ...]``; ``max_half_width``: constraints
    on the calibrated uncertainty (a design the surrogate cannot vouch for is infeasible). Returns the front and
    ``n_picks`` designs spread along it (both ends and the knee) for verification."""
    from pinneapple_design.design_optimizer.methods import run
    from pinneapple_design.design_optimizer.study import (
        Constraint,
        DesignProblem,
        DesignSpace,
        FunctionEvaluator,
        Objective,
        Variable,
    )
    names = list(bounds)
    fixed = dict(fixed or {})
    free = [k for k in names if k not in fixed]
    lo = np.array([bounds[k][0] for k in names])
    hi = np.array([bounds[k][1] for k in names])
    outs = [o for o, _ in objectives] + [f"{k}_half_width" for k in (max_half_width or {})]

    def full(Xf):
        X = np.zeros((len(Xf), len(names)))
        for j, k in enumerate(names):
            X[:, j] = Xf[:, free.index(k)] if k in free else fixed[k]
        return X

    def fn(Xf):
        X = full(np.atleast_2d(Xf))
        P = predict((X - lo) / (hi - lo))
        out = {o: P[o][0] for o, _ in objectives}
        for k in max_half_width or {}:
            out[f"{k}_half_width"] = P[k][1]
        return out
    space = DesignSpace([Variable(k, *bounds[k]) for k in free])
    prob = DesignProblem("surrogate design", space, [Objective(o, s) for o, s in objectives],
                         [Constraint(f"{k}_half_width", upper=float(v), kind="validity")
                          for k, v in (max_half_width or {}).items()],
                         FunctionEvaluator(fn, outs, name="surrogate", fidelity="surrogate"))
    res = run(prob, "nsga2", budget=budget, seed=seed)
    ev = res.evaluations
    front = res.pareto()
    Xf, F = ev.X[front], ev.F[front]
    order = np.argsort(F[:, 0])
    Xf, F = Xf[order], F[order]
    picks = []
    if len(Xf):
        Fn = (F - F.min(0)) / np.maximum(F.max(0) - F.min(0), 1e-12)
        knee = int(np.argmin(np.linalg.norm(Fn, axis=1)))
        for i in sorted({0, knee, len(Xf) - 1})[:n_picks]:
            picks.append({**{k: float(v) for k, v in zip(names, full(Xf[i:i + 1])[0], strict=False)},
                          "predicted": {o: float(np.ravel(v)[0]) for o, v in fn(Xf[i:i + 1]).items()}})
    return {"objectives": [list(o) for o in objectives], "fixed": fixed, "free": free,
            "front": [{**{k: float(v) for k, v in zip(names, full(x[None])[0], strict=False)},
                       **{o: float(f[j]) * (1 if s == "min" else -1) for j, (o, s) in enumerate(objectives)}}
                      for x, f in zip(Xf, F, strict=False)],
            "picks": picks, "n_evaluations": int(len(ev.X)), "wall_s": float(res.wall_s)}


def random_points(bounds: Mapping[str, tuple[float, float]], k: int, seed: int = 0) -> list[dict[str, float]]:
    """``k`` uniform random designs (the yardstick an acquisition rule must beat)."""
    rng = np.random.default_rng(seed)
    return [{n: float(lo + rng.random() * (hi - lo)) for n, (lo, hi) in bounds.items()} for _ in range(k)]


def acquire(uncertainty: Callable[[np.ndarray], np.ndarray], bounds: Mapping[str, tuple[float, float]],
            done_unit: np.ndarray, k: int, *, n_pool: int = 4096, min_dist: float = 0.15, seed: int = 0
            ) -> list[dict[str, float]]:
    """``k`` designs with the largest ``uncertainty(U)`` (e.g. a GP's predictive std of the target relative to its
    scale) from a scrambled Sobol pool, each at least ``min_dist`` (unit-cube distance) from the cases already run
    and from the other picks, so one uncertain corner does not take the whole batch."""
    from scipy.stats import qmc
    names = list(bounds)
    pool = qmc.Sobol(len(names), seed=seed).random(n_pool)
    u = np.asarray(uncertainty(pool), float)
    taken = np.atleast_2d(done_unit) if len(done_unit) else np.zeros((0, len(names)))
    picks = []
    for i in np.argsort(u)[::-1]:
        if len(taken) and np.min(np.linalg.norm(taken - pool[i], axis=1)) < min_dist:
            continue
        picks.append(i)
        taken = np.vstack([taken, pool[i]])
        if len(picks) == k:
            break
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])
    return [{**{n: float(v) for n, v in zip(names, lo + pool[i] * (hi - lo), strict=False)}, "acquisition": float(u[i])}
            for i in picks]
