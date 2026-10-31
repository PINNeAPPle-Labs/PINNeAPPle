"""Unit-cell campaigns and the equal-budget benchmark of an operator surrogate against Bayesian optimisation.

* :func:`cell_simulator` -- a :class:`~..antenna.runner.CLISimulator` on :mod:`.tcda` (one process per cell, every
  frequency and scan angle in it, SuperLU), with the peak memory and wall time of each run;
* :func:`campaign` -- a space-filling DOE of cells (:data:`CELL_RANGES`, generic) run through it;
* :func:`to_arrays` -- the (design, frequency, scan) table as ``X (n, d)`` and ``Y (n, n_f, n_scan)`` (active
  reflection in dB);
* :func:`fit_fno` -- an FNO from the design parameters to the whole (frequency x scan) map, trained with
  :mod:`pinneapple_analysis.surrogate` (its FNO candidate, budget-limited Adam, best validation checkpoint), and
  :func:`heldout` its error on cells it never saw;
* :func:`equal_budget` -- the same number of solver calls for two strategies on the worst-case active reflection
  over the band and the scan range: (a) DOE of ``budget - n_verify`` cells, an FNO, an optimiser on the FNO and the
  solver on its ``n_verify`` best candidates; (b) Bayesian optimisation calling the solver directly
  (:mod:`pinneapple_design.design_optimizer`, ``"bayesian"``). The best value is always the solver's, never the
  surrogate's. The DOE alone is reported too (random search with ``budget - n_verify`` calls).
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..antenna.runner import CLISimulator, ResultTable

__all__ = ["CELL_RANGES", "cell_simulator", "campaign", "to_arrays", "worst_case", "fit_fno", "heldout",
           "equal_budget", "solver_from"]

#: generic sampling ranges of :class:`.tcda.TCDACell` (mm; er_s dimensionless)
CELL_RANGES: dict[str, tuple[float, float]] = {"h": (6.0, 14.0), "w": (1.0, 4.0), "g": (0.4, 2.0), "t_s": (0.5, 5.0),
                                               "er_s": (1.5, 6.0)}


def cell_simulator(workdir: str, scans: Sequence[float], *, phi: float = 0.0, timeout: float = 3600.0,
                   log=None, resume: bool = False) -> CLISimulator:
    """Runner of :mod:`.tcda` with every frequency and the scan angles ``scans`` (deg) in one process per cell."""
    from . import tcda
    return CLISimulator([sys.executable, tcda.__file__], workdir, per_frequency=False, timeout=timeout,
                        extra_args=["--scan", *[f"{t:g}" for t in scans], "--phi", f"{phi:g}"],
                        env={"EMERGE_STD_LOGLEVEL": "WARNING"}, log=log, resume=resume)


def campaign(n: int, freqs: Sequence[float], scans: Sequence[float], workdir: str, *,
             ranges: Mapping[str, tuple[float, float]] | None = None, fixed: Mapping[str, float] | None = None,
             method: str = "lhs", seed: int = 0, tag: str = "c", log=None) -> tuple[ResultTable, CLISimulator]:
    """``n`` cells from a ``method`` sample of ``ranges`` (plus ``fixed`` parameters), all solved."""
    from pinneapple_data.parameter_sampling import sample_parameters
    ranges = dict(ranges or CELL_RANGES)
    designs = [dict(r) | dict(fixed or {}) for r in sample_parameters(ranges, n, method=method, seed=seed)]
    sim = cell_simulator(workdir, scans, log=log)
    t = sim.run(designs, freqs, tags=[f"{tag}{i:04d}" for i in range(n)])
    return t, sim


def to_arrays(table: ResultTable, names: Sequence[str], freqs: Sequence[float], scans: Sequence[float],
              column: str = "S11_dB") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``X (n, d)``, ``Y (n, n_f, n_scan)`` and the tags of the cells with every (frequency, scan) solved."""
    X, Y, tags = [], [], []
    fi = {round(float(f), 6): i for i, f in enumerate(freqs)}
    si = {round(float(s), 6): j for j, s in enumerate(scans)}
    for tag in table.tags:
        rows = [r for r in table.rows if str(r["tag"]) == tag]
        grid = np.full((len(freqs), len(scans)), np.nan)
        for r in rows:
            i, j = fi.get(round(float(r["f_GHz"]), 6)), si.get(round(float(r["theta_deg"]), 6))
            if i is not None and j is not None:
                grid[i, j] = float(r[column])
        if np.isfinite(grid).all():
            X.append([float(rows[0][k]) for k in names])
            Y.append(grid)
            tags.append(tag)
    return np.array(X), np.array(Y), tags


def worst_case(Y: np.ndarray) -> np.ndarray:
    """Worst active reflection (dB) over the band and the scan range, per cell.

    Returns
    -------
    worst : np.ndarray [dB]
        Worst active reflection per cell.
    """
    return np.asarray(Y).reshape(len(Y), -1).max(1)


def fit_fno(X: np.ndarray, Y: np.ndarray, *, wall_s: float = 60.0, seed: int = 0, val_frac: float = 0.15,
            width: int = 24, threads: int = 1):
    """FNO from the parameters to the (n_f, n_scan) map: ``pinneapple_analysis.surrogate`` FNO candidate, trained
    for ``wall_s`` seconds on ``threads`` CPU threads (one thread: a small FNO runs ~100x more steps per second than
    with four threads that compete with each other or with a solver). Returns ``(predict, info)``,
    ``predict(X) -> (n, n_f, n_scan)``."""
    import torch

    from pinneapple_analysis.cost.equal_budget import ComputeBudget
    from pinneapple_analysis.surrogate.methods import TrainContext, fit_candidate
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    nv = max(1, int(round(val_frac * n))) if n >= 8 else 0
    va, tr = idx[:nv], idx[nv:]
    shape = Y.shape[1:]
    flat = Y.reshape(n, -1)
    ctx = TrainContext(X[tr], flat[tr], X[va], flat[va], ComputeBudget(wall_s=wall_s), seed=seed, grid_shape=shape,
                       options={"width": width})
    old = torch.get_num_threads()
    torch.set_num_threads(max(1, int(threads)))
    try:
        fitted, secs = fit_candidate("fno", ctx)
    finally:
        torch.set_num_threads(old)

    def predict(Xq):
        return np.asarray(fitted.predict(np.atleast_2d(np.asarray(Xq, float)))).reshape((-1,) + shape)

    return predict, {"train_s": round(float(secs), 2), "n_train": int(len(tr)), "n_val": int(nv),
                     **getattr(fitted, "info", {})}


def heldout(predict: Callable, X: np.ndarray, Y: np.ndarray) -> dict[str, float]:
    """Errors on cells the model never saw: RMSE over every (f, scan) point and of the worst case, in dB."""
    P = predict(X)
    e = (P - Y).ravel()
    w = worst_case(P) - worst_case(Y)
    return {"n": int(len(X)), "rmse_dB": float(np.sqrt(np.mean(e ** 2))), "max_abs_dB": float(np.abs(e).max()),
            "worst_case_rmse_dB": float(np.sqrt(np.mean(w ** 2)))}


def equal_budget(solve: Callable[[list[dict[str, float]], str], tuple[np.ndarray, np.ndarray]],
                 ranges: Mapping[str, tuple[float, float]], budget: int, *, n_verify: int = 4, seed: int = 0,
                 fno_wall_s: float = 60.0, n_candidates: int = 4000, log: Callable[[str], None] = print,
                 doe: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    """Both strategies with ``budget`` solver calls each (one call = one cell, all frequencies and scans).

    ``solve(designs, label) -> (X (k, d), Y (k, n_f, n_scan))`` runs the solver (rows it could not solve are left
    out, and still count). ``doe``: an already solved DOE of ``budget - n_verify`` cells to reuse for arm (a)."""
    from pinneapple_data.parameter_sampling import sample_parameters
    from pinneapple_design.design_optimizer.methods import run
    from pinneapple_design.design_optimizer.study import (
        DesignProblem,
        DesignSpace,
        FunctionEvaluator,
        Objective,
        Variable,
    )
    names = list(ranges)
    out: dict[str, Any] = {"budget": budget, "n_verify": n_verify}
    # (a) DOE + FNO + optimiser on the FNO + verification
    t0 = time.perf_counter()
    n_doe = budget - n_verify
    if doe is None:
        designs = [{k: float(r[k]) for k in names} for r in sample_parameters(dict(ranges), n_doe, method="lhs",
                                                                             seed=seed)]
        Xd, Yd = solve(designs, "doe")
    else:
        Xd, Yd = doe[0][:n_doe], doe[1][:n_doe]
    predict, info = fit_fno(Xd, Yd, wall_s=fno_wall_s, seed=seed)
    C = np.array([[float(r[k]) for k in names] for r in sample_parameters(dict(ranges), n_candidates, method="sobol",
                                                                        seed=seed + 7)])
    pw = worst_case(predict(C))
    pick = C[np.argsort(pw)[:n_verify]]
    Xv, Yv = solve([dict(zip(names, map(float, x), strict=True)) for x in pick], "verify")
    wv = worst_case(Yv) if len(Yv) else np.array([])
    pred_v = worst_case(predict(Xv)) if len(Xv) else np.array([])
    trace_a = list(np.minimum.accumulate(np.r_[worst_case(Yd), wv])) if len(Yd) + len(wv) else []
    out["fno_optimizer"] = {"calls": int(n_doe + n_verify), "best_dB": float(np.min(np.r_[worst_case(Yd), wv])),
                            "best_verified_candidate_dB": float(wv.min()) if len(wv) else None,
                            "predicted_vs_solver_dB": [[float(p), float(t)] for p, t in zip(pred_v, wv, strict=True)],
                            "best_design": dict(zip(names, map(float, (np.r_[Xd, Xv] if len(Xv) else Xd)[
                                int(np.argmin(np.r_[worst_case(Yd), wv]))]), strict=True)),
                            "fno": info, "wall_s": round(time.perf_counter() - t0, 1), "trace": trace_a}
    out["doe_only"] = {"calls": int(n_doe), "best_dB": float(worst_case(Yd).min()) if len(Yd) else None}
    log(f"FNO + optimiser: best {out['fno_optimizer']['best_dB']:.2f} dB in {n_doe + n_verify} solver calls")
    # (b) Bayesian optimisation on the solver
    t0 = time.perf_counter()
    calls = {"n": 0}

    def fn(Xb):
        designs = [dict(zip(names, map(float, x), strict=True)) for x in np.atleast_2d(Xb)]
        calls["n"] += len(designs)
        Xs, Ys = solve(designs, "bo")
        val = np.full(len(designs), np.nan)
        for i, x in enumerate(np.atleast_2d(Xb)):
            hit = np.flatnonzero(np.all(np.isclose(Xs, x[None]), axis=1)) if len(Xs) else []
            if len(hit):
                val[i] = worst_case(Ys[hit[:1]])[0]
        return {"worst_dB": np.where(np.isfinite(val), val, 0.0)}

    space = DesignSpace([Variable(k, float(lo), float(hi)) for k, (lo, hi) in ranges.items()])
    prob = DesignProblem("unit_cell_worst_case", space, [Objective("worst_dB", "min", "dB")], [],
                         FunctionEvaluator(fn, ["worst_dB"], name="EMerge unit cell", fidelity="reference"))
    res = run(prob, "bayesian", budget=budget, seed=seed)
    y = res.evaluations.Y["worst_dB"] if res.evaluations is not None else np.array([])
    out["bayesian"] = {"calls": int(calls["n"]), "best_dB": float(np.min(y)) if len(y) else None,
                       "best_design": (space.as_dicts(res.evaluations.X[int(np.argmin(y))])[0] if len(y) else None),
                       "status": res.status, "message": res.message, "wall_s": round(time.perf_counter() - t0, 1),
                       "trace": list(np.minimum.accumulate(y)) if len(y) else []}
    log(f"Bayesian optimisation: best {out['bayesian']['best_dB']:.2f} dB in {calls['n']} solver calls")
    return out


def solver_from(sim: CLISimulator, names: Sequence[str], freqs: Sequence[float], scans: Sequence[float],
                fixed: Mapping[str, float] | None = None):
    """A ``solve`` callable for :func:`equal_budget` from a :func:`cell_simulator`."""
    count = {"k": 0}

    def solve(designs, label):
        tags = []
        for _ in designs:
            tags.append(f"{label}{count['k']:04d}")
            count["k"] += 1
        t = sim.run([dict(d) | dict(fixed or {}) for d in designs], freqs, tags=tags)
        X, Y, _ = to_arrays(t, names, freqs, scans)
        return X, Y

    return solve
