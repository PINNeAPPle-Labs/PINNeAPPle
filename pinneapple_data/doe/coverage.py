"""How well a set of cases covers a parameter space, which planned cases repeat what already ran, and in which
order to run them.

All distances are in the unit hypercube of the free parameters (``ParameterSpace.to_unit``), so a parameter in
metres and one in pascals weigh the same; categories and integers sit at the centres of their cells.
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .designs import greedy_maximin


def coverage(unit: np.ndarray, names: list[str] | None = None, *, n_probe: int = 2 ** 14, seed: int = 0,
             bins_2d: int | None = None, levels: dict[str, int] | None = None) -> dict[str, Any]:
    """Space-coverage metrics of points in [0, 1]^d.

    ``discrepancy``      centred L2 discrepancy (scipy ``qmc.discrepancy``, method "CD"): lower is more uniform
    ``min_distance``     smallest distance between two points (0: duplicates)
    ``mean_nn_distance`` mean distance to the nearest other point
    ``fill_distance``    largest distance from any probe point of the cube to its nearest case: the biggest hole
    ``ideal_spacing``    (1/n)^(1/d), the spacing of a regular grid with n points, to read the distances against
    ``projection_1d``    per parameter: fraction of n equal bins holding at least one case; for a discrete parameter
                         with k values (``levels``), of min(n, k) bins
    ``projection_2d``    per pair of parameters: fraction of b x b cells holding a case (b = floor(sqrt(n)))
    """
    unit = np.atleast_2d(np.asarray(unit, float))
    n, d = unit.shape
    names = list(names) if names else [f"x{i}" for i in range(d)]
    out: dict[str, Any] = {"n": int(n), "dim": int(d), "ideal_spacing": float((1.0 / max(n, 1)) ** (1.0 / d))}
    if n == 0:
        return out
    clipped = np.clip(unit, 0.0, 1.0)
    out["discrepancy"] = float(qmc.discrepancy(clipped, method="CD")) if n > 1 else None
    tree = cKDTree(unit)
    if n > 1:
        dist, _ = tree.query(unit, k=2)
        out["min_distance"] = float(dist[:, 1].min())
        out["mean_nn_distance"] = float(dist[:, 1].mean())
    probe = qmc.Sobol(d, seed=seed).random(n_probe)
    out["fill_distance"] = float(tree.query(probe)[0].max())
    p1 = {}
    for j, nm in enumerate(names):
        b1 = min(n, int((levels or {}).get(nm, n)))
        p1[nm] = float(len(np.unique(np.minimum((clipped[:, j] * b1).astype(int), b1 - 1))) / b1)
    out["projection_1d"] = p1
    b = bins_2d or max(2, int(math.floor(math.sqrt(n))))
    p2 = {}
    for i, j in itertools.combinations(range(d), 2):
        cells = np.minimum((clipped[:, [i, j]] * b).astype(int), b - 1)
        p2[f"{names[i]}|{names[j]}"] = float(len({tuple(c) for c in cells}) / (b * b))
    out["projection_2d"] = p2
    if p2:
        worst = min(p2, key=p2.get)
        out["projection_2d_mean"] = float(np.mean(list(p2.values())))
        out["projection_2d_worst"] = {"pair": worst, "filled": p2[worst]}
    out["projection_2d_bins"] = b
    return out


def redundancy(planned_unit: np.ndarray, existing_unit: np.ndarray | None = None, *, tol: float | None = None,
               existing_ids: list[str] | None = None) -> dict[str, Any]:
    """Which planned cases (nearly) repeat an existing case or another planned case.

    ``tol`` is a distance in the unit cube (default 1 % of its diagonal, 0.01 * sqrt(d)). Returns per planned case
    the nearest existing case and distance, and the flags; plus the near-duplicate pairs inside the existing data
    (compute that was already spent twice)."""
    planned_unit = np.atleast_2d(np.asarray(planned_unit, float))
    n, d = planned_unit.shape
    tol = float(tol) if tol is not None else 0.01 * math.sqrt(d)
    out: dict[str, Any] = {"tol": tol, "n_planned": int(n)}
    near_existing = np.full(n, np.nan)
    near_id: list[Any] = [None] * n
    if existing_unit is not None and len(existing_unit):
        ex = np.atleast_2d(np.asarray(existing_unit, float))
        dist, idx = cKDTree(ex).query(planned_unit)
        near_existing = dist
        near_id = [existing_ids[i] if existing_ids is not None else int(i) for i in idx]
        pairs = cKDTree(ex).query_pairs(tol)
        out["existing_duplicate_pairs"] = len(pairs)
        out["n_existing"] = int(len(ex))
        dup_ex = sorted({j for p in pairs for j in p})
        out["existing_redundant"] = [existing_ids[i] if existing_ids is not None else int(i) for i in dup_ex]
    within = np.zeros(n, dtype=bool)
    if n > 1:
        for i, j in cKDTree(planned_unit).query_pairs(tol):
            within[max(i, j)] = True                     # keep the first of a pair, flag the later one
    redundant_existing = np.nan_to_num(near_existing, nan=np.inf) <= tol
    out["cases"] = [{"nearest_existing": near_id[i],
                     "distance_to_existing": None if np.isnan(near_existing[i]) else float(near_existing[i]),
                     "repeats_existing": bool(redundant_existing[i]), "repeats_planned": bool(within[i])}
                    for i in range(n)]
    out["n_repeat_existing"] = int(redundant_existing.sum())
    out["n_repeat_planned"] = int(within.sum())
    return out


def prioritize(planned_unit: np.ndarray, existing_unit: np.ndarray | None = None) -> dict[str, Any]:
    """Run order: greedy maximin over the planned cases, starting from what already exists, so every prefix of the
    list covers the space as evenly as possible (stop the campaign early and the cases run are still spread out).
    ``score`` is each case's distance to everything before it (existing included): the information it adds."""
    planned_unit = np.atleast_2d(np.asarray(planned_unit, float))
    order = greedy_maximin(planned_unit, len(planned_unit), fixed=existing_unit)
    score = np.empty(len(planned_unit))
    seen = list(existing_unit) if existing_unit is not None and len(existing_unit) else []
    for k, i in enumerate(order):
        ref = np.array(seen + [planned_unit[j] for j in order[:k]])
        score[i] = float(np.linalg.norm(ref - planned_unit[i], axis=1).min()) if len(ref) else float("inf")
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    return {"order": order.tolist(), "rank": rank.tolist(),
            "score": [None if not np.isfinite(s) else float(s) for s in score]}
