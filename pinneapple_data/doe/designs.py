"""Designs of experiments over a :class:`~pinneapple_data.contracts.ParameterSpace`.

Every design is built in the unit hypercube of the free parameters and mapped to values by the space (log scales,
integers, categories), then filtered by the space's bounds and constraints (and an optional ``feasible`` callable
for checks that are not expressions, such as a geometry validity test).

=======================  =====================================================================================
``lhs``                  Latin hypercube (scipy.stats.qmc), one point per stratum in every 1-D projection
``maximin``              the Latin hypercube with the largest minimum distance among ``n_restarts`` random ones
``sobol`` / ``halton``   scrambled low-discrepancy sequences (Sobol balance needs n = 2^m)
``random``               uniform random (reference only: clusters and gaps)
``full_factorial``       every combination of ``levels`` per parameter (all values of categories / integers)
``fractional_factorial`` two-level 2^(k-p) design with generated factors; reports the generators and resolution
``infill``               sequential: new points as far as possible from the existing cases (greedy maximin over a
                         Sobol candidate pool); with ``outputs`` the distance is weighted up where the existing
                         outputs vary most among neighbours (adaptive refinement)
=======================  =====================================================================================

When constraints remove points, more candidates are drawn and the design keeps the feasible ones: for ``lhs``,
``sobol``, ``halton`` and ``random`` the first ``n`` in sequence order, for ``maximin`` and ``infill`` a greedy
maximin subset. The Latin-hypercube stratification is then only approximate; the coverage metrics show by how much.
"""
from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from ..contracts import ParameterSpace

METHODS = ("lhs", "maximin", "sobol", "halton", "random", "full_factorial", "fractional_factorial", "infill")


@dataclass
class Design:
    """The cases of a design: values (DataFrame with every parameter), unit coordinates and how they were made."""
    cases: Any                                   # pandas DataFrame
    unit: np.ndarray                             # (n, dim) in [0, 1]
    method: str
    info: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cases)


def design(space: ParameterSpace, n: int, method: str = "lhs", *, seed: int = 0, existing=None,
           outputs: Sequence[str] = (), levels: int | dict[str, int] = 3, n_restarts: int = 20,
           feasible: Callable[[Any], np.ndarray] | None = None, max_candidates: int = 200_000) -> Design:
    """``n`` cases of ``space`` by ``method`` (see the module docstring). ``existing`` is a DataFrame of cases
    already run (a column per parameter, plus ``outputs`` for adaptive infill)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; one of {METHODS}")
    if space.dim == 0:
        raise ValueError("the parameter space has no free parameter")
    d = space.dim
    if method == "full_factorial":
        unit, info = _full_factorial_unit(space, levels)
        return _finish(space, unit, method, info, feasible)
    if method == "fractional_factorial":
        unit, info = fractional_factorial_unit(d, n)
        return _finish(space, unit, method, info, feasible)
    if n < 1:
        raise ValueError("n must be at least 1")
    if method == "infill":
        if existing is None or len(existing) == 0:
            raise ValueError("infill needs existing cases")
        ex_unit = space.to_unit(existing)
        weights = _output_weights(ex_unit, existing, outputs) if outputs else None
        m = 1 << int(math.ceil(math.log2(max(4096, 64 * n))))
        pool = _feasible_pool(space, qmc.Sobol(d, seed=seed).random_base2(int(math.log2(m))), feasible)
        idx = greedy_maximin(pool, n, fixed=ex_unit, weights_fixed=weights)
        info = {"pool": len(pool), "existing": len(ex_unit), "adaptive": bool(outputs), "outputs": list(outputs)}
        return _make(space, pool[idx], method, info)

    def draw(m: int, k: int) -> np.ndarray:
        s = seed + 7919 * k
        if method in ("lhs", "maximin"):
            return qmc.LatinHypercube(d, seed=s).random(m)
        if method == "sobol":
            with warnings.catch_warnings():                   # reported in Design.info instead
                warnings.simplefilter("ignore", UserWarning)
                return qmc.Sobol(d, seed=s).random(m)
        if method == "halton":
            return qmc.Halton(d, seed=s).random(m)
        return np.random.default_rng(s).random((m, d))

    if method == "maximin":
        best, best_d, tried = None, -1.0, 0
        for k in range(n_restarts):
            cand = _fill_to(space, n, draw, k, feasible, max_candidates, maximin_select=True)
            tried += 1
            md = min_distance(cand)
            if md > best_d:
                best, best_d = cand, md
        return _make(space, best, method, {"restarts": tried, "min_distance": best_d})
    unit = _fill_to(space, n, draw, 0, feasible, max_candidates, maximin_select=False)
    info = {}
    if method == "sobol" and n & (n - 1):
        info["note"] = f"n = {n} is not a power of 2: the Sobol balance properties hold only for 2^m points"
    return _make(space, unit, method, info)


def _fill_to(space, n, draw, k, feasible, max_candidates, maximin_select) -> np.ndarray:
    m, unit = n, None
    while True:
        cand = draw(m, k)
        ok = _mask(space, cand, feasible)
        if ok.sum() >= n or m >= max_candidates:
            unit = cand[ok]
            break
        rate = max(ok.mean(), 1.0 / m)
        m = min(int(math.ceil(1.3 * n / rate)) + 8, max_candidates)
    if len(unit) < n:
        raise ValueError(f"only {len(unit)} of {n} cases satisfy the constraints after {m} candidates; "
                         "relax the constraints or lower the number of cases")
    if len(unit) == n:
        return unit
    if maximin_select:
        return unit[greedy_maximin(unit, n)]
    return unit[:n]


def _mask(space: ParameterSpace, unit: np.ndarray, feasible) -> np.ndarray:
    cases = space.from_unit(unit)
    ok = space.feasible(cases)
    if feasible is not None:
        ok &= np.asarray(feasible(cases), dtype=bool)
    return ok


def _feasible_pool(space, pool, feasible) -> np.ndarray:
    pool = pool[_mask(space, pool, feasible)]
    if not len(pool):
        raise ValueError("no candidate satisfies the constraints")
    return pool


def _make(space: ParameterSpace, unit: np.ndarray, method: str, info: dict) -> Design:
    cases = space.from_unit(unit)
    return Design(cases=cases, unit=space.to_unit(cases), method=method, info=info)


def _finish(space, unit, method, info, feasible) -> Design:
    ok = _mask(space, unit, feasible)
    info = {**info, "removed_by_constraints": int((~ok).sum())}
    if not ok.any():
        raise ValueError("no factorial point satisfies the constraints")
    return _make(space, unit[ok], method, info)


def _full_factorial_unit(space: ParameterSpace, levels: int | dict[str, int]) -> tuple[np.ndarray, dict]:
    axes, used = [], {}
    for p in space.free:
        if p.kind == "categorical":
            k = len(p.values)
            ax = (np.arange(k) + 0.5) / k
        elif p.kind == "integer" and (int(p.high - p.low) + 1) <= (levels.get(p.name, 3) if isinstance(levels, dict)
                                                                       else levels):
            k = int(p.high - p.low) + 1
            ax = (np.arange(k) + 0.5) / k
        else:
            k = int(levels.get(p.name, 3) if isinstance(levels, dict) else levels)
            if k < 2:
                raise ValueError(f"{p.name}: at least 2 levels")
            if p.kind == "integer":
                vals = np.unique(np.round(np.linspace(p.low, p.high, k)))
                ax = p.to_unit(vals)
                k = len(vals)
            else:
                ax = np.linspace(0.0, 1.0, k)
        axes.append(ax)
        used[p.name] = k
    size = int(np.prod([len(a) for a in axes]))
    if size > 100_000:
        raise ValueError(f"full factorial of {size:,} cases: lower the levels")
    unit = np.array(list(itertools.product(*axes)), dtype=float)
    return unit, {"levels": used, "size": size}


def fractional_factorial_unit(k: int, n: int) -> tuple[np.ndarray, dict]:
    """Two-level 2^(k-p) design with at least ``n`` runs (or 2^k if smaller): the first k-p factors are crossed in
    full, each further factor is the product of a distinct subset of them (largest subsets first, which keeps the
    resolution high). Returns unit coordinates (0 / 1) and the generators and resolution."""
    if k < 2:
        raise ValueError("a fractional factorial needs at least 2 free parameters")
    base = min(k, max(1, int(math.ceil(math.log2(max(n, 2))))))
    if 2 ** base < k + 1 and base < k:
        base = min(k, int(math.ceil(math.log2(k + 1))))       # enough runs to estimate all main effects
    names = [chr(ord("A") + i) if i < 26 else f"X{i}" for i in range(k)]
    signs = np.array(list(itertools.product((-1, 1), repeat=base)), dtype=int)[:, ::-1]
    subsets = [frozenset(c) for r in range(base, 1, -1) for c in itertools.combinations(range(base), r)]
    if k - base > len(subsets):
        raise ValueError(f"cannot alias {k} factors in {2 ** base} runs")
    extra = list(range(base, k))
    if math.comb(len(subsets), len(extra)) <= 20_000:         # small: every choice of generators, best resolution
        choices = itertools.combinations(subsets, len(extra))
        best = max(choices, key=lambda ch: _aberration([sub | {f} for sub, f in zip(ch, extra, strict=True)]))
    else:                                                     # large: greedy, one generator at a time
        best = []
        for f in extra:
            left = [sub for sub in subsets if sub not in best]
            best.append(max(left, key=lambda sub: (_aberration([b | {g} for b, g in zip(best, extra, strict=False)]
                                                                + [sub | {f}]), len(sub))))
    cols, generators, words = [signs[:, i] for i in range(base)], {}, []
    for sub, f in zip(best, extra, strict=True):
        cols.append(np.prod(signs[:, sorted(sub)], axis=1))
        generators[names[f]] = "".join(names[i] for i in sorted(sub))
        words.append(sub | {f})
    design_ = np.column_stack(cols)
    info = {"runs": 2 ** base, "factors": k, "p": k - base, "generators": generators,
            "resolution": _resolution(words), "factor_names": names}
    return (design_ + 1) / 2.0, info


def _aberration(words: list[frozenset]) -> tuple:
    """Sort key of a set of defining words: higher resolution first, then fewer words of that length."""
    group = {frozenset()}
    for w in words:
        group |= {g ^ w for g in group}
    lengths = sorted(len(g) for g in group if g)
    return (lengths[0], -lengths.count(lengths[0])) if lengths else (99, 0)


def _resolution(words: list[frozenset]) -> int | None:
    """Length of the shortest word of the defining relation generated by ``words`` (None for a full factorial)."""
    group = {frozenset()}
    for w in words:
        group |= {g ^ w for g in group}
    lengths = [len(g) for g in group if g]
    return min(lengths) if lengths else None


def greedy_maximin(points: np.ndarray, n: int, fixed: np.ndarray | None = None,
                   weights_fixed: np.ndarray | None = None) -> np.ndarray:
    """Indices of ``n`` of ``points`` picked one by one, each the farthest from those already picked and from
    ``fixed``. ``weights_fixed`` (one per fixed point, >= 1) favours the neighbourhoods to refine: a candidate's
    distance to the fixed points is multiplied by the weight of its nearest fixed point."""
    points = np.asarray(points, float)
    n = min(n, len(points))
    if fixed is not None and len(fixed):
        tree = cKDTree(fixed)
        dist, near = tree.query(points)
        if weights_fixed is not None:
            dist = dist * np.asarray(weights_fixed, float)[near]
        start = int(np.argmax(dist))
    else:
        dist = np.full(len(points), np.inf)
        start = int(np.argmin(np.linalg.norm(points - 0.5, axis=1)))       # the most central point first
    chosen = [start]
    dist = np.minimum(dist, np.linalg.norm(points - points[start], axis=1))
    for _ in range(n - 1):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return np.asarray(chosen)


def _output_weights(ex_unit: np.ndarray, existing, outputs: Sequence[str], k: int = 6) -> np.ndarray:
    """Per existing case: 1 + (how much the outputs vary among its k nearest neighbours, normalised to [0, 1])."""
    y = np.column_stack([np.asarray(existing[o], float) for o in outputs])
    y = (y - np.nanmean(y, 0)) / (np.nanstd(y, 0) + 1e-12)
    k = min(k + 1, len(ex_unit))
    _, nb = cKDTree(ex_unit).query(ex_unit, k=k)
    nb = np.atleast_2d(nb)
    var = np.nanmean(np.nanstd(y[nb], axis=1), axis=1)
    var = np.nan_to_num(var, nan=0.0)
    rng = var.max() - var.min()
    return 1.0 + 2.0 * ((var - var.min()) / rng if rng > 0 else np.zeros_like(var))


def min_distance(unit: np.ndarray) -> float:
    if len(unit) < 2:
        return float("nan")
    dist, _ = cKDTree(unit).query(unit, k=2)
    return float(dist[:, 1].min())
