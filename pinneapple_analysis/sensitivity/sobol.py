"""Variance-based (Sobol) sensitivity indices with bootstrap confidence intervals.

Design: two independent quasi-random matrices ``A`` and ``B`` (a scrambled Sobol sequence in ``2 d`` dimensions,
mapped to each input's distribution through its inverse CDF) and the matrices ``AB_i`` (``A`` with column ``i`` from
``B``); for second-order indices also ``BA_i``. Estimators:

- first order ``S_i``: Saltelli et al. (2010), Comput. Phys. Commun. 181:259, eq. (b) of table 2,
  ``mean(f(B) (f(AB_i) - f(A))) / V``;
- total ``ST_i``: Jansen (1999), ``mean((f(A) - f(AB_i))^2) / (2 V)``;
- second order ``S_ij``: Saltelli (2002), Comput. Phys. Commun. 145:280,
  ``mean(f(BA_i) f(AB_j) - f(A) f(B)) / V - S_i - S_j``;

with ``V`` the variance of ``[f(A), f(B)]``. Confidence intervals: percentile bootstrap over the rows of the design
(Archer et al. 1997). Cost: ``N (d + 2)`` model evaluations, ``N (2 d + 2)`` with second order.

:func:`first_order_given_data` estimates first-order indices from an existing sample (a dataset, a campaign's runs)
without a designed experiment, by the correlation-ratio estimator on equal-count bins.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc

from pinneapple_analysis.uncertainty.distributions import as_space, to_values


def _as_2d(Y, n: int) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2 or Y.shape[0] != n:
        raise ValueError(f"the model must return (n,) or (n, m) for n = {n} inputs; got {Y.shape}")
    return Y


def _estimate(yA, yB, yAB, yBA=None):
    """Indices from evaluations; arrays (..., N, m) for A/B and (d, ..., N, m) for AB/BA. Returns S1, ST (d, ..., m)
    and S2 (d, d, ..., m) or None."""
    V = np.concatenate([yA, yB], axis=-2).var(axis=-2)
    V = np.where(V > 0, V, np.nan)
    S1 = np.mean(yB * (yAB - yA), axis=-2) / V
    ST = 0.5 * np.mean((yA - yAB) ** 2, axis=-2) / V
    S2 = None
    if yBA is not None:
        d = yAB.shape[0]
        S2 = np.full((d, d) + S1.shape[1:], np.nan)
        for i in range(d):
            for j in range(i + 1, d):
                Vij = np.mean(yBA[i] * yAB[j] - yA * yB, axis=-2) / V
                S2[i, j] = S2[j, i] = Vij - S1[i] - S1[j]
    return S1, ST, S2


@dataclass
class SobolIndices:
    """Sobol indices per output (rows) and input (columns), with ``level`` bootstrap intervals ``*_ci`` of shape
    ``(2, m, d)`` (lower, upper). ``S2`` is ``(m, d, d)`` when second order was requested."""
    names: list[str]
    outputs: list[str]
    n_base: int
    n_evals: int
    variance: np.ndarray
    S1: np.ndarray
    ST: np.ndarray
    S1_ci: np.ndarray
    ST_ci: np.ndarray
    S2: np.ndarray | None = None
    S2_ci: np.ndarray | None = None
    level: float = 0.95
    n_boot: int = 0
    method: str = "Saltelli (2010) first order, Jansen (1999) total, Saltelli (2002) second order"
    X: np.ndarray = field(default=None, repr=False)
    Y: np.ndarray = field(default=None, repr=False)

    @property
    def interaction(self) -> np.ndarray:
        """``ST - S1``: the share of variance from interactions of each input with the others."""
        return self.ST - self.S1

    def ranking(self, output: int = 0) -> list[str]:
        """Inputs by decreasing total index for one output."""
        return [self.names[i] for i in np.argsort(-np.nan_to_num(self.ST[output], nan=-1.0))]

    def table(self, output: int = 0) -> list[dict]:
        rows = []
        for i in np.argsort(-np.nan_to_num(self.ST[output], nan=-1.0)):
            rows.append({"name": self.names[i], "S1": float(self.S1[output, i]),
                         "S1_ci": [float(self.S1_ci[0, output, i]), float(self.S1_ci[1, output, i])],
                         "ST": float(self.ST[output, i]),
                         "ST_ci": [float(self.ST_ci[0, output, i]), float(self.ST_ci[1, output, i])],
                         "interaction": float(self.interaction[output, i])})
        return rows

    def to_dict(self) -> dict:
        d = {"method": self.method, "names": self.names, "outputs": self.outputs, "n_base": self.n_base,
             "n_evals": self.n_evals, "variance": self.variance.tolist(), "level": self.level,
             "n_boot": self.n_boot, "S1": self.S1.tolist(), "ST": self.ST.tolist(),
             "S1_ci": self.S1_ci.tolist(), "ST_ci": self.ST_ci.tolist()}
        if self.S2 is not None:
            d["S2"] = np.nan_to_num(self.S2, nan=0.0).tolist()
            d["S2_ci"] = np.nan_to_num(self.S2_ci, nan=0.0).tolist()
        return d


def saltelli_design(space, n: int, second_order: bool = False, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Inputs ``X`` to evaluate, stacked as ``[A; B; AB_1..AB_d; (BA_1..BA_d)]``, and the layout to read them back.
    ``n`` is rounded up to a power of two (the balance property of the Sobol sequence)."""
    sp = as_space(space)
    d = len(sp)
    m = int(np.ceil(np.log2(max(int(n), 2))))
    U = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random_base2(m)
    N = U.shape[0]
    UA, UB = U[:, :d], U[:, d:]
    blocks = [UA, UB]
    for i in range(d):
        ab = UA.copy()
        ab[:, i] = UB[:, i]
        blocks.append(ab)
    if second_order:
        for i in range(d):
            ba = UB.copy()
            ba[:, i] = UA[:, i]
            blocks.append(ba)
    X = to_values(sp, np.vstack(blocks))
    return X, {"N": N, "d": d, "second_order": second_order, "names": [p.name for p in sp]}


def sobol_from_evaluations(Y, layout: dict, outputs: Sequence[str] | None = None, n_boot: int = 200,
                           level: float = 0.95, seed: int = 0, X: np.ndarray | None = None) -> SobolIndices:
    """Indices from model outputs ``Y`` on the inputs of :func:`saltelli_design` (e.g. runs done elsewhere: a solver
    campaign, a cluster job)."""
    N, d, so = layout["N"], layout["d"], layout["second_order"]
    nblocks = 2 + d * (2 if so else 1)
    Y = _as_2d(Y, N * nblocks)
    if not np.all(np.isfinite(Y)):
        raise ValueError(f"{int((~np.isfinite(Y)).sum())} model outputs are NaN or infinite; fix or drop those runs")
    m = Y.shape[1]
    R = Y.reshape(nblocks, N, m)
    yA, yB, yAB = R[0], R[1], R[2:2 + d]
    yBA = R[2 + d:] if so else None
    S1, ST, S2 = _estimate(yA, yB, yAB, yBA)
    alpha = (1 - level) / 2
    S1_ci = ST_ci = S2_ci = None
    if n_boot:
        idx = np.random.default_rng(seed).integers(0, N, size=(n_boot, N))
        bS1, bST, bS2 = _estimate(yA[idx], yB[idx], yAB[:, idx], None if yBA is None else yBA[:, idx])
        # bS1: (d, n_boot, m) -> quantiles over the bootstrap axis
        S1_ci = np.nanquantile(bS1, [alpha, 1 - alpha], axis=1)
        ST_ci = np.nanquantile(bST, [alpha, 1 - alpha], axis=1)
        if bS2 is not None:
            with warnings.catch_warnings():                                   # the diagonal is NaN by design
                warnings.simplefilter("ignore", RuntimeWarning)
                S2_ci = np.nanquantile(bS2, [alpha, 1 - alpha], axis=2)
    else:
        S1_ci = np.stack([S1, S1])
        ST_ci = np.stack([ST, ST])
        S2_ci = None if S2 is None else np.stack([S2, S2])
    def tr(a):                                                                # (..., d, m) -> (..., m, d)
        return np.moveaxis(a, -1, -2)

    out = SobolIndices(names=list(layout["names"]), outputs=list(outputs or [f"y{j}" for j in range(m)]),
                       n_base=N, n_evals=N * nblocks,
                       variance=np.concatenate([yA, yB]).var(axis=0), S1=S1.T, ST=ST.T,
                       S1_ci=tr(S1_ci), ST_ci=tr(ST_ci), level=level, n_boot=int(n_boot), X=X, Y=Y)
    if S2 is not None:
        out.S2 = np.moveaxis(S2, -1, 0)                                       # (m, d, d)
        out.S2_ci = np.moveaxis(S2_ci, -1, 1)                                 # (2, m, d, d)
    return out


def sobol_indices(fn, space, n: int = 1024, second_order: bool = False, n_boot: int = 200, level: float = 0.95,
                  seed: int = 0, outputs: Sequence[str] | None = None) -> SobolIndices:
    """First-order and total Sobol indices (and optionally second order) of ``fn`` over ``space``.

    Parameters
    ----------
    fn : model evaluated on an ``(n, d)`` array, returning ``(n,)`` or ``(n, m)``.
    space : uncertain inputs (see :func:`pinneapple_analysis.uncertainty.distributions.as_space`).
    n : base sample size ``N`` (rounded up to a power of two); cost ``N (d + 2)`` evaluations.
    n_boot, level : bootstrap resamples and confidence level of the intervals.
    """
    X, layout = saltelli_design(space, n, second_order, seed)
    return sobol_from_evaluations(fn(X), layout, outputs, n_boot, level, seed, X=X)


def first_order_given_data(X, y, names: Sequence[str] | None = None, n_bins: int | None = None,
                           n_boot: int = 200, level: float = 0.95, seed: int = 0) -> dict:
    """First-order indices from an existing sample ``(X, y)`` with independent inputs: ``Var(E[y | x_i]) / Var(y)``
    estimated on ``n_bins`` equal-count bins of each input (default ``sqrt(n)``, at most 50), with the bias of the bin
    means (within-bin variance / bin size) subtracted. Interactions and totals need a designed experiment
    (:func:`sobol_indices`); correlated inputs make these indices hard to interpret."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, d = X.shape
    if n < 50:
        raise ValueError(f"{n} rows are too few for given-data indices (need at least 50)")
    nb = int(n_bins or min(50, max(5, int(np.sqrt(n)))))
    names = list(names) if names is not None else [f"x{i}" for i in range(d)]

    def est(Xs, ys):
        V = ys.var()
        out = np.zeros(d)
        for i in range(d):
            order = np.argsort(Xs[:, i], kind="stable")
            parts = np.array_split(ys[order], nb)
            means = np.array([p.mean() for p in parts])
            sizes = np.array([len(p) for p in parts])
            within = np.array([p.var(ddof=1) if len(p) > 1 else 0.0 for p in parts])
            between = np.sum(sizes * (means - ys.mean()) ** 2) / len(ys)
            bias = within.sum() / len(ys)                    # E[between] = Var(E[y|x]) + sum(within) / n
            out[i] = (between - bias) / V if V > 0 else np.nan
        return out

    S1 = est(X, y)
    rng = np.random.default_rng(seed)
    boots = np.array([est(X[b], y[b]) for b in rng.integers(0, n, size=(n_boot, n))]) if n_boot else S1[None]
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(X, rowvar=False) if d > 1 else np.ones((1, 1))
    C = np.nan_to_num(np.atleast_2d(C))
    off = np.abs(C - np.eye(d))
    i, j = np.unravel_index(int(np.argmax(off)), off.shape)
    return {"method": "given data: correlation ratio on equal-count bins, bias-corrected", "names": names,
            "n": n, "n_bins": nb, "S1": S1.tolist(), "S1_ci": [lo.tolist(), hi.tolist()], "level": level,
            "input_correlation": C.tolist(),
            "max_input_correlation": {"pair": [names[i], names[j]], "r": float(C[i, j])} if d > 1 else None}


def sobol_from_runs(space, n: int, run_ids, Y, second_order: bool = False, seed: int = 0, n_boot: int = 200,
                    level: float = 0.95, outputs: Sequence[str] | None = None) -> SobolIndices:
    """Indices from runs of the design ``saltelli_design(space, n, second_order, seed)`` done elsewhere (OpenFOAM,
    CalculiX, a campaign runner): ``run_ids`` are the row numbers of the design (any order), ``Y`` their outputs.
    Every run of the design is needed; a missing or failed run raises ``ValueError`` naming it."""
    X, layout = saltelli_design(space, n, second_order, seed)
    ids = np.asarray(run_ids).astype(int).ravel()
    Y = _as_2d(Y, ids.size)
    if np.any((ids < 0) | (ids >= X.shape[0])):
        raise ValueError(f"run ids must be in 0..{X.shape[0] - 1}")
    full = np.full((X.shape[0], Y.shape[1]), np.nan)
    full[ids] = Y
    missing = np.nonzero(~np.all(np.isfinite(full), axis=1))[0]
    if missing.size:
        head = ", ".join(map(str, missing[:10])) + (" ..." if missing.size > 10 else "")
        raise ValueError(f"{missing.size} of {X.shape[0]} design runs are missing or not finite: {head}")
    return sobol_from_evaluations(full, layout, outputs, n_boot, level, seed, X=X)


__all__ = ["SobolIndices", "sobol_indices", "saltelli_design", "sobol_from_evaluations", "sobol_from_runs",
           "first_order_given_data"]
