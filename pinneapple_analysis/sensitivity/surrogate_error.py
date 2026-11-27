"""The surrogate's own error inside the Sobol intervals, and the stability of the ranking (#927).

:func:`sobol_indices` reports percentile-bootstrap intervals over the Saltelli sample. Those cover the Monte Carlo
error of the indices *for the model as given*; they do not cover the model's prediction error, so a surrogate with a
few dozen training points gets the same narrow intervals as the solver it stands for. This module propagates that
second source:

- :func:`gp_posterior_paths`: posterior sample paths of a fitted Gaussian process, the ``N_s`` realisations of
  Marrel, Iooss, Laurent & Roustant (2009), *Calculations of Sobol indices for the Gaussian process metamodel*,
  Reliab. Eng. Syst. Saf. 94(3):742. Computing the indices on every path gives a distribution instead of one number;
- :func:`ensemble_paths`: one path per member of an ensemble (MLP ensemble, deep ensemble, a zoo model with members);
- :func:`paths_from_residuals`: outputs perturbed with resampled calibration residuals (signed residuals of the
  Surrogate Studio / Uncertainty Lab). This one is weaker by construction: it ignores the residuals' correlation
  structure, and the docstring says so.

:func:`sobol_from_paths` (and its convenience wrapper :func:`sobol_with_paths`) turns those paths into

- a **combined interval**, from a nested bootstrap that resamples both the path and the rows of the design, reported
  in the usual ``S1_ci``/``ST_ci`` so that everything downstream (the freeze rule included) uses it;
- the **decomposition**: the width from Monte Carlo sampling alone (the old interval), from the surrogate alone, and
  the variance share of each;
- the **ranking stability**: per input the probability that its rank is unchanged and that it is in the top *k*,
  and the Kendall tau (1938) between each draw and the reported ranking.

The measurement that backs this is in ``tests/test_sensitivity_surrogate_error.py``: on Ishigami fitted with a GP on
30, 100 and 300 points, the coverage of the combined interval against the exact indices is counted, next to the
coverage of the bootstrap-only interval, which under-covers for the poor surrogate.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import kendalltau

from pinneapple_analysis.sensitivity.sobol import SobolIndices, _as_2d, _estimate, saltelli_design


# --------------------------------------------------------------------------------------------- error sources
def gp_posterior_paths(gp, X, n_paths: int = 100, seed: int = 0, jitter: float = 1e-9,
                       subtract_noise: bool = False) -> np.ndarray:
    """``n_paths`` draws of a fitted Gaussian process at ``X``, conditioned on its training data.

    ``gp`` is a fitted ``sklearn.gaussian_process.GaussianProcessRegressor`` (one output) or a sequence of them (one
    per output). Each draw is a function, so the indices computed on it see the surrogate's error as a whole, with
    the spatial correlation the GP believes in — unlike perturbing the output point by point.

    ``subtract_noise`` removes the fitted white-noise variance from the diagonal of the posterior covariance, giving
    paths of the *latent* function rather than of a noisy observation of it (the default matches what
    ``gp.predict(return_std=True)`` reports).
    """
    gps = [gp] if hasattr(gp, "predict") else list(gp)
    if not gps:
        raise ValueError("give at least one fitted Gaussian process")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    outs, noise = [], None
    for g in gps:
        if not hasattr(g, "L_"):
            raise ValueError("the Gaussian process must be fitted before its paths are drawn")
        K_trans = g.kernel_(X, g.X_train_)
        V = np.linalg.solve(np.tril(g.L_), K_trans.T)                       # L v = k(x, Z) -> (n_train, n)
        # ``alpha_`` lives in the space of the normalised training targets; take the paths back to the data's units
        ys = float(getattr(g, "_y_train_std", 1.0) or 1.0)
        ym = float(getattr(g, "_y_train_mean", 0.0) or 0.0)
        mean = ys * (K_trans @ g.alpha_) + ym
        cov = ys ** 2 * (g.kernel_(X) - V.T @ V)
        if subtract_noise:
            if noise is None:
                noise = _noise_variance(g, X)
            cov = cov - ys ** 2 * noise
        outs.append((mean, cov))
    rng = np.random.default_rng(seed)
    paths = np.empty((int(n_paths), X.shape[0], len(outs)))
    for j, (mean, cov) in enumerate(outs):
        L = _cholesky(cov, jitter)
        z = rng.standard_normal((X.shape[0], int(n_paths)))
        paths[:, :, j] = (mean[:, None] + L @ z).T
    return paths


def _noise_variance(gp, X) -> np.ndarray:
    """Variance the fitted WhiteKernel contributes at ``X`` (a scalar, or a per-point vector for ARD noise)."""
    k = gp.kernel_
    parts = getattr(k, "k2", None), getattr(k, "k1", None)
    for p in parts:
        if p is not None and p.__class__.__name__ == "WhiteKernel" and getattr(p, "noise_level", None):
            return float(p.noise_level)
    return np.zeros(X.shape[0])


def _cholesky(cov: np.ndarray, jitter: float) -> np.ndarray:
    """Lower Cholesky factor of a symmetric covariance matrix, with jitter increased until it succeeds."""
    cov = 0.5 * (cov + cov.T)
    scale = float(np.mean(np.diag(cov))) or 1.0
    j = jitter * scale
    for _ in range(6):
        try:
            return np.linalg.cholesky(cov + j * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            j = max(j * 100.0, 1e-12)
    raise np.linalg.LinAlgError("the posterior covariance is not positive definite, even with jitter")


def ensemble_paths(members, X, outputs: Sequence[str] | None = None) -> np.ndarray:
    """One path per ensemble member: ``members`` are callables ``X -> (n,) or (n, m)`` (the members of an ensemble,
    a zoo model's members). Indices on each path are the indices of that member; the spread across paths is the
    ensemble spread, moved from the output to the indices."""
    ms = list(members)
    if len(ms) < 2:
        raise ValueError(f"an ensemble needs at least two members; got {len(ms)}")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.stack([_as_2d(f(X), X.shape[0]) for f in ms], axis=0)


def paths_from_residuals(pred, residuals, n_paths: int = 100, seed: int = 0) -> np.ndarray:
    """``pred`` plus resampled calibration residuals, ``n_paths`` times.

    Use the **signed** residuals ``y - pred`` of the calibration split (``conformal_scores`` gives absolute scores;
    take the sign from the calibration set). Each draw resamples the residuals independently at every design point,
    which is the weak part: the surrogate's error is spatially correlated, and this ignores it. It over-states the
    variance of the indices when the errors are smooth and under-states it when they are biased in blocks.
    """
    pred = _as_2d(pred, np.shape(pred)[0] if np.ndim(pred) > 1 else np.size(pred))
    r = np.asarray(residuals, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError("give at least two finite calibration residuals")
    rng = np.random.default_rng(seed)
    draw = rng.choice(r, size=(int(n_paths),) + pred.shape)
    return pred[None] + draw


# --------------------------------------------------------------------------------------------- the estimator
@dataclass
class SurrogateError:
    """Where the width of the reported interval comes from, and how stable the ranking is.

    ``bootstrap_only``, ``surrogate_only`` and ``combined`` each hold ``S1_ci``/``ST_ci`` of shape ``(2, m, d)``
    like :class:`~pinneapple_analysis.sensitivity.sobol.SobolIndices`. ``variance_share`` splits the variance of the
    index over the combined draws into the part that the row resampling explains and the part the path draws
    explain; ``dominates`` says the surrogate part is the larger one.
    """
    method: str
    sources: list[str]
    n_paths: int
    n_draws: int
    level: float
    bootstrap_only: dict
    surrogate_only: dict
    combined: dict
    width: dict
    variance_share: dict
    ranking: dict
    dominates: bool
    names: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def table(self, output: int = 0) -> list[dict]:
        """Per input, in the reported order: the two intervals side by side and the ranking stability."""
        rows = []
        for i in np.argsort(-np.nan_to_num(np.asarray(self.combined["ST_med"])[output], nan=-1.0)):
            rk = self.ranking.get(self.outputs[output] if output < len(self.outputs) else str(output), {})
            rows.append({"name": self.names[i],
                         "ST_combined": float(self.combined["ST_med"][output, i]),
                         "ST_ci_combined": [float(self.combined["ST_ci"][0, output, i]),
                                            float(self.combined["ST_ci"][1, output, i])],
                         "ST_ci_bootstrap_only": [float(self.bootstrap_only["ST_ci"][0, output, i]),
                                                  float(self.bootstrap_only["ST_ci"][1, output, i])],
                         "ST_ci_surrogate_only": [float(self.surrogate_only["ST_ci"][0, output, i]),
                                                  float(self.surrogate_only["ST_ci"][1, output, i])],
                         "width_bootstrap_only": float(self.width["bootstrap_only"]["ST"][output, i]),
                         "width_surrogate_only": float(self.width["surrogate_only"]["ST"][output, i]),
                         "width_combined": float(self.width["combined"]["ST"][output, i]),
                         "share_surrogate": float(self.variance_share["surrogate"]["ST"][output, i]),
                         "rank_stability": (rk.get("rank_stability") or [0.0])[i] if rk else None,
                         "top_k": ({k: float(v[i]) for k, v in (rk.get("p_top_k") or {}).items()} if rk else None)})
        return rows

    def to_dict(self) -> dict:
        return {"method": self.method, "sources": list(self.sources), "n_paths": self.n_paths,
                "n_draws": self.n_draws, "level": self.level,
                "bootstrap_only": _ci_dict(self.bootstrap_only), "surrogate_only": _ci_dict(self.surrogate_only),
                "combined": {**_ci_dict(self.combined),
                             "S1": np.asarray(self.combined["S1"]).tolist(),
                             "ST": np.asarray(self.combined["ST"]).tolist()},
                "width": {k: {o: np.asarray(v).tolist() for o, v in d.items()} for k, d in self.width.items()},
                "variance_share": {k: {o: np.asarray(v).tolist() for o, v in d.items()}
                                   for k, d in self.variance_share.items()},
                "dominates": bool(self.dominates), "names": self.names, "outputs": self.outputs,
                "ranking": self.ranking}


def _ci_dict(d: Mapping) -> dict:
    return {k: np.asarray(v).tolist() for k, v in d.items() if k.endswith("_ci")}


def _gather(R: np.ndarray, path_idx: np.ndarray, row_idx: np.ndarray | None, d: int) -> tuple:
    """Select one path per draw and, when given, resample the rows of each draw. Shapes for :func:`_estimate`:
    ``(d, D, N, m)`` for the cross blocks and ``(D, N, m)`` for A and B."""
    R = R[path_idx]                                                          # (D, nblocks, N, m)
    A, B, AB = R[:, 0], R[:, 1], np.moveaxis(R[:, 2:2 + d], 1, 0)            # (D,N,m), (d,D,N,m)
    if row_idx is not None:
        take = row_idx[:, :, None]
        A, B, AB = (np.take_along_axis(a, take, axis=1) if a.ndim == 3 else
                    np.take_along_axis(a, take[None], axis=2) for a in (A, B, AB))
    return A, B, AB


def _draws(R, layout: dict, path_idx, row_idx) -> tuple[np.ndarray, np.ndarray]:
    d = layout["d"]
    S1, ST, _ = _estimate(*_gather(R, path_idx, row_idx, d))
    return S1, ST                                                            # (d, D, m)


def _shares(var_mc: np.ndarray, var_s: np.ndarray, var_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tot = np.where(np.isfinite(var_mc) & np.isfinite(var_s), var_mc + var_s, np.nan)
    tot = np.where(tot > 0, tot, np.nan)
    return var_mc / tot, var_s / tot


def ranking_stability(ST_draws, names: Sequence[str], top_k: Sequence[int] = (1, 2, 3),
                      reference: np.ndarray | None = None, outputs: Sequence[str] | None = None) -> dict:
    """How stable a ranking by total index is across draws.

    ``ST_draws`` is ``(n_draws, m, d)`` (one total-index vector per draw). For each output it reports, per input:
    ``rank_stability`` (the fraction of draws where its rank is the reference rank), ``p_top_k`` per ``k``,
    ``rank_median`` and the 5-95 % band of its rank; and per output the Kendall tau of every draw against the
    reference ranking. The reference is the ranking of the median total index (or of ``reference`` when given).
    """
    ST = np.asarray(ST_draws, dtype=float)
    if ST.ndim != 3:
        raise ValueError(f"ST_draws must be (n_draws, m, d); got {ST.shape}")
    D, m, d = ST.shape
    names = list(names)
    keys = list(outputs) if outputs is not None else [f"y{k}" for k in range(m)]
    if len(keys) != m:
        raise ValueError(f"got {len(keys)} output names for {m} outputs")
    out = {}
    for k in range(m):
        st = np.nan_to_num(ST[:, k, :], nan=-1.0)
        ref = np.asarray(reference[k] if reference is not None and np.ndim(reference) > 1 else
                         (reference if reference is not None else np.median(st, axis=0)), dtype=float)
        ref_order = np.argsort(-ref, kind="stable")
        ref_rank = np.empty(d, dtype=int)
        ref_rank[ref_order] = np.arange(d)
        ranks = np.argsort(np.argsort(-st, axis=1, kind="stable"), axis=1, kind="stable")
        tau = [float(kendalltau(ref_order, np.argsort(-st[r], kind="stable")).statistic) for r in range(D)]
        out[keys[k]] = {"reference": [names[i] for i in ref_order],
                        "rank_median": [names[i] for i in np.argsort(np.median(ranks, axis=0))],
                        "rank_stability": [float(np.mean(ranks[:, i] == ref_rank[i])) for i in range(d)],
                        "rank_ci": [[float(np.quantile(ranks[:, i], 0.05)),
                                     float(np.quantile(ranks[:, i], 0.95))] for i in range(d)],
                        "p_top_k": {int(kk): [float(np.mean(ranks[:, i] < int(kk))) for i in range(d)]
                                    for kk in top_k},
                        "kendall_tau": {"mean": float(np.mean(tau)),
                                        "p05": float(np.quantile(tau, 0.05)),
                                        "p95": float(np.quantile(tau, 0.95))}}
    return out


def sobol_from_paths(paths, layout: dict, outputs: Sequence[str] | None = None, level: float = 0.95,
                     n_draws: int = 200, top_k: Sequence[int] = (1, 2, 3), seed: int = 0,
                     method: str = "", sources: Sequence[str] | None = None) -> SobolIndices:
    """Sobol indices of a design whose outputs are not one number but a set of ``paths``.

    ``paths`` is ``(n_paths, n_design)`` or ``(n_paths, n_design, m)`` on the design of
    :func:`~pinneapple_analysis.sensitivity.sobol.saltelli_design` (the ``layout`` it returns). The reported
    ``S1``/``ST`` come from the mean over the paths (the surrogate's point prediction); their ``*_ci`` are the
    **combined** interval, resampling the path and the rows of the design together. ``result.surrogate_error`` holds
    the decomposition and the ranking stability, and ``result.to_dict()["sources"]`` says what the interval includes.
    """
    N, d, so = layout["N"], layout["d"], layout["second_order"]
    nblocks = 2 + d * (2 if so else 1)
    P = np.asarray(paths, dtype=float)
    if P.ndim == 2:
        P = P[:, :, None]
    if P.ndim != 3 or P.shape[1] != N * nblocks:
        raise ValueError(f"paths must be (n_paths, {N * nblocks}) or (n_paths, {N * nblocks}, m) for N = {N}, "
                         f"d = {d}; got {P.shape}")
    if not np.all(np.isfinite(P)):
        raise ValueError(f"{int((~np.isfinite(P)).sum())} values of the paths are NaN or infinite")
    if P.shape[0] < 2:
        raise ValueError(f"a surrogate-error interval needs at least two paths; got {P.shape[0]}")
    n_paths, m = P.shape[0], P.shape[2]
    R = P.reshape(n_paths, nblocks, N, m)
    mean = P.mean(axis=0)
    Rm = mean.reshape(1, nblocks, N, m)
    rng = np.random.default_rng(seed)

    S1, ST, _ = _estimate(Rm[:, 0], Rm[:, 1], np.moveaxis(Rm[:, 2:2 + d], 1, 0))
    S1, ST = S1[:, 0, :].T, ST[:, 0, :].T                                     # (m, d)

    # Monte Carlo only: the rows resampled on the point prediction (exactly the old interval)
    boot = rng.integers(0, N, size=(int(n_draws), N))
    mS1, mST = _draws(Rm, layout, np.zeros(int(n_draws), dtype=int), boot)
    # surrogate only: one path per draw, no row resampling
    pidx = rng.integers(0, n_paths, size=(int(n_draws),))
    sS1, sST = _draws(R, layout, pidx, np.tile(np.arange(N), (int(n_draws), 1)))
    # both at once: nested bootstrap
    cS1, cST = _draws(R, layout, pidx, boot)
    a = (1 - level) / 2
    q = [a, 1 - a]

    def cid(x):                                          # (d, D, m) -> (2, m, d)
        return np.moveaxis(np.nanquantile(x, q, axis=1), -1, -2)

    boot_only = {"S1_ci": cid(mS1), "ST_ci": cid(mST)}
    surr_only = {"S1_ci": cid(sS1), "ST_ci": cid(sST)}
    comb = {"S1_ci": cid(cS1), "ST_ci": cid(cST), "S1_med": np.median(cS1, axis=1).T,
            "ST_med": np.median(cST, axis=1).T, "S1": S1, "ST": ST}
    width = {"bootstrap_only": {o: boot_only[f"{o}_ci"][1] - boot_only[f"{o}_ci"][0] for o in ("S1", "ST")},
             "surrogate_only": {o: surr_only[f"{o}_ci"][1] - surr_only[f"{o}_ci"][0] for o in ("S1", "ST")},
             "combined": {o: comb[f"{o}_ci"][1] - comb[f"{o}_ci"][0] for o in ("S1", "ST")}}
    share_mc, share_s = {}, {}
    for o, x_mc, x_s, x_c in (("S1", mS1, sS1, cS1), ("ST", mST, sST, cST)):
        v_mc = np.nanvar(x_mc, axis=1).T                                     # (m, d)
        v_s = np.nanvar(x_s, axis=1).T
        mc, s = _shares(v_mc, v_s, np.nanvar(x_c, axis=1).T)
        share_mc[o], share_s[o] = mc, s
    dominated = bool(np.nanmax(share_s["ST"]) > 0.5 or np.nanmax(share_s["S1"]) > 0.5)
    names = list(layout["names"])
    outs = list(outputs or [f"y{j}" for j in range(m)])
    err = SurrogateError(
        method=method or "nested bootstrap over surrogate paths and the Saltelli sample",
        sources=list(sources or ["monte_carlo", "surrogate"]), n_paths=n_paths, n_draws=int(n_draws),
        level=level, bootstrap_only=boot_only, surrogate_only=surr_only, combined=comb, width=width,
        variance_share={"monte_carlo": share_mc, "surrogate": share_s}, dominates=dominated,
        ranking=ranking_stability(np.moveaxis(cST, 0, 2), names, top_k, reference=np.median(cST, axis=1).T,
                                  outputs=outs),
        names=names, outputs=outs)
    out = SobolIndices(names=names, outputs=outs, n_base=N, n_evals=0, variance=np.concatenate(
        [mean[:N], mean[N:2 * N]]).var(axis=0) if N else np.zeros(m), S1=S1, ST=ST,
        S1_ci=comb["S1_ci"], ST_ci=comb["ST_ci"], level=level, n_boot=int(n_draws), X=None, Y=mean,
        sources=list(err.sources), surrogate_error=err)
    out.method = f"{out.method}; intervals: {err.method} over {n_paths} paths"
    return out


def sobol_with_paths(paths_fn: Callable[[np.ndarray], np.ndarray], space, n: int = 1024,
                     second_order: bool = False, level: float = 0.95, n_draws: int = 200, seed: int = 0,
                     outputs: Sequence[str] | None = None, top_k: Sequence[int] = (1, 2, 3),
                     method: str = "", sources: Sequence[str] | None = None) -> SobolIndices:
    """:func:`sobol_from_paths` on the design of ``saltelli_design(space, n, second_order, seed)``:
    ``paths_fn`` receives the ``(n_design, d)`` inputs and returns ``(n_paths, n_design[, m])``."""
    X, layout = saltelli_design(space, n, second_order, seed)
    return sobol_from_paths(paths_fn(X), layout, outputs, level, n_draws, top_k, seed, method, sources)


__all__ = ["SurrogateError", "gp_posterior_paths", "ensemble_paths", "paths_from_residuals", "ranking_stability",
           "sobol_from_paths", "sobol_with_paths"]
