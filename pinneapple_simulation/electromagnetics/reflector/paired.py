"""The paired PO/direct experiment on the two-dimensional compact range: the sweep, the learned residual and the
multi-fidelity comparison.

The question the experiment asks is a question about numbers, not about a solver: for this design, how far is
physical optics from an exact surface-current solution, does the difference follow the design variables closely
enough to be learned, and at a fixed budget of exact solves does a learned correction beat spending them blindly.

* :func:`paired_sweep` -- every design solved twice, PO and exact (:mod:`.planar2d`), written to a JSON-lines
  checkpoint as it finishes;
* :class:`ResidualModel` / :func:`leave_out_validation` -- a Gaussian process on ``direct - PO`` and its error on
  designs it never saw, including a whole edge profile held out;
* :func:`compare_strategies` -- ``direct``, ``po`` and ``multi`` at the same number of exact solves, repeated over
  seeds;
* :func:`cost_summary` -- the measured cost of each solver, so the budget can be stated in seconds as well as in
  solves.
"""
from __future__ import annotations

import itertools
import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from . import planar2d as P

__all__ = ["METRICS", "paired_sweep", "load_rows", "ResidualModel", "fit_residual", "leave_out_validation",
           "objective", "compare_strategies", "cost_summary"]

METRICS = ("amp_pp_dB", "taper_dB", "ripple_pp_dB", "phase_ripple_pp_deg", "usable_width")


def _row(d: P.Design, po: Mapping[str, Any], dr: Mapping[str, Any]) -> dict[str, Any]:
    row = {"tag": d.tag, **d.inputs(), "n_segments": int(po["n_segments"]),
           "t_po_s": po["seconds"], "t_direct_s": dr["seconds"], "mom_s": dr["mom_seconds"],
           "condition": dr["condition"], "po_distance": dr["po_distance"]}
    for k in METRICS:
        row[f"po_{k}"] = po[k]
        row[f"direct_{k}"] = dr[k]
        row[f"residual_{k}"] = dr[k] - po[k]
    return row


def paired_sweep(designs: Sequence[tuple[P.Design, P.Range2D]], out_jsonl: str, *, workers: int = 1,
                 log: Callable[[str], None] | None = None) -> list[dict]:
    """Solve every design by PO and by the exact solver, appending each result to ``out_jsonl`` as it finishes and
    skipping the designs already there (a killed run resumes)."""
    from concurrent.futures import ProcessPoolExecutor
    done = {r["tag"]: r for r in load_rows(out_jsonl)}
    todo = [(d, rng) for d, rng in designs if d.tag not in done]
    if log:
        log(f"paired sweep: {len(todo)} designs to solve, {len(done)} in the checkpoint")
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)) or ".", exist_ok=True)

    def keep(row):
        with open(out_jsonl, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        done[row["tag"]] = row
        if log:
            log(f"  {row['tag']}: ripple {row['po_ripple_pp_dB']:.2f} -> {row['direct_ripple_pp_dB']:.2f} dB,"
                f" phase ripple {row['po_phase_ripple_pp_deg']:.1f} -> {row['direct_phase_ripple_pp_deg']:.1f} deg,"
                f" usable {row['po_usable_width']:.3f} -> {row['direct_usable_width']:.3f} m")
    if workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(workers) as ex:
            for row in ex.map(_solve, todo):
                keep(row)
    else:
        for job in todo:
            keep(_solve(job))
    return [done[d.tag] for d, _ in designs if d.tag in done]


def _solve(job) -> dict[str, Any]:
    d, rng = job
    po = P.po_metrics(rng)
    dr = P.direct_metrics(rng)
    return _row(d, po, dr)


def load_rows(path: str) -> list[dict]:
    """The rows of a JSON-lines checkpoint, skipping a line cut by a kill."""
    out = []
    if os.path.isfile(path):
        with open(path) as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except json.JSONDecodeError:
                        continue
    return out


# ------------------------------------------------------------------------------------ the learned correction
class ResidualModel:
    """Gaussian process on the residual ``direct - PO`` of one quantity, over the design inputs of
    :class:`~.planar2d.Design`.

    Squared-exponential kernel, one isotropic length scale and a nugget by maximum marginal likelihood over a
    small grid, inputs standardised by the range of the training set. :meth:`predict` returns the mean and the
    standard deviation of the residual.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.length_scale = 0.3
        self.nugget = 1e-3
        self.y_mean = 0.0
        self.X: np.ndarray | None = None

    def _kernel(self, A: np.ndarray, B: np.ndarray, length_scale: float) -> np.ndarray:
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / max(length_scale, 1e-6) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray, *,
            grid: Sequence[float] = (0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8),
            nuggets: Sequence[float] = (1e-5, 1e-4, 1e-3, 1e-2)) -> ResidualModel:
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float)
        lo, hi = X.min(0), X.max(0)
        self._lo = lo
        self._span = np.where(hi - lo > 0, hi - lo, 1.0)
        Xs = (X - lo) / self._span
        self.y_mean = float(y.mean()) if len(y) else 0.0
        yc = y - self.y_mean
        self._sd = float(np.std(yc)) if len(y) > 1 else 1.0
        best = (-np.inf, 0.3, 1e-3)
        for ls, ng in itertools.product(grid, nuggets):
            K = self._kernel(Xs, Xs, ls) + ng * np.eye(len(Xs))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
            ll = -0.5 * yc @ alpha - np.log(np.diag(L)).sum()
            if ll > best[0]:
                best = (ll, ls, ng)
        _, self.length_scale, self.nugget = best
        self.X = Xs
        K = self._kernel(Xs, Xs, self.length_scale) + self.nugget * np.eye(len(Xs))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, yc))
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, float))
        Xs = (X - self._lo) / self._span
        Ks = self._kernel(Xs, self.X, self.length_scale)
        mean = self.y_mean + Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        std = np.sqrt(np.maximum(1.0 - (v ** 2).sum(0), 0.0)) * self._sd
        return mean, std


def _features(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.array([[r["size"], r["q"], r["roll_factor"], r["blend"], r["aspect"]] for r in rows], float)


def fit_residual(rows: Sequence[Mapping[str, Any]], metric: str = "ripple_pp_dB") -> ResidualModel:
    y = np.array([r[f"residual_{metric}"] for r in rows], float)
    return ResidualModel(metric).fit(_features(rows), y)


def leave_out_validation(rows: Sequence[Mapping[str, Any]], metric: str = "ripple_pp_dB", *, folds: int = 5,
                         seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    """``PO`` against ``PO + learned residual`` on designs the model never saw.

    Two splits: ``random`` (``folds``-fold over the designs, averaged over ``seeds``) and ``profile`` (one whole
    edge profile -- every size and every feed of one roll/blend/aspect triple -- held out at a time, which is the
    honest test of "a profile we have not built yet").
    """
    X = _features(rows)
    truth = np.array([r[f"direct_{metric}"] for r in rows], float)
    po = np.array([r[f"po_{metric}"] for r in rows], float)
    prof = np.array([[r["roll_factor"], r["blend"], r["aspect"]] for r in rows], float)
    base = {"metric": metric, "n": len(rows), "po_rmse": float(np.sqrt(np.mean((po - truth) ** 2))),
            "po_mae": float(np.mean(np.abs(po - truth))), "po_bias": float(np.mean(po - truth))}

    def score(pred):
        ok = np.isfinite(pred)
        rmse = float(np.sqrt(np.mean((pred[ok] - truth[ok]) ** 2)))
        return {"rmse": rmse, "mae": float(np.mean(np.abs(pred[ok] - truth[ok]))),
                "bias": float(np.mean(pred[ok] - truth[ok])),
                "rmse_vs_po": float(np.sqrt(np.mean((po[ok] - truth[ok]) ** 2))),
                "fraction_removed": float(1 - rmse / max(np.sqrt(np.mean((po[ok] - truth[ok]) ** 2)), 1e-12))}

    preds = []
    for s in seeds:
        rng = np.random.default_rng(s)
        fold = rng.permutation(len(rows)) % folds
        out = np.full(len(rows), np.nan)
        for f in range(folds):
            tr, te = fold != f, fold == f
            m = ResidualModel(metric).fit(X[tr], (truth - po)[tr])
            mu, _ = m.predict(X[te])
            out[te] = po[te] + mu
        preds.append(out)
    random_split = np.nanmean(np.stack(preds), 0)
    loo = np.full(len(rows), np.nan)
    for p in np.unique(prof, axis=0):
        te = np.all(prof == p, axis=1)
        if te.all() or (~te).sum() < 12:
            continue
        m = ResidualModel(metric).fit(X[~te], (truth - po)[~te])
        mu, _ = m.predict(X[te])
        loo[te] = po[te] + mu
    base["random"] = score(random_split)
    base["profile"] = score(loo)
    base["rows"] = [{"tag": r["tag"], "truth": float(t), "po": float(p), "corrected": float(c),
                     "corrected_profile": float(lp)}
                    for r, t, p, c, lp in zip(rows, truth, po, random_split, loo, strict=True)]
    return base


# ----------------------------------------------------------------------------------- the optimization study
def objective(row: Mapping[str, Any], source: str, *, amp_limit: float = 3.0, penalty: float = 0.1) -> float:
    """What the strategies maximize: the usable quiet-zone width, discounted for the peak-to-peak amplitude ripple
    above ``amp_limit`` dB -- a zone that holds a tolerance only where the ripple is small."""
    return float(row[f"{source}_usable_width"]) - penalty * max(0.0, float(row[f"{source}_amp_pp_dB"]) - amp_limit)


def compare_strategies(rows: Sequence[Mapping[str, Any]], *, n_direct: int = 12, seeds: Sequence[int] = (0, 1, 2, 3),
                       amp_limit: float = 3.0, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Four strategies at ``n_direct`` exact solves each, repeated over ``seeds``.

    The exact result of every design is already in ``rows``, so a strategy's exact solves are *revealed* to it;
    the strategies differ only in which designs they spend them on. All of them know the PO value of the whole
    space, because PO is cheap -- that is the information a real campaign has.

    * ``direct`` -- the budget is a random sample of the space;
    * ``po`` -- the budget goes to the PO optimum and its nearest neighbours in rank (trust the cheap model);
    * ``multi`` -- the first half of the budget is a random paired sample to fit a residual model, the second half
      goes to the top of the *corrected* ranking (one batch, one model);
    * ``sequential`` -- refits the residual model on every design already solved exactly and spends each further
      solve on the best-predicted un-solved design, one at a time (the workflow a campaign actually runs).
    """
    truth = np.array([objective(r, "direct", amp_limit=amp_limit) for r in rows])
    po = np.array([objective(r, "po", amp_limit=amp_limit) for r in rows])
    X = _features(rows)
    out: dict[str, Any] = {"n_designs": len(rows), "n_direct": n_direct, "amp_limit": amp_limit,
                           "oracle": float(truth.max()), "po_best": float(po.max()),
                           "po_regret_of_best": float(truth.max() - truth[int(np.argmax(po))]),
                           "po_rank_of_oracle": int((po > po[int(np.argmax(truth))]).sum()) + 1, "runs": []}
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        perm = rng.permutation(len(rows))
        n_fit = max(4, min(n_direct - 2, n_direct // 2))
        picks = {"direct": perm[:n_direct].astype(int)}
        picks["po"] = np.unique(np.concatenate([[int(np.argmax(po))], perm[: n_direct - 1]]).astype(int))
        fit_idx = perm[:n_fit].astype(int)
        model = ResidualModel("objective").fit(X[fit_idx], (truth - po)[fit_idx])
        mu, _ = model.predict(X)
        ranked = [i for i in np.argsort(-(po + mu)) if i not in set(fit_idx.tolist())]
        picks["multi"] = np.unique(np.concatenate([fit_idx, np.asarray(ranked[: n_direct - n_fit], int)]).astype(int))
        # sequential: refit after every reveal
        seen = list(fit_idx)
        while len(seen) < n_direct:
            m = ResidualModel("objective").fit(X[seen], (truth - po)[seen])
            mu_s, _ = m.predict(X)
            cand = [i for i in np.argsort(-(po + mu_s)) if i not in set(seen)]
            seen.append(int(cand[0]))
        picks["sequential"] = np.asarray(sorted(seen), int)
        run = {"seed": int(s)}
        for name, pick in picks.items():
            v = truth[pick]
            run[name] = {"best": float(v.max()), "mean": float(v.mean()), "worst": float(v.min()),
                         "n": int(len(pick)), "rank_of_winner": int((truth > v.max()).sum()),
                         "percent_of_oracle": float(100 * v.max() / max(truth.max(), 1e-12))}
        out["runs"].append(run)
        if log:
            log(f"  seed {s}: direct {run['direct']['best']:.4f}, po {run['po']['best']:.4f}, "
                f"multi {run['multi']['best']:.4f}, sequential {run['sequential']['best']:.4f} "
                f"(oracle {out['oracle']:.4f})")
    for name in ("direct", "po", "multi", "sequential"):
        v = np.array([r[name]["best"] for r in out["runs"]])
        out[f"{name}_mean"] = float(v.mean())
        out[f"{name}_median"] = float(np.median(v))
        out[f"{name}_worst"] = float(v.min())
        out[f"{name}_best"] = float(v.max())
        out[f"{name}_percent_mean"] = float(100 * v.mean() / max(out["oracle"], 1e-12))
    return out


def cost_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The measured cost of each solver on this space (median, min and max over the designs), the ratio, and the
    segmented-contour size."""
    tp = np.array([r["t_po_s"] for r in rows], float)
    td = np.array([r["t_direct_s"] for r in rows], float)
    n = np.array([r["n_segments"] for r in rows], float)
    rate = td / np.maximum(tp, 1e-12)
    return {"n_designs": len(rows),
            "segments": {"min": int(n.min()), "median": int(np.median(n)), "max": int(n.max())},
            "po_seconds": {"median": float(np.median(tp)), "min": float(tp.min()), "max": float(tp.max())},
            "direct_seconds": {"median": float(np.median(td)), "min": float(td.min()), "max": float(td.max())},
            "ratio_median": float(np.median(rate)), "ratio_min": float(rate.min()), "ratio_max": float(rate.max()),
            "total_po_s": float(tp.sum()), "total_direct_s": float(td.sum()),
            "po_distance_median": float(np.median([r["po_distance"] for r in rows])),
            "po_distance_min": float(np.min([r["po_distance"] for r in rows])),
            "po_distance_max": float(np.max([r["po_distance"] for r in rows])),
            "seconds_per_design": {"po": float(np.median(tp)), "direct": float(np.median(td))}}
