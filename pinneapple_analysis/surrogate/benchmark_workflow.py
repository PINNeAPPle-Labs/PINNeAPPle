"""The standard model-benchmarking workflow (#1191): candidates on one protocol, ranked with intervals, a winner
named only when it is separated from the others.

Candidates are library families trained here (``gp``, ``mlp_ensemble``, ``quadratic``) and/or models that already
exist (``pinneapple_analysis.surrogate.model_source`` predictors: zoo, exported, TorchScript, callables), evaluated on
the same points. Protocol, recorded in the result:

* data: a Sobol design over a tool (a fresh design per seed) or a table (a fresh split per seed); every candidate of a
  seed sees the same training rows and the same test points;
* metrics per candidate and seed: relative RMSE and max error on the test points (inside the box), the same on an
  extrapolation set ``extrapolate`` beyond the box (tools only), interval coverage when the model gives one, fit and
  prediction time;
* ranking: test squared errors pooled over seeds; for the best candidate against each other one, a paired bootstrap
  of the mean difference; the best is the winner only when every 95 % interval excludes zero, otherwise the tied
  candidates are named.
"""
from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

FAMILIES = ("gp", "mlp_ensemble", "quadratic")


def _call(fn, kw, output):
    r = fn(**kw)
    return float(r[output]) if isinstance(r, Mapping) else float(r)


def benchmark_models(*, families: Sequence[str] = FAMILIES, predictors: Sequence[Any] = (),
                     tool: str | Callable[..., Any] | None = None, bounds: Mapping[str, tuple[float, float]] | None
                     = None, output: str | None = None, fixed: Mapping[str, Any] | None = None,
                     table: Mapping[str, Sequence[float]] | None = None, inputs: Sequence[str] | None = None,
                     n_train: int = 60, n_test: int = 200, seeds: Sequence[int] = (0, 1, 2),
                     extrapolate: float = 0.1, n_boot: int = 2000) -> dict[str, Any]:
    from scipy.stats import qmc

    from pinneapple_autolab.study.benchmark import make_scalar
    fixed = dict(fixed or {})
    fn = None
    if tool is not None:
        from pinneapple_autolab.study.verify import resolve_model
        fn = resolve_model(tool)
        inputs = list(bounds)
    lo = np.array([bounds[k][0] for k in inputs], float) if bounds else None
    hi = np.array([bounds[k][1] for k in inputs], float) if bounds else None
    names = list(families) + [f"{p.source}:{p.name}" for p in predictors]
    errs: dict[str, list[np.ndarray]] = {n: [] for n in names}
    ext: dict[str, list[np.ndarray]] = {n: [] for n in names}
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        if fn is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Xtr = qmc.scale(qmc.Sobol(len(inputs), seed=seed).random(n_train), lo, hi)
                Xte = qmc.scale(qmc.Sobol(len(inputs), seed=seed + 1000).random(n_test), lo, hi)
            ytr = np.array([_call(fn, {**fixed, **dict(zip(inputs, x, strict=True))}, output) for x in Xtr])
            yte = np.array([_call(fn, {**fixed, **dict(zip(inputs, x, strict=True))}, output) for x in Xte])
            rng = np.random.default_rng(seed + 2000)
            span = hi - lo
            # outside the box: each point beyond one face by up to `extrapolate` of the span
            Xex = lo + rng.random((max(20, n_test // 4), len(inputs))) * span
            j = rng.integers(0, len(inputs), len(Xex))
            side = rng.integers(0, 2, len(Xex))
            off = rng.random(len(Xex)) * extrapolate * span[j]
            Xex[np.arange(len(Xex)), j] = np.where(side == 0, lo[j] - off, hi[j] + off)
            yex = []
            for x in Xex:                               # beyond the box a tool may refuse (a negative rate): drop it
                try:
                    yex.append(_call(fn, {**fixed, **dict(zip(inputs, x, strict=True))}, output))
                except (ValueError, ArithmeticError):
                    yex.append(np.nan)
            yex = np.array(yex)
            keep = np.isfinite(yex)
            n_ext_dropped = int((~keep).sum())
            Xex, yex = Xex[keep], yex[keep]
        else:
            X = np.column_stack([np.asarray(table[k], float) for k in inputs])
            y = np.asarray(table[output], float)
            idx = np.random.default_rng(seed).permutation(len(X))
            k = int(0.75 * len(X))
            Xtr, ytr, Xte, yte = X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]
            lo, hi = Xtr.min(0), Xtr.max(0)
            Xex = yex = None
            n_ext_dropped = 0
        scale = max(float(np.ptp(np.r_[ytr, yte])), 1e-300)
        def U(Z, lo=lo, hi=hi):                    # the unit cube of this seed's training box
            return (Z - lo) / (hi - lo)
        for fam in families:
            m = make_scalar(fam, seed)
            t0 = time.perf_counter()
            m.fit(U(Xtr), ytr)
            fit_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            mu, sd = m.predict(U(Xte))
            pred_s = (time.perf_counter() - t1) / len(Xte)
            e = (mu - yte) / scale
            errs[fam].append(e)
            cover = float(np.mean(np.abs(mu - yte) <= 1.645 * sd)) if sd is not None else None
            row = {"candidate": fam, "seed": seed, "rel_rmse": float(np.sqrt(np.mean(e ** 2))),
                   "max_rel": float(np.max(np.abs(e))), "coverage90_gaussian": cover, "fit_s": fit_s,
                   "predict_s": pred_s}
            if Xex is not None and len(Xex):
                row["extrapolation_points"] = int(len(Xex))
                row["extrapolation_dropped"] = n_ext_dropped
                mx, _ = m.predict(U(Xex))
                ee = (mx - yex) / scale
                ext[fam].append(ee)
                row["extrapolation_rel_rmse"] = float(np.sqrt(np.mean(ee ** 2)))
            rows.append(row)
        for p in predictors:
            nm = f"{p.source}:{p.name}"
            t1 = time.perf_counter()
            mu = np.asarray(p.predict({k: Xte[:, i] for i, k in enumerate(inputs)})[output], float)
            pred_s = (time.perf_counter() - t1) / len(Xte)
            e = (mu - yte) / scale
            errs[nm].append(e)
            rows.append({"candidate": nm, "seed": seed, "rel_rmse": float(np.sqrt(np.mean(e ** 2))),
                         "max_rel": float(np.max(np.abs(e))), "coverage90_gaussian": None, "fit_s": 0.0,
                         "predict_s": pred_s, "note": "trained elsewhere: evaluated only"})
    pooled = {n: np.concatenate(v) ** 2 for n, v in errs.items()}
    mean = {n: float(np.mean(v)) for n, v in pooled.items()}
    order = sorted(names, key=lambda n: mean[n])
    best = order[0]
    rng = np.random.default_rng(12345)
    n = len(pooled[best])
    pairs = []
    for other in order[1:]:
        d = pooled[other] - pooled[best]                 # paired: the same test points
        boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
        lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
        pairs.append({"best": best, "other": other, "mean_sq_diff": float(d.mean()),
                      "ci95": [float(lo_ci), float(hi_ci)], "separated": bool(lo_ci > 0)})
    tied = [p["other"] for p in pairs if not p["separated"]]
    summary = [{"candidate": c, "rel_rmse_mean": float(np.mean([r["rel_rmse"] for r in rows if r["candidate"] == c])),
                "rel_rmse_spread": float(np.std([r["rel_rmse"] for r in rows if r["candidate"] == c])),
                "extrapolation_rel_rmse": (float(np.mean(np.sqrt([np.mean(e ** 2) for e in ext[c]])))
                                           if ext[c] else None)} for c in order]
    return {"protocol": {"n_train": n_train, "n_test": n_test, "seeds": list(seeds), "extrapolate": extrapolate,
                         "data": "tool" if fn is not None else "table", "candidates": names,
                         "ranking": "pooled test squared error; paired bootstrap of the difference, 95 %"},
            "rows": rows, "summary": summary, "pairs": pairs,
            "winner": best if not tied else None, "tied_with_best": [best] + tied if tied else [],
            "verdict": f"{best} wins (separated from every other candidate)" if not tied else
                       f"no single winner: {', '.join([best] + tied)} are not separated at 95 %"}
