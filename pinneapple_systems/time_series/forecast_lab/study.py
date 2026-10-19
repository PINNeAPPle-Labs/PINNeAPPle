"""The evaluation protocol: fix the test period first, tune on purged walk-forward folds of the rest, fit once,
score once on the untouched test period, and test the differences.

* :class:`Protocol` -- the split: the last ``test_frac`` of the series is the test period, chosen before any
  tuning; the development part ends with a validation block (early stopping, spread calibration, choosing a
  family's representative). Training labels never reach into the validation block (purge = largest horizon by
  construction), validation labels never reach into the test period.
* :func:`tune` -- the same configuration budget for every model, scored by purged walk-forward cross-validation
  on the development part (each fold with its own purged inner early-stopping block).
* :func:`evaluate` -- MAE, RMSE, CRPS, central-interval coverage and PIT per horizon.
* :func:`run_zoo` -- all of it for a list of :class:`ModelSpec`, plus skill against persistence and climatology,
  Diebold-Mariano tests against persistence and against the best baseline on the validation block, Holm (per
  horizon, across the zoo) and Benjamini-Hochberg (all model x horizon tests) adjustments, and for each family the
  last horizon with significant skill.
* :func:`overfitting_study` -- train / cross-validation / test errors of training variants under several
  cross-validation schemes (shuffled k-fold, walk-forward without purge, purged walk-forward, purged k-fold with
  embargo), so the train-test gap and each scheme's optimism are measured.
* :func:`surrogate_trained` -- the same model trained on an IAAFT (or shuffled) surrogate of the development data
  and scored on the real test period: a model that does not beat its surrogate-trained twin has learned nothing
  beyond the linear correlations (or nothing at all).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..metrics_ext.comparison import benjamini_hochberg, diebold_mariano, holm
from ..metrics_ext.probabilistic import pit_histogram
from ..validation.splitters import (
    PurgedKFoldSplitter,
    PurgedWalkForwardSplitter,
    ShuffledKFoldSplitter,
)
from .base import Forecaster, SeriesData, forecast_origins, targets_at

__all__ = ["Protocol", "ModelSpec", "tune", "evaluate", "run_zoo", "merge_zoo", "overfitting_study",
           "surrogate_trained", "skill_horizon"]

LEVELS = (0.5, 0.8, 0.95)


@dataclass
class Protocol:
    """Split of a series of length ``n`` into development (train + validation) and an untouched test period.

    ``min_history`` is the first usable origin for every model (the longest input window in the zoo), so all
    models train and are scored on the same origins. ``n_folds`` purged walk-forward folds of ``fold_frac`` of the
    development origins each are used for tuning. ``test_stride`` thins the test origins (consecutive h-step
    errors are correlated anyway; the DM test accounts for the overlap).
    """
    horizons: Sequence[int]
    test_frac: float = 0.2
    val_frac: float = 0.15
    n_folds: int = 3
    fold_frac: float = 0.12
    inner_val_frac: float = 0.15
    min_history: int = 64
    test_stride: int = 1
    max_train: int | None = None

    @property
    def H(self) -> int:
        return int(np.max(self.horizons))

    def dev_end(self, n: int) -> int:
        return int(round(n * (1 - self.test_frac)))

    def final_split(self, n: int) -> dict[str, np.ndarray]:
        dev = self.dev_end(n)
        val_start = dev - int(round(self.val_frac * dev))
        train = forecast_origins(0, val_start, self.horizons, min_history=self.min_history)
        if self.max_train is not None:
            train = train[-int(self.max_train):]
        val = forecast_origins(val_start, dev, self.horizons)
        test = forecast_origins(dev, n, self.horizons, stride=self.test_stride)
        return {"train": train, "val": val, "test": test, "dev_end": dev, "val_start": val_start}

    def inner(self, origins: np.ndarray) -> dict[str, np.ndarray]:
        """Split training origins into inner-train / inner-validation (early stopping) with a purge of H."""
        k = int(round(self.inner_val_frac * len(origins)))
        val = origins[len(origins) - k:]
        train = origins[origins < val[0] - self.H] if k else origins
        return {"train": train, "val": val}

    def cv_folds(self, n: int) -> list[dict[str, np.ndarray]]:
        dev = self.dev_end(n)
        o = forecast_origins(0, dev, self.horizons, min_history=self.min_history)
        sp = PurgedWalkForwardSplitter(n_folds=self.n_folds, val_size=int(self.fold_frac * len(o)), purge=self.H,
                                       window=self.max_train)
        folds = []
        for s in sp.split(o):
            inner = self.inner(o[s.train_idx])
            folds.append({"train": inner["train"], "inner_val": inner["val"], "val": o[s.val_idx]})
        return folds

    def describe(self, n: int) -> dict[str, Any]:
        s = self.final_split(n)
        return {"n": n, "horizons": list(map(int, self.horizons)), "dev_end": s["dev_end"],
                "val_start": s["val_start"], "n_train_origins": len(s["train"]), "n_val_origins": len(s["val"]),
                "n_test_origins": len(s["test"]), "test_stride": self.test_stride, "n_folds": self.n_folds,
                "purge": self.H, "min_history": self.min_history}


@dataclass
class ModelSpec:
    """A forecaster in the zoo: ``factory(**config)`` builds an unfitted model; ``grid`` lists the configurations
    to try (one entry means no tuning)."""
    name: str
    family: str
    factory: Callable[..., Forecaster]
    grid: list[dict[str, Any]] = field(default_factory=lambda: [{}])


def _rmse_skill(model: Forecaster, data: SeriesData, origins) -> float:
    """Mean over horizons of RMSE(model) / RMSE(persistence): < 1 beats persistence."""
    y = targets_at(data, origins, model.horizons)
    f = model.predict(data, origins).mean
    p = data.y[origins][:, None]
    num = np.sqrt(np.nanmean((f - y) ** 2, axis=0))
    den = np.sqrt(np.mean((p - y) ** 2, axis=0)) + 1e-12
    return float(np.mean(num / den))


def tune(spec: ModelSpec, data: SeriesData, protocol: Protocol, *, budget: int | None = None,
         seed: int = 0) -> dict[str, Any]:
    """Score every configuration (or ``budget`` of them, drawn at random) by purged walk-forward CV."""
    grid = list(spec.grid)
    if budget is not None and len(grid) > budget:
        pick = np.random.default_rng(seed).choice(len(grid), budget, replace=False)
        grid = [grid[i] for i in sorted(pick)]
    t0 = time.perf_counter()
    if len(grid) == 1:
        return {"best": grid[0], "table": [{"config": grid[0], "cv_score": None}], "seconds": 0.0, "n_configs": 1}
    folds = protocol.cv_folds(len(data))
    table = []
    for cfg in grid:
        scores, error = [], None
        for f in folds:
            try:
                m = spec.factory(**cfg).fit(data, f["train"], f["inner_val"], protocol.horizons)
                scores.append(_rmse_skill(m, data, f["val"]))
            except Exception as e:  # noqa: BLE001 -- a configuration that fails is recorded and skipped
                scores.append(float("nan"))
                error = repr(e)[:200]
        row = {"config": cfg, "cv_score": float(np.nanmean(scores)) if np.isfinite(scores).any() else None,
               "fold_scores": scores}
        if error:
            row["error"] = error
        table.append(row)
    ok = [r for r in table if r.get("cv_score") is not None]
    best = min(ok, key=lambda r: r["cv_score"])["config"] if ok else grid[0]
    return {"best": best, "table": table, "seconds": time.perf_counter() - t0, "n_configs": len(grid)}


def evaluate(model: Forecaster, data: SeriesData, origins, *, levels=LEVELS, keep: bool = False
             ) -> dict[str, Any]:
    """Point and probabilistic scores per horizon on ``origins``."""
    origins = np.asarray(origins, int)
    y = targets_at(data, origins, model.horizons)
    t0 = time.perf_counter()
    f = model.predict(data, origins)
    pred_s = time.perf_counter() - t0
    err = f.mean - y
    crps = f.crps(y)
    pit = f.pit(y)
    out: dict[str, Any] = {
        "mae": np.nanmean(np.abs(err), axis=0).tolist(),
        "rmse": np.sqrt(np.nanmean(err ** 2, axis=0)).tolist(),
        "crps": np.nanmean(crps, axis=0).tolist(),
        "coverage": {f"{lv:g}": (f.coverage(y, lv).tolist() if f.coverage(y, lv) is not None else None)
                     for lv in levels},
        "pit": {"pooled": pit_histogram(pit) if pit is not None else None,
                "per_horizon": [pit_histogram(pit[:, j]) for j in range(pit.shape[1])] if pit is not None else None},
        "predict_seconds": pred_s, "n_origins": int(len(origins)),
        "diverged_fraction": model.info.get("diverged_fraction", 0.0),
    }
    if keep:
        out["_err"], out["_crps"], out["_forecast"], out["_y"] = err, crps, f, y
    return out


def skill_horizon(horizons, skill, p_values=None, alpha: float = 0.05) -> int | None:
    """Largest horizon h such that the skill is positive (and, with ``p_values``, significant) at every horizon
    up to h; None when there is no skill at the first horizon."""
    last = None
    for j, h in enumerate(horizons):
        ok = skill[j] is not None and np.isfinite(skill[j]) and skill[j] > 0
        if p_values is not None:
            ok = ok and p_values[j] is not None and np.isfinite(p_values[j]) and p_values[j] < alpha
        if not ok:
            break
        last = int(h)
    return last


def run_zoo(data: SeriesData, specs: Sequence[ModelSpec], protocol: Protocol, *, budget: int | None = None,
            keep_forecasts: Sequence[str] = (), keep_models: bool = False, log: Callable[[str], None] = print,
            seed: int = 0) -> dict[str, Any]:
    """Tune, fit and score every model of the zoo on one series. Returns a JSON-able dict (``_forecasts`` holds
    the test forecasts of ``keep_forecasts`` for figures, ``_models`` the fitted models when ``keep_models``; both
    are dropped by ``to_jsonable(drop_private=True)``). Zoos run in parts (e.g. one process per model) are
    combined by :func:`merge_zoo`."""
    H = list(map(int, protocol.horizons))
    split = protocol.final_split(len(data))
    res: dict[str, Any] = {"series": data.name, "dt": data.dt, "protocol": protocol.describe(len(data)),
                           "models": {}, "_forecasts": {}, "_models": {}}
    errs: dict[str, dict[str, np.ndarray]] = {}
    for spec in specs:
        t0 = time.perf_counter()
        try:
            tuned = tune(spec, data, protocol, budget=budget, seed=seed)
            model = spec.factory(**tuned["best"]).fit(data, split["train"], split["val"], H)
            ev_test = evaluate(model, data, split["test"], keep=True)
            ev_val = evaluate(model, data, split["val"])
            tr = split["train"][:: max(1, len(split["train"]) // 1500)]
            ev_train = evaluate(model, data, tr)
        except Exception as e:  # noqa: BLE001 -- a model that cannot be fitted is reported, not hidden
            log(f"  {spec.name}: FAILED {e!r}")
            res["models"][spec.name] = {"family": spec.family, "error": repr(e)[:500]}
            continue
        errs[spec.name] = {"se": ev_test.pop("_err"), "crps": ev_test.pop("_crps")}
        fc, y = ev_test.pop("_forecast"), ev_test.pop("_y")
        if spec.name in keep_forecasts:
            res["_forecasts"][spec.name] = {"forecast": fc, "y": y, "origins": split["test"]}
        if keep_models:
            res["_models"][spec.name] = model
        res["_forecasts"].setdefault("_y", y)
        res["models"][spec.name] = {
            "family": spec.family, "n_params": int(model.n_params()), "fit_seconds": model.fit_seconds,
            "tune_seconds": tuned["seconds"], "n_configs": tuned["n_configs"], "best_config": tuned["best"],
            "cv_table": tuned["table"], "val_score": _rmse_skill(model, data, split["val"]),
            "info": model.describe(), "test": ev_test, "val": {k: ev_val[k] for k in ("rmse", "crps")},
            "train": {k: ev_train[k] for k in ("rmse", "crps")},
            "total_seconds": time.perf_counter() - t0,
        }
        log(f"  {spec.name:32s} {spec.family:12s} params={model.n_params():8d} "
            f"fit={model.fit_seconds:7.1f}s val_score={res['models'][spec.name]['val_score']:.3f} "
            f"test_rmse={np.round(ev_test['rmse'], 3).tolist()}")
    _add_comparisons(res, errs, H)
    return res


def merge_zoo(parts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine :func:`run_zoo` results of the same series and protocol run in parts (each with its models' test
    forecasts kept): the models and forecasts are pooled and the comparisons (skill, DM tests against persistence
    and the best baseline, Holm and BH adjustments, family representatives) are recomputed over the whole zoo."""
    first = parts[0]
    H = list(map(int, first["protocol"]["horizons"]))
    res: dict[str, Any] = {"series": first["series"], "dt": first["dt"], "protocol": first["protocol"],
                           "models": {}, "_forecasts": {}, "_models": {}}
    errs = {}
    for p in parts:
        if p["protocol"] != first["protocol"]:
            raise ValueError("parts were run with different protocols")
        for name, m in p["models"].items():
            m = {k: v for k, v in m.items() if k not in ("dm_vs_persistence", "dm_vs_best_baseline")}
            res["models"][name] = m
            if "test" not in m:
                continue
            rec = p["_forecasts"].get(name)
            if rec is None:
                raise ValueError(f"{name}: run_zoo was called without keep_forecasts for it")
            res["_forecasts"][name] = rec
            errs[name] = {"se": rec["forecast"].mean - rec["y"], "crps": rec["forecast"].crps(rec["y"])}
        res["_models"].update(p.get("_models", {}))
        res["_forecasts"].setdefault("_y", p["_forecasts"].get("_y"))
    _add_comparisons(res, errs, H)
    return res


def _add_comparisons(res, errs, H):
    models = res["models"]
    ok = {k: v for k, v in models.items() if "test" in v}
    if "persistence" not in ok:
        return
    pers = np.asarray(ok["persistence"]["test"]["rmse"])
    pers_mae = np.asarray(ok["persistence"]["test"]["mae"])
    clim = np.asarray(ok["climatology"]["test"]["crps"]) if "climatology" in ok else None
    baselines = [k for k, v in ok.items() if v["family"] == "baseline"]
    best_base = []
    for j in range(len(H)):
        best_base.append(min(baselines, key=lambda b: ok[b]["val"]["rmse"][j]))
    res["best_baseline_by_val"] = dict(zip(map(str, H), best_base, strict=False))
    for name, v in ok.items():
        t = v["test"]
        t["skill_rmse"] = (1 - np.asarray(t["rmse"]) / pers).tolist()
        t["skill_mae"] = (1 - np.asarray(t["mae"]) / pers_mae).tolist()
        if clim is not None:
            t["crps_skill_vs_climatology"] = (1 - np.asarray(t["crps"]) / clim).tolist()
        dm_p, dm_b = [], []
        for j, h in enumerate(H):
            e1 = errs[name]["se"][:, j]
            stride = res["protocol"]["test_stride"]
            lag = int(np.ceil(h / stride)) - 1
            rp = diebold_mariano(e1, errs["persistence"]["se"][:, j], max_lag=lag) if name != "persistence" else None
            b = best_base[j]
            rb = diebold_mariano(e1, errs[b]["se"][:, j], max_lag=lag) if name != b else None
            dm_p.append(rp.to_dict() if rp else None)
            dm_b.append({**rb.to_dict(), "baseline": b} if rb else None)
        v["dm_vs_persistence"] = dm_p
        v["dm_vs_best_baseline"] = dm_b
    # multiple-comparison control across the zoo: per horizon (Holm) and over all model x horizon tests (BH)
    learned = [k for k, v in ok.items() if v["family"] != "baseline"]
    for key in ("dm_vs_persistence", "dm_vs_best_baseline"):
        flat = {}
        for j in range(len(H)):
            pj = {m: (ok[m][key][j] or {}).get("p_value_better", np.nan) for m in learned}
            adj = holm(pj)
            for m in learned:
                if ok[m][key][j] is not None:
                    ok[m][key][j]["p_holm"] = adj[m]
                    flat[(m, j)] = ok[m][key][j]["p_value_better"]
        if flat:
            bh = benjamini_hochberg({f"{m}|{j}": p for (m, j), p in flat.items()})
            for (m, j) in flat:
                ok[m][key][j]["q_bh"] = bh[f"{m}|{j}"]
    # skill horizon of every model and of each family's representative (chosen on the validation block)
    for v in ok.values():
        sk = v["test"]["skill_rmse"]
        pp_ = [(d or {}).get("p_holm") for d in v["dm_vs_persistence"]]
        pb = [(d or {}).get("p_holm") for d in v["dm_vs_best_baseline"]]
        v["skill_horizon_vs_persistence"] = skill_horizon(H, sk, pp_ if v["family"] != "baseline" else None)
        base_rmse = np.array([ok[best_base[j]]["test"]["rmse"][j] for j in range(len(H))])
        v["skill_vs_best_baseline"] = (1 - np.asarray(v["test"]["rmse"]) / base_rmse).tolist()
        v["skill_horizon_vs_best_baseline"] = (skill_horizon(H, v["skill_vs_best_baseline"], pb)
                                               if v["family"] != "baseline" else None)
    fam = {}
    for name, v in ok.items():
        f = v["family"]
        if f not in fam or v["val_score"] < ok[fam[f]]["val_score"]:
            fam[f] = name
    res["family_representative"] = fam


# ---------------------------------------------------------------------------
# Anti-overfitting study
# ---------------------------------------------------------------------------

def _mean_rmse(model, data, origins) -> float:
    y = targets_at(data, origins, model.horizons)
    f = model.predict(data, origins).mean
    return float(np.mean(np.sqrt(np.nanmean((f - y) ** 2, axis=0)))) / float(np.std(data.y))


def overfitting_study(data: SeriesData, protocol: Protocol, variants: dict[str, Callable[[], Forecaster]], *,
                      schemes: Sequence[str] = ("purged_walk_forward",), schemes_for: Sequence[str] | None = None,
                      log: Callable[[str], None] = print) -> dict[str, Any]:
    """For each training variant: its error on its training origins, on the validation block and on the untouched
    test period, the gap test - train, and the cross-validation estimate under each scheme with its optimism
    (test - CV). The error is the mean over horizons of RMSE / RMSE of persistence *on the same origins* (1 = no
    better than persistence), so periods of different volatility compare; ``rmse_over_std`` also gives the mean
    RMSE divided by the standard deviation of the whole series.

    Schemes: ``"purged_walk_forward"`` (the protocol's folds), ``"walk_forward_no_purge"``, ``"purged_kfold"``
    (contiguous folds, purge H before and embargo H after each validation fold), ``"shuffled_kfold"``. They are run
    for the variants in ``schemes_for`` (all when None).
    """
    H = protocol.horizons
    split = protocol.final_split(len(data))
    dev_o = forecast_origins(0, split["dev_end"], H, min_history=protocol.min_history)
    out: dict[str, Any] = {"series": data.name, "variants": {}}
    for name, make in variants.items():
        t0 = time.perf_counter()
        m = make().fit(data, split["train"], split["val"], H)
        tr = split["train"][:: max(1, len(split["train"]) // 1500)]
        row = {"train": _rmse_skill(m, data, tr), "val": _rmse_skill(m, data, split["val"]),
               "test": _rmse_skill(m, data, split["test"]),
               "rmse_over_std": {"train": _mean_rmse(m, data, tr), "val": _mean_rmse(m, data, split["val"]),
                                 "test": _mean_rmse(m, data, split["test"])}, "n_params": int(m.n_params()),
               "fit_seconds": m.fit_seconds, "info": {k: v for k, v in m.info.items()
                                                      if k in ("epochs_run", "best_epoch", "n_members")}}
        row["gap_test_minus_train"] = row["test"] - row["train"]
        row["cv"] = {}
        if schemes_for is None or name in schemes_for:
            for sch in schemes:
                row["cv"][sch] = _cv_estimate(make, data, protocol, dev_o, sch)
                row["cv"][sch]["optimism_test_minus_cv"] = row["test"] - row["cv"][sch]["estimate"]
        row["seconds"] = time.perf_counter() - t0
        out["variants"][name] = row
        log(f"  {name:34s} train={row['train']:.3f} val={row['val']:.3f} test={row['test']:.3f} "
            f"cv={ {k: round(v['estimate'], 3) for k, v in row['cv'].items()} }")
    return out


def _cv_estimate(make, data, protocol, dev_o, scheme) -> dict[str, Any]:
    H = protocol.H
    k = protocol.n_folds
    if scheme == "purged_walk_forward":
        folds = [(f["train"], f["inner_val"], f["val"]) for f in protocol.cv_folds(len(data))]
    else:
        if scheme == "walk_forward_no_purge":
            sp = PurgedWalkForwardSplitter(n_folds=k, val_size=int(protocol.fold_frac * len(dev_o)), purge=0)
        elif scheme == "purged_kfold":
            sp = PurgedKFoldSplitter(n_folds=k, purge=H, embargo=H)
        elif scheme == "shuffled_kfold":
            sp = ShuffledKFoldSplitter(n_folds=k)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        folds = []
        for s in sp.split(dev_o):
            tr = dev_o[s.train_idx]
            if scheme == "shuffled_kfold":     # the naive practice: early stopping on a random part as well
                rng = np.random.default_rng(s.fold)
                iv = np.sort(rng.choice(tr, max(1, int(protocol.inner_val_frac * len(tr))), replace=False))
                folds.append((np.setdiff1d(tr, iv), iv, dev_o[s.val_idx]))
            elif scheme == "walk_forward_no_purge":
                n_iv = int(protocol.inner_val_frac * len(tr))
                folds.append((tr[: len(tr) - n_iv], tr[len(tr) - n_iv:], dev_o[s.val_idx]))
            else:                              # purged k-fold: inner early-stopping block purged on both sides
                n_iv = int(protocol.inner_val_frac * len(tr))
                iv = tr[len(tr) - n_iv:]
                train = tr[(tr < iv[0] - H) | (tr > iv[-1] + H)]
                folds.append((train, iv, dev_o[s.val_idx]))
    scores = []
    for tr, iv, va in folds:
        m = make().fit(data, np.sort(tr), None if iv is None else np.sort(iv), protocol.horizons)
        scores.append(_rmse_skill(m, data, np.sort(va)))
    return {"estimate": float(np.mean(scores)), "folds": scores}


# ---------------------------------------------------------------------------
# Surrogate-trained twin
# ---------------------------------------------------------------------------

def surrogate_trained(make: Callable[[], Forecaster], data: SeriesData, protocol: Protocol, *,
                      kind: str = "iaaft", seed: int = 0) -> dict[str, Any]:
    """Skill (vs persistence, mean over horizons and per horizon) on the real test period of a univariate model
    trained on the real development data and of the same model trained on a surrogate of it."""
    from ..predictability import iaaft_surrogate, shuffle_surrogate
    H = protocol.horizons
    split = protocol.final_split(len(data))
    dev = split["dev_end"]
    uni = SeriesData(data.y[:, None], 0, data.dt, data.name)
    gen = iaaft_surrogate if kind == "iaaft" else shuffle_surrogate
    sur = SeriesData(gen(data.y[:dev], np.random.default_rng(seed))[:, None], 0, data.dt, data.name + "_surrogate")
    y = targets_at(uni, split["test"], H)
    pers = np.sqrt(np.mean((uni.y[split["test"]][:, None] - y) ** 2, axis=0))
    out = {}
    for tag, train_data in (("real", uni), (kind, sur)):
        m = make().fit(train_data, split["train"], split["val"], H)
        f = m.predict(uni, split["test"]).mean
        skill = 1 - np.sqrt(np.mean((f - y) ** 2, axis=0)) / pers
        out[tag] = {"skill_rmse": skill.tolist(), "mean_skill": float(np.mean(skill))}
    out["real_minus_surrogate"] = out["real"]["mean_skill"] - out[kind]["mean_skill"]
    return out
