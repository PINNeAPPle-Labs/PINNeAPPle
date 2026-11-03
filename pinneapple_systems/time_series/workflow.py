"""The standard time-series prediction and modeling workflow (#1197): analysis -> baselines -> statistical, learned
and physical models -> rolling-origin backtest with skill against the baselines -> calibrated (conformal)
intervals with their measured coverage.

Protocol
--------
1. **Data**: a bundled series (``arma``: ARMA(2, 1) of known order; ``lorenz``: the Lorenz-63 system with all three
   variables observed, target ``x``; ``seasonal``: a daily-cycle series with an exogenous covariate, gaps, injected
   outliers and a column that leaks the future target) or a CSV (``value_column``, ``covariates``).
2. **Analysis**: missing values (count, longest gap, linear fill), outliers (Hampel filter, replaced by the
   window median), seasonality (dominant period and its autocorrelation), stationarity (ADF and KPSS; a unit root
   makes the ARIMA difference once), predictability (permutation entropy, and the largest Lyapunov exponent by
   Rosenstein's method with the automatic linear-region fit, reported as the Lyapunov time ``1 / lambda``) and
   leakage (a covariate that equals the target or a future target, ``|corr| >= leak_corr`` at lag 0 to
   ``max(horizons)``, is dropped; the backtest folds are checked to keep every training label before the
   validation block and every validation label before the test block).
3. **Models** on one protocol (:class:`.forecast_lab.base.Forecaster`): the baselines ``persistence``,
   ``seasonal_naive`` (when a period is found) and ``mean``, which every model must beat; ``arima`` (order by BIC,
   :mod:`.statistical`) and ``ets`` (additive Holt-Winters); ``gbm`` (gradient boosting on lags of every column,
   when scikit-learn is installed) and ``tcn`` (a small temporal convolutional network); and ``ode`` (the
   Lorenz-63 equations with their parameters fitted to the training data) when the series has a physical model.
4. **Backtest**: ``n_folds`` rolling origins over the last ``test_fraction`` of the series (expanding training
   window, a validation block of ``val_size`` samples before each test block, purged by ``max(horizons)``). Skill
   per horizon = ``1 - RMSE / RMSE(best baseline at that horizon)``; a model *beats the baselines* when its skill
   exceeds ``skill_margin`` at some horizon and is not below ``-skill_margin`` at any (beyond the memory of a
   stationary process the best forecast is the mean, so skill 0 at long horizons is the optimum, not a failure).
5. **Intervals**: split-conformal per horizon on the validation residuals of each fold, coverage measured on the
   test blocks; the check passes when the recommended model's coverage is within ``coverage_tolerance`` of the
   nominal level.

The recommendation is the model with the lowest mean RMSE over the horizons among those that beat the baselines.
"""
from __future__ import annotations

import csv
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .forecast_lab.base import SeriesData, forecast_origins, targets_at
from .forecast_lab.baselines import Climatology, Persistence, SeasonalNaive, dominant_period
from .predictability import permutation_entropy, rosenstein_lyapunov
from .statistical import (
    ARMAForecaster,
    HoltWintersForecaster,
    adf_test,
    arma_sample,
    conformal_quantile,
    hampel,
    kpss_test,
)

__all__ = ["time_series_workflow", "load_series", "SERIES", "as_tables"]

SERIES = {
    "arma": "ARMA(2, 1), phi = (0.75, -0.5), theta = 0.6, mean 5, 1000 samples (known order)",
    "lorenz": "Lorenz-63 (sigma 10, rho 28, beta 8/3) every 0.05, all three variables observed with 5 % noise, "
              "target x (largest Lyapunov exponent from the equations by Benettin's method)",
    "seasonal": "daily cycle (period 24) + an exogenous covariate + AR(1) noise, 1440 samples, with 1 % gaps, "
                "6 injected outliers and a column that leaks the next target value",
}


def load_series(name: str, seed: int = 0) -> dict[str, Any]:
    """A bundled series as ``obs`` (T, d) with the target in column 0, ``columns``, ``dt`` and its ground truth."""
    rng = np.random.default_rng(seed)
    if name == "arma":
        y = arma_sample(1000, [0.75, -0.5], [0.6], mu=5.0, rng=rng)
        return {"obs": y[:, None], "columns": ["y"], "dt": 1.0, "truth": {"order": (2, 1)}, "name": name}
    if name == "lorenz":
        from .synthetic import lorenz63
        s = lorenz63(n=3000, dt=0.05, noise=0.05, seed=seed, observed=(0, 1, 2), lyapunov_steps=40000)
        return {"obs": s.obs, "columns": ["x", "y", "z"], "dt": s.dt, "name": name,
                "truth": {"lyapunov": float(s.lyapunov), "lyapunov_time": float(s.lyapunov_time)}}
    if name == "seasonal":
        n = 1440
        t = np.arange(n)
        cov = np.zeros(n)
        for i in range(1, n):
            cov[i] = 0.95 * cov[i - 1] + 0.3 * rng.standard_normal()
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = 0.7 * noise[i - 1] + 0.5 * rng.standard_normal()
        y = 20.0 + 4.0 * np.sin(2 * np.pi * t / 24.0) + 1.5 * cov + noise
        out_idx = np.sort(rng.choice(np.arange(50, n - 50), 6, replace=False))
        y_obs = y.copy()
        y_obs[out_idx] += rng.choice([-1, 1], 6) * 8.0 * y.std()
        leak = np.concatenate([y[1:], [y[-1]]]) + 0.05 * rng.standard_normal(n)
        gaps = np.sort(rng.choice(np.arange(10, n - 10), 14, replace=False))
        y_obs[gaps] = np.nan
        obs = np.column_stack([y_obs, cov, leak])
        return {"obs": obs, "columns": ["y", "temperature", "y_next_reported"], "dt": 1.0, "name": name,
                "truth": {"period": 24, "outliers": out_idx.tolist(), "gaps": gaps.tolist(),
                          "leaky_column": "y_next_reported"}}
    raise ValueError(f"unknown series {name!r}; choose one of {sorted(SERIES)} or give csv_path")


def _load_csv(path: str, value_column: str, covariates: Sequence[str]) -> dict[str, Any]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    cols = [value_column] + [c for c in covariates if c != value_column]
    for c in cols:
        if c not in rows[0]:
            raise ValueError(f"{path}: no column {c!r} (columns: {list(rows[0])})")

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    obs = np.array([[num(r[c]) for c in cols] for r in rows], float)
    return {"obs": obs, "columns": cols, "dt": 1.0, "name": path, "truth": {}}


def _fill(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    bad = ~np.isfinite(x)
    if not bad.any():
        return x, 0, 0
    idx = np.arange(len(x))
    y = x.copy()
    y[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    run = longest = 0
    for b in bad:
        run = run + 1 if b else 0
        longest = max(longest, run)
    return y, int(bad.sum()), int(longest)


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    return float(x[lag:] @ x[:-lag] / (x @ x)) if 0 < lag < len(x) else float("nan")


def _leakage(obs: np.ndarray, columns: Sequence[str], H: int, thr: float) -> list[dict[str, Any]]:
    y = obs[:, 0]
    out = []
    for j in range(1, obs.shape[1]):
        x = obs[:, j]
        best = (0.0, 0)
        for k in range(0, H + 1):
            a, b = x[: len(x) - k], y[k:]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 10:
                continue
            c = float(np.corrcoef(a[ok], b[ok])[0, 1])
            if abs(c) > abs(best[0]):
                best = (c, k)
        out.append({"column": columns[j], "max_abs_corr": abs(best[0]), "at_lead": best[1],
                    "leaks": abs(best[0]) >= thr})
    return out


def _make_models(names: Sequence[str], period: int, d: int, n_cols: int, seed: int, physics: str,
                 epochs: int) -> tuple[dict[str, Any], dict[str, str]]:
    from .forecast_lab.learned import NeuralForecaster
    made: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for nm in names:
        if nm == "persistence":
            made[nm] = lambda: Persistence()
        elif nm == "mean":
            made[nm] = lambda: Climatology()
        elif nm == "seasonal_naive":
            if period > 1:
                made[nm] = lambda: SeasonalNaive(period=period)
            else:
                skipped[nm] = "no seasonal period found"
        elif nm == "arima":
            made[nm] = lambda: ARMAForecaster(max_p=4, max_q=3, d=d)
        elif nm == "ets":
            made[nm] = lambda: HoltWintersForecaster(period=period if period > 1 else 0)
        elif nm == "gbm":
            try:
                import sklearn  # noqa: F401
            except ImportError:
                skipped[nm] = "scikit-learn is not installed (pip install scikit-learn)"
                continue
            from .forecast_lab.learned import GBMForecaster
            made[nm] = lambda: GBMForecaster(n_lags=max(8, min(period, 48)) if period > 1 else 12,
                                             max_iter=200, seed=seed)
        elif nm == "tcn":
            made[nm] = lambda: NeuralForecaster(arch="tcn", n_lags=max(16, min(period, 48)) if period > 1 else 24,
                                                hidden=32, layers=2, epochs=epochs, patience=8, seed=seed,
                                                max_train=4000, name="tcn")
        elif nm == "ode":
            if physics == "lorenz63":
                from .forecast_lab.physical import ODEForecaster
                from .synthetic import lorenz63_rhs
                made[nm] = lambda: ODEForecaster(lorenz63_rhs, {"sigma": 9.0, "rho": 26.0, "beta": 2.5},
                                                 name="ode")
            else:
                skipped[nm] = "no physical model is known for this series (give one through a formulated ODE)"
        else:
            raise ValueError(f"unknown model {nm!r}")
    return made, skipped


def time_series_workflow(*, series: str = "seasonal", csv_path: str = "", value_column: str = "",
                         covariates: Sequence[str] = (), horizons: Sequence[int] = (1, 2, 4, 8, 16),
                         models: Sequence[str] = ("persistence", "seasonal_naive", "mean", "arima", "ets", "gbm",
                                                  "tcn", "ode"),
                         n_folds: int = 3, test_fraction: float = 0.3, val_size: int = 0, period: int = 0,
                         level: float = 0.9, coverage_tolerance: float = 0.05, leak_corr: float = 0.98,
                         lyapunov: str = "auto", epochs: int = 40, skill_margin: float = 0.02,
                         outlier_window: int = 12, outlier_sigmas: float = 5.0, seed: int = 0) -> dict[str, Any]:
    """Run the pipeline of the module docstring and return the analysis, the backtest table and every check."""
    if csv_path:
        d = _load_csv(csv_path, value_column, covariates)
    else:
        d = load_series(series, seed)
    obs = np.asarray(d["obs"], float)
    columns = list(d["columns"])
    H = np.asarray(sorted(int(h) for h in horizons), int)
    if H.min() < 1:
        raise ValueError("horizons must be >= 1")
    checks: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {"n": int(len(obs)), "columns": columns}

    # ---- analysis -----------------------------------------------------------------------------------------
    filled = []
    miss = {}
    for j in range(obs.shape[1]):
        col, nmiss, longest = _fill(obs[:, j])
        filled.append(col)
        miss[columns[j]] = {"missing": nmiss, "longest_gap": longest}
    obs = np.column_stack(filled)
    analysis["missing"] = miss
    hp = hampel(obs[:, 0], half_window=outlier_window, n_sigmas=outlier_sigmas)
    analysis["outliers"] = {"count": hp["count"], "index": hp["index"], "rule": hp["rule"]}
    obs[:, 0] = hp["cleaned"]
    y = obs[:, 0]
    leak = _leakage(obs, columns, int(H.max()), leak_corr)
    analysis["leakage"] = leak
    dropped = [r["column"] for r in leak if r["leaks"]]
    keep = [0] + [j for j in range(1, obs.shape[1]) if columns[j] not in dropped]
    obs = obs[:, keep]
    columns = [columns[j] for j in keep]
    per = int(period) if period else int(round(dominant_period(y)))
    acf_p = _acf(y, per) if per > 1 else float("nan")
    if not period and not (np.isfinite(acf_p) and acf_p > 0.3):
        per = 0
    analysis["seasonality"] = {"period": per, "acf_at_period": acf_p,
                               "rule": "dominant spectral period kept when its autocorrelation exceeds 0.3"
                               if not period else "given"}
    adf, kps = adf_test(y), kpss_test(y)
    unit_root = (not adf["unit_root_rejected_5pct"]) and kps["stationarity_rejected_5pct"]
    analysis["stationarity"] = {"adf": adf, "kpss": kps, "unit_root": bool(unit_root),
                                "verdict": "unit root: the ARIMA differences once" if unit_root else
                                ("stationary" if adf["unit_root_rejected_5pct"] and not kps["stationarity_rejected_5pct"]
                                 else "ambiguous (the tests disagree): no differencing")}
    pe = permutation_entropy(y, order=5)
    pred: dict[str, Any] = {"permutation_entropy": pe}
    run_ly = lyapunov == "always" or (lyapunov == "auto" and (d["name"] == "lorenz" or pe < 0.6))
    if run_ly:
        ly = rosenstein_lyapunov(y, dim=5, delay=3, dt=d["dt"], n_steps=80, fit="auto")
        pred["lyapunov"] = ly["lyapunov"]
        pred["lyapunov_fit_steps"] = ly["fit"]
        pred["lyapunov_time"] = 1.0 / ly["lyapunov"] if ly["lyapunov"] > 0 else float("inf")
    else:
        pred["lyapunov"] = None
        pred["lyapunov_note"] = ("not estimated: permutation entropy >= 0.6 (no deterministic structure for a "
                                 "Lyapunov exponent; the predictability horizon is the backtest's skill horizon)")
    analysis["predictability"] = pred

    # ---- backtest -----------------------------------------------------------------------------------------
    data = SeriesData(obs, 0, d["dt"], d["name"])
    n = len(obs)
    hmax = int(H.max())
    vs = int(val_size) if val_size else max(4 * hmax, int(0.1 * n))
    test_start = int(n * (1 - test_fraction))
    edges = np.linspace(test_start, n, n_folds + 1).astype(int)
    physics = "lorenz63" if d["name"] == "lorenz" else ""
    made, skipped = _make_models(list(models), per, 1 if unit_root else 0, obs.shape[1], seed, physics, epochs)
    min_hist = 49
    errs: dict[str, list[np.ndarray]] = {k: [] for k in made}
    cover: dict[str, list[np.ndarray]] = {k: [] for k in made}
    width: dict[str, list[np.ndarray]] = {k: [] for k in made}
    fit_s: dict[str, float] = {k: 0.0 for k in made}
    info: dict[str, Any] = {}
    leak_ok = True
    for k in range(n_folds):
        a, b = int(edges[k]), int(edges[k + 1])
        val_lo = a - vs
        tr_o = forecast_origins(0, val_lo - hmax, H, min_history=min_hist)      # labels end before val_lo - hmax
        va_o = forecast_origins(val_lo, a, H, min_history=min_hist)             # labels end before a
        te_o = forecast_origins(a, b, H, min_history=min_hist)
        leak_ok &= bool(tr_o.max() + hmax < va_o.min() and va_o.max() + hmax < te_o.min())
        Yv, Yt = targets_at(data, va_o, H), targets_at(data, te_o, H)
        for nm, make in made.items():
            t0 = time.perf_counter()
            try:
                m = make().fit(data, tr_o, va_o, H)
                pv, pt = m.predict(data, va_o).mean, m.predict(data, te_o).mean
            except Exception as exc:  # noqa: BLE001 -- a model that fails is reported, not fatal
                skipped[nm] = f"failed in fold {k}: {type(exc).__name__}: {exc}"
                continue
            fit_s[nm] += time.perf_counter() - t0
            q = np.array([conformal_quantile(np.abs(Yv[:, j] - pv[:, j]), 1 - level) for j in range(len(H))])
            errs[nm].append(Yt - pt)
            cover[nm].append(np.abs(Yt - pt) <= q[None, :])
            width[nm].append(np.broadcast_to(2 * q, pt.shape))
            if k == n_folds - 1:
                info[nm] = m.describe()
    names = [nm for nm in made if nm not in skipped and errs[nm]]
    for nm in list(made):
        if nm in skipped:
            errs.pop(nm, None)
    rmse = {nm: np.sqrt(np.mean(np.concatenate(errs[nm]) ** 2, axis=0)) for nm in names}
    cov = {nm: np.mean(np.concatenate(cover[nm]), axis=0) for nm in names}
    wid = {nm: np.mean(np.concatenate(width[nm]), axis=0) for nm in names}
    baselines = [nm for nm in ("persistence", "seasonal_naive", "mean") if nm in rmse]
    if not baselines:
        raise ValueError("at least one baseline (persistence, seasonal_naive, mean) is needed")
    best_base = np.min(np.stack([rmse[b] for b in baselines]), axis=0)
    skill = {nm: 1.0 - rmse[nm] / best_base for nm in names}
    beats = {nm: bool(np.all(skill[nm] >= -skill_margin) and np.any(skill[nm] > skill_margin))
             for nm in names if nm not in baselines}
    cands = [nm for nm, ok in beats.items() if ok]
    rec = min(cands, key=lambda nm: float(np.mean(rmse[nm]))) if cands else \
        min(baselines, key=lambda nm: float(np.mean(rmse[nm])))
    out: dict[str, Any] = {"data": {"name": d["name"], "description": SERIES.get(d["name"], csv_path),
                                    "dt": d["dt"], "dropped_columns": dropped},
                           "analysis": analysis, "horizons": H.tolist(), "level": level,
                           "folds": {"n": n_folds, "test_start": test_start, "val_size": vs,
                                     "edges": edges.tolist()},
                           "models": {nm: {"rmse": rmse[nm].tolist(), "skill": skill[nm].tolist(),
                                           "coverage": cov[nm].tolist(), "interval_width": wid[nm].tolist(),
                                           "baseline": nm in baselines, "beats_baselines": beats.get(nm),
                                           "fit_seconds": fit_s[nm], "info": info.get(nm, {})} for nm in names},
                           "skipped": skipped, "recommended": rec}
    rec_skill = skill[rec]
    pos = np.nonzero(1.0 - rmse[rec] / rmse["mean"] > 0)[0] if "mean" in rmse else np.arange(len(H))
    sk_h = int(H[pos.max()]) if len(pos) else 0
    pred["skill_horizon_steps"] = sk_h
    pred["skill_horizon_time"] = sk_h * d["dt"]
    if pred.get("lyapunov_time") and np.isfinite(pred["lyapunov_time"]):
        pred["skill_horizon_lyapunov_times"] = sk_h * d["dt"] / pred["lyapunov_time"]

    # ---- checks -------------------------------------------------------------------------------------------
    checks.append({"code": "leakage.folds", "status": "pass" if leak_ok else "fail",
                   "detail": f"every fold: training labels end > {hmax} samples before the validation block, "
                             "validation labels before the test block" if leak_ok else "a fold overlaps"})
    checks.append({"code": "leakage.columns", "status": "pass",
                   "detail": (f"dropped {', '.join(dropped)} (|corr| >= {leak_corr} with the target at a lead "
                              f"<= {hmax})" if dropped else "no covariate equals the target or a future target")})
    checks.append({"code": "models.beat_baselines", "status": "pass" if cands else "fail",
                   "detail": (f"{', '.join(cands)} beat the best baseline (skill > {skill_margin} at some "
                              f"horizon, >= -{skill_margin} at all)" if cands else
                              "no model beats the best baseline (the series may be unpredictable beyond them)")})
    c_rec = cov[rec]
    dev = float(np.max(np.abs(c_rec - level)))
    checks.append({"code": "intervals.coverage", "status": "pass" if dev <= coverage_tolerance else "fail",
                   "detail": f"{rec}: conformal {level:.0%} coverage on the test blocks "
                             f"{', '.join(f'h{h}: {c:.3f}' for h, c in zip(H, c_rec))} (largest gap {dev:.3f}, "
                             f"allowed {coverage_tolerance})"})
    tr = d.get("truth", {})
    if "order" in tr and "arima" in info:
        got = tuple(info["arima"]["order"][::2])
        checks.append({"code": "arma.order", "status": "pass" if got == tuple(tr["order"]) else "fail",
                       "detail": f"BIC selected ARMA{got} on the last fold's training data; true ARMA{tuple(tr['order'])}"})
    if "lyapunov" in tr and pred.get("lyapunov") is not None:
        rel = abs(pred["lyapunov"] - tr["lyapunov"]) / tr["lyapunov"]
        checks.append({"code": "predictability.lyapunov_time", "status": "pass" if rel <= 0.15 else "fail",
                       "detail": f"Lyapunov time {pred['lyapunov_time']:.3f} from the observed x (Rosenstein, fit "
                                 f"steps {pred['lyapunov_fit_steps']}) against {tr['lyapunov_time']:.3f} from the "
                                 f"equations (Benettin): {rel:.1%} apart (allowed 15 %)"})
    if "period" in tr:
        checks.append({"code": "analysis.period", "status": "pass" if per == tr["period"] else "fail",
                       "detail": f"found {per}, true {tr['period']}"})
    if "outliers" in tr:
        found = set(analysis["outliers"]["index"])
        rec_o = len(found & set(tr["outliers"])) / len(tr["outliers"])
        checks.append({"code": "analysis.outliers", "status": "pass" if rec_o == 1.0 else "fail",
                       "detail": f"{rec_o:.0%} of the {len(tr['outliers'])} injected outliers found; "
                                 f"{len(found - set(tr['outliers']))} other samples flagged"})
    if "leaky_column" in tr:
        checks.append({"code": "leakage.injected", "status": "pass" if tr["leaky_column"] in dropped else "fail",
                       "detail": f"the column {tr['leaky_column']!r} that reports the next target was "
                                 f"{'dropped' if tr['leaky_column'] in dropped else 'kept'}"})
    out["checks"] = checks
    out["passed"] = all(c["status"] != "fail" for c in checks)
    out["summary"] = (f"{d['name']}: recommended {rec} (skill {', '.join(f'{s:+.2f}' for s in rec_skill)} at "
                      f"h = {H.tolist()} against the best baseline; {level:.0%} conformal coverage "
                      f"{float(np.mean(c_rec)):.3f}); "
                      + (f"Lyapunov time {pred['lyapunov_time']:.3g}; " if pred.get("lyapunov") else "")
                      + f"{sum(c['status'] != 'fail' for c in checks)}/{len(checks)} checks pass")
    return out


def as_tables(result: dict[str, Any]) -> dict[str, list[list[Any]]]:
    H = result["horizons"]
    rows = []
    for nm, m in result["models"].items():
        rows.append([nm, "baseline" if m["baseline"] else ("yes" if m["beats_baselines"] else "no"),
                     " / ".join(f"{v:.4g}" for v in m["rmse"]), " / ".join(f"{v:+.3f}" for v in m["skill"]),
                     " / ".join(f"{v:.3f}" for v in m["coverage"]), round(m["fit_seconds"], 2)])
    return {"backtest": rows, "horizons": [H],
            "checks": [[c["code"], c["status"], c["detail"]] for c in result["checks"]],
            "skipped": [[k, v] for k, v in result["skipped"].items()]}
