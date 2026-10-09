"""Data-driven process optimization for industrial plants (plain machine learning, no physics model).

From historical operating data:

1. **Clean** with the data-health report: duplicate timestamps dropped, error codes, impossible values,
   stuck-sensor and spike spans set to missing (:func:`clean_for_modeling`).
2. **Learn** the plant: a gradient-boosting model (scikit-learn ``HistGradientBoostingRegressor``) predicts
   the KPI (e.g. plant power) from the *levers* the operator controls (setpoints, speeds) and the *context*
   the operator does not (weather, load). It is scored on the most recent 20-30 % of the record, which it
   never saw (time split, no shuffling). Its L2 regularisation is chosen by forward-chaining time-series
   cross-validation inside the training part, and the per-fold errors are reported.
3. **Optimize** each test-period hour: try lever combinations and keep the one with the best predicted
   KPI that also
   * stays inside the operating envelope: the lever + context combination must resemble situations in
     the data (nearest-neighbour distance), because the model cannot know what it has never seen;
   * respects constraints on other measured outputs (e.g. CHW return temperature <= 13 C), each with its own
     model;
   * moves each lever at most a set fraction of its range from the current value (gradual change).
4. **Report** the predicted saving with a range from an ensemble of bootstrapped models, the recommended
   setpoint schedule against the most influential context variable, partial-dependence curves and
   feature importance.

The saving is a model estimate from correlations in historical data, not a guarantee: the right next step
is a supervised trial (A/B by day). The built-in example (:func:`example_plant_operations`) is a simulated
chilled-water plant whose true physics is known, so the predicted saving can be checked against the true one.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_health import Issue, _issue_masks, _parse_time, _to_numeric, detect_time_column
from .physical_units import split_header

_TARGET = re.compile(r"power|energy|kw|consumption|demand|fuel|gas|steam|cost|kwh|emission|co2|specific|efficien|cop|"
                     r"yield|throughput|production|output", re.I)
_LEVER = re.compile(r"setpoint|set_point|\bsp\b|_sp\b|sp_|speed|position|valve|damper|command|cmd|frequency|vfd|opening|"
                    r"stage|staging|ratio|feed|reflux|dosing|pressure_sp|target", re.I)
_TIMEY = re.compile(r"^(time|timestamp|date)", re.I)


# ── roles ───────────────────────────────────────────────────────────────────
def suggest_roles(columns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Guess target / levers / context from column names (to be reviewed by a person)."""
    num = [c for c in columns if c.get("kind") == "numeric" and (c.get("n_unique") or 0) > 3]
    names = [c["name"] for c in num]
    levers = [n for n in names if _LEVER.search(split_header(n)[0].replace(" ", "_"))]
    cand = [n for n in names if n not in levers and _TARGET.search(split_header(n)[0])]
    plant = [n for n in cand if re.search(r"plant|total|site|facility", n, re.I)]
    target = (plant or cand or [None])[0]
    outcome = re.compile(r"return|leaving|outlet|discharge|exhaust|_out\b|lwt|chwr|cwr", re.I)
    constraints = [{"column": n, "op": "<=", "value": round(float(c.get("p99") or 0), 2)} for c in num
                   for n in [c["name"]] if n not in levers and n != target and outcome.search(n)][:2]
    context = [n for n in names if n not in levers and n != target and not _TARGET.search(split_header(n)[0])
               and not outcome.search(n)]
    goal = "maximize" if target and re.search(r"cop|efficien|yield|throughput|production|output", target, re.I) else "minimize"
    return {"target": target, "goal": goal, "levers": levers, "context": context, "constraints": constraints}


# ── cleaning ────────────────────────────────────────────────────────────────
def clean_for_modeling(df: pd.DataFrame, report: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """Time-sorted numeric table with the problems found by the health report set to missing."""
    log: List[str] = []
    tcol = report["summary"].get("time_column")
    if tcol:
        t, _ = _parse_time(df[tcol])
    else:
        _, t, _ = detect_time_column(df)
    numeric: Dict[str, pd.Series] = {}
    for col in report["columns"]:
        if col.get("kind") != "numeric":
            continue
        v, _ = _to_numeric(df[col["name"]])
        numeric[col["name"]] = v.astype(float)
    out = pd.DataFrame(numeric)
    if t is not None:
        out.insert(0, "__time", t.to_numpy())
        out = out[out["__time"].notna()].sort_values("__time", kind="stable")
        n0 = len(out)
        out = out.drop_duplicates("__time", keep="last").reset_index(drop=True)
        if n0 - len(out):
            log.append(f"Dropped {n0 - len(out)} rows with duplicate timestamps.")
    else:
        out.insert(0, "__time", np.arange(len(out)))
    issues = [Issue(**{k: v for k, v in i.items() if k in Issue.__dataclass_fields__}) for i in report["issues"]]
    for i in issues:
        if i.column in out and i.title.startswith("Error code"):
            k = int((out[i.column] == i.evidence.get("value")).sum())
            out.loc[out[i.column] == i.evidence.get("value"), i.column] = np.nan
            log.append(f"'{i.column}': {k} error-code values ({i.evidence.get('value'):g}) set to missing.")
    tt = pd.Series(pd.to_datetime(out["__time"])) if t is not None else None
    series = {c: out[c].reset_index(drop=True) for c in out.columns if c != "__time"}
    masks = _issue_masks([i for i in issues if i.column in series and i.category in ("validity", "signal_quality")
                          and i.title not in ("Constant column",) and not i.title.startswith("Clipped")],
                         series, tt)
    for c, m in masks.items():
        if m.any():
            out.loc[m, c] = np.nan
            log.append(f"'{c}': {int(m.sum())} samples in flagged spans (stuck, spikes, impossible values, unit "
                       "switch) set to missing.")
    return out.rename(columns={"__time": "time"}), log


# ── model ───────────────────────────────────────────────────────────────────
L2_GRID = (1e-3, 0.1, 1.0, 10.0)


def _model(seed: int = 0, quick: bool = False, l2: float = 1e-3):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=150 if quick else 300, learning_rate=0.08, max_leaf_nodes=31,
                                         min_samples_leaf=20, l2_regularization=l2, random_state=seed)


def time_series_cv(X: np.ndarray, y: np.ndarray, n_folds: int = 4, l2_grid: Sequence[float] = L2_GRID,
                   seed: int = 0) -> Dict[str, Any]:
    """Forward-chaining cross-validation on the training period: fold k trains on the first k blocks and is scored
    on the next one (never on the past), for every L2 regularisation strength in ``l2_grid``. Picks the strength
    with the lowest mean validation MAE; the per-fold errors show how stable the model is over time."""
    n = len(y)
    edges = np.linspace(0, n, n_folds + 2).astype(int)        # block 0 is only ever used for training
    grid = []
    for l2 in l2_grid:
        folds = []
        for k in range(1, n_folds + 1):
            a, b = edges[k], edges[k + 1]
            if a < 50 or b - a < 20:
                continue
            m = _model(seed, quick=True, l2=l2).fit(X[:a], y[:a])
            mt = _metrics(y[a:b], m.predict(X[a:b]), float(y[:a].mean()))
            folds.append({"train_rows": int(a), "val_rows": int(b - a), "mae": mt["mae"], "r2": mt["r2"]})
        maes = [f["mae"] for f in folds]
        grid.append({"l2": float(l2), "folds": folds, "mean_mae": float(np.mean(maes)) if maes else float("nan"),
                     "std_mae": float(np.std(maes)) if maes else float("nan")})
    usable = [g for g in grid if g["folds"]]
    if not usable:
        return {"n_folds": 0, "grid": grid, "chosen_l2": 1e-3, "folds": [], "mean_mae": None, "std_mae": None,
                "mean_r2": None, "note": "too few rows for time-series cross-validation"}
    best = min(usable, key=lambda g: g["mean_mae"])
    return {"n_folds": len(best["folds"]), "scheme": "forward chaining (expanding window, no shuffling)",
            "chosen_l2": best["l2"], "folds": best["folds"], "mean_mae": best["mean_mae"], "std_mae": best["std_mae"],
            "mean_r2": float(np.mean([f["r2"] for f in best["folds"]])),
            "grid": [{"l2": g["l2"], "mean_mae": g["mean_mae"], "std_mae": g["std_mae"]} for g in grid]}


def _metrics(y: np.ndarray, p: np.ndarray, y_train_mean: float) -> Dict[str, float]:
    err = p - y
    ss = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": 1 - float(np.sum(err ** 2)) / ss, "mape_pct": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-9)) * 100),
            "baseline_mae": float(np.mean(np.abs(y - y_train_mean))), "bias": float(np.mean(err))}


def _envelope(X: np.ndarray, quantile: float = 0.95):
    """Standardiser + nearest-neighbour index + distance threshold that defines 'seen before'."""
    from sklearn.neighbors import NearestNeighbors
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    nn = NearestNeighbors(n_neighbors=6).fit(Z)
    d, _ = nn.kneighbors(Z[: min(len(Z), 4000)])
    thr = float(np.quantile(d[:, 5], quantile))                 # distance to the 5th neighbour, excluding self
    return mu, sd, nn, thr


def fit_and_optimize(data: pd.DataFrame, target: str, levers: Sequence[str], context: Sequence[str],
                     goal: str = "minimize", constraints: Sequence[Dict[str, Any]] = (),
                     bounds: Optional[Dict[str, Tuple[float, float]]] = None, max_move: float = 0.5,
                     test_fraction: float = 0.25, grid: int = 7, n_ensemble: int = 5, max_rows: int = 600,
                     seed: int = 0, cv_folds: int = 4) -> Dict[str, Any]:
    """Train on the older part of the record, evaluate and optimize on the newest part. The L2 regularisation
    of the KPI model is chosen by forward-chaining cross-validation inside the training part (``cv_folds``;
    0 keeps the default) and the fold errors go in the report under ``model.cv``."""
    if not target or target not in data:
        raise ValueError("Choose the KPI column to optimize.")
    levers = [c for c in levers if c in data and c != target]
    context = [c for c in context if c in data and c not in levers and c != target]
    if not levers:
        raise ValueError("Choose at least one lever (a setpoint, speed or position the operator can change).")
    if len(levers) > 4:
        raise ValueError("Use at most 4 levers at a time (the search grows as grid^levers).")
    cons = [c for c in constraints if c.get("column") in data and c.get("op") in ("<=", ">=") and c.get("value") is not None]
    feats = list(levers) + list(context)
    d = data.dropna(subset=feats + [target]).reset_index(drop=True)
    if len(d) < 200:
        raise ValueError(f"Only {len(d)} complete rows after cleaning; at least 200 are needed.")
    n_test = max(50, int(len(d) * test_fraction))
    tr, te = d.iloc[:-n_test], d.iloc[-n_test:]
    Xtr, Xte = tr[feats].to_numpy(float), te[feats].to_numpy(float)
    ytr, yte = tr[target].to_numpy(float), te[target].to_numpy(float)
    rng = np.random.default_rng(seed)

    # time-series cross-validation on the training period picks the regularisation
    ts_cv = time_series_cv(Xtr, ytr, cv_folds, seed=seed) if cv_folds else None
    l2 = ts_cv["chosen_l2"] if ts_cv else 1e-3

    # main model + bootstrap ensemble (block bootstrap by day-sized chunks keeps autocorrelation)
    main = _model(seed, l2=l2).fit(Xtr, ytr)
    pte = main.predict(Xte)
    metrics = _metrics(yte, pte, float(ytr.mean()))
    block = max(24, len(tr) // 40)
    starts = np.arange(0, len(tr), block)
    ens = []
    for k in range(n_ensemble):
        pick = rng.choice(starts, len(starts), replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, len(tr))) for s in pick])
        ens.append(_model(seed + 1 + k, quick=True, l2=l2).fit(Xtr[idx], ytr[idx]))
    cmodels = {}
    for c in cons:
        dc = d.dropna(subset=[c["column"]])
        trc = dc.iloc[: max(1, len(dc) - n_test)]
        m = _model(seed + 50).fit(trc[feats].to_numpy(float), trc[c["column"]].to_numpy(float))
        cmodels[c["column"]] = m
        c["test_mae"] = float(np.mean(np.abs(m.predict(Xte) - te[c["column"]].to_numpy(float)))) if c["column"] in te else None

    # importance (permutation on the test period)
    from sklearn.inspection import permutation_importance
    pim = permutation_importance(main, Xte, yte, n_repeats=4, random_state=seed, scoring="neg_mean_absolute_error")
    importance = sorted([{"feature": f, "role": "lever" if f in levers else "context",
                          "importance": float(pim.importances_mean[i]), "unit_of_target": True}
                         for i, f in enumerate(feats)], key=lambda r: -r["importance"])

    # search space: training-data range of each lever (2nd-98th percentile), narrowed by user bounds
    lo_hi = {}
    for j, lv in enumerate(levers):
        lo, hi = np.percentile(Xtr[:, j], [2, 98])
        if bounds and lv in bounds:
            lo, hi = max(lo, bounds[lv][0]), min(hi, bounds[lv][1])
        lo_hi[lv] = (float(lo), float(hi))
    grids = [np.linspace(*lo_hi[lv], grid) for lv in levers]
    combos = np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(levers), -1).T
    mu, sd, nn, thr = _envelope(Xtr)
    sel = np.linspace(0, len(te) - 1, min(max_rows, len(te))).astype(int)
    rows = te.iloc[sel]
    Xr = Xte[sel]
    sign = 1.0 if goal == "minimize" else -1.0
    rec = np.zeros((len(rows), len(levers)))
    best_pred, cur_pred = np.zeros(len(rows)), main.predict(Xr)
    blocked = {"envelope": 0, "constraint": 0, "move": 0}
    fixed = {"needed": 0, "found": 0}
    nctx = len(context)
    for i in range(len(rows)):
        cand = np.hstack([combos, np.repeat(Xr[i:i + 1, len(levers):], len(combos), 0)]) if nctx else combos.copy()
        ok = np.ones(len(cand), bool)
        span = np.array([lo_hi[lv][1] - lo_hi[lv][0] or 1.0 for lv in levers])
        move = np.abs(cand[:, :len(levers)] - Xr[i, :len(levers)]) / span
        mv = (move <= max_move + 1e-9).all(1)
        blocked["move"] += int((~mv).sum())
        ok &= mv
        dist, _ = nn.kneighbors((cand - mu) / sd, n_neighbors=5)
        env = dist[:, 4] <= thr
        blocked["envelope"] += int((ok & ~env).sum())
        ok &= env
        for c in cons:
            pc = cmodels[c["column"]].predict(cand)
            margin = c.get("test_mae") or 0.0                  # keep one model error away from the limit
            okc = pc <= c["value"] - margin if c["op"] == "<=" else pc >= c["value"] + margin
            blocked["constraint"] += int((ok & ~okc).sum())
            ok &= okc
        cur_ok = True
        for c in cons:                                         # keep the current setting only if it is feasible
            pc = float(cmodels[c["column"]].predict(Xr[i:i + 1])[0])
            cur_ok &= pc <= c["value"] if c["op"] == "<=" else pc >= c["value"]
        cand_cur = np.vstack([Xr[i:i + 1], cand[ok]]) if cur_ok or not ok.any() else cand[ok]
        if not cur_ok:
            fixed["needed"] += 1
            fixed["found"] += int(ok.any())
        p = main.predict(cand_cur)
        j = int(np.argmin(sign * p))
        rec[i], best_pred[i] = cand_cur[j, :len(levers)], p[j]
    gain = sign * (cur_pred - best_pred)                     # >= 0: improvement in the KPI's own units
    rel = gain / np.maximum(np.abs(cur_pred), 1e-9) * 100
    # ensemble: the same recommendations scored by each bootstrapped model
    Xrec = Xr.copy()
    Xrec[:, :len(levers)] = rec
    ens_pct = []
    for m in ens:
        a, b = m.predict(Xr), m.predict(Xrec)
        ens_pct.append(float(np.sum(sign * (a - b)) / np.sum(np.abs(a)) * 100))
    total_pct = float(np.sum(gain) / np.sum(np.abs(cur_pred)) * 100)
    ens_pct.append(total_pct)

    # setpoint schedule against the most important context variable
    sched = None
    top_ctx = next((r["feature"] for r in importance if r["role"] == "context"), None)
    if top_ctx:
        cv = rows[top_ctx].to_numpy(float)
        edges = np.unique(np.quantile(cv, np.linspace(0, 1, 6)))
        b = np.clip(np.searchsorted(edges, cv, side="right") - 1, 0, len(edges) - 2)
        sched = {"context": top_ctx, "bins": []}
        for k in range(len(edges) - 1):
            m = b == k
            if not m.any():
                continue
            sched["bins"].append({"from": float(edges[k]), "to": float(edges[k + 1]), "n": int(m.sum()),
                                  "current": {lv: float(np.median(Xr[m, j])) for j, lv in enumerate(levers)},
                                  "recommended": {lv: float(np.median(rec[m, j])) for j, lv in enumerate(levers)},
                                  "saving_pct": float(np.sum(gain[m]) / np.sum(np.abs(cur_pred[m])) * 100)})

    # partial dependence of the KPI on each lever (test rows, other features as observed)
    pdp = {}
    for j, lv in enumerate(levers):
        vals = np.linspace(*lo_hi[lv], 15)
        X = Xr.copy()
        ys = []
        for v in vals:
            X[:, j] = v
            ys.append(float(main.predict(X).mean()))
        pdp[lv] = {"x": vals.tolist(), "y": ys, "current_median": float(np.median(Xr[:, j]))}

    t = rows["time"]
    tx = [pd.Timestamp(x).strftime("%Y-%m-%d %H:%M") if not isinstance(x, (int, np.integer)) else int(x) for x in t]
    out = {
        "target": target, "goal": goal, "levers": list(levers), "context": list(context), "constraints": cons,
        "rows": {"train": int(len(tr)), "test": int(len(te)), "optimized": int(len(rows)),
                 "train_period": [str(tr["time"].iloc[0]), str(tr["time"].iloc[-1])],
                 "test_period": [str(te["time"].iloc[0]), str(te["time"].iloc[-1])]},
        "model": {"type": "HistGradientBoostingRegressor (scikit-learn)", "features": feats, **metrics,
                  "l2_regularization": l2, "cv": ts_cv},
        "importance": importance, "bounds": lo_hi, "max_move": max_move,
        "saving": {"pct": total_pct, "range_pct": [min(ens_pct), max(ens_pct)], "ensemble_pct": ens_pct,
                   "per_hour_units": float(np.mean(gain)), "rows_improved_pct": float(np.mean(rel > 0.5) * 100),
                   "unit_note": "in the target's own unit, per sample"},
        "blocked_candidates": blocked, "constraint_fixes": fixed, "envelope_threshold": thr,
        "schedule": sched, "partial_dependence": pdp,
        "series": {"x": tx, "actual": te[target].to_numpy(float)[sel].tolist(), "predicted": cur_pred.tolist(),
                   "optimized": best_pred.tolist(),
                   "levers": {lv: {"current": Xr[:, j].tolist(), "recommended": rec[:, j].tolist()} for j, lv in enumerate(levers)}},
        "_rec": rec, "_rows": rows,
    }
    return out


def _clean_json(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean_json(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ── example: a chilled-water plant with known physics ───────────────────────
_CAP = 1400.0          # chiller capacity, kW


def _wetbulb(t: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Stull (2011) wet-bulb approximation, °C."""
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def plant_physics(oat, rh, load, chws_sp, cws_sp, pump_pct):
    """True plant behaviour used by the example (not visible to the ML model)."""
    oat, rh, load = np.asarray(oat, float), np.asarray(rh, float), np.asarray(load, float)
    chws = np.asarray(chws_sp, float)
    twb = _wetbulb(oat, rh)
    cws = np.maximum(np.asarray(cws_sp, float), twb + 2.5)          # the tower cannot beat wet bulb + 2.5 K
    plr = np.clip(load / _CAP, 0.1, 1.0)
    lift = cws + 5.0 - (chws - 1.0)                                  # condensing minus evaporating, ~K
    cop = 0.62 * (chws - 1.0 + 273.15) / np.maximum(lift, 8.0) * (1 - 0.35 * (plr - 0.75) ** 2)
    chiller = load / cop
    approach = np.maximum(cws - twb, 2.5)
    tower = 38.0 * np.clip((load * 1.2 / (_CAP * 1.2)) * (4.0 / approach) ** 1.4, 0.15, 1.0) ** 3
    flow = 220.0 * np.asarray(pump_pct, float) / 100.0                # m3/h
    pump = 32.0 * (np.asarray(pump_pct, float) / 100.0) ** 3 + 2.0
    chwr = chws + load / (flow / 3600 * 997 * 4.186)
    return {"chiller": chiller, "tower": tower, "pump": pump, "plant": chiller + tower + pump, "chwr": chwr, "cws": cws,
            "twb": twb, "cop": cop}


def example_plant_operations(seed: int = 11, days: int = 60) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """60 days of 15-minute data from a chilled-water plant run by operators with manual setpoint habits."""
    rng = np.random.default_rng(seed)
    n = days * 96
    t = pd.date_range("2026-01-05", periods=n, freq="15min")
    h = (t.hour + t.minute / 60).to_numpy()
    day = np.arange(n) / 96
    oat = 26 + 5.5 * np.sin((h - 9) / 24 * 2 * np.pi) + 2.5 * np.sin(day / 9) + rng.normal(0, 0.4, n)
    rh = np.clip(72 - 2.4 * (oat - 26) + 6 * np.sin(day / 5) + rng.normal(0, 3, n), 30, 98)
    occ = ((h > 7) & (h < 20) & (t.dayofweek < 6)).astype(float)
    load = np.clip(300 + 520 * occ + 22 * (oat - 26) + rng.normal(0, 25, n), 150, 1350)
    # operator habits: setpoints changed every 1-3 days, pump speed by shift
    def piecewise(levels, min_len, max_len):
        out, i = np.empty(n), 0
        while i < n:
            L = int(rng.integers(min_len, max_len)) * 96
            out[i:i + L] = rng.choice(levels)
            i += L
        return out
    chws_sp = piecewise([6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0], 1, 4)
    cws_sp = piecewise([24.0, 25.5, 27.0, 28.5, 30.0, 31.5], 1, 3)
    shift = ((h >= 6) & (h < 18)).astype(int) + 2 * (day.astype(int) % 3)
    pump_pct = np.choose(shift % 4, [70.0, 80.0, 90.0, 100.0]) + rng.normal(0, 1.5, n)
    true = plant_physics(oat, rh, load, chws_sp, cws_sp, pump_pct)
    noise = lambda s: rng.normal(0, s, n)  # noqa: E731
    df = pd.DataFrame({
        "Timestamp": t.strftime("%Y-%m-%d %H:%M"), "OAT [°C]": (oat + noise(0.1)).round(2),
        "RH_outdoor [%]": (rh + noise(0.5)).round(1), "Cooling_load [kW]": (load + noise(5)).round(1),
        "CHW_supply_setpoint [°C]": chws_sp, "CW_supply_setpoint [°C]": cws_sp,
        "CHW_pump_speed [%]": pump_pct.round(1), "CHW_return_temp [°C]": (true["chwr"] + noise(0.08)).round(2),
        "Chiller_power [kW]": (true["chiller"] + noise(3)).round(1), "Tower_fan_power [kW]": (true["tower"] + noise(0.5)).round(1),
        "CHW_pump_power [kW]": (true["pump"] + noise(0.3)).round(1),
        "Plant_power [kW]": (true["plant"] + noise(3.5)).round(1)})
    # a few realistic data problems for the cleaning step
    a = int(20.3 * 96)
    df.loc[a:a + 90, "OAT [°C]"] = round(float(df.loc[a, "OAT [°C]"]), 2)          # frozen sensor
    for k in rng.choice(np.arange(200, n - 200), 6, replace=False):
        df.loc[k:k + 3, "Plant_power [kW]"] = -999                                      # meter offline
    return df, {"description": "Chilled-water plant, 15-min data, 60 days, operator setpoint habits",
                "levers": ["CHW_supply_setpoint [°C]", "CW_supply_setpoint [°C]", "CHW_pump_speed [%]"],
                "context": ["OAT [°C]", "RH_outdoor [%]", "Cooling_load [kW]"], "target": "Plant_power [kW]",
                "constraints": [{"column": "CHW_return_temp [°C]", "op": "<=", "value": 13.0}]}


def true_saving(result: Dict[str, Any]) -> Dict[str, Any]:
    """For the example: score the recommendations with the plant's true physics."""
    rows, rec = result["_rows"], result["_rec"]
    lv = result["levers"]
    cur = {k: rows[k].to_numpy(float) for k in lv}
    new = {k: rec[:, j] for j, k in enumerate(lv)}

    def run(v):
        return plant_physics(rows["OAT [°C]"], rows["RH_outdoor [%]"], rows["Cooling_load [kW]"],
                             v.get("CHW_supply_setpoint [°C]", rows["CHW_supply_setpoint [°C]"]),
                             v.get("CW_supply_setpoint [°C]", rows["CW_supply_setpoint [°C]"]),
                             v.get("CHW_pump_speed [%]", rows["CHW_pump_speed [%]"]))
    a, b = run(cur), run(new)
    pct = float((a["plant"].sum() - b["plant"].sum()) / a["plant"].sum() * 100)
    viol = float(np.mean(b["chwr"] > 13.0 + 0.3) * 100)
    viol0 = float(np.mean(a["chwr"] > 13.0 + 0.3) * 100)
    return {"true_saving_pct": pct, "true_constraint_violations_pct": viol, "true_constraint_violations_before_pct": viol0,
            "true_mean_kw_saved": float(np.mean(a["plant"] - b["plant"]))}


__all__ = ["suggest_roles", "clean_for_modeling", "fit_and_optimize", "example_plant_operations", "plant_physics",
           "true_saving"]
