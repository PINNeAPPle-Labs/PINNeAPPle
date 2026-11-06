"""The tabular half of a component-scale campaign: a design vector in, a handful of scalar quantities out.

The crash corpus of a pole impact comes with a tabular benchmark (seven design variables -> peak wall force,
deceleration, internal energy, plastic work, kinetic-energy absorption), and the interesting question there is not
which net is fashionable but whether a gradient-boosted tree beats a linear model by enough to justify it. This
module runs that comparison once, with the same split and the same fixed hyperparameters for every family (a
zero-tuning protocol: no model gets a search the others did not), and reports R^2, MAE, RMSE and MAPE per target plus
the mean, with the feature sensitivity of the tree models.

Every model is optional: scikit-learn is required, XGBoost/LightGBM/CatBoost are used when installed, and a model
that is not installed is reported as unavailable with the reason instead of being quietly dropped from the table.

    from pinneapple_arena.fieldbench.tabular import load_table, benchmark_tabular, write_table_report
    table = load_table("runs/pole_impact/pole_impact_quantities.csv", inputs=DESIGN_KEYS, targets=TARGET_KEYS)
    result = benchmark_tabular(table, seed=42)
    print(result["table"])
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np

__all__ = ["load_table", "available_regressors", "benchmark_tabular", "write_table_report", "feature_sensitivity",
           "probe_family", "OPTIONAL_FAMILIES"]

TARGET_LABELS = {
    "peak_wall_force": "peak pole force [N]",
    "peak_deceleration": "peak deceleration [m/s²]",
    "peak_internal_energy": "peak internal energy [J]",
    "peak_plastic_work": "peak plastic work [J]",
    "ke_absorption": "kinetic-energy absorption [-]",
    "impact_duration": "impact duration [s]",
}


def _metric_set(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    nz = np.abs(y_true) > 1e-12
    return {"r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
            "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape": float(np.mean(np.abs(err[nz] / y_true[nz])) * 100) if nz.any() else float("nan")}


def available_regressors(include_optional: bool = True) -> dict[str, bool]:
    """Which model families this machine can build (scikit-learn always, the boosters when installed)."""
    have = {"sklearn": importlib.util.find_spec("sklearn") is not None,
            "xgboost": importlib.util.find_spec("xgboost") is not None,
            "lightgbm": importlib.util.find_spec("lightgbm") is not None,
            "catboost": importlib.util.find_spec("catboost") is not None}
    if not have["sklearn"]:
        return {"sklearn": False}
    return have if include_optional else {"sklearn": have["sklearn"]}


def _factories(seed: int) -> dict:
    """Every family with hyperparameters fixed across targets (the zero-tuning protocol of the corpus)."""
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    out = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "lasso": lambda: make_pipeline(StandardScaler(), Lasso(alpha=1e-2, max_iter=50_000)),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance")),
        "svr": lambda: make_pipeline(StandardScaler(), SVR(C=10.0)),
        # early stopping inside the MLP needs a validation split of its own, which a small campaign cannot afford
        # (it raises on a few dozen rows), so the network gets a fixed budget like the other families
        "mlp": lambda: make_pipeline(StandardScaler(),
                                     MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu",
                                                  learning_rate_init=1e-3, max_iter=800, early_stopping=False,
                                                  random_state=seed)),
        "random_forest": lambda: RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1),
        "extra_trees": lambda: ExtraTreesRegressor(n_estimators=400, random_state=seed, n_jobs=-1),
    }
    if importlib.util.find_spec("xgboost") is not None:
        import xgboost as xgb
        out["xgboost"] = lambda: xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6,
                                                  early_stopping_rounds=50, random_state=seed, n_jobs=-1)
    if importlib.util.find_spec("lightgbm") is not None:
        import lightgbm as lgb
        out["lightgbm"] = lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=63,
                                                    random_state=seed, verbose=-1)
    if importlib.util.find_spec("catboost") is not None:
        from catboost import CatBoostRegressor
        out["catboost"] = lambda: CatBoostRegressor(iterations=400, learning_rate=0.03, depth=6,
                                                    random_seed=seed, verbose=0)
    return out


def probe_family(factory, x, y, timeout_s: float = 120.0) -> str | None:
    """Fit one family on a handful of rows **in a child process**; return the reason it is unusable, or ``None``.

    A model whose wheels crash the interpreter (a segmentation fault in a booster compiled against another NumPy, say)
    cannot be caught by ``try/except`` in this process, and a benchmark that dies halfway tells nobody which row was
    at fault. The probe runs the smallest possible fit in a fork, so a crash becomes a reported row."""
    import multiprocessing as mp

    def _child(conn, factory, x, y):
        try:
            model = factory()
            model.fit(x, y)
            conn.send(None)
        except Exception as exc:                                        # noqa: BLE001 - reported, not raised
            conn.send(f"{type(exc).__name__}: {exc}")
        finally:
            conn.close()

    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, factory, x, y), daemon=True)
    proc.start()
    child.close()
    try:
        if parent.poll(timeout_s):
            return parent.recv()
        proc.join(timeout=1.0)
        return "the fit did not answer in time"
    finally:
        parent.close()
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=5.0)
        if proc.exitcode not in (0, None) and proc.exitcode is not None:
            return f"the process died with exit code {proc.exitcode} (a crash in the library, not a fit error)"


def load_table(csv_path: str, *, inputs: list[str], targets: list[str], filter_ok: bool = True) -> dict:
    """Read a campaign's quantities table into ``{"x", "y", "targets", "inputs", "rows"}``.

    Rows marked ``out_of_envelope`` (a run that left the strain envelope of the solver) are dropped when
    ``filter_ok`` is set, and the count of dropped rows is reported — never silently."""
    import csv as _csv

    with open(csv_path) as fh:
        rows = list(_csv.DictReader(fh))
    dropped = 0
    if filter_ok and rows and "out_of_envelope" in rows[0]:
        kept = [r for r in rows if str(r["out_of_envelope"]).lower() not in ("true", "1")]
        dropped = len(rows) - len(kept)
        rows = kept
    x = np.array([[float(r[k]) for k in inputs] for r in rows], float)
    y = np.array([[float(r[k]) for k in targets] for r in rows], float)
    return {"x": x, "y": y, "inputs": list(inputs), "targets": list(targets), "rows": rows,
            "dropped_out_of_envelope": dropped}


def benchmark_tabular(table: dict, *, seed: int = 42, test_fraction: float = 0.15, val_fraction: float = 0.15,
                      standardize_targets: bool = False) -> dict:
    """Fit every available family on the same split and score it per target, with the mean over targets.

    The split is frozen by ``seed`` (70/15/15 as the reference campaign does) and the same rows are used for every
    model, so the comparison is paired."""
    from sklearn.model_selection import train_test_split

    x, y = table["x"], table["y"]
    targets = table["targets"]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_fraction, random_state=seed)
    x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=val_fraction / (1.0 - test_fraction),
                                              random_state=seed)
    scores, predictions, unavailable = {}, {}, {}
    probe_x, probe_y = x_tr[:8], (y_tr[:8, 0] - np.mean(y_tr[:, 0])) / (np.std(y_tr[:, 0]) + 1e-12)
    for name, factory in _factories(seed).items():
        # only the optional third-party boosters are probed in a child process: their wheels are the ones that can die
        # in the interpreter (a segfault cannot be caught here), while the scikit-learn families are safe in-process
        if name in OPTIONAL_FAMILIES:
            why = probe_family(factory, probe_x, probe_y)
            if why is not None:
                unavailable[name] = why
                continue
        per_target = {}
        try:
            for j, target in enumerate(targets):
                # the targets span 0.99 (an absorption fraction) to 10^6 N: every family is fitted on the standardised
                # target and scored on the physical one, otherwise a regularisation strength means a different thing
                # per target and the linear models do not converge at all
                mu, sd = float(np.mean(y_tr[:, j])), float(np.std(y_tr[:, j])) + 1e-12
                model = factory()
                fit_kwargs = {}
                if name in ("xgboost", "lightgbm", "catboost"):
                    fit_kwargs["eval_set"] = [(x_va, (y_va[:, j] - mu) / sd)]
                    if name == "lightgbm":
                        import lightgbm as lgb
                        fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
                model.fit(x_tr, (y_tr[:, j] - mu) / sd, **fit_kwargs)
                y_hat = np.asarray(model.predict(x_te), float).ravel() * sd + mu
                per_target[target] = _metric_set(y_te[:, j], y_hat)
                predictions.setdefault(name, {})[target] = y_hat
        except Exception as exc:                                   # reported, not dropped
            unavailable[name] = f"{type(exc).__name__}: {exc}"
            continue
        scores[name] = {"targets": per_target,
                        "mean": {k: float(np.mean([per_target[t][k] for t in targets]))
                                 for k in ("r2", "mae", "rmse", "mape")}}
    ranked = sorted(((n, s) for n, s in scores.items()), key=lambda kv: -kv[1]["mean"]["r2"])
    return {"scores": scores, "ranked": [n for n, _ in ranked], "predictions": predictions,
            "unavailable": unavailable, "n_train": int(len(x_tr)), "n_test": int(len(x_te)),
            "targets": targets, "inputs": table["inputs"], "table": _tabulate(ranked, targets)}


def _tabulate(ranked, targets) -> str:
    header = "| rank | model | " + " | ".join(t.replace("peak_", "").replace("_", " ") for t in targets) + " | mean R² |"
    sep = "|---" * (len(targets) + 3) + "|"
    lines = [header, sep]
    for i, (name, s) in enumerate(ranked, 1):
        cells = " | ".join(f"{s['targets'][t]['r2']:.3f}" for t in targets)
        lines.append(f"| {i} | {name} | {cells} | {s['mean']['r2']:.3f} |")
    return "\n".join(lines)


OPTIONAL_FAMILIES = ("xgboost", "lightgbm", "catboost")


def feature_sensitivity(table: dict, *, seed: int = 42,
                        models=("catboost", "lightgbm", "xgboost", "extra_trees")) -> dict:
    """Permutation importance of every design variable, averaged over the tree models and the targets that exist.

    Returned as ``{target: {input: relative importance}}`` with the importances of each target normalised to one."""
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    factories = _factories(seed)
    usable = [m for m in models if m in factories]
    if not usable:
        return {}
    probe_x, probe_y = table["x"][:8], table["y"][:8, 0]
    usable = [m for m in usable if probe_family(factories[m], probe_x, probe_y) is None]
    if not usable:
        return {}
    x, y = table["x"], table["y"]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.15, random_state=seed)
    out: dict[str, dict[str, float]] = {}
    for j, target in enumerate(table["targets"]):
        acc = np.zeros(len(table["inputs"]))
        models_used = 0
        for name in usable:
            model = factories[name]()
            try:
                model.fit(x_tr, y_tr[:, j])
            except Exception:                                        # noqa: BLE001 - reported by the caller's table
                continue
            r = permutation_importance(model, x_te, y_te[:, j], n_repeats=20, random_state=seed)
            imp = np.maximum(r.importances_mean, 0.0)
            if imp.sum() <= 0:                                       # a held-out set too small to be informative
                tree = getattr(model, "feature_importances_", None)
                imp = np.asarray(tree, float) if tree is not None else imp
            acc += imp / imp.sum() if imp.sum() > 0 else imp
            models_used += 1
        out[target] = ({k: float(v) for k, v in zip(table["inputs"], acc / max(acc.sum(), 1e-12), strict=False)}
                       if models_used else {})
    return out


def write_table_report(result: dict, path: str, sensitivity: dict | None = None) -> str:
    """Write the tabular leaderboard (markdown) and the same numbers as JSON."""
    lines = ["# Tabular surrogate benchmark", "",
             f"{result['n_train']} training rows, {result['n_test']} held out. Every family uses fixed "
             f"hyperparameters (no per-model tuning).", "", result["table"], ""]
    if result.get("unavailable"):
        lines += ["## Not available on this machine", ""]
        for name, why in result["unavailable"].items():
            lines.append(f"- `{name}`: {why}")
        lines.append("")
    if sensitivity:
        lines += ["## Permutation importance of the design variables", ""]
        for target, imp in sensitivity.items():
            top = sorted(imp.items(), key=lambda kv: -kv[1])[:4]
            lines.append(f"- **{target}**: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
        lines.append("")
    text = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(text)
    with open(path.replace(".md", ".json"), "w") as fh:
        json.dump({"scores": result["scores"], "ranked": result["ranked"], "n_train": result["n_train"],
                   "n_test": result["n_test"], "unavailable": result.get("unavailable", {}),
                   "sensitivity": sensitivity or {}}, fh, indent=2)
    return path
