"""One call per study case, and the tables of the guide, built from the stored results.

* :func:`run_case` -- predictability before modelling, the whole zoo under the protocol, the surrogate-trained
  twins, fan-chart data; returns a JSON-able dict.
* :func:`run_overfitting` -- the anti-overfitting study of a case.
* :func:`markdown_tables` -- the result tables (Markdown) from a list of case results and overfitting studies, so
  every number in the guide is read from the committed JSON.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..predictability import predictability_report
from .learned import GBMForecaster
from .presets import BUDGET, case, overfitting_variants
from .study import overfitting_study, run_zoo, surrogate_trained

__all__ = ["run_case", "run_overfitting", "lyapunov_study", "fan_panels", "surprise_record", "memorise_vs_learn", "markdown_tables"]

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def fan_panels(zoo: dict, data, title: str, n_origins: int = 2, families=("physical", "hybrid", "data")
               ) -> list[dict]:
    """Fan-chart data for the representatives of ``families`` at ``n_origins`` test origins (spread over the
    test period)."""
    panels = []
    fc_all = zoo["_forecasts"]
    H = np.asarray(zoo["protocol"]["horizons"])
    for fam in families:
        name = zoo.get("family_representative", {}).get(fam)
        if name is None or name not in fc_all:
            continue
        rec = fc_all[name]
        o = rec["origins"]
        pick = np.linspace(len(o) * 0.2, len(o) * 0.8, n_origins).astype(int)
        q = rec["forecast"].quantiles(QUANTILES)                    # (5, n, nh)
        for i in pick:
            t0 = int(o[i])
            back = int(min(2 * H.max(), t0))
            panels.append({
                "title": f"{title}: {fam} ({name}), origin t={t0}", "family": fam,
                "t_hist": (np.arange(t0 - back, t0 + 1) * data.dt).tolist(),
                "y_hist": data.y[t0 - back:t0 + 1].tolist(),
                "t_future": (np.arange(t0, t0 + H.max() + 1) * data.dt).tolist(),
                "y_future": data.y[t0:t0 + H.max() + 1].tolist(),
                "horizons_t": ((t0 + np.concatenate([[0], H])) * data.dt).tolist(),
                "quantiles": np.concatenate([np.full((len(QUANTILES), 1), data.y[t0]), q[:, i, :]], axis=1).tolist(),
            })
    return panels


def surprise_record(zoo: dict, data, horizon_index: int = 0) -> dict[str, Any] | None:
    """Per test origin: the absolute error at one horizon of the best non-baseline model (by validation score) and
    of the best baseline at that horizon (by validation RMSE), with the target dates when ``data.meta["dates"]``
    exists. Feeds :func:`~.plots.plot_surprise_calendar`."""
    fc = zoo.get("_forecasts", {})
    models = {k: v for k, v in zoo["models"].items() if "test" in v}
    learned = [k for k, v in models.items() if v["family"] != "baseline" and k in fc]
    H = zoo["protocol"]["horizons"]
    base = zoo.get("best_baseline_by_val", {}).get(str(H[horizon_index]))
    if not learned or base not in fc:
        return None
    best = min(learned, key=lambda k: models[k]["val_score"])
    y = fc[best]["y"][:, horizon_index]
    em = np.abs(fc[best]["forecast"].mean[:, horizon_index] - y)
    eb = np.abs(fc[base]["forecast"].mean[:, horizon_index] - y)
    o = np.asarray(fc[best]["origins"]) + int(H[horizon_index])
    dates = data.meta.get("dates")
    return {"horizon": int(H[horizon_index]), "model": best, "baseline": base,
            "target_index": o.tolist(), "dates": [dates[i] for i in o] if dates else None,
            "abs_error_model": em.tolist(), "abs_error_baseline": eb.tolist()}


def run_case(name: str, *, n: int = 6000, budget: int = BUDGET, seed: int = 0, cache_dir: str | None = None,
             n_surrogates: int = 99, log: Callable[[str], None] = print) -> dict[str, Any]:
    from pinneapple_core.jsonable import to_jsonable
    t0 = time.perf_counter()
    c = case(name, n=n, seed=seed, cache_dir=cache_dir)
    log(f"[{name}] predictability")
    pred = predictability_report(c.data.y, horizons=c.protocol.horizons, dt=c.data.dt, n_surrogates=n_surrogates,
                                 seed=seed)
    log(f"[{name}] zoo of {len(c.specs)} models")
    zoo = run_zoo(c.data, c.specs, c.protocol, budget=budget, keep_forecasts=[s.name for s in c.specs],
                  log=log, seed=seed)
    log(f"[{name}] surrogate-trained twins")
    gbm = lambda: GBMForecaster(n_lags=8, max_iter=200, max_leaf_nodes=15)  # noqa: E731
    sur = {kind: surrogate_trained(gbm, c.data, c.protocol, kind=kind, seed=seed) for kind in ("iaaft", "shuffle")}
    fans = fan_panels(zoo, c.data, c.title)
    surprise = surprise_record(zoo, c.data)
    out = {"case": name, "title": c.title, "truth": c.truth, "n": len(c.data), "dt": c.data.dt,
           "predictability": pred, "zoo": zoo, "surrogate_trained": sur, "fan_panels": fans,
           "surprise": surprise,
           "seconds": time.perf_counter() - t0}
    return to_jsonable(out, digits=4, drop_private=True, unknown="repr")


def run_overfitting(name: str, *, n: int = 6000, seed: int = 0, cache_dir: str | None = None,
                    schemes=("shuffled_kfold", "walk_forward_no_purge", "purged_walk_forward", "purged_kfold"),
                    log: Callable[[str], None] = print) -> dict[str, Any]:
    from pinneapple_core.jsonable import to_jsonable
    c = case(name, n=n, seed=seed, cache_dir=cache_dir)
    v = overfitting_variants(c, physics=name.startswith("lorenz63"))
    with_cv = [k for k in v if k.startswith("overfit")
               or k in ("MLP 64x2 + early stopping", "GBM 7 leaves, 100 rounds, leaf>=50, l2=1")]
    st = overfitting_study(c.data, c.protocol, v, schemes=schemes, schemes_for=with_cv, log=log)
    st["title"] = c.title
    return to_jsonable(st, digits=4, unknown="repr")


def lyapunov_study(dt_ints=(0.1, 0.05), total_times=(10000.0, 50000.0, 200000.0), seeds=(0, 1, 2),
                   log: Callable[[str], None] = print) -> dict[str, Any]:
    """Convergence of the Benettin estimate of the Mackey-Glass (tau = 17) largest Lyapunov exponent with the
    integration step, the length of the run and the seed (initial history and perturbation direction)."""
    from ..synthetic import mackey_glass_lyapunov
    rows = []
    for dt in dt_ints:
        for T in total_times:
            vals, t0 = [], time.perf_counter()
            for s in seeds:
                vals.append(mackey_glass_lyapunov(dt_int=dt, total_time=T, seed=s))
            rows.append({"dt_int": dt, "total_time": T, "values": vals, "mean": float(np.mean(vals)),
                         "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                         "seconds": time.perf_counter() - t0})
            log(f"  MG Lyapunov dt={dt} T={T}: {np.round(vals, 5).tolist()}")
    return {"system": "Mackey-Glass tau=17, beta=0.2, gamma=0.1, n=10", "units": "per unit time, natural log",
            "method": "Benettin two-trajectory renormalisation on the discretised history", "rows": rows}


def memorise_vs_learn(name: str, variants=("overfit: MLP 512x3, 32 lags, no early stop", "MLP 64x2 + early stopping"),
                      *, horizon_index: int = 0, window: int = 200, n: int = 6000, seed: int = 0,
                      cache_dir: str | None = None) -> dict[str, Any]:
    """Refit two training variants of :func:`~.presets.overfitting_variants` on the final split of case ``name``
    and return, for one horizon, the truth and both forecasts over the last ``window`` training origins and the
    first ``window`` test origins, with each variant's RMSE on the whole training and test sets."""
    from .base import targets_at
    c = case(name, n=n, seed=seed, cache_dir=cache_dir)
    v = overfitting_variants(c)
    s = c.protocol.final_split(len(c.data))
    H = c.protocol.horizons
    tr, te = s["train"][-window:], s["test"][:window]
    out = {"case": name, "title": c.title, "horizon": int(H[horizon_index]),
           "train_index": (tr + H[horizon_index]).tolist(), "test_index": (te + H[horizon_index]).tolist(),
           "truth_train": targets_at(c.data, tr, H)[:, horizon_index].tolist(),
           "truth_test": targets_at(c.data, te, H)[:, horizon_index].tolist(), "variants": {}}
    for key in variants:
        m = v[key]().fit(c.data, s["train"], s["val"], H)
        rm = {}
        for part, o in (("train", s["train"]), ("test", s["test"])):
            e = m.predict(c.data, o).mean[:, horizon_index] - targets_at(c.data, o, H)[:, horizon_index]
            rm[part] = float(np.sqrt(np.mean(e ** 2)))
        out["variants"][key] = {"pred_train": m.predict(c.data, tr).mean[:, horizon_index].tolist(),
                                "pred_test": m.predict(c.data, te).mean[:, horizon_index].tolist(),
                                "rmse": rm, "n_params": int(m.n_params())}
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def markdown_tables(results: Sequence[dict], overfit: Sequence[dict] = ()) -> dict[str, str]:
    """Markdown tables: predictability, skill horizon per family, scores of the representatives, the zoo (wall
    time, parameters), surrogate-trained twins, and the overfitting study."""
    T: dict[str, str] = {}
    rows = ["| series | AR order (R² 1-step) | perm. entropy | sample entropy | Rosenstein λ (true λ) | "
            "NPE / std | p shuffle | p IAAFT | p TRA IAAFT | verdict |", "|---" * 10 + "|"]
    for r in results:
        p = r["predictability"]
        st = p["surrogate_tests"]
        lam = r["truth"].get("lyapunov")
        ly = p.get("rosenstein", {}).get("lyapunov")
        lin = st["npe_shuffle"]["p_value"] <= 0.05
        nonlin = st["npe_iaaft"]["p_value"] <= 0.05 or st["tra_iaaft"]["p_value"] <= 0.05
        verdict = "nonlinear structure" if nonlin else ("linear structure only" if lin else "no structure found")
        rows.append(f"| {r['title']} | {p['ar']['order']} ({_f(p['ar']['r2_one_step'])}) | "
                    f"{_f(p['permutation_entropy'])} | {_f(p['sample_entropy'])} | {_f(ly)}"
                    f"{' (' + _f(lam, 3) + ')' if lam else ''} | {_f(st['npe_iaaft']['observed'])} | "
                    f"{_f(st['npe_shuffle']['p_value'])} | {_f(st['npe_iaaft']['p_value'])} | "
                    f"{_f(st['tra_iaaft']['p_value'])} | {verdict}{' (tested on differences)' if p['tested_on'] == 'differences' else ''} |")
    T["predictability"] = "\n".join(rows)

    fams = ["data", "data+physics", "physical", "hybrid"]
    rows = ["| series | horizons | " + " | ".join(f"{f}: best (val) / last h with skill" for f in fams) + " |",
            "|---" * (2 + len(fams)) + "|"]
    for r in results:
        z = r["zoo"]
        H = z["protocol"]["horizons"]
        lam, dt = r["truth"].get("lyapunov"), r["dt"]
        def unit(h, dt=dt, lam=lam):
            return f"{h} ({_f(h * dt * lam)} LT)" if lam else f"{h}"
        cells = []
        for f in fams:
            name = z.get("family_representative", {}).get(f)
            if not name:
                cells.append("–")
                continue
            m = z["models"][name]
            hp, hb = m["skill_horizon_vs_persistence"], m["skill_horizon_vs_best_baseline"]
            cells.append(f"{name}: {unit(hp) if hp else 'none'} / {unit(hb) if hb else 'none'}")
        rows.append(f"| {r['title']} | {', '.join(map(str, H))} | " + " | ".join(cells) + " |")
    T["skill_horizon"] = "\n".join(rows)

    rows = ["| series | model | family | RMSE skill vs persistence per h | CRPS skill vs climatology per h | "
            "coverage of the central 80 % and 95 % intervals (mean over h) | PIT deviation (pooled) | params | final fit [s] |", "|---" * 9 + "|"]
    for r in results:
        z = r["zoo"]
        names = ["persistence", "climatology", "ar", "phase_climatology"] + [z["family_representative"][f] for f in fams
                                                         if f in z.get("family_representative", {})]
        for name in names:
            m = z["models"].get(name)
            if not m or "test" not in m:
                continue
            t = m["test"]
            cov = t["coverage"].get("0.95")
            cov80 = t["coverage"].get("0.8")
            covs = f"{_f(np.mean(cov80))} (80 %), {_f(np.mean(cov))} (95 %)" if cov else "–"
            pit = (t["pit"]["pooled"] or {}).get("deviation") if t["pit"]["pooled"] else None
            rows.append(f"| {r['title']} | {name} | {m['family']} | "
                        f"{' '.join(_f(s) for s in t['skill_rmse'])} | "
                        f"{' '.join(_f(s) for s in t.get('crps_skill_vs_climatology', []))} | {covs} | {_f(pit)} | "
                        f"{m['n_params']} | {_f(m['fit_seconds'], 1)} |")
    T["scores"] = "\n".join(rows)

    rows = ["| series | model | family | params | tuning configs | tuning [s] | final fit [s] | test RMSE skill "
            "(mean over h) | significant vs best baseline (Holm) at h |", "|---" * 9 + "|"]
    for r in results:
        z = r["zoo"]
        H = z["protocol"]["horizons"]
        for name, m in z["models"].items():
            if "test" not in m:
                rows.append(f"| {r['title']} | {name} | {m['family']} | failed: {m.get('error', '')[:80]} |||||")
                continue
            sig = [str(h) for h, d in zip(H, m["dm_vs_best_baseline"], strict=False) if d and (d.get("p_holm") or 1) < 0.05]
            rows.append(f"| {r['title']} | {name} | {m['family']} | {m['n_params']} | {m['n_configs']} | "
                        f"{_f(m['tune_seconds'], 1)} | {_f(m['fit_seconds'], 1)} | "
                        f"{_f(float(np.mean(m['test']['skill_rmse'])))} | {', '.join(sig) or 'none'} |")
    T["zoo"] = "\n".join(rows)

    rows = ["| series | GBM trained on real data | on IAAFT surrogate | on shuffled surrogate | real − IAAFT |",
            "|---" * 5 + "|"]
    for r in results:
        s = r["surrogate_trained"]
        rows.append(f"| {r['title']} | {_f(s['iaaft']['real']['mean_skill'])} | {_f(s['iaaft']['iaaft']['mean_skill'])}"
                    f" | {_f(s['shuffle']['shuffle']['mean_skill'])} | {_f(s['iaaft']['real_minus_surrogate'])} |")
    T["surrogate_trained"] = "\n".join(rows)

    rows = ["| series | variant | params | train | val | test | test − train | CV: shuffled k-fold | "
            "CV: walk-forward, no purge | CV: purged walk-forward | CV: purged k-fold + embargo |", "|---" * 11 + "|"]
    for st in overfit:
        for name, v in st["variants"].items():
            cv = v["cv"]
            def g(k, cv=cv):
                return _f(cv[k]["estimate"], 3) if k in cv else ""
            rows.append(f"| {st['title']} | {name} | {v['n_params']} | {_f(v['train'], 3)} | {_f(v['val'], 3)} | "
                        f"{_f(v['test'], 3)} | {_f(v['gap_test_minus_train'], 3)} | {g('shuffled_kfold')} | "
                        f"{g('walk_forward_no_purge')} | {g('purged_walk_forward')} | {g('purged_kfold')} |")
    T["overfitting"] = "\n".join(rows)
    return T
