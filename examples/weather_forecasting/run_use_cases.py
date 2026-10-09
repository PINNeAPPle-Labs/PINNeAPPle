"""Weather forecasting use cases: a global PINNeAPPle forecast model on ERA5, how many days it is good for, and
extreme events forecast against what happened.

Run after ``pinneapple_physics.weather.train.train`` (and ``score_baselines.py`` for the reference models):

    python run_use_cases.py <store> <model.pt> <out_dir>

Writes to ``out_dir``: ``skill_<var>.png`` (ACC and RMSE by lead against IFS HRES, Pangu-Weather, Keisler,
NeuralGCM, persistence and climatology), ``horizon.json`` (useful and no-skill horizons per variable, and the
per-forecast horizon from the ensemble checked against the real one), ``<event>.gif`` (ERA5 against the forecast,
lead running, skill strip) and ``events.json`` (how many days ahead each event was well forecast).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pinneapple_physics.weather.data import Era5Store, climatology, climatology_at
from pinneapple_physics.weather.evaluate import (
    SCORED,
    PublishedForecasts,
    acc,
    first_crossing,
    horizons,
    score_forecasts,
)
from pinneapple_physics.weather.events import EVENTS, event_states, region_mask
from pinneapple_physics.weather.train import Forecaster
from pinneapple_physics.weather.viz import forecast_gif, skill_figure

STEPS = 40                      # 10 days of 6-hour steps
LEADS_H = np.arange(STEPS + 1) * 6
MODEL = "PINNeAPPle"
TOLERANCE = {"t2m": 2.0, "msl": 500.0, "tp6": 0.35}     # event score: K, Pa, relative error of the total


def model_forecast(fc: Forecaster, store: Era5Store):
    def fn(i0, steps):
        x = np.asarray(store.state[i0 - 1:i0 + 1], np.float32)
        r = fc.rollout(x[1], x[0], store.times[i0], steps)[0]
        return store.denorm(r)
    return fn


# ------------------------------------------------------------------ ensemble and per-forecast horizon

def seasonal_perturbations(store: Era5Store, time, members: int, scale: float, rng) -> np.ndarray:
    """Differences of two ERA5 states from the same season of other training years, scaled: perturbations with the
    structure of real weather (not white noise)."""
    t = np.datetime64(time, "ns")
    doy = (t - t.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
    pool = store.years(1990, 2017)
    pd = (store.times[pool] - store.times[pool].astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
    pool = pool[np.abs(((pd - doy) + 182) % 365 - 182) < 30]
    out = []
    for _ in range(members):
        a, b = rng.choice(pool, 2, replace=False)
        out.append((np.asarray(store.state[a], np.float32) - np.asarray(store.state[b], np.float32)) / np.sqrt(2))
    p = scale * np.stack(out)
    p -= p.mean(0, keepdims=True)                       # the ensemble mean starts from the analysis
    return p


def ensemble_scores(fc, store, clim, inits, members, scale, var="z500", seed=0):
    """Per forecast and lead: ACC of the ensemble mean and the normalised spread (spread / climatological std)."""
    rng = np.random.default_rng(seed)
    c = store.channels.index(var)
    w = store.lat_weights()
    accs, spreads = [], []
    for i0 in inits:
        x = np.asarray(store.state[i0 - 1:i0 + 1], np.float32)
        p = seasonal_perturbations(store, store.times[i0], members, scale, rng)
        r = fc.rollout(x[1], x[0], store.times[i0], STEPS, perturb=p)[:, :, c]       # (m, lead, lat, lon)
        truth = np.asarray(store.state[i0:i0 + STEPS + 1, c], np.float32)
        cl = np.stack([climatology_at(clim, store.times[i0 + s])[c] for s in range(STEPS + 1)])
        mean = r.mean(0)
        accs.append(acc(mean, truth, cl, w))
        spreads.append(np.sqrt(np.mean(r.var(0) * w[:, None], axis=(-2, -1))))
    return np.array(accs), np.array(spreads)


def calibrate_spread(accs, spreads, threshold=0.6):
    """Spread at which the ACC crosses ``threshold`` on average: bin forecasts by spread, interpolate."""
    a, s = accs[:, 1:].ravel(), spreads[:, 1:].ravel()
    order = np.argsort(s)
    a, s = a[order], s[order]
    bins = np.array_split(np.arange(len(s)), 20)
    bs = np.array([s[b].mean() for b in bins])
    ba = np.array([a[b].mean() for b in bins])
    i = np.argmax(ba < threshold) if (ba < threshold).any() else len(ba) - 1
    if i == 0:
        return float(bs[0])
    return float(bs[i - 1] + (threshold - ba[i - 1]) * (bs[i] - bs[i - 1]) / (ba[i] - ba[i - 1]))


# ------------------------------------------------------------------ events

def event_score(field, lat, lon, ev, w):
    m = region_mask(lat, lon, ev.region)
    v = field[..., m]
    ww = np.broadcast_to(w[:, None], m.shape)[m]
    if ev.score == "min":
        return v.min(-1)
    if ev.score == "max":
        return v.max(-1)
    if ev.score == "sum":
        return (v * ww).sum(-1) / ww.sum()
    return (v * ww).sum(-1) / ww.sum()


def run_event(ev, fc, store, clim, out, skill, horizon_by_var, pub, log=print):
    leads_needed = STEPS
    times, states = event_states(store, ev, leads_needed, cache_dir=out / "_event_cache", log=log)
    c = store.channels.index(ev.var)
    w = store.lat_weights()
    phys = store.denorm(states.transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)            # (T, C, lat, lon)
    r = store.denorm(fc.rollout(states[1], states[0], times[1], STEPS)[0].transpose(1, 0, 2, 3)
                     ).transpose(1, 0, 2, 3)
    peak = np.datetime64(ev.peak, "ns")
    k_peak = int((peak - times[1]) / np.timedelta64(6, "h"))
    # how far ahead: forecasts started 1..6 days before the peak (when the states exist)
    ahead = []
    for days in range(1, 7):
        t_init = peak - np.timedelta64(24 * days, "h")
        try:
            i0 = store.index(t_init)
            x = np.asarray(store.state[i0 - 1:i0 + 1], np.float32)
            f = store.denorm(fc.rollout(x[1], x[0], t_init, 4 * days)[0, -1])
            obs = store.denorm(np.asarray(store.state[i0 + 4 * days], np.float32))
        except KeyError:
            j = int((t_init - times[1]) / np.timedelta64(6, "h")) + 1
            if j < 1:
                continue
            f = store.denorm(fc.rollout(states[j], states[j - 1], t_init, 4 * days)[0, -1])
            obs = phys[j + 4 * days]
        accum = ev.var == "tp6"
        sf = event_score(f[c], store.lat, store.lon, ev, w)
        so = event_score(obs[c], store.lat, store.lon, ev, w)
        err = abs(sf - so) / max(abs(so), 1e-9) if accum else abs(sf - so)
        row = {"days_ahead": days, "forecast": float(sf), "observed": float(so), "error": float(err)}
        for name in ("IFS HRES", "Pangu-Weather"):
            try:
                i0 = store.index(t_init)
                g = pub(name)(i0, 4 * days)
                if g is not None and np.isfinite(g[-1, c]).all():
                    sg = event_score(g[-1, c], store.lat, store.lon, ev, w)
                    row[name] = float(abs(sg - so) / max(abs(so), 1e-9) if accum else abs(sg - so))
            except KeyError:
                pass
        ahead.append(row)
        log(f"{ev.key}: {days} days ahead, error {err:.3g}")
    tol = TOLERANCE.get(ev.var, np.inf)
    good = [a["days_ahead"] for a in ahead if a["error"] <= tol]
    well_ahead = 0
    for d in range(1, 7):
        if d in good:
            well_ahead = d
        else:
            break
    hz = horizon_by_var.get(ev.var, horizon_by_var["t2m"])
    acc_curve = skill[MODEL].get(ev.var, skill[MODEL]["t2m"])["acc"]
    extra = {n: (LEADS_H, skill[n][ev.var]["acc"]) for n in ("IFS HRES", "Pangu-Weather")
             if n in skill and ev.var in skill[n] and np.isfinite(skill[n][ev.var]["acc"][1:]).any()}
    n_frames = min(STEPS, k_peak + 8)
    gif = forecast_gif(out / f"{ev.key}.gif", ev.var, phys[1:n_frames + 2, c], r[:n_frames + 1, c], store.lat,
                       store.lon, times[1:n_frames + 2], skill_leads_h=LEADS_H, skill_acc=acc_curve, horizon=hz,
                       region=ev.region, title=ev.title, model_name=MODEL, vmin=ev.vmin, vmax=ev.vmax,
                       extra_series=extra)
    return {"event": ev.key, "title": ev.title, "what": ev.what, "variable": ev.var, "score": ev.score,
            "tolerance": tol, "ahead": ahead, "well_forecast_days_ahead": well_ahead, "gif": gif.name}


def main(store_path, model_path, out_dir, inits_every_days=3, members=8):
    store = Era5Store(store_path)
    clim = climatology(store)
    fc = Forecaster.load(model_path, store)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads((Path(store_path) / "baselines_2020.json").read_text())
    inits = [store.index(t) for t in base["_meta"]["inits"]]
    res = score_forecasts(store, clim, {MODEL: model_forecast(fc, store)}, inits, STEPS)
    skill = {MODEL: {v: {k: np.nanmean(a, 0) for k, a in d.items()} for v, d in res[MODEL].items()}}
    for n, d in base.items():
        if n != "_meta":
            skill[n] = {v: {k: np.array(x, float) for k, x in dd.items() if k != "n"} for v, dd in d.items()}
    horizon_by_var, table = {}, {}
    for v in SCORED:
        clim_rmse = skill["climatology"][v]["rmse"]
        hz = {n: horizons(LEADS_H, s[v]["acc"], s[v]["rmse"], clim_rmse) for n, s in skill.items()
              if v in s and np.isfinite(s[v]["acc"][1:]).any() and n != "climatology"}
        horizon_by_var[v] = hz[MODEL]
        table[v] = {n: {"useful_days": h["useful_hours"] / 24, "no_skill_days": h["no_skill_hours"] / 24,
                        "rmse_day3": float(skill[n][v]["rmse"][12]), "rmse_day5": float(skill[n][v]["rmse"][20]),
                        "acc_day5": float(skill[n][v]["acc"][20])} for n, h in hz.items()}
        curves = {n: skill[n][v]["acc"] for n in hz}
        skill_figure(out / f"skill_{v}.png", LEADS_H, curves, {n: hz[n]["useful_hours"] for n in hz}, v)
        print(v, {n: round(t["useful_days"], 2) for n, t in table[v].items()}, flush=True)
    # ensemble: calibrate the spread on 2019, check the per-forecast horizon on 2020
    val = store.years(2019, 2019)
    val = val[(store.times[val].astype("datetime64[h]").astype(int) % 24 == 0)][::6]
    best = None
    for scale in (0.05, 0.1, 0.2):
        a, s = ensemble_scores(fc, store, clim, val[:20], members, scale)
        err = 1 - a[:, 12].mean()
        ratio = s[:, 12].mean() / max(err, 1e-9)
        print(f"ensemble scale {scale}: spread/err-proxy at day 3 {ratio:.2f}", flush=True)
        if best is None or abs(np.log(ratio)) < abs(np.log(best[1])):
            best = (scale, ratio)
    scale = best[0]
    a_val, s_val = ensemble_scores(fc, store, clim, val, members, scale)
    s_star = calibrate_spread(a_val, s_val)
    a_te, s_te = ensemble_scores(fc, store, clim, inits[::2], members, scale, seed=1)
    pred = np.array([first_crossing(LEADS_H, s, s_star, below=False) for s in s_te]) / 24
    real = np.array([first_crossing(LEADS_H, a, 0.6, below=True) for a in a_te]) / 24
    ok = np.isfinite(pred) & np.isfinite(real)
    per_forecast = {"variable": "z500", "members": members, "perturbation_scale": scale, "spread_at_acc_0.6": s_star,
                    "n": int(ok.sum()), "mae_days": float(np.mean(np.abs(pred[ok] - real[ok]))),
                    "corr": float(np.corrcoef(pred[ok], real[ok])[0, 1]) if ok.sum() > 2 else None,
                    "real_range_days": [float(np.min(real[ok])), float(np.max(real[ok]))],
                    "constant_horizon_mae_days": float(np.mean(np.abs(np.median(real[ok]) - real[ok]))),
                    "pairs": [[float(p), float(r)] for p, r in zip(pred[ok], real[ok], strict=True)]}
    print("per-forecast horizon:", {k: v for k, v in per_forecast.items() if k != "pairs"}, flush=True)
    (out / "horizon.json").write_text(json.dumps({"table": table, "per_forecast": per_forecast,
                                                   "leads_h": LEADS_H.tolist(),
                                                   "acc": {n: {v: skill[n][v]["acc"].tolist() for v in skill[n]}
                                                           for n in skill}}, default=float))
    pub = PublishedForecasts(store, names=("IFS HRES", "Pangu-Weather"))
    events = [run_event(ev, fc, store, clim, out, skill, horizon_by_var, pub) for ev in EVENTS]
    (out / "events.json").write_text(json.dumps(events, default=float))
    for e in events:
        print(f"{e['event']}: well forecast {e['well_forecast_days_ahead']} days ahead", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
