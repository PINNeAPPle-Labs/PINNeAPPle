"""Study-level steps of the ENSO study, each returning JSON-able results (the example script only calls these).

* :func:`solver_checks` -- every solver against theory (Kelvin / Rossby speeds and crossing times, convergence,
  Gill vs closed form, recharge oscillator vs eigenvalues for several time steps, delayed oscillator vs its root).
* :func:`enso_scores` -- ACC per lead, skill by target month, categorical scores for every model of a merged zoo.
* :func:`case_panels` -- plume data (quantiles at the evaluated leads) for the case-study origins.
* :func:`spring_barrier_hindcasts` -- blocked cross-validated hindcasts of cheap models over the development
  period (more forecasts per target month than the test period has).
* :func:`phase_portrait` -- observed (h, T) and the fitted recharge oscillators' paths and realisations.
* :func:`shallow_water_pinn_study` -- the equatorial-wave PINN on synthetic data from the 2-D model (known c, r)
  and on the observations of a window, with the Hovmoller fields of observations, the ERA5-forced 2-D ocean and
  the PINN.
* :func:`slab_frames` -- monthly SST and thermocline-depth fields for the 3-D render of an event.
* :func:`markdown_tables` -- the guide's tables from the stored results.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import evaluation as E
from . import intermediate as IM
from . import oscillators as O

__all__ = ["solver_checks", "enso_scores", "case_panels", "skill_against", "spring_barrier_hindcasts", "phase_portrait",
           "shallow_water_pinn_study", "slab_frames", "markdown_tables", "QUANTILES"]

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _ym_float(m: str) -> float:
    return int(m[:4]) + (int(m[5:7]) - 0.5) / 12


def solver_checks() -> dict[str, Any]:
    out: dict[str, Any] = {"kelvin": IM.kelvin_wave_test(), "rossby": IM.rossby_wave_test(),
                           "kelvin_convergence": IM.convergence_study(), "gill": IM.gill_test()}
    p = O.ROParams.constant(R=-0.05, F1=0.02, F2=1.5, eps=0.03)
    e = O.ro_eigen(**p.mean())
    rows = []
    for sub in (1, 2, 4, 8, 16):
        x = O.simulate_recharge(p, [1.0, 0.0], 600, sub=sub, stochastic=False)[0, 0, :, 0]
        st = O.oscillation_stats(x, 1.0)
        rows.append({"sub": sub, "period": st["period"], "growth": st["growth"],
                     "period_rel_error": st["period"] / e["period"] - 1, "growth_rel_error": st["growth"] / e["growth"] - 1})
    out["recharge_oscillator"] = {"params": p.mean(), "period_theory": e["period"], "growth_theory": e["growth"],
                                  "rows": rows}
    a, b, tau = 1 / 6, 1 / 3, 6.0
    r = O.do_roots(a, b, tau)[0]
    drows = []
    for dt in (1.0, 0.5, 0.25, 0.1):
        x = O.simulate_delayed(a, b, 0.0, tau, np.full(int(round(tau / dt)) + 1, 0.01), int(600 / dt), dt)
        st = O.oscillation_stats(x, dt, skip=int(100 / dt))
        drows.append({"dt": dt, **st, "period_rel_error": st["period"] / (2 * np.pi / r.imag) - 1,
                      "growth_rel_error": st["growth"] / r.real - 1})
    out["delayed_oscillator"] = {"a": a, "b": b, "tau": tau, "root": [r.real, r.imag],
                                 "period_theory": 2 * np.pi / r.imag, "rows": drows}
    return out


def _forecast_of(merged, name):
    rec = merged["_forecasts"][name]
    return rec["forecast"], rec["y"], np.asarray(rec["origins"])


def enso_scores(merged: dict[str, Any], month_of_year: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """ACC per lead, target-month stratified ACC / RMSE and categorical scores of every model in ``merged``."""
    H = merged["protocol"]["horizons"]
    out = {}
    for name, m in merged["models"].items():
        if "test" not in m or name not in merged["_forecasts"]:
            continue
        f, y, o = _forecast_of(merged, name)
        out[name] = {"family": m["family"], "acc": E.acc(f.mean, y),
                     "by_target_month": E.by_target_month(f.mean, y, o, month_of_year, H, min_n=6),
                     "categorical": E.categorical(f.mean, y, threshold)}
    return out


def case_panels(parts: dict[str, dict], merged: dict[str, Any], data, *, families=("physical", "hybrid", "data",
                                                                                     "data+physics"),
                events: dict[str, Any] | None = None, months_before: int = 18,
                leads: Sequence[int] = (1, 3, 6, 9, 12, 18, 24)) -> list[dict]:
    """Plume panels: for every case-study origin, the representative of each family (from ``merged``) with its
    quantiles at the evaluated leads. ``parts[name]["_cases"]`` holds forecasts from origins after the development
    period; ``events[label][name]`` (from :func:`.study.event_hindcast`) those of refitted models.

    Each panel's ``sample`` says whether the forecasts are out of sample and why: "test period", "after the test
    period", "refit: data to <month>" (an event refit; no data after that month entered the fit) or "in-sample
    (training period)" (a model fitted on data that include the target years); the title carries it."""
    months = data.meta["months"]
    Hall = list(merged["protocol"]["horizons"])
    keep = [Hall.index(h) for h in leads if h in Hall]
    H = np.asarray(Hall)[keep]
    reps = merged.get("family_representative", {})
    proto = merged["protocol"]
    dev_end = int(proto["dev_end"])
    last_test = int(proto["n"]) - max(Hall) - 1
    by_origin: dict[tuple[str, int], list[dict]] = {}
    sample: dict[tuple[str, int], str] = {}
    for fam in families:
        name = reps.get(fam)
        if name is None:
            continue
        srcs = [(lab, rec, None) for lab, rec in parts.get(name, {}).get("_cases", {}).items()]
        srcs += [(lab, ev[name], ev[name].get("data_end")) for lab, ev in (events or {}).items() if name in ev]
        for lab, rec, data_end in srcs:
            q = rec["forecast"].quantiles(QUANTILES)[:, :, keep]        # (5, n, n_leads)
            for i, o in enumerate(rec["origins"]):
                o = int(o)
                if data_end is not None:
                    tag = f"out-of-sample: refit, data to {data_end}"
                elif o >= dev_end:
                    tag = "out-of-sample: test period" if o <= last_test else "out-of-sample: after the test period"
                else:
                    tag = "in-sample (training period)"
                sample.setdefault((lab, o), tag)
                by_origin.setdefault((lab, o), []).append(
                    {"family": fam, "name": name, "t": [_ym_float(months[min(o + h, len(months) - 1)])
                                                       if o + h < len(months) else
                                                       _ym_float(months[o]) + h / 12 for h in H],
                     "q": q[:, i, :].tolist()})
    panels = []
    for (lab, o), fams in sorted(by_origin.items(), key=lambda kv: kv[0][1]):
        lo, hi = max(0, o - months_before), min(len(months), o + int(max(Hall)) + 1)
        panels.append({"title": f"{lab}: forecasts from {months[o]} [{sample[(lab, o)]}]", "event": lab,
                       "origin": months[o], "sample": sample[(lab, o)],
                       "origin_t": _ym_float(months[o]), "t_obs": [_ym_float(m) for m in months[lo:hi]],
                       "y_obs": data.y[lo:hi].tolist(), "fams": fams})
    return panels


def skill_against(merged: dict[str, Any], reference: str = "climatology", alpha: float = 0.05) -> dict[str, Any]:
    """RMSE skill 1 - RMSE / RMSE(reference) on the test period for every model of ``merged`` (with its kept
    forecasts), and one-sided Diebold-Mariano tests (squared errors, HLN, long-run variance over the h-step
    overlap) of each non-baseline model against the reference, Holm-adjusted across those models at each lead."""
    from ..metrics_ext.comparison import diebold_mariano, holm
    H = list(merged["protocol"]["horizons"])
    stride = merged["protocol"].get("test_stride", 1)
    fc = merged["_forecasts"]
    ref = fc[reference]
    e_ref = ref["forecast"].mean - ref["y"]
    rmse_ref = np.sqrt(np.mean(e_ref ** 2, axis=0))
    out: dict[str, Any] = {"reference": reference, "horizons": H, "models": {}}
    learned = [n for n, m in merged["models"].items() if "test" in m and m["family"] != "baseline" and n in fc]
    pvals = {n: [] for n in learned}
    for name, m in merged["models"].items():
        if "test" not in m or name not in fc:
            continue
        e = fc[name]["forecast"].mean - fc[name]["y"]
        sk = 1 - np.sqrt(np.mean(e ** 2, axis=0)) / rmse_ref
        out["models"][name] = {"family": m["family"], "skill_rmse": sk.tolist()}
        if name in pvals:
            for j, h in enumerate(H):
                r = diebold_mariano(e[:, j], e_ref[:, j], max_lag=int(np.ceil(h / stride)) - 1)
                pvals[name].append(r.p_value_better)
    for j in range(len(H)):
        adj = holm({n: pvals[n][j] for n in learned})
        for n in learned:
            out["models"][n].setdefault("p_holm", []).append(float(adj[n]))
    for n in learned:
        out["models"][n]["significant"] = [p < alpha for p in out["models"][n]["p_holm"]]
    return out


def spring_barrier_hindcasts(data, prot, makers: dict[str, Any], *, block: int = 48) -> dict[str, Any]:
    """Blocked, purged cross-validated hindcasts over the development period for each maker; returns, per model,
    the target-month stratified ACC / RMSE and the ACC per lead."""
    H = prot.horizons
    dev_end = prot.final_split(len(data))["dev_end"]
    last = dev_end - max(H) - 1
    moy = np.asarray(data.meta["month_of_year"])
    out = {}
    for name, make in makers.items():
        hc = E.blocked_hindcast(make, data, H, block=block, first=prot.min_history, last=last)
        out[name] = {"by_target_month": E.by_target_month(hc["mean"], hc["y"], hc["origins"], moy, H, min_n=8),
                     "acc": E.acc(hc["mean"], hc["y"]), "n_origins": int(len(hc["origins"])),
                     "first": data.meta["months"][int(hc["origins"][0])],
                     "last": data.meta["months"][int(hc["origins"][-1])]}
    return out


def phase_portrait(data, prot, *, seasonal=("R", "F1", "F2", "eps"), seed: int = 0,
                   onsets: Sequence[str] = ("1997-04", "2009-06", "2015-03", "2023-04"), n_path: int = 48
                   ) -> dict[str, Any]:
    """Observed (h, T) over the record (and its 3-month running mean), the deterministic paths of the fitted
    recharge oscillators (constant and seasonal) for ``n_path`` months from the observed state at each El Nino
    onset in ``onsets``, and one stochastic realisation of each with the record's length; parameters fitted on the
    training segment."""
    from ..forecast_lab.base import train_end
    sp = prot.final_split(len(data))
    end = train_end(sp["train"], prot.horizons)
    hc = list(data.meta["columns"]).index("wwv")
    moy = np.asarray(data.meta["month_of_year"])
    T, h = data.y, data.obs[:, hc]
    fits = {"ro": O.fit_recharge(T[:end], h[:end], moy[:end]),
            "ro_seasonal_multiplicative": O.fit_recharge(T[:end], h[:end], moy[:end], seasonal=seasonal,
                                                         multiplicative=True)}
    out = {"observed": {"T": T.tolist(), "h": h.tolist(), "months": list(data.meta["months"])}, "models": {}}
    n = len(T)
    for name, p in fits.items():
        det = O.simulate_recharge(p, [T[0], h[0]], n - 1, stochastic=False, month_offsets=moy[:1])[0, 0]
        sto = O.simulate_recharge(p, [T[0], h[0]], n - 1, seed=seed, month_offsets=moy[:1])[0, 0]
        e = O.ro_eigen(**p.mean())
        months = list(data.meta["months"])
        paths = {}
        for on in onsets:
            if on in months:
                i = months.index(on)
                P = O.simulate_recharge(p, [T[i], h[i]], n_path, stochastic=False, month_offsets=moy[i:i + 1])[0, 0]
                paths[on] = {"T": P[:, 0].tolist(), "h": P[:, 1].tolist()}
        out["models"][name] = {"params": p.to_dict(), "onset_paths": paths,
                               "deterministic": {"T": det[:, 0].tolist(), "h": det[:, 1].tolist()},
                               "stochastic": {"T": sto[:, 0].tolist(), "h": sto[:, 1].tolist(),
                                              "std_T": float(sto[:, 0].std()), "std_h": float(sto[:, 1].std())},
                               "eigen": {"growth": e["growth"], "period": e["period"]}}
    out["observed"]["std_T"], out["observed"]["std_h"] = float(T.std()), float(h.std())
    k = np.ones(3) / 3
    out["observed"]["T_smooth"] = np.convolve(T, k, "same").tolist()
    out["observed"]["h_smooth"] = np.convolve(h, k, "same").tolist()
    lags = list(range(-12, 13, 3))
    out["lag_corr_T_vs_h"] = {"lags": lags, "corr": [float(np.corrcoef(T[max(0, k):n + min(0, k)],
                                                                       h[max(0, -k):n + min(0, -k)])[0, 1]) for k in lags],
                              "note": "corr(T(t + lag), h(t)); positive lag: h leads"}
    return out


def shallow_water_pinn_study(ds, window: tuple[str, str] = ("1996-01", "1999-12"), *, epochs: int = 6000,
                             n_col: int = 4096, pde_weight: float = 100.0, spin_months: int = 36, seed: int = 0
                             ) -> dict[str, Any]:
    """The equatorial-wave PINN (:class:`.pinn.EquatorialWavePINN`) on (a) synthetic equatorial h from the 2-D
    shallow-water model forced by ERA5 stress (known c = 2.7 m/s, r = 1/900 d, closed west wall) and (b) observed
    2S-2N D20 and SST anomalies, both over ``window``; plus the Hovmoller fields (observed, the ERA5-forced 2-D
    model scaled by its fitted amplitude factor, the PINN)."""
    from .pinn import EquatorialWavePINN, project_forcing
    g = ds.grid
    i0, i1 = ds.months.index(window[0]), ds.months.index(window[1]) + 1
    j0 = max(0, i0 - spin_months)
    oc = IM.ShallowWaterOcean(g, c=2.7, r_days=900.0, dtype="float32")
    hfull, _ = IM.forced_ocean_run(oc, ds.taux[j0:i1])
    h2d = hfull[i0 - j0:]
    eq = np.abs(g.lat) < 2
    heq = h2d[:, eq].mean(1)
    h_syn = np.stack([np.interp(ds.d20_lon, g.lon, heq[t]) for t in range(len(heq))])
    obs_h = ds.d20_eq[i0:i1]
    A, B = h_syn.ravel(), obs_h.ravel()
    scale = float(A @ B / (A @ A))
    tx = ds.taux[i0:i1]
    X0, X2 = project_forcing(tx, g.lat)
    tau_eq = tx[:, eq].mean(1)
    T_eq = ds.sst_maps[i0:i1][:, eq].mean(1)
    T_obs = np.stack([np.interp(ds.d20_lon, g.lon, T_eq[t]) for t in range(len(T_eq))])
    out: dict[str, Any] = {"window": list(window), "model_vs_obs": {
        "scale": scale, "corr": float(np.corrcoef(A, B)[0, 1]),
        "rmse_scaled": float(np.sqrt(np.mean((scale * A - B) ** 2))), "obs_std": float(B.std())}}
    syn = EquatorialWavePINN(ds.d20_lon, h_syn, None, g.lon_u, X0, X2, tau_eq, sst=False, pde_weight=pde_weight,
                             seed=seed)
    out["synthetic"] = {**syn.fit(epochs=epochs, n_col=n_col), "truth": {"c_m_s": 2.7, "damping_days": 900.0,
                                                                        "a_W_two_mode_closed_wall": 2.0}}
    real = EquatorialWavePINN(ds.d20_lon, obs_h, T_obs, g.lon_u, X0, X2, tau_eq, sst=True, pde_weight=pde_weight,
                              seed=seed)
    out["observed"] = real.fit(epochs=epochs, n_col=n_col)
    pg = real.predict_grid(ds.d20_lon)
    hc = np.linspace(-40, 40, 21)
    out["closure"] = {"h": hc.tolist(), "lon": [160.0, 210.0, 250.0],
                      "g": [real.closure_curve(lo, hc).tolist() for lo in (160.0, 210.0, 250.0)]}
    out["hovmoller"] = {"lon": ds.d20_lon.tolist(), "t": [_ym_float(m) for m in ds.months[i0:i1]],
                        "h_obs": obs_h.tolist(), "h_model2d": (scale * h_syn).tolist(), "h_pinn": pg["h"].tolist(),
                        "T_obs": T_obs.tolist(), "T_pinn": pg["T"].tolist(),
                        "lon_model": g.lon.tolist(), "h_model2d_full": (scale * heq).tolist()}
    return out


def slab_frames(ds, oisst, casts, start: str, end: str, *, spin_months: int = 36, lon=(121.0, 279.0),
                lat=(-19.5, 19.5)) -> list[dict]:
    """Monthly frames for :func:`.render.render_frames`: full SST [deg C] from OISST (1 deg) and thermocline depth
    = the D20 climatology of the casts (2 deg lat x 10 deg lon bins, base period, smoothed and interpolated to
    the SST grid) plus the anomaly of the ERA5-forced 2-D ocean model, scaled to D20 by its fitted factor."""
    from scipy.ndimage import gaussian_filter

    from . import data as D
    g = ds.grid
    i0, i1 = ds.months.index(start), ds.months.index(end) + 1
    j0 = max(0, i0 - spin_months)
    oc = IM.ShallowWaterOcean(g, c=2.7, r_days=900.0, dtype="float32")
    hfull, _ = IM.forced_ocean_run(oc, ds.taux[j0:i1])
    eq = np.abs(g.lat) < 2
    heq = hfull[:, eq].mean(1)
    sim = np.stack([np.interp(ds.d20_lon, g.lon, heq[t]) for t in range(len(heq))])
    obs = ds.d20_eq[j0:i1]
    scale = float(sim.ravel() @ obs.ravel() / (sim.ravel() @ sim.ravel()))
    lon_e, lat_e = np.arange(120, 281, 10.0), np.arange(-20, 21, 2.0)
    clim = D.d20_climatology(casts, lon_e, lat_e, ds.base)
    lon_c, lat_c = 0.5 * (lon_e[1:] + lon_e[:-1]), 0.5 * (lat_e[1:] + lat_e[:-1])
    # fill empty bins from their row (latitude) neighbours, then smooth
    for r in range(clim.shape[0]):
        ok = np.isfinite(clim[r])
        if ok.sum() >= 2:
            clim[r] = np.interp(lon_c, lon_c[ok], clim[r, ok])
    for c in range(clim.shape[1]):
        ok = np.isfinite(clim[:, c])
        if ok.sum() >= 2:
            clim[:, c] = np.interp(lat_c, lat_c[ok], clim[ok, c])
    clim = gaussian_filter(np.nan_to_num(clim, nan=float(np.nanmean(clim))), 0.8)
    from scipy.interpolate import RegularGridInterpolator
    la = (oisst.lat >= lat[0]) & (oisst.lat <= lat[1])
    lo = (oisst.lon >= lon[0]) & (oisst.lon <= lon[1])
    LA, LO = np.meshgrid(oisst.lat[la], oisst.lon[lo], indexing="ij")
    P = np.column_stack([LA.ravel(), LO.ravel()])
    base = RegularGridInterpolator((lat_c, lon_c), clim, bounds_error=False, fill_value=None)(P).reshape(LA.shape)
    frames = []
    for k in range(i0, i1):
        anom = RegularGridInterpolator((g.lat, g.lon), hfull[k - j0] * scale, bounds_error=False,
                                       fill_value=None)(P).reshape(LA.shape)
        m = ds.months[k]
        sst = oisst.values["sst"][oisst.time.index(m)][la][:, lo]
        frames.append({"lon": oisst.lon[lo], "lat": oisst.lat[la], "sst": sst, "d20": np.clip(base + anom, 5, None), "d20_anom": anom,
                       "label": f"{['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][int(m[5:7]) - 1]} {m[:4]}",
                       "month": m})
    return frames


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def markdown_tables(res: dict[str, Any], leads: Sequence[int] = (1, 3, 6, 9, 12, 18, 24)) -> dict[str, str]:
    """Markdown tables of the study from the stored results (``res``: the study JSON)."""
    T: dict[str, str] = {}
    z = res["zoo"]
    H = z["protocol"]["horizons"]
    li = [H.index(h) for h in leads if h in H]
    sc = res["enso_scores"]
    fams = ["baseline", "data", "data+physics", "physical", "hybrid"]
    order = sorted(z["models"], key=lambda n: (fams.index(z["models"][n]["family"]), n))
    rows = ["| model | family | " + " | ".join(f"ACC h={H[i]}" for i in li) + " | RMSE skill vs persistence (h="
            + ",".join(str(H[i]) for i in li) + ") | RMSE skill vs best baseline (same h) | significant vs best "
            "baseline (Holm) at h | params | fit [s] |", "|---" * (len(li) + 7) + "|"]
    for n in order:
        m = z["models"][n]
        if "test" not in m:
            rows.append(f"| {n} | {m['family']} | failed: {m.get('error', '')[:80]} |" + " |" * (len(li) + 5))
            continue
        a = sc.get(n, {}).get("acc", [None] * len(H))
        sig = [str(h) for h, d in zip(H, m["dm_vs_best_baseline"], strict=False) if d and (d.get("p_holm") or 1) < 0.05]
        rows.append(f"| {n} | {m['family']} | " + " | ".join(_f(a[i]) for i in li) + " | "
                    + " ".join(_f(m["test"]["skill_rmse"][i]) for i in li) + " | "
                    + " ".join(_f(m["skill_vs_best_baseline"][i]) for i in li) + f" | {', '.join(sig) or 'none'} | "
                    f"{m['n_params']} | {_f(m['fit_seconds'], 1)} |")
    T["zoo"] = "\n".join(rows)

    rep = z.get("family_representative", {})
    rows = ["| family | representative (val.) | last lead with significant skill vs persistence | vs best baseline | "
            "CRPS skill vs climatology (h=" + ",".join(str(H[i]) for i in li) + ") | 95 % coverage (mean over h) |"
            " PIT deviation |", "|---" * 7 + "|"]
    for fam in fams:
        n = rep.get(fam)
        if not n:
            continue
        m = z["models"][n]
        t = m["test"]
        cov = t["coverage"].get("0.95")
        pit = (t["pit"]["pooled"] or {}).get("deviation") if t["pit"]["pooled"] else None
        rows.append(f"| {fam} | {n} | {m['skill_horizon_vs_persistence'] or 'none'} | "
                    f"{m['skill_horizon_vs_best_baseline'] or 'none'} | "
                    + " ".join(_f(t.get('crps_skill_vs_climatology', [None] * len(H))[i]) for i in li)
                    + f" | {_f(np.mean(cov)) if cov else '–'} | {_f(pit)} |")
    T["families"] = "\n".join(rows)
    T["best_baseline"] = "| lead | " + " | ".join(map(str, H)) + " |\n|---" * 1 + "|---" * len(H) + "|\n| best baseline (val.) | " + \
        " | ".join(z["best_baseline_by_val"][str(h)] for h in H) + " |"

    rows = ["| model | lead | El Nino hit rate | La Nina hit rate | neutral hit rate | El Nino false-alarm ratio | "
            "La Nina false-alarm ratio | accuracy | Heidke skill |", "|---" * 9 + "|"]
    for n in [rep.get(f) for f in fams if rep.get(f)]:
        if n not in sc:
            continue
        for h in (3, 6, 12):
            if h not in H:
                continue
            c = sc[n]["categorical"]["per_horizon"][H.index(h)]
            rows.append(f"| {n} | {h} | {_f(c['hit_rate']['el_nino'])} | {_f(c['hit_rate']['la_nina'])} | "
                        f"{_f(c['hit_rate']['neutral'])} | {_f(c['false_alarm_ratio']['el_nino'])} | "
                        f"{_f(c['false_alarm_ratio']['la_nina'])} | {_f(c['accuracy'])} | {_f(c['heidke'])} |")
    T["categorical"] = "\n".join(rows)

    if "skill_vs_climatology" in res:
        sv = res["skill_vs_climatology"]
        Hs = sv["horizons"]
        lj = [Hs.index(h) for h in leads if h in Hs]
        rows = ["| model | family | RMSE skill vs climatology (h=" + ",".join(str(Hs[i]) for i in lj) + ") | "
                "significant vs climatology (one-sided DM, Holm) at h |", "|---" * 4 + "|"]
        for n in order:
            v = sv["models"].get(n)
            if v is None:
                continue
            sig = [str(h) for h, k in zip(Hs, v.get("significant", []), strict=False) if k]
            rows.append(f"| {n} | {v['family']} | " + " ".join(_f(v["skill_rmse"][i]) for i in lj) + " | "
                        + (", ".join(sig) or "none" if v["family"] != "baseline" else "–") + " |")
        T["skill_vs_climatology"] = "\n".join(rows)

    if "spring_barrier" in res:
        rows = ["| model | origins | ACC h=6, target Jan-Mar | ACC h=6, target Jun-Aug | ACC h=6, target Sep-Nov |",
                "|---" * 5 + "|"]
        for n, v in res["spring_barrier"].items():
            A = np.asarray(v["by_target_month"]["acc"], float)
            j = v["by_target_month"]["horizons"].index(6)
            rows.append(f"| {n} | {v['first']} to {v['last']} ({v['n_origins']}) | {_f(np.nanmean(A[0:3, j]))} | "
                        f"{_f(np.nanmean(A[5:8, j]))} | {_f(np.nanmean(A[8:11, j]))} |")
        T["spring_barrier"] = "\n".join(rows)
    return T
