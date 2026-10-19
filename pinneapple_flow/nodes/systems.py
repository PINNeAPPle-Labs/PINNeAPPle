"""System nodes: digital twin on a stream, forecasting a near-random series, aircraft take-off (flight library)."""
from __future__ import annotations

import importlib.util
import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import load_artifact


def _mqtt_ok():
    return importlib.util.find_spec("paho") is not None


@node("twin.connect", "Digital twin (stream -> twin runtime)", category="Digital twin",
      inputs=[Port("stream", "series", required=False, help="recorded readings (t, h, T, q_in, Q, T_in, T_amb)")],
      outputs=[Port("twin", "twin"), Port("series", "series"), Port("metrics", "metrics")],
      params=[Param("source", "enum", "simulated", choices=["simulated", "input", "mqtt"],
                    help="simulated plant stream, the connected stream, or live MQTT (needs paho-mqtt)"),
              Param("duration_s", "float", 7200.0, minimum=60.0, maximum=7 * 86400.0,
                    help="simulated plant time"),
              Param("fault", "enum", "leak", choices=["none", "leak", "heater", "sensor_bias", "sensor_drift",
                                                      "stuck_sensor", "valve"]),
              Param("fault_magnitude", "float", 5e-4, help="leak: m^3/s; heater: fraction; sensor: units"),
              Param("fault_start_frac", "float", 0.5, minimum=0.0, maximum=1.0),
              Param("broker", "str", "127.0.0.1:1883", help="MQTT host:port (source = mqtt)"),
              Param("topic", "str", "pinneapple/tank-1/sensors"),
              Param("listen_s", "float", 30.0, minimum=1.0, maximum=3600.0),
              Param("seed", "int", 4, minimum=0)],
      calls="pinneapple_systems.operations.live.replay / LiveTwin + TwinRuntime (ensemble Kalman filter)")
def twin(ctx, inputs, params):
    """Feed a stream of sensor readings into the twin runtime (an ensemble Kalman filter on the heated-tank model):
    state estimates with uncertainty, hidden states (leak flow, heater efficiency), alerts, a forecast."""
    from pinneapple_systems.operations.live import LiveTwin, demo_record, make_twin, replay
    from pinneapple_systems.operations.plant import Fault
    from pinneapple_systems.operations.twin import LumpedTankModel
    model = LumpedTankModel(dt=10.0)
    src = params["source"]
    ctx.decide("twin.enkf", "ensemble Kalman filter (48 members) on the heated-tank model")
    ctx.decide("twin.parameter_state", "leak flow and heater efficiency estimated as states")
    ctx.decide("twin.alert_thresholds", "AlertRules defaults")
    t0 = time.perf_counter()
    truth = None
    if src == "simulated":
        dur = float(params["duration_s"])
        faults = [] if params["fault"] == "none" else [
            Fault(params["fault"], dur * float(params["fault_start_frac"]), dur + 10.0,
                  float(params["fault_magnitude"]), target="T" if params["fault"].startswith(("sensor", "stuck"))
                  else "")]
        table = demo_record(dur, 10.0, seed=int(params["seed"]), faults=faults)
        truth = {k: table[k] for k in ("t", "q_leak") if k in table}
        tw, res = replay(table, model)
    elif src == "input":
        if not inputs.get("stream"):
            raise StepFailed("source 'input' needs a connected stream")
        s = load_artifact(inputs["stream"])
        names = inputs["stream"].get("channels") or []
        table = {"t": s["t"], **{n: s["values"][:, i] for i, n in enumerate(names)}}
        tw, res = replay(table, model)
    else:
        if not _mqtt_ok():
            raise StepFailed("live MQTT needs paho-mqtt (pip install paho-mqtt)")
        host, _, port = params["broker"].partition(":")
        live = LiveTwin(make_twin(model), source="mqtt", topic=params["topic"], broker_host=host,
                        broker_port=int(port or 1883), embedded_broker=False)
        live.start()
        end = time.time() + float(params["listen_s"])
        while time.time() < end:
            ctx.check_stop()
            time.sleep(0.5)
        live.stop()
        tw = live.twin
        res = {"n": len(tw.history), "history": None, "alerts": [a.to_dict() for a in tw.alerts],
               "snapshot": tw.snapshot()}
    secs = time.perf_counter() - t0
    hist = res.get("history") or {}
    ctx.metric("readings", res.get("n"))
    ctx.metric("alerts", len(res.get("alerts") or []))
    ctx.metric("assimilation_seconds", round(secs, 2))
    alerts = res.get("alerts") or []
    if alerts:
        ctx.table("alerts", ["t", "kind", "message"],
                  [[a.get("t"), a.get("kind", a.get("rule", "")), a.get("message", a.get("text", ""))]
                   for a in alerts[:30]])
    if hist and "q_leak" in hist:
        est = np.asarray(hist["q_leak"], float)
        ctx.metric("q_leak_final_estimate", float(est[-1]), "m^3/s, ensemble mean at the last reading")
        if src == "simulated" and params["fault"] == "leak":
            ctx.metric("q_leak_true", float(params["fault_magnitude"]), "the leak injected in the simulated plant")
        first_alert = next((a.get("t") for a in alerts), None)
        if first_alert is not None and params["fault"] != "none" and src == "simulated":
            ctx.metric("detection_delay_s", float(first_alert) - float(params["duration_s"])
                       * float(params["fault_start_frac"]), "first alert after the fault started")
    if hist:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.asarray(hist["t"], float) / 60.0
        keys = [k for k in ("h", "T", "q_leak", "eta") if k in hist]
        fig, axs = plt.subplots(len(keys), 1, figsize=(7, 1.9 * len(keys)), sharex=True)
        for ax, k in zip(np.atleast_1d(axs), keys, strict=True):
            m, s = np.asarray(hist[k], float), np.asarray(hist.get(f"{k}_sd", np.zeros(len(t))), float)
            ax.plot(t, m, label=f"{k} (twin)")
            ax.fill_between(t, m - 2 * s, m + 2 * s, alpha=0.25)
            if f"obs_{k}" in hist:
                ax.plot(t, np.asarray(hist[f"obs_{k}"], float), ".", ms=1.5, alpha=0.4, label="sensor")
            if truth is not None and k in truth:
                ax.plot(np.asarray(truth["t"]) / 60, truth[k], "k--", lw=1, label="plant (truth)")
            ax.set_ylabel(k)
            ax.legend(fontsize=7, loc="upper left")
        for a in alerts[:1]:
            for ax in np.atleast_1d(axs):
                ax.axvline(float(a.get("t", 0)) / 60, color="r", lw=0.8)
        np.atleast_1d(axs)[-1].set_xlabel("time [min]")
        ctx.figure(fig, "twin", "twin estimates (+-2 sd), sensors and the first alert")
    s = ctx.save_npz("series", t=np.asarray(hist.get("t", []), float),
                     values=np.stack([np.asarray(hist[k], float) for k in ("h", "T") if k in hist], 1)
                     if hist else np.zeros((0, 2)))
    s.update({"channels": ["h", "T"]})
    snap = ctx.save_json("twin", {"snapshot": res.get("snapshot"), "alerts": alerts,
                                  "forecast": res.get("forecast"), "revalidation": res.get("revalidation")})
    ctx.summary(f"{res.get('n')} readings assimilated, {len(alerts)} alert(s)")
    return {"twin": snap, "series": s, "metrics": ctx.report["metrics"]}


ENSO_DEFAULT = ["persistence", "climatology", "ar", "ro", "delayed_oscillator", "lim", "gbm"]   # im_nudged: slow


@node("forecast.enso", "El Nino forecast (ENSO models)", category="Time series",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure", required=False)],
      params=[Param("models", "list", list(ENSO_DEFAULT),
                    help="names from pinneapple_systems.time_series.enso.study.zoo(): baselines, oscillators "
                         "(ro, delayed_oscillator), the intermediate coupled model (im_nudged, im_enkf), LIM, CNN, "
                         "PINNs, hybrids"),
              Param("quick", "bool", True, help="the study's quick settings (small epochs / members, budget 1)"),
              Param("test_start", "str", "2012-01", help="first month of the untouched test period"),
              Param("leads", "list", [3, 6, 9, 12], help="leads (months) shown in the summary table")],
      calls="pinneapple_systems.time_series.enso: dataset.build_dataset + to_series, study.protocol + run_model, "
            "pipeline.skill_against + enso_scores; forecast_lab.study.merge_zoo")
def enso(ctx, inputs, params):
    """Forecast the Nino 3.4 index with the ENSO library's models under its protocol (tuned on the development
    period, scored on the untouched test period): RMSE skill against climatology and persistence with Holm-adjusted
    Diebold-Mariano tests, and the anomaly correlation per lead. The observations (OISST, WOD thermocline, ERA5
    wind stress) are downloaded once into the PINNeAPPle data cache."""
    from pinneapple_systems.time_series.enso import dataset as DS
    from pinneapple_systems.time_series.enso import pipeline, study
    from pinneapple_systems.time_series.forecast_lab.study import merge_zoo
    known = {s.name for s in study.zoo(True)}
    bad = [m for m in params["models"] if m not in known]
    if bad:
        raise StepFailed(f"unknown ENSO models {bad}; known: {sorted(known)}")
    names = list(dict.fromkeys(["persistence", "climatology", *params["models"]]))
    ctx.decide("forecast.baselines", f"untouched test period from {params['test_start']}; skill against "
                                    "climatology and persistence", value=params["test_start"])
    try:
        ds = DS.build_dataset(log=lambda *a: ctx.log(" ".join(map(str, a))))
    except Exception as exc:  # noqa: BLE001 - no network and no cache: say so, never substitute data
        raise StepFailed(f"ENSO observations unavailable ({type(exc).__name__}: {exc}); they are downloaded once "
                         "from the NOAA / ERA5 buckets into the data cache") from None
    data = DS.to_series(ds)
    prot = study.protocol(ds.months, test_start=params["test_start"])
    parts = []
    t0 = time.perf_counter()
    for n in names:
        ctx.check_stop()
        ctx.log(f"model {n}")
        parts.append(study.run_model(n, data, prot, quick=bool(params["quick"]),
                                     budget=1 if params["quick"] else study.FP.BUDGET, log=ctx.log))
    merged = merge_zoo(parts)
    H = list(merged["protocol"]["horizons"])
    vs_clim = pipeline.skill_against(merged, "climatology")
    scores = pipeline.enso_scores(merged, data.meta["month_of_year"])
    leads = [h for h in params["leads"] if h in H]
    rows = []
    for n, m in merged["models"].items():
        if "test" not in m:
            continue
        sc = vs_clim["models"].get(n, {})
        sk_c = sc.get("skill_rmse") or [None] * len(H)
        sig = sc.get("significant") or [None] * len(H)
        sk_p = m["test"].get("skill_rmse") or [None] * len(H)
        acc = (scores.get(n) or {}).get("acc") or [None] * len(H)
        for h in leads:
            i = H.index(h)
            rows.append([n, m["family"], h, sk_c[i], sig[i], sk_p[i] if isinstance(sk_p, list) else sk_p, acc[i]])
    ctx.table("test period: RMSE skill vs climatology / persistence, anomaly correlation",
              ["model", "family", "lead [months]", "skill vs climatology", "significant (Holm)",
               "skill vs persistence", "ACC"], rows)
    for h in leads:
        cand = [(r[3], r[0]) for r in rows if r[2] == h and isinstance(r[3], (int, float))
                and r[1] != "baseline"]
        if cand:
            v, n = max(cand)
            ctx.metric(f"best_skill_vs_climatology_lead{h}", f"{n} ({v:.2f})",
                       "1 - RMSE / RMSE(climatology) on the test period")
    ctx.metric("test_period", f"{params['test_start']} to {ds.months[-1]}")
    ctx.metric("study_seconds", round(time.perf_counter() - t0, 1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for n in names:
        sc = vs_clim["models"].get(n)
        if sc and n != "climatology":
            ax[0].plot(H, sc["skill_rmse"], "o-", ms=3, label=n)
        if n in scores:
            ax[1].plot(H, scores[n]["acc"], "o-", ms=3, label=n)
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set_xlabel("lead [months]")
    ax[0].set_ylabel("RMSE skill vs climatology")
    ax[1].set_xlabel("lead [months]")
    ax[1].set_ylabel("anomaly correlation")
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fa = ctx.figure(fig, "skill", f"Nino 3.4 forecasts, test {params['test_start']} to {ds.months[-1]}")
    best = ctx.report["metrics"].get(f"best_skill_vs_climatology_lead{leads[1] if len(leads) > 1 else leads[0]}")
    best = best.get("value") if isinstance(best, dict) else best
    ctx.summary(f"{len(names)} models on Nino 3.4 (test from {params['test_start']}); best non-baseline skill vs "
                f"climatology at {leads[1] if len(leads) > 1 else leads[0]} months: {best}")
    return {"metrics": ctx.report["metrics"], "figure": fa}


@node("forecast.lab", "Forecast lab (near-random series)", category="Time series",
      inputs=[Port("series", "series", required=False)],
      outputs=[Port("metrics", "metrics")],
      params=[Param("source", "enum", "elnino", choices=["elnino", "input", "white_noise", "random_walk",
                                                         "ornstein_uhlenbeck", "lorenz63_noise0.1"],
                    help="elnino: monthly SST anomalies 1950-2010 (statsmodels); synthetic cases of forecast_lab"),
              Param("horizons", "list", [1, 3, 6, 12]),
              Param("models", "list", ["persistence", "climatology", "ar", "exp_smoothing", "seasonal_naive", "gbm"]),
              Param("budget", "int", 2, minimum=1, maximum=50, help="configurations tuned per model"),
              Param("n_surrogates", "int", 39, minimum=19, maximum=999),
              Param("n", "int", 1500, minimum=200, maximum=20000, help="length of synthetic cases")],
      calls="pinneapple_systems.time_series.predictability.predictability_report + forecast_lab.study.run_zoo")
def forecast(ctx, inputs, params):
    """Predictability first (surrogate tests, entropy), then a zoo of forecasters on purged walk-forward folds with
    an untouched test period; skill against persistence per horizon."""
    from pinneapple_systems.time_series.forecast_lab import presets
    from pinneapple_systems.time_series.forecast_lab.base import SeriesData
    from pinneapple_systems.time_series.forecast_lab.study import Protocol, run_zoo
    from pinneapple_systems.time_series.predictability import predictability_report
    H = [int(h) for h in params["horizons"]]
    src = params["source"]
    if src == "elnino":
        from pinneapple_data.datasets.real_world import _load_statsmodels_elnino
        d = _load_statsmodels_elnino()
        if d.get("source") != "statsmodels.datasets.elnino":
            raise StepFailed("the El Nino data needs statsmodels (pip install statsmodels); no placeholder is used")
        y = np.asarray(d["signal"], float)
        data = SeriesData(y[:, None], 0, 1.0, "elnino_sst", meta={"source": d["source"]})
        min_hist = 36
    elif src == "input":
        if not inputs.get("series"):
            raise StepFailed("source 'input' needs a connected series")
        s = load_artifact(inputs["series"])
        y = np.asarray(s["values"], float)[:, 0]
        data = SeriesData(y[:, None], 0, 1.0, "input")
        min_hist = 32
    else:
        c = presets.case(src, n=int(params["n"]))
        data = c.data
        y = data.y
        min_hist = 64
    ctx.metric("length", int(len(y)))
    ctx.decide("forecast.predictability_first", "predictability report (surrogate tests) before the models")
    ctx.decide("forecast.baselines", f"baselines and learned models, untouched test period, horizons {H}", H)
    t0 = time.perf_counter()
    rep = predictability_report(y, horizons=tuple(H), n_surrogates=int(params["n_surrogates"]))
    for k in ("tested_on", "ar1_coefficient", "permutation_entropy", "sample_entropy"):
        if isinstance(rep.get(k), (str, int, float, np.floating)):
            ctx.metric(f"predictability_{k}", float(rep[k]) if not isinstance(rep[k], str) else rep[k])
    tests = rep.get("surrogate_tests") or {}
    trows = []
    for name, t in tests.items():
        if isinstance(t, dict):
            trows.append([name, t.get("statistic"), t.get("observed"), t.get("p_value", t.get("p"))])
    if trows:
        ctx.table("surrogate tests (is the series more predictable than its surrogates?)",
                  ["test", "statistic", "observed", "p-value"], trows)
    specs = [s for s in presets.baseline_specs(arima=False) + presets.data_specs(lags=(12, 24), seq_lags=24,
                                                                                  max_train=None)
             if s.name in set(params["models"])]
    if not specs:
        raise StepFailed(f"no known model in {params['models']}")
    if "persistence" not in {s.name for s in specs}:
        specs = [s for s in presets.baseline_specs(arima=False) if s.name == "persistence"] + specs
    proto = Protocol(H, min_history=min_hist, test_stride=1, val_frac=0.2, fold_frac=0.2, n_folds=2)
    res = run_zoo(data, specs, proto, budget=int(params["budget"]), log=ctx.log)
    secs = time.perf_counter() - t0
    ctx.metric("study_seconds", round(secs, 1))
    models = res.get("models") or {}
    pers = np.asarray((models.get("persistence", {}).get("test") or {}).get("rmse") or [np.nan] * len(H), float)
    rows, out = [], {}
    for name, m in models.items():
        rmse = np.asarray((m.get("test") or {}).get("rmse") or [np.nan] * len(H), float)
        skill = 1.0 - rmse / pers
        rows.append([name, m.get("family"), _fmt(rmse), _fmt(skill)])
        out[name] = {"rmse": _fmt(rmse), "skill_vs_persistence": _fmt(skill)}
    ctx.table(f"untouched test period: RMSE and skill (1 - RMSE / RMSE persistence) at horizons {H}",
              ["model", "family", "RMSE per horizon", "skill per horizon"], rows)
    best = min(out, key=lambda k: np.nanmean(np.asarray(out[k]["rmse"], float))) if out else None
    if best:
        ctx.metric("best_model_by_mean_test_rmse", best)
        ctx.metric("best_model_skill_per_horizon", out[best]["skill_vs_persistence"])
    ctx.summary(f"{len(rows)} models on {data.name} ({len(y)} points); best mean test RMSE: {best}")
    return {"metrics": {"models": out, "series": data.name,
                        "predictability": {k: v for k, v in rep.items() if isinstance(v, (str, bool, int, float))}}}


def _fmt(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [round(float(x), 4) if x is not None and np.isfinite(x) else None for x in v]
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    return v


def _flight_available():
    if importlib.util.find_spec("pinneapple_simulation.flight") is None:
        return False, "needs pinneapple_simulation.flight (the flight-dynamics library; an older checkout lacks it)"
    return True, ""


def _render_takeoff(ctx, ac, hist, m, params) -> None:
    """Blender (Cycles, CPU) still at the 35 ft screen and a short clip of the take-off scene; skipped with a
    warning when bpy is not installed."""
    from pinneapple_tools.visualization.studio import blender as studio_blender
    try:
        import bpy
        studio_blender._check_bpy(bpy)
    except Exception as exc:  # noqa: BLE001 - bpy missing or too old: the take-off still runs
        ctx.warn(f"no Blender render: {exc}")
        return
    import time

    from pinneapple_simulation.flight.visuals import flight_scene
    n = int(params["frames"])
    sc = flight_scene(ac, hist, n_frames=n, title=f"{ac.info.get('type', params['aircraft'])}: take-off")
    t = np.asarray(hist.t, float)
    t35 = m.get("t_35ft") if isinstance(m.get("t_35ft"), (int, float)) else t[-1] / 2
    k = int(np.clip(round((t35 - t[0]) / max(t[-1] - t[0], 1e-9) * (n - 1)), 0, n - 1))
    out = ctx.path("render")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    still = studio_blender.render_still(sc, str(out / "takeoff_still.png"), k, size=(960, 540),
                                        samples=int(params["samples"]) * 2, camera=params["camera"])
    clip = studio_blender.render_animation(sc, str(out), size=(480, 270), samples=int(params["samples"]), fps=12,
                                           mp4=True, camera=params["camera"], hud_keys=["V", "h", "nz"],
                                           name="takeoff_clip")
    ctx.metric("render_seconds", round(time.perf_counter() - t0, 1), "Cycles on this machine's CPU, still + clip")
    ctx.report["figures"].append({"path": str(still), "name": "takeoff_render",
                                  "caption": f"Blender Cycles, frame {k} of {n} (the 35 ft screen)"})
    if clip.get("mp4"):
        ctx.view("video", {"path": str(clip["mp4"]), "poster": str(still),
                           "caption": f"{n} frames, 480x270, {params['samples']} samples, camera {params['camera']}"})
    for f in clip.get("frames") or []:                   # the mp4 holds them
        try:
            __import__("os").remove(f)
        except OSError:
            pass


@node("flight.takeoff", "Aircraft take-off (6-DOF)", category="Systems",
      outputs=[Port("metrics", "metrics"), Port("series", "series"), Port("figure", "figure", required=False)],
      params=[Param("aircraft", "str", "light_ga", help="name in pinneapple_simulation.flight.catalog"),
              Param("kind", "enum", "light", choices=["light", "transport"]),
              Param("render", "enum", "none", choices=["none", "blender"],
                    help="blender: a Cycles still at the 35 ft screen and a short clip (needs bpy >= 4.2)"),
              Param("camera", "enum", "side", choices=["side", "chase_low", "chase", "tower", "follow"]),
              Param("frames", "int", 24, minimum=4, maximum=480, help="clip frames"),
              Param("samples", "int", 8, minimum=1, maximum=512, help="Cycles samples per pixel")],
      calls="pinneapple_simulation.flight.takeoff.simulate_takeoff + takeoff_metrics + raymer_variants; "
            "pinneapple_simulation.flight.visuals.flight_scene + pinneapple_tools.visualization.studio.blender."
            "render_animation / render_still",
      available=_flight_available)
def takeoff(ctx, inputs, params):
    """Simulate a take-off with the 6-DOF flight library (vortex lattice in the loop) and compare the distance
    with Raymer's closed form."""
    from pinneapple_simulation.flight import catalog
    from pinneapple_simulation.flight.takeoff import (
        raymer_variants,
        simulate_takeoff,
        takeoff_metrics,
    )
    ac = catalog.build(params["aircraft"])
    ctx.decide("flight.raymer", "take-off distance compared with Raymer's closed form")
    hist = simulate_takeoff(ac, kind=params["kind"], record_gamma=params["render"] == "blender")
    m = takeoff_metrics(ac, hist)
    rv = raymer_variants(ac, m, params["kind"])        # the same two closed forms the flight guide quotes
    for k, v in m.items():
        if isinstance(v, (int, float)):
            ctx.metric(k, v)
    labels = {"raymer_1p1Vs": "Raymer sec. 17.8, V_LOF = 1.1 V_stall, rotation 1 s (light) / 3 s (transport)",
              "raymer_sim_VLOF": "Raymer sec. 17.8 with the simulated V_LOF and rotation time"}
    for name, v in rv.items():
        ctx.metric(f"{name}_total_35ft_m", v["total_35ft_m"], labels[name])
        ctx.metric(f"sim_over_{name}", v["sim_over_raymer"], "simulated distance to 35 ft / closed form")
    x = np.asarray(hist.x)
    s = ctx.save_npz("trajectory", t=np.asarray(hist.t), values=x[:, :3])
    s.update({"channels": ["north", "east", "down"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(x[:, 0] - x[0, 0], -x[:, 2] + x[0, 2], label="6-DOF simulation")
    ax.axhline(10.668, color="k", lw=0.8, ls="--", label="35 ft screen")
    for (name, v), c in zip(rv.items(), ("r", "m"), strict=True):
        ax.axvline(v["total_35ft_m"], color=c, lw=0.8, ls=":", label=labels[name].split(", ", 1)[-1][:40])
    ax.set_xlabel("distance along the runway [m]")
    ax.set_ylabel("height [m]")
    ax.legend(fontsize=8)
    fig_a = ctx.figure(fig, "takeoff", "take-off trajectory")
    if params["render"] == "blender":
        _render_takeoff(ctx, ac, hist, m, params)
    a, b = rv["raymer_1p1Vs"], rv["raymer_sim_VLOF"]
    ctx.summary(f"ground roll {m['ground_roll_m']:.0f} m, 35 ft at {m['distance_35ft_m']:.0f} m; Raymer 17.8: "
                f"{a['total_35ft_m']:.0f} m with V_LOF = 1.1 V_s ({100 * (a['sim_over_raymer'] - 1):+.0f}%), "
                f"{b['total_35ft_m']:.0f} m with the simulated V_LOF and rotation ({100 * (b['sim_over_raymer'] - 1):+.0f}%)")
    return {"metrics": ctx.report["metrics"], "series": s, "figure": fig_a}
