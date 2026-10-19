"""Automatic digital twin: a live-data node (simulated sensors, file, MQTT, OPC UA) and a twin node that builds the
twin from the identified equations (pinneapple_flow.twin on pinneapple_systems' TwinRuntime)."""
from __future__ import annotations

import importlib.util
import json
import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import load_artifact
from ._common import bounds_of, need_problem, save_solution, system_of


def _channels(n):
    return [f"s{i}" for i in range(n)]


def _model(problem, n, dt, estimate):
    from pinneapple_simulation.numerical_solvers.symbolic_fd import FDError

    from ..twin import PDETwinModel
    sys_ = system_of(problem)
    try:
        return PDETwinModel(sys_, bounds_of(problem), n=n, dt=dt, estimate=estimate)
    except FDError as exc:
        raise StepFailed(f"twin model: {exc}") from None


@node("data.stream", "Live data (sensors)", category="Digital twin",
      inputs=[Port("problem", "problem", required=False, help="the model of the asset (simulated source)")],
      outputs=[Port("stream", "series")],
      params=[Param("source", "enum", "simulated", choices=["simulated", "file", "mqtt", "opcua"],
                    help="simulated sensors on the identified model (a finer grid), a CSV in the data folder, an "
                         "MQTT topic (paho-mqtt) or OPC UA nodes (asyncua)"),
              Param("sensors", "list", [0.2, 0.4, 0.6, 0.8], help="sensor positions (one coordinate per axis)"),
              Param("t_end", "float", 3.0, minimum=0.0), Param("every", "int", 2, minimum=1),
              Param("dt", "float", 0.01, minimum=1e-6), Param("noise", "float", 0.002, minimum=0.0),
              Param("plant_n", "int", 81, minimum=5, maximum=2001, help="grid of the simulated asset"),
              Param("fault", "dict", {}, help="simulated change of the asset: {t: 1.5, alpha: 0.15}"),
              Param("file", "str", "", help="CSV with columns t, s0, s1, ... (data folder)"),
              Param("broker", "str", "127.0.0.1:1883"), Param("topic", "str", "pinneapple/asset/sensors"),
              Param("opcua_url", "str", "", help="opc.tcp://host:4840"),
              Param("opcua_nodes", "list", [], help="node ids, one per sensor"),
              Param("listen_s", "float", 30.0, minimum=1.0, maximum=3600.0), Param("seed", "int", 0)],
      calls="pinneapple_flow.twin.simulate_sensors | CSV | paho-mqtt | asyncua")
def stream(ctx, inputs, params):
    """Sensor readings for a twin: simulated on the model of the asset (with an optional injected change), read from
    a file, or listened to live on MQTT / OPC UA."""
    src = params["source"]
    sensors = [s if isinstance(s, list) else [s] for s in params["sensors"]]
    meta = {"sensors": sensors, "source": src}
    if src == "simulated":
        if not inputs.get("problem"):
            raise StepFailed("the simulated source needs the problem (the model of the asset)")
        from ..twin import simulate_sensors
        problem = need_problem(inputs["problem"])
        coefs = system_of(problem).coefficients
        estimate = {k: v for k, v in coefs.items() if k in params["fault"]} or dict(list(coefs.items())[:1])
        plant = _model(problem, int(params["plant_n"]), float(params["dt"]), estimate)
        cols = simulate_sensors(plant, sensors=sensors, t_end=float(params["t_end"]), every=int(params["every"]),
                                noise=float(params["noise"]), fault=params["fault"] or None, seed=int(params["seed"]))
        meta.update(noise=float(params["noise"]), fault=params["fault"], truth=[k for k in cols if
                                                                               k.startswith("truth_")])
        ctx.metric("plant_grid", int(params["plant_n"]))
    elif src == "file":
        from ..config import safe_data_path
        from .data import _read_csv
        head, arr = _read_csv(safe_data_path(params["file"]).read_text())
        if "t" not in head:
            raise StepFailed("the CSV needs a 't' column")
        cols = {h: arr[:, i] for i, h in enumerate(head)}
    elif src == "mqtt":
        if importlib.util.find_spec("paho") is None:
            raise StepFailed("MQTT needs paho-mqtt (pip install paho-mqtt)")
        cols = _listen_mqtt(ctx, params, len(sensors))
    else:
        if importlib.util.find_spec("asyncua") is None:
            raise StepFailed("OPC UA needs asyncua (pip install asyncua)")
        cols = _poll_opcua(ctx, params)
    chans = [c for c in _channels(len(sensors)) if c in cols]
    if not chans or len(cols.get("t", [])) < 2:
        raise StepFailed("no readings")
    extra = {k: v for k, v in cols.items() if k.startswith("truth_")}
    s = ctx.save_npz("stream", t=np.asarray(cols["t"], float), values=np.stack([cols[c] for c in chans], 1),
                     **{k: np.asarray(v, float) for k, v in extra.items()})
    s.update({"channels": chans, "meta": meta})
    ctx.metric("readings", int(len(cols["t"])))
    ctx.metric("sensors", len(chans))
    ctx.summary(f"{len(cols['t'])} readings from {len(chans)} sensors ({src})")
    return {"stream": s}


def _listen_mqtt(ctx, params, n):
    import paho.mqtt.client as mqtt
    host, _, port = params["broker"].partition(":")
    rows: list[dict] = []
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) if hasattr(mqtt, "CallbackAPIVersion") else mqtt.Client()

    def on_message(_c, _u, msg):
        try:
            d = json.loads(msg.payload)
            if "t" in d:
                rows.append({k: float(v) for k, v in d.items() if isinstance(v, (int, float))})
        except (ValueError, TypeError):
            pass
    client.on_message = on_message
    client.connect(host, int(port or 1883))
    client.subscribe(params["topic"])
    client.loop_start()
    end = time.time() + float(params["listen_s"])
    while time.time() < end:
        ctx.check_stop()
        time.sleep(0.2)
    client.loop_stop()
    client.disconnect()
    return {k: np.array([r.get(k, np.nan) for r in rows]) for k in ["t", *_channels(n)]}


def _poll_opcua(ctx, params):
    import asyncio

    from asyncua import Client

    async def poll():
        out = {"t": [], **{c: [] for c in _channels(len(params["opcua_nodes"]))}}
        async with Client(url=params["opcua_url"]) as client:
            nodes = [client.get_node(n) for n in params["opcua_nodes"]]
            t0 = time.time()
            while time.time() - t0 < float(params["listen_s"]):
                vals = [await nd.read_value() for nd in nodes]
                out["t"].append(time.time() - t0)
                for i, v in enumerate(vals):
                    out[f"s{i}"].append(float(v))
                await asyncio.sleep(max(float(params["dt"]) * int(params["every"]), 0.05))
        return out
    return asyncio.run(poll())


@node("twin.auto", "Digital twin (automatic)", category="Digital twin",
      inputs=[Port("problem", "problem"), Port("stream", "series")],
      outputs=[Port("twin", "twin"), Port("solution", "solution"), Port("metrics", "metrics")],
      params=[Param("estimate", "list", [], help="coefficients to calibrate live (default: the first one)"),
              Param("n", "int", 21, minimum=5, maximum=201, help="nodes of the twin's coarse grid"),
              Param("dt", "float", 0.0, minimum=0.0, help="twin step (0: the stream's sampling interval)"),
              Param("band", "float", 0.3, minimum=0.0, help="anomaly when a coefficient leaves nominal x (1 +- band)"),
              Param("n_ens", "int", 48, minimum=8, maximum=500),
              Param("sigma_obs", "float", 0.0, minimum=0.0, help="sensor noise (0: from the stream, else 1 %)"),
              Param("use_llm", "bool", True, help="let the configured LLM review the twin design (twin_draft)")],
      calls="pinneapple_flow.twin.PDETwinModel + run_twin (pinneapple_systems.operations.twin.TwinRuntime, EnKF)")
def twin_auto(ctx, inputs, params):
    """Build the twin from the identified equations: the PDE on a coarse grid as the fast model, the coefficients
    as calibrated states, the ensemble Kalman filter for the state, alerts when a coefficient leaves its band or the
    data stop agreeing with the model; dashboard and the estimated field (for the 3-D view)."""
    from ..twin import run_twin
    problem = need_problem(inputs["problem"])
    st_val = inputs["stream"]
    s = load_artifact(st_val)
    meta = (st_val or {}).get("meta") or {}
    sensors = meta.get("sensors")
    if not sensors:
        raise StepFailed("the stream has no sensor positions (use a live-data node)")
    T = np.asarray(s["t"], float)
    readings = {"t": T, **{f"s{i}": s["values"][:, i] for i in range(s["values"].shape[1])}}
    coefs = system_of(problem).coefficients
    names = params["estimate"] or list(coefs)[:1]
    missing = [k for k in names if k not in coefs]
    if missing:
        raise StepFailed(f"cannot estimate {missing}: not coefficients of the equations ({list(coefs)})")
    estimate = {k: coefs[k] for k in names}
    dt = float(params["dt"]) or float(np.median(np.diff(T)))
    model = _model(problem, int(params["n"]), dt, estimate)
    sig = float(params["sigma_obs"]) or float(meta.get("noise") or 0.0) or 0.01 * float(np.std(s["values"]))
    band = float(params["band"])
    limits = {k: (v * (1 - band), v * (1 + band)) if v > 0 else (v - band * abs(v), v + band * abs(v))
              for k, v in estimate.items()}
    ctx.decide("twin.design", "twin = identified PDE + live data + EnKF + calibration + alerts")
    ctx.decide("twin.coarse_model", f"identified PDE on a {model.grid_shape} grid, step {dt:g}", value=dt)
    ctx.decide("twin.enkf", f"ensemble Kalman filter, {params['n_ens']} members", value=params["n_ens"])
    ctx.decide("twin.parameter_state", f"calibrate {names} as random-walk states", value=names)
    ctx.auto("twin.alert_thresholds", f"anomaly band nominal x (1 +- {band:g}); chi-square 0.999 consistency",
             "band", value=limits)
    _llm_review(ctx, params, problem)
    t0 = time.perf_counter()
    rt, res = run_twin(model, readings, sensors=sensors, sigma_obs=sig, limits=limits, n_ens=int(params["n_ens"]),
                       progress=lambda i, n: ctx.check_stop())
    secs = time.perf_counter() - t0
    h = res["history"]
    ctx.metric("readings", len(T))
    ctx.metric("assimilation_seconds", round(secs, 2), "wall clock on this machine")
    for k in names:
        ctx.metric(f"{k}_final", h[k][-1], "ensemble mean at the last reading")
        ctx.metric(f"{k}_final_sd", h[f"{k}_sd"][-1])
        if f"truth_{k}" in s:
            ctx.metric(f"{k}_true_final", float(np.asarray(s[f"truth_{k}"])[-1]), "the simulated asset")
    alerts = [a for a in res["alerts"] if not a.get("cleared")]
    ctx.metric("alerts", len(alerts))
    fault = meta.get("fault") or {}
    first = next((a for a in alerts if a["kind"] in ("parameter", "consistency")), None)
    if first is not None and fault.get("t") is not None:
        ctx.metric("detection_delay", float(first["t"]) - float(fault["t"]), "first alert after the injected change")
    elif fault and first is None:
        ctx.warn("the injected change raised no alert")
    pred = np.array([[p if p is not None else np.nan for p in h[f"pred{i}"]] for i in range(len(sensors))])
    obs = np.array([h[f"obs{i}"] for i in range(len(sensors))])
    rmse = float(np.sqrt(np.nanmean((pred - obs) ** 2)))
    ctx.metric("one_step_rmse", rmse, "predicted (before the update) vs observed, all sensors")
    if alerts:
        ctx.table("alerts", ["t", "kind", "severity", "message"],
                  [[a["t"], a["kind"], a["severity"], a["message"]] for a in alerts[:30]])
    _dashboard(ctx, h, s, names, sensors, alerts, fault)
    F = res["fields"].reshape(len(T), *model.grid_shape)
    coords = list(model.axes)
    axes = {**model.axes, "t": T}
    vals = np.moveaxis(F, 0, -1)[..., None] if len(coords) == 1 else np.moveaxis(F, 0, -1)[..., None]
    sol = save_solution(ctx, "twin_field", axes, vals, coords + ["t"], model.fields[:1], "twin", "twin estimate")
    tw = ctx.save_json("twin", {"model": model.to_dict(), "final": res["final"], "alerts": res["alerts"],
                                "snapshot": res["snapshot"], "sensors": sensors, "obs_names": res["obs_names"]})
    ctx.summary(f"{len(T)} readings, {', '.join(f'{k} {h[k][-1]:.4g}' for k in names)}, {len(alerts)} alert(s)")
    return {"twin": tw, "solution": sol, "metrics": ctx.report["metrics"]}


def _llm_review(ctx, params, problem):
    if not params["use_llm"]:
        return
    from ..config import Settings
    s = Settings.from_env()
    if s.llm_provider in ("", "none", "openai_compatible"):
        return
    try:
        from pinneapple_llm.twin_draft import draft_digital_twin
        d = draft_digital_twin("digital twin of " + "; ".join(problem["equations"]["pdes"]), provider=s.llm_provider,
                               model=s.llm_model or None, api_key=s.llm_api_key or None)
        from ..decisions import llm_decision
        ctx.report["decisions"].append(llm_decision(f"{s.llm_provider}/{s.llm_model or 'default'}", topic="twin",
                                                    what=f"twin_draft proposal: {d.config_kwargs}", node=ctx.node_id,
                                                    rationale=d.reasoning).to_dict())
        ctx.log("twin_draft proposal recorded (DigitalTwinConfig fields; not applied to this EnKF twin)")
    except Exception as exc:  # noqa: BLE001 - optional review
        ctx.warn(f"LLM twin review skipped: {type(exc).__name__}: {exc}")


def _dashboard(ctx, h, s, names, sensors, alerts, fault):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.asarray(h["t"])
    fig, axs = plt.subplots(2 + len(names), 1, figsize=(7.5, 2.1 * (2 + len(names))), sharex=True,
                            constrained_layout=True)
    for i in range(len(sensors)):
        axs[0].plot(t, h[f"obs{i}"], ".", ms=2, alpha=0.5)
        axs[0].plot(t, [p if p is not None else np.nan for p in h[f"pred{i}"]], "-", lw=1,
                    label=f"sensor {i} @ {sensors[i]}")
    axs[0].set_ylabel("sensors")
    axs[0].legend(fontsize=7, ncol=2)
    for j, k in enumerate(names):
        ax = axs[1 + j]
        m, sd = np.asarray(h[k]), np.asarray(h[f"{k}_sd"])
        ax.plot(t, m, label=f"{k} (twin)")
        ax.fill_between(t, m - 2 * sd, m + 2 * sd, alpha=0.25)
        if f"truth_{k}" in s:
            ax.plot(t, s[f"truth_{k}"][:len(t)], "k--", lw=1, label="asset (truth)")
        ax.set_ylabel(k)
        ax.legend(fontsize=7)
    axs[-1].semilogy(t, [v if v else np.nan for v in h["nis"]], lw=0.8)
    axs[-1].set_ylabel("NIS")
    axs[-1].set_xlabel("t")
    for a in alerts[:5]:
        for ax in axs:
            ax.axvline(float(a["t"]), color="r", lw=0.7)
    if fault.get("t") is not None:
        for ax in axs:
            ax.axvline(float(fault["t"]), color="k", lw=0.7, ls=":")
    ctx.figure(fig, "twin_dashboard", "sensors vs twin, calibrated coefficients (+-2 sd), consistency; red: alerts, "
                                      "dotted: injected change")
