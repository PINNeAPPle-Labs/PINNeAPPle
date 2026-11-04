"""The standard digital-twin workflow (:mod:`pinneapple_systems.digital_twin.workflow`, #1193): a state-space model
and a data source (a simulated plant with a known drift, a CSV replay or a connected stream), an ensemble Kalman
filter of the states and the slow parameters, NIS / NEES consistency against their chi-square bands, anomaly flags,
CUSUM drift detection with recalibration through the inverse workflow, a forecast with intervals, what-if runs and
a dashboard page."""
from __future__ import annotations

from pathlib import Path

from ..registry import Param, Port, node

DRIFT = {"param": "UA", "at": 200, "to": 35.0, "ramp": 0}
SPIKE = {"at": 120, "size": 12.0}


@node("twin.workflow", "Digital twin: assimilation, consistency, drift -> recalibration, forecast", category="Digital twin",
      inputs=[Port("stream", "series", required=False,
                   help="readings with a 't' column and the model's inputs and outputs (source = input)")],
      outputs=[Port("metrics", "metrics")],
      params=[Param("model", "str", "thermal_plant",
                    help="a built-in twin model or 'module:function' returning a StateSpaceModel"),
              Param("model_params", "dict", {}, help="keyword arguments of the model builder"),
              Param("source", "enum", "simulated", choices=["simulated", "csv", "input"],
                    help="a simulated plant (known drift and spike: every check scored), a CSV replay, or the "
                         "connected stream"),
              Param("csv_path", "str", "", help="CSV with columns t, the inputs and the outputs (source = csv)"),
              Param("n_steps", "int", 400, minimum=50, help="simulated readings"),
              Param("dt", "float", 5.0, minimum=1e-6, help="sampling interval of the simulated plant"),
              Param("drift", "dict", DRIFT, help="simulated drift {param, at, to, ramp}"),
              Param("spike", "dict", SPIKE, help="simulated sensor spike {at, size (sigmas)}"),
              Param("plant_params", "dict", {"UA": 52.5},
                    help="the simulated plant's true parameters (the twin starts at the model's nominal values)"),
              Param("members", "int", 200, minimum=20, maximum=5000),
              Param("param_walk", "float", 2e-4, minimum=0.0, help="random-walk std of the log parameters per step"),
              Param("anomaly_z", "float", 4.0, minimum=1.0, help="normalised innovation of an anomaly"),
              Param("cusum_k", "float", 0.5, minimum=0.0), Param("cusum_h", "float", 8.0, minimum=0.5),
              Param("max_detection_delay", "int", 20, minimum=1),
              Param("recalibration_window", "int", 60, minimum=10, help="readings the inverse workflow refits"),
              Param("horizon", "int", 120, minimum=0, help="forecast steps"),
              Param("limits", "dict", {"T": 80.0}, help="output -> limit for the what-if exceedance probability"),
              Param("nees_runs", "int", 20, minimum=2, advanced=True),
              Param("seed", "int", 0)],
      calls="pinneapple_systems.digital_twin.workflow.digital_twin_workflow (etkf_analysis, "
            "inverse_problems.workflow.inverse_workflow)")
def twin_workflow(ctx, inputs, params):
    """Run the twin over the readings and write the dashboard next to the run."""
    from pinneapple_systems.digital_twin.workflow import digital_twin_workflow as run
    source = str(params["source"])
    data = None
    if source == "input":
        from ..runtime import StepFailed, load_artifact
        if inputs.get("stream") is None:
            raise StepFailed("source = input needs a connected stream")
        data = load_artifact(inputs["stream"])
        source = "arrays"
    r = run(model=str(params["model"]), model_params=dict(params.get("model_params") or {}), source=source,
            data=data, csv_path=str(params.get("csv_path") or "") or None, n_steps=int(params["n_steps"]),
            dt=float(params["dt"]), drift=dict(params.get("drift") or {}), spike=dict(params.get("spike") or {}),
            plant_params=dict(params.get("plant_params") or {}) or None, members=int(params["members"]),
            param_walk=float(params["param_walk"]), anomaly_z=float(params["anomaly_z"]),
            cusum_k=float(params["cusum_k"]), cusum_h=float(params["cusum_h"]),
            max_detection_delay=int(params["max_detection_delay"]),
            recalibration_window=int(params["recalibration_window"]), horizon=int(params["horizon"]),
            limits=dict(params.get("limits") or {}), nees_runs=int(params["nees_runs"]), seed=int(params["seed"]),
            out_dir=str(Path(ctx.workdir) / "twin"))
    ctx.metric("model", r["model"]["name"], r["model"]["description"])
    ctx.metric("readings", r["n_steps"], f"source: {r['source']}, dt = {r['dt']:g}")
    for k, v in r["final"]["theta"].items():
        ctx.metric(f"final.{k}", v, f"90 % band {r['final']['theta_90'][k]}")
    if r["nis"]["before"] is not None:
        ctx.metric("nis.before", r["nis"]["before"], "mean normalised innovation squared (1 for a consistent filter)")
    if r["nis"]["after"] is not None:
        ctx.metric("nis.after_recalibration", r["nis"]["after"], "mean NIS after the recalibration")
    ctx.metric("cusum.arl0", r["cusum"]["arl0"], f"in-control average run length (k = {r['cusum']['k']:g}, "
                                                 f"h = {r['cusum']['h']:g}; Siegmund)")
    for rc in r["recalibrations"]:
        for k, v in (rc.get("applied") or {}).items():
            ctx.metric(f"recalibrated.{k}", v, f"inverse workflow on steps {rc['window'][0]}-{rc['window'][1] - 1}; "
                                               f"95 % {rc['intervals'][k]}")
    fc = r["forecast"]
    if "coverage_90" in fc:
        ctx.metric("forecast.coverage_90", fc["coverage_90"], f"withheld readings inside the 90 % band; RMSE "
                                                              f"{fc['rmse']:.3g}")
    ctx.table("events", ["step", "t", "kind", "detail"], [[e["step"], e["t"], e["kind"], e["detail"]]
                                                          for e in r["events"]])
    ctx.table("what_if", ["scenario", "output", "peak mean", "peak 5 %", "peak 95 %", "P(exceed)"],
              [[w["name"], w["output"], w["peak_mean"], w["peak_lo"], w["peak_hi"], w.get("p_exceed", "")]
               for w in r["what_if"]])
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    if r.get("dashboard"):
        ctx.metric("dashboard", r["dashboard"], "self-contained HTML page of the twin")
    ctx.metric("verdict", r["verdict"], ", ".join(r["failed"]) or "every check passes")
    ctx.summary(r["summary"])
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Digital twin: assimilation, consistency, drift detection and recalibration, forecast"
    fb = FlowBuilder(title, "A state-space model and a data source (a simulated plant with a known parameter drift, "
                            "a CSV replay or a live stream); an ensemble Kalman filter of the states and slow "
                            "parameters; NIS and NEES against their chi-square bands; anomaly flags; CUSUM drift "
                            "detection with recalibration through the inverse workflow; a forecast with intervals, "
                            "what-if runs and a dashboard (pinneapple_systems.digital_twin.workflow).")
    fb.add("twin", "twin.workflow", dict(params), label="Digital twin")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("twin", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="twin.workflow", title="Digital twin with live data assimilation (standard workflow)",
        domain="digital_twin",
        keywords={"digital twin": 3, "gêmeo digital": 3, "gemeo digital": 3, "twin digital": 2,
                  "data assimilation": 3, "assimilação de dados": 3, "kalman": 2, "ensemble kalman": 3,
                  "drift detection": 3, "detecção de deriva": 3, "deriva": 1, "recalibration": 2,
                  "recalibração": 2, "anomaly detection": 2, "detecção de anomalias": 2, "live sensor data": 2,
                  "dados de sensores": 2, "sensor stream": 2, "monitoramento": 1, "monitoring": 1},
        # 5: "digital twin" alone stays with the planner's PDE / tank twins; with drift, assimilation,
        # recalibration, Kalman or anomaly words it is this workflow
        min_score=5, nodes=("twin.workflow",), build=lambda text, det: build_flow(),
        inputs=("the model (states, slow parameters, inputs, outputs)", "the data source and its sampling",
                "the sensor noise and the anomaly / drift thresholds"),
        uncited="the workflow cites Bishop et al. (2001), Hunt et al. (2007), Burgers et al. (1998), Bar-Shalom et al. "
                "(2001), Page (1954) and Siegmund (1985) in the library module; they are not in the flow reference "
                "file yet",
        description="pinneapple_systems.digital_twin.workflow.digital_twin_workflow"))


_register_capabilities()
