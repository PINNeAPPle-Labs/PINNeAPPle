"""The standard inverse-modeling workflow (:mod:`pinneapple_analysis.inverse_problems.workflow`, #1189):
identifiability before the fit (structural rank, the combinations the data cannot separate, the D-optimal next
measurements), estimation with held-out rows and intervals, and the checks that say whether the estimate can be
trusted (fit, held-out, bounds, practical identifiability, white residuals, synthetic recovery)."""
from __future__ import annotations

from ..registry import Param, Port, node

TOOL = "pinneapple_physics.reactors.kinetics:first_order_pfr_conversion"
PARAMS = {"A": {"lower": 1e4, "upper": 1e9, "initial": 1e6, "unit": "1/s"},
          "Ea": {"lower": 3e4, "upper": 1e5, "initial": 5e4, "unit": "J/mol"}}


def _synthetic(tool: str, truth: dict, sigma: float, seed: int):
    import numpy as np

    from pinneapple_autolab.study.verify import resolve_model
    fn = resolve_model(tool)
    T = np.repeat(np.linspace(320.0, 380.0, 7), 3)
    tau = np.tile([20.0, 60.0, 180.0], 7)
    y = np.array([fn(T=t, tau=u, **truth)["X"] for t, u in zip(T, tau, strict=True)])
    y = y + sigma * np.random.default_rng(seed).standard_normal(len(y))
    return {"inputs": {"T": T.tolist(), "tau": tau.tolist()}, "values": y.tolist(), "sigma": sigma}


@node("inverse.workflow", "Inverse modeling: identifiability, estimate, trust checks", category="Inverse",
      outputs=[Port("metrics", "metrics")],
      params=[Param("tool", "str", TOOL, help="forward model: library tool id with keyword inputs"),
              Param("output", "str", "X", help="the measured output of the tool"),
              Param("parameters", "dict", PARAMS, help="name -> {lower, upper, initial, unit}"),
              Param("fixed", "dict", {}, help="inputs held fixed"),
              Param("measurements_csv", "str", "", help="CSV: the condition columns, 'value' and 'sigma'"),
              Param("conditions", "list", ["T", "tau"], help="the condition columns of the CSV"),
              Param("synthetic_truth", "dict", {"A": 5e6, "Ea": 6e4},
                    help="without a CSV: synthetic measurements from these values with 1 % noise (stated)"),
              Param("candidates", "dict", {"T": [300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400],
                                           "tau": [60.0] * 11}, help="possible next measurements"),
              Param("holdout_fraction", "float", 0.2, minimum=0.0, maximum=0.5),
              Param("bootstrap", "int", 50, minimum=0), Param("n_recovery", "int", 10, minimum=0),
              Param("seed", "int", 0)],
      calls="pinneapple_analysis.inverse_problems.workflow.inverse_workflow (identifiability, calibration)")
def inverse_workflow(ctx, inputs, params):
    from pinneapple_analysis.inverse_problems.workflow import inverse_workflow as run
    if params.get("measurements_csv"):
        import csv
        with open(params["measurements_csv"], newline="") as f:
            rows = list(csv.DictReader(f))
        meas = {"inputs": {c: [float(r[c]) for r in rows] for c in params["conditions"]},
                "values": [float(r["value"]) for r in rows], "sigma": [float(r.get("sigma", 1.0)) for r in rows]}
        source = f"{len(rows)} measurements from {params['measurements_csv']}"
    else:
        meas = _synthetic(str(params["tool"]), dict(params["synthetic_truth"]), 0.01, int(params["seed"]))
        source = f"synthetic: {len(meas['values'])} points from {params['synthetic_truth']} with sigma 0.01"
    ctx.metric("data", source)
    r = run(tool=str(params["tool"]), measurements=meas, output=str(params["output"]),
            parameters=dict(params["parameters"]), fixed=dict(params["fixed"] or {}),
            candidates=dict(params["candidates"]) or None, holdout_fraction=float(params["holdout_fraction"]),
            bootstrap=int(params["bootstrap"]), n_recovery=int(params["n_recovery"]), seed=int(params["seed"]))
    for k, v in (r.get("estimate") or {}).items():
        ctx.metric(f"estimate.{k}", v, f"95 % interval {r['intervals'][k]}")
    for s in r["identifiability"]["suggestions"][:3]:
        ctx.metric(f"next_measurement.{s['label']}", s.get("gain_logdet"), "D-optimal gain in log det F")
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    bad = [c["code"] for c in r["checks"] if c["status"] == "fail"]
    ctx.metric("verdict", "fail" if bad else "pass", ", ".join(bad) or "every check passes")
    ctx.summary((", ".join(f"{k} = {v:.4g}" for k, v in r["estimate"].items()) if r.get("estimated") else
                 "not estimated: structurally non-identifiable") + ("; fail: " + ", ".join(bad) if bad else ""))
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Inverse modeling: identifiability, estimate, trust checks"
    fb = FlowBuilder(title, "Identifiability before the fit, estimation with held-out rows and intervals, and the "
                            "checks that say whether the estimate can be trusted: fit, held-out, bounds, practical "
                            "identifiability, white residuals, synthetic recovery "
                            "(pinneapple_analysis.inverse_problems.workflow).")
    fb.add("inverse", "inverse.workflow", dict(params), label="Inverse modeling")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("inverse", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="inverse.workflow", title="Inverse modeling / parameter discovery (standard workflow)", domain="inverse",
        keywords={"inverse": 2, "inverso": 2, "parameter estimation": 3, "estimar parâmetros": 3,
                  "estimação de parâmetros": 3, "calibrate": 2, "calibrar": 2, "calibração": 2,
                  "identifiability": 3, "identificabilidade": 3, "parameter discovery": 3,
                  "descoberta de parâmetros": 3, "fit the model": 2, "ajustar o modelo": 2},
        min_score=3, nodes=("inverse.workflow",), build=lambda text, det: build_flow(),
        inputs=("the forward model", "the measurements and their noise", "the parameters with their bounds"),
        uncited="the method cites Raue et al. (2009) for profile likelihood and Wald & Wolfowitz (1940) in the "
                "library modules; they are not in the flow reference file yet",
        description="pinneapple_analysis.inverse_problems.workflow.inverse_workflow"))


_register_capabilities()
