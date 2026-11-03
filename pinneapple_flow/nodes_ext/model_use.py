"""Use a trained model instead of training (:mod:`pinneapple_analysis.surrogate.model_source`, #1190): a zoo model,
a model exported by the surrogate workflow, a TorchScript or ONNX file with its declared card, or a library tool as a
callable; the applicability gate (validity range, units, reference solves or the zoo's stored reference) before
use; forward predictions or an inverse estimate through the model, re-verified by the reference solver."""
from __future__ import annotations

from ..registry import Param, Port, node

REF = "pinneapple_design.thermal.heatsink.surrogate:forced_reference"
KNOWN = {"base_width_mm": 100.0, "base_depth_mm": 100.0, "base_thickness_mm": 6.0, "n_fins": 16.0,
         "fin_thickness_mm": 1.5, "fin_height_mm": 40.0, "k_w_mk": 200.0, "source_width_mm": 50.0,
         "source_depth_mm": 50.0, "t_ambient_c": 25.0, "power_w": 150.0}


@node("model.use", "Use a trained model (zoo / yours), gated and re-verified", category="Surrogates",
      outputs=[Port("metrics", "metrics")],
      params=[Param("source", "enum", "zoo", choices=["zoo", "workflow", "torchscript", "onnx"],
                    help="zoo model, a surrogate-workflow export, or a file with its declared card"),
              Param("name", "str", "heatsink_forced_mlp", help="zoo model name"),
              Param("path", "str", "", help="workflow export (surrogate.pkl) or model file"),
              Param("card", "dict", {}, help="file models: {inputs, outputs, bounds, units, claimed_rel_error}"),
              Param("mode", "enum", "inverse", choices=["forward", "inverse"]),
              Param("query", "dict", {}, help="forward: input -> list of values"),
              Param("known", "dict", KNOWN, help="inverse: the known inputs"),
              Param("unknown", "list", ["air_velocity_m_s"], help="inverse: the inputs to estimate"),
              Param("observed", "dict", {"r_base_k_w": 0.36}, help="inverse: measured outputs (default: the solver's "
                                                                       "value at 2.5 m/s, 0.3600 K/W)"),
              Param("reference", "str", REF, help="library tool id of the solver that checks the model ('' : none)")],
      calls="pinneapple_analysis.surrogate.model_source.open_model / gate / use_model")
def model_use(ctx, inputs, params):
    """Open the model, gate it, use it, re-verify with the reference."""
    from pinneapple_analysis.surrogate.model_source import gate, open_model, use_model
    card = dict(params.get("card") or {})
    p = open_model(str(params["source"]), name=params.get("name") or None, path=params.get("path") or None,
                   inputs=card.get("inputs", ()), outputs=card.get("outputs", ()), bounds=card.get("bounds"),
                   units=card.get("units"), claimed_rel_error=card.get("claimed_rel_error"))
    ref = None
    if params.get("reference"):
        from pinneapple_autolab.study.verify import resolve_model
        ref = resolve_model(str(params["reference"]))
    if params["mode"] == "forward":
        r = use_model(p, mode="forward", query=dict(params["query"]), reference=ref)
    else:
        known = dict(params["known"])
        g = gate(p, {**{k: [v] for k, v in known.items()},
                     **{k: [0.5 * (p.bounds[k][0] + p.bounds[k][1])] for k in params["unknown"]}}, reference=ref,
                 n_reference=1)
        r = use_model(p, mode="inverse", observed=dict(params["observed"]), unknown=list(params["unknown"]),
                      known=known, reference=ref)
        r["gate_before"] = g
    gates = (r.get("gate_before") or {}).get("checks", []) + r["gate"]["checks"] + r.get("checks", [])
    ctx.table("gate", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in gates])
    ctx.metric("model", f"{p.source}:{p.name}", f"{len(p.inputs)} inputs, outputs {p.outputs}")
    if r.get("estimate"):
        for k, v in r["estimate"].items():
            ctx.metric(f"estimate.{k}", v, "bounded least squares through the model")
        ctx.metric("misfit_rel", r["misfit_rel"], "relative misfit of the model at the estimate")
        for o, v in (r.get("reverified") or {}).items():
            ctx.metric(f"reverified.{o}.rel_diff", v["rel_diff"], "reference solver at the estimate vs observed")
    for o, v in (r.get("predictions") or {}).items():
        ctx.metric(f"prediction.{o}", v)
    bad = [c["code"] for c in gates if c["status"] == "fail"]
    ctx.metric("verdict", "fail" if bad or not r["used"] else "pass", ", ".join(bad) or "gate passed")
    ctx.summary(f"{p.source}:{p.name} {params['mode']}: " + ("refused by the gate: " + ", ".join(bad) if bad else
                                                            "used, gate passed"))
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Use a trained model (zoo or yours) instead of training, gated and re-verified"
    fb = FlowBuilder(title, "A zoo model, a surrogate-workflow export or a model file with its card; the "
                            "applicability gate (validity range, units, reference solves) before use; forward or "
                            "inverse through the model; the result re-verified by the reference solver "
                            "(pinneapple_analysis.surrogate.model_source).")
    fb.add("model", "model.use", dict(params), label="Trained model")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("model", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="model.use", title="Use a trained model instead of training", domain="surrogates",
        keywords={"pretrained": 3, "pré-treinado": 3, "pre-trained": 3, "trained model": 3, "modelo treinado": 3,
                  "model zoo": 3, "zoo": 2, "my model": 2, "meu modelo": 2, "onnx": 2, "torchscript": 2,
                  "sem treinar": 2, "without training": 2},
        min_score=3, nodes=("model.use",), build=lambda text, det: build_flow(),
        inputs=("the model (zoo name or file with its card)", "the query or the observations", "a reference solver"),
        uncited="the gate follows the zoo's validity and trust rules (pinneapple_hub.zoo); no external reference",
        description="pinneapple_analysis.surrogate.model_source.use_model"))


_register_capabilities()
