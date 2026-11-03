"""The standard model-benchmarking workflow (:mod:`pinneapple_analysis.surrogate.benchmark_workflow`, #1191): library
families trained on the same rows and models that already exist (zoo, exported, files), on the same test points
over several seeds, ranked with paired-bootstrap intervals; a winner only when it is separated."""
from __future__ import annotations

from ..registry import Param, Port, node

TOOL = "pinneapple_design.aero.wing_design:wing_performance"


@node("benchmark.models", "Model benchmarking (equal data, intervals, winner only when separated)",
      category="Benchmarks", outputs=[Port("metrics", "metrics")],
      params=[Param("tool", "str", TOOL, help="the reference: a library tool id"),
              Param("bounds", "dict", {"aspect_ratio": [5.0, 12.0], "taper": [0.3, 1.0], "twist": [-6.0, 0.0]}),
              Param("output", "str", "drag_N"), Param("fixed", "dict", {}),
              Param("families", "list", ["gp", "mlp_ensemble", "quadratic"]),
              Param("zoo", "list", [], help="zoo models to evaluate on the same points"),
              Param("n_train", "int", 40, minimum=5), Param("n_test", "int", 100, minimum=10),
              Param("seeds", "list", [0, 1, 2]), Param("extrapolate", "float", 0.1, minimum=0.0, maximum=1.0)],
      calls="pinneapple_analysis.surrogate.benchmark_workflow.benchmark_models")
def benchmark_models(ctx, inputs, params):
    from pinneapple_analysis.surrogate.benchmark_workflow import benchmark_models as run
    from pinneapple_analysis.surrogate.model_source import open_model
    preds = [open_model("zoo", name=n) for n in params.get("zoo") or []]
    r = run(families=list(params["families"]), predictors=preds, tool=str(params["tool"]),
            bounds={k: tuple(v) for k, v in dict(params["bounds"]).items()}, output=str(params["output"]),
            fixed=dict(params["fixed"] or {}), n_train=int(params["n_train"]), n_test=int(params["n_test"]),
            seeds=[int(s) for s in params["seeds"]], extrapolate=float(params["extrapolate"]))
    ctx.table("ranking", ["candidate", "rel RMSE (mean over seeds)", "spread", "extrapolation rel RMSE"],
              [[s["candidate"], round(s["rel_rmse_mean"], 6), round(s["rel_rmse_spread"], 6),
                None if s["extrapolation_rel_rmse"] is None else round(s["extrapolation_rel_rmse"], 6)]
               for s in r["summary"]])
    ctx.table("paired bootstrap (best vs each)", ["other", "mean squared-error difference", "95 % interval",
                                                  "separated"],
              [[p["other"], p["mean_sq_diff"], p["ci95"], p["separated"]] for p in r["pairs"]])
    for s in r["summary"]:
        ctx.metric(f"{s['candidate']}.rel_rmse", s["rel_rmse_mean"], "mean over seeds, inside the box")
    ctx.metric("winner", r["winner"] or "none", r["verdict"])
    ctx.summary(r["verdict"])
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Model benchmarking: equal data, intervals, a winner only when separated"
    fb = FlowBuilder(title, "Library families trained on the same rows and existing models on the same test points "
                            "over several seeds, ranked with paired-bootstrap intervals "
                            "(pinneapple_analysis.surrogate.benchmark_workflow).")
    fb.add("bench", "benchmark.models", dict(params), label="Benchmark")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("bench", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="benchmark.models", title="Model benchmarking (standard workflow)", domain="benchmarks",
        keywords={"benchmark": 3, "benchmarking": 3, "compare models": 3, "comparar modelos": 3,
                  "which model is best": 3, "qual modelo é melhor": 3, "model comparison": 3,
                  "comparação de modelos": 3, "ranking": 1},
        min_score=3, nodes=("benchmark.models",), build=lambda text, det: build_flow(),
        inputs=("the reference (tool or table)", "the candidate families and existing models"),
        uncited="paired bootstrap of test errors (Efron & Tibshirani 1993); not in the flow reference file yet",
        description="pinneapple_analysis.surrogate.benchmark_workflow.benchmark_models"))


_register_capabilities()
