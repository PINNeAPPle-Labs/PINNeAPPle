"""The standard surrogate-modeling workflow (:mod:`pinneapple_analysis.surrogate.workflow`, #1186): a design of
experiments over a library tool (or a CSV table), the family ``recommend_architecture`` picks (or none, with its
reason), the equal-budget comparison of the library's families, checks on new tool calls, and an exported model
that refuses to extrapolate."""
from __future__ import annotations

from pathlib import Path

from ..registry import Param, Port, node

DEFAULT_TOOL = "pinneapple_design.aero.wing_design:wing_performance"
DEFAULT_BOUNDS = {"aspect_ratio": [5.0, 12.0], "taper": [0.3, 1.0], "twist": [-6.0, 0.0]}
DEFAULT_OUTPUTS = ["drag_N", "M_root_Nm", "cl_local_max_landing"]


@node("surrogate.modeling", "Surrogate modeling (recommended family, checked, exported)", category="Surrogates",
      outputs=[Port("metrics", "metrics")],
      params=[Param("tool", "str", DEFAULT_TOOL, help="library tool id 'module:function' (keyword inputs)"),
              Param("bounds", "dict", DEFAULT_BOUNDS, help="varied input -> [low, high]"),
              Param("outputs", "list", DEFAULT_OUTPUTS, help="outputs of the tool to model"),
              Param("fixed", "dict", {}, help="inputs held fixed"),
              Param("table_csv", "str", "", help="instead of a tool: a CSV with the inputs and outputs columns"),
              Param("inputs", "list", [], help="table only: the input columns"),
              Param("n_doe", "int", 60, minimum=8), Param("n_fresh", "int", 200, minimum=10),
              Param("n_uses", "int", 100_000, minimum=1, help="predictions the model will serve (decides whether "
                                                                "it pays off)"),
              Param("accuracy_target", "float", 0.05, minimum=1e-4, maximum=1.0),
              Param("force", "bool", False, advanced=True, help="train even when no model pays off (to compare)"),
              Param("seed", "int", 0)],
      calls="pinneapple_analysis.surrogate.workflow.surrogate_modeling (pinneapple_autolab.study.benchmark, "
            "architecture_recommendation.recommend_architecture)")
def surrogate_modeling(ctx, inputs, params):
    """Run the workflow and export the model next to the run."""
    from pinneapple_analysis.surrogate.workflow import surrogate_modeling as run
    kw = dict(n_doe=int(params["n_doe"]), n_fresh=int(params["n_fresh"]), n_uses=int(params["n_uses"]),
              accuracy_target=float(params["accuracy_target"]), force=bool(params["force"]), seed=int(params["seed"]),
              out_dir=str(Path(ctx.workdir) / "surrogate"))
    if params.get("table_csv"):
        import csv
        with open(params["table_csv"], newline="") as f:
            rows = list(csv.DictReader(f))
        cols = {k: [float(r[k]) if r[k] not in ("", None) else float("nan") for r in rows] for k in rows[0]}
        r = run(table=cols, inputs=list(params["inputs"]), outputs=list(params["outputs"]), **kw)
    else:
        r = run(tool=str(params["tool"]), bounds={k: tuple(v) for k, v in dict(params["bounds"]).items()},
                outputs=list(params["outputs"]), fixed=dict(params["fixed"] or {}), **kw)
    rec = r["recommendation"]
    ctx.metric("recommendation", rec["recommended"][0], (rec.get("decision_path") or [rec["reasoning"]])[0])
    if r["cost"]["solve_seconds"] is not None:
        ctx.metric("cost.solve_seconds", r["cost"]["solve_seconds"], "measured: mean over the design of experiments")
    for o, v in (r.get("results") or {}).items():
        ctx.metric(f"{o}.family", v["family"], v["reason"])
        ctx.metric(f"{o}.rel_rmse", v["rel_rmse"], "relative RMSE of the output range on solves never trained on")
        ctx.metric(f"{o}.coverage90", v["coverage90"], "share of those solves inside the 90 % interval")
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    if r.get("model_path"):
        ctx.metric("model_path", r["model_path"], "load with pinneapple_analysis.surrogate.workflow.load_surrogate")
    bad = [c["code"] for c in r["checks"] if c["status"] == "fail"]
    ctx.metric("verdict", "fail" if bad else "pass", ", ".join(bad) or "every check passes")
    ctx.summary(("no surrogate needed: " + (rec.get("decision_path") or [""])[0]) if not r["trained"] else
                ", ".join(f"{o}: {v['family']} {v['rel_rmse']:.2%}" for o, v in r["results"].items())
                + ("; fail: " + ", ".join(bad) if bad else "; verified"))
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Surrogate modeling: recommended family, checked on new solves, exported"
    fb = FlowBuilder(title, "A design of experiments over a tool (or a table), the family recommend_architecture "
                            "picks or none, the library's families compared on one protocol, checks on new tool "
                            "calls, and an exported model that refuses to extrapolate "
                            "(pinneapple_analysis.surrogate.workflow).")
    fb.add("surrogate", "surrogate.modeling", dict(params), label="Surrogate modeling")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("surrogate", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="surrogate.modeling", title="Surrogate modeling (standard workflow)", domain="surrogates",
        keywords={"surrogate": 3, "surrogate model": 3, "modelo substituto": 3, "metamodel": 3, "metamodelo": 3,
                  "response surface": 2, "superfície de resposta": 2, "emulator": 2, "emulador": 2,
                  "gaussian process": 2, "kriging": 2},
        min_score=3, nodes=("surrogate.modeling",), build=lambda text, det: build_flow(),
        inputs=("a tool and the ranges of its inputs, or a table", "how many predictions the model will serve",
                "the accuracy needed"),
        uncited="the protocol cites Rasmussen & Williams (2006), Vovk (2015) and Forrester et al. (2008) in "
                "pinneapple_autolab.study.benchmark; they are not in the flow reference file yet",
        description="pinneapple_analysis.surrogate.workflow.surrogate_modeling"))


_register_capabilities()
