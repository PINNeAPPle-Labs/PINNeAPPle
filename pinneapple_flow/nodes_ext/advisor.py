"""The pipeline advisor as a flow node: a problem description in, the suggested study pipeline out (plan JSON, the
Mermaid source, the SVG and an HTML page), with what each stage costs and what it must pass.

The node wraps :func:`pinneapple_advisor.advise` -- the ten questions answered in order (physics, 3-D, mesh, what the
user already has, dataset, solver and cost, sweep or surrogate, architecture, dataset recipe, checks and views) --
and writes the same four files the CLI writes, next to the run.
"""
from __future__ import annotations

import json

from ..registry import Param, Port, node


@node("advisor.pipeline", "Suggested study pipeline from a problem description", category="Study design",
      outputs=[Port("plan", "report"), Port("metrics", "metrics")],
      params=[Param("text", "str", "transient heat in a 3D bracket, I have the STEP but no mesh",
                    help="the problem in a sentence or more (English or Portuguese)"),
              Param("answers", "dict", {}, help="brief fields answered up front, e.g. {dimension: 3, goal: inverse}"),
              Param("language", "str", "pt", choices=["pt", "en"]),
              Param("offline", "bool", True, help="no language model: the keyword rules read the sentence"),
              Param("model", "str", "", help="Ollama model name, when offline is false")],
      calls="pinneapple_advisor.advise, pinneapple_advisor.extract.detect, pinneapple_advisor.brief.Brief, "
            "pinneapple_advisor.questions.next_questions, pinneapple_advisor.compose.build, "
            "pinneapple_advisor.estimate.CostModel, pinneapple_advisor.capabilities.verify_tools, "
            "pinneapple_advisor.render.svg, pinneapple_advisor.schema.StudyPlan, pinneapple_advisor.cli.main")
def pipeline(ctx, inputs, params):
    """Compose the pipeline and write plan.json, pipeline.mermaid, pipeline.svg and pipeline.html into the run."""
    from pinneapple_advisor import advise
    from pinneapple_advisor.schema import text_of

    lang = str(params["language"])
    plan = advise(str(params["text"]), answers=dict(params["answers"] or {}),
                  llm=False if bool(params["offline"]) else None, model=str(params["model"] or ""),
                  language=lang)
    for name, kind in (("plan", "json"), ("pipeline", "mermaid"), ("pipeline", "svg"), ("pipeline", "html")):
        ext = {"json": "json", "mermaid": "mmd", "svg": "svg", "html": "html"}[kind]
        ctx.path(f"{name}.{ext}").write_text(plan.diagram(kind) if kind != "json" else plan.json(),
                                             encoding="utf-8")
    ctx.view("html", {"path": str(ctx.path("pipeline.html")), "title": "Suggested pipeline"})
    total = plan.total_estimate()
    ctx.metric("stages.needed", len(plan.needed()))
    ctx.metric("campaign.hours", round(total.total_seconds / 3600, 3),
               f"estimated cost model, {total.confidence}")
    for decision in plan.decisions:
        ctx.metric(f"choice.{decision.topic}", str(decision.choice), decision.source)
    ctx.table("stages", ["#", "stage", "status", "estimate"],
              [[i, text_of(s.title, lang), s.status,
                "" if s.estimate is None else f"{s.estimate.total_seconds / 60:.1f} min"]
               for i, s in enumerate(plan.stages, 1)])
    return {"plan": ctx.save_json("plan", json.loads(plan.json())), "metrics": ctx.report["metrics"]}


def build_flow(text: str = "", **params):
    """The advisor as a two-node flow: the pipeline, then the report of what it wrote."""
    from ..planner.build import FlowBuilder
    title = "Suggested study pipeline"
    fb = FlowBuilder(title, "The ten questions answered in order -- physics, geometry, mesh, dataset, solver, "
                            "surrogate, architecture, dataset recipe, checks and views -- with the PINNeAPPle call "
                            "of every step (pinneapple_advisor.advise).")
    fb.add("advise", "advisor.pipeline", {"text": text, **params}, label="Pipeline advisor")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("advise", "plan", "report", "items")
    return fb


def _build(text: str, _det):
    return build_flow(text)


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="advisor.pipeline", title="Suggest the study pipeline for a problem",
        domain="study design", example="",
        keywords={"pipeline": 3, "what should i run": 3, "study plan": 3, "which surrogate": 2.5,
                  "qual arquitetura": 2.5, "pinelines": 2, "roteiro": 2, "solver or surrogate": 3,
                  "dataset": 1.5, "mesh": 1.5},
        min_score=3, nodes=("advisor.pipeline",), build=_build, inputs=("a problem description",),
        uncited="a recommendation over the library's own APIs; nothing is executed",
        description="pinneapple_advisor.advise"))


_register_capabilities()
