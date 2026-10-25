"""Problem design without an LLM (``examples/problem_designer/03_offline_spec_to_report``): a partly known
:class:`pinneapple_problemdesign.schema.ProblemSpec`, its validators (warnings and safe assumptions), the FNO-first
plan and the Markdown / JSON design report."""
from __future__ import annotations

import html

from ..registry import Param, Port, node

HEAT_EXCHANGER = {
    "title": "Heat Exchanger: Outlet Temperature Forecast",
    "goal": "Forecast outlet temperature 6 hours ahead to optimize energy usage", "task_type": "forecasting",
    "inputs": ["T_in (°C)", "flow_rate (kg/s)", "T_ambient (°C)", "pressure (bar)"], "outputs": ["T_out (°C)"],
    "frequency": "10min", "input_window": "48 steps (8h)", "horizon": "6h",
    "domain_context": "Industrial heat exchanger. Maintenance causes intermittent missing data. We expect seasonal "
                      "drift across weeks/months.",
    "data": {"sources": ["s3://plant-data/hex/parquet/"], "format": "parquet", "sampling": "10min",
             "variables_observed": ["T_in", "flow_rate", "T_ambient", "pressure", "T_out"],
             "target_variables": ["T_out"], "known_quality_issues": ["missing points during maintenance",
                                                                     "sensor spikes"],
             "missingness": "bursty gaps during maintenance windows", "train_span": "2023-01-01 .. 2025-01-01",
             "val_split_policy": "last 3 months as validation (time-based)", "labels_available": True},
    "validation": {"primary_metrics": ["MAE"], "acceptance_criteria": "MAE < 0.5°C at 6h horizon",
                   "robustness_tests": ["missingness stress", "drift across weeks", "extreme ambient temp"]},
}


@node("problemdesign.offline_report", "Design report from a problem spec (no LLM)", category="Problem design",
      outputs=[Port("metrics", "metrics"), Port("report", "report")],
      params=[Param("spec", "dict", HEAT_EXCHANGER, help="ProblemSpec fields; 'data' / 'validation' as dicts")],
      calls="pinneapple_problemdesign.schema.ProblemSpec / DataSpec / ValidationSpec / DesignReport + "
            "elicitation.validators.validate_and_suggest + knowledge.mapping.build_plan_fno_first + "
            "renderers.report_md.render_markdown_report / report_json.render_json_report")
def offline_report(ctx, inputs, params):
    """Validators, plan and report of a partly known spec, as in the example (gaps empty)."""
    import json

    from pinneapple_problemdesign.elicitation.validators import validate_and_suggest
    from pinneapple_problemdesign.knowledge.mapping import build_plan_fno_first
    from pinneapple_problemdesign.renderers.report_json import render_json_report
    from pinneapple_problemdesign.renderers.report_md import render_markdown_report
    from pinneapple_problemdesign.schema import DataSpec, DesignReport, ProblemSpec, ValidationSpec
    raw = dict(params["spec"])
    kw = {k: v for k, v in raw.items() if k not in ("data", "validation")}
    if "data" in raw:
        kw["data"] = DataSpec(**raw["data"])
    if "validation" in raw:
        kw["validation"] = ValidationSpec(**raw["validation"])
    spec = ProblemSpec(**kw)
    warnings, assumptions = validate_and_suggest(spec)
    spec.assumptions.extend(assumptions)
    plan = build_plan_fno_first(spec, [])
    report = DesignReport(spec=spec, gaps=[], plan=plan)
    md = render_markdown_report(report, warnings)
    js = render_json_report(report)
    ctx.path("offline_report.md").write_text(md, encoding="utf-8")
    page = ctx.path("offline_report.html")
    page.write_text("<!doctype html><meta charset='utf-8'><title>Design report</title><body style='font:13px "
                    "system-ui;margin:16px'><pre style='white-space:pre-wrap'>" + html.escape(md) + "</pre>",
                    encoding="utf-8")
    ctx.view("html", {"path": str(page), "title": "Design report"})
    ctx.metric("warnings", len(warnings))
    ctx.metric("assumptions", len(assumptions))
    ctx.metric("plan_steps", len(plan.steps))
    ctx.metric("report_md_chars", len(md))
    ctx.table("plan", ["step", "why"], [[s.title, s.why] for s in plan.steps])
    return {"metrics": ctx.report["metrics"], "report": ctx.save_json("offline_report", json.loads(js))}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="problemdesign.offline_report", title="A design report and plan from a problem spec, without an LLM",
        domain="problem design", example="problem_designer",
        keywords={"design report": 3, "problem spec": 3, "problem design": 3, "offline report": 3,
                  "requirements": 1.5, "plan the study": 2},
        min_score=3, nodes=("problemdesign.offline_report",), inputs=("what is known of the problem",),
        uncited="a structured template of the problem-design package: no physics decision",
        template="problem_designer/03_offline_spec_to_report",
        description="examples/problem_designer/03_offline_spec_to_report.py"))


_register_capabilities()
