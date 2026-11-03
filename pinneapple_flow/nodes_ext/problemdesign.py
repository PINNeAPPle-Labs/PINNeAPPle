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


# ------------------------------------------------------------------------------------- a drafted model, run (#1246)
# illustrative values of a cylindrical-cell-sized slab (not a datasheet; tests/test_problemdesign_formulate.py):
# they only make the battery draft solvable, every one of them is the user's to replace
CELL_VALUES = {"rho": 2500, "cp": 1000, "k": 1.0, "I": 5, "R0": 0.03, "EaR": 2000, "Tref": 298.15, "dUdT": 1e-4,
               "Vc": 1.65e-5, "h": 10, "Tamb": 298.15, "T0": 298.15}


@node("problemdesign.run_draft", "Run a drafted model (formulate) on a box: FD or PINN, MMS order, balance",
      category="Problem design", outputs=[Port("metrics", "metrics")],
      params=[Param("phenomenon", "enum", "battery_cell_thermal",
                    choices=["battery_cell_thermal", "solid_heat", "room_air_species", "arrhenius"],
                    help="the formulate draft (pinneapple_problemdesign.formulate.known())"),
              Param("dims", "int", 3, minimum=1, maximum=3),
              Param("steady", "bool", False),
              Param("values", "dict", CELL_VALUES, help="constants of the draft (SI); illustrative defaults"),
              Param("box", "list", [0.065, 0.03, 0.0085], help="box lengths per axis, m (x, y, z)"),
              Param("method", "enum", "auto", choices=["auto", "fd", "pinn"],
                    help="auto: box finite differences when the draft fits them, else a PINN on the draft's loss"),
              Param("t_end", "float", 600.0, minimum=0.0, help="s (0: one diffusion time)"),
              Param("n", "int", 11, minimum=5, help="nodes per axis"),
              Param("n_t", "int", 100, minimum=4, help="time steps (Crank-Nicolson)"),
              Param("adam_steps", "int", 2000, minimum=0, advanced=True, help="PINN only"),
              Param("lbfgs_steps", "int", 200, minimum=0, advanced=True, help="PINN only"),
              Param("n_evaluations", "int", 0, minimum=0, advanced=True,
                    help="runs a study needs (0: none): recommend_architecture answers whether a surrogate pays")],
      calls="pinneapple_problemdesign.formulate.formulate + pinneapple_problemdesign.run_draft.run_draft")
def run_drafted_model(ctx, inputs, params):
    """The draft composed from its laws, run as a job: the field, the MMS order, the balance, the assumptions."""
    from pinneapple_problemdesign.formulate import formulate
    from pinneapple_problemdesign.run_draft import run_draft
    d = formulate(str(params["phenomenon"]), dims=int(params["dims"]), steady=bool(params["steady"]),
                  values={k: float(v) for k, v in dict(params["values"]).items()})
    budget = {"n": int(params["n"]), "n_t": int(params["n_t"]), "adam_steps": int(params["adam_steps"]),
              "lbfgs_steps": int(params["lbfgs_steps"])}
    if float(params["t_end"]) > 0:
        budget["t_end"] = float(params["t_end"])
    if int(params["n_evaluations"]):
        budget["n_evaluations"] = int(params["n_evaluations"])
    r = run_draft(d, geometry={"box": [float(v) for v in params["box"]]}, method=str(params["method"]), budget=budget)
    ctx.metric("status", r.status, r.reason)
    if r.status != "done":
        ctx.summary(r.summary())
        return {"metrics": ctx.report["metrics"]}
    ctx.metric("method", r.method, r.reason)
    m = r.metrics
    for k in ("max", "min", "mean"):
        ctx.metric(f"{r.field_name}.{k}", m[k], f"{r.unit} at t = {m['t_end']:g} s")
    if r.mms.get("levels"):
        ctx.metric("mms.observed_order", r.mms["observed_order"], f"expected {r.mms['expected_order']:g}; grids "
                   + "/".join(str(x["n"]) for x in r.mms["levels"]))
        ctx.table("manufactured solution", ["nodes per axis", "h (m)", "max error", "order"],
                  [[x["n"], x["h"], x["max_error"], x.get("order", "")] for x in r.mms["levels"]])
    if r.balance:
        ctx.metric("balance.imbalance_rel", r.balance["imbalance_rel"], r.balance["what"])
    if "reference.fd_max_error_rel" in m:
        ctx.metric("reference.fd_max_error_rel", m["reference.fd_max_error_rel"], "PINN vs the FD solution")
    ctx.metric("solve_seconds", m.get("solve_seconds"), "measured")
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r.checks])
    ctx.table("assumptions", ["id", "assumption", "applied", "criterion", "evaluated"],
              [[a["id"], a["title"], a.get("applied"), a.get("limit", ""), a.get("evaluated", "")]
               for a in r.assumptions])
    if r.history.get("t") and len(r.history["t"]) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for k in ("max", "mean", "min"):
            ax.plot(r.history["t"], r.history[k], label=k)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(f"{r.field_name} ({r.unit})")
        ax.legend()
        fig.tight_layout()
        ctx.figure(fig, "history.png", f"{r.field_name} over time ({r.method})")
        plt.close(fig)
    bad = [c["code"] for c in r.checks if c["status"] == "fail"]
    ctx.metric("verdict", "fail" if bad else "pass", ", ".join(bad) or "every check passes or is inconclusive")
    ctx.save_json("draft_run", r.to_dict())
    ctx.summary(r.summary())
    return {"metrics": ctx.report["metrics"]}


def build_run_draft_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Run a drafted model: the formulated laws solved on a box, with the MMS order and the balance"
    fb = FlowBuilder(title, "A model composed by pinneapple_problemdesign.formulate (laws with units and sources) run "
                            "as a job: box finite differences (Crank-Nicolson, Newton) when the draft is a scalar "
                            "diffusion-reaction law, else a PINN on the draft's own loss; the manufactured solution "
                            "at the expected order, the balance of the drafted law, every assumption re-stated "
                            "(pinneapple_problemdesign.run_draft).")
    fb.add("run", "problemdesign.run_draft", dict(params), label="Run the drafted model")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("run", "metrics", "report", "items")
    return fb


def _register_run_draft_capability():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="problemdesign.run_draft", title="Run a drafted (formulated) model as a job", domain="problem design",
        keywords={"drafted model": 3, "model draft": 3, "modelo proposto": 3, "modelo formulado": 3,
                  "run the draft": 3, "rodar o modelo": 2, "run the model": 2, "formulate": 2, "formular": 1.5,
                  "manufactured solution": 1.5, "solução manufaturada": 1.5, "battery cell": 1, "célula": 1},
        min_score=3, nodes=("problemdesign.run_draft",), build=lambda text, det: build_run_draft_flow(),
        inputs=("the phenomenon and its constants", "the box", "the method (auto / fd / pinn)"),
        uncited="the laws carry their sources in pinneapple_problemdesign.formulate (Incropera, Bernardi et al. 1985, "
                "Bird, Stewart & Lightfoot); they are not in the flow reference file yet",
        description="pinneapple_problemdesign.run_draft.run_draft"))


_register_run_draft_capability()
