"""Futures nodes: the What If Engine of :mod:`pinneapple_futures` and the demonstrators of the Lab.

``futures.ask`` turns a speculative question into a plan (which equations contain the parameter that changed, and
which experiment can run it) and, at the caller's choice, runs it. ``futures.run`` runs one demonstrator directly.
"""
from __future__ import annotations

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node

#: kept literal so registering the nodes imports nothing heavy; tests/test_futures.py checks it equals
#: pinneapple_futures.names()
EXPERIMENTS = ["futures.exoplanet_laboratory", "futures.mars_habitat", "futures.programmable_matter"]
DIMS = ["first", "0D", "1D"]
_TAGS = ("section:validation", "futures", "speculative", "laboratory")


@node("futures.ask", "What if (speculative question to experiment)", category="Validate",
      outputs=[Port("metrics", "metrics"), Port("report", "report")],
      params=[Param("question", "str", "What if Earth's gravity were 2.5 times stronger?",
                    help="the change the question asks about; the engine reads gravity, temperature, pressure, "
                         "insolation and albedo"),
              Param("run", "bool", True, help="run the experiment the question maps to (off: only the plan)"),
              Param("quick", "bool", True, help="reduced sizes (seconds); off for the committed resolution")],
      calls="pinneapple_futures.whatif.ask(question) -> Brief.stages(); Brief.run() over "
            "pinneapple_futures.core.SpeculativeExperiment (pinneapple_futures.schema Scenario and Proposal, "
            "pinneapple_futures.plausibility.assess, pinneapple_futures.numerics)",
      tags=_TAGS)
def what_if(ctx, inputs, params):
    """Read a speculative question: what changed, which equations contain it, and the experiment that runs it."""
    import pinneapple_futures as F

    brief = F.ask(params["question"])
    ctx.table("Plan", ["step", "content"], [[k, v] for k, v in brief.stages().items()])
    ctx.table("Which equations contain it", ["phenomenon", "law", "effect"],
              [[p, law, effect] for p, law, effect in brief.dependencies()])
    for name, d in brief.changes.items():
        ctx.metric(f"change[{name}]", d.value, d.unit)
    for i, w in enumerate(brief.warnings):
        ctx.metric(f"warning[{i}]", 1.0, w)
    if not brief.understood:
        raise StepFailed("the Lab could not read this change; it reads gravity, temperature, pressure, "
                         "insolation and albedo")
    out = {"metrics": brief.to_dict(), "report": None}
    if not params["run"]:
        ctx.summary(f"{params['question']}: plan only, experiments {', '.join(brief.experiments) or 'none'}")
        return out
    if not brief.answerable:
        raise StepFailed(f"no registered Futures experiment contains {brief.parameter!r}")
    r = brief.run(quick=bool(params["quick"]))
    out["report"] = _record(ctx, r)
    return out


@node("futures.run", "Futures Lab experiment", category="Validate",
      outputs=[Port("metrics", "metrics"), Port("report", "report")],
      params=[Param("experiment", "enum", EXPERIMENTS[0], choices=EXPERIMENTS,
                    help="the demonstrator (pp.futures.names() lists them)"),
              Param("dim", "enum", "first", choices=DIMS, help="the dimension to run in (first = its first)"),
              Param("quick", "bool", True, help="reduced sizes (seconds); off for the committed resolution")],
      calls="pinneapple_futures.core.run(name, dim) over the pinneapple_futures.labs demonstrators "
            "(mars_habitat, programmable_matter, exoplanet), assessed by pinneapple_futures.plausibility.assess "
            "and published by pinneapple_futures.report.publish",
      tags=_TAGS)
def run_experiment(ctx, inputs, params):
    """Run a demonstrator of the Lab: comparisons, physics checks, trust report and the evidence assessment."""
    import pinneapple_futures as F

    e = F.get(params["experiment"])
    dim = None if params["dim"] == "first" else params["dim"]
    if dim is not None and dim not in e.dims:
        raise StepFailed(f"{e.name} runs in {', '.join(e.dims)}, not {dim}")
    r = e.run(dim, quick=bool(params["quick"]))
    return {"metrics": r.to_dict(), "report": _record(ctx, r)}


@node("futures.atlas", "Futures Lab atlas", category="Analysis", outputs=[Port("report", "report")],
      params=[Param("status", "enum", "any", choices=["any", "available", "partial", "planned"],
                    help="which subareas to list in the table")],
      calls="pinneapple_futures.atlas.load().to_markdown() / validate(): every pointer against the code that "
            "exists", tags=("section:validation", "futures", "atlas"))
def atlas(ctx, inputs, params):
    """The eight directions of the Lab: available (a demonstrator), partial (library code), planned (nothing yet)."""
    import pinneapple_futures as F

    a = F.atlas()
    rows = [r for r in a.rows() if params["status"] in ("any", r["status"])]
    ctx.table("Atlas", ["direction", "subarea", "status", "covered by"],
              [[r["title"], r["name"], r["status"],
                ", ".join(p.get("name") or p.get("module") for p in r.get("covered_by", []))] for r in rows])
    for k, v in a.counts().items():
        ctx.metric(f"subareas_{k}", v)
    problems = F.validate()
    ctx.summary(f"{len(a.areas)} directions, {len(problems)} problems with the pointers")
    path = ctx.path("atlas.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Directions of the Futures Lab\n\n" + a.to_markdown() + "\n")
    return {"report": {"path": str(path), "kind": "md"}}


def _record(ctx, r) -> dict:
    """The tables and metrics every futures result contributes to a flow run."""
    for c in r.comparisons:
        ctx.metric(f"rel_error[{c.quantity}]", c.rel_error, f"vs {c.kind.replace('_', ' ')}: {c.source}")
    ctx.table("Comparisons", ["quantity", "computed", "reference", "unit", "rel. error", "tolerance", "source"],
              [[c.quantity, c.computed, c.reference, c.unit, c.rel_error, c.tol, c.source] for c in r.comparisons])
    ctx.table("Evidence", ["axis", "passed", "assessed", "evidence"],
              [[a.axis, a.passed, a.total, ", ".join(a.evidence)] for a in (r.assessment.axes.values()
                                                                            if r.assessment else [])])
    ctx.summary(f"{r.title} [{r.dim}]: {'PASS' if r.passed else 'FAIL'}, trust {r.decision}, "
                f"{r.assessment.headline() if r.assessment else 'no assessment'}")
    paths = r.save(str(ctx.path(r.experiment.replace(".", "_"))))
    return {"path": paths["result"], "kind": "json"}
