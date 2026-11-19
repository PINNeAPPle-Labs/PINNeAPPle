"""Text to simulation and text to Physics AI as flow nodes (library: :mod:`pinneapple_intent`).

A sentence is compiled into a study contract and run through its fidelity ladder; the node reports the quantity of
interest, the decision the trust report reached, and one row per check, and it writes the ``contract.json``,
``run.json`` and ``report.md`` the Studio shows and a replay can rebuild.
"""
from __future__ import annotations

from ..registry import Param, Port, node

_TAGS = ("section:studio", "text to simulation", "intent", "verification", "replay", "physics ai")


@node("intent.compile", "Compile a sentence into a study contract", category="Design",
      outputs=[Port("metrics", "metrics")],
      params=[Param("text", "text", "a 20 mm pipe, 3 m long, carrying 0.5 L/min of water, viscosity 1 mPa*s",
                    max_length=2000, help="the study in plain language, with its numbers and units"),
              Param("mode", "enum", "simulation", choices=["simulation", "physics_ai"],
                    help="physics_ai adds the surrogate rung"),
              Param("answers", "dict", {}, help="answers to the questions the sentence leaves open")],
      calls="pinneapple_intent.compile:compile_study (pp.compile_study)", tags=_TAGS)
def intent_compile(ctx, inputs, params):
    """The contract the sentence compiles to — the family, the equations, the ladder and every value with the place
    it came from — or the questions it still needs, or the refusal saying what would be missing."""
    import pinneapple as pp
    contract = pp.compile_study(params["text"], mode=params["mode"], answers=params["answers"] or None)
    if not hasattr(contract, "status"):                       # out of scope: no family, or a refusal
        payload = contract.to_dict()
        ctx.summary(str(contract))
        ctx.table("reply", ["key", "value"], [[k, str(v)[:300]] for k, v in payload.items()])
        ctx.metric("status", str(payload.get("status", payload.get("kind", "out_of_scope"))))
        return {"metrics": ctx.report["metrics"]}
    ctx.metric("status", contract.status)
    ctx.metric("family", contract.family)
    ctx.metric("contract_id", contract.id)
    ctx.table("values", ["quantity", "value", "provenance", "source"],
              [[row["name"], row["value"], row["provenance"], row["source"]] for row in contract.table()])
    ctx.table("questions", ["slot", "question", "unit"],
              [[q.slot, q.text, q.unit] for q in contract.questions])
    ctx.table("ladder", ["rung", "what it is", "available", "reason"],
              [[r["name"], r["label"], str(r["available"]), r["reason"]]
               for r in contract.to_dict()["workflow"]["ladder"]])
    ctx.save_json("contract", contract.to_dict())
    ctx.summary(contract.summary())
    return {"metrics": ctx.report["metrics"]}


@node("intent.study", "Run a sentence end to end and check it", category="Design",
      outputs=[Port("metrics", "metrics")],
      params=[Param("text", "text", "a 12 mm steel wall with k = 200 W/mK, 300 degC inside and air at 25 degC "
                                    "outside with h = 12 W/m2K", max_length=2000),
              Param("mode", "enum", "simulation", choices=["simulation", "physics_ai"]),
              Param("answers", "dict", {})],
      calls="pinneapple_intent.execute:study (pp.study) / pinneapple_intent.replay:replay", tags=_TAGS)
def intent_study(ctx, inputs, params):
    """Compile, run the fidelity ladder, and report the quantity of interest with the decision the trust report
    reached: the analytic rung against the solver, the change under refinement, and (in physics_ai mode) the
    surrogate on designs held out of its fit."""
    import pinneapple as pp
    outcome = pp.study(params["text"], mode=params["mode"], answers=params["answers"] or None,
                       out_dir=str(ctx.path("study")), progress=ctx.log)
    if not hasattr(outcome, "measurements"):                  # an open question, or a refusal
        ctx.metric("decision", "not run")
        ctx.table("reply", ["key", "value"],
                  [[k, str(v)[:300]] for k, v in (outcome.to_dict() if hasattr(outcome, "to_dict") else {}).items()])
        ctx.summary(str(outcome))
        return {"metrics": ctx.report["metrics"]}
    run = outcome
    quantity = run.outputs()["qoi"]
    ctx.metric("family", run.family)
    ctx.metric("mode", run.contract.mode)
    ctx.metric("decision", run.decision)
    ctx.metric(quantity["name"], float(quantity["value"]) if quantity["value"] is not None else float("nan"),
               quantity["unit"])
    for name, measurement in run.measurements.items():
        ctx.metric(f"{name}_ran", int(measurement.ran), measurement.reason or measurement.rung)
        for key, value in measurement.outputs.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                ctx.metric(f"{name}.{key}", float(value))
    ctx.table("checks", ["check", "result", "value", "limit", "detail"],
              [[c.get("name"), "pass" if c.get("passed") else "FAIL", c.get("value"), c.get("limit"),
                str(c.get("detail", ""))[:300]] for c in run.checks])
    ctx.save_json("run", run.to_dict())
    report = ctx.path("report.md")
    report.write_text(run.report_markdown(), encoding="utf-8")
    ctx.log(f"report: {report}")
    ctx.summary(run.summary())
    return {"metrics": ctx.report["metrics"]}
