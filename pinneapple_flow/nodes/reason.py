"""Reasoning nodes: requirement, concept, check a failure mode, check every requirement, countermeasure, the chain of
whys and its report (library: :mod:`pinneapple_reason`)."""
from __future__ import annotations

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact, load_artifact

_CASES = ["roof", "footbridge", "power_board", "pumped_main"]
_TAGS = ("section:optimization", "reasoning", "failure modes", "design")


#: the built-in library (kept literal so registering the nodes imports nothing heavy; tests/test_reason.py checks
#: that these lists equal pinneapple_reason.LIBRARY)
MODES = ["bending_strength", "column_separation", "deflection", "fatigue", "flexural_buckling", "galvanic_corrosion",
         "joule_heating", "lap_water_ingress", "lateral_torsional_buckling", "pedestrian_resonance", "pump_cavitation",
         "radiant_heat_gain", "underside_condensation", "water_hammer", "wind_uplift"]
COUNTERMEASURES = ["ceiling_insulation", "ceiling_ventilated_attic", "coat_cathode", "column_midheight_brace",
                   "deepen_girders", "forced_air", "heatsink_natural", "higher_pressure_class", "increase_slope",
                   "insulating_kit", "larger_suction_pipe", "lateral_bracing", "longer_lap", "lower_pump",
                   "radiant_barrier", "slow_valve_closure", "thermal_vias", "thicker_section", "tile_clips",
                   "tuned_mass_damper", "wider_flanges"]


def _reqs(value):
    from pinneapple_reason import Requirement
    if not value:
        raise StepFailed("connect a requirement")
    return [Requirement.from_dict(r) for r in value["requirements"]]


def _concept(value):
    from pinneapple_reason import Concept
    if not value:
        raise StepFailed("connect a concept")
    return Concept.from_dict(value)


def _vrow(v):
    return [v.requirement, v.mode, "PASS" if v.passed else "FAIL", v.demand, v.capacity, v.unit]


@node("reason.requirement", "Requirement", category="Design",
      outputs=[Port("requirement", "requirement")],
      params=[Param("case", "enum", "roof", choices=_CASES + ["custom"], help="a worked case, or your own"),
              Param("custom", "list", [], help="[{'id', 'text', 'modes': [...], 'limits': {...}}] (case = custom)")],
      calls="pinneapple_reason.cases / pinneapple_reason.chain.Requirement", tags=_TAGS)
def requirement(ctx, inputs, params):
    """What the design must do and which failure modes would stop it."""
    from pinneapple_reason import Requirement, cases
    if params["case"] == "custom":
        reqs = [Requirement.from_dict(r) for r in params["custom"] or []]
        if not reqs:
            raise StepFailed("give at least one requirement")
        for r in reqs:
            for m in r.modes:
                if m not in MODES:
                    raise StepFailed(f"unknown failure mode {m!r}")
    else:
        reqs = cases.get(params["case"])[0]
    ctx.table("Requirements", ["id", "text", "modes"], [[r.id, r.text["en"], ", ".join(r.modes)] for r in reqs])
    return {"requirement": {"requirements": [r.to_dict() for r in reqs]}}


@node("reason.concept", "Concept", category="Design",
      outputs=[Port("concept", "concept")],
      params=[Param("case", "enum", "roof", choices=_CASES, help="the naive concept of a worked case"),
              Param("overrides", "dict", {}, help="parameter values to change (SI units)"),
              Param("elements", "list", [], help="replace the element list (empty: keep the case's)")],
      calls="pinneapple_reason.cases / pinneapple_reason.chain.Concept", tags=_TAGS)
def concept(ctx, inputs, params):
    """A version of the design: parameters, elements and where each value came from."""
    from pinneapple_reason import cases
    c = cases.get(params["case"])[1]
    if params["overrides"]:
        c = c.with_changes(dict(params["overrides"]), source="flow parameter")
        c.version = 0
    if params["elements"]:
        c.elements = [str(e) for e in params["elements"]]
    ctx.user_choice("reason.concept", "concept", c.name)
    return {"concept": c.to_dict()}


@node("reason.failure_mode", "Failure mode check", category="Design",
      inputs=[Port("concept", "concept")], outputs=[Port("metrics", "metrics")],
      params=[Param("mode", "enum", "lap_water_ingress", choices=MODES, help="failure mode of the library"),
              Param("limits", "dict", {}, help="acceptance-limit overrides")],
      calls="pinneapple_reason.library.LIBRARY.mode(name).evaluate (pinneapple_reason.modes)", tags=_TAGS)
def failure_mode(ctx, inputs, params):
    """One failure mode of the library on the concept: demand, capacity, mechanism and every measured number."""
    from pinneapple_reason.library import LIBRARY
    try:
        v = LIBRARY.mode(params["mode"]).evaluate(_concept(inputs.get("concept")), "", params["limits"] or {})
    except KeyError as exc:
        raise StepFailed(str(exc)) from exc
    ctx.summary(v.mechanism["en"])
    ctx.metric("margin", v.margin, "normalised margin, >= 0 passes")
    return {"metrics": v.to_dict()}


@node("reason.verify", "Check requirements", category="Design",
      inputs=[Port("concept", "concept"), Port("requirement", "requirement")],
      outputs=[Port("metrics", "metrics"), Port("finding", "params")],
      calls="pinneapple_reason.engine.Reasoner.verify / first_failure", tags=_TAGS)
def verify(ctx, inputs, params):
    """Every failure mode of every requirement on the concept, and the first one that fails."""
    from pinneapple_reason import Reasoner
    vs = Reasoner().verify(_concept(inputs.get("concept")), _reqs(inputs.get("requirement")))
    ctx.table("Checks", ["requirement", "mode", "status", "demand", "capacity", "unit"], [_vrow(v) for v in vs])
    f = Reasoner.first_failure(vs)
    ctx.metric("passing", sum(v.passed for v in vs))
    ctx.summary("every requirement passes" if f is None else f"{f.mode} fails: {f.proof}")
    return {"metrics": {"checks": [v.to_dict() for v in vs]}, "finding": {} if f is None else f.to_dict()}


@node("reason.countermeasure", "Countermeasure", category="Design",
      inputs=[Port("concept", "concept"), Port("requirement", "requirement")],
      outputs=[Port("concept", "concept"), Port("metrics", "metrics")],
      params=[Param("countermeasure", "enum", "auto", choices=["auto"] + COUNTERMEASURES,
                    help="auto: the engine's choice for the first failure")],
      calls="pinneapple_reason.engine.Reasoner.evaluate_options / choose", tags=_TAGS)
def countermeasure(ctx, inputs, params):
    """Try the countermeasures for the first failure, apply the chosen one (or the one you name) and report all."""
    from pinneapple_reason import Reasoner
    c, reqs = _concept(inputs.get("concept")), _reqs(inputs.get("requirement"))
    eng = Reasoner()
    vs = eng.verify(c, reqs)
    f = eng.first_failure(vs)
    if f is None:
        ctx.summary("every requirement passes: nothing to change")
        return {"concept": c.to_dict(), "metrics": {"options": []}}
    opts, trials = eng.evaluate_options(f, c, reqs, vs, {c.key()})
    name = params["countermeasure"]
    if name == "auto":
        idx, why = eng.choose(opts)
    else:
        idx = next((i for i, o in enumerate(opts) if o.countermeasure == name and o.feasible), None)
        why = {"en": "chosen in the flow", "pt": "escolhida no fluxo"}
    ctx.table("Options", ["countermeasure", "feasible", "fixes", "margin after", "cost"],
              [[o.countermeasure, o.feasible, o.fixes_target, o.target_margin_after, o.cost.get("index")]
               for o in opts])
    if idx is None:
        raise StepFailed(f"no applicable countermeasure for {f.mode}" if name == "auto" else
                         f"{name!r} cannot be applied to this failure")
    ctx.summary(f"{f.mode} fails ({f.proof}); applied {opts[idx].countermeasure}: {why['en']}")
    return {"concept": trials[idx].to_dict(),
            "metrics": {"finding": f.to_dict(), "options": [o.to_dict() for o in opts], "chosen": idx}}


@node("reason.chain", "Chain of whys", category="Design",
      inputs=[Port("requirement", "requirement"), Port("concept", "concept")],
      outputs=[Port("chain", "chain"), Port("metrics", "metrics")],
      params=[Param("max_steps", "int", 12, minimum=1, maximum=40)],
      calls="pinneapple_reason.engine.reason", tags=_TAGS)
def chain(ctx, inputs, params):
    """From the naive concept, check, find the first failure, apply the best countermeasure, check again, until every
    requirement passes."""
    from pinneapple_reason import reason
    ch = reason(_reqs(inputs.get("requirement")), _concept(inputs.get("concept")), "flow",
                max_steps=int(params["max_steps"]), progress=ctx.log)
    p = ch.save(ctx.path("chain.json"))
    ctx.table("Path", ["version", "failure found", "countermeasure"], [list(x) for x in ch.path()])
    ctx.summary(ch.explain("en")[:2000])
    return {"chain": artifact(p, "json"),
            "metrics": {"status": ch.status, "versions": len(ch.steps), "evaluations": ch.meta["evaluations"]}}


@node("reason.report", "Reasoning report", category="Design",
      inputs=[Port("chain", "chain")], outputs=[Port("report", "report"), Port("figure", "figure")],
      params=[Param("lang", "enum", "pt", choices=["pt", "en"])],
      calls="pinneapple_reason.report.write_report (timeline, margins; pinneapple_reason.trust card)", tags=_TAGS)
def report(ctx, inputs, params):
    """The chain as a report: timeline version 0 -> N and why, margins, figures, trust report and six-stage card."""
    from pinneapple_reason import Chain, write_report
    if not inputs.get("chain"):
        raise StepFailed("connect a chain")
    ch = Chain.from_dict(load_artifact(inputs["chain"]))
    paths = write_report(ch, ctx.workdir / "reason_report", lang=params["lang"])
    fig = artifact(paths["fig_timeline"], "png")
    ctx.report["figures"].append({"path": str(paths["fig_timeline"]), "name": "timeline", "caption": "v0 -> vN"})
    return {"report": artifact(paths["html"], "html"), "figure": fig}
