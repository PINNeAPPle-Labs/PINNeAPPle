"""Reactor sizing as a closed loop (:mod:`pinneapple_physics.reactors.design`, #1179): the reactor model (mole, energy
and, for a packed bed, momentum balances), a learned model only when the measured cost says it pays off, the
smallest reactor that meets the conversion, selectivity and hot-spot limits, and the design re-verified by the model
with its safety margins (Semenov, parametric sensitivity, stable steady state)."""
from __future__ import annotations

from ..registry import Param, Port, node

CASES = ("phthalic_anhydride_bed", "propylene_glycol_cstr")


def spec_for(case: str, params: dict):
    from pinneapple_physics.reactors import presets
    from pinneapple_physics.reactors.design import ReactorSpec
    if case == "phthalic_anhydride_bed":
        b = presets.phthalic_anhydride_bed(float(params.get("p_xylene_atm", 0.011)))
        V0 = b["V"]
        return ReactorSpec(b["network"], b["feed"], b["key"], float(params["X_target"]),
                          {"V": (0.3 * V0, 3 * V0), "Tc": (610.0, 650.0)}, cooling=b["cooling"],
                          desired=b["desired"], T_limit=float(params["T_limit"]),
                          margin_min=float(params["margin_min"]), assumptions=b["assumptions"], source=b["source"])
    c = presets.propylene_glycol_cstr()
    return ReactorSpec(c["network"], c["feed"], c["key"], float(params["X_target"]),
                      {"V": (0.2, 5.0), "UA": (0.0, 3e4)}, reactor="cstr", cooling=c["cooling"],
                      desired=c["desired"], T_limit=float(params["T_limit"]), tanks=int(params.get("tanks", 1)),
                      assumptions=["liquid of constant volumetric flow (density change neglected)",
                                   c["note"]], source=c["source"])


@node("reactor.design", "Reactor sizing (closed loop, re-verified)", category="Design",
      outputs=[Port("metrics", "metrics")],
      params=[Param("case", "enum", CASES[0], choices=list(CASES),
                    help="cooled multitubular packed bed (o-xylene oxidation) or cooled CSTR (propylene glycol)"),
              Param("X_target", "float", 0.75, minimum=0.01, maximum=0.999, help="required conversion of the key "
                                                                                  "reactant"),
              Param("T_limit", "float", 673.15, minimum=200.0, help="highest temperature allowed anywhere, K (your "
                                                                     "catalyst's or product's limit)"),
              Param("margin_min", "float", 10.0, minimum=0.0, help="coolant-temperature margin to the runaway "
                                                                    "boundary, K (packed bed)"),
              Param("p_xylene_atm", "float", 0.011, minimum=1e-4, maximum=0.05, advanced=True),
              Param("tanks", "int", 1, minimum=1, maximum=10, advanced=True, help="CSTRs in series"),
              Param("force_surrogate", "bool", False, advanced=True,
                    help="train and use the Gaussian process even when the model is cheap enough (to compare)"),
              Param("maxiter", "int", 60, minimum=5), Param("popsize", "int", 12, minimum=4),
              Param("seed", "int", 0)],
      calls="pinneapple_physics.reactors.design.design_reactor (models.pfr / cstr_series, sizing.*, "
            "architecture_recommendation.recommend_architecture)")
def reactor_design(ctx, inputs, params):
    """Size the reactor, then verify the design with the model."""
    from pinneapple_physics.reactors.design import design_reactor, to_json
    spec = spec_for(str(params["case"]), params)
    r = to_json(design_reactor(spec, maxiter=int(params["maxiter"]), popsize=int(params["popsize"]),
                               seed=int(params["seed"]), force_surrogate=bool(params["force_surrogate"])))
    for k, v in r["design"].items():
        ctx.metric(f"design.{k}", v, "optimum, re-solved by the model")
    for k, v in r["outputs"].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            ctx.metric(f"model.{k}", v, "reactor model at the design")
    ctx.metric("recommendation", r["recommendation"]["recommended"][0],
               (r["recommendation"].get("decision_path") or [r["recommendation"]["reasoning"]])[0])
    ctx.metric("cost.solve_seconds", r["cost"]["solve_seconds"], "measured: mean of three solves")
    sens = r["safety"].get("sensitivity") or {}
    if sens.get("margin_K") is not None:
        ctx.metric("safety.runaway_margin_K", sens["margin_K"], "Morbidelli-Varma boundary minus the coolant")
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.table("assumptions", ["assumption"], [[a] for a in r["assumptions"]])
    bad = [c["code"] for c in r["checks"] if c["status"] == "fail"]
    ctx.metric("verdict", "fail" if bad else "pass", ", ".join(bad) or "every check passes or is inconclusive")
    ctx.summary(f"{params['case']}: V = {r['design']['V']:.4g} m^3, X = {r['outputs']['X']:.3f}, "
                f"T_max = {r['outputs']['T_max']:.1f} K, " + ("fail: " + ", ".join(bad) if bad else "verified"))
    return {"metrics": ctx.report["metrics"]}


def build_flow(case: str = CASES[0], **params):
    """The reactor-sizing flow: the design node and its report."""
    from ..planner.build import FlowBuilder
    title = {"phthalic_anhydride_bed": "Reactor sizing: cooled packed bed (o-xylene to phthalic anhydride)",
             "propylene_glycol_cstr": "Reactor sizing: cooled CSTR (propylene glycol)"}[case]
    fb = FlowBuilder(title, "The smallest reactor meeting the conversion, hot-spot and runaway limits: the reactor "
                            "model, a learned model only when its measured cost pays off, optimisation, and the "
                            "design re-verified by the model (pinneapple_physics.reactors.design).")
    fb.add("design", "reactor.design", {"case": case, **params}, label="Reactor sizing")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("design", "metrics", "report", "items")
    return fb


def _build(text: str, _det):
    t = text.lower()
    case = "propylene_glycol_cstr" if any(w in t for w in ("cstr", "stirred", "tanque", "agitado", "liquid",
                                                           "líquid")) else "phthalic_anhydride_bed"
    return build_flow(case)


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="reactor.sizing", title="Reactor sizing with safety margins (closed loop)", domain="chemical reactors",
        keywords={"reactor": 2, "reator": 2, "reactor sizing": 3, "dimensionar": 1.5, "dimensionamento": 1.5,
                  "conversion": 1, "conversão": 1, "runaway": 2, "hot spot": 1.5, "packed bed": 2,
                  "leito fixo": 2, "cstr": 2, "pfr": 2, "plug flow": 2,
                  # the goal verbs: "size a reactor", "design a reactor", "projetar um reator" reach 3 with the noun
                  "size": 1.5, "sizing": 1.5, "design": 1, "projetar": 1, "otimiz": 1, "optimi": 1},
        requires_all=(("reactor", "reator", "cstr", "pfr", "packed bed", "leito fixo", "plug flow"),),
        min_score=3, nodes=("reactor.design",), build=_build,
        inputs=("the reaction kinetics and heats of reaction", "the feed", "the conversion target and the "
                                                                          "temperature limit"),
        uncited="the method cites Fogler (2016), Levenspiel (1999), Froment & Bischoff (2011) and Morbidelli & "
                "Varma (1988) in pinneapple_physics.reactors; they are not in the flow reference file yet",
        description="pinneapple_physics.reactors.design.design_reactor"))


_register_capabilities()
