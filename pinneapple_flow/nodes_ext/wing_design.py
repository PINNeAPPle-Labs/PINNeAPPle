"""Wing design as a closed loop (:mod:`pinneapple_design.aero.wing_design`, #1181): the vortex lattice and a friction
drag build-up, a learned model only when its measured cost pays off, the planform with the least cruise drag within
the root-bending, stall and span limits, and the design re-verified (finer lattice, Munk's bound, lifting line)."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("aero.wing_design", "Wing planform design (closed loop, re-verified)", category="Design",
      outputs=[Port("metrics", "metrics")],
      params=[Param("weight_N", "float", 10_500.0, minimum=1.0), Param("area_m2", "float", 16.2, minimum=0.1),
              Param("cruise_V", "float", 62.0, minimum=1.0), Param("cruise_rho", "float", 1.007, minimum=0.01),
              Param("landing_V", "float", 31.0, minimum=1.0),
              Param("cl_max", "float", 1.6, minimum=0.3, maximum=3.0, help="section cl_max (measured)"),
              Param("n_load", "float", 3.8, minimum=1.0, help="limit load factor"),
              Param("M_root_max", "float", 60_000.0, minimum=0.0, help="half-wing root bending limit, N m"),
              Param("span_max", "float", 12.0, minimum=0.5, help="largest span allowed, m"),
              Param("maxiter", "int", 30, minimum=3), Param("popsize", "int", 10, minimum=4), Param("seed", "int", 0)],
      calls="pinneapple_design.aero.wing_design.design_wing (vlm.solve, lifting_line, "
            "architecture_recommendation.recommend_architecture)")
def wing_design(ctx, inputs, params):
    """Optimise the planform and verify the design."""
    from pinneapple_design.aero.wing_design import WingSpec, design_wing
    keys = ("weight_N", "area_m2", "cruise_V", "cruise_rho", "landing_V", "cl_max", "n_load", "M_root_max",
            "span_max")
    spec = WingSpec(**{k: float(params[k]) for k in keys})
    r = design_wing(spec, maxiter=int(params["maxiter"]), popsize=int(params["popsize"]), seed=int(params["seed"]))
    for k, v in r["design"].items():
        ctx.metric(f"design.{k}", v, "optimum, re-solved by the lattice")
    for k in ("drag_N", "CDi", "CD0", "e", "M_root_Nm", "span_m", "cl_local_max_landing", "stall_station"):
        ctx.metric(f"model.{k}", r["outputs"][k], "vortex lattice + friction build-up at the design")
    ctx.metric("baseline.drag_N", r["baseline"]["drag_N"], "the library's light aircraft wing (AR 7.5, taper 0.7, "
                                                           "-3 deg)")
    ctx.metric("recommendation", r["recommendation"]["recommended"][0],
               (r["recommendation"].get("decision_path") or [r["recommendation"]["reasoning"]])[0])
    ctx.metric("cost.solve_seconds", r["cost"]["solve_seconds"], "measured: mean of three lattice solves")
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.table("assumptions", ["assumption"], [[a] for a in r["assumptions"]])
    bad = [c["code"] for c in r["checks"] if c["status"] == "fail"]
    ctx.metric("verdict", "fail" if bad else "pass", ", ".join(bad) or "every check passes or is inconclusive")
    ctx.summary(f"AR {r['design']['aspect_ratio']:.2f}, taper {r['design']['taper']:.2f}, twist "
                f"{r['design']['twist']:.1f} deg: drag {r['outputs']['drag_N']:.0f} N (baseline "
                f"{r['baseline']['drag_N']:.0f} N); " + ("fail: " + ", ".join(bad) if bad else "verified"))
    return {"metrics": ctx.report["metrics"]}


def build_flow(**params):
    from ..planner.build import FlowBuilder
    title = "Wing design: least cruise drag within the structural and stall limits"
    fb = FlowBuilder(title, "Vortex lattice and friction build-up, a learned model only when its measured cost pays "
                            "off, the planform optimised for cruise drag within the root-bending, stall and span "
                            "limits, and the design re-verified (pinneapple_design.aero.wing_design).")
    fb.add("wing", "aero.wing_design", dict(params), label="Wing design")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("wing", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="aero.wing_design", title="Wing planform design (closed loop)", domain="aerospace",
        keywords={"wing": 2, "asa": 1.5, "planform": 3, "aspect ratio": 2, "alongamento": 2, "induced drag": 2,
                  "arrasto induzido": 2, "taper": 1.5, "afilamento": 1.5, "washout": 2, "torção": 1,
                  "aircraft": 1, "aeronave": 1, "avião": 1,
                  # the goal verbs: "otimizar a asa", "optimise the wing", "design the wing" reach 3 with the noun
                  "otimiz": 1.5, "optimi": 1.5, "projetar": 1, "projeto da asa": 1.5, "design": 1,
                  "dimensionar": 1, "size the wing": 1},
        requires_all=(("wing", "asa", "planform", "aspect ratio", "alongamento"),), min_score=3,
        nodes=("aero.wing_design",), build=lambda text, det: build_flow(),
        inputs=("the aircraft weight and wing area", "the cruise and landing speeds", "the section's cl_max",
                "the structural and span limits"),
        uncited="the method cites Raymer (2018) and Munk / Prandtl in pinneapple_design.aero.wing_design; they are "
                "not in the flow reference file yet",
        description="pinneapple_design.aero.wing_design.design_wing"))


_register_capabilities()
