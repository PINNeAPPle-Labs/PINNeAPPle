"""A 3-D aircraft aerodynamics surrogate from a plain-text request, graded by VeriPhysics, shown where the grade
rejects it and where it accepts it (:mod:`pinneapple_design.aero.aircraft_surrogate_demo`, #1438)."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("aero.aircraft_surrogate", "3-D aircraft aerodynamics surrogate (from text, graded)", category="Design",
      outputs=[Port("metrics", "metrics")],
      params=[Param("request", "text", "", max_length=2000,
                    help="the problem in words: planform values and angle of attack; no geometry needed"),
              Param("rounds", "int", 3, minimum=1, maximum=3,
                    help="1 quick model, 2 full model on the library's design space, 3 design space extended"),
              Param("cinematic", "enum", "off", choices=["auto", "on", "off"],
                    help="Cycles renders with streamlines (needs Blender's bpy). Off by default: measured in the "
                         "engineering app, the three rounds took 186 s without the renders and 1160 s with them, "
                         "beyond a 900 s job limit"),
              Param("orbit_frames", "int", 28, minimum=2, maximum=120), Param("map_n", "int", 8, minimum=3,
                                                                                maximum=16),
              Param("seed", "int", 0)],
      calls="pinneapple_design.aero.aircraft_surrogate_demo.aircraft_surrogate_demo "
            "(aircraft_surrogate_study, vlm.solve, pinneapple_veriphysics.policy.decide)")
def aircraft_surrogate(ctx, inputs, params):
    """Parse the request, train and grade three rounds, map where each model accepts a request, render."""
    from pinneapple_design.aero.aircraft_surrogate_demo import aircraft_surrogate_demo
    r = aircraft_surrogate_demo(params.get("request") or "", out_dir=str(ctx.workdir), rounds=int(params["rounds"]),
                                cinematic=str(params["cinematic"]), orbit_frames=int(params["orbit_frames"]),
                                map_n=int(params["map_n"]), seed=int(params["seed"]))
    req = r["request"]
    for k, v in req["design"].items():
        ctx.metric(f"request.{k}", v, "default (not in the text)" if k in req["defaulted"] else "read from the text")
    for rr in r["rounds"]:
        k = rr["round"]
        ctx.metric(f"round{k}.model_verdict", rr["model_verdict"], "VeriPhysics grade (policy.decide) of the model")
        ctx.metric(f"round{k}.request_verdict", rr["request_verdict"],
                   "outside the validated design space" if rr["request_flagged"] else "inside the validated space")
        ctx.metric(f"round{k}.field_error_held_out", rr["field_error_held_out"], "relative L2, held-out designs")
        ctx.metric(f"round{k}.request_field_error", rr["request_field_error"], "relative L2 against the solver")
    o = r["outputs"]
    ctx.metric("request_verdict", o["request_verdict"], "the last round's answer to the request")
    ctx.metric("cost.total_seconds", o["total_seconds"], "measured")
    for f in r["files"]:
        if f.endswith(".png"):
            ctx.report["figures"].append({"path": str(ctx.path(f)), "name": f[:-4], "caption": f})
    ctx.metric("files", ", ".join(r["files"]), "written in this node's folder")
    ctx.table("checks", ["check", "status", "detail"], [[c["code"], c["status"], c["detail"]] for c in r["checks"]])
    ctx.table("assumptions", ["assumption"], [[a] for a in r["assumptions"]])
    ctx.summary(" → ".join(f"round {rr['round']}: model {rr['model_verdict']}, request {rr['request_verdict']}"
                           for rr in r["rounds"])
                + f"; request error against the solver {o['request_field_error']:.4f}")
    return {"metrics": ctx.report["metrics"]}


def build_flow(text: str = "", **params):
    from ..planner.build import FlowBuilder
    title = "3-D aircraft aerodynamics surrogate, graded by VeriPhysics"
    fb = FlowBuilder(title, "From the request in words: the planform is read from the text, the vortex lattice gives "
                            "the training data, the surrogate is graded by VeriPhysics, rejected and then accepted "
                            "once its design space covers the request (pinneapple_design.aero.aircraft_surrogate_demo).")
    fb.add("surrogate", "aero.aircraft_surrogate", dict(params, request=text), label="Aircraft surrogate")
    fb.add("report", "export.report", {"title": title}, label="Report")
    fb.connect("surrogate", "metrics", "report", "items")
    return fb


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="aero.aircraft_surrogate", title="3-D aircraft aerodynamics surrogate (graded)", domain="aerospace",
        keywords={"surrogate": 2, "modelo substituto": 2, "metamodel": 1.5, "aerodinâmica": 1.5,
                  "aerodynamic": 1.5, "aeronave": 1.5, "aircraft": 1.5, "avião": 1.5, "airplane": 1.5, "3d": 1,
                  "3-d": 1, "treinar": 1, "train": 1, "veriphysics": 1.5, "campo de pressão": 1,
                  "pressure field": 1, "asa": 0.5, "wing": 0.5},
        requires_all=(("surrogate", "modelo substituto", "metamodel"),
                      ("aeronave", "aircraft", "avião", "airplane", "aerodinâmica", "aerodynamic")),
        min_score=4, nodes=("aero.aircraft_surrogate",), build=lambda text, det: build_flow(text),
        inputs=("the planform in words (wing area, aspect ratio, taper, sweep, twist)", "the angle of attack"),
        uncited="pinneapple_design.aero.vlm (the vortex lattice) and the surrogate carry no literature reference "
                "in the flow reference file yet",
        description="pinneapple_design.aero.aircraft_surrogate_demo.aircraft_surrogate_demo"))


_register_capabilities()
