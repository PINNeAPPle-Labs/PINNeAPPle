"""Aircraft aerodynamics from the geometry library (``examples/geometry_library/03_airliner_vlm``): the vortex
lattice of the ``airliner_a320`` entry (:func:`pinneapple_design.aero.vlm.solve`) and the friction-drag build-up of
its mesh (:func:`pinneapple_design.aero.buildup.friction_drag`)."""
from __future__ import annotations

import math

from ..registry import Param, Port, node


@node("aero.airliner_vlm", "Airliner vortex lattice + friction drag (geometry library)", category="Aerodynamics",
      outputs=[Port("metrics", "metrics")],
      params=[Param("entry", "str", "airliner_a320", help="pp.geometries entry"),
              Param("mass_kg", "float", 64000.0), Param("mach", "float", 0.78),
              Param("T_K", "float", 216.65, help="ISA 11 000 m"), Param("rho", "float", 0.3639),
              Param("a_mps", "float", 295.07), Param("span_efficiency", "float", 0.85,
                                                     help="real wing against the lattice's induced drag")],
      calls="pinneapple.geometries.build + pinneapple_design.aero.airliner.Airliner + pinneapple_design.aero.vlm.solve "
            "(CL, neutral_point, trim, coefficients) + pinneapple_design.aero.buildup.friction_drag")
def airliner_vlm(ctx, inputs, params):
    """Lift slope, neutral point (wing + tail), trim at the cruise lift coefficient, induced drag; CD0 from the
    library mesh; L/D. Published A320 class: CL_alpha about 5.5-6 /rad, cruise L/D about 16-18."""
    import pinneapple as pp
    from pinneapple_design.aero.airliner import Airliner
    from pinneapple_design.aero.buildup import friction_drag
    from pinneapple_design.aero.vlm import solve
    g = pp.geometries.build(str(params["entry"]))
    p = g.params
    ac = Airliner(length=p["length"], wing_area=p["wing_area"], aspect_ratio=p["aspect_ratio"], taper=p["taper"],
                  sweep_le=p["sweep_le"], dihedral=p["dihedral"], twist=p["twist"], fan_diameter=p["fan_diameter"],
                  engine_station=p["engine_station"], sharklets=p["sharklets"])
    x_ref = ac.mac_le_x + 0.25 * ac.mac
    sol = solve(ac, x_ref)
    M, T, rho = float(params["mach"]), float(params["T_K"]), float(params["rho"])
    V = M * float(params["a_mps"])
    q = 0.5 * rho * V * V
    CL = float(params["mass_kg"]) * 9.81 / (q * ac.wing_area)
    alpha, it = sol.trim(CL)
    c = sol.coefficients(alpha, it)
    d = friction_drag(g, ac, rho=rho, V=V, T=T, mach=M)
    CD = d["CD0"] + c["CDi"] / float(params["span_efficiency"])
    ctx.metric("span_m", float(ac.span))
    ctx.metric("mac_m", float(ac.mac))
    ctx.metric("x_ref_m", float(x_ref), "quarter MAC")
    ctx.metric("CL_alpha_per_rad", float(sol.CL[1]), "published A320 class 5.5-6")
    ctx.metric("neutral_point_x_m", float(sol.neutral_point), "wing + tail only (no fuselage moment)")
    ctx.metric("CL_cruise", CL)
    ctx.metric("alpha_trim_deg", math.degrees(alpha))
    ctx.metric("tail_incidence_deg", math.degrees(it))
    ctx.metric("S_wet_m2", float(d["S_wet"]))
    ctx.metric("CD0", float(d["CD0"]))
    ctx.metric("CDi", float(c["CDi"]))
    ctx.metric("CD", float(CD))
    ctx.metric("L_over_D", CL / CD, "published A320 class 16-18")
    ctx.table("zero-lift drag by component", ["component", "CD0"],
              [[k, round(v, 5)] for k, v in sorted(d["components"].items(), key=lambda kv: -kv[1])])
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="aero.airliner_vlm", title="Airliner lift slope, trim and drag by vortex lattice", domain="aerodynamics",
        example="geometry_library", keywords={"vortex lattice": 3, "vlm": 3, "airliner": 1.5, "a320": 2,
                                              "lift slope": 2, "neutral point": 2, "induced drag": 2, "l/d": 1},
        min_score=3, nodes=("aero.airliner_vlm",), references=("raymer2018aircraft",),
        template="geometry_library/03_airliner_vlm", description="examples/geometry_library/03_airliner_vlm.py"))


_register_capabilities()
