"""Viscous vortex domains nodes: the two cases of :mod:`pinneapple_simulation.vortex_domains`, run as flow steps.

``vortex.cylinder`` is the verification case -- a cylinder whose drag at ``Re = 40`` is compared with Dennis & Chang
(``Cd = 1.3375``) and whose shedding at ``Re = 200`` with Williamson & Brown's correlation. ``vortex.spoiler`` is the
reference case: a NACA 0012 at zero incidence with a spoiler that deploys to 90 degrees, reporting the lift, the drag
and the radius of the vorticity handed to the fluid at the wall.
"""
from __future__ import annotations

import time

import numpy as np

from ..registry import Param, Port, node


@node("vortex.cylinder", "Cylinder in a stream (viscous vortex domains)", category="Viscous vortex domains",
      outputs=[Port("metrics", "metrics")],
      params=[Param("re", "float", 40.0, minimum=1.0, help="Reynolds number on the diameter"),
              Param("panels", "int", 64, minimum=16, maximum=512, help="panels around the cylinder"),
              Param("t_end", "float", 2.0, minimum=0.01, help="end time (diameters travelled by the free stream)")],
      calls="pinneapple_simulation.vortex_domains.study.run_study + cases.cylinder_case")
def cylinder(ctx, inputs, params):
    """Run the cylinder and report the mean coefficients over the last third, plus the physics checks."""
    from pinneapple_simulation.vortex_domains.cases import cylinder_case
    from pinneapple_simulation.vortex_domains.study import run_study

    t0 = time.perf_counter()
    result = run_study(cylinder_case(re=float(params["re"]), panels=int(params["panels"]),
                                     t_end=float(params["t_end"])))
    c = result.coefficients
    ctx.metric("Re", float(params["re"]))
    ctx.metric("Cd", c["cd"], "mean over the last third of the run")
    ctx.metric("Cl", c["cl"], "mean over the last third of the run")
    if "strouhal" in c:
        ctx.metric("St", c["strouhal"], "dominant frequency of the lift times D / U")
    ctx.metric("Cd_reference", 1.3375, "Dennis & Chang 1970, Re = 40")
    ctx.metric("wall_s", round(time.perf_counter() - t0, 1))
    for check in result.checks:
        ctx.metric(f"check_{check['name']}", 1.0 if check["ok"] else 0.0, check["detail"])
    ctx.summary(result.summary())
    return {"metrics": ctx.report["metrics"]}


@node("vortex.spoiler", "NACA 0012 with a deploying spoiler (viscous vortex domains)",
      category="Viscous vortex domains", outputs=[Port("metrics", "metrics")],
      params=[Param("re", "float", 40000.0, minimum=1.0),
              Param("panels", "int", 120, minimum=40, maximum=400, help="panels on the section"),
              Param("t_end", "float", 0.6, minimum=0.05),
              Param("deploy_angle_deg", "float", 90.0, minimum=0.0, maximum=90.0),
              Param("deploy_start", "float", 0.4, minimum=0.0)],
      calls="pinneapple_simulation.vortex_domains.cases.naca0012_spoiler_case + solver.VVD2D")
def spoiler(ctx, inputs, params):
    """Deploy the spoiler and report how the loads change: mean lift and drag before and after it stands up."""
    from pinneapple_simulation.vortex_domains.cases import naca0012_spoiler_case

    case = naca0012_spoiler_case(re=float(params["re"]), panels=int(params["panels"]),
                                 spoiler_panels=16, deploy_angle_deg=float(params["deploy_angle_deg"]),
                                 t_start=float(params["deploy_start"]), deploy_time=0.5,
                                 t_end=float(params["t_end"]))
    sim = case.build()
    before: list[tuple[float, float, float]] = []

    def sample(s) -> None:
        c = s.coefficients
        before.append((s.time, c["cl"], c["cd"]))

    sim.run(progress=sample)
    history = np.asarray(before)
    t, cl, cd = history[:, 0], history[:, 1], history[:, 2]
    early = t < float(params["deploy_start"])
    late = t > float(params["deploy_start"]) + 0.6
    ctx.metric("Cl_before", float(np.mean(cl[early])) if np.any(early) else float("nan"))
    ctx.metric("Cl_after", float(np.mean(cl[late])) if np.any(late) else float("nan"))
    ctx.metric("Cd_before", float(np.mean(cd[early])) if np.any(early) else float("nan"))
    ctx.metric("Cd_after", float(np.mean(cd[late])) if np.any(late) else float("nan"))
    ctx.metric("domains", float(len(sim.domains)), "vortex domains in the wake at the end")
    ctx.metric("circulation_shed", float(sim.shed), "circulation handed to the fluid at the walls")
    ctx.summary(f"Cl {ctx.report['metrics']['Cl_before']['value']:+.2f} -> "
                f"{ctx.report['metrics']['Cl_after']['value']:+.2f}, "
                f"Cd {ctx.report['metrics']['Cd_before']['value']:+.2f} -> "
                f"{ctx.report['metrics']['Cd_after']['value']:+.2f} at Re = {float(params['re']):g}")
    return {"metrics": ctx.report["metrics"]}
