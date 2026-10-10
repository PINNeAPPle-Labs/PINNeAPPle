"""3-D vehicles in OpenFOAM with photoreal renders: a road car and a launch vehicle.

``vehicle_cfd`` builds the vehicle (``pinneapple_design.geometry.vehicles3d``), runs steady RANS around it with the
generic ``ExternalFlow`` (snappyHexMesh, simpleFoam k-omega SST, wall functions; half model), and keeps the
coefficients, the skin pressure and friction, streamlines and wake planes. It renders the vehicle with Blender Cycles
(a beauty shot under a physical sky, the skin Cp, the streamlines coloured by speed, the wake vorticity) and writes
the three.js viewer, so the run can be browsed in 3-D from the lab server.

* ``car``    -- a coupé at 30 m/s over a road moving with the flow, wheels on the road. Checked: drag coefficient
               in the range of passenger cars with exposed, non-rotating wheels; pressure drag dominant.
* ``rocket`` -- a two-stage launcher with four tail fins at 4 degrees angle of attack (low-speed flight right after
               lift-off, incompressible). Checked against Barrowman's equations: normal-force slope and the centre
               of pressure from the CFD skin forces; the static margin.

    python -m pinneapple_lab run vehicle_cfd -p vehicle=car
    python -m pinneapple_lab run vehicle_cfd -p vehicle=rocket
"""
from __future__ import annotations

import math
import os
import shutil

import numpy as np

from ..spec import Experiment, register


def _parts_and_flow(p):
    from pinneapple_design.geometry.vehicles3d import LaunchVehicle, RoadCar
    from pinneapple_simulation.numerical_solvers.external_flow import ExternalFlow
    if p["vehicle"] == "car":
        car = RoadCar(style=p["style"], slant_deg=p["slant_deg"])
        flow = ExternalFlow(car.cfd_bodies(), speed=p["speed"], ground=0.0, ground_moving=True, half_model=True,
                            ref_area=car.frontal_area(), resolution=p["resolution"], iterations=p["iterations"],
                            title=f"Road car ({car.style}), {p['speed']:g} m/s")
        return car, car.parts(), flow
    rocket = LaunchVehicle()
    flow = ExternalFlow(rocket.cfd_bodies(), speed=p["speed"], alpha=p["alpha"], half_model=True,
                        ref_area=rocket.ref_area(), resolution=p["resolution"], iterations=p["iterations"],
                        surface_level=tuple(p["surface_level"]) if p.get("surface_level") else None,
                        title=f"Launch vehicle, {p['speed']:g} m/s, {p['alpha']:g} deg")
    return rocket, rocket.parts(plume=False), flow


@register
class VehicleCFD(Experiment):
    name = "vehicle_cfd"
    version = "1"
    description = ("A parametric 3-D road car or launch vehicle in OpenFOAM (snappyHexMesh + simpleFoam, half model): "
                   "drag, lift, skin pressure, streamlines and wake, checked against reference ranges (car) or "
                   "Barrowman's stability equations (rocket); Blender Cycles renders and a 3-D viewer.")
    tags = ["automotive", "aerospace", "cfd", "openfoam", "3d", "render"]
    case_param = "vehicle"
    limitations = {
        "car": ["steady RANS on a coarse mesh (no grid study yet)", "non-rotating wheels, no underbody or cooling-flow "
                "detail", "a parametric body, not a production car"],
        "rocket": ["incompressible at 50 m/s: no Mach effects (transonic / supersonic flight)",
                   "the plume is only drawn, not simulated"],
    }
    params = {"vehicle": "car", "style": "fastback", "slant_deg": 22.0, "speed": 30.0, "alpha": 4.0,
              "resolution": "coarse", "surface_level": None, "iterations": 600, "procs": 2, "samples": 96,
              "keep_case": False}
    space = {"vehicle": ["car", "rocket"], "style": ["fastback", "notchback"], "slant_deg": (12.0, 35.0)}

    def run(self, ctx):
        import pinneapple as pp
        p = dict(ctx.params)
        if p["vehicle"] == "rocket" and p["speed"] == 30.0:
            p["speed"] = 50.0
        veh, parts, flow = _parts_and_flow(p)
        ctx.input("vehicle", {k: v for k, v in veh.__dict__.items() if isinstance(v, (int, float, str))})
        lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ctx.dir))))
        case = os.path.join(lab_root, "_cases", os.path.basename(ctx.dir))
        with ctx.stage("cfd"):
            res = flow.solve(case, procs=int(p["procs"]), log=ctx.log)
        c = res.coefficients
        for k in ("CD", "CL", "CD_pressure", "CD_friction", "drag_N", "lift_N", "cells"):
            ctx.metric(k, c[k])
        ctx.output("coefficients", c)
        ctx.metric("Re", res.info["Re"])
        S = res.surface

        if p["vehicle"] == "car":
            ctx.metric("frontal_area_m2", flow.ref_area)
            ctx.check("drag_in_passenger_car_range", value=c["CD"], min=0.22, max=0.6,
                      detail="0.25-0.40 for production cars; exposed, non-rotating wheels and no underbody "
                             "detail add drag")
            ctx.check("pressure_drag_dominates", value=c["CD_pressure"] / c["CD"], min=0.7)
        else:
            alpha = math.radians(p["alpha"])
            q = 0.5 * p["speed"] ** 2
            Fz = S["force"][:, 2]
            N = 2 * Fz.sum()                                      # half model: both halves push the same way
            cn_alpha = N / (q * veh.ref_area() * alpha)
            x_cp = float((S["xyz"][:, 0] * Fz).sum() / Fz.sum())
            bw = veh.barrowman()
            ctx.output("barrowman", bw)
            ctx.metric("CN_alpha_cfd", cn_alpha)
            ctx.metric("CN_alpha_barrowman", bw["CN_alpha"])
            ctx.metric("x_cp_cfd_m", x_cp)
            ctx.metric("x_cp_barrowman_m", bw["x_cp"])
            x_cg = 0.6 * veh.length                                # a loaded launcher: CG about 60 % from the nose
            ctx.metric("static_margin_calibers_cfd", (x_cp - x_cg) / veh.diameter)
            ctx.metric("static_margin_calibers_barrowman", (bw["x_cp"] - x_cg) / veh.diameter)
            ctx.check("cp_location_vs_barrowman", value=abs(x_cp - bw["x_cp"]) / veh.length, max=0.08,
                      detail="|x_cp CFD - x_cp Barrowman| / length")
            ctx.check("normal_force_slope_vs_barrowman", value=cn_alpha, reference=bw["CN_alpha"], rtol=0.35)
            ctx.check("statically_stable", value=x_cp - x_cg, min=0.0, detail="centre of pressure behind the CG")

        with ctx.stage("datasets"):
            ds = ctx.dataset("surface", description="Skin pressure and friction coefficients on the wall faces "
                             "(half model, y >= 0) with the force per face", units={"xyz": "m"})
            ds.add(xyz=S["xyz"].astype(np.float32), cp=S["cp"].astype(np.float32), cf=S["cf"].astype(np.float32),
                   force=S["force"].astype(np.float32), vehicle=p["vehicle"], CD=c["CD"], CL=c["CL"])
            ln = ctx.dataset("streamlines", description="Streamlines (points) and the speed along them, |U|/U")
            for L, sp in zip(res.lines, res.line_speed, strict=True):
                ln.add(points=np.asarray(L, np.float32), speed=np.asarray(sp, np.float32))

        with ctx.stage("render"):
            from pinneapple_design.geometry.vehicles3d import road
            sc = pp.viz.Scene.from_parts(parts, axes="z_up", title=flow.title)
            aero = [s.name for s in sc.surfaces if s.name.startswith(("body", "wheel", "rocket", "fin")) and
                    not s.name.endswith("_rim")]
            sc.map_field("cp", S["xyz"], S["cp"], mirror_y=True, surfaces=aero, label="Pressure coefficient Cp (OpenFOAM)")
            full = res.to_scene(sc)                                # adds lines and slices to the same scene
            glb = os.path.join(ctx.dir, "outputs", "scene.glb")
            os.makedirs(os.path.dirname(glb), exist_ok=True)
            full.save(glb)
            pp.viz.web_viewer(full, os.path.join(ctx.dir, "viewer"))
            n = int(p["samples"])
            renders = []
            if p["vehicle"] == "car":
                beauty = pp.viz.Scene.from_parts(parts + [road()], axes="z_up", title=flow.title)
                renders.append(("beauty", dict(scene=beauty, view=(-1.0, -1.25, 0.28), background="sky", distance=0.62,
                                               sun_elevation=24.0, colorbar_on=False)))
                renders.append(("beauty_rear", dict(scene=beauty, view=(1.0, -1.1, 0.3), background="sky",
                                                    distance=0.66, sun_elevation=24.0, colorbar_on=False)))
            else:
                up = veh.upright(veh.parts())
                beauty = pp.viz.Scene.from_parts(up, axes="z_up", title=flow.title)
                renders.append(("beauty", dict(scene=beauty, view=(-0.8, -1.0, -0.35), background="sky", distance=0.5,
                                               sun_elevation=18.0, colorbar_on=False, size=(1000, 1500))))
                renders.append(("beauty_low", dict(scene=beauty, view=(-0.5, -1.0, -0.75), background="sky",
                                                   distance=0.42, lens=35, sun_elevation=30.0, colorbar_on=False,
                                                   size=(1000, 1500))))
            renders += [("cp", dict(scene=full, field="cp", view="iso", background="studio")),
                        ("streamlines", dict(scene=full, lines=True, view="iso", background="studio")),
                        ("wake_vorticity", dict(scene=full, slice="wake: vorticity", view="back", background="studio")),
                        ("mid_plane_speed", dict(scene=full, slice="mid plane: speed", view="side", background="studio"))]
            for name, kw in renders:
                out = os.path.join(ctx.dir, "figures", f"{name}.jpg")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                try:
                    pp.viz.render(kw.pop("scene"), out, samples=n if name.startswith("beauty") else max(32, n // 2),
                                  **kw)
                    ctx.files["figures"].append(os.path.relpath(out, ctx.dir))
                except Exception as exc:                          # noqa: BLE001 - a missing render is logged
                    ctx.log(f"render {name} failed: {exc}")
        if not p["keep_case"]:
            shutil.rmtree(case, ignore_errors=True)
