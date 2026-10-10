"""A parametric road car in OpenFOAM, from geometry to photoreal pictures, with PINNeAPPle.

    python examples/studio/04_road_car_cfd.py [coarse|medium|fine] [processes] [fastback|notchback]

A coupé (``pinneapple_design.geometry.vehicles3d.RoadCar``: body with greenhouse, lights and wheels) at 30 m/s over a
road moving with the flow, half model, steady RANS (k-omega SST, wall functions). Prints the drag and lift
coefficients on the frontal area, writes the 3-D viewer and renders a beauty shot, the skin Cp, the streamlines and
the wake. The same run is in the lab as ``vehicle_cfd`` (``python -m pinneapple_lab run vehicle_cfd -p vehicle=car``).
Needs OpenFOAM (FOAM_BASHRC pointing to its bashrc); the pictures need ``pip install bpy``.
"""
import json
import os
import sys

import pinneapple as pp
from pinneapple_design.geometry.vehicles3d import RoadCar, road

HERE = os.path.dirname(os.path.abspath(__file__))
res_name = sys.argv[1] if len(sys.argv) > 1 else "coarse"
procs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
style = sys.argv[3] if len(sys.argv) > 3 else "fastback"
out = os.path.join(HERE, "_out", f"car_{style}")
os.makedirs(out, exist_ok=True)

car = RoadCar(style=style)
flow = pp.cfd.ExternalFlow(car.cfd_bodies(), speed=30.0, ground=0.0, half_model=True, ref_area=car.frontal_area(),
                           resolution=res_name, iterations=600, title=f"Road car ({style}), 30 m/s")
result = flow.solve(os.path.join(out, f"case_{res_name}"), procs=procs)
c = result.coefficients
print(f"CD {c['CD']:.3f} (pressure {c['CD_pressure']:.3f} + friction {c['CD_friction']:.3f}), CL {c['CL']:.3f}, "
      f"frontal area {flow.ref_area:.2f} m2, {c['cells']:,} cells")
json.dump(c, open(os.path.join(out, f"coefficients_{res_name}.json"), "w"), indent=1)

scene = pp.viz.Scene.from_parts(car.parts(), axes="z_up", title=flow.title)
scene = result.to_scene(scene)                  # skin Cp and Cf on the car, streamlines, mid-plane and wake slices
pp.viz.web_viewer(scene, os.path.join(out, "viewer"))
try:
    import bpy  # noqa: F401
    beauty = pp.viz.Scene.from_parts(car.parts() + [road()], axes="z_up")
    pp.viz.render(beauty, os.path.join(out, "beauty.jpg"), view=(-1.0, -1.25, 0.28), background="sky",
                  distance=0.62, sun_elevation=24, colorbar_on=False)
    pp.viz.render(scene, os.path.join(out, "cp.jpg"), field="cp", samples=64)
    pp.viz.render(scene, os.path.join(out, "streamlines.jpg"), lines=True, samples=64)
    pp.viz.render(scene, os.path.join(out, "wake_vorticity.jpg"), slice="wake: vorticity", view="back", samples=48)
except ImportError:
    print("pip install bpy for the Blender renders; the viewer is in", os.path.join(out, "viewer"))
os._exit(0)
