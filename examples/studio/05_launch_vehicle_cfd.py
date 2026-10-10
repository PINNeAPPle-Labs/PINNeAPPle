"""A launch vehicle in OpenFOAM, checked against Barrowman's stability equations, with photoreal pictures.

    python examples/studio/05_launch_vehicle_cfd.py [coarse|medium|fine] [processes]

A two-stage launcher with four tail fins (``pinneapple_design.geometry.vehicles3d.LaunchVehicle``) at 50 m/s and
4 degrees angle of attack (low-speed flight just after lift-off; incompressible), half model. The normal-force slope
and the centre of pressure come from the CFD skin forces and are compared with Barrowman's equations; the static
margin follows. Renders the vehicle on its plume under a physical sky, the skin Cp and the streamlines.
Also in the lab: ``python -m pinneapple_lab run vehicle_cfd -p vehicle=rocket``.
"""
import math
import os
import sys

import pinneapple as pp
from pinneapple_design.geometry.vehicles3d import LaunchVehicle

HERE = os.path.dirname(os.path.abspath(__file__))
res_name = sys.argv[1] if len(sys.argv) > 1 else "coarse"
procs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
out = os.path.join(HERE, "_out", "rocket")
os.makedirs(out, exist_ok=True)

rocket = LaunchVehicle()
alpha, speed = 4.0, 50.0
flow = pp.cfd.ExternalFlow(rocket.cfd_bodies(), speed=speed, alpha=alpha, half_model=True, ref_area=rocket.ref_area(),
                           resolution=res_name, iterations=600, title="Launch vehicle, 50 m/s, 4 deg")
res = flow.solve(os.path.join(out, f"case_{res_name}"), procs=procs)
S = res.surface
Fz = S["force"][:, 2]
cn_alpha = 2 * Fz.sum() / (0.5 * speed ** 2 * rocket.ref_area() * math.radians(alpha))
x_cp = (S["xyz"][:, 0] * Fz).sum() / Fz.sum()
bw = rocket.barrowman()
print(f"CN_alpha {cn_alpha:.2f} /rad (Barrowman {bw['CN_alpha']:.2f}); centre of pressure {x_cp:.2f} m from the nose "
      f"(Barrowman {bw['x_cp']:.2f} m); CD {res.coefficients['CD']:.3f}")

scene = pp.viz.Scene.from_parts(rocket.parts(plume=False), axes="z_up", title=flow.title)
scene = res.to_scene(scene)
pp.viz.web_viewer(scene, os.path.join(out, "viewer"))
try:
    import bpy  # noqa: F401
    up = pp.viz.Scene.from_parts(rocket.upright(rocket.parts()), axes="z_up")
    pp.viz.render(up, os.path.join(out, "beauty.jpg"), view=(-0.8, -1.0, -0.35), background="sky", distance=0.75,
                  sun_elevation=18, colorbar_on=False)
    pp.viz.render(scene, os.path.join(out, "cp.jpg"), field="cp", samples=64)
    pp.viz.render(scene, os.path.join(out, "streamlines.jpg"), lines=True, samples=64)
except ImportError:
    print("pip install bpy for the Blender renders; the viewer is in", os.path.join(out, "viewer"))
os._exit(0)
