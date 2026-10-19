"""External flow around the Ahmed body in OpenFOAM, from geometry to pictures, with PINNeAPPle.

    python examples/studio/02_ahmed_body_cfd.py [coarse|medium|fine] [processes]

The Ahmed body (Ahmed, Ramm & Faltin 1984) with a 25 degree slant, 40 m/s, a road moving with the flow, half model.
Measured drag coefficient: about 0.285. With this setup (steady RANS, k-omega SST, wall functions):
coarse 0.321 (0.15 M cells, 4 min on 2 cores), medium 0.298 (0.20 M cells, 10 min).
Needs OpenFOAM (FOAM_BASHRC pointing to its bashrc); the pictures need ``pip install bpy``.
"""
import json
import os
import sys

import pinneapple as pp

HERE = os.path.dirname(os.path.abspath(__file__))
res_name = sys.argv[1] if len(sys.argv) > 1 else "coarse"
procs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
out = os.path.join(HERE, "_out", "ahmed")
os.makedirs(out, exist_ok=True)

flow = pp.cfd.ExternalFlow({"ahmed": pp.bodies.ahmed_body(slant_deg=25)}, speed=40.0, ground=0.0, half_model=True,
                           resolution=res_name, iterations=500, title="Ahmed body, 25 deg slant")
result = flow.solve(os.path.join(out, f"case_{res_name}"), procs=procs)
c = result.coefficients
print(f"CD {c['CD']:.3f} (pressure {c['CD_pressure']:.3f} + friction {c['CD_friction']:.3f}), CL {c['CL']:.3f}, "
      f"{c['cells']:,} cells; measured CD about 0.285")
json.dump(c, open(os.path.join(out, f"coefficients_{res_name}.json"), "w"), indent=1)

scene = result.to_scene()                       # skin Cp and Cf, streamlines by speed, mid-plane and wake slices
pp.viz.web_viewer(scene, os.path.join(out, "viewer"))
try:
    import bpy  # noqa: F401
    pp.viz.render(scene, os.path.join(out, "cp.jpg"), field="cp", samples=64)
    pp.viz.render(scene, os.path.join(out, "streamlines.jpg"), lines=True, samples=64)
    pp.viz.render(scene, os.path.join(out, "mid_plane_speed.jpg"), slice="mid plane: speed", view="side", samples=48)
    pp.viz.render(scene, os.path.join(out, "wake_vorticity.jpg"), slice="wake: vorticity", view="back", samples=48)
except ImportError:
    print("pip install bpy for the Blender renders; the viewer is in", os.path.join(out, "viewer"))
os._exit(0)
