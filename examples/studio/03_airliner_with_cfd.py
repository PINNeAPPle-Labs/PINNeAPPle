"""The aero app's airliner with its OpenFOAM skin pressure, through the generic studio.

    python examples/studio/03_airliner_with_cfd.py

Builds the A320-class airliner from ``pinneapple_design.aero``, maps the stored OpenFOAM 3D skin Cp onto it (the run
of apps/aero_optimizer/tools/verify3d_airliner.py) and exports glTF/USD, a viewer and a Blender render.
"""
import json
import os

import numpy as np

import pinneapple as pp
from pinneapple_design.aero.airliner3d import airliner_from, baseline_x_al

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "..", "apps", "aero_optimizer", "model")
out = os.path.join(HERE, "_out", "airliner")
os.makedirs(out, exist_ok=True)

x = baseline_x_al()
parts = airliner_from(x).build("high")
scene = pp.viz.Scene.from_parts(parts, title="A320-class airliner")                     # aircraft axes
V = json.load(open(os.path.join(MODEL, "verification3d.json")))
rec = next(d for d in V["designs"] if np.allclose(d["x"], x, atol=1e-4))
z = np.load(os.path.join(MODEL, "cfd3d", f"{rec['id']}.npz"))
aero = [p.name for p in parts if p.aero]                       # the surfaces that were in the CFD geometry
scene.map_field("cp", z["xyz"], z["cp"], mirror_y=True, surfaces=aero, label="Pressure coefficient Cp (OpenFOAM)")
scene.add_lines(list(z["lines"]) + [L * [1, -1, 1] for L in z["lines"]], list(z["line_speed"]) * 2, label="Speed |U|/U∞")
scene.save(os.path.join(out, "airliner.glb"))
pp.viz.web_viewer(scene, os.path.join(out, "viewer"))
try:
    import bpy  # noqa: F401
    pp.viz.render(scene, os.path.join(out, "cp.jpg"), field="cp", view="iso", samples=64)
except ImportError:
    print("pip install bpy for the render")
os._exit(0)
