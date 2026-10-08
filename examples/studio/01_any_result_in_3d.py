"""Any simulation result in 3D, with three calls: load, export, render.

    python examples/studio/01_any_result_in_3d.py [result file]      (default: a CalculiX cantilever)

Works for anything ``pinneapple_data.cae`` reads (OpenFOAM case or zip, CalculiX .frd, Abaqus .inp, Gmsh, VTK) and for
STL/OBJ geometry. Vectors are shown as magnitude, stress tensors as von Mises.
Writes into examples/studio/_out/beam/: .glb (Blender, Unreal, Unity, three.js), .usda (Omniverse), an interactive
viewer, and a Blender render if ``bpy`` is installed (pip install bpy).
"""
import os
import sys

import pinneapple as pp

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "apps", "interop_hub", "examples", "calculix_cantilever.frd")
out = os.path.join(HERE, "_out", "beam")
os.makedirs(out, exist_ok=True)

scene = pp.viz.Scene.from_file(src)                                   # 1. load
scene.title = "CalculiX cantilever"
scene.labels.update({"STRESS": "von Mises stress (MPa)", "DISP": "Displacement (mm)"})
print("fields on the surface:", scene.field_names())

scene.save(os.path.join(out, "beam.glb"))                             # 2. export
scene.save(os.path.join(out, "beam.usda"))
print("viewer:", pp.viz.web_viewer(scene, os.path.join(out, "viewer")),
      "-> python -m http.server -d", os.path.join(out, "viewer"))

try:                                                                   # 3. render (Blender Cycles)
    import bpy  # noqa: F401
    pp.viz.render(scene, os.path.join(out, "beam_stress.jpg"), field="STRESS", samples=64)
    print("render:", os.path.join(out, "beam_stress.jpg"))
except ImportError:
    print("pip install bpy for the Blender render")
os._exit(0)                                                            # bpy can crash while tearing down
