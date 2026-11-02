"""Runs in a Python with ``bpy``: install the PINNeAPPle STEP extension zip and run every operator on a generated
STEP file, asserting what each one must produce. Used by tests/test_blender_step_extension.py.

    python step_extension_smoke.py <extension.zip> <work_dir>
"""
import json
import os
import sys

import bpy
import numpy as np

addon_utils = __import__("addon_utils")                 # importable only once bpy is

zipf, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
assert bpy.ops.extensions.package_install_files(filepath=zipf, repo="user_default",
                                                enable_on_install=True) == {"FINISHED"}
assert any(m.__name__.endswith("pinneapple_step") for m in addon_utils.modules())
core = sys.modules["bl_ext.user_default.pinneapple_step.core"]
brep = core.brep()
import smlibpy as sm  # noqa: E402

plate = sm.Box(80, 50, 6)
for x in (-25, 25):
    for y in (-15, 15):
        plate = plate - sm.Cylinder(4, 6).moved(sm.Pos(x, y, 0))
step = os.path.join(work, "parts.step")
brep.write_step([plate, sm.Torus(20, 5).moved(sm.Pos(0, 0, 40))], step)
s = bpy.context.scene.pinneapple_step
out = {"brep": brep.__name__}

assert bpy.ops.pinneapple.import_step(filepath=step) == {"FINISHED"}
objs = [o for o in bpy.data.objects if "pnp_step_path" in o]
assert len(objs) == 2
dims = sorted(tuple(round(d, 6) for d in o.dimensions) for o in objs)
assert dims == [(0.05, 0.05, 0.01), (0.08, 0.05, 0.006)], dims        # real size, metres
assert {o["pnp_mesher"] for o in objs} == {"fast", "kernel"}           # the torus falls back to the kernel
for o in objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = objs[0]

before = [len(o.data.polygons) for o in objs]
s.quality = "fine"
assert bpy.ops.pinneapple.remesh() == {"FINISHED"}
after = [len(o.data.polygons) for o in objs]
assert all(a > b for a, b in zip(after, before, strict=True)) and all(o["pnp_quality"] == "fine" for o in objs)

assert bpy.ops.pinneapple.quality_report() == {"FINISHED"}
for o in objs:
    r = json.loads(o["pnp_quality_report"])
    assert r["watertight"] and r["self_intersections"] == 0 and r["chordal_error_over_tolerance"] <= 1.05

for mode in ("MEAN", "GAUSS", "DEVIATION", "ZEBRA", "HIGHLIGHT"):
    s.diagnostic = mode
    assert bpy.ops.pinneapple.diagnostic_shading() == {"FINISHED"}
    for o in objs:
        assert len(o.data.color_attributes["pnp_diagnostic"].data) == len(o.data.vertices)
        assert o.data.materials[0].name == "PNP Diagnostic"
    if mode == "MEAN":                                                  # the torus: H at most (R + 2r)/(2r(R + r))
        torus = max(objs, key=lambda o: o.dimensions[2])
        d = json.loads(torus["pnp_diagnostic"])
        R, r = 0.020, 0.005
        assert abs(d["max"] - (R + 2 * r) / (2 * r * (R + r))) < 1.0
        out["torus_H_max"] = d["max"]

assert bpy.ops.pinneapple.zebra_material() == {"FINISHED"} and objs[0].data.materials[0].name == "PNP Zebra"
assert bpy.ops.pinneapple.stripe_world() == {"FINISHED"} and bpy.context.scene.world.name == "PNP Stripes"

s.diagnostic = "MEAN"
bpy.ops.pinneapple.diagnostic_shading()
for ext in (".glb", ".obj", ".stl", ".usda"):
    p = os.path.join(work, "out" + ext)
    assert bpy.ops.pinneapple.bake(filepath=p, format=ext) == {"FINISHED"} and os.path.getsize(p) > 0
assert '"name":"' + objs[0].name + '"' in open(os.path.join(work, "out.glb"), "rb").read().decode("latin-1")

assert bpy.ops.pinneapple.send_meshdata(directory=work) == {"FINISHED"}
for o in objs:
    d = np.load(os.path.join(work, bpy.path.clean_name(o.name) + ".npz"))
    assert d["faces"].shape[0] == len(o.data.polygons) and d["normals"].shape == d["faces"].shape

bpy.ops.preferences.addon_disable(module="bl_ext.user_default.pinneapple_step")
out["ok"] = True
print("PNP_SMOKE " + json.dumps(out))
