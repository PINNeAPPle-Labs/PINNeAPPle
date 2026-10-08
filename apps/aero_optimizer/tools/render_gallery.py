"""Photoreal Blender (Cycles) renders for the app gallery: the baseline and the optimizer's picks, plus the skin
pressure and streamlines of the designs run in OpenFOAM 3D.

    pip install bpy==4.2.0
    python apps/aero_optimizer/tools/render_gallery.py [--samples 128]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pinneapple_design.aero.aircraft3d import airframe_from, baseline_x           # noqa: E402
from pinneapple_design.aero.airframe import to_glb                               # noqa: E402
from pinneapple_design.aero.case3d import surface_scalars                        # noqa: E402
from pinneapple_design.aero.render_blender import render                         # noqa: E402

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = os.path.join(MODEL_DIR, "renders")
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp()
    D = json.load(open(os.path.join(MODEL_DIR, "default_search3d.json")))
    jobs = [("01_cessna172_class_hero", baseline_x(), "hero", False, None, "Baseline: Cessna 172 class (NACA 2412, AR 7.5)")]
    for k, (pick, view) in enumerate((("balanced", "hero"), ("fastest", "hero_front"), ("safest", "hero"))):
        x = np.array(D["details"][str(D["picks"][pick])]["x"])
        d = D["details"][str(D["picks"][pick])]
        jobs.append((f"0{k + 2}_{pick}_{view}", x, view, False, None,
                     f"{pick.capitalize()} pick: {d['vmax_kt']:.0f} kt, {d['co2_100km']:.1f} kg CO2/100 km, stall {d['v_stall_kt']:.0f} kt, AR {d['plan']['aspect_ratio']:.1f}"))
    V = json.load(open(os.path.join(MODEL_DIR, "verification3d.json"))) if os.path.exists(os.path.join(MODEL_DIR, "verification3d.json")) else {"designs": []}
    for k, v in enumerate(V["designs"]):
        jobs.append((f"1{k}_{v['id']}_openfoam_pressure", np.array(v["x"]), "hero", True, v, f"OpenFOAM 3D: skin pressure and streamlines, {v['label']}"))
    captions = {}
    for name, x, view, cfd, v, cap in jobs:
        if a.only and a.only not in name:
            continue
        af = airframe_from(x)
        parts = af.build()
        scal, lines = None, None
        if cfd:
            z = np.load(os.path.join(MODEL_DIR, "cfd3d", f"{v['id']}_cp.npz"))
            scal = surface_scalars(af, parts, {"xyz": z["xyz"], "cp": z["cp"].astype(np.float32)})
            lines = os.path.join(tmp, name + "_lines.npz")
            np.savez(lines, lines=np.array([np.asarray(L) for L in v["lines"]], dtype=object))
        glb = os.path.join(tmp, name + ".glb")
        open(glb, "wb").write(to_glb(parts, scal))
        path = os.path.join(out, name + ".jpg")
        render(glb, path, view=view, samples=a.samples, size=(1600, 900), cp=cfd, lines_npz=lines)
        captions[name + ".jpg"] = cap
        print("rendered", path, flush=True)
    p = os.path.join(MODEL_DIR, "verification3d.json")
    V = json.load(open(p)) if os.path.exists(p) else {"designs": []}
    V.setdefault("captions", {}).update(captions)
    json.dump(V, open(p, "w"))


if __name__ == "__main__":
    main()
