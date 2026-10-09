"""Run an airliner in OpenFOAM 3D (half model, approach condition) and store its fields for the app.

    python apps/aero_optimizer/tools/verify3d_airliner.py WORK_DIR --label "A320 class" [--x 13 numbers] [--skip-run]

Writes apps/aero_optimizer/model/cfd3d/<id>.npz (skin Cp and Cf at the CFD faces, streamlines with their speed,
the wake and symmetry-plane slices) and a summary in model/verification3d.json (forces next to the vortex lattice).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pinneapple_design.aero.airliner3d import airliner_from, baseline_x_al        # noqa: E402
from pinneapple_design.aero.case3d import read_result3d, run_case3d, write_case3d  # noqa: E402
from pinneapple_design.aero.vlm import solve                                      # noqa: E402

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))


def design_id(x) -> str:
    return hashlib.sha1(np.round(np.asarray(x, float), 5).tobytes()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--label", required=True)
    ap.add_argument("--x", nargs=13, type=float, default=None)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=70.0)
    ap.add_argument("--iterations", type=int, default=700)
    ap.add_argument("--skip-run", action="store_true", help="the case in WORK_DIR is already run")
    a = ap.parse_args()
    case = a.work
    if a.skip_run:
        info = json.load(open(os.path.join(case, "info.json")))
        x = np.array(info["x"])
    else:
        x = np.array(a.x) if a.x else baseline_x_al()
        af = airliner_from(x)
        info = write_case3d(case, af, alpha_deg=a.alpha, speed=a.speed, iterations=a.iterations, level_wing=(5, 6), level_fus=(4, 5))
        info["x"] = x.tolist()
        json.dump(info, open(os.path.join(case, "info.json"), "w"))
        run_case3d(case, 4, log=lambda s: print(s, flush=True))
    af = airliner_from(x)
    r = read_result3d(case, info, af)
    sol = solve(af, 16.9, nc=6, ns_wing=24, ns_tail=8)
    m = sol.coefficients(math.radians(info["alpha"]), 0.0)
    did = design_id(x)
    os.makedirs(os.path.join(MODEL_DIR, "cfd3d"), exist_ok=True)
    lines = r["streamlines"]
    sw, ss = r["slice_wake"], r["slice_sym"]
    np.savez_compressed(os.path.join(MODEL_DIR, "cfd3d", f"{did}.npz"),
                        xyz=r["surface"]["xyz"].astype(np.float32), cp=r["surface"]["cp"].astype(np.float32),
                        cf=r["surface"]["cf"].astype(np.float32),
                        lines=np.stack(lines).astype(np.float32), line_speed=np.stack(r["streamline_speed"]).astype(np.float32),
                        wake_speed=sw["speed"].astype(np.float32), wake_vort=sw["vorticity"].astype(np.float32),
                        sym_cp=ss["cp"].astype(np.float32), sym_speed=ss["speed"].astype(np.float32),
                        wake_box=np.array([sw["x"], *sw["y"], *sw["z"]], np.float32),
                        sym_box=np.array([ss["y"], *ss["x"], *ss["z"]], np.float32))
    rec = {"id": did, "label": a.label, "x": x.tolist(), "alpha": info["alpha"], "speed": info["speed"], "cells": r["cells"],
           "CL": r["CL"], "CD": r["CD"], "CD_pressure": r["CD_pressure"], "CD_friction": r["CD_friction"],
           "patches": r["patches"], "model": {"CL": m["CL"], "CDi": m["CDi"]}, "date": time.strftime("%Y-%m-%d"),
           "ranges": {"cp": [float(np.percentile(r["surface"]["cp"], 1)), float(np.percentile(r["surface"]["cp"], 99))],
                      "cf": [float(np.percentile(r["surface"]["cf"], 1)), float(np.percentile(r["surface"]["cf"], 99))]}}
    p = os.path.join(MODEL_DIR, "verification3d.json")
    V = json.load(open(p)) if os.path.exists(p) else {"designs": []}
    V["designs"] = [d for d in V["designs"] if d["id"] != did and len(d.get("x", [])) == 13] + [rec]
    V["note"] = ("Half model with a symmetry plane, snappyHexMesh (~0.4 M cells), simpleFoam k-ω SST with wall functions, "
                 f"approach at {info['speed']:.0f} m/s and α {info['alpha']:.0f}°, sea level, clean wing, tail at zero incidence, "
                 "no nacelles or sharklets. The vortex lattice models wing and tail only; the CFD adds the fuselage's lift.")
    json.dump(V, open(p, "w"), indent=1)
    print(f"{a.label}: OpenFOAM CL {r['CL']:.4f} (wing {r['patches']['wing']['CL']:.4f}) CD {r['CD']:.4f} | vortex lattice CL {m['CL']:.4f}")


if __name__ == "__main__":
    main()
