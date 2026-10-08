"""Run whole aircraft in OpenFOAM 3D (half model) and compare with the vortex-lattice model at the same angle.

    python apps/aero_optimizer/tools/verify3d.py WORK_DIR --label "Cessna 172 class" [--x 12 numbers] [--alpha 2]

Writes apps/aero_optimizer/model/cfd3d/<id>_cp.npz (skin pressure) and appends to model/verification3d.json
(forces, the model's numbers, streamlines for the viewer).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pinneapple_design.aero.aircraft import Polar                           # noqa: E402
from pinneapple_design.aero.aircraft3d import Aircraft3D, airframe_from, baseline_x  # noqa: E402
from pinneapple_design.aero.case3d import read_result3d, run_case3d, write_case3d   # noqa: E402
from pinneapple_design.aero.geometry import REFERENCE                        # noqa: E402
from pinneapple_design.aero.optimize import ALPHAS, Engine                   # noqa: E402

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--label", required=True)
    ap.add_argument("--x", nargs=12, type=float, default=None)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=55.0)
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--skip-run", action="store_true")
    a = ap.parse_args()
    x = np.array(a.x) if a.x else baseline_x()
    did = hashlib.sha1(np.round(x, 5).tobytes()).hexdigest()[:10]
    case = os.path.join(a.work, did)
    af = airframe_from(x)
    if not a.skip_run:
        info = write_case3d(case, af, alpha_deg=a.alpha, speed=a.speed, iterations=a.iterations, procs=a.procs)
        json.dump(info, open(os.path.join(case, "info.json"), "w"))
        t = run_case3d(case, a.procs, log=lambda s: print(s, flush=True))
    info = json.load(open(os.path.join(case, "info.json")))
    res = read_result3d(case, info, af)
    eng = Engine.load(os.path.join(MODEL_DIR, "surrogate.pt"))
    co = eng.mlp_coefficients(np.stack([x[:6], REFERENCE["NACA 0012"]]))
    wp = Polar(ALPHAS, co["cl"][0], co["cd"][0], co["cm"][0])
    tp = Polar(ALPHAS, co["cl"][1], co["cd"][1], co["cm"][1])
    m = Aircraft3D(x, wp, tp, nc=6, ns_wing=24, ns_tail=8).at_alpha(a.alpha, 0.0, a.speed, rho=1.225)
    print(f"{a.label}: OpenFOAM CL {res['CL']:.4f} CD {res['CD']:.5f} (p {res['CD_pressure']:.5f} + f {res['CD_friction']:.5f}) | "
          f"model CL {m['CL']:.4f} CD {m['CD']:.5f}", flush=True)
    os.makedirs(os.path.join(MODEL_DIR, "cfd3d"), exist_ok=True)
    np.savez_compressed(os.path.join(MODEL_DIR, "cfd3d", f"{did}_cp.npz"), xyz=res["surface"]["xyz"].astype(np.float32),
                        cp=res["surface"]["cp"].astype(np.float16))
    lines = [L.round(3).tolist() for L in res["streamlines"]]
    lines += [[[p[0], -p[1], p[2]] for p in L] for L in lines]
    rec = {"id": did, "label": a.label, "x": x.tolist(), "alpha": a.alpha, "speed": a.speed, "cells": res["cells"],
           "CL": res["CL"], "CD": res["CD"], "CD_pressure": res["CD_pressure"], "CD_friction": res["CD_friction"],
           "patches": res["patches"], "model": m, "lines": lines, "date": time.strftime("%Y-%m-%d")}
    p = os.path.join(MODEL_DIR, "verification3d.json")
    V = json.load(open(p)) if os.path.exists(p) else {"designs": []}
    V["designs"] = [d for d in V["designs"] if d["id"] != did] + [rec]
    V["note"] = ("Half model with a symmetry plane, snappyHexMesh (~0.6 M cells), simpleFoam k-ω SST with wall functions, "
                 f"{a.speed:.0f} m/s, sea level, tail at zero incidence, no propeller, gear or struts; the model is "
                 "evaluated the same way (clean airframe, same angle).")
    json.dump(V, open(p, "w"))


if __name__ == "__main__":
    main()
