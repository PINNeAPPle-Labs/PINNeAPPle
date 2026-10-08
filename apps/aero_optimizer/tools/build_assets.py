"""Summaries the app shows: the training set, the CFD validation, the OpenFOAM verification rounds.

    python apps/aero_optimizer/tools/build_assets.py DATA_DIR [VERIFY_DIR] --out apps/aero_optimizer/model
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pinneapple_design.aero.geometry import outline, properties                  # noqa: E402

# Run on the production grid (O-grid 228 x 88 cells, far field 200 chords), Re 6e6, k-omega SST, against the NASA
# Turbulence Modeling Resource NACA 0012 validation case (SST, fine grids).
CFD_VALIDATION = [
    {"case": "NACA 0012, α 0°, drag", "here": "Cd 0.00812", "ref": "Cd ≈ 0.0081"},
    {"case": "NACA 0012, α 10°, lift", "here": "Cl 1.086", "ref": "Cl ≈ 1.09"},
    {"case": "NACA 0012, α 10°, drag", "here": "Cd 0.0140", "ref": "Cd ≈ 0.0123"},
]
CFD_NOTE = ("Lift within 0.5 %, drag at zero lift within 1 %; at high lift the drag reads ~14 % high on this 20k-cell grid "
            "(the far field at 25 chords instead of 200 gave +48 %, which is why it is at 200). The same grid topology is "
            "used for every design, so the error is shared and the ranking holds.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("verify", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.data, "runs", "*.json")))]
    lhs = [r for r in runs if r["set"] == "lhs"]
    st = Counter(r.get("status") for r in runs)
    secs = [r["seconds"] for r in runs if r.get("seconds")]
    shapes = {}
    for r in lhs:
        shapes.setdefault(r["shape_id"], r["shape"])
    ref2412 = sorted([r for r in runs if r["shape_id"] == "NACA 2412" and "Cl" in r], key=lambda r: r["alpha"])
    val = []
    if ref2412:
        a_ = np.array([r["alpha"] for r in ref2412])
        cl = np.array([r["Cl"] for r in ref2412])
        att = (a_ >= -2) & (a_ <= 6)
        slope, icpt = np.polyfit(a_[att], cl[att], 1)
        val.append(f"OpenFOAM, NACA 2412 (the Cessna 172 airfoil): zero-lift angle {-icpt / slope:.1f}° and lift slope "
                   f"{slope:.3f} per degree, against −2.1° and ~0.105 in the NACA wind-tunnel data (Abbott & von Doenhoff)")
    val.append("OpenFOAM against NASA's turbulence-model validation (NACA 0012, Re 6 million, SST): zero-lift drag "
               "0.00812 vs ≈0.0081, lift at 10° 1.086 vs ≈1.09")
    ds = {
        "runs": len(runs),
        "kpis": [{"l": "OpenFOAM runs", "v": f"{len(runs)}", "s": f"{len(lhs)} on {len(shapes)} airfoils + {len(runs) - len(lhs)} NACA polars"},
                 {"l": "Converged", "v": f"{100 * (st['converged'] + st['steady-ish']) / max(1, len(runs)):.0f} %",
                  "s": f"{st['unsteady']} unsteady near stall (averaged), {st['failed'] + st['timeout']} failed"},
                 {"l": "CPU per run", "v": f"{np.median(secs):.0f} s" if secs else "—", "s": "20k cells, 1200 iterations, one core"},
                 {"l": "Angles of attack", "v": "−2° … 14°", "s": "4 per airfoil, one in each band"}],
        "status": dict(st),
        "cfd_validation": CFD_VALIDATION, "cfd_note": CFD_NOTE, "validation": val,
        "outlines": [outline(np.array(s), 40) for s in list(shapes.values())],
        "points": [{"id": r["id"], "alpha": r["alpha"], "cl": r.get("Cl"), "cd": r.get("Cd"), "status": r.get("status"),
                    "t": properties(np.array(r["shape"]))["thickness"]} for r in runs if "Cl" in r],
    }
    os.makedirs(a.out, exist_ok=True)
    json.dump(ds, open(os.path.join(a.out, "dataset.json"), "w"))
    rounds = []
    for vd in a.verify:
        for f in sorted(glob.glob(os.path.join(vd, "verification_round*.json"))):
            v = json.load(open(f))
            D = v["designs"]
            ok = [d for d in D if d["openfoam"]["feasible"] == d["surrogate"]["feasible"]]
            ev = [abs(d["surrogate"]["vmax_kt"] - d["openfoam"]["vmax_kt"]) for d in D]
            ec = [abs(d["surrogate"]["co2_100km"] - d["openfoam"]["co2_100km"]) / d["openfoam"]["co2_100km"] * 100 for d in D]
            es = [abs(d["surrogate"]["v_stall_kt"] - d["openfoam"]["v_stall_kt"]) for d in D]
            v["summary"] = (f"Round {v['round']}: {len(D)} designs from the Pareto front run in OpenFOAM (9 angles each): "
                            f"top speed within {max(ev):.1f} kt, CO₂ within {max(ec):.1f} %, stall speed within {max(es):.1f} kt "
                            f"of the surrogate; {len(ok)}/{len(D)} with the same flyable verdict.")
            rounds.append(v)
    json.dump({"rounds": rounds}, open(os.path.join(a.out, "verification.json"), "w"))
    print(json.dumps({k: v for k, v in ds.items() if k not in ("outlines", "points")}, indent=1))
    for r in rounds:
        print(r["summary"])


if __name__ == "__main__":
    main()
