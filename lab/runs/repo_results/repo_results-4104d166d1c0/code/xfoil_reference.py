"""XFOIL reference polars for the NACA 2412 baseline (XFOIL 6.99, viscous panel method with e^N transition).

    python xfoil_reference.py OUT_DIR [--re 4e6] [--xfoil xfoil]

Two transition settings, both at Re 4e6, Mach 0:
  free     - natural transition, Ncrit = 9 (XFOIL default; a clean wind tunnel / smooth model)
  tripped  - transition forced at 1 % chord on both sides (close to the fully turbulent k-omega SST CFD)
for two shapes: the exact NACA 2412 (XFOIL's own NACA generator, "NACA 2412") and the 6-weight CST fit of it that
the CFD uses (geometry/naca2412_cst_coordinates.dat). Writes OUT_DIR/xfoil_<shape>_<transition>.pol (XFOIL polar
file) and OUT_DIR/xfoil_summary.json (.log with the end of the XFOIL console if a run fails).

Note: the xfoil 6.99 package of Ubuntu 24.04 is built with floating-point traps (-ffpe-trap=invalid,zero,
-finit-real=inf) and stops with SIGFPE; the reference results were made with the same 6.99 sources rebuilt with
plain -O2 (see README.md). Any other XFOIL 6.99 build works.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CST_DAT = os.path.join(HERE, "..", "geometry", "naca2412_cst_coordinates.dat")
ALPHAS = (-2.0, 14.0, 1.0)


def run_xfoil(load: str, re_: float, xtr: float | None, polar: str, xfoil: str = "xfoil") -> list[dict]:
    """One XFOIL session: sweep 0 -> -2 deg, re-initialise the boundary layer, sweep 1 -> 14 deg; returns the
    converged points of the saved polar (XFOIL leaves non-converged points out)."""
    if os.path.exists(polar):
        os.remove(polar)
    cwd, polar_name = os.path.dirname(os.path.abspath(polar)), os.path.basename(polar)   # XFOIL reads short paths
    vpar = f"VPAR\nXTR\n{xtr}\n{xtr}\n\n" if xtr is not None else ""
    lo, hi, d = ALPHAS
    cmds = (f"PLOP\nG F\n\n{load}\nPANE\nOPER\nVISC {re_:.0f}\nMACH 0\nITER 300\n{vpar}PACC\n{polar_name}\n\n"
            f"ASEQ 0 {lo} -{d}\nINIT\nASEQ {d} {hi} {d}\nPACC\n\nQUIT\n")
    r = subprocess.run([xfoil], input=cmds, text=True, capture_output=True, timeout=600, check=False,
                       cwd=cwd)
    if not os.path.exists(polar):                     # keep the console only when something went wrong
        with open(polar[:-4] + ".log", "w") as f:
            f.write(r.stdout[-20000:])
        return []
    rows, started = [], False
    for line in open(polar):
        if line.strip().startswith("------"):
            started = True
        elif started and line.strip():
            v = [float(x) for x in line.split()]
            rows.append({"alpha": v[0], "Cl": v[1], "Cd": v[2], "Cdp": v[3], "Cm": v[4], "xtr_top": v[5],
                         "xtr_bot": v[6]})
    return sorted(rows, key=lambda r: r["alpha"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--re", type=float, default=4e6)
    ap.add_argument("--xfoil", default="xfoil")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {"xfoil": "XFOIL 6.99", "reynolds": a.re, "mach": 0.0, "ncrit": 9, "panels": "PANE default (160)",
           "polars": {}}
    shutil.copy(CST_DAT, os.path.join(a.out, "cst.dat"))
    shapes = {"naca2412_exact": "NACA 2412", "naca2412_cst": "LOAD cst.dat"}
    for sname, load in shapes.items():
        for tname, xtr in (("free", None), ("tripped", 0.01)):
            rows = run_xfoil(load, a.re, xtr, os.path.join(a.out, f"xfoil_{sname}_{tname}.pol"), a.xfoil)
            out["polars"][f"{sname}_{tname}"] = rows
            at4 = [r for r in rows if abs(r["alpha"] - 4.0) < 1e-6]
            print(sname, tname, len(rows), "points;", "alpha 4:", at4[0] if at4 else "not converged")
    # surface pressure of the exact NACA 2412, tripped, alpha 4 (for the Cp comparison with the CFD)
    cmds = (f"PLOP\nG F\n\nNACA 2412\nPANE\nOPER\nVISC {a.re:.0f}\nITER 300\nVPAR\nXTR\n0.01\n0.01\n\nALFA 4\n"
            "CPWR cp.txt\n\nQUIT\n")
    subprocess.run([a.xfoil], input=cmds, text=True, capture_output=True, timeout=120, check=False, cwd=a.out)
    os.replace(os.path.join(a.out, "cp.txt"), os.path.join(a.out, "xfoil_cp_alpha4_tripped.txt"))
    os.remove(os.path.join(a.out, "cst.dat"))
    for f in os.listdir(a.out):                       # boundary-layer dump files XFOIL leaves behind
        if f.endswith(".bl"):
            os.remove(os.path.join(a.out, f))
    with open(os.path.join(a.out, "xfoil_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
