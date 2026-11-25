"""Rebuild the classical CFD validation results: the published tables, and the OpenFOAM runs compared with them (#272).

    python examples/cfd/classical_cases/run_cases.py --out examples/cfd/classical_cases/results
    python examples/cfd/classical_cases/run_cases.py --only reference          # no solver needed
    python examples/cfd/classical_cases/run_cases.py --only cavity --re 100 400
    python examples/cfd/classical_cases/run_cases.py --only ladder-ahmed       # reads the Ahmed campaign's records

It writes one JSON per case next to the tables it was compared with, so the README's table can be regenerated and the
numbers in it can be checked. Without OpenFOAM the ``reference`` part still runs (it is transcribed data) and the
solver parts say so instead of failing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pinneapple_simulation.cfd_validation import (  # noqa: E402
    cavity,
    foam,
    ladder,
    ladder_ahmed,
    reference,
    step,
    tmr,
    warm_start,
)


def _dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"  wrote {path}")


def do_reference(out: str) -> None:
    print("reference tables (no solver needed)")
    _dump(os.path.join(out, "reference.json"),
          {"profiles": reference.as_dicts(),
           "nusselt": {"citation": reference.DEV_VAHL_DAVIS,
                       "values": {f"{k:g}": v for k, v in sorted(reference.DEV_VAHL_DAVIS_NU.items())}},
           "armaly_reattachment": {"citation": "Armaly et al. (1983), J. Fluid Mech. 127, 473-496",
                                   "values": {}, "note": "the published curve is a figure; not transcribed"}})


def do_warm(out: str, cases: str) -> None:
    """#494: a surrogate over Re fitted from solved cases, then a cold and a warm run at held-out Reynolds numbers."""
    print("warm start: snapshots at Re 50..800 on 64 x 64, benchmark at Re 400 and 1000 (held out)")
    snaps = warm_start.snapshots(os.path.join(cases, "warm"), (50, 100, 200, 300, 600, 800), n=64)
    surrogate = warm_start.OperatorSurrogate(modes=8).fit(snaps)
    result = {"surrogate": {"modes": 8, "fitted_on": [float(r) for r in sorted(snaps)],
                            "energy": surrogate.energy, "seconds": round(surrogate.fitted_seconds, 2)},
              "benchmarks": []}
    for re in (400, 1000):
        got = warm_start.benchmark(re, surrogate, os.path.join(cases, "warm"), n=64)
        print(f"  Re = {re}: cold {got.cold_iterations} it / {got.cold_seconds:.1f} s, "
              f"warm {got.warm_iterations} it / {got.warm_seconds:.1f} s -> {got.iteration_ratio:.2f}x iterations, "
              f"{got.speedup:.2f}x wall time; field difference {got.field_difference}")
        result["benchmarks"].append(got.to_dict())
    _dump(os.path.join(out, "warm_start.json"), result)


def do_ladder(out: str, cases: str, *, reuse: bool = False) -> None:
    """#294: the cost-accuracy table, and the calibration's effect on the ranking."""
    print("ladder: NACA 0012 at alpha = 0, 4, 8 on three RANS levels")
    designs = [ladder.design("0012", a) for a in (0.0, 4.0, 8.0)]
    got = ladder.ladder(designs, os.path.join(cases, "ladder"), log=lambda m: print(m), reuse=reuse)
    got["calibration"] = ladder.calibrate(got["rows"], "rans-coarse", designs=[0.0, 4.0])
    for row in got["table"]:
        print(f"  {row['rung']:<14} {row['cells']:>7} cells  {row['seconds_per_design']:>7.1f} s/design  "
              f"CL err {row['CL_error_rel']}  spearman {row['spearman_CL']}  Cp rms {row['Cp_error_rms']}")
    _dump(os.path.join(out, "ladder.json"), got)


def do_ladder_ahmed(out: str) -> None:
    """#294: the same table on the second family, the Ahmed body's variants, read from its campaign's records."""
    print("ladder: Ahmed body at the six slant angles of Ahmed et al. (1984), campaign and L2 meshes")
    got = ladder_ahmed.ladder(log=lambda m: print(m))
    for row in got["table"]:
        print(f"  {row['rung']:<10} {row['cells']:>7} cells  {row['seconds_per_design']:>7.1f} s/design  "
              f"CD err {row['CD_error_rel']}  Cp rms {row['Cp_error_rms']}  spearman {row['spearman_CD']}")
    _dump(os.path.join(out, "ladder_ahmed.json"), got)


def do_tmr(out: str, cases: str) -> None:
    """#265: the TMR flat plate with Spalart-Allmaras and k-omega SST on three grids."""
    print("NASA TMR 2DZP flat plate: SpalartAllmaras and kOmegaSST on three grids")
    got = tmr.study(os.path.join(cases, "tmr"))
    for model, entry in got["models"].items():
        print(f"  {model:<16} cf(0.97) = {[round(v, 6) for v in entry['cf']]}  order {entry.get('observed_order')}"
              f"  extrapolated {entry.get('extrapolated')}  law {entry['cf_law']:.6f}")
    _dump(os.path.join(out, "tmr_flat_plate.json"), got)


def do_cavity(out: str, reynolds: list[int], meshes: list[int], cases: str, *, reuse: bool = False) -> None:
    result = {"case": "ghia-1982", "solver": "simpleFoam", "scheme": "linearUpwind grad(U)",
              "case_dir": "system/ in pinneapple_simulation.cfd_validation.cavity", "runs": []}
    for re in reynolds:
        print(f"cavity Re = {re}: meshes {meshes}")
        t0 = time.time()
        got = cavity.study(re, os.path.join(cases, "cavity"), meshes=meshes, reuse=reuse)
        got["wall_seconds"] = round(time.time() - t0, 1)
        for rec in got["runs"]:
            print(f"  n = {rec['n']:>4}  {rec['cells']:>6} cells  {rec['iterations']:>6} it  "
                  f"{rec['seconds']:>7.1f} s  converged={rec['converged']}")
        for n, err in got["errors"].items():
            print(f"    n = {n:>4}: u l2 {err['u']['l2']:.4f} ({100 * err['u']['l2_relative']:.2f} %), "
                  f"v l2 {err['v']['l2']:.4f} ({100 * err['v']['l2_relative']:.2f} %)")
        result["runs"].append(got)
    _dump(os.path.join(out, "cavity.json"), result)


def do_step(out: str, reynolds: list[int], meshes: list[int], cases: str) -> None:
    result = {"case": "armaly-1983", "solver": "simpleFoam", "scheme": "linearUpwind grad(U)", "runs": []}
    for re in reynolds:
        print(f"step Re = {re}: meshes {meshes}")
        t0 = time.time()
        got = step.study(re, os.path.join(cases, "step"), meshes=meshes)
        got["wall_seconds"] = round(time.time() - t0, 1)
        for rec in got["runs"]:
            print(f"  m = {rec['cells_per_step']:>3}  {rec['cells']:>6} cells  {rec['iterations']:>6} it  "
                  f"{rec['seconds']:>7.1f} s  x1/S = {rec['x1_over_S']}  converged={rec['converged']}")
        result["runs"].append(got)
    _dump(os.path.join(out, "step.json"), result)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
                    help="where the JSON lands (the numbers committed next to the README)")
    ap.add_argument("--cases", default="/tmp/pinneapple-cfd-cases",
                    help="where the OpenFOAM cases are written (never committed: they are hundreds of files)")
    ap.add_argument("--only", nargs="+", default=["reference", "cavity", "step", "warm", "ladder", "ladder-ahmed",
                                                  "tmr"],
                    choices=["reference", "cavity", "step", "warm", "ladder", "ladder-ahmed", "tmr"])
    ap.add_argument("--reuse", action="store_true", help="read the cases already solved under --cases")
    ap.add_argument("--re", nargs="+", type=int, default=[100, 400, 1000], help="Reynolds numbers for the cavity")
    ap.add_argument("--meshes", nargs="+", type=int, default=list(cavity.MESHES))
    ap.add_argument("--step-re", nargs="+", type=int, default=[100, 200, 400, 800])
    ap.add_argument("--step-meshes", nargs="+", type=int, default=list(step.MESHES))
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cases, exist_ok=True)
    if "reference" in args.only:
        do_reference(args.out)
    if "cavity" in args.only or "step" in args.only or "ladder" in args.only:
        found = foam.bashrc()
        if found is None:
            print("no OpenFOAM installation found (set FOAM_BASHRC): the solver cases are skipped", file=sys.stderr)
            return 1
        print(f"OpenFOAM: {found}")
    if "cavity" in args.only:
        do_cavity(args.out, args.re, args.meshes, args.cases, reuse=args.reuse)
    if "step" in args.only:
        do_step(args.out, args.step_re, args.step_meshes, args.cases)
    if "warm" in args.only:
        do_warm(args.out, args.cases)
    if "ladder" in args.only:
        do_ladder(args.out, args.cases, reuse=args.reuse)
    if "ladder-ahmed" in args.only:
        do_ladder_ahmed(args.out)
    if "tmr" in args.only:
        do_tmr(args.out, args.cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
