"""Rebuild the classical CFD validation results: the published tables, and the OpenFOAM runs compared with them (#272).

    python examples/cfd/classical_cases/run_cases.py --out examples/cfd/classical_cases/results
    python examples/cfd/classical_cases/run_cases.py --only reference          # no solver needed
    python examples/cfd/classical_cases/run_cases.py --only cavity --re 100 400

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

from pinneapple_simulation.cfd_validation import cavity, foam, reference, step     # noqa: E402


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


def do_cavity(out: str, reynolds: list[int], meshes: list[int]) -> None:
    result = {"case": "ghia-1982", "solver": "simpleFoam", "scheme": "linearUpwind grad(U)",
              "case_dir": "system/ in pinneapple_simulation.cfd_validation.cavity", "runs": []}
    for re in reynolds:
        print(f"cavity Re = {re}: meshes {meshes}")
        t0 = time.time()
        got = cavity.study(re, os.path.join(out, "_cases"), meshes=meshes)
        got["wall_seconds"] = round(time.time() - t0, 1)
        for rec in got["runs"]:
            print(f"  n = {rec['n']:>4}  {rec['cells']:>6} cells  {rec['iterations']:>6} it  "
                  f"{rec['seconds']:>7.1f} s  converged={rec['converged']}")
        for n, err in got["errors"].items():
            print(f"    n = {n:>4}: u l2 {err['u']['l2']:.4f} ({100 * err['u']['l2_relative']:.2f} %), "
                  f"v l2 {err['v']['l2']:.4f} ({100 * err['v']['l2_relative']:.2f} %)")
        result["runs"].append(got)
    _dump(os.path.join(out, "cavity.json"), result)


def do_step(out: str, reynolds: list[int], meshes: list[int]) -> None:
    result = {"case": "armaly-1983", "solver": "simpleFoam", "scheme": "linearUpwind grad(U)", "runs": []}
    for re in reynolds:
        print(f"step Re = {re}: meshes {meshes}")
        t0 = time.time()
        got = step.study(re, os.path.join(out, "_cases"), meshes=meshes)
        got["wall_seconds"] = round(time.time() - t0, 1)
        for rec in got["runs"]:
            print(f"  m = {rec['cells_per_step']:>3}  {rec['cells']:>6} cells  {rec['iterations']:>6} it  "
                  f"{rec['seconds']:>7.1f} s  x1/S = {rec['x1_over_S']}  converged={rec['converged']}")
        result["runs"].append(got)
    _dump(os.path.join(out, "step.json"), result)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--only", nargs="+", default=["reference", "cavity", "step"],
                    choices=["reference", "cavity", "step"])
    ap.add_argument("--re", nargs="+", type=int, default=[100, 400, 1000], help="Reynolds numbers for the cavity")
    ap.add_argument("--meshes", nargs="+", type=int, default=list(cavity.MESHES))
    ap.add_argument("--step-re", nargs="+", type=int, default=[100, 200, 400, 800])
    ap.add_argument("--step-meshes", nargs="+", type=int, default=list(step.MESHES))
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if "reference" in args.only:
        do_reference(args.out)
    if "cavity" in args.only or "step" in args.only:
        found = foam.bashrc()
        if found is None:
            print("no OpenFOAM installation found (set FOAM_BASHRC): the solver cases are skipped", file=sys.stderr)
            return 1
        print(f"OpenFOAM: {found}")
    if "cavity" in args.only:
        do_cavity(args.out, args.re, args.meshes)
    if "step" in args.only:
        do_step(args.out, args.step_re, args.step_meshes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
