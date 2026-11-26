#!/usr/bin/env python3
"""The evidence behind #177: a recovered structural parameter, its interval, and whether the data could see it.

Three questions, in the order an identification report asks them:

1. **Are the data real?** A C3D20R brick cantilever is solved in **CalculiX** at a known Young's modulus and a
   known tip load, the deflection is read at four gauge stations, and the gap between that 3-D solution and
   the Timoshenko beam the inversion uses is **measured** -- it is not assumed to be zero.
2. **Did we get it back?** The same readings are inverted three ways: modulus unknown (load on a load cell),
   load unknown (material certified), and both unknown. The truth is available because the data were
   generated at known values, so the interval is checked against it.
3. **What does it look like when it cannot be?** With both unknowns the displacements only see ``P/E``: the
   report says so, with the dead direction, instead of quoting two confident intervals.

    $ python scripts/structural_identification.py              # the whole study
    $ python scripts/structural_identification.py --write      # regenerate the stored measurements (needs ccx)

Exit code 1 when a parameter that the data *can* identify comes back with the truth outside its interval: a
study whose job is to detect a miscalibrated interval should not report success when it finds one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinneapple_analysis.inverse_problems.structural import (  # noqa: E402
    GEOMETRY,
    IDENTIFICATION_FIXTURE,
    ObservationSet,
    identify_cantilever,
    load_measurements,
    timoshenko_deflection,
)

#: the truth of the synthetic study: aluminium and 800 N
TRUE = {"E": 70e9, "P": 800.0}
STATIONS = (0.25, 0.5, 0.75, 1.0)
NOISE = 1e-6            # an LVDT is a micrometre
MODEL_FORM = 0.01       # replaced by the measured gap when CalculiX runs


def measured_with_calculix() -> ObservationSet:
    """Solve the cantilever in CalculiX and read the deflection at the gauge stations."""
    import tempfile

    from pinneapple_simulation.external_solvers.calculix import (
        cantilever_hex_mesh,
        ccx_backend,
        read_frd,
        run_ccx,
        write_inp,
    )

    if ccx_backend() is None:
        raise RuntimeError("CalculiX is not available: install ccx or Docker, or use the stored measurements")
    L, b, h = GEOMETRY["L"], GEOMETRY["b"], GEOMETRY["h"]
    model = cantilever_hex_mesh(L, b, h, nx=20, ny=2, nz=2, element_type="C3D20R")
    model.E, model.nu = TRUE["E"], GEOMETRY["nu"]
    tip = model.nsets["TIP"]
    model.boundary = [("FIXED", 1, 3, 0.0)]
    model.cloads = [(str(n), 3, -TRUE["P"] / len(tip)) for n in tip]
    with tempfile.TemporaryDirectory() as workdir:
        write_inp(model, str(Path(workdir) / "beam.inp"))
        run = run_ccx(workdir, "beam")
        if run.returncode != 0:
            raise RuntimeError(f"ccx failed (exit {run.returncode}):\n{run.stdout[-2000:]}")
        field = read_frd(run.frd)
    ids = np.asarray(field.node_ids)
    coords = np.asarray(field.coords)
    uz = field.field("DISP")[:, 2]
    stations = np.asarray(STATIONS)
    reference = timoshenko_deflection(stations, E=TRUE["E"], P=TRUE["P"])
    measured = np.empty(stations.size)
    for i, x in enumerate(stations):
        pick = np.abs(coords[:, 0] - x) <= (L / 40.0) / 2.0 + 1e-12
        measured[i] = -float(np.mean(uz[pick]))            # the load is along -z, the deflection is +z down
    gap = float(np.max(np.abs(np.abs(measured) - reference) / np.abs(reference)))
    generator = np.random.default_rng(0)
    sigma = np.sqrt(NOISE**2 + (gap * np.abs(measured)) ** 2)
    return ObservationSet(stations=stations, measured=measured + generator.normal(0.0, NOISE, stations.size),
                          sigma=sigma, truth=reference, noise=NOISE, model_form=gap,
                          truth_parameters=dict(TRUE)), ids.size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="regenerate the stored measurements with CalculiX")
    args = ap.parse_args(argv)

    if args.write:
        measurements, nodes = measured_with_calculix()
        IDENTIFICATION_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        IDENTIFICATION_FIXTURE.write_text(json.dumps(measurements.to_dict(), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {IDENTIFICATION_FIXTURE} ({nodes} nodes)")
    else:
        measurements = load_measurements()

    print(f"stations {np.round(measurements.stations, 3).tolist()}")
    print(f"measured  {np.round(measurements.measured, 8).tolist()}  m")
    print(f"beam      {np.round(measurements.truth, 8).tolist()}  m")
    print(f"noise {measurements.noise:g} m, model-form gap {measurements.model_form:.4%}\n")

    ok = True
    for unknown in ("modulus", "load", "both"):
        ident = identify_cantilever(measurements, unknown=unknown)
        report = ident.verdict()
        print(f"[{unknown}]")
        for name, row in report["parameters"].items():
            width = "open" if row["relative_width"] is None else f"{row['relative_width']:.2%}"
            inside = {True: "inside", False: "OUTSIDE", None: "no truth"}[row["inside"]]
            print(f"  {name}: {row['value']:.6g} [{row['low']:.6g}, {row['high']:.6g}] "
                  f"truth {row['truth']:.6g} -> {inside} (width {width})")
        for text in report["dead"]:
            print(f"  dead direction: {text}")
        print(f"  condition number {report['condition_number']:.3g}")
        if unknown != "both" and not all(r["inside"] for r in report["parameters"].values()):
            ok = False
        if unknown == "both" and not report["dead"]:
            print("  ! both parameters came back as identifiable: the dead direction was not detected")
            ok = False
        print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
