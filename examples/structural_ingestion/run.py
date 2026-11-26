"""Structural ingestion: one explicit-FEA run as one object, and the normalisation that makes it a training set.

Three coupon impacts at coupon scale are run by the library's own explicit crash solver, each with its own geometry —
one family per design, which is what a split has to hold out. Each run is then read back through the two paths #1500
names, and the numbers below are measured on this machine rather than asserted:

1. ``CrashResult`` -> ``StructuralRun`` in one call: frames, time axis, parts with their elements, units, provenance;
2. the same run written as a legacy-VTK time series (the format ``anim_to_vtk`` writes) and read back — per-field
   discrepancy, at the precision the format stores;
3. the same run laid out as a staged OpenRadioss case (``frame_000.vtk`` + ``disp_XXX.txt``/``time_XXX.txt``) and read
   back — displacement only past frame 0, and the run says which fields that costs;
4. ``build_training_set``: dedup, node decimation with the error it introduces, unit conversion to SI,
   non-dimensionalisation, the part hierarchy preserved, and a split by geometry family;
5. the Zarr round trip with its checksum, at ``float64`` (bit-identical) and at ``float32`` (documented).

Run: ``python examples/structural_ingestion/run.py``      (``--smoke``: smaller meshes, two designs)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from pinneapple_data.cae import StructuralRun, build_training_set  # noqa: E402
from pinneapple_data.cae.vtk_series import write_vtk_frame  # noqa: E402
from pinneapple_simulation.numerical_solvers.crash_fem import CrashSolver, Material, RigidWall  # noqa: E402
from pinneapple_simulation.numerical_solvers.solid_fem import box_mesh  # noqa: E402

STEEL = Material(rho=7.8e3, E=2.1e11, nu=0.3, yield_stress=6e8, hardening=4e8, erosion_strain=0.4)
#: length, width, height, cells along each, end time — three geometries, one per family
DESIGNS = [(0.060, 0.020, 0.020, 12, 4, 4, 1.2e-4),
           (0.050, 0.020, 0.020, 10, 4, 4, 1.0e-4),
           (0.070, 0.024, 0.020, 14, 4, 4, 1.4e-4)]


def coupon(design, family: str, frames: int = 6) -> StructuralRun:
    """One coupon driven into a rigid wall: the library's explicit solver, wrapped in the run container."""
    length, width, height, nx, ny, nz, end = design
    mesh = box_mesh(length, width, height, nx, ny, nz)
    parts = np.where(np.arange(len(mesh.elements)) < len(mesh.elements) // 2, 1, 2)
    solver = CrashSolver(mesh, material=STEEL, parts=parts, contact=RigidWall(x0=length + 5e-4),
                         fixed=mesh.face_nodes("x-"), initial_velocity=(30.0, 0.0, 0.0), end_time=end,
                         frames=frames, record_velocity=False)
    result = solver.run()
    return StructuralRun.from_crash_result(result, part_names={1: "front", 2: "rear"}, units="SI", case=family,
                                           fixed_nodes=solver.fixed)


def staged_case(root: str, run: StructuralRun) -> str:
    """Write the artifact layout the OpenRadioss runner leaves on disk, from a run of this library's solver.

    ``frame_000.vtk`` is the full first frame and every step has its reduced displacement dump; this is what
    ``external_solvers.openradioss.runner`` produces with ``anim_to_vtk`` inside its container.
    """
    frames = os.path.join(root, "frames")
    os.makedirs(frames, exist_ok=True)
    write_vtk_frame(os.path.join(frames, "frame_000.vtk"), run.frames[0], time=float(run.times[0]),
                    part_ids=run.part_of_cell())
    disp = run.field("displacement")
    for k in range(run.n_frames):
        np.savetxt(os.path.join(frames, f"disp_{k:03d}.txt"), disp[k], fmt="%.6e")
        with open(os.path.join(frames, f"time_{k:03d}.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{run.times[k]:.6e}\n")
    return root


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute difference, as a fraction of the field's own largest value."""
    peak = float(np.max(np.abs(a))) or 1.0
    return float(np.max(np.abs(a - b))) / peak


def figure(metrics: dict, path: str) -> str:
    """What the pipeline costs: the campaign on the left, the error each step introduces on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for name, series in metrics["histories"].items():
        ax1.plot(np.array(series["time"]) * 1e6, np.array(series["peak_displacement"]) * 1e3, marker="o", ms=3,
                 label=name)
    ax1.set_xlabel("time (µs)")
    ax1.set_ylabel("peak displacement (mm)")
    ax1.set_title(f"{len(metrics['histories'])} coupons, explicit FE", fontsize=10)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    fields = list(metrics["two_paths"]["vtk"].keys())
    x = np.arange(len(fields))
    w = 0.38
    ax2.bar(x - w / 2, [metrics["two_paths"]["vtk"][f] for f in fields], w, label="VTK round trip")
    decim = metrics["normalisation"]["decimation_error"]
    ax2.bar(x + w / 2, [decim.get(f, np.nan) for f in fields], w, label="node decimation")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(fields, rotation=15, fontsize=8)
    ax2.set_ylabel("relative error introduced")
    ax2.set_title("what each step costs, per field", fontsize=10)
    ax2.grid(alpha=0.25, axis="y")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="smaller meshes and two designs")
    ap.add_argument("--points", type=int, default=120, help="node budget for the decimation step (0 = keep every node)")
    ap.add_argument("--out", default=HERE, help="where results.json and the figure go")
    args = ap.parse_args(argv)

    designs = DESIGNS[:2] if args.smoke else DESIGNS
    runs = []
    print(f"1. {len(designs)} coupons through the explicit solver, into StructuralRun")
    for k, design in enumerate(designs):
        family = f"coupon_{k}"
        run = coupon(design, family, frames=4 if args.smoke else 6)
        runs.append(run)
        print(f"   {family}: {run.n_frames} frames, {run.n_points} nodes, {run.n_cells} cells, "
              f"{[p.name for p in run.parts]}, t = 0..{run.times[-1] * 1e6:.1f} µs, "
              f"checksum {run.checksum()[:12]}")

    print("\n2. the same run through two paths — the VTK series (what anim_to_vtk writes) against the in-memory run")
    work = os.path.join(os.path.dirname(HERE), "_out", "structural_ingestion")   # ignored by git
    vtk_error: dict[str, float] = {}
    for run in runs:
        series = os.path.join(work, f"vtk_{run.family()}")
        run.to_vtk_series(series, precision="single")
        back = StructuralRun.from_vtk_series(series, units="SI")
        for name in run.fields:
            vtk_error[name] = max(vtk_error.get(name, 0.0),
                                  relative(run.field(name), back.field(name)))
    for name, err in sorted(vtk_error.items()):
        print(f"   {name:<16} relative discrepancy {err:.3e}  (unit {runs[0].units[name]})")

    print("\n3. the staged OpenRadioss layout, read by the campaign's own artifact contract")
    case = StructuralRun.from_openradioss(staged_case(os.path.join(work, "radioss_case"), runs[0]), units="SI",
                                          part_names={1: "front", 2: "rear"})
    rad_error = relative(runs[0].field("displacement"), case.field("displacement"))
    print(f"   displacement relative discrepancy {rad_error:.3e} over {case.n_frames} frames; "
          f"elemental fields present at {case.info['fields_partial'].get('von_mises', 0)} frame(s) of "
          f"{case.n_frames} — that is what the reduction costs, and the run says so")

    print("\n4. the training set: dedup, decimation, units, non-dimensionalisation, split by family")
    budget = None if args.points in (0, None) else args.points
    ts = build_training_set(runs, max_points=budget, keep_cells=True, hold_out=(runs[-1].family(),))
    decimated = [d for d in ts.dropped if d["step"] == "decimate"]
    for d in decimated:
        print(f"   {d['family']}: {d['kept_points']} of {d['kept_points'] + d['dropped_points']} nodes, "
              f"{d['kept_cells']} cells kept — error "
              + ", ".join(f"{k} {v:.3f}" for k, v in d["error"].items()))
    print(f"   split {ts.split}   scales {({k: round(v, 6) for k, v in ts.scales.items()})}")
    print(f"   samples {ts.n_samples}   families {ts.families}")

    print("\n5. the Zarr round trip")
    zpath = os.path.join(work, "run.zarr")
    runs[0].to_zarr(zpath)
    back = StructuralRun.from_zarr(zpath)
    f32 = os.path.join(work, "run_f32.zarr")
    runs[0].to_zarr(f32, dtype="float32")
    back32 = StructuralRun.from_zarr(f32)
    print(f"   float64: checksum {'identical' if back.checksum() == runs[0].checksum() else 'DIFFERENT'} "
          f"({back.checksum()[:12]})")
    print(f"   float32: displacement relative difference "
          f"{relative(runs[0].field('displacement'), back32.field('displacement')):.3e} for half the bytes")

    metrics = {
        "designs": len(runs),
        "frames": sum(r.n_frames for r in runs),
        "points": sum(r.n_points for r in runs),
        "cells": sum(r.n_cells for r in runs),
        "parts": sum(len(r.parts) for r in runs),
        "histories": {r.family(): {"time": r.times.tolist(),
                                   "peak_displacement": np.abs(r.field("displacement")).max(axis=(1, 2)).tolist()}
                      for r in runs},
        "two_paths": {"vtk": vtk_error, "openradioss": rad_error},
        "normalisation": {"decimation_error": decimated[0]["error"] if decimated else {},
                          "kept_cells": [d["kept_cells"] for d in decimated],
                          "scales": ts.scales, "split": ts.split},
        "zarr": {"float64_checksum_match": back.checksum() == runs[0].checksum(),
                 "float32_relative": relative(runs[0].field("displacement"), back32.field("displacement"))},
    }
    fig = figure(metrics, os.path.join(args.out, "ingestion.png"))
    metrics["figure"] = os.path.relpath(fig, HERE)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1, sort_keys=True)

    print("\nmeasured:")
    print(f"   worst VTK discrepancy over every field: {max(vtk_error.values()):.3e}")
    print(f"   OpenRadioss layout displacement:       {rad_error:.3e}")
    print(f"   float32 store:                         {metrics['zarr']['float32_relative']:.3e}")
    print(f"results written to {os.path.relpath(args.out, HERE)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
