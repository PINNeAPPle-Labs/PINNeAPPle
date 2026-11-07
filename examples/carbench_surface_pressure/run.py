"""Surface pressure over a family of car bodies — the vehicle-aerodynamics surrogate benchmark, recreated here.

What is recreated, and with what:

| the benchmark | here |
|---|---|
| 8,121 parametric cars from 8 archetypes (3 roof lines x underbody x wheels) | the library's parametric ``RoadCar`` with the same three roof lines and open/closed arches, sampled over 13 numeric design variables |
| labels: steady RANS (k-omega SST) on ~450k-point surface meshes | labels: 3-D lattice-Boltzmann (D3Q19, Smagorinsky) in the library's wind tunnel, wall-face kinematic pressure, a few thousand faces per car |
| 11 surrogates (point, graph, transformer, triplane) | the point and operator families the library has natively (Transolver, its lite variant, PointNet, a wide residual point model, EdgeConv, a part-aware point model) plus the hierarchical model below |
| per-geometry relative L2 (mean over test geometries), pooled R^2, bootstrap intervals, error percentiles, latency | the same, in ``pinneapple_arena.fieldbench`` |
| the three-resolution protocol (subsampled / interpolated / patchwise) | ``PatchwiseProtocol``, scored at the end of the run |
| a cross-category zero-shot split | the same, with the roof line as the category |

What is *not* the same, stated up front: the label is a coarse LBM field, not a RANS one, so absolute error levels are
not comparable with the published table — what is comparable is the protocol, the ranking of families, and the cost.
The dataset records its resolution and the Reynolds number reachable at that resolution for that reason.

Run: ``python examples/carbench_surface_pressure/run.py --designs 48 --parallel 8``
(``--smoke``: 4 designs, tiny resolution, 2 epochs — a minute on a laptop).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import pinneapple_lab  # noqa: E402
from pinneapple_arena.fieldbench import (  # noqa: E402
    FieldDataset,
    PatchwiseProtocol,
    build_model_specs,
    mean_relative_l2,
    paired_bootstrap,
    run_benchmark,
    write_report,
)


def label_quality(csv_path: str, failures_path: str) -> dict:
    """The label-quality numbers of a campaign, read back from the coefficients table it wrote.

    The table is the campaign's own record (one row per labelled design), so a ``--reuse`` run reports the same
    resolution, Reynolds number and density excursion as the run that produced the labels."""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"{csv_path} has no labelled design")
    n_failed = 0
    if os.path.exists(failures_path):
        with open(failures_path) as fh:
            n_failed = len(json.load(fh))
    return {"n_failed": n_failed,
            "cd_mean": float(np.mean([float(r["cd"]) for r in rows])),
            "max_density_deviation": float(np.max([float(r["max_density_deviation"]) for r in rows])),
            "re_mean": float(np.mean([float(r["re"]) for r in rows])),
            "cells_per_length": float(np.mean([float(r["cells_per_length"]) for r in rows])),
            "steps": float(np.mean([float(r["steps"]) for r in rows]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", type=int, default=48)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--cells-per-length", type=int, default=40)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--road", action="store_true",
                    help="put the road under the car (the ride height has to be resolved by several cells for the "
                         "ground effect not to be a staircase nozzle; the default labels the body in free air)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--scale", choices=("small", "paper"), default="small")
    ap.add_argument("--max-faces", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse the dataset if it is already there")
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    a = ap.parse_args()

    designs, cpl, steps, epochs = (4, 20, 300, 2) if a.smoke else (a.designs, a.cells_per_length, a.steps, a.epochs)
    os.makedirs(a.out, exist_ok=True)
    npz = os.path.join(a.out, "surface_pressure.npz")

    with pinneapple_lab.experiment(
            "examples/carbench_surface_pressure",
            title="Surface-pressure surrogate benchmark over a parametric car family",
            description="Geometry point cloud in, surface kinematic pressure out, over a parametric car family "
                        "(three roof lines, open/closed wheel arches); labels from the library's lattice-Boltzmann "
                        "tunnel; six surrogate families ranked with per-geometry relative L2, bootstrap intervals "
                        "and the paired significance test.",
            domain="external aerodynamics",
            methods=["lattice Boltzmann (D3Q19, Smagorinsky)", "physics-attention transformer", "PointNet",
                     "EdgeConv", "hierarchical part-aware surrogate"],
            modules=["pinneapple_simulation.pipelines.car_pressure", "pinneapple_arena.fieldbench"],
            out_dir=HERE, register=False,
            links={"guide": "docs/guides/crash_and_surface_benchmarks.md",
                   "script": "examples/carbench_surface_pressure/run.py"}) as ex:
        # ---------------------------------------------------------------- 1. the labels
        if a.reuse and os.path.exists(npz):
            dataset = FieldDataset.load(npz)
            print(f"reused {npz}: {len(dataset)} geometries")
        else:
            from pinneapple_simulation.pipelines.car_pressure import build_pressure_dataset
            build_pressure_dataset(designs, a.out, seed=a.seed, parallel=a.parallel,
                                   cells_per_length=cpl, steps=steps, averaging=1, u_lattice=0.08,
                                   Re=300.0, max_faces=a.max_faces, road=bool(a.road), verbose=True)
            dataset = FieldDataset.load(npz)
        # the label-quality numbers are read back from the campaign's own coefficients table, so they are recorded by
        # a --reuse run as well (the labels are the same run's: the table is written next to the dataset)
        labels = label_quality(os.path.join(a.out, "coefficients.csv"), os.path.join(a.out, "failures.json"))
        ex.log_metric("geometries_labelled", len(dataset), "geometries",
                      f"lattice-Boltzmann, {labels['cells_per_length']:.0f} cells per car length, "
                      f"{labels['steps']:.0f} steps, Re {labels['re_mean']:.0f}")
        ex.log_metric("label_max_density_deviation", labels["max_density_deviation"], "fraction of rho",
                      "largest density excursion of the lattice over the campaign: how far the label is from "
                      "the incompressible limit")
        ex.log_metric("label_cd_mean", labels["cd_mean"], "drag coefficient",
                      "mean over the labelled designs, coarse lattice, front area of every design")
        ex.log_metric("failed_designs", labels["n_failed"], "designs", "listed in failures.json")
        ex.log_metric("wall_faces_per_geometry", int(np.mean([s.n_points for s in dataset.samples])), "faces",
                      "wall faces of the voxelized body, = input points of the surrogate")
        ex.log_metric("categories", len(dataset.categories), "categories", ", ".join(sorted(dataset.categories)))

        # ---------------------------------------------------------------- 2. the benchmark
        specs = build_model_specs(transient=False, scale=a.scale, n_parts=4)
        result = run_benchmark(specs, dataset, epochs=epochs, batch_size=a.batch, device=None, patience=12,
                               seed=a.seed, verbose=True, n_boot=2000)
        print("\n" + result["table"] + "\n")
        write_report(result, os.path.join(a.out, "leaderboard.md"))

        # ---------------------------------------------------------------- 3. the resolutions and the significance
        best = result["scores"][int(np.argmin([s.rel_l2 for s in result["scores"]]))]
        second = sorted(result["scores"], key=lambda s: s.rel_l2)[1]
        paired = paired_bootstrap(best.scores, second.scores)
        ex.log_metric("best_rel_l2", best.rel_l2, "relative L2", f"{best.name}, mean over {result['n_test']} "
                                                                   f"held-out geometries, 95% CI {best.extra['ci95']}")
        ex.log_metric("best_r2", best.r2, "R^2", f"{best.name}, pooled over all test points")
        ex.log_metric("pair_gap", paired["mean_difference"], "relative L2",
                      f"{best.name} minus {second.name}, paired bootstrap, win rate {paired['win_rate']:.2f}, "
                      f"p_perm {paired['p_permutation']:.3g}")
        protocol = PatchwiseProtocol("subsampled")
        subs = [protocol.score(s.points, s.values[:, 0], s.points[::2],
                               result["predictions"][best.name][i][::2]) for i, s in enumerate(
            [dataset[j] for j in dataset.splits["test"]])]
        ex.log_metric("subsampled_rel_l2", mean_relative_l2([s["pred"] for s in subs], [s["ref"] for s in subs]),
                      "relative L2", f"{best.name} scored on every second wall face only")
        ex.check("a surrogate beats the mean field", best.rel_l2 < 1.0,
                 detail=f"best {best.name} Rel L2 {best.rel_l2:.3f}", kind="regression")
        ex.check("labels are physical", float(np.mean([m["cd"] for m in [s.meta for s in dataset.samples]])) > 0.0,
                 detail="drag coefficient of every geometry is positive", kind="physics")
        ex.add_file(os.path.join(a.out, "leaderboard.md"))
        ex.add_file(os.path.join(a.out, "coefficients.csv"))
        print(f"leaderboard: {os.path.join(a.out, 'leaderboard.md')}")
        print(f"winner: {best.name}  Rel L2 {best.rel_l2:.4f}  R² {best.r2:.4f}")


if __name__ == "__main__":
    main()
