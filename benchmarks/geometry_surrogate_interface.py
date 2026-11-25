"""The geometry-agnostic surrogate interface, measured on a real solver (#293).

The question this case answers is not "which model is best" — that needs the car benchmarks — but the one a user asks
first: **does the interface work, and what does a model behind it cost and give on real simulation data?**

1. **Geometry -> data.** A family of 2-D bodies (CST thickness and camber weights, the airfoil family the library's
   aerodynamic work uses), labelled by the Hess-Smith panel method: the inviscid reference the library already treats
   as ground truth, cheap enough that a label costs milliseconds. Each body becomes a :class:`GeometrySample` — the
   panel mid-points, the outward normal of each panel, the panel length, the angle of attack as a condition, the
   ``cp`` field on every point, and ``cl``/``cd`` as integrals.
2. **Geometry -> model.** Two families behind the same interface: ``transolver`` (physics-attention over learned
   slices, Wu et al. 2024) and ``pointnet`` (shared MLP plus a global max pool, Qi et al. 2017). Same features, same
   optimiser, same budget — the only difference is the architecture, which is what makes the comparison a comparison.
   ``gino`` (Li et al. 2023) is *not* trained here and the JSON says why; the interface it would plug into is the one
   measured.
3. **The split.** Training shapes are a Latin hypercube inside the documented CST ranges; held out are shapes from
   the same box that no model ever saw. Three angles of attack per shape, so the conditions are non-trivial.
4. **The report.** Relative L2 of the ``cp`` field against the solver, the error of the lift coefficient *integrated
   from the predicted field* with the solver's own pressure integration, the rank correlation of the designs by lift
   (a field error that does not reorder the designs costs nothing at the decision level), the parameters, the training
   time and the peak memory.

    python -m benchmarks.geometry_surrogate_interface            # the protocol, a few minutes on one CPU
    python -m benchmarks.geometry_surrogate_interface --quick    # the test budget, well under a minute
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np

from pinneapple_analysis.surrogate.geometry import (
    GeometrySample,
    PointNetGeometry,
    TransolverGeometry,
    available_families,
    panel_integrals,
    samples_from_panel,
)
from pinneapple_design.aero.geometry import BOUNDS, REFERENCE, latin_hypercube, valid

__all__ = ["PROTOCOL", "OUT", "bodies", "run"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "benchmarks", "_out", "geometry_surrogate_interface.json")

#: what is trained, on what, with which budget. ``seeds`` are training seeds over one design set: the spread between
#: them is reported next to the mean, because a single seed is a point and not a distribution.
PROTOCOL = {
    "train_shapes": 36,
    "held_shapes": 8,
    "alphas_deg": [0.0, 4.0, 8.0],
    "n_per_side": 60,
    "epochs": 60,
    "batch": 8,
    "lr": 2e-3,
    "hidden": 96,
    "depth": 4,
    "slices": 16,
    "seeds": [0, 1],
    "families": ["transolver", "pointnet"],
}

QUICK = dict(PROTOCOL) | {"train_shapes": 10, "held_shapes": 4, "alphas_deg": [0.0, 4.0], "n_per_side": 20,
                          "epochs": 4, "seeds": [0]}


def bodies(n: int, seed: int = 0) -> list[np.ndarray]:
    """``n`` valid CST shapes from a Latin hypercube over the documented ranges (invalid draws are redrawn)."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    while len(out) < n:
        candidate = latin_hypercube(1, BOUNDS, rng)[0]
        if valid(candidate)[0]:
            out.append(candidate)
    return out


def _rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if os.uname().sysname == "Darwin" else peak / 1024


def run(*, protocol: dict | None = None, verbose: bool = True) -> dict:
    protocol = dict(PROTOCOL if protocol is None else protocol)
    shapes = bodies(protocol["train_shapes"] + protocol["held_shapes"], seed=0)
    train_shapes = shapes[:protocol["train_shapes"]]
    held_shapes = shapes[protocol["train_shapes"]:]

    t = time.perf_counter()
    train: list[GeometrySample] = samples_from_panel(train_shapes, alphas=tuple(protocol["alphas_deg"]),
                                                    n_per_side=protocol["n_per_side"])
    held: list[GeometrySample] = samples_from_panel(held_shapes, alphas=tuple(protocol["alphas_deg"]),
                                                   n_per_side=protocol["n_per_side"])
    label_seconds = time.perf_counter() - t
    if verbose:
        print(f"dados: {len(train)} amostras de treino e {len(held)} de teste, {train[0].n_points} painéis por "
              f"corpo, {label_seconds:.2f} s de solver")
    # the labels are the solver's, and the integrator is the solver's own: check they agree before training anything
    truth = train[0]
    check = panel_integrals(fields=truth.fields, normals=truth.normals, elements=truth.elements,
                            conditions=truth.conditions)
    if abs(check["cl"] - truth.integrals["cl"]) > 1e-9:
        raise AssertionError(f"the integrator does not reproduce the solver's Cl ({check['cl']} vs "
                             f"{truth.integrals['cl']})")

    catalogue = available_families()
    report: dict = {
        "schema": "pinneapple.geometry_surrogate_interface/1",
        "protocol": {k: v for k, v in protocol.items()},
        "data": {"train_samples": len(train), "held_samples": len(held), "points_per_body": train[0].n_points,
                 "solver": "pinneapple_design.aero.panel.panel_solve (Hess-Smith), Cp on the panel mid-points",
                 "label_seconds": label_seconds, "integrator_matches_solver": check["cl"]},
        "families": catalogue,
        "models": {},
    }

    for family in protocol["families"]:
        rows = []
        for seed in protocol["seeds"]:
            model = (TransolverGeometry if family == "transolver" else PointNetGeometry)(
                fields=("cp",), hidden=protocol["hidden"], depth=protocol["depth"], seed=seed,
                **({"slices": protocol["slices"]} if family == "transolver" else {}),
                integrate=panel_integrals)
            t = time.perf_counter()
            model.fit(train, epochs=protocol["epochs"], batch=protocol["batch"], lr=protocol["lr"], seed=seed)
            history = model.fit_report
            fit_seconds = time.perf_counter() - t
            evaluation = model.evaluate(held)
            summary = evaluation.summary()
            row = {"seed": seed, "parameters": history["parameters"], "fit_seconds": fit_seconds,
                   "loss_first": history["loss_first"], "loss_last": history["loss_last"],
                   "relative_l2_mean": summary["relative_l2_mean"],
                   "relative_l2_worst": summary["relative_l2_worst"],
                   "cl_error_mean": summary["integral_error:cl"]["mean"],
                   "cl_error_worst": summary["integral_error:cl"]["worst"],
                   "rank_r2": summary.get("rank_r2"), "peak_rss_mb": _rss_mb(),
                   "predict_seconds": _predict_seconds(model, held)}
            rows.append(row)
            if verbose:
                print(f"{family:11s} seed {seed}: {row['parameters']:6d} parâmetros, {fit_seconds:5.1f} s de treino, "
                      f"L2 rel. {row['relative_l2_mean']:.3f} (pior {row['relative_l2_worst']:.3f}), "
                      f"erro de Cl {row['cl_error_mean']:.4f}, R² do ranking {row['rank_r2']}")
        report["models"][family] = {
            "seeds": rows,
            "relative_l2_mean": float(np.mean([r["relative_l2_mean"] for r in rows])),
            "cl_error_mean": float(np.mean([r["cl_error_mean"] for r in rows])),
            "fit_seconds_mean": float(np.mean([r["fit_seconds"] for r in rows])),
            "parameters": rows[0]["parameters"],
        }
    return report


def _predict_seconds(model, samples: list[GeometrySample], repeats: int = 3) -> float:
    """Cost of one prediction on a held-out body, median of a few calls: what an online query costs."""
    times = []
    for s in samples[:repeats]:
        t = time.perf_counter()
        model.predict(s.points, s.normals, s.conditions, s.elements)
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="the test budget: fewer shapes, fewer epochs")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    report = run(protocol=QUICK if args.quick else PROTOCOL)
    report["mode"] = "quick" if args.quick else "protocol"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nrelatório em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
