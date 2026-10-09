"""Run the OpenFOAM training set: LHS airfoil shapes x angles of attack, plus NACA reference polars.

    python apps/aero_optimizer/tools/generate_dataset.py OUT_DIR [--shapes 100] [--alphas 4] [--workers 4]

Each run leaves OUT_DIR/runs/<id>.json (inputs, coefficients, convergence, wall Cp/Cf) and <id>.npz (cell fields
p, Ux, Uy, nut on the shared O-grid). Re-running skips finished runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pinneapple_design.aero.case import read_internal, run_case          # noqa: E402
from pinneapple_design.aero.geometry import BOUNDS, REFERENCE, latin_hypercube, valid  # noqa: E402

RE = 4e6
ALPHA_BINS = [(-2, 2), (2, 6), (6, 10), (10, 14)]


def plan(n_shapes: int, n_alpha: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    shapes = []
    while len(shapes) < n_shapes:
        for s in latin_hypercube(n_shapes, BOUNDS, rng):
            if valid(s)[0] and len(shapes) < n_shapes:
                shapes.append(s)
    jobs = []
    for k, s in enumerate(shapes):
        bins = ALPHA_BINS if n_alpha == 4 else [(-2 + 16 * i / n_alpha, -2 + 16 * (i + 1) / n_alpha) for i in range(n_alpha)]
        for b, (lo, hi) in enumerate(bins):
            jobs.append({"id": f"s{k:03d}_a{b}", "set": "lhs", "shape_id": f"s{k:03d}", "shape": s.tolist(),
                         "alpha": float(rng.uniform(lo, hi))})
    for name, s in REFERENCE.items():
        for a in (-2, 0, 2, 4, 6, 8, 10, 12, 14):
            jobs.append({"id": f"{name.replace(' ', '')}_a{a:+03d}", "set": "reference", "shape_id": name,
                         "shape": s.tolist(), "alpha": float(a)})
    return jobs


def one(job, out):
    case = os.path.join(out, "cases", job["id"])
    r = run_case(case, np.array(job["shape"]), job["alpha"], reynolds=RE)
    rec = {**job, "reynolds": RE, **{k: v for k, v in r.items() if k not in ("time_dir",)}}
    d = r.get("time_dir")
    if d:
        p = read_internal(os.path.join(d, "p"))
        U = read_internal(os.path.join(d, "U"))
        nut = read_internal(os.path.join(d, "nut"))
        np.savez_compressed(os.path.join(out, "runs", job["id"] + ".npz"), p=p.astype(np.float32),
                            Ux=U[:, 0].astype(np.float32), Uy=U[:, 1].astype(np.float32), nut=nut.astype(np.float32))
    with open(os.path.join(out, "runs", job["id"] + ".json"), "w") as f:
        json.dump(rec, f)
    shutil.rmtree(case, ignore_errors=True)
    return job["id"], r.get("status"), r.get("seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shapes", type=int, default=100)
    ap.add_argument("--alphas", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", choices=["lhs", "reference"], default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    jobs = [j for j in plan(a.shapes, a.alphas) if (a.only is None or j["set"] == a.only)
            and not os.path.exists(os.path.join(a.out, "runs", j["id"] + ".json"))]
    jobs.sort(key=lambda j: j["set"] != "reference")                 # references first: validation early
    print(f"{len(jobs)} runs to do", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, j, a.out) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            try:
                print(i + 1, *f.result(), flush=True)
            except Exception as e:                                    # noqa: BLE001
                print(i + 1, "error", e, flush=True)


if __name__ == "__main__":
    main()
