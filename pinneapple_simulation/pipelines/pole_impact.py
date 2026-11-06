"""Pole-impact dataset of the bumper-beam coupon: the crashworthiness quantities and the displacement sequence.

Two datasets come out of one campaign, which is how the crash benchmark is built:

- the **tabular** one — seven design variables in, the five crashworthiness quantities out (peak wall force, peak
  deceleration, peak internal energy, peak plastic work, kinetic-energy absorption, plus the impact duration): the
  component-scale regression benchmark, where gradient-boosted trees and neural nets are compared;
- the **field** one — the undeformed mesh plus its part labels in, the displacement of every node at every frame out:
  the structure-scale field benchmark where a part-aware surrogate is compared with point and operator models.

Both are written from the same runs, so a design cannot be in one and not the other, and the JSON manifest records
which solver produced them (this library's explicit hexahedral crash solver, not OpenRadioss), the end time, the
frame count, the mesh size and the design box.

    from pinneapple_simulation.pipelines.pole_impact import build_pole_impact_dataset
    summary = build_pole_impact_dataset(120, "runs/pole_impact", parallel=8)
    print(summary["quantities"], summary["n_samples"])
"""
from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["build_pole_impact_dataset", "quantities_table", "run_design"]


def run_design(design: dict, *, end_time: float = 1e-3, frames: int = 11, n_parts: int = 3) -> dict:
    """One design: the result, the node-level sample and the quantity row."""
    from pinneapple_simulation.numerical_solvers.crash_cases import pole_impact

    res = pole_impact(design, end_time=end_time, frames=frames, n_parts=n_parts)
    mesh = res.mesh
    node_part = np.zeros(len(mesh.nodes), int)
    for e, p in enumerate(res.parts):
        for node in mesh.elements[e]:
            if node_part[node] == 0:
                node_part[node] = int(p)
    qoi = {k: float(v) for k, v in res.qoi().items()}
    return {"design": dict(design), "qoi": qoi, "points": np.asarray(mesh.nodes, np.float32),
            "displacement": np.asarray(res.displacement, np.float32), "times": np.asarray(res.times, np.float32),
            "parts": node_part, "out_of_envelope": bool(res.out_of_envelope),
            "eroded_end": int(res.eroded[-1].sum()),
            "energies": {k: float(v[-1]) for k, v in res.energies.items()}, "dt": float(res.dt),
            "n_steps": int(len(res.time_history))}


def _worker(args):
    design, kwargs = args
    try:
        return run_design(design, **kwargs)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "design": dict(design)}


def quantities_table(rows: list[dict]) -> list[dict]:
    """The tabular view: design vector + the seven reported quantities, one row per design."""
    out = []
    for r in rows:
        if "error" in r:
            continue
        row = dict(r["design"])
        row.update(r["qoi"])
        row["out_of_envelope"] = r["out_of_envelope"]
        row["eroded_elements"] = r["eroded_end"]
        out.append(row)
    return out


def build_pole_impact_dataset(n_designs: int, out_dir: str, *, seed: int = 0, parallel: int = 1,
                              end_time: float = 1e-3, frames: int = 11, n_parts: int = 3,
                              designs: list[dict] | None = None, verbose: bool = True) -> dict:
    """Sample designs, run them (or ``parallel`` of them at a time) and write both datasets.

    Writes ``<out_dir>/pole_impact_fields.npz`` (a field dataset of displacement sequences),
    ``<out_dir>/pole_impact_quantities.csv`` (the tabular benchmark), ``<out_dir>/campaign.json`` (the manifest) and
    ``<out_dir>/failures.json``. Returns a summary dict."""
    from pinneapple_arena.fieldbench.protocol import FieldDataset, FieldSample, normalize_geometry
    from pinneapple_simulation.numerical_solvers.crash_cases import sample_design

    os.makedirs(out_dir, exist_ok=True)
    if designs is None:
        designs = sample_design(rng=np.random.default_rng(seed), n=int(n_designs))
    jobs = [(d, {"end_time": end_time, "frames": frames, "n_parts": n_parts}) for d in designs]
    results = []
    if parallel and parallel > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(int(parallel)) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, jobs)):
                results.append(res)
                if verbose:
                    print(f"  [{i + 1}/{len(jobs)}] {res.get('error', 'ok')}", flush=True)
    else:
        for i, job in enumerate(jobs):
            res = _worker(job)
            results.append(res)
            if verbose:
                print(f"  [{i + 1}/{len(jobs)}] {res.get('error', 'ok')}", flush=True)

    samples, rows, failures = [], [], []
    for i, res in enumerate(results):
        if "error" in res:
            failures.append(res)
            continue
        sid = f"pole{i:04d}"
        n_nodes = res["points"].shape[0]
        samples.append(FieldSample(id=sid, points=res["points"], values=res["displacement"],
                                   category="pole_impact", parts=res["parts"], part_ids=res["parts"],
                                   thickness=np.ones(n_nodes, np.float32), times=res["times"],
                                   meta={"design": res["design"], "dt": res["dt"], "energies": res["energies"]}))
        rows.append({**res, "id": sid})
    if not samples:
        raise RuntimeError("no design completed: see the failures listed above")
    dataset = normalize_geometry(FieldDataset(name=f"pole_impact_fields[{len(samples)}]", samples=samples))
    npz = os.path.join(out_dir, "pole_impact_fields.npz")
    dataset.save(npz)
    table = quantities_table(rows)
    csv = os.path.join(out_dir, "pole_impact_quantities.csv")
    keys = list(table[0])
    with open(csv, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in table:
            fh.write(",".join(f"{r[k]:.10g}" if isinstance(r[k], (float, np.floating)) else str(r[k])
                              for k in keys) + "\n")
    summary = {"fields": npz, "quantities": csv, "n_samples": len(samples), "n_failed": len(failures),
               "end_time_s": end_time, "frames": frames, "n_parts": n_parts,
               "nodes": int(samples[0].points.shape[0]), "elements_note": "see campaign.json",
               "out_of_envelope": int(sum(bool(r["out_of_envelope"]) for r in rows)),
               "eroded_mean": float(np.mean([r["eroded_end"] for r in rows])),
               "splits": {k: len(v) for k, v in dataset.splits.items()}}
    with open(os.path.join(out_dir, "campaign.json"), "w") as fh:
        json.dump({"generator": "pinneapple_simulation.numerical_solvers.crash_fem.CrashSolver",
                   "case": "bumper beam + two crash boxes against a rigid pole (sled view)",
                   "solver_note": "explicit hexahedral dynamics, J2 plasticity, penalty contact; not OpenRadioss",
                   "summary": summary, "designs": [r["design"] for r in rows]}, fh, indent=2)
    with open(os.path.join(out_dir, "failures.json"), "w") as fh:
        json.dump(failures, fh, indent=2)
    return summary
