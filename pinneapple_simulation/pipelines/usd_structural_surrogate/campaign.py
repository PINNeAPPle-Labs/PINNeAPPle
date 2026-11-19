"""The CalculiX campaign: N designs meshed and solved by ``ccx``, each turned into one grid sample.

    from pinneapple_simulation.pipelines.usd_structural_surrogate import DesignSpace, build_campaign
    space = DesignSpace.from_interface()
    summary = build_campaign(space, 24, "runs/campaign", element_size=0.022, grid_n=(40, 40, 56), parallel=6)

One process per design (``ccx`` is single-job). Every sample is ``<out>/samples/sample_XXX.npz`` holding

``design``      (4,)          the four design variables, in metres
``occupancy``   (nx,ny,nz)    which cells are inside the part
``u``           (nx,ny,nz,3)  displacement on the grid, metres under the reference load
``vm``          (nx,ny,nz)    von Mises on the grid, pascals under the reference load
``nodes``/``elements``/``u_nodes``/``vm_nodes``
                              the finite-element result the grid was sampled from, kept so a prediction can be
                              checked against the solver at the nodes themselves and not only through the grid

and ``<out>/manifest.json`` records the design space, the asset, the solver settings and per-sample timings. A run
that already has a sample is skipped, so a campaign can be resumed after an interruption.
"""
from __future__ import annotations

import json
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .design import DesignSpace, run_design
from .grid import GridSpec, fill_outside, rasterize_barycentric
from .grid import occupancy as grid_occupancy

__all__ = ["build_campaign", "load_samples", "default_grid"]


def default_grid(space: DesignSpace) -> GridSpec:
    """The fixed window that holds every design of ``space``, with half a plate radius of margin.

    The box is fixed for the whole family -- that is what makes every sample land on the same lattice -- and sized
    from the space's own bounds (the tallest ``h``, the largest flare) rather than from one design.
    """
    r_plate = space.plate_diameter / 2.0
    r_flare = space.bounds["r"][1] + space.bounds["dc"][1] / 2.0
    rad = max(r_plate, r_flare)
    pad = 0.08 * rad
    return GridSpec.around([-rad - pad, -rad - pad, 0.0],
                           [rad + pad, rad + pad, space.bounds["h"][1] + pad * 0.5],
                           n=[40, 40, 56])


def _worker(job: dict[str, Any]) -> dict[str, Any]:
    """One design: mesh, solve, sample onto the grid, save. Runs in its own process."""
    t0 = time.time()
    out = Path(job["out"])
    tag = job["tag"]
    path = out / "samples" / f"sample_{tag}.npz"
    try:
        res = run_design(job["design"], str(out / "work" / tag), element_size=job["element_size"],
                         order=job["order"], threads=job["threads"], keep=False)
        grid = GridSpec.from_dict(job["grid"])
        occ = grid_occupancy(res["nodes"], res["elements"], grid, res["element_type"])
        u = fill_outside(rasterize_barycentric(res["nodes"], res["u"], grid, res["elements"],
                                               res["element_type"], inside=occ), occ, grid)
        vm = fill_outside(rasterize_barycentric(res["nodes"], res["von_mises"], grid, res["elements"],
                                                res["element_type"], inside=occ), occ, grid)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, design=np.array([res["design"][k] for k in job["names"]], np.float32),
                            occupancy=occ.astype(np.uint8), u=u.astype(np.float32), vm=vm.astype(np.float32),
                            nodes=res["nodes"].astype(np.float32), elements=res["elements"].astype(np.int32),
                            u_nodes=res["u"].astype(np.float32), vm_nodes=res["von_mises"].astype(np.float32),
                            names=np.array(job["names"]))
        return {"tag": tag, "ok": True, "path": str(path), "seconds": round(time.time() - t0, 2),
                "n_nodes": res["info"]["n_nodes"], "n_elements": res["info"]["n_elements"],
                "occupied_cells": int(occ.sum()), "design": res["design"]}
    except Exception as e:                                          # one bad design must not kill the campaign
        return {"tag": tag, "ok": False, "seconds": round(time.time() - t0, 2), "design": job["design"],
                "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}


def build_campaign(space: DesignSpace, n: int, out: str | Path, *, element_size: float = 0.022,
                   order: int = 1, grid: GridSpec | None = None, parallel: int = 4, seed: int = 0,
                   method: str = "lhs", threads: int = 1, designs: Sequence[dict[str, float]] | None = None,
                   resume: bool = True) -> dict[str, Any]:
    """Solve ``n`` designs and write the dataset. Returns the manifest (also written to ``<out>/manifest.json``)."""
    out = Path(out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    grid = grid or default_grid(space)
    chosen = list(designs) if designs is not None else space.sample(n, seed=seed, method=method)
    names = list(space.bounds)

    jobs = []
    for i, d in enumerate(chosen):
        tag = f"{i:04d}"
        if resume and (out / "samples" / f"sample_{tag}.npz").exists():
            jobs.append({"tag": tag, "skip": True, "design": d})
            continue
        jobs.append({"tag": tag, "design": d, "out": str(out), "element_size": element_size, "order": order,
                     "grid": grid.as_dict(), "names": names, "threads": threads})

    t0 = time.time()
    results: list[dict[str, Any]] = [{"tag": j["tag"], "ok": True, "skipped": True, "design": j["design"]}
                                     for j in jobs if j.get("skip")]
    todo = [j for j in jobs if not j.get("skip")]
    if parallel <= 1:
        results += [_worker(j) for j in todo]
    else:
        with ProcessPoolExecutor(max_workers=parallel) as ex:
            futures = [ex.submit(_worker, j) for j in todo]
            for f in as_completed(futures):
                results.append(f.result())

    ok = [r for r in results if r["ok"]]
    manifest = {
        "schema": "pinneapple.usd_structural_surrogate/1",
        "space": space.as_dict(),
        "grid": grid.as_dict(),
        "solver": {"name": "CalculiX ccx", "element": "C3D4" if order == 1 else "C3D10",
                   "element_size_m": element_size, "step": "linear static, unit horizontal load"},
        "reference_load_n": space.reference_load,
        "n_requested": len(chosen), "n_ok": len(ok), "n_failed": len(results) - len(ok),
        "sampling": {"method": method, "seed": seed},
        "seconds": round(time.time() - t0, 2),
        "samples": sorted(results, key=lambda r: r["tag"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_samples(out: str | Path, fields: Sequence[str] = ("design", "occupancy", "u", "vm")) -> dict[str, Any]:
    """Every sample of a campaign, as arrays.

    Fields that have the same shape in every sample (``design``, ``occupancy``, ``u``, ``vm``) come back stacked as
    ``(N, ...)``; fields whose shape changes with the design (``nodes``, ``elements``, ``u_nodes``, ``vm_nodes``)
    come back as a list of ``N`` arrays, which is what they are -- a different mesh per design.
    """
    out = Path(out)
    files = sorted((out / "samples").glob("sample_*.npz"))
    if not files:
        raise FileNotFoundError(f"no samples under {out / 'samples'}")
    per: dict[str, list] = {f: [] for f in fields}
    for p in files:
        with np.load(p) as z:
            for f in fields:
                per[f].append(z[f])
    stacked: dict[str, Any] = {}
    for f, v in per.items():
        try:
            stacked[f] = np.stack(v)
        except ValueError:                                  # ragged: one mesh per design
            stacked[f] = list(v)
    return stacked
