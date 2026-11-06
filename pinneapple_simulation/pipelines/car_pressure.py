"""Surface-pressure dataset of a parametric car family, from the 3-D lattice-Boltzmann tunnel.

This is the data generator of the surface-pressure benchmark: a design vector drives the library's parametric road car
(``pinneapple_design.geometry.vehicles3d.RoadCar``) or a user-supplied mesh, the body is voxelized into the LBM tunnel,
the flow is run to a steady state, and the **kinematic pressure on the body surface** is sampled at the centres of the
wall faces — the field a surrogate is then asked to reproduce from the geometry alone.

Three roof lines (fastback, notchback, estateback), an underbody (the diffuser angle) and open or closed wheel arches
make the categories of the family; the numeric variables (lengths, angles, wheel size, ride height) are the design
space inside a category. Holding a whole category out gives the zero-shot cross-category experiment.

What this replaces, and what it does not: the published vehicle-aerodynamics benchmark labels its designs with
steady RANS (k-omega SST) on ~450k-point surface meshes; this generator labels them with a coarse lattice-Boltzmann
run (Smagorinsky LES) on a few thousand wall faces. The *protocol* is the same — geometry point cloud in, surface
pressure out, per-geometry splits — the physics is coarser, and the absolute error levels are those of a coarse LBM
label, not of a RANS one. Every dataset records the resolution, the Reynolds number reachable at that resolution and
the model used, so a result cannot be read as if it came from the finer reference.

    from pinneapple_simulation.pipelines.car_pressure import (AERO_DESIGN_SPACE, sample_designs,
                                                              simulate_pressure, build_pressure_dataset)
    designs = sample_designs(48, seed=0)
    sample = simulate_pressure(designs[0], cells_per_length=40, steps=2000)
    print(sample["points"].shape, sample["pressure"].mean())
"""
from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["AERO_DESIGN_SPACE", "ROOF_LINES", "sample_designs", "car_surfaces", "simulate_pressure",
           "build_pressure_dataset", "design_category"]

ROOF_LINES = ("fastback", "notchback", "estateback")

# numeric design variables (SI) and their ranges; the category variables are sampled separately
AERO_DESIGN_SPACE = {
    "length": (4.2, 5.0),
    "width": (1.75, 1.95),
    "roof_height": (1.30, 1.50),
    "clearance": (0.10, 0.18),
    "deck_height": (0.95, 1.12),
    "windshield_deg": (22.0, 34.0),
    "slant_deg": (14.0, 32.0),
    "boat_tail": (0.02, 0.12),
    "diffuser_deg": (0.0, 12.0),
    "wheel_radius": (0.31, 0.37),
    "wheel_width": (0.20, 0.28),
    "wheelbase": (2.65, 3.00),
    "front_overhang": (0.85, 1.00),
}


def design_category(design: dict) -> str:
    """The category string a split is made on: roof line + wheel arches."""
    return f"{design.get('roof_line', 'fastback')}/{design.get('wheels', 'open')}"


def sample_designs(n: int, seed: int = 0, roof_lines=ROOF_LINES, wheels=("open", "closed")) -> list[dict]:
    """``n`` designs: a Latin-hypercube-like sample of the numeric box, with the category cycled so that every
    (roof line, wheels) combination gets the same number of designs."""
    rng = np.random.default_rng(seed)
    combos = [(r, w) for r in roof_lines for w in wheels]
    out = []
    for i in range(int(n)):
        roof, wheel = combos[i % len(combos)]
        d = {"roof_line": roof, "wheels": wheel}
        for key, (lo, hi) in AERO_DESIGN_SPACE.items():
            u = (i // len(combos) + rng.random()) / max(1, (n + len(combos) - 1) // len(combos))
            d[key] = float(lo + (hi - lo) * min(max(u, 0.0), 1.0))
        if roof == "estateback":                       # an estate is the fastback profile with the roof carried back
            d["slant_deg"] = float(rng.uniform(68.0, 82.0))
            d["deck_height"] = float(rng.uniform(1.08, 1.20))
        out.append(d)
    return out


def car_surfaces(design: dict):
    """The closed surfaces of the design (body and wheels) as ``{name: (vertices, faces)}``, in metres."""
    from pinneapple_design.geometry.vehicles3d import RoadCar

    roof = design.get("roof_line", "fastback")
    style = "notchback" if roof == "notchback" else "fastback"
    # closed wheel arches: the body sits lower, so its skirt covers the wheels (the open/closed pair of the family)
    clearance = float(design.get("clearance", 0.13))
    if design.get("wheels", "open") == "closed":
        clearance *= 0.5
    car = RoadCar(length=float(design.get("length", 4.6)), width=float(design.get("width", 1.86)),
                  roof_height=float(design.get("roof_height", 1.36)),
                  clearance=clearance,
                  deck_height=float(design.get("deck_height", 1.0)),
                  windshield_deg=float(design.get("windshield_deg", 28.0)),
                  slant_deg=float(design.get("slant_deg", 22.0)), style=style,
                  boat_tail=float(design.get("boat_tail", 0.06)),
                  diffuser_deg=float(design.get("diffuser_deg", 6.0)),
                  wheel_radius=float(design.get("wheel_radius", 0.35)),
                  wheel_width=float(design.get("wheel_width", 0.25)),
                  wheelbase=float(design.get("wheelbase", 2.80)),
                  front_overhang=float(design.get("front_overhang", 0.92)), n_x=160, n_ring=96)
    return dict(car.cfd_bodies())


def _voxelize(bodies: dict, shape, dx: float, origin, road_z: float = 0.0) -> np.ndarray:
    """Solid mask of the union of the closed surfaces on the tunnel grid."""
    import trimesh

    from pinneapple_design.geometry.ops.lbm_bridge import mesh_to_obstacle_mask
    bounds = {"x": (origin[0], origin[0] + shape[0] * dx), "y": (origin[1], origin[1] + shape[1] * dx),
              "z": (road_z, road_z + shape[2] * dx)}
    mask = np.zeros(tuple(shape), dtype=bool)
    for verts, faces in bodies.values():
        mesh = trimesh.Trimesh(vertices=np.asarray(verts, float), faces=np.asarray(faces, int), process=False)
        mask |= np.asarray(mesh_to_obstacle_mask(mesh, bounds, tuple(shape), oracle_supersample=1), dtype=bool)
    return mask


def simulate_pressure(design: dict, *, cells_per_length: int = 40, steps: int = 2500, averaging: int = 3,
                      Re: float = 400.0, u_lattice: float = 0.06, smagorinsky: float = 0.12,
                      shapes: dict | None = None, device: str = "cpu", max_faces: int = 6000,
                      seed: int = 0) -> dict:
    """Run one design and return its surface samples and its aerodynamic coefficients.

    Returns a dict with ``points`` (M, 3) in metres, ``pressure`` (M,) kinematic pressure in lattice units (the cell is
    one unit of length and the step one of time, so ``p = c_s² (rho - 1)``), ``normal`` (M, 3) the outward wall normal,
    ``cd``/``cl`` and ``frontal_area`` [m²], ``re`` and ``re_max`` (the lattice's own Reynolds number at that
    resolution), and ``surface_area``.

    ``averaging`` runs the solver ``averaging`` times the requested steps and averages the wall pressure over the
    samples of the last third of the run — a steady RANS label is a mean field, and a coarse LBM behind a car sheds."""
    from pinneapple_simulation.numerical_solvers.lbm_tunnel import WindTunnelLBM

    bodies = car_surfaces(design)
    lo = np.min([np.min(np.asarray(v, float), axis=0) for v, _ in bodies.values()], axis=0)
    hi = np.max([np.max(np.asarray(v, float), axis=0) for v, _ in bodies.values()], axis=0)
    L = float(hi[0] - lo[0])
    n = max(16, int(cells_per_length))
    dx = L / n
    clearance = max(float(design.get("clearance", 0.13)), 2.0 * dx)
    tau_min = 0.505
    re_max = u_lattice * n * 3.0 / (tau_min - 0.5)
    re = float(min(Re, re_max))
    nu = u_lattice * n / max(re, 1e-9)
    # tunnel proportions: 0.7 car lengths upstream, 1.5 downstream, three car widths across, 2.6 car heights tall —
    # tight enough that a design is a couple of minutes of lattice time, wide enough that the blockage is a constant
    # bias of the whole campaign rather than a per-design surprise (recorded with every dataset)
    x0, x1 = lo[0] - 0.7 * L, hi[0] + 1.5 * L
    y_half = 1.5 * float(hi[1] - lo[1])
    z_top = clearance + 2.6 * float(hi[2] - lo[2])
    shape = (int(round((x1 - x0) / dx)), int(round(2 * y_half / dx)), int(round(z_top / dx)))
    if shapes:
        shape = tuple(int(v) for v in shapes.get("shape", shape))
    origin = (x0, -y_half, 0.0)
    mask = _voxelize(bodies, shape, dx, origin)

    tunnel = WindTunnelLBM(shape, nu=nu, u_in=(u_lattice, 0.0, 0.0), smagorinsky=smagorinsky, device=device,
                           faces={"x-": "inlet", "x+": "outlet", "y-": "wall", "y+": "wall", "z-": "moving",
                                  "z+": "wall"})
    tunnel.add_solid(mask)
    total = int(steps) * max(1, int(averaging))
    averaged = tunnel.run_averaged(total, average_from=int(total * 2 / 3), every=max(1, total // 60))
    wall = tunnel.wall_pressure()
    cells = wall["cells"]
    if cells.shape[0] == 0:
        raise RuntimeError("no wall cell was found: the body is not resolved at this resolution")
    # outward normal of every wall cell, from the mask gradient
    grad = np.zeros((cells.shape[0], 3))
    for k in range(3):
        for s in (-1, 1):
            nb = cells.copy()
            nb[:, k] += s
            inside = np.all((nb >= 0) & (nb < np.asarray(shape)), axis=1)
            free = np.zeros(cells.shape[0], dtype=bool)
            free[inside] = ~mask[nb[inside, 0].astype(int), nb[inside, 1].astype(int), nb[inside, 2].astype(int)]
            grad[free, k] += s
    norm = np.linalg.norm(grad, axis=1)
    normal = np.divide(grad, np.where(norm > 0, norm, 1.0)[:, None])
    points = cells * dx + np.asarray(origin) + 0.5 * dx
    pressure = tunnel.surface_pressure(cells, field=averaged["field"])
    face_area = (dx ** 2) * np.maximum(norm, 1.0)

    # coefficients from the momentum-exchange force on the body and its frontal area
    force = tunnel.force_on_solid()
    frontal = float(np.max(mask.sum(axis=(1, 2))) * dx ** 2)
    q = 0.5 * u_lattice ** 2
    cd = float(force[0] / max(q * frontal, 1e-30))
    cl = float(force[2] / max(q * frontal, 1e-30))
    if points.shape[0] > max_faces:                          # keep a bounded, reproducible subset of wall faces
        rng = np.random.default_rng(seed)
        keep = rng.choice(points.shape[0], size=int(max_faces), replace=False)
        points, pressure, normal, face_area = points[keep], pressure[keep], normal[keep], face_area[keep]
    return {"points": np.asarray(points, np.float32), "pressure": np.asarray(pressure, np.float32),
            "normal": np.asarray(normal, np.float32), "cd": cd, "cl": cl, "frontal_area": frontal,
            "surface_area": float(np.sum(face_area)), "re": re, "re_max": float(re_max),
            "shape": tuple(int(s) for s in shape), "dx": float(dx), "u_lattice": float(u_lattice),
            "steps": int(total), "cells_per_length": n, "design": dict(design)}


def _worker(args):
    design, kwargs = args
    import torch
    torch.set_num_threads(1)
    try:
        return simulate_pressure(design, **kwargs)
    except Exception as exc:                                # a design that cannot be meshed or run is reported
        return {"error": f"{type(exc).__name__}: {exc}", "design": dict(design)}


def build_pressure_dataset(n_designs: int, out_dir: str, *, seed: int = 0, parallel: int = 1,
                           verbose: bool = True, designs: list[dict] | None = None, **sim_kwargs) -> dict:
    """Sample designs, run them (or ``parallel`` of them at a time) and write the dataset plus a coefficients table.

    Writes ``<out_dir>/surface_pressure.npz`` (a :class:`pinneapple_arena.fieldbench.protocol.FieldDataset`) and
    ``<out_dir>/coefficients.csv`` (design vector and Cd/Cl/areas per design, the tabular view of the same campaign),
    and returns a summary dict. A design whose run failed is listed in ``<out_dir>/failures.json``, never silently
    dropped."""
    from pinneapple_arena.fieldbench.protocol import FieldDataset, FieldSample, normalize_geometry

    os.makedirs(out_dir, exist_ok=True)
    designs = sample_designs(n_designs, seed=seed) if designs is None else designs
    jobs = [(d, sim_kwargs) for d in designs]
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
        d = res["design"]
        sample_id = f"car{i:04d}"
        samples.append(FieldSample(id=sample_id, points=res["points"],
                                   values=np.asarray(res["pressure"], np.float32).reshape(-1, 1),
                                   category=design_category(d),
                                   meta={"design": d, "cd": res["cd"], "cl": res["cl"], "re": res["re"],
                                         "normal": None}))
        rows.append({"id": sample_id, "category": sample_id and design_category(d), **d,
                     "cd": res["cd"], "cl": res["cl"], "frontal_area": res["frontal_area"],
                     "surface_area": res["surface_area"], "n_faces": int(res["points"].shape[0]),
                     "re": res["re"], "re_max": res["re_max"], "cells_per_length": res["cells_per_length"],
                     "steps": res["steps"]})
    if not samples:
        raise RuntimeError("no design completed: see the failures listed above")
    dataset = normalize_geometry(FieldDataset(name=f"car_surface_pressure[{len(samples)}]", samples=samples))
    npz = os.path.join(out_dir, "surface_pressure.npz")
    dataset.save(npz)
    csv = os.path.join(out_dir, "coefficients.csv")
    keys = list(rows[0])
    with open(csv, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    with open(os.path.join(out_dir, "failures.json"), "w") as fh:
        json.dump(failures, fh, indent=2)
    summary = {"dataset": npz, "coefficients": csv, "n_samples": len(samples), "n_failed": len(failures),
               "categories": dataset.categories, "re_mean": float(np.mean([r["re"] for r in rows])),
               "splits": {k: len(v) for k, v in dataset.splits.items()}}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
