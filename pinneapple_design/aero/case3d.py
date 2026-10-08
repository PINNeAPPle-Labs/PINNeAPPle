"""3D OpenFOAM case of a whole aircraft (half model, symmetry plane): STL from the Airframe, snappyHexMesh,
simpleFoam k-omega SST with wall functions, run in parallel. Forces, surface pressure and streamlines are computed
from the written fields (no function objects needed).

    write_case3d(case, airframe, alpha_deg=2.0, speed=55.0)  ->  run_case3d(case, procs=4)  ->  read_result3d(case)
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List

import numpy as np

from pinneapple_simulation.numerical_solvers.external_flow import (FOAM_BASHRC, HDR, CaseFields, latest_time,  # noqa: F401
                                                                    run_case, snappy_dict, write_flow_fields)

from .airframe import Airframe, Part, to_stl
BODIES = ("fuselage", "wing", "htail", "vtail")


def _w(case, rel, cls, body):
    p = os.path.join(case, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(HDR % (cls, os.path.basename(rel)) + body)


def _bodies(af: Airframe) -> Dict[str, List[Part]]:
    parts = af.build("cfd")
    out = {b: [] for b in BODIES}
    for p in parts:
        if not p.aero or p.name.endswith("_left"):
            continue                                            # half model: right side + the fuselage
        for b in BODIES:
            if p.name.startswith(b):
                out[b].append(p)
    # the fuselage was split by material: merge back into one closed surface
    return out


def write_case3d(case: str, af: Airframe, alpha_deg: float = 2.0, speed: float = 55.0, nu: float = 1.5e-5,
                 iterations: int = 800, level_wing=(5, 6), procs: int = 4, level_fus=(4, 5)) -> Dict[str, Any]:
    if os.path.isdir(case):
        shutil.rmtree(case)
    tri = os.path.join(case, "constant", "triSurface")
    os.makedirs(tri)
    bodies = _bodies(af)
    for b, parts in bodies.items():
        data = to_stl(parts, b)
        with open(os.path.join(tri, f"{b}.stl"), "wb") as f:
            f.write(data)
    L, b2 = af.length, af.span / 2
    # domain and background cells scale with the fuselage (light aircraft: 8.2 m -> 1 m cells)
    sL = L / 8.2
    xmin, xmax, ymax, zmin, zmax = -12.0 * sL, 32.0 * sL, max(18.0 * sL, b2 * 2.6), -14.0 * sL, 14.0 * sL
    h = 1.0 * sL
    nx, ny, nz = int((xmax - xmin) / h), int(ymax / h), int((zmax - zmin) / h)
    _w(case, "system/blockMeshDict", "dictionary", f"""convertToMeters 1;
vertices ( ({xmin} 0 {zmin}) ({xmax} 0 {zmin}) ({xmax} {ymax} {zmin}) ({xmin} {ymax} {zmin})
           ({xmin} 0 {zmax}) ({xmax} 0 {zmax}) ({xmax} {ymax} {zmax}) ({xmin} {ymax} {zmax}) );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    farfield {{ type patch; faces ( (0 4 7 3) (1 2 6 5) (3 7 6 2) (0 3 2 1) (4 5 6 7) ); }}
    symmetry {{ type symmetryPlane; faces ( (0 1 5 4) ); }}
);
mergePatchPairs ();
""")
    _w(case, "system/surfaceFeatureExtractDict", "dictionary", "\n".join(
        f"{b}.stl {{ extractionMethod extractFromSurface; extractFromSurfaceCoeffs {{ includedAngle 150; }} "
        f"subsetFeatures {{ nonManifoldEdges no; openEdges yes; }} writeObj no; }}" for b in BODIES) + "\n")
    lw = level_wing
    levels = {"fuselage": level_fus, "wing": lw, "htail": lw, "vtail": (lw[0], lw[0])}
    geom = "\n".join(f"    {b}.stl {{ type triSurfaceMesh; name {b}; }}" for b in BODIES)
    feats = "\n".join(f"        {{ file \"{b}.eMesh\"; level {levels[b][1]}; }}" for b in BODIES)
    surf = "\n".join(f"        {b} {{ level ({levels[b][0]} {levels[b][1]}); patchInfo {{ type wall; inGroups (aircraft); }} }}" for b in BODIES)
    geom += (f"\n    near {{ type searchableBox; min ({-1.5 * sL} 0 {-3.0 * sL}); max ({L + 3.0 * sL} {b2 + 1.5 * sL} {3.5 * sL}); }}"
             f"\n    wake {{ type searchableBox; min ({L - 1 * sL} 0 {-3.0 * sL}); max ({L + 12 * sL} {b2 + 2.5 * sL} {3.0 * sL}); }}")
    regions = "        near { mode inside; levels ((1E15 3)); }\n        wake { mode inside; levels ((1E15 2)); }"
    _w(case, "system/snappyHexMeshDict", "dictionary",
       snappy_dict(geom, feats, surf, regions, (-7.53 * sL, 6.27 * sL, 5.31 * sL)))   # location: off any cell face
    write_flow_fields(case, speed, alpha_deg, 0.0, nu, iterations, procs, "aircraft", True)
    return {"alpha": alpha_deg, "speed": speed, "nu": nu, "span": af.span, "S_half": af.wing_area / 2, "mac": af.mac}


def run_case3d(case: str, procs: int = 4, log=print) -> Dict[str, float]:
    return run_case(case, procs, log)


def read_result3d(case: str, info: Dict[str, Any], af: Airframe, n_lines: int = 34) -> Dict[str, Any]:
    """Forces (pressure + wall-function shear), Cp and Cf on every aircraft face, streamlines around the right half,
    the wake and symmetry-plane slices. ``n_lines`` is kept for compatibility (the seeds follow the wing)."""
    cf = CaseFields(case)
    V, nu = info["speed"], info["nu"]
    a = math.radians(info["alpha"])
    drag_dir, lift_dir = np.array([math.cos(a), 0, math.sin(a)]), np.array([-math.sin(a), 0, math.cos(a)])
    q = 0.5 * V * V
    S = info["S_half"]
    out: Dict[str, Any] = {"time": cf.time, "cells": cf.n_cells, "patches": {}}
    Fp, Fv = np.zeros(3), np.zeros(3)
    faces_xyz, faces_cp, faces_cf = [], [], []
    for b in cf.patches():
        if b["name"] not in BODIES:
            continue
        w = cf.wall(b, nu, q)
        fpress, fvisc = w["f_pressure"], w["f_viscous"]
        Fp += fpress.sum(0)
        Fv += fvisc.sum(0)
        out["patches"][b["name"]] = {"faces": int(b["nFaces"]), "CL": float((fpress + fvisc).sum(0) @ lift_dir / (q * S)),
                                     "CD": float((fpress + fvisc).sum(0) @ drag_dir / (q * S))}
        faces_xyz.append(w["xyz"])
        faces_cp.append(w["cp"])
        faces_cf.append(w["cf"])
    Ftot = Fp + Fv
    out.update(CL=float(Ftot @ lift_dir / (q * S)), CD=float(Ftot @ drag_dir / (q * S)),
               CD_pressure=float(Fp @ drag_dir / (q * S)), CD_friction=float(Fv @ drag_dir / (q * S)))
    out["surface"] = {"xyz": np.concatenate(faces_xyz), "cp": np.concatenate(faces_cp), "cf": np.concatenate(faces_cf)}
    # streamlines in the near field: inverse-distance velocity from the 8 nearest cells, RK2
    b2 = af.span / 2
    vel = cf.velocity((np.array([-2.0, -np.inf, -4.0]), np.array([af.length + 14, b2 + 3, 5.0])))
    # seeds: a row just above each wing section's leading edge (suction side), a few under it, a ring around the
    # nose and a cluster at the tip (vortex roll-up); traced until well behind the tail
    zw = af.wing_z_root()
    seeds = []
    for eta in np.r_[np.linspace(0.12, 0.92, 9), 0.97, 1.0]:
        le = np.asarray(af._wing_point(eta, 0.0), float)
        seeds += [le + [-3.0, 0.0, 0.22], le + [-3.0, 0.0, -0.35]]
    for eta, dz in ((1.0, -0.15), (1.0, 0.45), (1.02, 0.1)):
        le = np.asarray(af._wing_point(min(eta, 1.0), 0.0), float)
        seeds.append(le + [-3.0, 0.25 if eta > 1 else 0.05, dz])
    r = 0.5 * af.width
    for ang in np.radians([20, 60, 100, 140]):
        seeds.append([-1.5, 0.15 + 1.1 * r * np.sin(ang), zw + 1.0 + 1.1 * r * np.cos(ang)])
    seeds = np.array(seeds)
    seeds[:, 1] = np.clip(seeds[:, 1], 0.15, b2 + 3)
    P = cf.trace(seeds, vel, h=0.12, x_end=af.length + 10, y_min=0.02, every=3, overshoot=6.0)
    lines = [P[k] for k in range(len(seeds))]
    out["streamlines"] = lines
    out["streamline_speed"] = [np.linalg.norm(vel(L), axis=1) / V for L in lines]

    # slice planes (what ParaView would show): a cross-flow plane behind the wing (speed, streamwise vorticity)
    # and the symmetry plane (pressure coefficient, speed). Points with no mesh cell around them
    # (inside the aircraft) are NaN.
    tip_te = af._wing_point(1.0, 1.0)
    xs_ = tip_te[0] + 0.6 * af.mac
    ny_, nz_ = 220, 110
    yy = np.linspace(0.0, 1.35 * b2, ny_)
    zz = np.linspace(zw - 0.32 * b2, zw + 0.32 * b2, nz_)
    Yg, Zg = np.meshgrid(yy, zz)
    Uv, pv, dmin = cf.sample(np.c_[np.full(Yg.size, xs_), Yg.ravel(), Zg.ravel()])
    Uv = Uv.reshape(nz_, ny_, 3)
    dy, dz = yy[1] - yy[0], zz[1] - zz[0]
    wx = np.gradient(Uv[..., 2], dy, axis=1) - np.gradient(Uv[..., 1], dz, axis=0)
    spd = np.linalg.norm(Uv, axis=-1) / V
    in_w = (dmin > 1.0).reshape(nz_, ny_)
    spd[in_w] = np.nan
    wx[in_w] = np.nan
    out["slice_wake"] = {"x": float(xs_), "y": [float(yy[0]), float(yy[-1])], "z": [float(zz[0]), float(zz[-1])],
                         "shape": [nz_, ny_], "speed": spd, "vorticity": wx * af.mac / V}
    nx2, nz2 = 300, 120
    xx = np.linspace(-0.15 * af.length, 1.25 * af.length, nx2)
    zz2 = np.linspace(-0.22 * af.length, 0.28 * af.length, nz2)
    Xg, Zg2 = np.meshgrid(xx, zz2)
    Uv2, pv2, d2 = cf.sample(np.c_[Xg.ravel(), np.full(Xg.size, 0.02 * af.width), Zg2.ravel()])
    inside = (d2 > 1.0).reshape(nz2, nx2)
    cp2 = (pv2 / q).reshape(nz2, nx2)
    sp2 = (np.linalg.norm(Uv2, axis=1) / V).reshape(nz2, nx2)
    cp2[inside] = np.nan
    sp2[inside] = np.nan
    out["slice_sym"] = {"y": 0.0, "x": [float(xx[0]), float(xx[-1])], "z": [float(zz2[0]), float(zz2[-1])],
                        "shape": [nz2, nx2], "cp": cp2, "speed": sp2}
    return out


def surface_scalars(af: Airframe, parts, surface: Dict[str, np.ndarray], key: str = "cp") -> Dict[str, np.ndarray]:
    """A surface field from the CFD faces onto the visual mesh vertices (nearest faces, inverse-distance, mirrored
    for the left side). key: "cp" (pressure coefficient) or "cf" (skin friction coefficient)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(surface["xyz"])
    vals = np.asarray(surface[key], np.float32)
    out = {}
    for p in parts:
        if not p.aero:
            continue
        v = p.vertices * [1, 1, 1]
        v[:, 1] = np.abs(v[:, 1])
        d, i = tree.query(v, k=4)
        w = 1 / (d + 1e-4) ** 2
        out[p.name] = ((vals[i] * w).sum(1) / w.sum(1)).astype(np.float32)
    return out
