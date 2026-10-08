"""Simulation Comparator: web API + single-page UI. Compare a candidate result with a reference, field by field:
simulation vs simulation, vs experiment, vs AI (PINN, FNO, DeepONet, surrogate), vs analytical or benchmark data.

Run:  uvicorn simulation_comparator.api:app --port 8089   (from apps/simulation_comparator)

Environment (all optional):
  CMP_USER / CMP_PASSWORD   HTTP Basic login on everything except /health
  CMP_MAX_MB                largest upload (both sides together) in MB (default 300)
  CMP_MAX_POINTS            largest reference / candidate in cells or points (default 2,000,000)
  CMP_MAX_HEAVY             concurrent comparisons per worker (default 2)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_data.cae import FormatError, read_any
from pinneapple_data.cae.compare import compare, fields_of, views
from pinneapple_data.cae.upload import expand

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
EXAMPLES = os.path.join(HERE, "..", "examples")
MAX_MB = float(os.environ.get("CMP_MAX_MB", "300"))
MAX_POINTS = int(os.environ.get("CMP_MAX_POINTS", "2000000"))
MODES = {"sim-sim": "Simulation vs simulation", "sim-exp": "Simulation vs experiment",
         "sim-ai": "Simulation vs AI", "ai-exp": "AI vs experiment", "sim-ref": "Simulation vs analytical / benchmark"}

EXAMPLES_INFO = {
    "mesh": {"title": "Grid convergence: pitzDaily on 12,225 vs 48,900 cells", "mode": "sim-sim",
             "reference": "pitzDaily_kEpsilon_fine_48900cells.zip", "candidate": "pitzDaily_kEpsilon_12225cells.zip",
             "note": "Same case (simpleFoam, k-ε) on the tutorial mesh and on a mesh refined 2× in each direction; "
                     "the fine run is the reference. Where is the coarse mesh wrong?"},
    "model": {"title": "Turbulence model: k-ε vs k-ω SST on pitzDaily", "mode": "sim-sim",
              "reference": "pitzDaily_kEpsilon_12225cells.zip", "candidate": "pitzDaily_kOmegaSST_12225cells.zip",
              "note": "Same mesh, two RANS models (both converged: 282 and 381 iterations). Same locations, no "
                      "interpolation: the differences are the models."},
    "ghia20": {"title": "Lid-driven cavity Re = 100, 20×20 vs Ghia et al. (1982)", "mode": "sim-ref",
               "reference": "ghia1982_cavity_Re100_u_centreline.csv", "candidate": "cavity_Re100_20x20.zip",
               "note": "icoFoam on 20×20 cells against the classic benchmark table (u along the vertical centreline)."},
    "ghia40": {"title": "Lid-driven cavity Re = 100, 40×40 vs Ghia et al. (1982)", "mode": "sim-ref",
               "reference": "ghia1982_cavity_Re100_u_centreline.csv", "candidate": "cavity_Re100_40x40.zip",
               "note": "The same on 40×40 cells: the error to the benchmark drops by 7×."},
    "pinn": {"title": "AI vs simulation: a PINN temperature map vs finite volumes", "mode": "sim-ai",
             "reference": "plate_finite_volume_reference.csv", "candidate": "plate_pinn_prediction.csv",
             "note": "The 2D heat-spreader plate of PINNeAPPle's inverse example (day 7): the PINN learned h and "
                     "the map from 8 thermocouples; the reference is an independent finite-volume solution."},
    "beam": {"title": "CalculiX cantilever vs Euler-Bernoulli beam theory", "mode": "sim-ref",
             "reference": "cantilever_euler_bernoulli.csv", "candidate": "calculix_cantilever_results.frd",
             "note": "Tip-loaded steel cantilever (C3D8I, NLGEOM) against w(x) = F x² (3L - x) / 6EI along its axis."},
}

app = FastAPI(title="Simulation Comparator", version=VERSION,
              description="Upload a reference and a candidate result (OpenFOAM case, VTK/VTU, CalculiX .frd, Gmsh, CSV/"
                          "HDF5/NPZ point tables) to /api/compare: the candidate is interpolated onto the reference "
                          "locations and every common field gets MAE, RMSE, maximum error and where, relative L2, "
                          "NRMSE and R², with the spatial error map.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="CMP")
_HEAVY = BusyLimiter("CMP_MAX_HEAVY")

VALIDATION = [
    "Lid-driven cavity at Re = 100 (icoFoam) against Ghia, Ghia & Shin (1982): relative L2 error of u on the "
    "centreline 1.26 % on 20×20 cells, 0.19 % on 40×40 (the expected mesh convergence towards the benchmark)",
    "CalculiX cantilever against Euler-Bernoulli: 0.7 % relative L2 on the deflection line (tip 7.57 vs 7.62 mm: "
    "shear deformation and geometric stiffening, as expected)",
    "Interpolation alone, on an analytic field between the two pitzDaily meshes: relative L2 0.04 %, NRMSE 0.09 % "
    "(tested), far below the differences between the results",
    "OpenFOAM boundary values (fixed values, noSlip, written wall values) are part of the interpolation, so points "
    "on walls and inlets compare correctly",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Scope & validation", "noun": "limitation",
            "items_title": "How the comparison works",
            "band": "The candidate is mapped onto the reference; metrics are computed at the reference locations.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can hide a difference", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Interpolation", "effect": "check",
                 "detail": "Linear (Delaunay) inside the candidate's points, nearest outside (counted in the table). "
                           "Very coarse candidates are smoothed by the interpolation itself, and where a boundary "
                           "condition jumps (an inlet meeting a wall) the interpolated value is an average: isolated "
                           "maxima there are interpolation, not physics; the p99 error is the robust figure.",
                 "today": "Compare on the coarser of the two meshes when possible (swap reference and candidate).",
                 "planned": "Conservative (volume-weighted) remapping between meshes."},
                {"topic": "Units", "effect": "optimistic",
                 "detail": "Values are compared as they are; units read from the files are shown and a mismatch is "
                           "flagged, not converted. OpenFOAM's incompressible p is kinematic (m²/s²).",
                 "today": "Make sure both sides use the same units (and p vs p/ρ).",
                 "planned": "Automatic conversion through the Data Standardizer's unit engine."},
                {"topic": "Time", "effect": "check",
                 "detail": "OpenFOAM: the latest time directory of each case unless chosen; transient histories are "
                           "not compared over time.", "today": "Pick the time of each side explicitly.",
                 "planned": "Time-series comparison of probes and monitors."},
                {"topic": "Global score", "effect": "check",
                 "detail": "The global MAE/RMSE is the mean over fields of each error divided by that field's "
                           "reference range: dimensionless, but it weighs all fields equally.",
                 "today": "Read the per-field table for decisions.", "planned": "User-weighted scores."},
            ]}


def _load(files: List[Tuple[str, bytes]], time: Optional[str]) -> object:
    fs = expand(files)
    m = read_any(fs, fields=True, time=time or None)
    n = max(m.n_cells, m.n_points)
    if n > MAX_POINTS:
        raise HTTPException(413, f"{n:,} cells/points: this server compares up to {MAX_POINTS:,}.")
    return m


def _summary(m) -> dict:
    return {"format": m.source.get("format"), "file": m.source.get("file"), "cells": m.n_cells, "points": m.n_points,
            "time": m.source.get("time"), "times": m.source.get("times"), "fields": fields_of(m)}


def _run(ref_files, cand_files, mode, mapping, method, tolerance, ref_time, cand_time) -> dict:
    if not ref_files or not cand_files:
        raise HTTPException(422, "Upload a reference and a candidate.")
    if sum(len(b) for _, b in ref_files + cand_files) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Uploads larger than {MAX_MB:g} MB.")
    try:
        mp = json.loads(mapping) if mapping else None
        if mp is not None and not isinstance(mp, dict):
            raise ValueError
    except ValueError as e:
        raise HTTPException(422, 'mapping must be a JSON object, e.g. {"T": "temperature"}') from e
    try:
        with _HEAVY:
            ref = _load(ref_files, ref_time)
            cand = _load(cand_files, cand_time)
            res = compare(ref, cand, mapping=mp, method=method)
            v = views(ref, res)
    except HTTPException:
        raise
    except (FormatError, ValueError, KeyError) as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(422, f"{type(e).__name__}: {e}") from e
    tol = tolerance / 100 if tolerance and tolerance > 0 else None
    for f in res["fields"]:
        e = f["summary"]["rel_l2"]
        f["verdict"] = None if tol is None or e is None else ("pass" if e <= tol else "fail")
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    out.update(view=v, mode=mode, mode_label=MODES.get(mode, mode), tolerance_pct=tolerance,
               reference=_summary(ref), candidate=_summary(cand), scope=scope(),
               usage={"reference_points": max(ref.n_cells, ref.n_points), "candidate_points": max(cand.n_cells, cand.n_points)})
    verdicts = [f["verdict"] for f in res["fields"] if f["verdict"]]
    out["status"] = None if not verdicts else ("PASS" if all(v == "pass" for v in verdicts) else "FAIL")
    return out


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "max_mb": MAX_MB, "modes": MODES,
            "examples": [{"id": k, **{kk: vv for kk, vv in v.items()}} for k, v in EXAMPLES_INFO.items()
                         if all(os.path.exists(os.path.join(EXAMPLES, v[s])) for s in ("reference", "candidate"))]}


@app.post("/api/compare")
async def api_compare(reference: List[UploadFile] = File(...), candidate: List[UploadFile] = File(...),
                      mode: str = Form("sim-sim"), mapping: str = Form(""), method: str = Form("auto"),
                      tolerance: float = Form(5.0), reference_time: str = Form(""), candidate_time: str = Form("")):
    """reference / candidate: the files of each result (a case zip, a .vtu, a .frd, a CSV with x,y,z columns...).
    mapping: optional JSON {reference field: candidate field}; method: auto | linear | nearest; tolerance: relative
    L2 error in % for the per-field PASS/FAIL (0 = no verdict)."""
    rf = [(f.filename or f"ref_{i}", await f.read()) for i, f in enumerate(reference)]
    cf = [(f.filename or f"cand_{i}", await f.read()) for i, f in enumerate(candidate)]
    if method not in ("auto", "linear", "nearest"):
        raise HTTPException(422, "method is auto, linear or nearest")
    return _run(rf, cf, mode, mapping, method, tolerance, reference_time, candidate_time)


@app.get("/api/example/{eid}")
def api_example(eid: str, tolerance: float = 5.0):
    if eid not in EXAMPLES_INFO:
        raise HTTPException(404, "No such example.")
    ex = EXAMPLES_INFO[eid]
    rd = lambda n: [(n, open(os.path.join(EXAMPLES, n), "rb").read())]  # noqa: E731
    out = _run(rd(ex["reference"]), rd(ex["candidate"]), ex["mode"], "", "auto", tolerance, "", "")
    out["example"] = {"id": eid, **ex}
    return out


@app.get("/api/example-file/{name}")
def api_example_file(name: str):
    names = {v[s] for v in EXAMPLES_INFO.values() for s in ("reference", "candidate")}
    if name not in names:
        raise HTTPException(404, "No such file.")
    with open(os.path.join(EXAMPLES, name), "rb") as f:
        return Response(f.read(), media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
