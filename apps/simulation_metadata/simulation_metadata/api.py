"""Simulation Metadata API + single-page UI.

Run:  uvicorn simulation_metadata.api:app --port 8084   (from apps/simulation_metadata)

Environment (all optional):
  SMD_USER / SMD_PASSWORD   HTTP Basic login on everything except /health
  SMD_MAX_MB                largest upload per request in MB (default 100)
  SMD_MAX_HEAVY             concurrent extractions per worker (default 2)
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_data.simulation_metadata import SCHEMA, extract, read_upload

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
EXAMPLES = os.path.join(HERE, "..", "examples")
MAX_MB = float(os.environ.get("SMD_MAX_MB") or "100")

EXAMPLE_INFO = {
    "openfoam_pitzDaily_converged": "OpenFOAM simpleFoam · backward-facing step, k-ε RANS · converged",
    "openfoam_pitzDaily_stopped_early": "Same case with endTime 60 · stopped before the residual targets",
    "openfoam_pitzDaily_diverged": "Same case with relaxation 0.99 and plain SIMPLE · blows up",
    "openfoam_cavity_transient": "OpenFOAM icoFoam · lid-driven cavity · transient, laminar",
    "calculix_cantilever_nlgeom": "CalculiX · steel cantilever, tip load, geometric nonlinearity",
}

app = FastAPI(title="Simulation Metadata API", version=VERSION,
              description="Upload simulation files (OpenFOAM case zip or files, CalculiX/Abaqus .inp with "
                          ".sta/.cvg/.dat, residual histories) to /api/extract and get one structured record: solver, "
                          f"mesh, time stepping, boundary conditions, parameters and a convergence verdict ({SCHEMA}).")
install(app, prefix="SMD")
_HEAVY = BusyLimiter("SMD_MAX_HEAVY")

VALIDATION = [
    "Five real runs made for this app (OpenFOAM v1912 simpleFoam and icoFoam tutorials, CalculiX 2.21): every "
    "verdict matches what the solver did — converged, stopped early while residuals were still falling, diverged with a "
    "floating-point exception, transient run completed, nonlinear FEA converged in 6 increments",
    "Mesh counts match checkMesh exactly (12,225 cells for pitzDaily, 400 for the cavity, 640 C3D8I elements / "
    "1,025 nodes for the beam)",
    "The beam's tip deflection read from the .dat file (7.58 mm) is within 1 % of Euler–Bernoulli "
    "(F L³ / 3 E I = 7.62 mm)",
    "Settings are read, never executed: #include / #codeStream / $macros are reported, not run",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Extraction scope & validation", "noun": "limitation",
            "items_title": "What the extractor assumes",
            "band": "Every value in the record names the file it came from.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can miss information", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Supported solvers", "effect": "optimistic",
                 "detail": "Full records for OpenFOAM and CalculiX/Abaqus decks; Fluent, STAR-CCM+ and SU2 are read from "
                           "their residual-history exports only (no solver settings or mesh).",
                 "today": "For other solvers, upload the residual history (CSV) to get the convergence verdict.",
                 "planned": "Fluent .cas/.trn journals, STAR-CCM+ reports, SU2 .cfg, Code_Aster .comm, Abaqus .msg/.sta."},
                {"topic": "Convergence verdict", "effect": "check",
                 "detail": "Based on residuals, the solver's own messages, Courant numbers and continuity errors. Flat "
                           "residuals do not prove that engineering quantities (drag, pressure drop) have settled.",
                 "today": "Include postProcessing/ monitors: their final variation is reported next to the residuals.",
                 "planned": "Monitor-based convergence and grid-convergence (GCI) across several uploaded runs."},
                {"topic": "Derived numbers", "effect": "check",
                 "detail": "The Reynolds number uses the inlet velocity and the smallest in-plane domain size, which is "
                           "not always the reference length engineers use.",
                 "today": "Treat it as an order of magnitude; enter your reference length in your own report.",
                 "planned": "User-defined reference length, area and velocity per case."},
                {"topic": "Units", "effect": "check",
                 "detail": "OpenFOAM is SI; CalculiX/Abaqus are unit-agnostic, so values are shown in the deck's own "
                           "consistent units (often N, mm, MPa).",
                 "today": "Check the deck's unit system before comparing loads or stiffness across models.",
                 "planned": "Unit-system detection from material constants (E ≈ 210000 → N-mm-MPa)."},
            ]}


def _run(files: List[Tuple[str, bytes]]) -> dict:
    if not files:
        raise HTTPException(422, "No files uploaded.")
    if sum(len(b) for _, b in files) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Upload larger than {MAX_MB:g} MB. Leave out result time directories and meshes in binary formats.")
    try:
        with _HEAVY:
            rec = extract(read_upload(files))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e)) from e
    rec["scope"] = scope()
    return rec


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "schema": SCHEMA, "max_mb": MAX_MB,
            "examples": [{"name": k, "description": v} for k, v in EXAMPLE_INFO.items()
                         if os.path.exists(os.path.join(EXAMPLES, f"{k}.zip"))]}


@app.post("/api/extract")
async def api_extract(files: List[UploadFile] = File(...)):
    """One or more files: a zipped case folder, or loose files (controlDict, fvSolution, log.simpleFoam, model.inp,
    model.sta, history.csv, ...). Returns the metadata record (JSON)."""
    return _run([(f.filename or f"upload_{i}", await f.read()) for i, f in enumerate(files)])


def _example_path(name: str) -> str:
    if name not in EXAMPLE_INFO:
        raise HTTPException(404, "No such example.")
    return os.path.join(EXAMPLES, f"{name}.zip")


@app.get("/api/example/{name}.zip")
def api_example_zip(name: str):
    p = _example_path(name)
    with open(p, "rb") as f:
        return Response(f.read(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})


@app.get("/api/example/{name}")
def api_example(name: str):
    p = _example_path(name)
    with open(p, "rb") as f:
        rec = _run([(f"{name}.zip", f.read())])
    rec["example"] = {"name": name, "description": EXAMPLE_INFO[name]}
    return rec


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
