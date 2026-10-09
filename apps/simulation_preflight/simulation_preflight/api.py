"""Simulation Preflight: web API + single-page UI. Before spending hours of solver time, find out whether the
simulation is set up correctly: mesh, materials, boundary and initial conditions, solver settings.

Run:  uvicorn simulation_preflight.api:app --port 8087   (from apps/simulation_preflight)

Environment (all optional):
  PFL_USER / PFL_PASSWORD   HTTP Basic login on everything except /health
  PFL_MAX_MB                largest upload in MB (default 200)
  PFL_MAX_HEAVY             concurrent checks per worker (default 2)
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_data.cae.upload import expand
from pinneapple_data.preflight import preflight

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
EXAMPLES = os.path.join(HERE, "..", "examples")
MAX_MB = float(os.environ.get("PFL_MAX_MB") or "200")

# name: (what it is, what the real solver did with it)
EXAMPLE_INFO = {
    "openfoam_pitzDaily_ok.zip": ("OpenFOAM simpleFoam · pitzDaily, k-ε, as shipped",
                                  "simpleFoam converged in 282 iterations"),
    "openfoam_pitzDaily_missing_patch_bc.zip": ("Same case, the outlet entry deleted from 0/k",
                                                "simpleFoam stopped at start-up: 'Cannot find patchField entry for outlet'"),
    "openfoam_pitzDaily_no_pressure_reference.zip": ("Same case, the outlet pressure changed to zeroGradient",
                                                     "simpleFoam stopped: 'Unable to set reference cell for field p'"),
    "openfoam_pitzDaily_epsilon_zero.zip": ("Same case, epsilon initialised to 0",
                                            "simpleFoam crashed in the first iteration (floating-point exception)"),
    "openfoam_pitzDaily_no_relaxation.zip": ("Same case, plain SIMPLE without relaxation factors",
                                             "simpleFoam crashed in the first iteration (floating-point exception)"),
    "openfoam_cavity_large_timestep.zip": ("OpenFOAM icoFoam · lid-driven cavity, time step 4× the tutorial's",
                                           "icoFoam ran to the end at Courant 3.4: no error, but the transient is not "
                                           "time-accurate"),
    "openfoam_cavity_missing_nu.zip": ("Same cavity, nu deleted from transportProperties",
                                       "icoFoam stopped at start-up: \"Entry 'nu' not found\""),
    "calculix_cantilever_ok.inp": ("CalculiX · steel cantilever, tip load, NLGEOM", "ccx converged; tip deflection "
                                   "7.57 mm (Euler-Bernoulli 7.62 mm)"),
    "calculix_heat_missing_conductivity.inp": ("CalculiX · steady heat transfer in the beam, no *CONDUCTIVITY",
                                               "ccx warned 'no conductivity' and aborted while factoring the system; "
                                               "no results"),
    "calculix_static_no_supports.inp": ("CalculiX · the beam loaded but not supported",
                                        "ccx reported 'Job finished' with no error; tip displacement 1.8e11 mm"),
    "calculix_frequency_inconsistent_units.inp": ("CalculiX · modal analysis, E in MPa but density in kg/m³",
                                                  "ccx: first frequency 0.16 Hz instead of 209 Hz, no warning"),
    "calculix_undefined_node_set.inp": ("CalculiX · *BOUNDARY on a node set that does not exist",
                                        "ccx stopped: '*ERROR reading *BOUNDARY: node set CLAMP'"),
    "calculix_inverted_element.inp": ("CalculiX · one element with its nodes in the wrong order",
                                      "ccx stopped: 'nonpositive jacobian determinant in element 1'"),
    "calculix_elements_without_section.inp": ("CalculiX · the last 40 elements left out of the section",
                                              "ccx stopped: 'no material was assigned to element 638'"),
}

app = FastAPI(title="Simulation Preflight", version=VERSION,
              description="Upload an OpenFOAM case (zip or folder) or a CalculiX/Abaqus .inp deck to /api/preflight "
                          "and find out, before running it, whether the mesh, materials, boundary and initial "
                          "conditions and solver settings are consistent: PASS / WARNING / FAIL with the location, "
                          "an explanation and a fix for every problem.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="PFL")
_HEAVY = BusyLimiter("PFL_MAX_HEAVY")

VALIDATION = [
    "Every FAIL rule was confirmed by running the solver on a broken copy of a real case (OpenFOAM v1912 "
    "simpleFoam/icoFoam, CalculiX 2.21): the error or the wrong result is quoted next to the finding",
    "The unmodified cases (pitzDaily, cavity, cantilever) pass; each broken copy is flagged for exactly the change "
    "that was made, at the file and line OpenFOAM or ccx later complains about",
    "Two failures the solver does not report are caught: an unsupported model (ccx finishes with a 1.8e11 mm "
    "displacement) and inconsistent units (first frequency 0.16 Hz instead of 209 Hz)",
    "Mesh checks reuse the Mesh Quality engine, which reproduces checkMesh to the reported digits",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Scope & validation", "noun": "limitation",
            "items_title": "What the pre-flight covers",
            "band": "A PASS means no rule found a problem, not that the results will be right.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can miss a problem", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Solvers covered", "effect": "optimistic",
                 "detail": "OpenFOAM incompressible (simpleFoam, pimpleFoam, pisoFoam, icoFoam), scalar, potential and "
                           "the buoyant/rho families; CalculiX and flat Abaqus decks (static, frequency, heat transfer, "
                           "dynamic, buckling). Other OpenFOAM solvers get the generic checks.",
                 "today": "For other solvers, read the generic findings (mesh, patch coverage, schemes, linear solvers).",
                 "planned": "interFoam, chtMultiRegionFoam, rhoCentralFoam; Abaqus part/assembly decks; Code_Aster."},
                {"topic": "Courant number", "effect": "conservative",
                 "detail": "Estimated before the run from the largest boundary velocity applied everywhere: an upper "
                           "bound (OpenFOAM measured 0.85 where the estimate gave 1.0).",
                 "today": "A warning near the limit may be fine; well above it, reduce the time step.",
                 "planned": "Estimate from a potentialFoam solution."},
                {"topic": "Physics plausibility", "effect": "check",
                 "detail": "Unit consistency (E vs density), viscosity range and Poisson's ratio are checked; whether "
                           "the model represents your physics (turbulence model choice, load magnitudes) is not.",
                 "today": "Use the findings as a checklist, not as a validation of the model.",
                 "planned": "Reynolds/Péclet-based model suggestions; y+ estimate from the mesh."},
                {"topic": "Nothing is executed", "effect": "conservative",
                 "detail": "Files are parsed, never run: #codeStream, coded conditions and user subroutines are not "
                           "evaluated.", "today": "Coded entries are treated as present and valid.", "planned": ""},
            ]}


def _snippets(rep: dict, fs: Dict[str, bytes]) -> None:
    root = rep.get("info", {}).get("root", ".")
    pre = "" if root in (".", "") else root.rstrip("/") + "/"
    for f in rep["findings"]:
        fn, ln = f.get("file"), f.get("line")
        if not fn or not ln:
            continue
        b = fs.get(pre + fn) or fs.get(fn) or next((v for k, v in fs.items() if k.endswith("/" + fn) or k == fn), None)
        if b is None:
            continue
        lines = b.decode("latin-1").splitlines()
        a, z = max(0, ln - 4), min(len(lines), ln + 4)
        f["snippet"] = {"start": a + 1, "lines": [l[:160] for l in lines[a:z]], "at": ln}


def _run(files: List[Tuple[str, bytes]]) -> dict:
    if not files:
        raise HTTPException(422, "No files uploaded.")
    if sum(len(b) for _, b in files) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Upload larger than {MAX_MB:g} MB. Leave out result time directories.")
    try:
        with _HEAVY:
            fs = expand(files)
            # result time directories are not needed for a pre-flight: drop them early
            fs = {k: v for k, v in fs.items() if not _is_result(k)}
            rep = preflight(fs)
            _snippets(rep, fs)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(422, f"{type(e).__name__}: {e}") from e
    rep["scope"] = scope()
    return rep


def _is_result(path: str) -> bool:
    import re
    parts = path.split("/")
    return any(re.fullmatch(r"\d+(\.\d+)?(e[-+]?\d+)?", p) and p not in ("0",) for p in parts[:-1]) or \
        path.lower().endswith((".frd", ".dat", ".sta", ".cvg", ".12d", ".vtk", ".vtu"))


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "max_mb": MAX_MB,
            "examples": [{"name": k, "description": v[0], "solver_result": v[1],
                          "group": "OpenFOAM" if k.startswith("openfoam") else "CalculiX"}
                         for k, v in EXAMPLE_INFO.items() if os.path.exists(os.path.join(EXAMPLES, k))]}


@app.post("/api/preflight")
async def api_preflight(files: List[UploadFile] = File(...)):
    """An OpenFOAM case (zip, or the files with their relative paths) or a CalculiX/Abaqus .inp deck (with its
    *INCLUDE files). Returns status, per-section summary and the findings."""
    return _run([(f.filename or f"upload_{i}", await f.read()) for i, f in enumerate(files)])


def _example_path(name: str) -> str:
    if name not in EXAMPLE_INFO:
        raise HTTPException(404, "No such example.")
    return os.path.join(EXAMPLES, name)


@app.get("/api/example/{name}/file")
def api_example_file(name: str):
    with open(_example_path(name), "rb") as f:
        return Response(f.read(), media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/example/{name}/log")
def api_example_log(name: str):
    _example_path(name)
    p = os.path.join(EXAMPLES, "solver_logs", name.rsplit(".", 1)[0] + ".log")
    if not os.path.exists(p):
        raise HTTPException(404, "No solver log for this example.")
    with open(p, encoding="latin-1") as f:
        return PlainTextResponse(f.read())


@app.get("/api/example/{name}")
def api_example(name: str):
    with open(_example_path(name), "rb") as f:
        rec = _run([(name, f.read())])
    rec["example"] = {"name": name, "description": EXAMPLE_INFO[name][0], "solver_result": EXAMPLE_INFO[name][1]}
    return rec


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
