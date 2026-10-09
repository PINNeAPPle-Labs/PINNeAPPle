"""Mesh Quality: web API + single-page UI. Upload a mesh, get its health: checkMesh-equivalent finite-volume metrics,
element shape metrics, connectivity, surface checks, problem regions located in space and what to do about them.

Run:  uvicorn mesh_quality.api:app --port 8088   (from apps/mesh_quality)

Environment (all optional):
  MQA_USER / MQA_PASSWORD   HTTP Basic login on everything except /health
  MQA_MAX_MB                largest upload in MB (default 200)
  MQA_MAX_CELLS             largest mesh in cells (default 3,000,000)
  MQA_MAX_HEAVY             concurrent analyses per worker (default 2)
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_data.cae import METRICS, FormatError, mesh_report, read_any, view_payload
from pinneapple_data.cae.upload import expand as read_upload

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
EXAMPLES = os.path.join(HERE, "..", "examples")
MAX_MB = float(os.environ.get("MQA_MAX_MB") or "200")
MAX_CELLS = int(os.environ.get("MQA_MAX_CELLS") or "3000000")

EXAMPLE_INFO = {
    "openfoam_pitzDaily.zip": ("OpenFOAM polyMesh · backward-facing step (pitzDaily), 12,225 hexahedra · the "
                               "metrics match checkMesh to the last digit", "cfd"),
    "openfoam_sheared_channel.zip": ("OpenFOAM blockMesh · a channel with one strongly sheared block · checkMesh: "
                                     "'Failed 1 mesh checks' (22 highly skew faces)", "cfd"),
    "gmsh_bracket_tet.msh": ("Gmsh · bracket with a hole, 8,828 tetrahedra · good mesh with a couple of slender "
                             "elements", "fea"),
    "assembly_with_errors.vtu": ("VTK · the bracket with an inverted element, a bolt touching it at one node and a "
                                 "stray element", "fea"),
    "calculix_cantilever_hex.inp": ("CalculiX deck · steel cantilever, 640 C3D8I hexahedra", "fea"),
    "bracket_surface_damaged.stl": ("STL surface · the bracket with a hole, flipped triangles and a loose part, "
                                    "as it would reach snappyHexMesh", "surface"),
}

app = FastAPI(title="Mesh Quality", version=VERSION,
              description="Upload a mesh (OpenFOAM polyMesh or case zip, Gmsh .msh, VTK .vtk/.vtu, STL, Abaqus/"
                          "CalculiX .inp, CalculiX .frd, and other meshio formats) to /api/check and get its health: "
                          "non-orthogonality, skewness, aspect ratio and volumes as OpenFOAM's checkMesh computes them, "
                          "scaled Jacobian and element skewness for FEA, connectivity, surface checks, the problem "
                          "regions and what to do about each issue.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="MQA")
_HEAVY = BusyLimiter("MQA_MAX_HEAVY")

VALIDATION = [
    "Finite-volume metrics reproduce OpenFOAM v1912 checkMesh on real meshes: pitzDaily (max aspect ratio 8.1407, "
    "non-orthogonality 5.95045° / average 1.63034°, skewness 0.260575), a Gmsh tetrahedral mesh read directly from "
    ".msh (69.027° / 21.8577°, skewness 0.816741, aspect ratio 7.91293) and a sheared blockMesh (skewness 4.32031 and "
    "the same count of highly skew faces)",
    "Scaled Jacobian is 1 for the ideal tetrahedron, hexahedron, wedge and pyramid in either node-order convention, "
    "negative for an inverted element",
    "STL: the closed bracket surface encloses exactly the volume of its tetrahedral mesh (1.11077e-4 m³); removed, "
    "flipped and stray triangles are all detected",
    "CalculiX .inp and .frd: 640 elements / 1,025 nodes, the tip displacement read from the .frd equals the .dat",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Scope & validation", "noun": "limitation",
            "items_title": "What the checks assume",
            "band": "Thresholds follow checkMesh (finite volume) and common FEA guidelines; each is stated in the report.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can miss a problem", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Thresholds are generic", "effect": "check",
                 "detail": "70° non-orthogonality, skewness 4, scaled Jacobian 0.2 are widely used limits; your solver "
                           "settings (non-orthogonal correctors, element formulation) move the real limit.",
                 "today": "Read warnings as 'look here first', not as a verdict on the results.",
                 "planned": "Per-solver threshold profiles (OpenFOAM, Fluent, Abaqus, CalculiX)."},
                {"topic": "Linear geometry", "effect": "optimistic",
                 "detail": "Quadratic elements are assessed on their corner nodes; curved mid-side nodes are not checked.",
                 "today": "For quadratic meshes, also check the mesher's own high-order validity (Gmsh 'ICN/SICN').",
                 "planned": "High-order Jacobian at the integration points."},
                {"topic": "Self-intersections", "effect": "optimistic",
                 "detail": "Surface checks find holes, non-manifold edges, flipped normals, zero-area and duplicate "
                           "faces; intersecting triangles are not searched.",
                 "today": "Run surfaceCheck (OpenFOAM) or MeshLab's self-intersection filter on suspect geometry.",
                 "planned": "Bounding-volume self-intersection search."},
                {"topic": "Formats", "effect": "check",
                 "detail": "Decomposed OpenFOAM cases (processor*) are not reassembled; Fluent .msh/.cas and CGNS "
                           "need meshio's support for the file's flavour.",
                 "today": "Reconstruct the case (reconstructParMesh) or export to VTK/Gmsh.",
                 "planned": "Fluent mesh files and parallel OpenFOAM cases."},
            ]}


def _clean(rep: dict) -> dict:
    rep = {k: v for k, v in rep.items() if not k.startswith("_")}
    return rep


def _worst_cells(mesh, rep, n: int = 40) -> List[dict]:
    vals, cents = rep["_cell_values"], rep.get("_centres")
    out = []
    for k, v in vals.items():
        spec = METRICS[k]
        if not np.isfinite(v).any():
            continue
        order = np.argsort(-np.nan_to_num(v, nan=-np.inf)) if spec["better"] == "low" else np.argsort(np.nan_to_num(v, nan=np.inf))
        for c in order[:n]:
            out.append({"metric": k, "cell": int(c), "label": mesh.cell_label(int(c)), "value": float(v[c]),
                        "centre": cents[c].tolist() if cents is not None else None})
    return out


def _run(files: List[Tuple[str, bytes]], target: str) -> dict:
    if not files:
        raise HTTPException(422, "No files uploaded.")
    if sum(len(b) for _, b in files) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Upload larger than {MAX_MB:g} MB.")
    if target not in ("auto", "cfd", "fea", "surface"):
        raise HTTPException(422, "target is auto, cfd, fea or surface")
    try:
        with _HEAVY:
            mesh = read_any(read_upload(files), fields=False)
            if mesh.n_cells > MAX_CELLS:
                raise HTTPException(413, f"{mesh.n_cells:,} cells: this server checks up to {MAX_CELLS:,}.")
            if mesh.n_cells == 0:
                raise HTTPException(422, "The file has points but no cells.")
            rep = mesh_report(mesh, target)
            view = view_payload(mesh, rep)
            worst = _worst_cells(mesh, rep)
    except HTTPException:
        raise
    except FormatError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(422, f"{type(e).__name__}: {e}") from e
    out = _clean(rep)
    out.update(view=view, worst_cells=worst, scope=scope(),
               metric_info={k: {kk: vv for kk, vv in v.items()} for k, v in METRICS.items()},
               files=[{"name": n, "bytes": len(b)} for n, b in files])
    return out


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "max_mb": MAX_MB, "max_cells": MAX_CELLS,
            "examples": [{"name": k, "description": v[0], "target": v[1]} for k, v in EXAMPLE_INFO.items()
                         if os.path.exists(os.path.join(EXAMPLES, k))]}


@app.post("/api/check")
async def api_check(files: List[UploadFile] = File(...), target: str = Form("auto")):
    """One mesh: a file (.msh, .vtk, .vtu, .stl, .inp, .frd, ...) or an OpenFOAM case / polyMesh folder as a zip or
    loose files. ``target``: auto, cfd (finite-volume thresholds), fea (element shape) or surface."""
    return _run([(f.filename or f"upload_{i}", await f.read()) for i, f in enumerate(files)], target)


def _example_path(name: str) -> str:
    if name not in EXAMPLE_INFO:
        raise HTTPException(404, "No such example.")
    return os.path.join(EXAMPLES, name)


@app.get("/api/example/{name}/file")
def api_example_file(name: str):
    with open(_example_path(name), "rb") as f:
        return Response(f.read(), media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/example/{name}")
def api_example(name: str, target: str = "auto"):
    with open(_example_path(name), "rb") as f:
        rec = _run([(name, f.read())], target)
    rec["example"] = {"name": name, "description": EXAMPLE_INFO[name][0]}
    return rec


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
