"""Simulation Interoperability Hub: web API + single-page UI. Upload a simulation result, get it in a neutral
engineering representation (geometry, mesh, coordinates, fields with quantity and unit, metadata) and export it to
VTK, HDF5, Parquet, CSV, NPZ, JSON or a PINNeAPPle dataset.

Run:  uvicorn interop_hub.api:app --port 8091   (from apps/interop_hub)

Environment (all optional):
  IOP_USER / IOP_PASSWORD   HTTP Basic login on everything except /health
  IOP_MAX_MB                largest upload in MB (default 500)
  IOP_MAX_HEAVY             concurrent conversions per worker (default 2)
  IOP_CACHE_MIN             minutes a converted upload stays available for export (default 30)
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
import tempfile
import time
import uuid
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_data.cae import FormatError, read_any
from pinneapple_data.cae.compare import surface
from pinneapple_data.cae.dataset import SCHEMA, UNIT_SYSTEMS, build_dataset, describe, export
from pinneapple_data.cae.upload import expand

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
HERE = os.path.dirname(__file__)
STATIC = os.path.join(HERE, "static")
EXAMPLES = os.path.join(HERE, "..", "examples")
MAX_MB = float(os.environ.get("IOP_MAX_MB", "500"))
CACHE_MIN = float(os.environ.get("IOP_CACHE_MIN", "30"))
CACHE = os.path.join(tempfile.gettempdir(), "pinneapple_interop")
os.makedirs(CACHE, exist_ok=True)
FORMATS = {"vtu": "VTK unstructured grid (.vtu): ParaView, VisIt, PyVista",
           "hdf5": "HDF5 (.h5): mesh, fields with units as attributes, metadata",
           "parquet": "Parquet (.parquet): one row per cell or point, units in the schema metadata",
           "csv": "CSV (.csv): one row per cell or point, units in the header",
           "npz": "NumPy (.npz): coordinates, fields and metadata arrays",
           "json": "JSON (.json): the dataset description, with the data when small",
           "pinneapple": "PINNeAPPle dataset (.zarr.zip): a PhysicalSample in the library's UPD Zarr store"}
EXAMPLE_INFO = {
    "openfoam_pitzDaily_result.zip": ("OpenFOAM simpleFoam result · pitzDaily, k-ε, converged (t = 282), polyhedral mesh", {}),
    "openfoam_cavity_Re100_result.zip": ("OpenFOAM icoFoam result · lid-driven cavity Re = 100, 40×40", {"rho": 1.0}),
    "calculix_cantilever.frd": ("CalculiX .frd · cantilever, displacements and stresses (deck in N-mm-t-s)",
                                {"unit_system": "N-mm-t-s"}),
    "gmsh_bracket_mesh.msh": ("Gmsh mesh · bracket, 8,828 tetrahedra, physical groups", {}),
}

app = FastAPI(title="Simulation Interoperability Hub", version=VERSION,
              description="Upload a simulation result (OpenFOAM case, CalculiX .frd/.inp, VTK/VTU, Gmsh, STL, CSV/HDF5/"
                          f"NPZ tables) to /api/inspect: it is normalised into one neutral dataset ({SCHEMA}) with "
                          "geometry, mesh, coordinates, fields (quantity, unit) and metadata, and can be exported to "
                          "VTU, HDF5, Parquet, CSV, NPZ, JSON or a PINNeAPPle dataset.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="IOP")
_HEAVY = BusyLimiter("IOP_MAX_HEAVY")

VALIDATION = [
    "OpenFOAM polyMesh → VTU: hexahedra, wedges, tetrahedra and pyramids rebuilt from the faces (other cells as VTK "
    "polyhedra); reread with VTK (PyVista): same 12,225 cells, total volume equal to the last digit "
    "(1.451604e-5 m³), smallest cell 1.6902e-10 m³ as checkMesh reports, same field ranges",
    "PINNeAPPle dataset: written with the library's own Zarr store and read back with "
    "pinneapple_data.serialization.load_zarr (coordinates, fields, units and source in the provenance)",
    "Every export is reread in the tests (VTU with VTK and meshio, HDF5, Parquet, CSV, NPZ, JSON) and checked "
    "against the source values",
    "CalculiX results get units from the deck's unit system (N-mm-t-s: mm, MPa) and convert to SI on request",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Scope & validation", "noun": "limitation",
            "items_title": "What the conversion covers",
            "band": "Values are carried over exactly; only declared unit conversions change them.",
            "tags": {"conservative": "Safe by default", "optimistic": "Can lose information", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Formats in", "effect": "check",
                 "detail": "OpenFOAM (ASCII and binary, one time per export), CalculiX .frd/.inp, VTK/VTU, Gmsh and "
                           "the other meshio formats, STL, CSV/Parquet/HDF5/NPZ tables. Abaqus .odb, ANSYS .rst and "
                           "COMSOL files are binary and proprietary.",
                 "today": "Export those to VTK/VTU (or CSV) from the vendor tool first.",
                 "planned": "VMAP (the NAFEMS-backed neutral format), CGNS and Exodus II readers/writers."},
                {"topic": "Units", "effect": "check",
                 "detail": "OpenFOAM is SI (incompressible p is kinematic, m²/s²: give ρ to get Pa); CalculiX/Abaqus "
                           "have no units: choose the deck's unit system to label and convert the fields.",
                 "today": "Check the unit column before using the data.", "planned": "Unit-system detection from E and ρ."},
                {"topic": "Boundary data", "effect": "optimistic",
                 "detail": "Exports carry the cell/point fields; OpenFOAM patch values are not written as separate "
                           "boundary datasets, and only one time step is exported at a time.",
                 "today": "Export each time you need.", "planned": "Patch data and time series in HDF5."},
                {"topic": "Uploads", "effect": "conservative",
                 "detail": f"A converted upload stays in the server's temporary storage for {CACHE_MIN:g} minutes so "
                           "it can be exported in several formats, then it is deleted.", "today": "", "planned": ""},
            ]}


def _gc():
    now = time.time()
    for p in glob.glob(os.path.join(CACHE, "*.pkl")):
        if now - os.path.getmtime(p) > CACHE_MIN * 60:
            try:
                os.remove(p)
            except OSError:
                pass


def _convert(files: List[Tuple[str, bytes]], time_: str, unit_system: str, to_si: bool, rho: Optional[float],
             location: str) -> dict:
    if not files:
        raise HTTPException(422, "No files uploaded.")
    if sum(len(b) for _, b in files) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Upload larger than {MAX_MB:g} MB.")
    if unit_system not in ("auto", "model units") and unit_system not in UNIT_SYSTEMS:
        raise HTTPException(422, "unit_system is auto, SI or N-mm-t-s")
    if location not in ("auto", "cell", "point"):
        raise HTTPException(422, "location is auto, cell or point")
    _gc()
    try:
        with _HEAVY:
            fs = expand(files)
            m = read_any(fs, fields=True, time=time_ or None)
            ds = build_dataset(m, unit_system=unit_system, to_si=to_si, rho=rho, location=location, files=fs)
            if not ds["metadata"].get("case") or ds["metadata"]["case"] in (".", ""):
                ds["metadata"]["case"] = files[0][0]
            sid = uuid.uuid4().hex
            with open(os.path.join(CACHE, f"{sid}.pkl"), "wb") as fh:
                pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)
            d = describe(ds)
            prev = _preview(ds)
    except HTTPException:
        raise
    except (FormatError, ValueError, KeyError) as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(422, f"{type(e).__name__}: {e}") from e
    return {"id": sid, "dataset": d, "preview": prev, "formats": FORMATS, "scope": scope(),
            "usage": {"bytes_in": sum(len(b) for _, b in files)},
            "options": {"time": time_, "unit_system": unit_system, "to_si": to_si, "rho": rho, "location": location}}


def _preview(ds: dict) -> dict:
    """Boundary surface coloured by each cell field (first 6 fields), for the 3D preview."""
    import numpy as np
    from pinneapple_data.cae.compare import _round
    m = ds["_mesh"]
    if not m.n_cells:
        return {}
    try:
        s = surface(m, max_tris=200_000)
    except Exception:
        return {}
    if s is None:
        return {}
    out = {"points": s["points"], "tris": s["tris"], "tri_cell": s["tri_cell"], "fields": {}}
    k = 0
    for name, f in ds["fields"].items():
        v = f["values"]
        mag = np.linalg.norm(v, axis=1) if v.ndim == 2 else v
        if f["location"] == "cell" and len(mag) == m.n_cells:
            out["fields"][name] = {"where": "cell", "values": _round(mag[s["cells"]])}
        elif f["location"] == "point" and len(mag) == m.n_points:
            out["fields"][name] = {"where": "point", "values": _round(mag[s["vertex_index"]])}
        else:
            continue
        k += 1
        if k >= 6:
            break
    return out


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "schema": SCHEMA, "formats": FORMATS, "unit_systems": ["auto", "SI", "N-mm-t-s"],
            "examples": [{"name": k, "description": v[0], "options": v[1]} for k, v in EXAMPLE_INFO.items()
                         if os.path.exists(os.path.join(EXAMPLES, k))]}


@app.post("/api/inspect")
async def api_inspect(files: List[UploadFile] = File(...), time: str = Form(""), unit_system: str = Form("auto"),
                      to_si: bool = Form(False), rho: Optional[float] = Form(None), location: str = Form("auto")):
    """Normalise an upload; returns the dataset description and an id for /api/export/{id}/{format}."""
    return _convert([(f.filename or f"upload_{i}", await f.read()) for i, f in enumerate(files)], time, unit_system,
                    to_si, rho, location)


@app.post("/api/convert/{fmt}")
async def api_convert(fmt: str, files: List[UploadFile] = File(...), time: str = Form(""),
                      unit_system: str = Form("auto"), to_si: bool = Form(False), rho: Optional[float] = Form(None),
                      location: str = Form("auto")):
    """One call: upload and get the file in the chosen format (vtu, hdf5, parquet, csv, npz, json, pinneapple)."""
    r = _convert([(f.filename or f"upload_{i}", await f.read()) for i, f in enumerate(files)], time, unit_system,
                 to_si, rho, location)
    return api_export(r["id"], fmt)


@app.get("/api/export/{sid}/{fmt}")
def api_export(sid: str, fmt: str):
    if fmt not in FORMATS:
        raise HTTPException(404, f"format is one of {', '.join(FORMATS)}")
    p = os.path.join(CACHE, f"{sid}.pkl")
    if not sid.isalnum() or not os.path.exists(p):
        raise HTTPException(404, f"Upload expired (kept {CACHE_MIN:g} minutes): convert it again.")
    with open(p, "rb") as fh:
        ds = pickle.load(fh)
    os.utime(p)
    try:
        with _HEAVY:
            blob, media, ext = export(ds, fmt)
    except Exception as e:
        raise HTTPException(422, f"{type(e).__name__}: {e}") from e
    base = os.path.splitext(os.path.basename(str(ds["metadata"].get("case") or "dataset").rstrip("/")) or "dataset")[0] or "dataset"
    return Response(blob, media_type=media, headers={"Content-Disposition": f'attachment; filename="{base}.{ext}"'})


@app.get("/api/example/{name}")
def api_example(name: str):
    if name not in EXAMPLE_INFO:
        raise HTTPException(404, "No such example.")
    desc, opts = EXAMPLE_INFO[name]
    with open(os.path.join(EXAMPLES, name), "rb") as fh:
        r = _convert([(name, fh.read())], "", opts.get("unit_system", "auto"), False, opts.get("rho"), "auto")
    r["example"] = {"name": name, "description": desc, "options": opts}
    return r


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
