"""MeshBridge: upload a CFD / FEM / CAD mesh file, get a portable NPZ bundle (node coordinates + every field) or a
repaired, optionally simplified triangle surface (STL / OBJ / PLY) back.

All the reading, repair and writing is ``pinneapple_simulation.external_solvers.cfd_formats.convert``; this file
is the HTTP layer. Which outputs a file allows depends on its format (``GET /v1/formats``): CGNS, Exodus II and
Fluent meshes give points and fields only (their readers return no element connectivity); Abaqus ``.inp``
(triangle shells, tetrahedra), STL, OBJ, PLY and STEP (with cadquery) give surfaces.

Run:  uvicorn meshbridge.api:app --port 8141   (from apps/meshbridge)

Environment (all optional; the rest in apps/_shared/README.md):
  MBR_USER / MBR_PASSWORD(_HASH)   login on everything except /health
  MBR_FREE_CONVERSIONS             free conversions per caller and month (default 200)
  MBR_UPLOAD_MAX_MB                largest upload (default 200)
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask

from pinneapple_simulation.external_solvers.cfd_formats import convert as cv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
MAX_MB = float(os.environ.get("MBR_UPLOAD_MAX_MB") or 200)

app = FastAPI(title="MeshBridge", version=VERSION,
              description="Legacy CFD/FEM/CAD mesh formats to a portable point bundle or a repaired surface.")
install(app, prefix="MBR", max_body_mb=MAX_MB + 10)
rt = install_runtime(app, prefix="MBR", name="meshbridge", units={"conversions": 200}, upload_max_mb=MAX_MB)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/v1/formats")
def formats():
    return {"inputs": cv.SUFFIX_TO_FORMAT, "npz_bundle_from": list(cv.POINT_FORMATS),
            "surface_from": list(cv.SURFACE_FORMATS), "surface_outputs": list(cv.SURFACE_OUTPUTS),
            "notes": ["CGNS / Exodus / Fluent readers follow the published specs and were not checked against files "
                      "from the real writers", "STEP input needs cadquery; STEP output is not offered"]}


@app.post("/v1/convert")
async def convert(request: Request, file: UploadFile = File(...),  # noqa: B008
                  output: Literal["npz", "stl", "obj", "ply"] = Form("npz"), repair: bool = Form(False),
                  simplify_target_faces: int | None = Form(None)):
    try:
        fmt = cv.detect_format(file.filename or "")
    except ValueError as e:
        raise HTTPException(415, str(e)) from None
    data = await file.read()
    if len(data) > MAX_MB * 2**20:
        raise HTTPException(413, f"upload larger than {MAX_MB:g} MB")
    rt.consume(request, "conversions")
    work = Path(tempfile.mkdtemp(prefix="meshbridge_"))
    src = work / f"input{Path(file.filename).suffix.lower()}"
    src.write_bytes(data)
    done = False
    try:
        if output == "npz":
            bundle = cv.read_point_bundle(str(src), fmt)
            buf = io.BytesIO()
            np.savez(buf, coords=bundle["coords"], **{f"field_{k}": v for k, v in bundle["fields"].items()})
            return Response(buf.getvalue(), media_type="application/octet-stream",
                            headers={"Content-Disposition": 'attachment; filename="bundle.npz"',
                                     "X-Fields": ",".join(bundle["fields"]), "X-Format": fmt})
        out = work / f"mesh.{output}"
        report = cv.convert_surface(str(src), str(out), fmt=fmt, repair=repair,
                                    simplify_target_faces=simplify_target_faces)
    except ImportError as e:
        raise HTTPException(422, f"this conversion needs an optional package on the server: {e}") from None
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e)) from None
    else:
        done = True
    finally:
        if not done or output == "npz":
            shutil.rmtree(work, ignore_errors=True)
    return FileResponse(out, filename=out.name, media_type="application/octet-stream",
                        headers={"X-Report": ";".join(f"{k}={v}" for k, v in report.items())},
                        background=BackgroundTask(shutil.rmtree, work, ignore_errors=True))
