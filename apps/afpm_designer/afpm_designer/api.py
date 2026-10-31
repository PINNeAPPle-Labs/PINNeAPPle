"""Axial-Flux Motor Designer: a quick quasi-3D estimate of an axial-flux PM machine, then a "verify in 3-D" that
solves the same design in full 3-D on the same grid and shows what the estimate left out, with a 3-D view of |B|.

Glue only: the physics is ``pinneapple_simulation.electromagnetics.machines3d.afpm_design`` (``estimate``,
``verify_3d``, ``field_view_of``) over the edge-element solver of ``machines3d.afpm``.

Run:  uvicorn afpm_designer.api:app --port 8191   (from apps/afpm_designer)

Environment (all optional): AFPM_USER / AFPM_PASSWORD (or AFPM_PASSWORD_HASH) for an HTTP Basic login on everything
except /health; AFPM_MAX_JOBS runs at once (default 1: a 3-D check takes about 2 GB); the rest in
apps/_shared/README.md.
"""
from __future__ import annotations

import os
import secrets
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinneapple_simulation.electromagnetics.machines3d.afpm_design import (
    estimate,
    field_view_of,
    grid_for,
    machine_from,
    verify_3d,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Axial-Flux Motor Designer", version=VERSION,
              description="Quasi-3D estimate of an axial-flux PM machine (cogging, tooth flux linkage, back-EMF "
                          "constant) and a verification in full 3-D on the same edge-element grid, with a 3-D view "
                          "of the field. Linear iron, no load.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="AFPM")
rt = install_runtime(app, prefix="AFPM", name="afpm_designer", units={"estimates": None, "checks_3d": None},
                     max_jobs=1)
_VIEWS: OrderedDict[str, tuple[bytes, dict]] = OrderedDict()
MAX_VIEWS = 12


class Design(BaseModel):
    poles: int = Field(8, ge=2, le=48)
    slots: int = Field(12, ge=3, le=48)
    r_in: float = Field(0.05, gt=0.01, le=0.3)
    r_out: float = Field(0.09, gt=0.02, le=0.4)
    gap: float = Field(0.001, ge=0.0003, le=0.005)
    magnet_thickness: float = Field(0.004, ge=0.001, le=0.02)
    magnet_arc: float = Field(0.8, gt=0.2, le=0.98)
    rotor_yoke: float = Field(0.008, ge=0.002, le=0.03)
    tooth_height: float = Field(0.02, ge=0.005, le=0.06)
    stator_yoke: float = Field(0.008, ge=0.002, le=0.03)
    slot_fraction: float = Field(0.5, gt=0.1, lt=0.9)
    Br: float = Field(1.2, gt=0.1, le=1.6)
    topology: Literal["single", "double_rotor"] = "single"


def _check(d: Design) -> dict:
    p = d.model_dump()
    if p["r_out"] <= p["r_in"] + 0.01:
        raise HTTPException(422, "r_out must exceed r_in by at least 10 mm")
    if p["poles"] % 2:
        raise HTTPException(422, "the number of poles must be even")
    try:
        grid_for(machine_from(p))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return p


def _keep_view(models, title: str) -> str:
    model, sol = models[0]
    fv = field_view_of(model, sol, title=title)
    token = secrets.token_urlsafe(16)
    _VIEWS[token] = (fv.glb(), fv.meta)
    while len(_VIEWS) > MAX_VIEWS:
        _VIEWS.popitem(last=False)
    return token


def _estimate_job(params: dict, ctx) -> dict:
    ctx.progress(0.05, "quasi-3D: one radial slice with flux-parallel faces")
    r = estimate(params, keep=True)
    ctx.progress(0.9, "3-D view of the field")
    r["view"] = _keep_view(r.pop("_models"), "quasi-3D estimate, |B| at no load")
    return r


def _verify_job(params: dict, ctx) -> dict:
    ctx.progress(0.05, "quasi-3D estimate on the grid of the check")
    e = estimate(params)
    ctx.progress(0.35, "full 3-D: air inside, outside, above and below")
    r = verify_3d(params, against=e, keep=True)
    ctx.progress(0.9, "3-D view of the field")
    r["view"] = _keep_view(r.pop("_models"), "3-D check, |B| at no load")
    r["estimate"] = {k: e[k] for k in ("cogging_pkpk_Nm", "tooth_linkage_Wb", "ke_tooth_V_s", "cogging_Nm",
                                       "thetas_deg", "tets", "seconds")}
    return r


rt.jobs.register("estimate", _estimate_job)
rt.jobs.register("verify", _verify_job)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/defaults")
def defaults():
    """The default design and the allowed ranges of the form."""
    return {"design": Design().model_dump(), "schema": Design.model_json_schema()}


@app.post("/api/estimate")
def post_estimate(request: Request, design: Design):
    """Queue a quasi-3D estimate (about 20 s for the default design)."""
    return rt.submit(request, "estimate", _check(design), units={"estimates": 1})


@app.post("/api/verify")
def post_verify(request: Request, design: Design):
    """Queue the check in full 3-D, with the estimate on the same grid (about 60 s for the default design: 22 s for
    the estimate, 35 s for the 3-D solve)."""
    return rt.submit(request, "verify", _check(design), units={"checks_3d": 1})


@app.get("/api/view/{token}.glb")                 # before /api/view/{token}, which would take "x.glb" as a token
def view_glb(token: str):
    if token not in _VIEWS:
        raise HTTPException(404, "no such view")
    return Response(_VIEWS[token][0], media_type="model/gltf-binary")


@app.get("/api/view/{token}")
def view_meta(token: str):
    if token not in _VIEWS:
        raise HTTPException(404, "no such view (views are kept for the last 12 runs)")
    return _VIEWS[token][1]


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
