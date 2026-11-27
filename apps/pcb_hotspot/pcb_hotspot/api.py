"""PCB Hotspot web API + single-page UI.

Run:  uvicorn pcb_hotspot.api:app --port 8081   (from apps/pcb_hotspot)

Environment (all optional):
  PCB_USER / PCB_PASSWORD   HTTP Basic login on everything except /health
  PCB_MAX_HEAVY             concurrent what-if / calibration runs per worker (default 2)
"""
from __future__ import annotations

import os
import sys
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from pinneapple_physics.closed_form.pcb_thermal import PACKAGES, Environment

from pinneapple_physics.pcb.engine import H_BAND, calibrate, evaluate, what_if
from pinneapple_physics.pcb.solver import Board, Component

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
STATIC = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="PCB Hotspot", version=VERSION,
              description="Board-level PCB thermal analysis with calibration to measurements (PINNeAPPle).")
install(app, prefix="PCB")
_HEAVY = BusyLimiter("PCB_MAX_HEAVY")

PackageName = Literal[tuple(PACKAGES)]  # type: ignore[valid-type]


class ComponentModel(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    x_mm: float
    y_mm: float
    power_w: float = Field(ge=0, le=500)
    package: PackageName = "QFN-32 5x5"
    side: Literal["top", "bottom"] = "top"
    tj_max_c: float = Field(125, gt=-40, le=250)
    theta_jb: float | None = Field(None, gt=0, le=1000)
    theta_jc: float | None = Field(None, gt=0, le=10000)
    w_mm: float | None = Field(None, gt=0.2, le=200)
    d_mm: float | None = Field(None, gt=0.2, le=200)
    h_mm: float | None = Field(None, gt=0.1, le=100)
    vias: int = Field(0, ge=0, le=2000)


class BoardModel(BaseModel):
    width_mm: float = Field(160, gt=5, le=600)
    depth_mm: float = Field(100, gt=5, le=600)
    n_copper: int = Field(4, ge=1, le=20)
    thickness_mm: float = Field(1.6, gt=0.2, le=6)
    outer_oz: float = Field(1.0, gt=0.2, le=6)
    inner_oz: float = Field(1.0, gt=0.2, le=6)
    outer_coverage: float = Field(0.3, ge=0, le=1)
    inner_coverage: float = Field(0.9, ge=0, le=1)
    chassis_edges: bool = False
    chassis_t_c: float | None = Field(None, gt=-60, le=200)


class EnvModel(BaseModel):
    t_ambient_c: float = Field(25, ge=-40, le=120)
    air_velocity_m_s: float = Field(0, ge=0, le=20)
    orientation: Literal["vertical", "horizontal"] = "vertical"
    emissivity: float = Field(0.9, ge=0, le=1)


class Case(BaseModel):
    board: BoardModel = BoardModel()
    environment: EnvModel = EnvModel()
    components: list[ComponentModel] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def unique_names(self):
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise ValueError("component names must be unique")
        return self

    def objects(self):
        return (Board(**self.board.model_dump()), [Component(**c.model_dump()) for c in self.components],
                Environment(**self.environment.model_dump()))


class Measurement(BaseModel):
    t_c: float = Field(gt=-60, le=300)
    x_mm: float | None = None
    y_mm: float | None = None
    side: Literal["top", "bottom"] = "top"
    component: str | None = None

    @model_validator(mode="after")
    def where(self):
        if self.component is None and (self.x_mm is None or self.y_mm is None):
            raise ValueError("each measurement needs x_mm/y_mm or a component name")
        return self


class CalibrationRequest(Case):
    measurements: list[Measurement] = Field(min_length=2, max_length=200)
    noise_std_c: float = Field(1.0, gt=0, le=20)


def _run(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "packages": PACKAGES, "h_uncertainty_band": H_BAND,
            "note": "Package theta values are typical; enter your datasheet's theta_JB and theta_JC(top)."}


@app.post("/api/solve")
def api_solve(case: Case, quick: bool = False):
    return _run(evaluate, *case.objects(), quick=quick)


@app.post("/api/whatif")
def api_whatif(case: Case):
    with _HEAVY:
        return _run(what_if, *case.objects())


@app.post("/api/calibrate")
def api_calibrate(req: CalibrationRequest):
    board, comps, env = req.objects()
    meas = [{k: v for k, v in m.model_dump().items() if v is not None} for m in req.measurements]
    with _HEAVY:
        return _run(calibrate, board, comps, env, meas, noise_std_c=req.noise_std_c)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
