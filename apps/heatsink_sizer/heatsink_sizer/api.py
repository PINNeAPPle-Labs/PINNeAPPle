"""HeatSink Sizer web API + single-page UI.

Run:  uvicorn heatsink_sizer.api:app --port 8080   (from apps/heatsink_sizer)

Environment (all optional):
  HSS_USER / HSS_PASSWORD   require HTTP Basic auth on everything except /health
  HSS_MAX_SIZING            concurrent /api/size runs per worker (default 2);
                            extra requests get 429 instead of queueing up
"""
from __future__ import annotations

import os
import sys
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinneapple_design.thermal.heatsink.engine import H_BAND, DesignInput, OperatingInput, evaluate_design
from pinneapple_design.thermal.heatsink.sizer import PROCESSES, SizingRequest, size
from pinneapple_design.thermal.heatsink.surrogate import COMMON_RANGES, MODE_RANGES, load_surrogates
from pinneapple_physics.closed_form.plate_fin_heatsink import MATERIALS

# apps/_shared: login, busy limiter and the vendored three.js used by every app
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
STATIC = os.path.join(os.path.dirname(__file__), "static")
# the app's trained surrogates (python -m pinneapple_design.thermal.heatsink.surrogate --artifacts <this dir>)
ARTIFACTS = os.environ.get("PINNEAPPLE_HEATSINK_ARTIFACTS") or os.path.join(os.path.dirname(__file__), "..",
                                                                            "artifacts")

app = FastAPI(title="HeatSink Sizer", version=VERSION,
              description="Plate-fin heat sink sizing with verified physics (PINNeAPPle).")
install(app, prefix="HSS")
_SURROGATES = load_surrogates(ARTIFACTS)
_SIZING = BusyLimiter("HSS_MAX_SIZING")

MaterialName = Literal[tuple(MATERIALS)]  # type: ignore[valid-type]


class DesignModel(BaseModel):
    base_width_mm: float = Field(80, gt=5, le=500)
    base_depth_mm: float = Field(80, gt=5, le=500)
    base_thickness_mm: float = Field(6, gt=0.5, le=50)
    n_fins: int = Field(24, ge=2, le=300)
    fin_thickness_mm: float = Field(1.2, gt=0.1, le=20)
    fin_height_mm: float = Field(35, gt=1, le=200)
    material: MaterialName = "al6063"


class OperatingModel(BaseModel):
    power_w: float = Field(100, gt=0, le=5000)
    t_ambient_c: float = Field(25, ge=-40, le=120)
    air_velocity_m_s: float = Field(2.0, ge=0, le=30)
    source_width_mm: float = Field(30, gt=1, le=500)
    source_depth_mm: float = Field(30, gt=1, le=500)
    source_x_mm: Optional[float] = Field(None, ge=0)
    source_y_mm: Optional[float] = Field(None, ge=0)
    tim_k_mm2_w: float = Field(10, ge=0, le=1000)
    t_limit_c: float = Field(85, gt=-40, le=250)


class EvaluateRequest(BaseModel):
    design: DesignModel
    operating: OperatingModel


class SizeModel(BaseModel):
    power_w: float = Field(100, gt=0, le=5000)
    t_limit_c: float = Field(70, gt=-40, le=250)
    t_ambient_c: float = Field(25, ge=-40, le=120)
    air_velocity_m_s: float = Field(2.0, ge=0, le=30)
    source_width_mm: float = Field(30, gt=1, le=200)
    source_depth_mm: float = Field(30, gt=1, le=200)
    tim_k_mm2_w: float = Field(10, ge=0, le=1000)
    max_width_mm: float = Field(100, gt=5, le=500)
    max_depth_mm: float = Field(100, gt=5, le=500)
    max_height_mm: float = Field(50, gt=10, le=300)
    materials: List[MaterialName] = ["al6063"]
    process: Literal[tuple(PROCESSES)] = "extruded"  # type: ignore[valid-type]
    max_pressure_drop_pa: Optional[float] = Field(None, gt=0)
    objective: Literal["mass", "volume", "temperature"] = "mass"


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "surrogates": sorted(_SURROGATES)}


@app.get("/api/meta")
def meta():
    return {
        "version": VERSION,
        "materials": {k: {"label": v["label"], "k_w_mk": v["k"], "density": v["rho"]}
                      for k, v in MATERIALS.items()},
        "processes": PROCESSES,
        "surrogates": {m: {"verdict": s.report["verdict"],
                           "design_space": {**COMMON_RANGES, **MODE_RANGES[m]}}
                       for m, s in _SURROGATES.items()},
        "convection_uncertainty_band": H_BAND,
    }


@app.post("/api/evaluate")
def api_evaluate(req: EvaluateRequest):
    try:
        return evaluate_design(DesignInput(**req.design.model_dump()),
                               OperatingInput(**req.operating.model_dump()))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/api/size")
def api_size(req: SizeModel):
    with _SIZING:
        try:
            return size(SizingRequest(**req.model_dump()), _SURROGATES)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e


@app.get("/api/surrogate/{mode}/report")
def surrogate_report(mode: Literal["forced", "natural"]):
    if mode not in _SURROGATES:
        raise HTTPException(status_code=404, detail=f"No trained {mode} surrogate.")
    return _SURROGATES[mode].report


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
