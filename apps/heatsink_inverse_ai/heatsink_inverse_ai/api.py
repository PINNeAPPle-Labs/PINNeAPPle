"""HeatSink Design AI: web API + single page for ``pinneapple_design.surrogate_inverse.heatsink_fins``.

A surrogate of the fin-array formula is trained once in a background thread at start-up; ``/api/optimize`` descends
through it and re-checks the design against the formula. ``/api/evaluate`` and ``/api/mesh.stl`` use the formula and
the mesh builder only (no network), ``/api/cross_check`` compares the formula with a finite-volume solve of one fin.

Run:  uvicorn heatsink_inverse_ai.api:app --port 8160   (from apps/heatsink_inverse_ai, apps/_shared on the path)

Environment (all optional):
  HSD_USER / HSD_PASSWORD (or HSD_PASSWORD_HASH)   HTTP Basic login on everything except /health (appkit)
  HSD_EPOCHS                                        surrogate training epochs (default 1500)
  HSD_THREADS                                       torch threads for the training (default 1)
"""
from __future__ import annotations

import os
import sys
import threading
import time

import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pinneapple_design.surrogate_inverse import heatsink_fins as hs

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

STATIC = os.path.join(os.path.dirname(__file__), "static")
EPOCHS = int(os.environ.get("HSD_EPOCHS") or 1500)
THREADS = int(os.environ.get("HSD_THREADS") or 1)

app = FastAPI(title="HeatSink Design AI", version="0.3.0",
              description="Gradient-based inverse design of a straight-fin heat sink through a frozen MLP surrogate of "
                          "the fin-array formula, re-checked against the formula.")
install(app, prefix="HSD")

_state: dict = {"fit": None, "error": None, "started": time.time()}


def _train() -> None:
    try:
        torch.set_num_threads(THREADS)
        _state["fit"] = hs.train_surrogate(epochs=EPOCHS)
    except Exception as exc:  # reported by /api/status, never hidden
        _state["error"] = f"{type(exc).__name__}: {exc}"


threading.Thread(target=_train, name="hsd-train", daemon=True).start()


class OptimizeBody(BaseModel):
    power_w: float
    cooling_profile: str = "standard_fan"


class DesignBody(BaseModel):
    fin_count: float
    fin_height_m: float
    fin_thickness_m: float
    power_w: float = 65.0
    cooling_profile: str = "standard_fan"


def _h(power_w: float, profile: str) -> float:
    if profile not in hs.COOLING_PROFILES:
        raise HTTPException(400, f"unknown cooling profile, use one of {list(hs.COOLING_PROFILES)}")
    if not 0 < power_w <= 400:
        raise HTTPException(400, "power_w must be between 0 and 400 W")
    return hs.COOLING_PROFILES[profile]


def _design(b: DesignBody) -> hs.HeatsinkDesign:
    if not (b.fin_height_m > 0 and b.fin_thickness_m > 0 and b.fin_count >= 1):
        raise HTTPException(400, "fin height and thickness must be positive, fin count at least 1")
    return hs.HeatsinkDesign(max(round(b.fin_count), 1), b.fin_height_m, b.fin_thickness_m)


@app.get("/health")
def health():
    return {"status": "ok", "surrogate_trained": _state["fit"] is not None}


@app.get("/api/status")
def status():
    fit = _state["fit"]
    return {
        "surrogate_trained": fit is not None,
        "training_error": _state["error"],
        "validation_relative_error": fit.val_mean_rel_err if fit else None,
        "validation_p95_relative_error": fit.val_p95_rel_err if fit else None,
        "training_time_s": fit.seconds if fit else None,
        "epochs": EPOCHS,
        "cooling_profiles": hs.COOLING_PROFILES,
        "bounds": {"fin_count": hs.FIN_COUNT_BOUNDS, "fin_height_m": hs.FIN_HEIGHT_BOUNDS_M,
                   "fin_thickness_m": hs.FIN_THICKNESS_BOUNDS_M},
        "baseline_design": hs.design_to_dict(hs.BASELINE_DESIGN),
    }


@app.post("/api/evaluate")
def evaluate(body: DesignBody):
    """The formula only (no surrogate): what the sliders show."""
    h = _h(body.power_w, body.cooling_profile)
    d = _design(body)
    return {"junction_temperature_c": hs.junction_temperature_c(d, body.power_w, h),
            "design": hs.design_to_dict(d),
            "violated_constraints": [k for k, v in hs.constraint_violations(d).items() if v > 0],
            "thermal_field": hs.thermal_field(d, body.power_w, h)}


@app.post("/api/mesh.stl")
def mesh_stl(body: DesignBody):
    return Response(content=hs.export_stl_bytes(_design(body)), media_type="model/stl")


@app.post("/api/cross_check")
def cross_check(body: DesignBody):
    h = _h(body.power_w, body.cooling_profile)
    d = _design(body)
    if any(v > 0 for v in hs.constraint_violations(d).values()):
        raise HTTPException(400, "the design breaks a geometric rule; fix the sliders first")
    out = hs.fin_cell_check(d, body.power_w, h)
    out["design"] = hs.design_to_dict(d)
    out["method_note"] = ("2-D finite volumes in one fin pitch (exact for the 3-D unit cell when nothing varies along "
                          "the depth), symmetry at the sides, the same h as the formula: it checks the 1-D fin theory, "
                          "not the airflow. The cell gets P/N of the power over one pitch, so with few, widely spaced "
                          "fins it differs from the formula's footprint by the edge effect (about 8 % at 3-5 fins).")
    return out


@app.post("/api/optimize")
def optimize(body: OptimizeBody):
    fit = _state["fit"]
    if fit is None:
        raise HTTPException(503, "surrogate not trained yet, try again shortly")
    h = _h(body.power_w, body.cooling_profile)
    base = hs.junction_temperature_c(hs.BASELINE_DESIGN, body.power_w, h)
    r = hs.optimize_design(fit.model, body.power_w, h)
    card = hs.verify_design(r.design, body.power_w, h, r.surrogate_predicted_temp_c)
    return {
        "input": {"power_w": body.power_w, "cooling_profile": body.cooling_profile, "h_conv": h},
        "baseline": {"design": hs.design_to_dict(hs.BASELINE_DESIGN), "junction_temperature_c": base},
        "optimized": {"design": hs.design_to_dict(r.design), "surrogate_predicted_temp_c": r.surrogate_predicted_temp_c},
        "verification": card.to_dict(),
        "temp_reduction_c": base - card.real_temperature_c,
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
