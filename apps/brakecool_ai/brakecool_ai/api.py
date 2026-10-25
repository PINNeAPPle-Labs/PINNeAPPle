"""BrakeCool AI: web API + single page for ``pinneapple_design.surrogate_inverse.brake_rotor``.

A surrogate of the vented-rotor fin-array formula is trained once in a background thread at start-up;
``/api/optimize`` descends through it and re-checks the design against the formula; ``/api/uncertainty`` propagates
machining tolerances through the formula (Monte Carlo). ``/api/evaluate`` and ``/api/mesh.stl`` use no network.

Run:  uvicorn brakecool_ai.api:app --port 8161   (from apps/brakecool_ai, apps/_shared on the path)

Environment (all optional):
  BRAKE_USER / BRAKE_PASSWORD (or BRAKE_PASSWORD_HASH)   HTTP Basic login on everything except /health (appkit)
  BRAKE_EPOCHS                                            surrogate training epochs (default 1500)
  BRAKE_THREADS                                           torch threads for the training (default 1)
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

from pinneapple_design.surrogate_inverse import brake_rotor as br

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

STATIC = os.path.join(os.path.dirname(__file__), "static")
EPOCHS = int(os.environ.get("BRAKE_EPOCHS") or 1500)
THREADS = int(os.environ.get("BRAKE_THREADS") or 1)

app = FastAPI(title="BrakeCool AI", version="0.3.0",
              description="Inverse design of vented brake-rotor vanes through a frozen MLP surrogate of a fin-array "
                          "model, re-checked against the model, with Monte Carlo of machining tolerances.")
install(app, prefix="BRAKE")

_state: dict = {"fit": None, "error": None, "started": time.time()}


def _train() -> None:
    try:
        torch.set_num_threads(THREADS)
        _state["fit"] = br.train_surrogate(epochs=EPOCHS)
    except Exception as exc:  # reported by /api/status
        _state["error"] = f"{type(exc).__name__}: {exc}"


threading.Thread(target=_train, name="brake-train", daemon=True).start()


class OptimizeBody(BaseModel):
    power_w: float
    cooling_profile: str = "highway"


class DesignBody(BaseModel):
    vane_count: float
    vane_gap_m: float
    vane_thickness_m: float
    power_w: float = 800.0
    cooling_profile: str = "highway"


class UncertaintyBody(DesignBody):
    n_samples: int = 200


def _h(power_w: float, profile: str) -> float:
    if profile not in br.COOLING_PROFILES:
        raise HTTPException(400, f"unknown cooling profile, use one of {list(br.COOLING_PROFILES)}")
    if not 0 < power_w <= 2000:
        raise HTTPException(400, "power_w must be between 0 and 2000 W (sustained braking power, not a single stop)")
    return br.COOLING_PROFILES[profile]


def _design(b: DesignBody) -> br.RotorDesign:
    if not (b.vane_gap_m > 0 and b.vane_thickness_m > 0 and b.vane_count >= 1):
        raise HTTPException(400, "vane gap and thickness must be positive, vane count at least 1")
    return br.RotorDesign(max(round(b.vane_count), 1), b.vane_gap_m, b.vane_thickness_m)


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
        "cooling_profiles": br.COOLING_PROFILES,
        "bounds": {"vane_count": br.VANE_COUNT_BOUNDS, "vane_gap_m": br.VANE_GAP_BOUNDS_M,
                   "vane_thickness_m": br.VANE_THICKNESS_BOUNDS_M},
        "baseline_design": br.design_to_dict(br.BASELINE_DESIGN),
    }


@app.post("/api/evaluate")
def evaluate(body: DesignBody):
    h = _h(body.power_w, body.cooling_profile)
    d = _design(body)
    return {"rotor_temperature_c": br.rotor_temperature_c(d, body.power_w, h),
            "design": br.design_to_dict(d),
            "violated_constraints": [k for k, v in br.constraint_violations(d).items() if v > 0],
            "thermal_field": br.thermal_field(d, body.power_w, h)}


@app.post("/api/mesh.stl")
def mesh_stl(body: DesignBody):
    return Response(content=br.export_stl_bytes(_design(body)), media_type="model/stl")


@app.post("/api/uncertainty")
def uncertainty(body: UncertaintyBody):
    h = _h(body.power_w, body.cooling_profile)
    d = _design(body)
    n = max(min(int(body.n_samples), 5000), 50)
    rep = br.monte_carlo_uncertainty(d, body.power_w, h, br.MAX_SAFE_ROTOR_TEMP_C, n_samples=n)
    return {"design": br.design_to_dict(d), "nominal_temperature_c": br.rotor_temperature_c(d, body.power_w, h),
            "tolerances": {"vane_thickness_tolerance_m": br.VANE_THICKNESS_TOLERANCE_M,
                           "vane_gap_tolerance_m": br.VANE_GAP_TOLERANCE_M}, **rep}


@app.post("/api/optimize")
def optimize(body: OptimizeBody):
    fit = _state["fit"]
    if fit is None:
        raise HTTPException(503, "surrogate not trained yet, try again shortly")
    h = _h(body.power_w, body.cooling_profile)
    base = br.rotor_temperature_c(br.BASELINE_DESIGN, body.power_w, h)
    r = br.optimize_design(fit.model, body.power_w, h)
    card = br.verify_design(r.design, body.power_w, h, r.surrogate_predicted_temp_c)
    return {
        "input": {"power_w": body.power_w, "cooling_profile": body.cooling_profile, "h_conv": h},
        "baseline": {"design": br.design_to_dict(br.BASELINE_DESIGN), "rotor_temperature_c": base},
        "optimized": {"design": br.design_to_dict(r.design), "surrogate_predicted_temp_c": r.surrogate_predicted_temp_c},
        "verification": card.to_dict(),
        "temp_reduction_c": base - card.real_temperature_c,
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
