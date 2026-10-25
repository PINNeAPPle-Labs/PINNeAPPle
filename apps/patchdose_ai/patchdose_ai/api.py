"""PatchDose AI: web API + single page for ``pinneapple_design.surrogate_inverse.transdermal_patch``.

A surrogate of Fick's-law membrane release is trained once in a background thread at start-up; ``/api/optimize``
finds the smallest patch for a target release rate and wear time through it and re-checks the design against the
formula; ``/api/simulate`` returns the release over the wear time. A mass-transfer design exercise with generic
coefficients, not a dosing or medical tool.

Run:  uvicorn patchdose_ai.api:app --port 8162   (from apps/patchdose_ai, apps/_shared on the path)

Environment (all optional):
  PATCH_USER / PATCH_PASSWORD (or PATCH_PASSWORD_HASH)   HTTP Basic login on everything except /health (appkit)
  PATCH_EPOCHS                                            surrogate training epochs (default 1500)
  PATCH_THREADS                                           torch threads for the training (default 1)
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

from pinneapple_design.surrogate_inverse import transdermal_patch as tp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

STATIC = os.path.join(os.path.dirname(__file__), "static")
EPOCHS = int(os.environ.get("PATCH_EPOCHS") or 1500)
THREADS = int(os.environ.get("PATCH_THREADS") or 1)

app = FastAPI(title="PatchDose AI", version="0.3.0",
              description="Inverse design of a membrane-controlled release patch (generic coefficients) through a "
                          "frozen MLP surrogate of Fick's law, re-checked against the formula. Not a medical tool.")
install(app, prefix="PATCH")

_state: dict = {"fit": None, "error": None, "started": time.time()}


def _train() -> None:
    try:
        torch.set_num_threads(THREADS)
        _state["fit"] = tp.train_surrogate(epochs=EPOCHS)
    except Exception as exc:  # reported by /api/status
        _state["error"] = f"{type(exc).__name__}: {exc}"


threading.Thread(target=_train, name="patch-train", daemon=True).start()


class OptimizeBody(BaseModel):
    target_rate_mg_day: float
    target_wear_days: float


class DesignBody(BaseModel):
    area_m2: float
    membrane_thickness_m: float
    reservoir_conc_kg_m3: float


class SimulateBody(DesignBody):
    duration_days: float | None = None
    n_points: int = 100


def _design(b: DesignBody) -> tp.PatchDesign:
    if not (b.area_m2 > 0 and b.membrane_thickness_m > 0 and b.reservoir_conc_kg_m3 > 0):
        raise HTTPException(400, "area, membrane thickness and concentration must be positive")
    return tp.PatchDesign(b.area_m2, b.membrane_thickness_m, b.reservoir_conc_kg_m3)


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
        "target_rate_bounds_mg_day": tp.TARGET_RATE_BOUNDS_MG_DAY,
        "target_wear_bounds_days": tp.TARGET_WEAR_BOUNDS_DAYS,
        "bounds": {"area_m2": tp.AREA_BOUNDS_M2, "membrane_thickness_m": tp.MEMBRANE_THICKNESS_BOUNDS_M,
                   "reservoir_conc_kg_m3": tp.RESERVOIR_CONC_BOUNDS_KG_M3},
        "baseline_design": tp.design_to_dict(tp.BASELINE_DESIGN),
    }


@app.post("/api/evaluate")
def evaluate(body: DesignBody):
    d = _design(body)
    viol = tp.constraint_violations(d, target_wear_days=0.0)
    return {"release_rate_mg_day": tp.release_rate_mg_day(d), "depletion_time_days": tp.depletion_time_days(d),
            "design": tp.design_to_dict(d),
            "violated_constraints": [k for k, v in viol.items() if v > 0 and k != "insufficient_wear_duration"],
            "concentration_field": tp.concentration_field(d)}


@app.post("/api/mesh.stl")
def mesh_stl(body: DesignBody):
    return Response(content=tp.export_stl_bytes(_design(body)), media_type="model/stl")


@app.post("/api/simulate")
def simulate(body: SimulateBody):
    d = _design(body)
    ts = tp.simulate_release_timeseries(d, duration_days=body.duration_days, n_points=max(min(body.n_points, 2000), 2))
    return {"design": tp.design_to_dict(d), "depletion_time_days": tp.depletion_time_days(d),
            "duration_days": ts[-1]["day"], "timeseries": ts, "total_delivered_mg": ts[-1]["cumulative_delivered_mg"]}


@app.post("/api/optimize")
def optimize(body: OptimizeBody):
    fit = _state["fit"]
    if fit is None:
        raise HTTPException(503, "surrogate not trained yet, try again shortly")
    lo, hi = tp.TARGET_RATE_BOUNDS_MG_DAY
    if not lo <= body.target_rate_mg_day <= hi:
        raise HTTPException(400, f"target_rate_mg_day must be between {lo} and {hi}")
    lo, hi = tp.TARGET_WEAR_BOUNDS_DAYS
    if not lo <= body.target_wear_days <= hi:
        raise HTTPException(400, f"target_wear_days must be between {lo} and {hi}")
    r = tp.optimize_design(fit.model, body.target_rate_mg_day, body.target_wear_days)
    card = tp.verify_design(r.design, body.target_rate_mg_day, body.target_wear_days, r.surrogate_predicted_rate_mg_day)
    return {
        "input": {"target_rate_mg_day": body.target_rate_mg_day, "target_wear_days": body.target_wear_days},
        "baseline": {"design": tp.design_to_dict(tp.BASELINE_DESIGN),
                     "rate_mg_day": tp.release_rate_mg_day(tp.BASELINE_DESIGN),
                     "depletion_days": tp.depletion_time_days(tp.BASELINE_DESIGN)},
        "optimized": {"design": tp.design_to_dict(r.design),
                      "surrogate_predicted_rate_mg_day": r.surrogate_predicted_rate_mg_day},
        "verification": card.to_dict(),
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
