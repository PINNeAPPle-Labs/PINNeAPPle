"""RideTune AI: web API + single page for ``pinneapple_design.surrogate_inverse.suspension_arm``.

A surrogate of the cantilever bending stress is trained once in a background thread at start-up; ``/api/optimize``
finds the lightest cross-section that keeps a fatigue-life target through it and re-checks the design against the
closed form; ``/api/sensitivity`` gives the local derivatives and elasticities of life and mass on the closed form.

Run:  uvicorn ridetune_ai.api:app --port 8163   (from apps/ridetune_ai, apps/_shared on the path)

Environment (all optional):
  RIDE_USER / RIDE_PASSWORD (or RIDE_PASSWORD_HASH)   HTTP Basic login on everything except /health (appkit)
  RIDE_EPOCHS                                          surrogate training epochs (default 1500)
  RIDE_THREADS                                         torch threads for the training (default 1)
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

from pinneapple_design.surrogate_inverse import suspension_arm as sa

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

STATIC = os.path.join(os.path.dirname(__file__), "static")
EPOCHS = int(os.environ.get("RIDE_EPOCHS") or 1500)
THREADS = int(os.environ.get("RIDE_THREADS") or 1)

app = FastAPI(title="RideTune AI", version="0.3.0",
              description="Minimum-mass cross-section of a cantilever suspension arm for a fatigue-life target through "
                          "a frozen MLP surrogate, re-checked against the closed form, with local sensitivities.")
install(app, prefix="RIDE")

_state: dict = {"fit": None, "error": None, "started": time.time()}


def _train() -> None:
    try:
        torch.set_num_threads(THREADS)
        _state["fit"] = sa.train_surrogate(epochs=EPOCHS)
    except Exception as exc:  # reported by /api/status
        _state["error"] = f"{type(exc).__name__}: {exc}"


threading.Thread(target=_train, name="ride-train", daemon=True).start()


class OptimizeBody(BaseModel):
    load_n: float
    target_life_cycles: float


class DesignBody(BaseModel):
    width_m: float
    thickness_m: float
    load_n: float = 1200.0


def _load(load_n: float) -> None:
    if not sa.LOAD_BOUNDS_N[0] <= load_n <= sa.LOAD_BOUNDS_N[1]:
        raise HTTPException(400, f"load_n must be between {sa.LOAD_BOUNDS_N}")


def _design(b: DesignBody) -> sa.ArmDesign:
    if not (b.width_m > 0 and b.thickness_m > 0):
        raise HTTPException(400, "width and thickness must be positive")
    return sa.ArmDesign(b.width_m, b.thickness_m)


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
        "load_bounds_n": sa.LOAD_BOUNDS_N,
        "target_life_bounds_cycles": sa.TARGET_LIFE_BOUNDS_CYCLES,
        "bounds": {"width_m": sa.WIDTH_BOUNDS_M, "thickness_m": sa.THICKNESS_BOUNDS_M},
        "baseline_design": sa.design_to_dict(sa.BASELINE_DESIGN),
    }


@app.post("/api/evaluate")
def evaluate(body: DesignBody):
    _load(body.load_n)
    d = _design(body)
    return {"bending_stress_pa": sa.bending_stress_pa(d, body.load_n),
            "fatigue_life_cycles": sa.fatigue_life_cycles(d, body.load_n),
            "design": sa.design_to_dict(d), "stress_field": sa.stress_field(d, body.load_n)}


@app.post("/api/sensitivity")
def sensitivity(body: DesignBody):
    _load(body.load_n)
    if not sa.WIDTH_BOUNDS_M[0] <= body.width_m <= sa.WIDTH_BOUNDS_M[1]:
        raise HTTPException(400, f"width_m must be between {sa.WIDTH_BOUNDS_M}")
    if not sa.THICKNESS_BOUNDS_M[0] <= body.thickness_m <= sa.THICKNESS_BOUNDS_M[1]:
        raise HTTPException(400, f"thickness_m must be between {sa.THICKNESS_BOUNDS_M}")
    return sa.sensitivity_to_dict(sa.compute_sensitivity(body.width_m, body.thickness_m, body.load_n))


@app.post("/api/mesh.stl")
def mesh_stl(body: DesignBody):
    return Response(content=sa.export_stl_bytes(_design(body)), media_type="model/stl")


@app.post("/api/optimize")
def optimize(body: OptimizeBody):
    fit = _state["fit"]
    if fit is None:
        raise HTTPException(503, "surrogate not trained yet, try again shortly")
    _load(body.load_n)
    lo, hi = sa.TARGET_LIFE_BOUNDS_CYCLES
    if not lo <= body.target_life_cycles <= hi:
        raise HTTPException(400, f"target_life_cycles must be between {lo:g} and {hi:g}")
    r = sa.optimize_design(fit.model, body.load_n, body.target_life_cycles)
    card = sa.verify_design(r.design, body.load_n, body.target_life_cycles, r.surrogate_predicted_stress_pa)
    base_mass = sa.BASELINE_DESIGN.mass_kg()
    return {
        "input": {"load_n": body.load_n, "target_life_cycles": body.target_life_cycles},
        "baseline": {"design": sa.design_to_dict(sa.BASELINE_DESIGN),
                     "life_cycles": sa.fatigue_life_cycles(sa.BASELINE_DESIGN, body.load_n)},
        "optimized": {"design": sa.design_to_dict(r.design),
                      "surrogate_predicted_stress_mpa": r.surrogate_predicted_stress_pa / 1e6},
        "verification": card.to_dict(),
        "mass_reduction_kg": base_mass - r.design.mass_kg(),
        "mass_reduction_pct": (1 - r.design.mass_kg() / base_mass) * 100,
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
