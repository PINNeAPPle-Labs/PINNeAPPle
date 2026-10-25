"""SoundShape AI: web API + single page for ``pinneapple_design.surrogate_inverse.reflex_port``.

A surrogate of the Helmholtz tuning formula is trained once in a background thread at start-up; ``/api/optimize``
tunes a bass-reflex port to a target frequency through it and ``/api/pareto_frontier`` sweeps the trade-off between
port volume and tuning accuracy; every design is re-checked against the formula.

Run:  uvicorn soundshape_ai.api:app --port 8164   (from apps/soundshape_ai, apps/_shared on the path)

Environment (all optional):
  SOUND_USER / SOUND_PASSWORD (or SOUND_PASSWORD_HASH)   HTTP Basic login on everything except /health (appkit)
  SOUND_EPOCHS                                            surrogate training epochs (default 1500)
  SOUND_THREADS                                           torch threads for the training (default 1)
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

from pinneapple_design.surrogate_inverse import reflex_port as rp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

STATIC = os.path.join(os.path.dirname(__file__), "static")
EPOCHS = int(os.environ.get("SOUND_EPOCHS") or 1500)
THREADS = int(os.environ.get("SOUND_THREADS") or 1)

app = FastAPI(title="SoundShape AI", version="0.3.0",
              description="Bass-reflex port tuning through a frozen MLP surrogate of the Helmholtz formula, with a "
                          "volume-vs-accuracy trade-off sweep; every design re-checked against the formula.")
install(app, prefix="SOUND")

_state: dict = {"fit": None, "error": None, "started": time.time()}


def _train() -> None:
    try:
        torch.set_num_threads(THREADS)
        _state["fit"] = rp.train_surrogate(epochs=EPOCHS)
    except Exception as exc:  # reported by /api/status
        _state["error"] = f"{type(exc).__name__}: {exc}"


threading.Thread(target=_train, name="sound-train", daemon=True).start()


class OptimizeBody(BaseModel):
    cabinet_volume_l: float
    target_freq_hz: float


class FrontierBody(OptimizeBody):
    n_points: int = 8


class DesignBody(BaseModel):
    radius_m: float
    length_m: float
    cabinet_volume_l: float = 20.0


def _volume(v: float) -> None:
    if not rp.CABINET_VOLUME_BOUNDS_L[0] <= v <= rp.CABINET_VOLUME_BOUNDS_L[1]:
        raise HTTPException(400, f"cabinet_volume_l must be between {rp.CABINET_VOLUME_BOUNDS_L}")


def _target(f: float) -> None:
    if not rp.TARGET_FREQ_BOUNDS_HZ[0] <= f <= rp.TARGET_FREQ_BOUNDS_HZ[1]:
        raise HTTPException(400, f"target_freq_hz must be between {rp.TARGET_FREQ_BOUNDS_HZ}")


def _design(b: DesignBody) -> rp.PortDesign:
    if not (b.radius_m > 0 and b.length_m > 0):
        raise HTTPException(400, "radius and length must be positive")
    return rp.PortDesign(b.radius_m, b.length_m)


def _fit():
    if _state["fit"] is None:
        raise HTTPException(503, "surrogate not trained yet, try again shortly")
    return _state["fit"]


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
        "bounds": {"radius_m": rp.PORT_RADIUS_BOUNDS_M, "length_m": rp.PORT_LENGTH_BOUNDS_M,
                   "cabinet_volume_l": rp.CABINET_VOLUME_BOUNDS_L, "target_freq_hz": rp.TARGET_FREQ_BOUNDS_HZ},
        "baseline_design": rp.design_to_dict(rp.BASELINE_DESIGN),
    }


@app.post("/api/evaluate")
def evaluate(body: DesignBody):
    _volume(body.cabinet_volume_l)
    d = _design(body)
    return {"resonant_frequency_hz": rp.resonant_frequency_hz(d, body.cabinet_volume_l),
            "design": rp.design_to_dict(d),
            "violated_constraints": [k for k, v in rp.constraint_violations(d, body.cabinet_volume_l).items() if v > 0],
            "acoustic_field": rp.acoustic_field(d, body.cabinet_volume_l)}


@app.post("/api/mesh.stl")
def mesh_stl(body: DesignBody):
    return Response(content=rp.export_stl_bytes(_design(body)), media_type="model/stl")


@app.post("/api/optimize")
def optimize(body: OptimizeBody):
    fit = _fit()
    _volume(body.cabinet_volume_l)
    _target(body.target_freq_hz)
    r = rp.optimize_design(fit.model, body.cabinet_volume_l, body.target_freq_hz)
    card = rp.verify_design(r.design, body.cabinet_volume_l, body.target_freq_hz, r.surrogate_predicted_freq_hz)
    return {
        "input": {"cabinet_volume_l": body.cabinet_volume_l, "target_freq_hz": body.target_freq_hz},
        "baseline": {"design": rp.design_to_dict(rp.BASELINE_DESIGN),
                     "resonant_frequency_hz": rp.resonant_frequency_hz(rp.BASELINE_DESIGN, body.cabinet_volume_l)},
        "optimized": {"design": rp.design_to_dict(r.design), "surrogate_predicted_freq_hz": r.surrogate_predicted_freq_hz},
        "verification": card.to_dict(),
    }


@app.post("/api/pareto_frontier")
def pareto_frontier(body: FrontierBody):
    fit = _fit()
    _volume(body.cabinet_volume_l)
    _target(body.target_freq_hz)
    n = max(3, min(body.n_points, 20))
    points = rp.sweep_pareto_frontier(fit.model, body.cabinet_volume_l, body.target_freq_hz, n_points=n)
    return {
        "input": {"cabinet_volume_l": body.cabinet_volume_l, "target_freq_hz": body.target_freq_hz, "n_points": n},
        "frontier": [{"volume_weight": p.volume_weight, "design": rp.design_to_dict(p.design),
                      "surrogate_predicted_freq_hz": p.surrogate_predicted_freq_hz,
                      "real_achieved_freq_hz": p.real_achieved_freq_hz, "real_freq_error_pct": p.real_freq_error_pct,
                      "port_volume_m3": p.port_volume_m3} for p in points],
        "method_note": ("Each point minimises volume_weight * port_volume / max_port_volume + (1 - volume_weight) * "
                        "(relative frequency error)^2 through the surrogate, from several starts, and keeps the best "
                        "candidate by the closed-form loss; the reported frequency and volume are from the formula."),
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
