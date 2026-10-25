"""Surrogate Speed: a trained heat-sink surrogate timed against a finite-volume solve of the fin unit cell, with the
agreement between them, over seven plate-fin designs.

Glue only: the study is ``pinneapple_design.thermal.fin_array_speed`` (closed form
``pinneapple_physics.closed_form.fin_array_conduction``, reference
``pinneapple_design.thermal.fin_cell``), designs, surrogate and mesh ``pinneapple_design.surrogate_inverse.heatsink_fins``. The
surrogate trains at startup (a few seconds); the sweep runs in a background thread and the page polls
``GET /api/sweep_status``; ``POST /api/run_sweep`` starts a new sweep.

Run:  uvicorn surrogate_speed.api:app --port 8185   (from apps/surrogate_speed)

Environment (all optional): SSPEED_USER / SSPEED_PASSWORD (or SSPEED_PASSWORD_HASH) for an HTTP Basic login on
everything except /health; SSPEED_AUTOSTART=0 to train and sweep on first request instead of at startup;
SSPEED_TRAIN_EPOCHS (default 1500) and SSPEED_GRID_CHECK=0 (skip the finer-grid check) to shorten; the rest in
apps/_shared/README.md.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pinneapple_design.surrogate_inverse import heatsink_fins as hf
from pinneapple_design.thermal import fin_array_speed as fas

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

VERSION = "1.0.0"
STATIC = Path(__file__).resolve().parent / "static"
EPOCHS = int(os.environ.get("SSPEED_TRAIN_EPOCHS") or 1500)
GRID_CHECK = (os.environ.get("SSPEED_GRID_CHECK") or "1") != "0"

_state: dict[str, Any] = {"surrogate": None, "val_rel_error": None, "training_time_s": None,
                          "sweep": {"status": "idle", "progress": 0, "total": len(fas.CURATED_DESIGNS),
                                    "designs": None, "summary": None, "started_at": None, "finished_at": None,
                                    "error": None}}
_lock = threading.Lock()


def _ensure_surrogate() -> None:
    with _lock:
        if _state["surrogate"] is None:
            t0 = time.time()
            sur, err = fas.train_surrogate(epochs=EPOCHS)
            _state.update(surrogate=sur, val_rel_error=err, training_time_s=time.time() - t0)


def _run() -> None:
    sweep = _state["sweep"]
    try:
        _ensure_surrogate()
        res = fas.run_sweep(_state["surrogate"], progress_cb=lambda i, n: sweep.update(progress=i),
                            refine=GRID_CHECK)
        sweep.update(designs=res["designs"], summary=res["summary"], status="done")
    except Exception as exc:  # noqa: BLE001 -- shown on the page
        sweep.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        sweep["finished_at"] = time.time()


def start_if_idle() -> bool:
    with _lock:
        if _state["sweep"]["status"] == "running":
            return False
        _state["sweep"].update(status="running", progress=0, error=None, started_at=time.time(), finished_at=None)
    threading.Thread(target=_run, daemon=True).start()
    return True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if (os.environ.get("SSPEED_AUTOSTART") or "1") != "0":
        start_if_idle()
    yield


app = FastAPI(title="Surrogate Speed", version=VERSION, lifespan=lifespan,
              description="A heat-sink surrogate timed against a finite-volume solve, with their agreement, over "
                          "seven plate-fin designs.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="SSPEED")


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/sweep_status")
def sweep_status():
    return {"surrogate_trained": _state["surrogate"] is not None, "validation_relative_error": _state["val_rel_error"],
            "training_time_s": _state["training_time_s"], "agreement_threshold_pct": fas.AGREEMENT_THRESHOLD_PCT,
            "reference_solver": "finite-volume fin pitch (pinneapple_design.thermal.fin_cell)",
            "sweep": _state["sweep"]}


@app.post("/api/run_sweep")
def run_sweep():
    started = start_if_idle()
    return {"started": started, "sweep": _state["sweep"]}


@lru_cache(maxsize=8)
def _mesh(name: str) -> dict:
    V, F = hf.mesh(fas.design_of(fas.design_by_name(name)))     # mm, base centred on z = 0, fin roots at +t_base/2
    return {"positions": np.round(V, 4).ravel().tolist(), "indices": F.ravel().tolist()}


@app.get("/api/mesh.json")
def mesh(design: str = "baseline"):
    """Triangle mesh (mm) of one of the seven designs."""
    try:
        return _mesh(design)
    except KeyError:
        raise HTTPException(404, f"unknown design '{design}'") from None


@app.get("/api/thermal_field")
def thermal_field(design: str = "baseline"):
    """Closed-form fin temperature profile parameters for colouring the mesh."""
    try:
        return fas.thermal_field(design)
    except KeyError:
        raise HTTPException(404, f"unknown design '{design}'") from None


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
