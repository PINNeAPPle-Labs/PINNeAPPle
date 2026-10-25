"""Design Optimization: a NACA 4-digit section redesigned by a surrogate-driven gradient loop to cut its profile drag
while holding its lift, re-run live on every page load.

Glue only: the study is ``pinneapple_design.design_optimizer.airfoil_drag.run_design_optimization`` (closed-form
ground truth in ``pinneapple_physics.closed_form.thin_airfoil``).

Run:  uvicorn design_optimization.api:app --port 8181   (from apps/design_optimization)

Environment (all optional): DOPT_USER / DOPT_PASSWORD (or DOPT_PASSWORD_HASH) for an HTTP Basic login on everything
except /health; DOPT_MAX_HEAVY concurrent runs per worker (default 2); the rest in apps/_shared/README.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pinneapple_design.design_optimizer.airfoil_drag import run_design_optimization

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Design Optimization", version=VERSION,
              description="A wing section redesigned live: an MLP surrogate of thin-airfoil lift and Hoerner profile "
                          "drag, and Adam steps on (camber, thickness) through it, every iterate re-checked on the "
                          "closed-form physics. GET /api/scenario runs one study (about 2 s on a CPU).")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="DOPT")
_HEAVY = BusyLimiter("DOPT_MAX_HEAVY", 2)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/scenario")
def scenario(seed: int | None = Query(None, ge=0, lt=2**32), n_iterations: int = Query(150, ge=1, le=300)):
    """Train a fresh surrogate and run the loop (``seed`` for a reproducible run)."""
    with _HEAVY:
        return run_design_optimization(n_iterations=n_iterations, seed=seed)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
