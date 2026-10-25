"""Parameter Discovery: a PINN recovers the hidden diffusivity of a bar from 256 noisy readings at one instant,
starting from a guess 3.3 times too high, with the Cramer-Rao bound of those readings next to the estimate.

Glue only: the study is ``pinneapple_analysis.inverse_problems.heat_diffusivity.run_inverse_discovery``. A run
(2,500 Adam steps) takes a minute or two on a CPU, so it runs in a background thread and the page polls
``GET /api/discovery/status``; ``POST /api/discovery/run`` starts a new one (409 while one is running).

Run:  uvicorn parameter_discovery.api:app --port 8163   (from apps/parameter_discovery)

Environment (all optional): PDISC_USER / PDISC_PASSWORD (or PDISC_PASSWORD_HASH) for an HTTP Basic login on
everything except /health; PDISC_AUTOSTART=0 to not start a run at startup; PDISC_EPOCHS (default 2500) and
PDISC_OBSERVATIONS (default 256) to shorten a run; the rest in apps/_shared/README.md.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pinneapple_analysis.inverse_problems import heat_diffusivity as hd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

VERSION = "1.0.0"
STATIC = Path(__file__).resolve().parent / "static"
EPOCHS = int(os.environ.get("PDISC_EPOCHS") or hd.N_EPOCHS)
OBSERVATIONS = int(os.environ.get("PDISC_OBSERVATIONS") or hd.N_OBSERVATIONS)
LOG_EVERY = max(1, EPOCHS // 500)

_state: dict[str, Any] = {"status": "idle", "progress": {"step": 0, "total": EPOCHS, "loss": None,
                                                          "alpha_estimate": None},
                          "result": None, "error": None, "started_at": None, "finished_at": None}
_lock = threading.Lock()


def _run() -> None:
    def progress(step: int, total: int, loss: float, alpha: float) -> None:
        _state["progress"] = {"step": step, "total": total, "loss": loss, "alpha_estimate": alpha}

    try:
        _state["result"] = hd.run_inverse_discovery(n_epochs=EPOCHS, n_observations=OBSERVATIONS,
                                                    log_every=LOG_EVERY, progress_cb=progress)
        _state["status"] = "done"
    except Exception as exc:  # noqa: BLE001 -- shown on the page
        _state.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        _state["finished_at"] = time.time()


def start_if_idle() -> bool:
    with _lock:
        if _state["status"] == "running":
            return False
        _state.update(status="running", error=None, started_at=time.time(), finished_at=None,
                      progress={"step": 0, "total": EPOCHS, "loss": None, "alpha_estimate": None})
        threading.Thread(target=_run, daemon=True).start()
        return True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if (os.environ.get("PDISC_AUTOSTART") or "1") != "0":
        start_if_idle()
    yield


app = FastAPI(title="Parameter Discovery", version=VERSION, lifespan=lifespan,
              description="Recover the diffusivity of the 1-D heat equation from noisy readings with a PINN trained "
                          "jointly with the unknown, plus the Fisher-information identifiability of those readings.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="PDISC")


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/discovery/status")
def discovery_status():
    return _state


@app.post("/api/discovery/run")
def discovery_run():
    if not start_if_idle():
        raise HTTPException(409, "a discovery run is already in progress")
    return {"started": True, "status": _state["status"]}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
