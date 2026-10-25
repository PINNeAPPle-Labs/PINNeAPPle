"""Model Benchmarking: four contrasting PINN architectures (Vanilla PINN, SIREN, PIELM, XTFC) trained live on the
viscous Burgers problem, ranked by accuracy and by Arena's physics-aware view.

Glue only: the run is ``pinneapple_arena.showcase.run_benchmark``. A full run takes minutes on a CPU, so it runs in
a background thread and the page polls ``GET /api/sweep_status``; ``POST /api/run_sweep`` starts a new run (409
while one is running).

Run:  uvicorn model_benchmarking.api:app --port 8162   (from apps/model_benchmarking)

Environment (all optional): MBENCH_USER / MBENCH_PASSWORD (or MBENCH_PASSWORD_HASH) for an HTTP Basic login on
everything except /health; MBENCH_AUTOSTART=0 to not start a run at startup; MBENCH_EPOCH_SCALE (default 1) scales
every architecture's epoch budget (e.g. 0.1 for a quick demo; the results then say so); the rest in
apps/_shared/README.md.
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

from pinneapple_arena import showcase

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

VERSION = "1.0.0"
STATIC = Path(__file__).resolve().parent / "static"
EPOCH_SCALE = float(os.environ.get("MBENCH_EPOCH_SCALE") or 1.0)
ARCHITECTURES = [{**a, "epochs": max(1, int(round(a["epochs"] * EPOCH_SCALE)))}
                 for a in showcase.DEFAULT_ARCHITECTURES]

_state: dict[str, Any] = {"sweep": {"status": "idle", "progress": 0, "total": len(ARCHITECTURES), "result": None,
                                    "started_at": None, "finished_at": None, "error": None,
                                    "epoch_scale": EPOCH_SCALE}}
_lock = threading.Lock()


def _run() -> None:
    sweep = _state["sweep"]

    def progress(i: int, n: int) -> None:
        sweep.update(progress=i, total=n)

    try:
        sweep["result"] = showcase.run_benchmark(architectures=ARCHITECTURES, progress_cb=progress)
        sweep["status"] = "done"
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
    if (os.environ.get("MBENCH_AUTOSTART") or "1") != "0":
        start_if_idle()
    yield


app = FastAPI(title="Model Benchmarking", version=VERSION, lifespan=lifespan,
              description="Four PINN architectures on viscous Burgers, trained live: relative L2, front-band error, "
                          "L-infinity, PDE residual, time and size, ranked by accuracy with Arena's physics-aware flag.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="MBENCH")


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/sweep_status")
def sweep_status():
    return {"sweep": _state["sweep"], "architectures": showcase.ARCHITECTURE_ORDER,
            "budgets": [{"name": a["display"], "type": a["type"], "epochs": a["epochs"]} for a in ARCHITECTURES]}


@app.post("/api/run_sweep")
def run_sweep():
    if not start_if_idle():
        raise HTTPException(409, "a benchmark run is already in progress")
    return {"started": True, "sweep": _state["sweep"]}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
