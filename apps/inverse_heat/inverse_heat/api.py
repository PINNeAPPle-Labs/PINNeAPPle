"""Inverse Heat Lab: web API + single-page UI. The convection coefficient h from a few thermocouples, in 1D, 2D and 3D.

Run:  uvicorn inverse_heat.api:app --port 8086 --workers 1   (from apps/inverse_heat; jobs live in the worker's memory)

Environment (all optional):
  IHL_USER / IHL_PASSWORD   HTTP Basic login on everything except /health
  IHL_MAX_JOBS              trainings running at the same time (default 2); more get HTTP 429
  IHL_THREADS               torch threads per training (default 1)
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import engine
from .snippets import SNIPPETS

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

VERSION = "1.0.0"
STATIC = os.path.join(os.path.dirname(__file__), "static")
MAX_JOBS = int(os.environ.get("IHL_MAX_JOBS", "2"))
THREADS = int(os.environ.get("IHL_THREADS", "1"))

app = FastAPI(title="Inverse Heat Lab", version=VERSION,
              description="Find the convection coefficient h from a few thermocouples with a physics-informed neural "
                          "network (PINNeAPPle): a pin fin (1D, also from your own readings), a heat-spreader plate "
                          "(2D) and a chip under a block (3D).")
install(app, prefix="IHL")

_JOBS: Dict[str, Dict] = {}
_LOCK = threading.Lock()

VALIDATION = [
    "1D fin, 10 noise draws: h = 24.8 ± 0.4 W/m²K for a true 25, as accurate as a least-squares fit of the analytic "
    "profile to the same readings (24.8 ± 0.3); heat dissipated 0.578 W vs 0.579 W analytic",
    "2D plate, 5 noise draws: h = 14.8 ± 0.3 for a true 15; hot spot 79.2 °C vs 79.1 °C from an independent "
    "finite-volume solution (grid-converged to 0.01 K)",
    "3D block, 3 noise draws: chip temperature 74.6 °C vs 74.5 °C, never measured; h from the energy balance on the "
    "learned field 149.5 for a true 150 (the network's own h parameter settles 7 % low, reported as is)",
    "Live runs on this page: the fin also reports the least-squares fit of the analytic solution to the same readings, "
    "and the plate is compared to the finite-volume solution cell by cell",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Scope & validation", "noun": "limitation",
            "items_title": "What the models assume",
            "band": "Every case is checked against an analytic or an independent numerical solution.",
            "tags": {"conservative": "Safe by default", "optimistic": "Review before use", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Steady state, uniform h", "effect": "check",
                 "detail": "The fin and the plate assume steady conduction and one h over the whole convective surface.",
                 "today": "Take readings at steady state; if h varies (a fan on one side), the result is an average.",
                 "planned": "Spatially varying h and transient readings."},
                {"topic": "1D fin: thin fin", "effect": "check",
                 "detail": "Temperature is taken uniform across the fin's cross-section (Biot number hD/2k much below 1); "
                           "the page shows the Biot number of your case.",
                 "today": "Below 0.1 the 1D model holds; above it, use a 2D/3D model.",
                 "planned": ""},
                {"topic": "No radiation, no contact resistance", "effect": "optimistic",
                 "detail": "Radiation adds to convection, so h here is an effective coefficient including it; a "
                           "thermocouple with poor contact reads low.",
                 "today": "Read the result as an effective h; check the sensor attachment.",
                 "planned": "A radiation term with the surface emissivity."},
                {"topic": "3D runs are not live", "effect": "conservative",
                 "detail": "Training the 3D block takes about 30 minutes on a CPU, so the page shows the full result "
                           "computed offline; the code to rerun it is on the Code tab.",
                 "today": "Run block_3d.py on your machine (a GPU makes it minutes).",
                 "planned": ""},
            ]}


class FinRequest(BaseModel):
    k: float = Field(16.0, gt=0, le=500, description="conductivity, W/m.K")
    d_mm: float = Field(5.0, gt=0, le=100)
    length_mm: float = Field(50.0, gt=0, le=1000)
    t_base: float = Field(80.0, ge=-50, le=1000)
    t_air: float = Field(25.0, ge=-50, le=500)
    sensors_mm: List[float] = Field(default_factory=lambda: [10, 20, 30, 40, 50])
    readings: Optional[List[float]] = Field(None, description="your readings (°C); omit to use synthetic ones")
    h_true: float = Field(25.0, gt=0, le=5000, description="synthetic mode only")
    noise: float = Field(0.5, ge=0, le=5, description="synthetic mode only, °C")
    h_guess: float = Field(100.0, gt=0, le=5000)


class PlateRequest(BaseModel):
    power: float = Field(8.0, ge=0.5, le=50)
    h_true: float = Field(15.0, ge=2, le=200)
    noise: float = Field(0.5, ge=0, le=3)
    h_guess: float = Field(60.0, ge=1, le=500)


def _start(kind: str, fn, **kw) -> Dict:
    with _LOCK:
        running = sum(j["status"] == "running" for j in _JOBS.values())
        if running >= MAX_JOBS:
            raise HTTPException(429, f"{running} trainings are running; try again in a minute")
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "kind": kind, "status": "running", "step": 0, "total": 1, "h": None, "history": [],
               "started": time.time(), "result": None, "error": None}
        _JOBS[jid] = job
        for old in [k for k, j in _JOBS.items() if time.time() - j["started"] > 3600]:
            _JOBS.pop(old, None)

    def progress(step, total, h):
        job.update(step=step, total=total, h=h)
        job["history"].append([step, h])

    def work():
        import torch
        torch.set_num_threads(THREADS)
        try:
            job["result"] = fn(progress=progress, **kw)
            job["status"] = "done"
        except engine.InputError as exc:
            job.update(status="error", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - reported to the page
            job.update(status="error", error=f"{type(exc).__name__}: {exc}")
        job["seconds"] = round(time.time() - job["started"], 1)

    threading.Thread(target=work, daemon=True).start()
    return {"id": jid}


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "scope": scope(), "max_jobs": MAX_JOBS,
            "plate_sensors_mm": engine.PLATE_SENSORS_MM, "snippets": SNIPPETS}


@app.post("/api/fin/run")
def fin_run(req: FinRequest):
    """Start a 1D fin training. With ``readings`` it uses yours; without, it makes noisy ones from ``h_true``."""
    kw = req.model_dump()
    synthetic = kw.pop("readings") is None
    h_true, noise = kw.pop("h_true"), kw.pop("noise")
    if synthetic:
        readings = engine.fin_synthetic_readings(k=req.k, d_mm=req.d_mm, length_mm=req.length_mm, t_base=req.t_base,
                                                 t_air=req.t_air, sensors_mm=req.sensors_mm, h_true=h_true,
                                                 noise=noise)
    else:
        readings = req.readings
    try:  # validate before starting the thread
        engine.run_fin(**kw, readings=readings, steps=0)
    except engine.InputError as exc:
        raise HTTPException(422, str(exc))
    return _start("fin", engine.run_fin, **kw, readings=readings, h_true=h_true if synthetic else None)


@app.post("/api/plate/run")
def plate_run(req: PlateRequest):
    return _start("plate", engine.run_plate, **req.model_dump())


@app.get("/api/jobs/{jid}")
def job(jid: str, history_from: int = 0):
    j = _JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "unknown job (results are kept for an hour)")
    return {k: v for k, v in j.items() if k != "history"} | {"history": j["history"][history_from:]}


@app.get("/api/plate/precomputed")
def plate_pre():
    return engine.plate_precomputed()


@app.get("/api/block/precomputed")
def block_pre():
    return engine.block_precomputed()


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/figures", StaticFiles(directory=os.path.join(engine.EXAMPLE, "results")), name="figures")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
