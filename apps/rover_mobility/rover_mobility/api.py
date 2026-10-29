"""RoverMobility: rigid-wheel mobility screening with the ``bekker_wong_surrogate_2d`` PINN.

``POST /v1/analyze_mobility`` (job) runs ``pinneapple_simulation.terramechanics.mobility.mobility_analysis``: train
the surrogate for a soil (cohesion, friction angle) and wheel (radius, width), find the sinkage that carries the
wheel load, read drawbar pull against slip, and check each constraint (R1 zero pull at zero slip, R2 Mohr-Coulomb
limit, R3 monotone up to slip 0.4, R4 torque covers thrust) on its own next to the guardrail. A pooled guardrail
residual of zero can hide a collapsed surrogate (pull ~ 0 everywhere): read ``max_drawbar_pull_N`` and the checks.

``GET /v1/soil_parameter_guide`` lists the library's soils with their sources
(``pinneapple_simulation.terramechanics.soils``). For full Wong-Reece wheel physics and rover missions use the
library (``WongReeceModel``, ``pinneapple_simulation.rover.autolab``); this app is the quick surrogate screen.

Run:  uvicorn rover_mobility.api:app --port 8147   (from apps/rover_mobility)

Environment (all optional; the rest in apps/_shared/README.md):
  ROVERMOB_USER / ROVERMOB_PASSWORD(_HASH)   login on everything except /health
  ROVERMOB_FREE_ANALYSES                     free analyses per caller and month (default 20)
  ROVERMOB_MAX_EPOCHS                        largest training budget per job (default 5000)
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
MAX_EPOCHS = int(os.environ.get("ROVERMOB_MAX_EPOCHS") or 5000)

app = FastAPI(title="RoverMobility", version=VERSION,
              description="Wheel-soil drawbar pull against slip from a Bekker-Wong PINN, each constraint re-checked.")
install(app, prefix="ROVERMOB")
rt = install_runtime(app, prefix="ROVERMOB", name="rover_mobility", units={"analyses": 20}, max_jobs=1)


class MobilityRequest(BaseModel):
    c_Pa: float = Field(1400.0, ge=0)
    phi_deg: float = Field(30.0, gt=0, lt=90)
    R_m: float = Field(0.125, gt=0)
    b_m: float = Field(0.060, gt=0)
    Fz_N: float = Field(50.0, gt=0)
    slip_min: float = Field(0.0, ge=0)
    slip_max: float = Field(0.7, gt=0, le=1)
    n_slip: int = Field(15, ge=2, le=200)
    epochs: int = Field(2000, ge=10)
    n_collocation: int = Field(1024, ge=64, le=8192)
    architecture: str = "modified_mlp"
    residual_threshold: float = Field(1e-2, gt=0)
    seed: int = Field(0, ge=0)


def analyze_job(p: dict, ctx) -> dict:
    from pinneapple_core.jsonable import to_jsonable
    from pinneapple_simulation.terramechanics.mobility import mobility_analysis

    ctx.progress(0.05, "training the surrogate")
    return to_jsonable(mobility_analysis(**p))


rt.jobs.register("analyze", analyze_job)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/v1/soil_parameter_guide")
def soil_parameter_guide():
    from pinneapple_simulation.terramechanics import get_soil, list_soils

    soils = []
    for name in list_soils():
        s = get_soil(name)
        soils.append({"name": name, "c_Pa": s.c, "phi_deg": s.phi_deg, "description": s.description,
                      "sources": {k: s.sources.get(k, "") for k in ("c", "phi_deg")}})
    return {"soils": soils, "note": "c_Pa and phi_deg feed POST /v1/analyze_mobility; screening values, not a "
                                    "substitute for a shear test of your terrain"}


@app.post("/v1/analyze_mobility")
def analyze(request: Request, req: MobilityRequest):
    p = req.model_dump()
    p["epochs"] = min(p["epochs"], MAX_EPOCHS)
    return rt.submit(request, "analyze", p, units={"analyses": 1})


@app.get("/v1/jobs/{job_id}")
def job(request: Request, job_id: str):
    return rt.job_for(request, job_id)
