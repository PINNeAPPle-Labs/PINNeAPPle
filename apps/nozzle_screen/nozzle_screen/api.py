"""NozzleScreen: early rocket-nozzle screening by two methods that share no code, and their disagreement.

* ``GET /v1/analytical_only``: the ideal 1-D isentropic nozzle (``pinneapple_physics.closed_form.rocket_nozzle``):
  exit Mach, pressure, temperature, c*, C_F, Isp, in microseconds.
* ``POST /v1/screen_nozzle`` (job): the same plus a PINN on the ``rocket_nozzle_cfd`` preset, re-checked by the
  guardrail, and the relative difference at the exit
  (``pinneapple_analysis.verification.nozzle_crosscheck.screen_nozzle``). Expect them to disagree today: the
  preset's residual is in raw SI units (~1e22-1e24 here), so the PINN half is a demonstration of the cross-check
  catching a bad model, not a second estimate to use.

Run:  uvicorn nozzle_screen.api:app --port 8146   (from apps/nozzle_screen)

Environment (all optional; the rest in apps/_shared/README.md):
  NOZZLE_USER / NOZZLE_PASSWORD(_HASH)   login on everything except /health
  NOZZLE_FREE_ESTIMATES, NOZZLE_FREE_SCREENS   free analytic estimates (10000) and PINN screens (20) per month
  NOZZLE_MAX_EPOCHS                      largest PINN budget per job (default 3000)
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
R_UNIVERSAL = 8.314462618
MAX_EPOCHS = int(os.environ.get("NOZZLE_MAX_EPOCHS") or 3000)

app = FastAPI(title="NozzleScreen", version=VERSION,
              description="Ideal-nozzle estimate and a PINN cross-check with their measured disagreement.")
install(app, prefix="NOZZLE")
rt = install_runtime(app, prefix="NOZZLE", name="nozzle_screen", units={"estimates": 10000, "screens": 20},
                     max_jobs=1)


class Nozzle(BaseModel):
    throat_radius_m: float = Field(..., gt=0)
    exit_radius_m: float | None = Field(None, gt=0)
    area_ratio: float | None = Field(None, gt=1.0)
    nozzle_length_m: float | None = Field(None, gt=0)
    p_chamber_pa: float = Field(..., gt=0)
    T_chamber_k: float = Field(..., gt=0)
    gamma: float = Field(1.2, gt=1.0, lt=1.7)
    R_gas_j_per_kgk: float | None = Field(None, gt=0)
    molecular_weight_kg_per_mol: float | None = Field(None, gt=0)
    p_atm_pa: float = Field(101325.0, ge=0)

    @model_validator(mode="after")
    def _check(self):
        if (self.exit_radius_m is None) == (self.area_ratio is None):
            raise ValueError("give exactly one of exit_radius_m and area_ratio")
        if self.R_gas_j_per_kgk is None and self.molecular_weight_kg_per_mol is None:
            raise ValueError("give R_gas_j_per_kgk or molecular_weight_kg_per_mol")
        if self.exit_radius_m is not None and self.exit_radius_m <= self.throat_radius_m:
            raise ValueError("exit_radius_m must be larger than throat_radius_m")
        return self

    def resolved(self) -> dict:
        exit_r = self.exit_radius_m or self.throat_radius_m * self.area_ratio ** 0.5
        R = self.R_gas_j_per_kgk or R_UNIVERSAL / self.molecular_weight_kg_per_mol
        return {"throat_radius": self.throat_radius_m, "exit_radius": exit_r, "nozzle_length": self.nozzle_length_m,
                "p_chamber": self.p_chamber_pa, "T_chamber": self.T_chamber_k, "gamma": self.gamma, "R_gas": R,
                "p_atm": self.p_atm_pa}


class ScreenRequest(Nozzle):
    epochs: int = Field(300, ge=10)
    n_collocation: int = Field(1024, ge=64, le=8192)
    n_bc_per_face: int = Field(300, ge=20, le=4000)
    architecture: str = "modified_mlp"
    residual_threshold: float = Field(1e-2, gt=0)
    tolerance: float = Field(0.30, gt=0)
    seed: int = Field(0, ge=0)


def screen_job(p: dict, ctx) -> dict:
    from pinneapple_analysis.verification.nozzle_crosscheck import screen_nozzle
    from pinneapple_core.jsonable import to_jsonable

    ctx.progress(0.05, "training the PINN")
    return to_jsonable(screen_nozzle(**p))


rt.jobs.register("screen", screen_job)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/v1/analytical_only")
def analytical_only(request: Request, throat_radius_m: float, p_chamber_pa: float, T_chamber_k: float,
                    exit_radius_m: float | None = None, area_ratio: float | None = None, gamma: float = 1.2,
                    R_gas_j_per_kgk: float | None = None, molecular_weight_kg_per_mol: float | None = None,
                    p_atm_pa: float = 101325.0):
    from pinneapple_analysis.verification.nozzle_crosscheck import screen_nozzle

    try:
        n = Nozzle(throat_radius_m=throat_radius_m, exit_radius_m=exit_radius_m, area_ratio=area_ratio,
                   p_chamber_pa=p_chamber_pa, T_chamber_k=T_chamber_k, gamma=gamma, R_gas_j_per_kgk=R_gas_j_per_kgk,
                   molecular_weight_kg_per_mol=molecular_weight_kg_per_mol, p_atm_pa=p_atm_pa)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    rt.consume(request, "estimates")
    return screen_nozzle(**n.resolved(), run_pinn=False)


@app.post("/v1/screen_nozzle")
def screen(request: Request, req: ScreenRequest):
    p = req.resolved()
    p.update(epochs=min(req.epochs, MAX_EPOCHS), n_collocation=req.n_collocation, n_bc_per_face=req.n_bc_per_face,
             architecture=req.architecture, residual_threshold=req.residual_threshold, tolerance=req.tolerance,
             seed=req.seed)
    return rt.submit(request, "screen", p, units={"screens": 1})


@app.get("/v1/jobs/{job_id}")
def job(request: Request, job_id: str):
    return rt.job_for(request, job_id)
