"""SatScreen: quick conjunction screening (closest approach and a 2-D Gaussian collision probability in
Clohessy-Wiltshire relative motion) and two-body + J2 propagation, from ``pinneapple_physics.astro.conjunction``.

No PINN is trained per request: CW has an exact closed form and J2 is integrated with DOP853. The presets
``space_debris_cw_relative_motion`` and ``satellite_j2_perturbation`` build and check the inputs and carry the
references. ``mode="description"`` lets the local LLM draft the inputs (``pinneapple_llm.draft_preset_kwargs``,
range-checked and retried); ``mode="elements"`` is the reliable path.

Scope: CW is linear relative motion about a circular reference orbit over short windows (no drag, J2, third bodies,
manoeuvres); the deputy is a relative state in the chief's Hill frame; sigma is a caller-supplied isotropic
placeholder, not a propagated covariance, so the Pc illustrates the method and is not an operational Pc.

Run:  uvicorn satscreen.api:app --port 8144   (from apps/satscreen)

Environment (all optional; the rest in apps/_shared/README.md):
  SATSCREEN_USER / SATSCREEN_PASSWORD(_HASH)   login on everything except /health
  SATSCREEN_LLM_PROVIDER / SATSCREEN_LLM_MODEL  description mode: ollama (default) + llama3.2:3b
  SATSCREEN_FREE_SCREENINGS                     free screenings and propagations per caller and month (default 500)
"""
from __future__ import annotations

import os
import sys
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import pinneapple_llm as pl
import pinneapple_physics as pp
from pinneapple_physics.astro import conjunction as cj

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install, install_runtime  # noqa: E402

VERSION = "1.0.0"
PROVIDER = os.environ.get("SATSCREEN_LLM_PROVIDER", "ollama")
MODEL = os.environ.get("SATSCREEN_LLM_MODEL") or ("llama3.2:3b" if PROVIDER == "ollama" else None)
# ranges an LLM-drafted value must fall in (description mode)
CW_BOUNDS = {"n": (1e-5, 2e-2), "x0": (-2000.0, 2000.0), "y0": (-5000.0, 5000.0), "z0": (-2000.0, 2000.0),
             "vx0": (-1.0, 1.0), "vy0": (-1.0, 1.0), "vz0": (-1.0, 1.0)}
J2_BOUNDS = {"mu": (3.9e5, 4.0e5), "J2": (1e-4, 1e-2), "Re": (6000.0, 6800.0), "a": (6500.0, 45000.0),
             "e": (0.0, 0.9), "inclination_deg": (0.0, 180.0)}

app = FastAPI(title="SatScreen", version=VERSION,
              description="Conjunction screening in Clohessy-Wiltshire relative motion, and J2 propagation.")
install(app, prefix="SATSCREEN")
rt = install_runtime(app, prefix="SATSCREEN", name="satscreen", units={"screenings": 500})


class Chief(BaseModel):
    a_km: float = Field(6771.0, gt=6378.0)
    mu_km3_s2: float = 398600.4418


class RelativeState(BaseModel):
    x0_km: float
    y0_km: float = 0.0
    z0_km: float = 0.0
    vx0_km_s: float = 0.0
    vy0_km_s: float = 0.0
    vz0_km_s: float = 0.0


class ScreenRequest(BaseModel):
    mode: Literal["elements", "description"] = "elements"
    description: str | None = Field(None, max_length=4000)
    chief: Chief | None = None
    deputy_relative_state: RelativeState | None = None
    duration_s: float | None = Field(None, gt=0)
    n_samples: int = Field(2000, ge=100, le=200_000)
    combined_hard_body_radius_km: float = Field(0.010, ge=0)
    combined_position_sigma_km: float = Field(1.0, gt=0)
    max_attempts: int = Field(4, ge=1, le=8)


class Satellite(BaseModel):
    a_km: float = Field(7000.0, gt=6378.0)
    e: float = Field(0.001, ge=0, lt=1)
    inclination_deg: float = Field(98.7, ge=0, le=180)
    mu_km3_s2: float = 398600.4418
    J2: float = 1.08262668e-3
    Re_km: float = 6378.137


class PropagateRequest(BaseModel):
    mode: Literal["elements", "description"] = "elements"
    description: str | None = Field(None, max_length=4000)
    satellite: Satellite | None = None
    duration_orbits: float = Field(1.0, gt=0, le=200)
    n_points: int = Field(50, ge=2, le=5000)
    max_attempts: int = Field(4, ge=1, le=8)


def _draft(description: str | None, preset: str, bounds: dict, attempts: int):
    if not description:
        raise HTTPException(422, "mode='description' needs a description")
    try:
        return pl.draft_preset_kwargs(description, preset, bounds=bounds, provider=PROVIDER, model=MODEL,
                                      max_attempts=attempts)
    except pl.DraftRejected as e:
        raise HTTPException(502, str(e)) from None
    except ConnectionError as e:
        raise HTTPException(503, f"the LLM is not reachable ({PROVIDER}): {e}") from None


def _ic(spec) -> dict[str, float]:
    return {c.fields[0]: float(np.asarray(c.value_fn(np.zeros((1, 1)), {})).ravel()[0]) for c in spec.conditions}


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.post("/v1/screen_conjunction")
def screen_conjunction(request: Request, req: ScreenRequest):
    rt.consume(request, "screenings")
    reasoning = None
    if req.mode == "elements":
        if req.chief is None or req.deputy_relative_state is None:
            raise HTTPException(422, "mode='elements' needs 'chief' and 'deputy_relative_state'")
        d = req.deputy_relative_state
        kwargs = dict(n=cj.cw_mean_motion(req.chief.a_km, req.chief.mu_km3_s2), x0=d.x0_km, y0=d.y0_km, z0=d.z0_km,
                      vx0=d.vx0_km_s, vy0=d.vy0_km_s, vz0=d.vz0_km_s)
    else:
        kwargs, reasoning, _ = _draft(req.description, "space_debris_cw_relative_motion", CW_BOUNDS, req.max_attempts)
    spec = pp.get_preset("space_debris_cw_relative_motion", **kwargs)
    ic = _ic(spec)
    n = float(spec.pde.params["n"])
    out = cj.screen_cw_conjunction(n, [ic[k] for k in ("x", "y", "z", "vx", "vy", "vz")], duration_s=req.duration_s,
                                   n_samples=req.n_samples, hard_body_radius_km=req.combined_hard_body_radius_km,
                                   sigma_km=req.combined_position_sigma_km)
    out.update({"preset": "space_debris_cw_relative_motion", "preset_kwargs": kwargs, "llm_reasoning": reasoning,
                "references": list(spec.references),
                "limits": "linear CW about a circular orbit, short windows; isotropic placeholder sigma, "
                          "not a propagated covariance"})
    return out


@app.post("/v1/propagate_j2")
def propagate_j2(request: Request, req: PropagateRequest):
    rt.consume(request, "screenings")
    reasoning = None
    if req.mode == "elements":
        if req.satellite is None:
            raise HTTPException(422, "mode='elements' needs 'satellite'")
        s = req.satellite
        kwargs = dict(mu=s.mu_km3_s2, J2=s.J2, Re=s.Re_km, a=s.a_km, e=s.e, inclination_deg=s.inclination_deg)
    else:
        kwargs, reasoning, _ = _draft(req.description, "satellite_j2_perturbation", J2_BOUNDS, req.max_attempts)
    spec = pp.get_preset("satellite_j2_perturbation", **kwargs)
    ic = _ic(spec)
    prm = spec.pde.params
    period = float(spec.pde.meta["orbital_period_s"])
    t = np.linspace(0.0, req.duration_orbits * period, req.n_points)
    traj = cj.propagate_two_body_j2([ic[k] for k in ("x", "y", "z", "vx", "vy", "vz")], t,
                                    mu=float(prm["mu"]), J2=float(prm["J2"]), Re=float(prm["Re"]))
    keys = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")
    return {"preset": "satellite_j2_perturbation", "preset_kwargs": kwargs, "llm_reasoning": reasoning,
            "orbital_period_s": period, "window_s": float(t[-1]),
            "trajectory": [{"t_s": float(ti), **{k: float(v) for k, v in zip(keys, row, strict=True)}}
                           for ti, row in zip(t, traj, strict=True)],
            "secular_rates_literature": {
                "nodal_regression_rad_s": spec.pde.meta["nodal_regression_rate_rad_s_literature"],
                "apsidal_precession_rad_s": spec.pde.meta["apsidal_precession_rate_rad_s_literature"],
                "source": "Vallado (2013), Fundamentals of Astrodynamics and Applications, 4th ed., ch. 9",
                "checked_in": "tests/test_j2_secular_validation.py (integrated rates within 1 %)"},
            "limits": "two-body + J2 only: no drag, J3+, Sun/Moon", "references": list(spec.references),
            "radius_range_km": [float(np.linalg.norm(traj[:, :3], axis=1).min()),
                                float(np.linalg.norm(traj[:, :3], axis=1).max())]}
