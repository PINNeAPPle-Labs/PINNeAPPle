"""Verification Gap: a vented brake rotor that passes its 500 C limit at nominal dimensions, and what a Monte Carlo
propagation of its machining tolerances shows underneath that pass.

Glue only: the rotor model is ``pinneapple_physics.closed_form.vented_rotor``, the study (sampling, corner stack-up,
six checks, TrustReport decision) is ``pinneapple_analysis.uncertainty.margin.margin_study`` and the mesh is
``pinneapple_design.thermal.meshes.vented_rotor_mesh``.

Run:  uvicorn verification_gap.api:app --port 8160   (from apps/verification_gap)

Environment (all optional): VGAP_USER / VGAP_PASSWORD (or VGAP_PASSWORD_HASH) for an HTTP Basic login on everything
except /health; VGAP_MAX_HEAVY concurrent studies per worker (default 4); the rest in apps/_shared/README.md.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pinneapple_analysis.uncertainty.margin import margin_study
from pinneapple_design.thermal.meshes import mesh_payload, vented_rotor_mesh
from pinneapple_physics.closed_form.vented_rotor import (
    VentedRotor,
    radial_profile,
    rotor_temperature_c,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
STATIC = Path(__file__).resolve().parent / "static"
ROTOR = VentedRotor()
POWER_W = 1480.0               # sustained braking (a long descent)
H_CITY = 40.0                  # W/(m2 K), low-speed airflow through the vents: the worst case
H_HIGHWAY = 120.0              # same rotor at highway speed, for comparison
LIMIT_C = 500.0                # thermal limit of this worked case
TOL_THICKNESS_M = 0.05e-3      # +- machining tolerance of the vane thickness
TOL_GAP_M = 0.1e-3             # +- of the vent gap
N_SAMPLES = 200

app = FastAPI(title="Verification Gap", version=VERSION,
              description="A design that passes at nominal dimensions, checked under its manufacturing tolerances: "
                          "Monte Carlo margin study, corner stack-up and six checks with a TrustReport decision.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="VGAP")
_HEAVY = BusyLimiter("VGAP_MAX_HEAVY", 4)


def _temperature(X: np.ndarray, h_conv: float = H_CITY) -> np.ndarray:
    return np.array([rotor_temperature_c(ROTOR.with_(vane_thickness_m=max(t, 1e-6), vane_gap_m=max(g, 1e-6)),
                                         POWER_W, h_conv) for t, g in np.atleast_2d(X)])


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/scenario")
def scenario(seed: int | None = Query(None, ge=0)):
    """A fresh 200-sample study (``seed`` for a reproducible one) in the shape the page reads."""
    with _HEAVY:
        st = margin_study(_temperature, [ROTOR.vane_thickness_m, ROTOR.vane_gap_m], [TOL_THICKNESS_M, TOL_GAP_M],
                          LIMIT_C, n=N_SAMPLES, seed=seed, check_seed=None if seed is None else seed + 1, unit="°C")
    s = st["summary"]
    city, highway = rotor_temperature_c(ROTOR, POWER_W, H_CITY), rotor_temperature_c(ROTOR, POWER_W, H_HIGHWAY)
    return {
        "nominal_c": st["nominal_value"], "safety_limit_c": LIMIT_C, "samples_c": st["samples"],
        "summary": {"n_samples": s["n_samples"], "mean_c": s["mean"], "std_c": s["std"], "min_c": s["min"],
                    "max_c": s["max"], "p95_c": s["p95"], "safe_fraction": s["safe_fraction"],
                    "safe_fraction_ci": s["safe_fraction_ci"], "verdict": s["verdict"],
                    "margin_at_p95_c": s["margin_at_p95"]},
        "verification": {"checks": st["checks"], "overall_verdict": st["overall_verdict"], "decision": st["decision"],
                         "trust": st["trust"], "coverage": st["trust"]["cobertura"], "coverage_note": st["scope"],
                         "corner": st["corner"]},
        "scenario": {"power_w": POWER_W, "condition": "sustained high-power braking, low-speed cooling airflow",
                     "vane_count": ROTOR.vane_count, "vane_gap_mm": ROTOR.vane_gap_m * 1000,
                     "vane_thickness_mm": ROTOR.vane_thickness_m * 1000,
                     "thickness_tolerance_mm": TOL_THICKNESS_M * 1000, "gap_tolerance_mm": TOL_GAP_M * 1000,
                     "seed": seed},
        "comparison": {
            "city": {"h_conv": H_CITY, "label": "Low-speed / stop-and-go (this scenario)", "temp_c": city,
                     "thermal_field": radial_profile(ROTOR, POWER_W, H_CITY)},
            "highway": {"h_conv": H_HIGHWAY, "label": "Highway-speed ram air (same rotor)", "temp_c": highway,
                        "thermal_field": radial_profile(ROTOR, POWER_W, H_HIGHWAY)}},
    }


@lru_cache(maxsize=1)
def _mesh() -> dict:
    V, F = vented_rotor_mesh(ROTOR.r_mean_m * 1000, ROTOR.radial_flow_length_m * 1000, ROTOR.ring_thickness_m * 1000,
                             ROTOR.vane_gap_m * 1000, int(ROTOR.vane_count), ROTOR.vane_thickness_m * 1000)
    return mesh_payload(V, F)


@app.get("/api/mesh.json")
def mesh():
    """Triangle mesh of the rotor (mm): two friction rings and the vanes between them."""
    return _mesh()


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
