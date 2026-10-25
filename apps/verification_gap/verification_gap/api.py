"""Verification Gap: a vented brake rotor that passes its 500 C limit at nominal dimensions, and what a Monte Carlo
propagation of its machining tolerances shows underneath that pass.

Glue only: the rotor model and mesh are ``pinneapple_design.surrogate_inverse.brake_rotor``, the study (sampling, corner stack-up,
six checks, TrustReport decision) is ``pinneapple_analysis.uncertainty.margin.margin_study``.

Run:  uvicorn verification_gap.api:app --port 8180   (from apps/verification_gap)

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
from pinneapple_design.surrogate_inverse import brake_rotor as br

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
STATIC = Path(__file__).resolve().parent / "static"
ROTOR = br.BASELINE_DESIGN       # 36 cast-iron vanes, 8 mm gap, 4 mm thick (0.754 m mean circumference)
POWER_W = 1480.0               # sustained braking (a long descent)
H_CITY = 40.0                  # W/(m2 K), low-speed airflow through the vents: the worst case
H_HIGHWAY = 120.0              # same rotor at highway speed, for comparison
LIMIT_C = 500.0                # thermal limit of this worked case
TOL_THICKNESS_M = br.VANE_THICKNESS_TOLERANCE_M   # +-0.05 mm (assumed band, not supplier data)
TOL_GAP_M = br.VANE_GAP_TOLERANCE_M               # +-0.1 mm
N_SAMPLES = 200

app = FastAPI(title="Verification Gap", version=VERSION,
              description="A design that passes at nominal dimensions, checked under its manufacturing tolerances: "
                          "Monte Carlo margin study, corner stack-up and six checks with a TrustReport decision.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="VGAP")
_HEAVY = BusyLimiter("VGAP_MAX_HEAVY", 4)


def _temperature(X: np.ndarray, h_conv: float = H_CITY) -> np.ndarray:
    return np.array([br.rotor_temperature_c(br.RotorDesign(ROTOR.vane_count, max(g, 1e-6), max(t, 1e-6)),
                                            POWER_W, h_conv) for t, g in np.atleast_2d(X)])


def _profile(h_conv: float) -> dict:
    r_mean_mm = br.MEAN_CIRCUMFERENCE_M * 1000 / (2 * np.pi)
    return {**br.thermal_field(ROTOR, POWER_W, h_conv), "radial_flow_length_m": br.RADIAL_FLOW_LENGTH_M,
            "r_outer_mm": r_mean_mm + br.RADIAL_FLOW_LENGTH_M * 500,
            "r_inner_mm": r_mean_mm - br.RADIAL_FLOW_LENGTH_M * 500}


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
    city = br.rotor_temperature_c(ROTOR, POWER_W, H_CITY)
    highway = br.rotor_temperature_c(ROTOR, POWER_W, H_HIGHWAY)
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
                     "thermal_field": _profile(H_CITY)},
            "highway": {"h_conv": H_HIGHWAY, "label": "Highway-speed ram air (same rotor)", "temp_c": highway,
                        "thermal_field": _profile(H_HIGHWAY)}},
    }


@lru_cache(maxsize=1)
def _mesh() -> dict:
    V, F = br.mesh(ROTOR)
    return {"positions": np.round(V, 4).ravel().tolist(), "indices": F.ravel().tolist()}


@app.get("/api/mesh.json")
def mesh():
    """Triangle mesh of the rotor (mm): two friction rings and the vanes between them."""
    return _mesh()


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
