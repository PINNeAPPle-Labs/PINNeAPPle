"""Small, physically grounded reference experiment for the trusted workflow runner.

1-D steady conduction through a slab, constant k, prescribed end temperatures.
Finite-volume cell-centered heat flux; exact linear analytical reference.
This is a demonstrator, not a replacement for industrial FEA/CFD.
"""
from __future__ import annotations

from math import isfinite


def slab_conduction(params: dict, ctx) -> dict:
    settings = params.get("constraints") or {}
    length = float(settings.get("length_m", 1.0))
    conductivity = float(settings.get("conductivity_W_mK", 10.0))
    left = float(settings.get("left_temperature_K", 400.0))
    right = float(settings.get("right_temperature_K", 300.0))
    cells = int(settings.get("cells", 20))
    if not all(map(isfinite, (length, conductivity, left, right))):
        raise ValueError("all physical parameters must be finite")
    if length <= 0 or conductivity <= 0 or cells < 2 or cells > 100000:
        raise ValueError("invalid slab dimensions, conductivity or cell count")
    dx = length / cells
    temperatures = [left + (right - left) * (i + 0.5) / cells for i in range(cells)]
    heat_flux = -conductivity * (right - left) / length
    return {
        "solver": "analytic_1d_steady_conduction",
        "units": {"temperature": "K", "heat_flux": "W/m^2", "length": "m"},
        "coordinates_m": [(i + 0.5) * dx for i in range(cells)],
        "temperature_K": temperatures,
        "heat_flux_W_m2": heat_flux,
        "boundary_temperatures_K": [left, right],
        "length_m": length,
        "conductivity_W_mK": conductivity,
    }


def verify_slab(params: dict, ctx) -> dict:
    upstream = ctx.upstream.get("simulate", {}).get("outputs") or {}
    if upstream.get("solver") != "analytic_1d_steady_conduction":
        return {"approved": False, "reason": "missing or unexpected reference solver"}
    t = upstream["temperature_K"]
    length = upstream["length_m"]
    left, right = upstream["boundary_temperatures_K"]
    k = upstream["conductivity_W_mK"]
    n = len(t)
    residual = max(abs(t[i] - (left + (right-left)*(i+0.5)/n)) for i in range(n))
    expected_flux = -k*(right-left)/length
    flux_error = abs(upstream["heat_flux_W_m2"]-expected_flux)
    approved = residual <= 1e-9 and flux_error <= 1e-9
    return {"approved": approved, "temperature_residual_K": residual,
            "flux_error_W_m2": flux_error, "reference": "steady 1D Fourier conduction"}


def reference_handlers():
    """Return trusted callbacks for an end-to-end local smoke test only."""
    def passthrough(stage):
        return lambda params, ctx: {"stage": stage, "reference_only": True}
    handlers = {stage: passthrough(stage) for stage in (
        "scope", "design", "data", "model", "optimize", "report")}
    handlers["simulate"] = slab_conduction
    handlers["verify"] = verify_slab
    return handlers
