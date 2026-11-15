"""Darcy-Weisbach straight circular pipe reference, not a CFD solver.

Fully developed steady incompressible Newtonian flow in a hydraulically smooth
circular pipe. Laminar: f_D=64/Re; turbulent smooth-pipe *Blasius*
f_D=0.3164 Re^-1/4, used only over approximately 4e3 <= Re <= 1e5.
Reject transitional Reynolds and unsupported turbulent ranges instead of
assigning them false certainty. No entry/exit/minor losses or roughness model.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PipeResult:
    reynolds: float
    regime: str
    darcy_friction_factor: float
    pressure_drop_pa: float
    valid: bool
    note: str


def straight_pipe_pressure_drop(velocity_m_s: float, diameter_m: float,
                                length_m: float = 1.0, density_kg_m3: float = 998.0,
                                viscosity_pa_s: float = 0.001) -> PipeResult:
    inputs = (velocity_m_s, diameter_m, length_m, density_kg_m3, viscosity_pa_s)
    if not all(math.isfinite(float(x)) for x in inputs):
        raise ValueError("all inputs must be finite")
    if velocity_m_s < 0 or diameter_m <= 0 or length_m <= 0 or density_kg_m3 <= 0 or viscosity_pa_s <= 0:
        raise ValueError("velocity must be nonnegative; geometry and fluid properties must be positive")
    reynolds = density_kg_m3 * velocity_m_s * diameter_m / viscosity_pa_s
    if reynolds == 0:
        return PipeResult(0.0, "stagnant", 0.0, 0.0, True, "zero-flow reference")
    if reynolds < 2300:
        regime = "laminar"
        f = 64.0 / reynolds
    elif reynolds < 4000:
        return PipeResult(reynolds, "transitional", math.nan, math.nan, False,
                          "2300 <= Re < 4000; no validated correlation selected")
    elif reynolds <= 100_000:
        regime = "turbulent_smooth_blasius"
        f = 0.3164 * reynolds ** (-0.25)
    else:
        return PipeResult(reynolds, "out_of_range", math.nan, math.nan, False,
                          "Blasius smooth-pipe approximation not asserted beyond Re=100000")
    dp = f * (length_m / diameter_m) * (density_kg_m3 * velocity_m_s ** 2 / 2.0)
    return PipeResult(reynolds, regime, f, dp, True,
                      "fully developed steady flow; major losses only; smooth pipe")


def generate_pipe_reference(velocities_m_s, diameters_m, **kwargs) -> list[dict]:
    """Produce explicit validity labels; do not silently train on invalid rows."""
    if len(velocities_m_s) != len(diameters_m):
        raise ValueError("velocities and diameters must have equal lengths")
    return [
        {"velocity_m_s": v, "diameter_m": d, **vars(straight_pipe_pressure_drop(v, d, **kwargs))}
        for v, d in zip(velocities_m_s, diameters_m)
    ]
