"""Analytical two-material Lame interference-fit reference, SI units.

Standalone kernel for issue #105. Not yet wired into PrePoMax/FEM.
"""
from __future__ import annotations

import math


def interference_fit_pressure(
    *, interference_m: float, interface_radius_m: float,
    ring_outer_radius_m: float, shaft_inner_radius_m: float,
    ring_young_pa: float, ring_poisson: float,
    shaft_young_pa: float, shaft_poisson: float,
    plane_strain: bool = True,
) -> float:
    """Pressure in Pa. Interference is radial, not diametral.

    Linear isotropic elasticity; no friction, plasticity, thermal or
    finite-length end effects.
    """
    vals = (interference_m, interface_radius_m, ring_outer_radius_m,
            shaft_inner_radius_m, ring_young_pa, ring_poisson,
            shaft_young_pa, shaft_poisson)
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("all inputs must be finite")
    a, c, b = interface_radius_m, ring_outer_radius_m, shaft_inner_radius_m
    if not (interference_m >= 0 and 0 <= b < a < c):
        raise ValueError("require radial interference >= 0 and 0 <= b < a < c")
    if ring_young_pa <= 0 or shaft_young_pa <= 0:
        raise ValueError("Young moduli must be positive")
    if not (-1 < ring_poisson < 0.5 and -1 < shaft_poisson < 0.5):
        raise ValueError("Poisson ratios must be in (-1, 0.5)")

    if plane_strain:
        ring = ((1 + ring_poisson) / ring_young_pa
                * (c*c + (1-2*ring_poisson)*a*a) / (c*c-a*a))
        shaft = ((1 + shaft_poisson) / shaft_young_pa
                 * ((1-2*shaft_poisson)*a*a + b*b) / (a*a-b*b))
    else:
        ring = ((c*c+a*a)/(c*c-a*a) + ring_poisson)/ring_young_pa
        shaft = ((a*a+b*b)/(a*a-b*b) - shaft_poisson)/shaft_young_pa
    compliance = a * (ring + shaft)
    if not math.isfinite(compliance) or compliance <= 0:
        raise ValueError("invalid elastic compliance")
    return interference_m / compliance
