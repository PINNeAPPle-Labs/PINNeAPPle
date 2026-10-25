"""Closed-form 2-D section aerodynamics: thin-airfoil lift of a NACA 4-digit camber line and a profile-drag
correlation for a streamlined section.

Lift (inviscid, small angles; Anderson, *Fundamentals of Aerodynamics*, 6th ed., sec. 4.8, the Glauert
Fourier-series solution of the cambered thin airfoil)::

    x = (1 - cos theta0) / 2
    A0 = alpha - (1/pi) int_0^pi dz/dx dtheta0
    A1 = (2/pi) int_0^pi dz/dx cos(theta0) dtheta0
    Cl = pi (2 A0 + A1)                     (symmetric section: Cl = 2 pi alpha)

Profile drag (viscous, attached flow): the turbulent flat-plate skin friction ``Cf = 0.074 / Re^0.2`` (Prandtl's
1/7-power law, Schlichting, *Boundary-Layer Theory*, ch. 21) times Hoerner's thickness form factor for a
streamlined 2-D section, both surfaces wetted (Hoerner, *Fluid-Dynamic Drag*, 1965, ch. 6)::

    Cd = 2 Cf (1 + 2 t/c + 60 (t/c)^4)       (referred to the chord)

Scope: no thickness effect on lift, no camber or lift effect on drag, no stall, no compressibility, no transition
(fully turbulent). Use a panel method (``pinneapple_design.aero.panel``) or RANS for anything beyond trends.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["naca4_camber_slope", "thin_airfoil_cl", "turbulent_flat_plate_cf", "hoerner_profile_cd"]


def naca4_camber_slope(x: np.ndarray, m: float, p: float) -> np.ndarray:
    """dz/dx of the NACA 4-digit mean line (max camber ``m`` at chord fraction ``p``), the two parabolic arcs."""
    x = np.asarray(x, dtype=float)
    if m == 0.0 or p <= 0.0:
        return np.zeros_like(x)
    fwd = 2.0 * m / p**2 * (p - x)
    aft = 2.0 * m / (1.0 - p) ** 2 * (p - x)
    return np.where(x < p, fwd, aft)


def thin_airfoil_cl(m: float, p: float, alpha_deg: float, n_quad: int = 4001) -> float:
    """Thin-airfoil lift coefficient of a NACA 4-digit camber line at ``alpha_deg`` (trapezoidal quadrature in
    theta0; the integrands are bounded, the kink at x = p converges as 1/n_quad)."""
    theta = np.linspace(0.0, math.pi, int(n_quad))
    dz = naca4_camber_slope(0.5 * (1.0 - np.cos(theta)), m, p)
    a0 = math.radians(alpha_deg) - np.trapezoid(dz, theta) / math.pi
    a1 = 2.0 / math.pi * np.trapezoid(dz * np.cos(theta), theta)
    return float(math.pi * (2.0 * a0 + a1))


def turbulent_flat_plate_cf(reynolds: float) -> float:
    """Mean skin-friction coefficient of a fully turbulent flat plate, ``0.074 / Re^0.2`` (5e5 < Re < 1e7)."""
    if reynolds <= 0:
        raise ValueError("reynolds must be positive")
    return 0.074 / reynolds**0.2


def hoerner_profile_cd(t_c: float, reynolds: float) -> float:
    """Profile drag coefficient of a streamlined 2-D section of thickness ratio ``t_c`` (Hoerner form factor, two
    wetted sides, referred to the chord)."""
    return float(2.0 * turbulent_flat_plate_cf(reynolds) * (1.0 + 2.0 * t_c + 60.0 * t_c**4))
