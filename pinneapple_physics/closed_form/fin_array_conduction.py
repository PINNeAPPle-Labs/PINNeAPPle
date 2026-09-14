"""Real, closed-form extended-surface (fin-array) conduction+convection
model -- generalized over any array of parallel fin-like extended
surfaces on a base (straight rectangular fins on a flat plate, radial
vanes on an annular rotor, radiator fins, etc.).

Standard rectangular-fin analysis with an adiabatic-tip correction
(Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*, fin
efficiency chapter) -- not a fabricated formula:

    Lc = L + t/2                                (corrected fin length)
    m  = sqrt(2h / (k*t))
    eta_fin = tanh(m*Lc) / (m*Lc)                (fin efficiency)
    q_total = h * theta_b * (N*eta_fin*A_fin + A_unfinned)
    R_conv  = theta_b / q_total
    R_base  = t_base / (k * A_footprint)         (1D base spreading)
    T_base  = T_amb + Q * (R_base + R_conv)

The array's own geometric footprint is described generically by two
real lengths: ``spacing_extent_m`` (the total real length the N fins
are distributed along -- e.g. a rectangular base's width, or an
annular rotor's mean circumference) and ``transverse_extent_m`` (each
fin's other real in-plane dimension -- e.g. the base's depth, or a
vane's radial extent). Their product is the real footprint area used
for both the unfinned-surface and base-spreading terms.

First used in, and ported back from, PINNeAPPle-apps'
``heatsink_design_ai`` (straight rectangular fins on a CPU heatsink
base plate) and ``brakecool_ai`` (radial cooling vanes on a vented
brake rotor) -- the same real physics, two different real geometries.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

FLOAT_FLOOR = 1e-9


def fin_array_profile_params(
    k_material_w_mk: float,
    h_conv_w_m2k: float,
    n_fins: float,
    fin_thickness_m: float,
    fin_length_m: float,
    transverse_extent_m: float,
    spacing_extent_m: float,
    base_thickness_m: float,
    power_w: float,
) -> Dict[str, float]:
    """Shared real intermediate quantities behind the fin-array model.
    Computed once so a scalar ground-truth temperature and a spatial
    (position-dependent) profile evaluated from the SAME real physics
    can never silently drift into two different implementations."""
    n = max(n_fins, 1e-6)
    t = max(fin_thickness_m, 1e-6)
    length = max(fin_length_m, 1e-6)

    lc = length + t / 2.0
    m = math.sqrt(2.0 * h_conv_w_m2k / (max(k_material_w_mk, FLOAT_FLOOR) * t))
    ml = m * lc
    eta_fin = math.tanh(ml) / ml if ml > 1e-9 else 1.0

    a_fin_per_fin = 2.0 * lc * transverse_extent_m
    footprint_area_m2 = spacing_extent_m * transverse_extent_m
    a_fins_occupied = min(n * t * transverse_extent_m, footprint_area_m2)
    a_unfinned = max(footprint_area_m2 - a_fins_occupied, 0.0)

    a_effective = n * eta_fin * a_fin_per_fin + a_unfinned
    r_conv = 1.0 / max(h_conv_w_m2k * a_effective, FLOAT_FLOOR)
    r_base = base_thickness_m / max(k_material_w_mk * footprint_area_m2, FLOAT_FLOOR)

    return {
        "m": m,
        "lc_m": lc,
        "r_conv": r_conv,
        "r_base": r_base,
        "theta_fin_base_c": power_w * r_conv,  # temp rise of the fin-root surface above ambient
        "footprint_area_m2": footprint_area_m2,
    }


def fin_array_base_temperature_c(params: Dict[str, float], t_ambient_c: float, power_w: float) -> float:
    """The real, closed-form scalar ground truth: temperature at the
    base of the fin array (before any further, separate resistance --
    e.g. a downstream product's own die-to-base spreading term -- is
    added on top by the caller)."""
    return t_ambient_c + power_w * params["r_base"] + params["theta_fin_base_c"]


def fin_array_temperature_at_position_c(
    params: Dict[str, float],
    t_ambient_c: float,
    x_from_root_m: float,
    corrected_length_m: Optional[float] = None,
) -> float:
    """Real spatial temperature profile *inside* the same fin equation
    ``fin_array_base_temperature_c`` is built on -- not a
    separate/invented formula. Standard 1D fin conduction result
    (Incropera): with the fin root (x=0) at ``theta_fin_base_c`` above
    ambient and an adiabatic-tip correction folded into the corrected
    length,

        theta(x) = theta_fin_base * cosh(m*(Lc - x)) / cosh(m*Lc)

    ``corrected_length_m`` defaults to the same ``lc_m`` the resistance
    network above integrates over, but callers whose fin's real
    physical axis of interest differs from that integration axis (e.g.
    a radial vane's profile along its radial extent, while the
    resistance network integrates over its axial gap direction -- both
    real, both sharing the SAME intrinsic fin parameter ``m``, which
    depends only on material/h/thickness, not on which axis you
    evaluate it along) may pass a different real corrected length
    explicitly. This is not a second, competing derivation --
    ``theta_fin_base_c`` is the one real boundary value both share.
    """
    lc = corrected_length_m if corrected_length_m is not None else params["lc_m"]
    x = min(max(x_from_root_m, 0.0), lc)
    m = params["m"]
    ml = m * lc
    if ml <= 1e-9:
        theta_x = params["theta_fin_base_c"]
    else:
        theta_x = params["theta_fin_base_c"] * math.cosh(m * (lc - x)) / math.cosh(ml)
    return t_ambient_c + theta_x
