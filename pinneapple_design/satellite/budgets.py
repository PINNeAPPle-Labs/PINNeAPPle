"""Closed-form budgets of a small Earth-observation satellite: orbit (sun-synchronous inclination, J2 drift, eclipse,
revisit), power (end-of-life array, battery depth of discharge), single-node thermal (hot and cold cases), RF link
(EIRP, path loss, G/T, Eb/N0) and the attitude disturbance torques with wheel and magnetorquer sizing.

The formulas are the standard ones of Wertz and Larson, *Space Mission Analysis and Design* (3rd ed., 1999):
power sizing with peak-power tracking (X_e = 0.60, X_d = 0.80, eq. 11-5), disturbance torques (table 11-9), the
momentum a wheel stores over a quarter orbit (0.707 T P / 4), the magnetorquer dipole ``T / B``. Component
properties (cell efficiency, battery energy density, catalogue-like wheel and torquer steps) are representative
values, not a vendor's data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MU = 3.986004418e14            # m^3/s^2
R_E = 6378137.0                # m
J2 = 1.08262668e-3
OMEGA_E = 7.2921159e-5         # rad/s, Earth rotation
OMEGA_SUN = 2 * math.pi / (365.2421897 * 86400.0)   # mean motion of the Sun (sun-synchronous node rate)
SOLAR = 1361.0                 # W/m^2 (Kopp and Lean 2011)
SIGMA = 5.670374419e-8
EARTH_IR = 237.0               # W/m^2, mean outgoing long-wave
ALBEDO = 0.30
C_LIGHT = 299792458.0
K_B_DB = 10 * math.log10(1.380649e-23)     # -228.599 dB(W/K/Hz)
M_DIPOLE = 7.96e15             # T m^3, Earth's dipole (SMAD)

__all__ = ["sso_inclination", "nodal_rate", "eclipse_fraction", "eclipse_fraction_sampled", "beta_angles_sso",
           "revisit", "solar_array", "battery", "Coating", "COATINGS", "MLI", "surface", "thermal_cases", "link_budget",
           "slant_range", "disturbance_torques", "wheel_and_torquer"]


# --------------------------------------------------------------------------- orbit
def nodal_rate(a: float, e: float, i_rad: float) -> float:
    """Secular J2 drift of the ascending node, rad/s: ``-3/2 n J2 (R_E / p)^2 cos i``."""
    n = math.sqrt(MU / a ** 3)
    p = a * (1 - e * e)
    return -1.5 * n * J2 * (R_E / p) ** 2 * math.cos(i_rad)


def sso_inclination(h_km: float, e: float = 0.0) -> float:
    """Inclination (deg) whose J2 node drift equals the Sun's mean motion."""
    a = R_E + h_km * 1e3
    c = -OMEGA_SUN * 2 * a ** 3.5 * (1 - e * e) ** 2 / (3 * J2 * R_E ** 2 * math.sqrt(MU))
    if abs(c) > 1:
        raise ValueError(f"no sun-synchronous inclination at {h_km} km")
    return math.degrees(math.acos(c))


def eclipse_fraction(h_km: float, beta_deg: float) -> float:
    """Fraction of a circular orbit in the Earth's cylindrical shadow:
    ``acos(sqrt(h^2 + 2 R h) / ((R + h) cos beta)) / pi`` for ``|beta| < asin(R / (R + h))``, else 0."""
    r = R_E + h_km * 1e3
    b = math.radians(beta_deg)
    if abs(math.sin(b)) >= R_E / r:
        return 0.0
    return math.acos(math.sqrt((h_km * 1e3) ** 2 + 2 * R_E * h_km * 1e3) / (r * math.cos(b))) / math.pi


def eclipse_fraction_sampled(h_km: float, beta_deg: float, n: int = 200_000) -> float:
    """The same fraction from geometry: points of the orbit behind the Earth (r . s < 0) within R_E of the Sun line."""
    r = R_E + h_km * 1e3
    u = (np.arange(n) + 0.5) / n * 2 * np.pi
    b = math.radians(beta_deg)
    s = np.array([math.cos(b), 0.0, math.sin(b)])          # Sun in the orbit frame (orbit in the x-y plane)
    pos = r * np.stack([np.cos(u), np.sin(u), np.zeros_like(u)], 1)
    along = pos @ s
    perp = np.linalg.norm(pos - along[:, None] * s, axis=1)
    return float(np.mean((along < 0) & (perp < R_E)))


def beta_angles_sso(i_deg: float, ltan_h: float, n: int = 365) -> np.ndarray:
    """Sun-orbit-plane angle over a year for a sun-synchronous orbit with local time of ascending node ``ltan_h``
    (mean Sun; the equation of time, at most 16 min, is neglected): ``sin beta = cos d sin i sin(O - a) + sin d cos i``
    with the node fixed relative to the mean Sun."""
    d = np.arange(n) * 365.2422 / n
    eps = math.radians(23.439)
    lam = np.radians(280.46 + 0.9856474 * d) + np.radians(1.915) * np.sin(np.radians(357.528 + 0.9856003 * d))
    dec = np.arcsin(math.sin(eps) * np.sin(lam))
    ra = np.arctan2(math.cos(eps) * np.sin(lam), np.cos(lam))
    ra_mean = np.radians(280.46 + 0.9856474 * d)
    node = ra_mean + math.radians((ltan_h - 12.0) * 15.0)
    i = math.radians(i_deg)
    return np.degrees(np.arcsin(np.cos(dec) * math.sin(i) * np.sin(node - ra) + np.sin(dec) * math.cos(i)))


def revisit(h_km: float, i_deg: float, fov_deg: float, ifov_urad: float) -> dict[str, float]:
    """Ground-track spacing at the equator, swath (flat Earth, nadir), the days to close the equatorial gap with
    one pass per orbit (``spacing / swath``) and the ground sample distance ``h IFOV``."""
    a = R_E + h_km * 1e3
    T = 2 * math.pi * math.sqrt(a ** 3 / MU)
    node = nodal_rate(a, 0.0, math.radians(i_deg))
    spacing = (OMEGA_E - node) * T * R_E
    swath = 2 * h_km * 1e3 * math.tan(math.radians(fov_deg / 2))
    return {"period_s": T, "orbits_per_day": 86400.0 / T, "track_spacing_km": spacing / 1e3, "swath_km": swath / 1e3,
            "revisit_days": spacing / swath, "gsd_m": h_km * 1e3 * ifov_urad * 1e-6}


# --------------------------------------------------------------------------- power
def solar_array(p_day_w: float, p_ecl_w: float, period_s: float, f_ecl: float, years: float, *,
                efficiency: float = 0.28, inherent_degradation: float = 0.77, incidence_deg: float = 23.5,
                degradation_per_year: float = 0.0275, kg_per_m2: float = 2.0, x_e: float = 0.60,
                x_d: float = 0.80) -> dict[str, float]:
    """Array power and area sized at end of life (SMAD eq. 11-5 to 11-8)."""
    t_e, t_d = f_ecl * period_s, (1 - f_ecl) * period_s
    p_sa = (p_ecl_w * t_e / x_e + p_day_w * t_d / x_d) / t_d
    p_bol = SOLAR * efficiency * inherent_degradation * math.cos(math.radians(incidence_deg))
    life = (1 - degradation_per_year) ** years
    p_eol = p_bol * life
    area = p_sa / p_eol
    return {"p_required_w": p_sa, "p_bol_w_m2": p_bol, "p_eol_w_m2": p_eol, "life_factor": life,
            "area_m2": area, "mass_kg": area * kg_per_m2, "p_bol_w": area * p_bol}


#: Li-ion depth of discharge allowed for a cycle count (typical published cycle-life curves; log interpolation)
DOD_TABLE = ((1e3, 0.80), (1e4, 0.50), (3e4, 0.30), (5e4, 0.25), (1e5, 0.20))


def battery(p_ecl_w: float, period_s: float, f_ecl: float, years: float, *, eta: float = 0.90,
            wh_per_kg: float = 100.0) -> dict[str, float]:
    """Capacity from the eclipse energy at the depth of discharge the cycle count allows."""
    cycles = years * 365.25 * 86400.0 / period_s
    c, d = np.log10([x for x, _ in DOD_TABLE]), [y for _, y in DOD_TABLE]
    dod = float(np.interp(math.log10(max(cycles, 1.0)), c, d))
    e_ecl = p_ecl_w * f_ecl * period_s / 3600.0
    cap = e_ecl / (dod * eta)
    return {"cycles": cycles, "dod": dod, "eclipse_wh": e_ecl, "capacity_wh": cap, "mass_kg": cap / wh_per_kg}


# --------------------------------------------------------------------------- thermal
@dataclass(frozen=True)
class Coating:
    name: str
    alpha: float
    eps: float


#: beginning-of-life optical properties (Gilmore, *Spacecraft Thermal Control Handbook*, appendix A, rounded)
COATINGS = (Coating("white paint", 0.20, 0.88), Coating("silvered teflon", 0.08, 0.78),
            Coating("black anodize", 0.88, 0.88), Coating("bare aluminium", 0.15, 0.05),
            Coating("gold", 0.30, 0.03), Coating("solar cells + white paint", 0.55, 0.85))


#: effective properties of multi-layer insulation seen from outside (an effective emittance of a few percent)
MLI = Coating("MLI", 0.03, 0.03)


def surface(coating: Coating, radiator_fraction: float) -> Coating:
    """Area-weighted optical properties of a box whose ``radiator_fraction`` of the surface carries ``coating`` and
    the rest MLI (the single-node approximation of a radiator)."""
    f = float(radiator_fraction)
    return Coating(f"{coating.name} on {f:.0%} + MLI", f * coating.alpha + (1 - f) * MLI.alpha,
                   f * coating.eps + (1 - f) * MLI.eps)


def thermal_cases(dims_m: tuple[float, float, float], h_km: float, coating: Coating, q_int_hot: float,
                  q_int_cold: float) -> dict[str, float]:
    """Single-node equilibrium of a box: hot = sunlit, subsolar albedo and Earth IR on the nadir face, full internal
    power; cold = eclipse, Earth IR only, the least internal power. Area to the Sun: the largest face; to the Earth:
    a face with view factor ``(R / (R + h))^2``. Radiates from its whole surface with the (effective) properties of
    ``coating`` (see :func:`surface`)."""
    a, b, c = sorted(dims_m)
    faces = [a * b, a * c, b * c]
    A_tot = 2 * sum(faces)
    A_sun, A_nadir = max(faces), min(faces)
    F = (R_E / (R_E + h_km * 1e3)) ** 2
    q_hot = coating.alpha * SOLAR * (A_sun + ALBEDO * F * A_nadir) + coating.eps * EARTH_IR * F * A_nadir + q_int_hot
    q_cold = coating.eps * EARTH_IR * F * A_nadir + q_int_cold
    T = lambda q: (q / (coating.eps * SIGMA * A_tot)) ** 0.25 - 273.15       # noqa: E731
    return {"t_hot_c": T(q_hot), "t_cold_c": T(q_cold), "q_hot_w": q_hot, "q_cold_w": q_cold, "area_m2": A_tot,
            "a_sun_m2": A_sun, "a_nadir_m2": A_nadir, "view_factor": F}


# --------------------------------------------------------------------------- link
def slant_range(h_km: float, elev_deg: float) -> float:
    """Range (m) to a satellite at altitude ``h`` seen at elevation ``elev`` from the ground."""
    e = math.radians(elev_deg)
    r = R_E + h_km * 1e3
    return math.sqrt(r * r - (R_E * math.cos(e)) ** 2) - R_E * math.sin(e)


def link_budget(*, p_tx_w: float, f_hz: float, h_km: float, elev_deg: float, g_tx_dbi: float, g_over_t_db_k: float,
                rate_bps: float, ebn0_req_db: float, line_loss_db: float = 1.0, atm_loss_db: float = 0.5,
                pol_loss_db: float = 0.5, point_loss_db: float = 1.0) -> dict[str, float]:
    """Downlink Eb/N0 and margin: ``EIRP - L_fs - L_other + G/T - 10 log k - 10 log R``."""
    d = slant_range(h_km, elev_deg)
    eirp = 10 * math.log10(p_tx_w) + g_tx_dbi - line_loss_db
    fspl = 20 * math.log10(4 * math.pi * d * f_hz / C_LIGHT)
    other = atm_loss_db + pol_loss_db + point_loss_db
    ebn0 = eirp - fspl - other + g_over_t_db_k - K_B_DB - 10 * math.log10(rate_bps)
    return {"range_km": d / 1e3, "eirp_dbw": eirp, "fspl_db": fspl, "other_losses_db": other,
            "g_over_t_db_k": g_over_t_db_k, "ebn0_db": ebn0, "margin_db": ebn0 - ebn0_req_db, "rate_bps": rate_bps}


# --------------------------------------------------------------------------- attitude
def disturbance_torques(*, h_km: float, rho_kg_m3: float, inertia: tuple[float, float, float],
                        area_m2: float, cp_offset_m: float, residual_dipole_am2: float, deviation_deg: float = 10.0,
                        cd: float = 2.2, reflectance: float = 0.6) -> dict[str, float]:
    """Worst-case torques (N m), SMAD table 11-9: gravity gradient, aerodynamic, solar pressure, magnetic (polar
    field ``2 M / R^3``)."""
    r = R_E + h_km * 1e3
    v = math.sqrt(MU / r)
    tg = 1.5 * MU / r ** 3 * (max(inertia) - min(inertia)) * math.sin(2 * math.radians(deviation_deg))
    ta = 0.5 * rho_kg_m3 * cd * area_m2 * v * v * cp_offset_m
    ts = SOLAR / C_LIGHT * area_m2 * (1 + reflectance) * cp_offset_m
    tm = residual_dipole_am2 * 2 * M_DIPOLE / r ** 3
    return {"gravity_gradient": tg, "aerodynamic": ta, "solar_pressure": ts, "magnetic": tm,
            "total": tg + ta + ts + tm, "b_min_t": M_DIPOLE / r ** 3, "b_max_t": 2 * M_DIPOLE / r ** 3}


#: representative small-satellite steps: wheels (momentum N m s, torque N m, mass kg, power W) and torquers (A m^2,
#: kg, W); not a vendor catalogue
WHEELS = ((0.002, 0.0002, 0.13, 0.6), (0.010, 0.001, 0.25, 1.0), (0.030, 0.002, 0.45, 1.5), (0.10, 0.005, 0.9, 3.0),
          (0.50, 0.020, 2.5, 8.0), (2.0, 0.05, 5.0, 15.0))
TORQUERS = ((0.2, 0.03, 0.2), (1.0, 0.08, 0.4), (5.0, 0.3, 0.8), (15.0, 0.8, 1.5), (50.0, 2.5, 3.0))


def wheel_and_torquer(torque_nm: float, period_s: float, b_min_t: float, inertia: tuple[float, float, float], *,
                      slew_deg: float = 30.0, slew_s: float = 60.0, margin: float = 2.0) -> dict[str, float]:
    """Wheel momentum ``0.707 T P / 4`` (cyclic disturbance over a quarter orbit) and slew torque ``4 theta I / t^2``,
    each with ``margin``; torquer dipole ``margin T / B_min``. Three wheels and three torquers."""
    h_need = margin * 0.707 * torque_nm * period_s / 4
    t_need = margin * max(4 * math.radians(slew_deg) * max(inertia) / slew_s ** 2, torque_nm)
    d_need = margin * torque_nm / b_min_t
    w = next((x for x in WHEELS if x[0] >= h_need and x[1] >= t_need), None)
    m = next((x for x in TORQUERS if x[0] >= d_need), None)
    return {"h_needed_nms": h_need, "torque_needed_nm": t_need, "dipole_needed_am2": d_need,
            "wheel": None if w is None else {"h_nms": w[0], "torque_nm": w[1], "mass_kg": w[2], "power_w": w[3]},
            "torquer": None if m is None else {"dipole_am2": m[0], "mass_kg": m[1], "power_w": m[2]},
            "mass_kg": (3 * w[2] if w else math.inf) + (3 * m[1] if m else math.inf),
            "power_w": (3 * w[3] if w else math.inf) + (m[2] if m else math.inf)}
