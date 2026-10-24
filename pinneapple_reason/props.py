"""Physical constants, properties and small closed-form relations the failure modes share, each with its source.

Air properties are not repeated here: :func:`air` reads the heat-sink model's table
(:func:`pinneapple_physics.closed_form.plate_fin_heatsink.air_properties`, Incropera & DeWitt Table A.4).
"""
from __future__ import annotations

import math

__all__ = ["G", "SIGMA", "WATER", "STEEL", "air", "saturation_pressure", "dew_point", "sky_temperature",
           "wind_convection", "atmospheric_pressure", "ISO6946", "SRC"]

G = 9.80665                       # standard gravity, m/s^2 (CGPM 1901)
SIGMA = 5.670374419e-8            # Stefan-Boltzmann constant, W/(m^2 K^4) (CODATA 2018)
FARADAY = 96485.33212             # C/mol (CODATA 2018)

SRC = {
    "crc": "CRC Handbook of Chemistry and Physics (properties of water at 20 °C)",
    "magnus": ("Alduchov & Eskridge (1996), 'Improved Magnus form approximation of saturation vapor pressure', "
               "J. Appl. Meteor. 35, 601-609 (a = 17.625, b = 243.04 °C)"),
    "swinbank": "Swinbank (1963), 'Long-wave radiation from clear skies', Q. J. R. Meteorol. Soc. 89, 339-348",
    "watmuff": ("Watmuff, Charters & Proctor (1977), wind heat-transfer coefficient h = 2.8 + 3.0 V, as given in "
                "Duffie & Beckman, Solar Engineering of Thermal Processes, sec. 3.15"),
    "iso6946": "ISO 6946:2017, Building components and building elements - Thermal resistance, Table 7 and Annex C",
    "iso2533": "ISO 2533:1975 Standard Atmosphere (troposphere barometric formula)",
    "ec3": "EN 1993-1-1:2005, Eurocode 3 - Design of steel structures, 3.2.6 (E, G, nu) and 6.3.2 (lateral-torsional "
           "buckling, Tables 6.3 and 6.4)",
    "incropera": "Incropera, DeWitt, Bergman & Lavine, Fundamentals of Heat and Mass Transfer, 6th ed.",
    "wylie": "Wylie & Streeter, Fluid Transients in Systems, Prentice Hall, 1993, ch. 2-3",
}

# water near 20 °C (CRC Handbook): density, dynamic viscosity, surface tension against air, bulk modulus
WATER = {"rho": 998.2, "mu": 1.002e-3, "sigma": 0.0728, "K": 2.18e9}
# structural steel (EN 1993-1-1, 3.2.6): E, shear modulus G, Poisson's ratio, density (EN 1991-1-1 Table A.4: 78.5 kN/m3)
STEEL = {"E": 210e9, "G": 81e9, "nu": 0.3, "rho": 7850.0}


def air(t_c: float, p_pa: float = 101325.0) -> dict[str, float]:
    """Air properties at ``t_c`` (k, nu, Pr, rho, cp, alpha, beta)."""
    from pinneapple_physics.closed_form.plate_fin_heatsink import air_properties
    return air_properties(t_c, p_pa)


def saturation_pressure(t_c: float) -> float:
    """Saturation vapour pressure of water over a flat water surface, Pa (Magnus form, Alduchov & Eskridge 1996;
    within 0.4 % of the reference values for -40..50 °C)."""
    return 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))


def dew_point(t_c: float, rh: float) -> float:
    """Dew-point temperature, °C, of air at ``t_c`` with relative humidity ``rh`` (0-1) (inverse Magnus form)."""
    if not 0 < rh <= 1:
        raise ValueError("relative humidity must be in (0, 1]")
    g = math.log(rh) + 17.625 * t_c / (243.04 + t_c)
    return 243.04 * g / (17.625 - g)


def sky_temperature(t_air_k: float) -> float:
    """Effective clear-sky temperature, K: T_sky = 0.0552 T_air^1.5 (Swinbank 1963)."""
    return 0.0552 * t_air_k ** 1.5


def wind_convection(v: float) -> float:
    """External convection coefficient of a surface in wind, W/(m^2 K): h = 2.8 + 3.0 V (Watmuff et al. 1977)."""
    return 2.8 + 3.0 * v


def atmospheric_pressure(z_m: float) -> float:
    """Standard-atmosphere pressure at altitude ``z_m``, Pa (ISO 2533, troposphere)."""
    return 101325.0 * (1 - 2.25577e-5 * z_m) ** 5.25588


# ISO 6946:2017 surface resistances (m^2 K/W) and Annex C convective coefficients (W/(m^2 K)) by heat-flow direction
ISO6946 = {"Rsi_up": 0.10, "Rsi_horizontal": 0.13, "Rsi_down": 0.17, "Rse": 0.04,
           "hc_up": 5.0, "hc_horizontal": 2.5, "hc_down": 0.7}
