"""The US Standard Atmosphere 1976 from the ground to 1000 km: temperature, pressure, density and speed of sound,
the model every ascent calculation below drags against.

Up to 84.852 km (geopotential) the atmosphere is the seven-layer table of NASA-TM-X-74335 (the 1976 standard): a
base geopotential altitude, a lapse rate, a base temperature and a base pressure per layer, with

    T(H) = T_b + L (H - H_b)
    p(H) = p_b (T_b / T) ** (g0 M / (R* L))      (L != 0)
    p(H) = p_b exp(-g0 M (H - H_b) / (R* T_b))   (L == 0)
    rho  = p M / (R* T),   a = sqrt(gamma R* T / M)

and the geopotential altitude H = R0 h / (R0 + h) over the geometric altitude h (R0 = 6356.766 km, the effective
radius of the 1976 standard). Above 84.852 km the same document's upper table (86 km to 1000 km) is used with the
density interpolated linearly in ln(rho) in geopotential altitude, which is what an ascent needs there: the drag at
100 km is already a thousandth of the dynamic pressure at 50 km.

    from pinneapple_simulation.launch import atmosphere
    atmosphere.state(5000.0)["rho"]          # density [kg/m^3] at 5 km
    atmosphere.speed_of_sound(11000.0)       # [m/s] at the tropopause

The model is dry and mid-latitude, no wind, no day-night or latitude variation: an ascent through it is reproducible
to the last digit, which is what a verification needs.
"""
from __future__ import annotations

import bisect
import math

import numpy as np

__all__ = ["LAYERS", "R_SPECIFIC", "G0", "R0_M", "state", "temperature", "pressure", "density", "speed_of_sound",
           "geopotential", "geometric", "TABLE_ALTITUDES_KM"]

G0 = 9.80665                 # standard gravity [m/s^2]
R_STAR = 8.31432             # universal gas constant [J/(mol K)] as used by the 1976 standard
M_AIR = 0.0289644            # molar mass of dry air [kg/mol]
R_SPECIFIC = R_STAR / M_AIR  # 287.053 [J/(kg K)]
GAMMA = 1.4                  # ratio of specific heats, below 86 km
R0_M = 6_356_766.0           # effective Earth radius of the 1976 standard [m] (geopotential altitude)

#: The seven layers: geopotential base altitude [km], lapse rate [K/km], base temperature [K], base pressure [Pa]
LAYERS = (
    (0.0, -6.5, 288.150, 101325.0),
    (11.0, 0.0, 216.650, 22632.06),
    (20.0, 1.0, 216.650, 5474.889),
    (32.0, 2.8, 228.650, 868.0187),
    (47.0, 0.0, 270.650, 110.9063),
    (51.0, -2.8, 270.65, 66.93887),
    (71.0, -2.0, 214.65, 3.956420),
    (84.852, 0.0, 186.946, 0.3734),
)

#: Upper atmosphere (NASA-TM-X-74335, part 3): geopotential altitude [km], temperature [K], density [kg/m^3]
UPPER = (
    (86.0, 186.87, 6.958e-6),
    (90.0, 186.87, 3.416e-6),
    (100.0, 195.08, 5.604e-7),
    (110.0, 240.00, 9.708e-8),
    (120.0, 360.00, 2.222e-8),
    (150.0, 634.39, 2.076e-9),
    (200.0, 854.56, 2.541e-10),
    (300.0, 976.01, 1.916e-11),
    (500.0, 999.24, 5.215e-13),
    (1000.0, 1000.0, 3.561e-15),
)

#: Altitudes at which the checks compare with the published table [km]
TABLE_ALTITUDES_KM = (0.0, 5.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.0)

_LOWER_H = [layer[0] for layer in LAYERS]
_UPPER_H = np.array([u[0] for u in UPPER])
_UPPER_T = np.array([u[1] for u in UPPER])
_UPPER_LNRHO = np.log(np.array([u[2] for u in UPPER]))


def geopotential(h_m):
    """Geopotential altitude [m] of a geometric altitude ``h_m`` [m] (R0 h / (R0 + h))."""
    h = np.asarray(h_m, float)
    return R0_M * h / (R0_M + h)


def geometric(H_m):
    """Geometric altitude [m] of a geopotential altitude ``H_m`` [m]."""
    H = np.asarray(H_m, float)
    return R0_M * H / (R0_M - H)


def _layer(H_km):
    """Index of the layer whose base is at or below ``H_km`` (geopotential) and that layer's constants."""
    k = 0
    for i, (Hb, *_rest) in enumerate(LAYERS):
        if H_km >= Hb:
            k = i
    return k, LAYERS[k]


def _lower(H_km: float):
    """Temperature [K] and pressure [Pa] at one geopotential altitude from the seven-layer table."""
    _, (Hb, L, Tb, pb) = _layer(H_km)
    if L == 0.0:
        T = Tb
        p = pb * np.exp(-G0 * M_AIR * (H_km - Hb) * 1000.0 / (R_STAR * Tb))
    else:
        T = Tb + L * (H_km - Hb)
        p = pb * (Tb / T) ** (G0 * M_AIR / (R_STAR * L * 1e-3))
    return T, p


def _upper(H_km: float):
    """Temperature [K], density [kg/m^3] and pressure [Pa] above 84.852 km from the upper table."""
    H = min(max(H_km, _UPPER_H[0]), _UPPER_H[-1])
    T = float(np.interp(H, _UPPER_H, _UPPER_T))
    rho = float(np.exp(np.interp(H, _UPPER_H, _UPPER_LNRHO)))
    return T, rho, rho * R_SPECIFIC * T


def _scalar(h: float):
    """``(T, p, rho)`` at one geometric altitude, in plain Python: the ascent calls this several million times."""
    H_km = R0_M * max(h, 0.0) / (R0_M + max(h, 0.0)) / 1000.0
    k = bisect.bisect_right(_LOWER_H, H_km) - 1
    k = min(max(k, 0), len(LAYERS) - 1)
    Hb, L, Tb, pb = LAYERS[k]
    if L == 0.0:
        T = Tb
        p = pb * math.exp(-G0 * M_AIR * (H_km - Hb) * 1000.0 / (R_STAR * Tb))
    else:
        T = Tb + L * (H_km - Hb)
        p = pb * (Tb / T) ** (G0 * M_AIR / (R_STAR * L * 1e-3))
    if H_km <= _LOWER_H[-1]:
        return T, p, p / (R_SPECIFIC * T)
    j = min(max(bisect.bisect_right(_UPPER_H, H_km) - 1, 0), len(_UPPER_H) - 2)
    f = (H_km - _UPPER_H[j]) / (_UPPER_H[j + 1] - _UPPER_H[j])
    Tu = _UPPER_T[j] + f * (_UPPER_T[j + 1] - _UPPER_T[j])
    rho = math.exp(_UPPER_LNRHO[j] + f * (_UPPER_LNRHO[j + 1] - _UPPER_LNRHO[j]))
    return Tu, rho * R_SPECIFIC * Tu, rho


def _evaluate(h_m):
    """``(T, p, rho)`` (K, Pa, kg/m^3) at geometric altitude ``h_m``: floats for a scalar, arrays otherwise."""
    if np.ndim(h_m) == 0:
        return _scalar(float(h_m))
    h = np.asarray(h_m, float)
    flat = np.atleast_1d(h).ravel()
    T = np.empty_like(flat)
    p = np.empty_like(flat)
    rho = np.empty_like(flat)
    top = LAYERS[-1][0]
    for i, hi in enumerate(flat):
        H_km = float(geopotential(max(hi, 0.0))) / 1000.0
        if H_km <= top:
            Ti, pi = _lower(H_km)
            T[i], p[i], rho[i] = Ti, pi, pi / (R_SPECIFIC * Ti)
        else:
            Ti, ri, pi = _upper(H_km)
            T[i], p[i], rho[i] = Ti, pi, ri
    if h.ndim == 0:
        return float(T[0]), float(p[0]), float(rho[0])
    shape = h.shape
    return T.reshape(shape), p.reshape(shape), rho.reshape(shape)


def temperature(h_m):
    """Static temperature [K] at geometric altitude ``h_m`` [m]."""
    return _evaluate(h_m)[0]


def pressure(h_m):
    """Static pressure [Pa] at geometric altitude ``h_m`` [m]."""
    return _evaluate(h_m)[1]


def density(h_m):
    """Density [kg/m^3] at geometric altitude ``h_m`` [m]."""
    return _evaluate(h_m)[2]


def speed_of_sound(h_m):
    """Speed of sound [m/s] (gamma R* T / M)^0.5 at geometric altitude ``h_m`` [m]."""
    T = temperature(h_m)
    return np.sqrt(GAMMA * R_SPECIFIC * T)


def state(h_m):
    """Everything the ascent needs at one altitude: ``T`` [K], ``p`` [Pa], ``rho`` [kg/m^3], ``a`` [m/s], ``H`` [m].

    The ascent calls this at every integrator substep, so a scalar altitude takes the plain-Python path."""
    h = float(h_m)
    T, p, rho = _scalar(h)
    return {"h_m": h, "H_m": R0_M * max(h, 0.0) / (R0_M + max(h, 0.0)), "T": T, "p": p, "rho": rho,
            "a": math.sqrt(GAMMA * R_SPECIFIC * T)}
