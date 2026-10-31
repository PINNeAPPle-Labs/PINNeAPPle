"""Ideal isothermal reactors with a first-order reaction, in closed form (Levenspiel, *Chemical Reaction Engineering*,
3rd ed., ch. 5):

    plug flow (PFR):               X = 1 - exp(-k tau)
    continuous stirred tank (CSTR): X = k tau / (1 + k tau)
    residence time:                tau = V / Q,   V = pi D^2 L / 4 for a tube of diameter D and length L

``k`` is the first-order rate constant [1/s], ``Q`` the volumetric flow [m3/s] and ``X`` the conversion of the
reactant [-]. Each function carries ``x_ranges`` (the input box it is declared for, used to say when a point is
outside it) and ``x_outputs`` (name, type, unit and description of each output; the module's ``X_OUTPUTS``).
"""
from __future__ import annotations

import math
from typing import TypedDict

__all__ = ["tube_volume", "pfr_conversion", "pfr_conversion_ktau", "cstr_conversion_ktau", "PFRResult"]

# what each tool returns, with units (read by the Studio catalog; ``x_outputs`` on each function points here)
X_OUTPUTS = {
    "tube_volume": [{"name": "V", "type": "float", "unit": "m3", "description": "volume of the tube"}],
    "pfr_conversion": [{"name": "X", "type": "float", "unit": "-", "description": "conversion of the reactant"},
                       {"name": "tau", "type": "float", "unit": "s", "description": "residence time V / Q"},
                       {"name": "V", "type": "float", "unit": "m3", "description": "volume of the tube"}],
    "pfr_conversion_ktau": [{"name": "X", "type": "float", "unit": "-", "description": "conversion of the reactant"}],
    "cstr_conversion_ktau": [{"name": "X", "type": "float", "unit": "-", "description": "conversion of the reactant"}],
}


class PFRResult(TypedDict):
    X: float
    tau: float
    V: float


def tube_volume(D: float, L: float) -> float:
    """Volume of a tube.

    Parameters
    ----------
    D : float
        inner diameter [m]
    L : float
        length [m]

    Returns
    -------
    V : float [m3]
        volume of the tube
    """
    return math.pi * D * D * L / 4.0


def pfr_conversion(D: float, L: float, k: float, Q: float) -> PFRResult:
    """Conversion of an ideal isothermal plug-flow tube with a first-order reaction, X = 1 - exp(-k V / Q).

    Parameters
    ----------
    D : float
        inner diameter of the tube [m]
    L : float
        length of the tube [m]
    k : float
        first-order rate constant [1/s]
    Q : float
        volumetric flow [m3/s]

    Returns
    -------
    X : float [-]
        conversion of the reactant
    tau : float [s]
        residence time V / Q
    V : float [m3]
        volume of the tube
    """
    if min(D, L, Q) <= 0 or k < 0:
        raise ValueError("D, L and Q must be positive and k non-negative")
    V = tube_volume(D, L)
    tau = V / Q
    return {"X": -math.expm1(-k * tau), "tau": tau, "V": V}


pfr_conversion.x_ranges = {"D": (1e-3, 1.0), "L": (0.01, 100.0), "k": (0.0, 10.0), "Q": (1e-7, 1.0)}
pfr_conversion.x_outputs = X_OUTPUTS["pfr_conversion"]


def pfr_conversion_ktau(k: float, tau: float) -> dict[str, float]:
    """Plug-flow conversion from the rate constant and the residence time, X = 1 - exp(-k tau).

    Parameters
    ----------
    k : float
        first-order rate constant [1/s]
    tau : float
        residence time [s]

    Returns
    -------
    X : float [-]
        conversion of the reactant
    """
    if k < 0 or tau < 0:
        raise ValueError("k and tau must be non-negative")
    return {"X": -math.expm1(-k * tau)}


pfr_conversion_ktau.x_ranges = {"k": (0.0, 10.0), "tau": (0.0, 1e4)}
pfr_conversion_ktau.x_outputs = X_OUTPUTS["pfr_conversion_ktau"]


def cstr_conversion_ktau(k: float, tau: float) -> dict[str, float]:
    """Conversion of an ideal stirred tank with a first-order reaction, X = k tau / (1 + k tau).

    Parameters
    ----------
    k : float
        first-order rate constant [1/s]
    tau : float
        residence time [s]

    Returns
    -------
    X : float [-]
        conversion of the reactant
    """
    if k < 0 or tau < 0:
        raise ValueError("k and tau must be non-negative")
    return {"X": k * tau / (1.0 + k * tau)}


cstr_conversion_ktau.x_ranges = {"k": (0.0, 10.0), "tau": (0.0, 1e4)}
cstr_conversion_ktau.x_outputs = X_OUTPUTS["cstr_conversion_ktau"]
