"""Initial (small-angle) transverse stability of a surface ship: metacentric height, free-surface correction, righting
arm and natural roll period.

The standard relations (e.g. Biran & Lopez-Pulido, *Ship Hydrostatics and Stability*, 2nd ed., 2014, ch. 2-6):

* ``KM = KB + BM``, ``BM = I_T / volume``, ``GM = KM - KG - FSC``;
* free-surface correction of a slack tank: ``FSC = rho_tank i_T / (rho_sea volume)``, with ``i_T`` the transverse
  second moment of the tank's free surface;
* box barge: ``KB = T/2`` and ``BM = B^2 / (12 T)``;
* righting arm:
  - small angles: ``GZ = GM sin(phi)``;
  - the *wall-sided* formula ``GZ = sin(phi) (GM + BM tan^2(phi) / 2)`` holds until the deck edge immerses or the
    bilge emerges;
* natural roll period: ``T_phi = 2 pi k_xx / sqrt(g GM)``, with ``k_xx`` the roll radius of gyration including
  added mass. When it is not known, ``0.35 B`` is a common first estimate (the IMO 2008 IS Code formula gives
  0.33-0.40 B for usual ships).

A negative ``GM`` means the upright position is unstable: the ship lolls to an angle or capsizes.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["box_barge", "free_surface_correction", "metacentric_height", "righting_arm", "roll_period",
           "initial_stability"]

G = 9.81


def box_barge(L: float, B: float, T: float, rho: float = 1025.0) -> dict[str, float]:
    """Hydrostatics of a rectangular barge floating upright.

    Parameters
    ----------
    L, B, T : float [m]
        Length, beam and draught.
    rho : float [kg/m^3]
        Water density.

    Returns
    -------
    volume : float [m^3]
        Displaced volume ``L B T``.
    displacement : float [kg]
        Displaced mass.
    KB : float [m]
        Height of the centre of buoyancy above the keel (``T/2``).
    BM : float [m]
        Transverse metacentric radius ``B^2 / (12 T)``.
    KM : float [m]
        Height of the transverse metacentre above the keel.
    I_T : float [m^4]
        Transverse second moment of the waterplane ``L B^3 / 12``.
    """
    vol = L * B * T
    I_T = L * B ** 3 / 12.0
    KB, BM = T / 2.0, I_T / vol
    return {"volume": vol, "displacement": rho * vol, "KB": KB, "BM": BM, "KM": KB + BM, "I_T": I_T}


def free_surface_correction(tanks: Sequence[dict[str, float]], volume: float, rho_sea: float = 1025.0) -> float:
    """Free-surface correction of slack tanks: ``sum(rho_tank i_T) / (rho_sea volume)``.

    Parameters
    ----------
    tanks : sequence of dict
        Each with ``l`` [m] (length) and ``b`` [m] (breadth) of a rectangular free surface, and ``rho`` [kg/m^3] of
        the liquid; or ``i_T`` [m^4] directly instead of ``l`` and ``b``.
    volume : float [m^3]
        Displaced volume of the ship.
    rho_sea : float [kg/m^3]
        Water density.

    Returns
    -------
    float [m]
        The virtual rise of the centre of gravity (subtract from GM).
    """
    total = 0.0
    for tk in tanks:
        i_t = float(tk["i_T"]) if "i_T" in tk else float(tk["l"]) * float(tk["b"]) ** 3 / 12.0
        total += float(tk.get("rho", 1000.0)) * i_t
    return total / (rho_sea * volume)


def metacentric_height(KB: float, BM: float, KG: float, fsc: float = 0.0) -> float:
    """``GM = KB + BM - KG - FSC`` [m] (all inputs in m)."""
    return KB + BM - KG - fsc


def righting_arm(GM: float, phi_deg, BM: float | None = None) -> np.ndarray:
    """Righting arm ``GZ`` [m] at heel angles ``phi_deg`` [deg].

    With ``BM`` given, the wall-sided formula ``sin(phi) (GM + BM tan^2(phi)/2)``; otherwise the small-angle
    ``GM sin(phi)``. Both assume the deck edge stays above water and the bilge below.
    """
    phi = np.radians(np.asarray(phi_deg, float))
    if BM is None:
        return GM * np.sin(phi)
    return np.sin(phi) * (GM + 0.5 * BM * np.tan(phi) ** 2)


def roll_period(GM: float, B: float | None = None, k_xx: float | None = None, g: float = G) -> float:
    """Natural roll period ``2 pi k_xx / sqrt(g GM)`` [s].

    ``k_xx`` [m] is the roll radius of gyration with added mass. When omitted it is estimated as ``0.35 B``
    (``B`` [m], the beam): a first estimate only. Returns ``inf`` for ``GM <= 0``.
    """
    if k_xx is None:
        if B is None:
            raise ValueError("give k_xx or the beam B")
        k_xx = 0.35 * B
    if GM <= 0:
        return math.inf
    return 2 * math.pi * k_xx / math.sqrt(g * GM)


def initial_stability(KB: float, BM: float, KG: float, volume: float, B: float,
                      tanks: Sequence[dict[str, float]] = (), k_xx: float | None = None, rho_sea: float = 1025.0,
                      min_gm: float = 0.15) -> dict[str, Any]:
    """Initial stability of an upright ship, with slack tanks.

    Parameters
    ----------
    KB, BM, KG : float [m]
        Centre of buoyancy, metacentric radius and centre of gravity heights.
    volume : float [m^3]
        Displaced volume.
    B : float [m]
        Beam (for the roll-period estimate).
    tanks : sequence of dict
        Slack tanks, as in :func:`free_surface_correction`.
    k_xx : float [m], optional
        Roll radius of gyration (default ``0.35 B``).
    min_gm : float [m]
        Smallest acceptable corrected GM. The default 0.15 m is the IMO 2008 IS Code general criterion.

    Returns
    -------
    GM_solid : float [m]
        GM without the free-surface correction.
    fsc : float [m]
        Free-surface correction.
    GM : float [m]
        Corrected GM.
    stable : bool
        ``GM > 0``.
    meets_min_gm : bool
        ``GM >= min_gm``.
    roll_period : float [s]
        Natural roll period (``inf`` when ``GM <= 0``).
    notes : list of str
        Plain-language remarks.
    """
    fsc = free_surface_correction(tanks, volume, rho_sea) if tanks else 0.0
    gm0 = metacentric_height(KB, BM, KG)
    gm = gm0 - fsc
    notes = []
    if gm <= 0:
        notes.append(f"GM = {gm:.3f} m <= 0: the upright position is unstable (angle of loll or capsize)")
    elif gm < min_gm:
        notes.append(f"GM = {gm:.3f} m is below the {min_gm:g} m criterion")
    if fsc > 0:
        notes.append(f"slack tanks raise G virtually by {fsc:.3f} m")
    return {"GM_solid": gm0, "fsc": fsc, "GM": gm, "stable": gm > 0, "meets_min_gm": gm >= min_gm,
            "roll_period": roll_period(gm, B=B, k_xx=k_xx), "notes": notes}
