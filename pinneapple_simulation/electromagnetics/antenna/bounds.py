"""Physical limits on the directivity of a travelling-wave (end-fire) or aperture (broadside) antenna.

An end-fire antenna of length ``L`` (a Vivaldi, a Yagi, a helix, a dielectric rod) behaves like a continuous
end-fire line source of the same length. Its maximum directivity (C. A. Balanis, *Antenna Theory*, 4th ed., ch. 6,
directivity of uniform end-fire arrays with ``N d = L``) is

* ordinary end-fire (phase velocity = c): ``D ~= 4 L / lambda``;
* Hansen-Woodyard end-fire (the increased-directivity condition, slow wave): ``D ~= 1.805 * 4 L / lambda``
  (7.22 L / lambda; this module uses the rounded design value 7.28 L / lambda, 0.04 dB higher), the practical upper
  bound of a single travelling-wave antenna.

A broadside aperture of physical area ``A`` and aperture efficiency ``eta`` has ``G = eta 4 pi A / lambda^2``
(Balanis ch. 12). This is the wrong limit for an end-fire antenna, whose aperture faces along the board: a printed
Vivaldi of 100 mm x 200 mm has a large "broadside" area but its end-fire mouth is only ``W x h``. Use
:func:`endfire_directivity` for it.

:func:`min_length` inverts the end-fire limit (the shortest antenna that can reach a target gain at a frequency)
and :func:`feasibility_frequency` gives the lowest frequency at which a board of length ``L`` can reach it. Example:
10 dBi at 2 GHz needs ``L >= 10 / 7.28 * 149.9 mm = 206 mm`` in the Hansen-Woodyard limit.
"""
from __future__ import annotations

import numpy as np

from ..constants import C0

__all__ = ["HW_FACTOR", "ORDINARY_FACTOR", "endfire_directivity", "endfire_directivity_dbi", "aperture_gain",
           "aperture_gain_dbi", "min_length", "feasibility_frequency", "to_dbi", "from_dbi", "wavelength"]

ORDINARY_FACTOR = 4.0                  # D = 4 L / lambda
# Hansen-Woodyard: D = 1.805 * 4 L / lambda = 7.22 L / lambda from the array factor; 7.28 L / lambda is the value
# this module uses (the one the end-fire design rule of thumb quotes). They differ by 0.04 dB.
HW_FACTOR = 7.28
_FACTORS = {"hw": HW_FACTOR, "hansen-woodyard": HW_FACTOR, "ordinary": ORDINARY_FACTOR}


def to_dbi(x):
    return 10.0 * np.log10(np.asarray(x, dtype=float))


def from_dbi(g):
    return 10.0 ** (np.asarray(g, dtype=float) / 10.0)


def wavelength(f_hz):
    """Free-space wavelength [m] at ``f_hz``."""
    return C0 / np.asarray(f_hz, dtype=float)


def _factor(kind: str) -> float:
    try:
        return _FACTORS[kind.lower()]
    except KeyError:
        raise ValueError(f"unknown end-fire limit {kind!r}; use 'hw' or 'ordinary'") from None


def endfire_directivity(length_m, f_hz, kind: str = "hw"):
    """Maximum directivity (linear) of an end-fire antenna of length ``length_m`` at ``f_hz``."""
    return _factor(kind) * np.asarray(length_m, dtype=float) / wavelength(f_hz)


def endfire_directivity_dbi(length_m, f_hz, kind: str = "hw"):
    return to_dbi(endfire_directivity(length_m, f_hz, kind))


def aperture_gain(area_m2, f_hz, efficiency: float = 1.0):
    """Broadside aperture gain (linear) ``eta 4 pi A / lambda^2``."""
    return efficiency * 4.0 * np.pi * np.asarray(area_m2, dtype=float) / wavelength(f_hz) ** 2


def aperture_gain_dbi(area_m2, f_hz, efficiency: float = 1.0):
    return to_dbi(aperture_gain(area_m2, f_hz, efficiency))


def min_length(gain_dbi, f_hz, kind: str = "hw"):
    """Shortest end-fire length [m] whose directivity limit reaches ``gain_dbi`` at ``f_hz``.

    It is a lower bound for the gain too (gain <= directivity)."""
    return from_dbi(gain_dbi) * wavelength(f_hz) / _factor(kind)


def feasibility_frequency(gain_dbi, length_m, kind: str = "hw"):
    """Lowest frequency [Hz] at which an end-fire antenna of length ``length_m`` can reach ``gain_dbi``.

    Below it the target is out of reach for any design of that length, whatever the flare, slots or director."""
    return from_dbi(gain_dbi) * C0 / (_factor(kind) * np.asarray(length_m, dtype=float))
