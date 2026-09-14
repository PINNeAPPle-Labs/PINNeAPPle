"""Real, closed-form Helmholtz-resonator tuning equation -- the same
family of equation that determines the resonant "chuff" frequency of a
bottle you blow across, and the standard bass-reflex port tuning
equation used throughout loudspeaker cabinet design (Small/Thiele port
design method; see e.g. Olson, *Acoustical Engineering*):

    A    = pi * r^2                      (port cross-sectional area)
    Leff = L + k_end * r                 (port length + end correction)
    f    = (c / 2*pi) * sqrt(A / (V * Leff))

This is a lumped-element model: it treats the port's air mass as
moving together as a single oscillating plug (valid because a
practical port is acoustically short relative to the wavelengths
involved) -- it does NOT resolve a spatial standing-wave/pressure
profile along the port's length, and callers should not invent one on
top of it.

First used in, and ported back from, PINNeAPPle-apps' ``soundshape_ai``
(a bass-reflex enclosure port design tool).
"""
from __future__ import annotations

import math

SPEED_OF_SOUND_M_S = 343.0  # room-temperature air


def helmholtz_resonant_frequency_hz(
    port_radius_m: float, port_length_m: float, cabinet_volume_l: float,
    end_correction_factor: float = 1.7,
) -> float:
    """The real, closed-form Helmholtz resonant frequency for a port of
    the given radius/length feeding a cabinet of the given internal
    volume. ``end_correction_factor`` defaults to 1.7 (a port flanged
    at both ends -- cabinet wall + interior -- a standard
    loudspeaker-design value); pass a different real value for a
    differently-terminated port geometry."""
    v_m3 = max(cabinet_volume_l, 1e-6) / 1000.0
    a = math.pi * port_radius_m ** 2
    l_eff = port_length_m + end_correction_factor * port_radius_m
    return (SPEED_OF_SOUND_M_S / (2.0 * math.pi)) * math.sqrt(a / (v_m3 * l_eff))


def port_volume_m3(port_radius_m: float, port_length_m: float) -> float:
    """Real port air volume -- used both as a real design constraint
    (a port shouldn't displace too much of the cabinet's own internal
    air) and as a real, physically-meaningful minimization objective in
    multi-objective (Pareto-frontier) port design."""
    return math.pi * port_radius_m ** 2 * port_length_m
