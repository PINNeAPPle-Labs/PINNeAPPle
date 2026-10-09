"""The four reference use cases as specs: steel ladle, pig-iron ladle, BOF converter, RH degasser.

Geometry is derived from the proportions published in the reference twins (a ~230 t steel ladle, ~300 t
pig-iron ladle, ~300 t BOF, RH snorkel vessel); it is *indicative*, not a drawing of a specific plant.
Replace dimensions with the plant's drawings for a real project - nothing else changes.

``get(key)`` returns ``(spec, hotspots)``; hot spots are only used for the synthetic demonstration.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import numpy as np

from .model import VesselSpec, ZoneSpec
from .synthetic import HotSpot

PRESETS: Dict[str, Callable[[], Tuple[VesselSpec, tuple]]] = {}


def _register(fn):
    PRESETS[fn.__name__] = fn
    return fn


def get(key: str):
    if key not in PRESETS:
        raise KeyError(f"unknown preset '{key}'; available: {sorted(PRESETS)}")
    return PRESETS[key]()


def fill_level(rin, y0, y1, rho, mass, n=300) -> float:
    """Bath level for ``mass`` [t] in a vessel of inner radius ``rin(y)`` (frustum integration)."""
    dy = (y1 - y0) / (n - 1)
    v = 0.0
    prev_y = y0
    for i in range(1, n):
        ym = y0 + (i - 0.5) * dy
        r = rin(ym) - 0.04
        v += math.pi * r * r * dy
        if v * rho >= mass:
            return y0 + i * dy
        prev_y = y0 + i * dy
    return prev_y


def _conical_ladle(key, title, H, r_ob, r_ot, wall_t, yf, bands, floor, floor_rows=10):
    """Ladle with a conical shell. ``bands``: (name, label, y0, y1, rows, e0, emin) from top to bottom."""
    rin = lambda y: r_ob + (r_ot - r_ob) * min(max(y / H, 0), 1) - wall_t
    zones = [ZoneSpec(n, lab, ((rin(y0), y0), (rin(y1), y1)), rows, 36, e0, emin, group="wall")
             for n, lab, y0, y1, rows, e0, emin in bands]
    zones.append(ZoneSpec("fundo", "Floor", ((0.0, yf), (rin(yf), yf)), floor_rows, 36, floor[0], floor[1], group="floor"))
    return VesselSpec(key, title, tuple(zones), x_name="heat", notes="Indicative geometry from reference proportions.")


@_register
def steel_ladle():
    # slag-line band fixed at y = 3.15..4.15 m in the reference twin
    spec = _conical_ladle("steel_ladle", "Steel ladle (~230 t)", 4.9, 2.05, 2.30, 0.30, 0.45, [
        ("borda", "Rim / free board", 4.15, 4.85, 7, 150, 50),
        ("linha_escoria", "Slag line", 3.15, 4.15, 10, 200, 60),
        ("zona_metal", "Metal zone - wall", 0.45, 3.15, 27, 180, 60)], floor=(320, 120))
    hs = (HotSpot("linha_escoria", 0.55, None, 0.25, amp=0.75),
          HotSpot("linha_escoria", 0.6, 180, 0.4, 35, amp=0.2),  # spout-side scraping
          HotSpot("zona_metal", 0.05, None, 0.08, amp=0.35),  # wall/floor corner
          HotSpot("zona_metal", 0.1, 180, 0.15, 30, amp=0.15),
          HotSpot("fundo", 0.4, 180, 0.35, 45, amp=0.65),  # alloy jet / impact block
          HotSpot("fundo", 0.55, 120, 0.2, 25, amp=0.25), HotSpot("fundo", 0.55, 240, 0.2, 25, amp=0.25),  # porous plugs
          HotSpot("borda", 0.5, None, 0.5, amp=0.1))
    return spec, hs


@_register
def pig_iron_ladle():
    H, r_ob, r_ot, wt, yf, y_top = 5.2, 2.22, 2.55, 0.32, 0.50, 5.15
    rin = lambda y: r_ob + (r_ot - r_ob) * min(max(y / H, 0), 1) - wt
    slag_y = fill_level(rin, yf, y_top, 6.9, 300) + 0.08
    le0, le1 = slag_y - 0.45, slag_y + 0.5
    spec = _conical_ladle("pig_iron_ladle", "Pig-iron ladle (~300 t)", H, r_ob, r_ot, wt, yf, [
        ("borda", "Rim and spout", le1, y_top, max(3, round((y_top - le1) * 10)), 230, 70),
        ("linha_escoria", "Slag line", le0, le1, 10, 250, 80),
        ("zona_metal", "Metal zone - wall", yf, le0, round((le0 - yf) * 10), 230, 70)], floor=(350, 120))
    hs = (HotSpot("linha_escoria", 0.45, None, 0.2, amp=0.6),
          HotSpot("linha_escoria", 0.5, 180, 0.4, 35, amp=0.25),  # slag dragged to the spout
          HotSpot("zona_metal", 0.75, None, 0.12, amp=0.25),  # KR rotor vortex band
          HotSpot("zona_metal", 0.4, 90, 0.2, 30, amp=0.25),  # torpedo jet splash
          HotSpot("fundo", 0.35, 90, 0.3, 40, amp=0.8),  # torpedo jet impact block
          HotSpot("borda", 0.6, 180, 0.4, 30, amp=0.7))  # spout
    return spec, hs


@_register
def bof_converter():
    R_IN, Y_CAP, BAR_H, CONE_H, R_MOUTH, NECK_H = 3.45, 1.0, 4.8, 2.4, 2.0, 0.55
    Y_JUN = Y_CAP
    Y_C0, Y_C1 = Y_JUN + BAR_H, Y_JUN + BAR_H + CONE_H
    pts = [(R_IN, Y_JUN), (R_IN, Y_C0), (R_MOUTH, Y_C1), (R_MOUTH, Y_C1 + NECK_H)]
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    LC = math.hypot(R_IN - R_MOUTH, CONE_H)
    U_C1 = BAR_H + LC - 0.4

    def seg(u0, u1):
        us = np.linspace(u0, u1, 24)
        return tuple((float(np.interp(u, arc, [p[0] for p in pts])), float(np.interp(u, arc, [p[1] for p in pts]))) for u in us)

    phi = np.linspace(0, math.pi / 2, 24)  # dished (elliptical) floor, centre -> barrel junction
    floor = tuple((float(R_IN * math.sin(p)), float(Y_JUN - Y_CAP * math.cos(p))) for p in phi)
    zones = (
        ZoneSpec("fundo", "Floor - plugs and impact", floor, 12, 36, 800, 300, group="floor"),
        ZoneSpec("barril_inferior", "Lower barrel - bath zone", seg(0, 1.4), 14, 36, 750, 250, group="wall"),
        ZoneSpec("linha_escoria", "Slag line - mid barrel", seg(1.4, 3.4), 20, 36, 800, 250, group="wall"),
        ZoneSpec("barril_superior", "Upper barrel", seg(3.4, BAR_H), 14, 36, 700, 250, group="wall"),
        ZoneSpec("cone", "Cone - tap-hole side", seg(BAR_H, U_C1), 17, 36, 650, 220, group="wall"),
        ZoneSpec("boca", "Mouth - chin", seg(U_C1, float(arc[-1])), 8, 36, 550, 200, group="wall"),
    )
    spec = VesselSpec("bof_converter", "BOF converter (~300 t)", zones, x_name="heat",
                      notes="Indicative geometry from reference proportions.")
    hs = (HotSpot("fundo", 0.9, 180, 0.2, 45, amp=0.55), HotSpot("fundo", 0.5, 90, 0.25, 20, amp=0.25),
          HotSpot("fundo", 0.5, 270, 0.25, 20, amp=0.25),
          HotSpot("linha_escoria", 0.5, None, 0.3, amp=0.55), HotSpot("linha_escoria", 0.5, 90, 0.5, 30, amp=0.2),  # trunnions
          HotSpot("barril_inferior", 0.2, None, 0.3, amp=0.3),
          HotSpot("cone", 0.7, 0, 0.3, 35, amp=0.45),  # tap-hole side
          HotSpot("boca", 0.6, None, 0.3, amp=0.2))
    return spec, hs


@_register
def rh_degasser():
    LEG_X, LEG_R_OUT = 1.05, 0.52
    LEG_TIP, VESSEL_FLOOR_OUT = -1.6, 1.25
    rl = LEG_R_OUT + 0.012
    zones = (
        ZoneSpec("vaso_inferior", "Lower vessel - wall", ((1.876, 1.66), (1.876, 3.75), (1.726, 3.95)), 12, 36, 250, 100, group="wall"),
        ZoneSpec("vaso_superior", "Upper vessel - wall and cone", ((1.726, 3.95), (1.726, 5.85), (1.296, 6.45), (0.626, 7.05), (0.626, 7.6)), 14, 36, 250, 100, group="wall"),
        ZoneSpec("piso", "Lower vessel - floor", ((0.0, 1.66), (1.876, 1.66)), 4, 36, 250, 100, group="floor"),
        ZoneSpec("perna_ascendente", "Up-snorkel", ((rl, LEG_TIP), (rl, VESSEL_FLOOR_OUT)), 8, 24, 250, 100, center=(-LEG_X, 0.0), group="leg"),
        ZoneSpec("perna_descendente", "Down-snorkel", ((rl, LEG_TIP), (rl, VESSEL_FLOOR_OUT)), 8, 24, 250, 100, center=(LEG_X, 0.0), theta0=math.pi, group="leg"),
    )
    spec = VesselSpec("rh_degasser", "RH degasser (snorkel vessel)", zones, x_name="heat", notes="Indicative geometry from reference proportions.")
    hs = (HotSpot("vaso_inferior", 0.18, None, 0.2, amp=0.5), HotSpot("vaso_inferior", 0.62, None, 0.18, amp=0.2),
          HotSpot("vaso_inferior", 0.2, 0, 0.35, 50, amp=0.2),
          HotSpot("vaso_superior", 0.03, None, 0.12, amp=0.2), HotSpot("vaso_superior", 0.88, None, 0.1, amp=0.3),
          HotSpot("piso", 0.5, 0, 0.5, 60, amp=0.5), HotSpot("piso", 0.5, 180, 0.5, 60, amp=0.5),
          HotSpot("perna_ascendente", 0.08, None, 0.1, amp=0.7), HotSpot("perna_ascendente", 0.25, None, 0.12, amp=0.4),
          HotSpot("perna_descendente", 0.08, None, 0.1, amp=0.7), HotSpot("perna_descendente", 0.25, None, 0.12, amp=0.4))
    return spec, hs


@_register
def oxyred_reactor():
    """Oxy-Red 2 pilot reactor (autoreduction / melting-reduction): shaft, tuyere zone, crucible, hearth.

    Sector 0 is the tap hole; the tuyeres sit between tap holes, evenly spaced. Wear limit 300 mm of a 450 mm lining.
    """
    n_tuy = 8
    zones = (
        ZoneSpec("cuba_superior", "Upper shaft - top", ((1.71, 7.5), (1.62, 11.0), (0.72, 11.85)), 14, 36, 450, 150, group="wall"),
        ZoneSpec("cuba_inferior", "Lower shaft - reduction", ((1.80, 4.0), (1.71, 7.5)), 12, 36, 450, 150, group="wall"),
        ZoneSpec("ventaneiras", "Tuyere zone", ((1.80, 2.6), (1.80, 4.0)), 6, 32, 450, 150, group="wall"),
        ZoneSpec("cadinho_parede", "Crucible - wall", ((1.80, 0.9), (1.80, 2.6)), 10, 36, 450, 150, group="wall"),
        ZoneSpec("soleira", "Crucible - hearth", ((0.0, 0.9), (1.80, 0.9)), 5, 36, 450, 150, group="floor"),
    )
    spec = VesselSpec("oxyred_reactor", "Oxy-Red reactor", zones, x_name="heat",
                      notes="Indicative geometry from the Oxy-Red twin proportions (hot face 3.6 m).")
    tuy = [(k + 0.5) / n_tuy * 360 for k in range(n_tuy)]
    hs = ((HotSpot("cadinho_parede", 0.6, None, 0.11, amp=0.45),            # slag line
           HotSpot("cadinho_parede", 0.17, 0, 0.18, 17, amp=0.55),          # tap hole
           HotSpot("cadinho_parede", 0.03, None, 0.08, amp=0.15)) +
          tuple(HotSpot("ventaneiras", 0.45, t, 0.24, 8, amp=0.5) for t in tuy) +      # O2 jets / post-combustion
          (HotSpot("ventaneiras", 0.95, None, 0.18, amp=0.1),
           HotSpot("cuba_inferior", 0.04, None, 0.2, amp=0.2),
           HotSpot("cuba_superior", 0.655, None, 0.07, amp=0.3),             # abrasion at the stock level
           HotSpot("soleira", 0.92, None, 0.14, amp=0.35), HotSpot("soleira", 0.9, 0, 0.22, 28, amp=0.3)))
    return spec, hs
