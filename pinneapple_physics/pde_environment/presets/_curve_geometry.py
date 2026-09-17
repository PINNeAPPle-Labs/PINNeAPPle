"""Parametric curve/polygon geometry for the two tag-based presets whose
domain is NOT a box or a circular cylinder: ``axial_compressor_cascade_2d``
(a circular-arc cascade blade) and ``car_external_aero`` (an Ahmed-body-
inspired car silhouette). See ``tag_geometry.py`` for where these are used
and the exact published source cited for each formula/ratio below.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# axial_compressor_cascade_2d: circular-arc camber-line blade.
#
# Source: Dixon & Hall, "Fluid Mechanics and Thermodynamics of
# Turbomachinery", the circular-arc camber line construction for a cascade
# blade. For a circular arc of constant curvature 1/R turning the flow from
# blade angle beta1 (at the leading edge) to beta2 (at the trailing edge)
# over a chord length `chord`:
#   theta_c = beta1 - beta2                      (camber/turning angle)
#   R = chord / (2*sin(theta_c/2))                (camber circle radius --
#       a direct consequence of the chord being the circle's chord for a
#       central angle theta_c: chord = 2*R*sin(theta_c/2))
#   stagger zeta = (beta1 + beta2) / 2            (a circular arc's chord
#       line bisects its turning angle BY CONSTRUCTION -- not a separate
#       empirical assumption, since the tangent deviates from the chord by
#       +-theta_c/2 at each end for a symmetric arc: beta1 = zeta+theta_c/2,
#       beta2 = zeta-theta_c/2 solve exactly to zeta=(beta1+beta2)/2)
# This preset's own params (flow_angle_in_deg, flow_angle_out_deg, chord)
# are read directly as beta1', beta2' (zero-incidence/deviation design-
# condition assumption: metal angles = design flow angles) and `chord` --
# no fabricated blade parameter (thickness, max-camber location, ...) is
# used; the blade is modelled as a zero-thickness cambered plate (a
# standard simplified/inviscid cascade-design idealization), so no interior
# exclusion is needed for collocation sampling.
# ---------------------------------------------------------------------------

def circular_arc_cascade_blade_points(
    n: int,
    rng: np.random.Generator,
    *,
    flow_angle_in_deg: float,
    flow_angle_out_deg: float,
    chord: float,
    x_le: float,
    y_le: float,
) -> np.ndarray:
    beta1 = np.radians(flow_angle_in_deg)
    beta2 = np.radians(flow_angle_out_deg)
    theta_c = beta1 - beta2
    zeta = 0.5 * (beta1 + beta2)
    s = rng.uniform(0.0, 1.0, size=n).astype(np.float64)

    if abs(theta_c) < 1e-8:
        # Degenerate zero-camber case: R -> infinity, the "arc" is a
        # straight flat plate at angle zeta -- the exact theta_c -> 0 limit
        # of the circular-arc formulas below (l'Hopital), not a special case.
        xp = s * chord
        yp = np.zeros_like(s)
    else:
        R = chord / (2.0 * np.sin(theta_c / 2.0))
        phi = theta_c / 2.0 - theta_c * s
        xp = R * (np.sin(theta_c / 2.0) - np.sin(phi))
        yp = R * (np.cos(phi) - np.cos(theta_c / 2.0))

    x = x_le + xp * np.cos(zeta) - yp * np.sin(zeta)
    y = y_le + xp * np.sin(zeta) + yp * np.cos(zeta)
    return np.column_stack([x, y]).astype(np.float32)


# ---------------------------------------------------------------------------
# car_external_aero: Ahmed-body-inspired 2D side-view silhouette.
#
# Source: Ahmed, S.R., Ramm, G., Faltin, G. (1984), "Some Salient Features
# Of The Time-Averaged Ground Vehicle Wake", SAE Technical Paper 840300 --
# the standard reference bluff-body used for automotive external-aero CFD
# validation. Published dimensions of the classic model: overall length
# 1044 mm, width 389 mm, height 288 mm, ground clearance 50 mm, front
# rounding radius ~100 mm, rear slant length 222 mm, slant angle
# configurable (25 deg or 35 deg are the two most commonly tested).
# Used here as RATIOS (clearance/L=50/1044, nose radius/L=100/1044, slant
# length/L=222/1044) applied to this preset's OWN car_length/car_height
# parameters, not the literal millimeter values -- the preset's own size is
# respected, only the real body's proportions and slant angle are borrowed.
# The nose is modelled as two quarter-circle fillets of that radius (front-
# bottom, front-top) joined by a short vertical front face, a closer match
# to the real body's corner rounding than one giant single arc spanning the
# full height would be.
# ---------------------------------------------------------------------------

_AHMED_CLEARANCE_RATIO = 50.0 / 1044.0
_AHMED_NOSE_RADIUS_RATIO = 100.0 / 1044.0
_AHMED_SLANT_LENGTH_RATIO = 222.0 / 1044.0
_AHMED_DEFAULT_SLANT_ANGLE_DEG = 25.0  # the classic, most-cited Ahmed-body configuration


def ahmed_body_polygon(
    car_length: float,
    car_height: float,
    *,
    slant_angle_deg: float = _AHMED_DEFAULT_SLANT_ANGLE_DEG,
    n_arc: int = 24,
    x0: float = 0.0,
) -> np.ndarray:
    """Closed polygon (M, 2) tracing the Ahmed-body-inspired silhouette,
    nose at x=x0, tail at x=x0+car_length, underbody at y=ground_clearance,
    roof at y=car_height. Vertices are ordered counter-clockwise starting
    at the bottom of the front fillet."""
    L, H = float(car_length), float(car_height)
    gc = _AHMED_CLEARANCE_RATIO * L
    nr = _AHMED_NOSE_RADIUS_RATIO * L
    sl = _AHMED_SLANT_LENGTH_RATIO * L
    nr = min(nr, 0.49 * (H - gc))  # guard against a degenerate/too-short body
    slant_rad = np.radians(slant_angle_deg)

    pts = []
    # A: front-bottom fillet, center (nr, gc+nr), phi in [180, 270] deg.
    phi = np.radians(np.linspace(180.0, 270.0, n_arc))
    pts.append(np.column_stack([nr + nr * np.cos(phi), gc + nr + nr * np.sin(phi)]))
    # C: front-top fillet, center (nr, H-nr), phi in [180, 90] deg (decreasing).
    phi = np.radians(np.linspace(180.0, 90.0, n_arc))
    pts.append(np.column_stack([nr + nr * np.cos(phi), H - nr + nr * np.sin(phi)]))
    # D: flat roof.
    pts.append(np.array([[L - sl, H]]))
    # E: slant.
    pts.append(np.array([[L, H - sl * np.tan(slant_rad)]]))
    # F: rear base (down to underbody height) -- implied by the next point.
    pts.append(np.array([[L, gc]]))
    outline = np.concatenate(pts, axis=0).astype(np.float64)
    return (outline + np.array([x0, 0.0]))[:, :]


# ---------------------------------------------------------------------------
# aircraft_wing_aerodynamics: symmetric NACA 4-digit airfoil outline.
#
# Source: Abbott, I.H. & Von Doenhoff, A.E., "Theory of Wing Sections"
# (1959), the standard public NACA 4-digit thickness distribution for a
# ZERO-CAMBER (symmetric) profile, e.g. NACA 0012 for thickness_ratio=0.12
# (this preset's own literature-chosen default -- see
# aircraft_wing_aerodynamics's own docstring and AUDIT_REPORT.md's Third
# follow-up pass for why 0.12/NACA 0012 was picked and why it is a
# LITERATURE default, not something this preset's original parameters
# ever specified):
#
#   y_t(x) = 5*t*(0.2969*sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)^2
#                  + 0.2843*(x/c)^3 - 0.1015*(x/c)^4),  x/c in [0, 1]
#
# This is the CLASSIC coefficient set (-0.1015): the resulting profile has
# a small, finite trailing-edge thickness (the well-known ~0.0021*c NACA
# 4-digit trailing-edge gap), exactly as this public formula predicts --
# not an approximation error (a different, "-0.1036" coefficient variant
# exists specifically to force a zero-thickness trailing edge, but was not
# substituted in since the cited formula uses -0.1015). Cosine spacing in
# x/c is standard practice for resolving the leading-edge curvature, not a
# physics choice.
# ---------------------------------------------------------------------------

def naca4_symmetric_polygon(
    chord: float,
    thickness_ratio: float,
    *,
    n: int = 80,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """Closed polygon (M, 2) tracing a symmetric (zero-camber) NACA 4-digit
    airfoil of the given ``chord`` and ``thickness_ratio`` t (e.g. t=0.12
    for NACA 0012), leading edge at (x0, y0), trailing edge at
    (x0+chord, y0). Vertices run leading edge -> upper surface -> trailing
    edge -> lower surface -> back toward the leading edge (the closing edge
    back to vertex 0 is implicit, same convention as
    :func:`ahmed_body_polygon`)."""
    t = float(thickness_ratio)
    c = float(chord)
    beta = np.linspace(0.0, np.pi, n)
    xc = 0.5 * (1.0 - np.cos(beta))  # cosine spacing, 0..1
    yt = 5.0 * t * (
        0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2 + 0.2843 * xc ** 3 - 0.1015 * xc ** 4
    )
    x = xc * c
    y = yt * c
    upper = np.column_stack([x, y])
    lower = np.column_stack([x[::-1], -y[::-1]])
    # upper[0] == lower[-1] == (0, 0) (leading edge, exact per the formula:
    # yt(0)=0) -- drop lower's final point to avoid a duplicate vertex; the
    # implicit closing edge (last vertex -> vertex 0) reconnects them.
    outline = np.concatenate([upper, lower[:-1]], axis=0).astype(np.float64)
    return outline + np.array([x0, y0])


def polygon_perimeter_sample(vertices: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly (by arc length) along the CLOSED polygon
    whose vertices are given in order (the closing edge back to vertices[0]
    is included)."""
    V = np.asarray(vertices, dtype=np.float64)
    V_next = np.roll(V, -1, axis=0)
    seg = V_next - V
    seg_len = np.sqrt((seg ** 2).sum(axis=1))
    total = seg_len.sum()
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    u = rng.uniform(0.0, total, size=n)
    idx = np.searchsorted(cum, u, side="right") - 1
    idx = np.clip(idx, 0, len(seg_len) - 1)
    t = (u - cum[idx]) / np.maximum(seg_len[idx], 1e-12)
    pts = V[idx] + seg[idx] * t[:, None]
    return pts.astype(np.float32)


def polygon_contains(vertices: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test (closed polygon,
    vertices in order). Returns a bool array, one per row of X (N, 2)."""
    V = np.asarray(vertices, dtype=np.float64)
    V_next = np.roll(V, -1, axis=0)
    X = np.asarray(X, dtype=np.float64)
    px, py = X[:, 0][:, None], X[:, 1][:, None]  # (N, 1)
    x1, y1 = V[None, :, 0], V[None, :, 1]  # (1, M)
    x2, y2 = V_next[None, :, 0], V_next[None, :, 1]

    cond = ((y1 > py) != (y2 > py))
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
    crosses = cond & (px < x_intersect)
    inside = (crosses.sum(axis=1) % 2) == 1
    return inside
