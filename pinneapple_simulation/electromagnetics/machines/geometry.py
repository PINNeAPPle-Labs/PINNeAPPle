"""Parametric cross-sections of radial-flux machines: V-shaped interior magnets and round-bottom slots.

A machine is described per *sector*: one rotor pole and one stator slot pitch, each a list of named closed
polygons inside an annular sector (straight edges; arcs are discretised by :func:`arc`). The mesher
(:mod:`.mesher`) meshes one sector with every polygon edge as a mesh edge and copies it around the circle,
so the regions of the mesh are the polygons exactly, and every pole and every slot has the same mesh.

Parametrisations (lengths in metres, angles in radians):

* :class:`VRotor` -- a lamination with one V of two rectangular magnets per pole, the parametrisation of
  pyleecan's ``HoleM50`` (Apache-2.0; P. Bonneel, E. Devillers et al.): opening ``W0`` and depth ``H0`` of the
  V, tangential-bridge thickness ``H1``, magnet thickness ``H3`` and width ``W4``, inter-pole rib ``W3``,
  centre-post width ``W1``. The V angle follows from them (:attr:`VRotor.v_angle`).
* :class:`RoundBottomSlots` -- open trapezoidal slots with a round bottom (pyleecan ``SlotW11``): opening
  ``W0`` x ``H0``, wedge ``H1``, body width ``W1`` at the top and ``W2`` at the bottom, depth ``H2``,
  bottom fillet radius ``R1``.

:func:`prius_2004` returns the 2004 Toyota Prius traction motor with the dimensions of pyleecan's
``Toyota_Prius.json`` (48 slots, 8 poles, 83.82 mm stack).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Polygon", "SectorGeometry", "arc", "VRotor", "RoundBottomSlots", "polygon_area", "prius_2004",
           "MachineGeometry", "geometry_errors", "ROTOR_VARIABLES"]

# design ranges of the V-rotor variables (name, lower, upper), metres, around the Prius values
# W4 18.9, H3 6.5, H0 10.96, W0 42, H1 1.5 mm
ROTOR_VARIABLES = (("W4", 0.0140, 0.0220), ("H3", 0.0045, 0.0075), ("H0", 0.0085, 0.0135), ("W0", 0.0360, 0.0450),
                   ("H1", 0.0010, 0.0020))

TWO_PI = 2.0 * math.pi


def polygon_area(xy: np.ndarray) -> float:
    """Signed area of a closed polygon (counter-clockwise positive)."""
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def arc(center: complex, radius: float, a0: float, a1: float, h: float, even: bool = False) -> np.ndarray:
    """Points of a circular arc from angle ``a0`` to ``a1`` (both included), spacing at most ``h``; ``even``: an even
    number of intervals, so the mid-angle is a vertex (arcs symmetric about a mirror axis)."""
    n = max(1, int(math.ceil(abs(a1 - a0) * radius / h)))
    if even:
        n += n % 2
    t = np.linspace(a0, a1, n + 1)
    return np.column_stack([center.real + radius * np.cos(t), center.imag + radius * np.sin(t)])


@dataclass
class Polygon:
    """A named closed polygon (vertices counter-clockwise or clockwise, last vertex not repeated).

    ``magnetization``: for magnets, the angle (in the sector frame) of the remanence of a north pole."""

    name: str
    xy: np.ndarray
    magnetization: float | None = None

    @property
    def area(self) -> float:
        return abs(polygon_area(self.xy))


@dataclass
class SectorGeometry:
    """Annular sector ``r_in < r < r_out``, ``-span/2 < phi < span/2``, holding polygons (later ones win where
    they overlap) in a background region ``fill``. ``arc_nodes_in``/``arc_nodes_out``: extra angles that must
    be nodes on the inner/outer arc (corners of polygons that touch the arcs)."""

    r_in: float
    r_out: float
    span: float
    fill: str
    polygons: list[Polygon] = field(default_factory=list)
    arc_nodes_in: list[float] = field(default_factory=list)
    arc_nodes_out: list[float] = field(default_factory=list)


def _rot(z, a):
    return z * complex(math.cos(a), math.sin(a))


def _pts(*zs) -> np.ndarray:
    """Vertices from complex points, dropping repeated consecutive points (coincident corners when a
    parameter is zero)."""
    out = []
    for z in zs:
        if not out or abs(z - out[-1]) > 1e-12:
            out.append(z)
    if len(out) > 1 and abs(out[0] - out[-1]) <= 1e-12:
        out.pop()
    return np.array([[z.real, z.imag] for z in out])


@dataclass
class VRotor:
    """Interior-PM rotor with one V of two magnets per pole (pyleecan ``HoleM50`` parametrisation)."""

    poles: int = 8
    r_out: float = 0.0802
    r_in: float = 0.05532
    H0: float = 0.01096  # depth of the V apex (centre post top) below the rotor surface
    H1: float = 0.0015   # tangential bridge (iron between the pocket and the air gap)
    H2: float = 0.001    # pocket clearance at the magnet ends
    H3: float = 0.0065   # magnet thickness
    W0: float = 0.042    # chord of the outer pocket ends
    W1: float = 0.0      # centre-post width
    W2: float = 0.0      # offset of the magnet along the pocket
    W3: float = 0.014    # inter-pole rib at radius r_out - H1
    W4: float = 0.0189   # magnet width
    magnet_segments: int = 1

    @property
    def pole_pitch(self) -> float:
        return TWO_PI / self.poles

    def _points(self) -> dict:
        """Corner points of the lower half of the V in the pole frame (pole axis on +x)."""
        R = self.r_out
        hssp = math.pi / self.poles
        alpham = 2.0 * math.asin(self.W0 / (2.0 * (R - self.H1)))
        harc = (R - self.H1) * (1.0 - math.cos(alpham / 2.0))
        gammam = math.atan((self.H0 - self.H1 - harc) / (self.W0 / 2.0 - self.W1 / 2.0))
        x78 = (self.H3 - self.H2) / math.cos(gammam)
        Z9 = complex(R - harc - self.H1, -self.W0 / 2.0)
        Z8 = complex(R - self.H0, -self.W1 / 2.0)
        Z7 = complex(R - self.H0 - x78, -self.W1 / 2.0)
        Z1 = (R - self.H1) * complex(math.cos(-hssp + math.asin(self.W3 / (2 * (R - self.H1)))),
                                     math.sin(-hssp + math.asin(self.W3 / (2 * (R - self.H1)))))
        rot = complex(math.cos(math.atan2((Z9 - Z8).imag, (Z9 - Z8).real)),
                      math.sin(math.atan2((Z9 - Z8).imag, (Z9 - Z8).real)))
        loc = {"Z8b": complex(self.W2, 0.0), "Z8c": complex(self.W2 + self.W4, 0.0),
               "Z5": complex(self.W2, -self.H3), "Z4": complex(self.W2 + self.W4, -self.H3)}
        loc["Z6"] = loc["Z5"] + 1j * self.H2
        loc["Z3"] = loc["Z4"] + 1j * self.H2
        P = {k: v * rot + Z8 for k, v in loc.items()}
        # Z2: on the line through Z3 and Z6, displaced from Z1 parallel to the inter-pole axis
        Z1r, Z3r, Z6r = _rot(Z1, hssp), _rot(P["Z3"], hssp), _rot(P["Z6"], hssp)
        d = Z6r - Z3r
        x = ((Z1r - Z3r) / d).imag / (1.0 / d).imag
        Z2 = _rot(Z1r - x, -hssp)
        P.update(Z1=Z1, Z2=Z2, Z7=Z7, Z8=Z8, Z9=Z9)
        return P

    @property
    def v_angle(self) -> float:
        """Opening angle of the V between the two magnets (radians; pi is a flat, tangential magnet)."""
        P = self._points()
        a = math.atan2((P["Z8c"] - P["Z8b"]).imag, (P["Z8c"] - P["Z8b"]).real)
        return 2.0 * abs(a)

    def sector(self, h: float) -> SectorGeometry:
        """One pole centred on +x: pockets (air) and magnets, magnetised normal to their wide face, outwards."""
        P = self._points()
        a9 = math.atan2(P["Z9"].imag, P["Z9"].real)
        a1 = math.atan2(P["Z1"].imag, P["Z1"].real)
        bridge = [complex(*p) for p in arc(0j, self.r_out - self.H1, a9, a1, h)[1:-1]]
        outer = [P["Z1"], P["Z2"], P["Z3"], P["Z8c"], P["Z9"], *bridge]
        magnet = [P["Z3"], P["Z4"], P["Z5"], P["Z6"], P["Z8b"], P["Z8c"]]
        t = P["Z8c"] - P["Z8b"]
        n = t * 1j / abs(t)
        if n.real < 0:
            n = -n
        ang = math.atan2(n.imag, n.real)
        polys = []
        if self.W1 == 0.0:  # the two inner pockets meet on the pole axis
            polys.append(Polygon("pocket_c", _pts(P["Z8b"], P["Z6"], P["Z7"], P["Z6"].conjugate(),
                                                 P["Z8b"].conjugate(), P["Z8"])))
        for side, sg in (("a", 1), ("b", -1)):
            f = (lambda z: z) if sg > 0 else (lambda z: z.conjugate())
            polys.append(Polygon(f"pocket_{side}", _pts(*map(f, outer))))
            if self.W1 > 0.0:
                polys.append(Polygon(f"pocket_{side}2", _pts(*map(f, [P["Z6"], P["Z7"], P["Z8"], P["Z8b"]]))))
            for k, seg in enumerate(_split_magnet([f(z) for z in magnet], self.magnet_segments)):
                polys.append(Polygon(f"magnet_{side}{k}", seg, magnetization=sg * ang))
        return SectorGeometry(self.r_in, self.r_out, self.pole_pitch, "rotor_iron", polys)


def _split_magnet(z: list[complex], n: int) -> list[np.ndarray]:
    """The magnet polygon (Z3, Z4, Z5, Z6, Z8b, Z8c) as one piece, or ``n`` electrically insulated pieces
    along its width."""
    if n <= 1:
        return [_pts(*z)]
    Z3, Z4, Z5, Z6, Z8b, Z8c = z
    lerp = lambda a, b, s: a + (b - a) * s  # noqa: E731
    return [_pts(lerp(Z5, Z4, k / n), lerp(Z5, Z4, (k + 1) / n), lerp(Z8b, Z8c, (k + 1) / n), lerp(Z8b, Z8c, k / n))
            for k in range(n)]


@dataclass
class RoundBottomSlots:
    """Stator lamination with open slots: opening, straight-sided body, semicircular or filleted bottom."""

    slots: int = 48
    r_in: float = 0.08095  # bore
    r_out: float = 0.13462
    W0: float = 0.00193
    H0: float = 0.001
    H1: float = 0.0
    W1: float = 0.005
    W2: float = 0.008
    H2: float = 0.0333
    R1: float = 0.004

    @property
    def slot_pitch(self) -> float:
        return TWO_PI / self.slots

    def _bottom(self, top: float, h: float) -> np.ndarray:
        x4 = top + self.H2 - self.R1
        if abs(self.R1 - self.W2 / 2) < 1e-12:  # semicircular bottom
            return arc(complex(x4, 0.0), self.R1, -math.pi / 2, math.pi / 2, h, even=True)
        lo = arc(complex(x4, -self.W2 / 2 + self.R1), self.R1, -math.pi / 2, 0.0, h)
        hi = arc(complex(x4, self.W2 / 2 - self.R1), self.R1, 0.0, math.pi / 2, h)
        return np.vstack([lo, hi])

    def sector(self, h: float) -> SectorGeometry:
        """One slot pitch centred on the slot axis (+x): regions ``slot`` (winding), ``opening`` and ``wedge``
        (air), in the background ``stator_iron``."""
        Rbo = self.r_in
        al = math.asin(self.W0 / (2 * Rbo))
        Z1 = Rbo * complex(math.cos(al), -math.sin(al))
        x2 = Z1.real + self.H0  # top of the opening
        top = x2 + self.H1  # bottom of the wedge = top of the winding
        bore = arc(0j, Rbo, al, -al, h, even=True)[1:-1]
        polys = [Polygon("opening", np.vstack([_pts(Z1, complex(x2, -self.W0 / 2), complex(x2, self.W0 / 2),
                                                    Z1.conjugate()), bore]))]
        body = [complex(top, -self.W1 / 2)]
        if self.H1 > 0:
            polys.append(Polygon("wedge", _pts(complex(x2, -self.W0 / 2), complex(top, -self.W1 / 2),
                                               complex(top, self.W1 / 2), complex(x2, self.W0 / 2))))
        elif self.W1 > self.W0:  # the opening's top edge is part of the slot boundary: keep its corners
            body = [complex(top, -self.W0 / 2)] + body
        winding = np.vstack([_pts(*body), self._bottom(top, h), _pts(*[z.conjugate() for z in reversed(body)])])
        polys.append(Polygon("slot", winding))
        return SectorGeometry(Rbo, self.r_out, self.slot_pitch, "stator_iron", polys, arc_nodes_in=[-al, al])

    def slot_area(self) -> float:
        """Exact area of the winding region: trapezoid plus the rounded bottom."""
        trap = 0.5 * (self.W1 + self.W2) * (self.H2 - self.R1)
        return trap + self.R1 * (self.W2 - 2 * self.R1) + 0.5 * math.pi * self.R1 ** 2


@dataclass
class MachineGeometry:
    """Rotor + stator + air gap. ``gap_layers`` rings split the gap (moving band in the middle)."""

    rotor: VRotor
    stator: RoundBottomSlots
    length: float = 0.08382
    stacking_factor: float = 0.95

    @property
    def airgap(self) -> float:
        return self.stator.r_in - self.rotor.r_out


def prius_2004() -> MachineGeometry:
    """2004 Toyota Prius traction motor: dimensions of pyleecan's ``Toyota_Prius.json`` (pyleecan 1.4.2,
    Apache-2.0), which models the machine of the ORNL evaluation reports (ORNL/TM-2004/185, ORNL/TM-2006/423)."""
    return MachineGeometry(VRotor(), RoundBottomSlots(), length=0.08382, stacking_factor=0.95)


def _segments_cross(p, q, r, s, tol: float) -> bool:
    """Proper intersection of segments pq and rs (touching at shared end points does not count)."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    if min(np.linalg.norm(a - b) for a in (p, q) for b in (r, s)) <= tol:
        return False
    d1, d2, d3, d4 = orient(r, s, p), orient(r, s, q), orient(p, q, r), orient(p, q, s)
    return (d1 * d2 < -tol ** 2) and (d3 * d4 < -tol ** 2)


def geometry_errors(sector: SectorGeometry, tol: float = 1e-9) -> list[str]:
    """Reasons why a sector is not a valid cross-section (empty when valid): polygons that self-intersect, cross each
    other, overlap (a vertex strictly inside another polygon), or leave the annular sector."""
    from matplotlib.path import Path

    errs = []
    half = sector.span / 2.0
    edges = []
    for k, pg in enumerate(sector.polygons):
        xy = pg.xy
        r = np.hypot(xy[:, 0], xy[:, 1])
        a = np.arctan2(xy[:, 1], xy[:, 0])
        if np.any(r < sector.r_in - tol) or np.any(r > sector.r_out + tol) or np.any(np.abs(a) > half + 1e-12):
            errs.append(f"{pg.name} leaves the sector")
        if polygon_area(xy) == 0.0:
            errs.append(f"{pg.name} has zero area")
        edges += [(k, xy[i], xy[(i + 1) % len(xy)]) for i in range(len(xy))]
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if _segments_cross(edges[i][1], edges[i][2], edges[j][1], edges[j][2], tol):
                a, b = sector.polygons[edges[i][0]].name, sector.polygons[edges[j][0]].name
                errs.append(f"{a} and {b} cross" if a != b else f"{a} intersects itself")
    for i, pa in enumerate(sector.polygons):
        path = Path(pa.xy)
        for j, pb in enumerate(sector.polygons):
            if i == j:
                continue
            inside = path.contains_points(pb.xy) & (_distance_to_boundary(pb.xy, pa.xy) > 1e3 * tol)
            if np.any(inside):
                errs.append(f"{pb.name} overlaps {pa.name}")
    return sorted(set(errs))


def _distance_to_boundary(pts: np.ndarray, xy: np.ndarray) -> np.ndarray:
    a, b = xy, np.roll(xy, -1, axis=0)
    ab = b - a
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-300), 0, 1)
    d = pts[:, None, :] - (a[None] + t[..., None] * ab[None])
    return np.sqrt((d * d).sum(-1)).min(1)
