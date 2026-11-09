"""Bodies of the VVD solver: closed polygons that move rigidly, and the generators the cases are built from.

A body is a closed polygon in its own frame plus a :class:`Motion`. Each time step the polygon is cut into straight
*panels*: the segment midpoints are the collocation points of the source sheet (no through flow) and of the vorticity
creation (no slip), and the panel normals of the wall-diffusion drift.

Every generator returns a counter-clockwise polygon with outward normals, so the solver never has to guess an
orientation.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DEG",
    "Assembly",
    "Body",
    "Motion",
    "PanelSet",
    "cylinder",
    "deploying_spoiler",
    "ellipse",
    "fixed",
    "hinge",
    "inside_polygon",
    "naca4",
    "naca0012",
    "plate",
    "polygon",
    "resample_polygon",
    "rotating",
    "translating",
]

DEG = np.pi / 180.0


# --------------------------------------------------------------------------------------------- rigid motion
@dataclass(frozen=True)
class Motion:
    """Rigid motion ``x(t) = center(t) + R(angle(t)) x_local`` with its first derivative.

    ``center`` and ``angle`` are callables of time; ``velocity`` and ``omega`` are their derivatives (both default
    to zero, i.e. a fixed body).
    """

    center: Callable[[float], np.ndarray] = lambda t: np.zeros(2)
    angle: Callable[[float], float] = lambda t: 0.0
    velocity: Callable[[float], np.ndarray] = lambda t: np.zeros(2)
    omega: Callable[[float], float] = lambda t: 0.0
    label: str = "fixed"

    def ring(self, t: float) -> tuple[np.ndarray, float]:
        return np.asarray(self.center(t), dtype=float), float(self.angle(t))

    def rates(self, t: float) -> tuple[np.ndarray, float]:
        return np.asarray(self.velocity(t), dtype=float), float(self.omega(t))


def fixed() -> Motion:
    """A body that stays where it is."""
    return Motion()


def translating(velocity: Sequence[float], center0: Sequence[float] = (0.0, 0.0), angle: float = 0.0,
                omega: float = 0.0) -> Motion:
    """Constant translation (and optional constant rotation) from ``center0``."""
    v = np.asarray(velocity, dtype=float)
    c0 = np.asarray(center0, dtype=float)
    return Motion(center=lambda t: c0 + v * t, angle=lambda t: angle + omega * t, velocity=lambda t: v,
                  omega=lambda t: omega, label="translating")


def rotating(center: Sequence[float], rate: float, angle0: float = 0.0) -> Motion:
    """Constant rotation about a fixed point."""
    c = np.asarray(center, dtype=float)
    return Motion(center=lambda t: c + np.zeros(2), angle=lambda t: angle0 + rate * t,
                  velocity=lambda t: np.zeros(2), omega=lambda t: rate, label="rotating")


def _ramp(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * s)                  # smooth (zero rate at both ends)


def hinge(pivot: Sequence[float], angle_of_t: Callable[[float], float],
          rate_of_t: Callable[[float], float] | None = None) -> Motion:
    """A body hinged at ``pivot`` (world coordinates at ``t = 0``), turning by ``angle_of_t``.

    The rate is the analytic derivative when given, otherwise a central difference of ``angle_of_t``.
    """
    p = np.asarray(pivot, dtype=float)
    if rate_of_t is None:
        h = 1e-6
        rate_of_t = lambda t: (angle_of_t(t + h) - angle_of_t(t - h)) / (2 * h)   # noqa: E731
    return Motion(center=lambda t: p + np.zeros(2), angle=angle_of_t, velocity=lambda t: np.zeros(2),
                  omega=rate_of_t, label="hinge")


def deploy(from_deg: float, to_deg: float, t_start: float, duration: float) -> tuple[Any, Any]:
    """Angle and rate of a smooth ramp between two angles (radians returned).

    Returns ``(angle_of_t, rate_of_t)``; both are zero outside ``[t_start, t_start + duration]``.
    """
    a0, a1 = from_deg * DEG, to_deg * DEG
    duration = max(float(duration), 1e-12)

    def angle(t: float) -> float:
        return a0 + (a1 - a0) * _ramp((t - t_start) / duration)

    def rate(t: float) -> float:
        s = (t - t_start) / duration
        if s <= 0.0 or s >= 1.0:
            return 0.0
        return (a1 - a0) * 0.5 * np.pi * np.sin(np.pi * s) / duration

    return angle, rate


# ---------------------------------------------------------------------------------------------------- panels
@dataclass
class PanelSet:
    """The panels of one or more bodies at one instant, in world coordinates."""

    a: np.ndarray                  # (m, 2) start of each panel
    b: np.ndarray                  # (m, 2) end of each panel
    mid: np.ndarray                # (m, 2) midpoint (collocation point)
    tangent: np.ndarray            # (m, 2) unit tangent along a -> b
    normal: np.ndarray             # (m, 2) unit outward normal
    length: np.ndarray             # (m,)
    velocity: np.ndarray           # (m, 2) surface velocity at the midpoint
    body: np.ndarray               # (m,) body index of each panel
    body_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.length)

    @property
    def vector(self) -> np.ndarray:
        """Outward normal scaled by the panel length (the ``dS`` of the wall-diffusion kernel)."""
        return self.normal * self.length[:, None]

    def of_body(self, index: int) -> np.ndarray:
        return self.body == index

    def surface_velocity_at(self, points: np.ndarray) -> np.ndarray:
        """Velocity of the nearest panel's surface point, for points that sit on the wall."""
        dist = ((np.atleast_2d(points)[:, None, :] - self.mid[None, :, :]) ** 2).sum(-1)
        return self.velocity[np.argmin(dist, axis=1)]


def _signed_area(v: np.ndarray) -> float:
    x, y = v[:, 0], v[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_ccw(v: np.ndarray) -> np.ndarray:
    return v if _signed_area(v) > 0 else v[::-1]


def polygon_area(v: np.ndarray) -> float:
    """Area enclosed by a closed polygon (positive for counter-clockwise)."""
    return abs(_signed_area(np.asarray(v, dtype=float)))


def polygon_centroid(v: np.ndarray) -> np.ndarray:
    """Area centroid of a closed polygon."""
    v = np.asarray(v, dtype=float)
    x, y = v[:, 0], v[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum()
    if abs(area) < 1e-14:
        return v.mean(axis=0)
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (3.0 * area)


class Body:
    """A closed polygonal body that moves rigidly.

    ``vertices`` are in the body's own frame (the frame :class:`Motion` maps to the world); the polygon is closed
    automatically and made counter-clockwise.
    """

    def __init__(self, vertices: np.ndarray, motion: Motion | None = None, name: str = "body",
                 thin: bool = False):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("a body needs at least three vertices of shape (n, 2)")
        if np.allclose(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        self.vertices = _ensure_ccw(vertices)
        self.motion = motion or fixed()
        self.name = name
        self.thin = thin
        self.area = polygon_area(self.vertices)
        self.local_centroid = polygon_centroid(self.vertices)

    # -- geometry ---------------------------------------------------------------------------------------
    def world_vertices(self, t: float) -> np.ndarray:
        center, angle = self.motion.ring(t)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        return self.vertices @ rot.T + center

    def panels(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(a, b, surface velocity at the midpoints)`` in world coordinates."""
        v = self.world_vertices(t)
        a = v
        b = np.roll(v, -1, axis=0)
        mid = 0.5 * (a + b)
        dcenter, omega = self.motion.rates(t)
        center, _ = self.motion.ring(t)
        vel = dcenter[None, :] + omega * np.stack([-(mid[:, 1] - center[1]), mid[:, 0] - center[0]], axis=1)
        return a, b, vel

    def closed(self) -> bool:
        return True

    def __repr__(self) -> str:                                       # pragma: no cover - debugging aid
        return f"Body({self.name!r}, {len(self.vertices)} vertices, area={self.area:.4g})"


class Assembly:
    """A set of bodies that share one flow field."""

    def __init__(self, bodies: Sequence[Body] | None = None):
        self.bodies = list(bodies or [])

    def add(self, body: Body) -> Body:
        self.bodies.append(body)
        return body

    def __len__(self) -> int:
        return len(self.bodies)

    def __iter__(self):
        return iter(self.bodies)

    def panels(self, t: float) -> PanelSet:
        a_all, b_all, v_all, owner = [], [], [], []
        for index, body in enumerate(self.bodies):
            a, b, vel = body.panels(t)
            a_all.append(a)
            b_all.append(b)
            v_all.append(vel)
            owner.append(np.full(len(a), index))
        if not a_all:
            empty = np.zeros((0, 2))
            return PanelSet(empty, empty, empty, empty, empty, np.zeros(0), empty, np.zeros(0, dtype=int))
        a = np.vstack(a_all)
        b = np.vstack(b_all)
        vel = np.vstack(v_all)
        mid = 0.5 * (a + b)
        d = b - a
        length = np.hypot(d[:, 0], d[:, 1])
        tangent = d / length[:, None]
        # counter-clockwise polygons: outward normal is -rotl(tangent) = (ty, -tx)
        normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
        return PanelSet(a=a, b=b, mid=mid, tangent=tangent, normal=normal, length=length, velocity=vel,
                        body=np.concatenate(owner), body_names=[b.name for b in self.bodies])

    def inside(self, points: np.ndarray, t: float) -> np.ndarray:
        """Which points are inside any body (ray casting, one polygon at a time)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        flag = np.zeros(len(points), dtype=bool)
        for body in self.bodies:
            flag |= inside_polygon(points, body.world_vertices(t))
        return flag

    def area(self, t: float | None = None) -> float:
        return float(sum(b.area for b in self.bodies))


def inside_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Even-odd (ray casting) point-in-polygon test, vectorised over the points."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    v = np.asarray(vertices, dtype=float)
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    xj, yj = v[-1]
    for (xi, yi) in v:
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_cross)
        xj, yj = xi, yi
    return inside


# ----------------------------------------------------------------------------------------------- generators
def polygon(vertices: np.ndarray, name: str = "polygon", motion: Motion | None = None,
            n_refine: int = 0) -> Body:
    """A body from explicit vertices (optionally subdivided into ``n_refine`` extra points per edge)."""
    v = np.asarray(vertices, dtype=float)
    if n_refine > 0:
        v = refine_polygon(v, n_refine)
    return Body(v, motion=motion, name=name)


def resample_polygon(vertices: np.ndarray, n: int) -> np.ndarray:
    """``n`` points equally spaced in arclength around a closed polygon.

    Cosine spacing (and any CAD curve) leaves panels next to a corner that are far shorter than the wall can
    resolve; resampling makes every panel comparable, which is what the stability rule of the method wants.
    """
    v = np.asarray(vertices, dtype=float)
    closed = np.vstack([v, v[:1]])
    seg = np.diff(closed, axis=0)
    dist = np.hypot(seg[:, 0], seg[:, 1])
    s_cum = np.concatenate([[0.0], np.cumsum(dist)])
    total = s_cum[-1]
    targets = np.linspace(0.0, total, int(n), endpoint=False)
    x = np.interp(targets, s_cum, closed[:, 0])
    y = np.interp(targets, s_cum, closed[:, 1])
    return np.column_stack([x, y])


def refine_polygon(vertices: np.ndarray, n_refine: int) -> np.ndarray:
    """Insert ``n_refine`` evenly spaced points on every edge of a closed polygon."""
    v = np.asarray(vertices, dtype=float)
    out: list[np.ndarray] = []
    for k in range(len(v)):
        a, b = v[k], v[(k + 1) % len(v)]
        for s in range(n_refine + 1):
            out.append(a + (b - a) * (s / (n_refine + 1.0)))
    return np.asarray(out)


def cylinder(diameter: float = 1.0, n: int = 120, center: Sequence[float] = (0.0, 0.0),
             motion: Motion | None = None, name: str = "cylinder") -> Body:
    """A circle of ``diameter`` cut into ``n`` equal panels."""
    radius = 0.5 * diameter
    theta = np.linspace(0.0, 2 * np.pi, int(n), endpoint=False)
    v = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]) + np.asarray(center, dtype=float)
    return Body(v, motion=motion, name=name)


def ellipse(semi_x: float = 0.5, semi_y: float = 0.25, n: int = 120, center: Sequence[float] = (0.0, 0.0),
            angle: float = 0.0, motion: Motion | None = None, name: str = "ellipse") -> Body:
    """An ellipse, optionally turned by ``angle`` in its own frame."""
    theta = np.linspace(0.0, 2 * np.pi, int(n), endpoint=False)
    v = np.column_stack([semi_x * np.cos(theta), semi_y * np.sin(theta)])
    c, s = np.cos(angle), np.sin(angle)
    v = v @ np.array([[c, s], [-s, c]]) + np.asarray(center, dtype=float)
    return Body(v, motion=motion, name=name)


def plate(length: float = 0.1, thickness: float = 0.004, hinge_at: Sequence[float] = (0.0, 0.0),
          angle: float = 0.0, n: int = 12, motion: Motion | None = None, name: str = "plate") -> Body:
    """A thin rectangle (a flat plate with a small thickness), hinged at its leading edge.

    The ``hinge_at`` point is the origin of the body frame, so a :func:`hinge` motion turns the plate about its
    leading edge.
    """
    t = 0.5 * thickness
    v = np.array([[0.0, -t], [length, -t], [length, t], [0.0, t]])
    if n > 4:
        v = refine_polygon(v, max(0, int(n) // 4 - 1))
    c, s = np.cos(angle), np.sin(angle)
    v = v @ np.array([[c, s], [-s, c]]) + np.asarray(hinge_at, dtype=float)
    return Body(v, motion=motion, name=name, thin=True)


def naca4(m: float = 0.0, p: float = 0.4, t: float = 0.12, chord: float = 1.0, n: int = 120,
          closed_te: bool = True, motion: Motion | None = None, name: str = "naca") -> Body:
    """A 4-digit NACA section (``m`` camber, ``p`` its position, ``t`` thickness), cosine-spaced panels.

    ``n`` is the total number of panels (half of them on each surface). ``closed_te`` collapses the last thickness
    term to zero so the trailing edge is a single point (the classic "closed" NACA form), which is what a sharp
    trailing edge needs.
    """
    n_half = max(6, int(n) // 2)
    beta = np.linspace(0.0, np.pi, n_half + 1)
    xc = 0.5 * (1.0 - np.cos(beta))                              # cosine spacing, 0 -> 1
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2 + 0.2843 * xc ** 3 - 0.1036 * xc ** 4)
    if closed_te:
        yt[-1] = 0.0
    if m > 0:
        yc = np.where(xc < p, m / p ** 2 * (2 * p * xc - xc ** 2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xc - xc ** 2))
        dyc = np.where(xc < p, 2 * m / p ** 2 * (p - xc), 2 * m / (1 - p) ** 2 * (p - xc))
    else:
        yc = np.zeros_like(xc)
        dyc = np.zeros_like(xc)
    theta = np.arctan(dyc)
    xu = xc - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = xc + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    upper = np.column_stack([xu, yu])[::-1]                       # trailing edge -> leading edge
    lower = np.column_stack([xl, yl])[1:]                         # leading edge -> trailing edge
    v = np.vstack([upper, lower]) * chord
    return Body(v, motion=motion, name=name)


def naca0012(n: int = 120, chord: float = 1.0, motion: Motion | None = None, name: str = "naca0012") -> Body:
    """The symmetric NACA 0012 section the reference case uses."""
    return naca4(m=0.0, p=0.4, t=0.12, chord=chord, n=n, motion=motion, name=name)


def deploying_spoiler(hinge_at: Sequence[float] = (0.8, 0.0), length: float = 0.10,
                      thickness: float = 0.004, n: int = 12, angle0_deg: float = 0.0,
                      angle1_deg: float = 90.0, t_start: float = 0.0, duration: float = 0.5,
                      name: str = "spoiler") -> Body:
    """A hinged spoiler that rises from flush with the surface to ``angle1_deg`` over ``duration``.

    The default is the reference case: a 10 % chord spoiler whose trailing edge sits 20 % of the chord ahead of the
    airfoil trailing edge, hinged at ``0.8 c`` and deploying to 90 degrees.
    """
    angle_of_t, rate_of_t = deploy(angle0_deg, angle1_deg, t_start, duration)
    # the plate's local frame has its leading edge at the origin; the hinge motion carries that origin to `hinge_at`
    outline = resample_polygon(plate(length=length, thickness=thickness).vertices, max(8, int(n)))
    return Body(outline, motion=hinge(hinge_at, angle_of_t, rate_of_t), name=name, thin=True)
