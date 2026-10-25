"""Offset paraboloid reflectors with optional rolled edges, as triangle meshes for :mod:`.po`.

The paraboloid is ``z = (x^2 + y^2) / (4F)`` (vertex at the origin, focus at ``(0, 0, F)``, reflected rays along
+z). The section is the part whose projection on the aperture plane lies inside a superellipse centred at
``(0, centre_y)`` with half sizes ``half_x``, ``half_y`` and exponent ``n`` (``n = 2``: an ellipse; large ``n``: a
rectangle with rounded corners). Past this junction contour an optional rolled edge continues the surface in each
vertical plane through the section centre: the profile leaves the paraboloid tangentially and with its curvature,
blends (``sin^2`` over the first ``blend`` fraction of the roll) to a constant curvature ``-1 / roll`` turning away
from the quiet zone, and stops when its tangent has turned to ``end_angle`` degrees (from the outward horizontal;
-90 = pointing straight back). ``roll_aspect`` stretches the roll's departure from the junction tangent line along
z (the tangent stays continuous): 1 gives the circular roll, larger values a deeper, elliptic-like one. This is a blended rolled edge in the spirit of Gupta and Burnside; the exact
profile of a given range (ellipse ratio, blending function) is a parameter, not built in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["OffsetReflector", "ReflectorMesh", "superellipse_radius", "mesh_reflector", "flat_disk"]


@dataclass
class OffsetReflector:
    focal: float                 # F [m]
    half_x: float                # half width of the section's projection [m]
    half_y: float                # half height [m]
    centre_y: float              # height of the section centre above the paraboloid axis [m]
    n: float = 6.0               # superellipse exponent of the junction contour
    roll: float = 0.0            # rolled-edge radius [m]; 0 = a sharp edge
    end_angle: float = -90.0     # tangent angle where the roll stops [deg]
    blend: float = 0.3           # fraction of the roll arc over which the curvature blends
    roll_aspect: float = 1.0     # depth / width of the roll (1: circular; > 1: an elongated, elliptic-like roll)

    def __post_init__(self):
        if min(self.focal, self.half_x, self.half_y, self.roll_aspect) <= 0 or self.roll < 0 or self.n < 2:
            raise ValueError("focal, half sizes > 0, roll >= 0 and n >= 2 are required")

    @property
    def focus(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.focal])

    @property
    def centre(self) -> np.ndarray:
        """The section centre on the paraboloid."""
        return np.array([0.0, self.centre_y, self.centre_y ** 2 / (4 * self.focal)])


@dataclass
class ReflectorMesh:
    vertices: np.ndarray         # (nv, 3)
    faces: np.ndarray            # (nf, 3)
    centres: np.ndarray          # (nf, 3)
    normals: np.ndarray          # (nf, 3), unit, on the concave (focus) side
    areas: np.ndarray            # (nf,)
    roll: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))   # triangles on the rolled edge

    @property
    def area(self) -> float:
        return float(self.areas.sum())


def superellipse_radius(phi, a: float, b: float, n: float):
    c, s = np.abs(np.cos(phi)), np.abs(np.sin(phi))
    return (c ** n / a ** n + s ** n / b ** n) ** (-1.0 / n)


def _roll_profile(theta0: float, kappa_p: float, roll: float, end_angle: float, blend: float, n_pts: int):
    """(du, dw) of the rolled edge in the meridian plane from the junction, ``n_pts`` points equally spaced in
    arc length, and its arc length."""
    kr = -1.0 / roll
    end = np.deg2rad(end_angle)
    if theta0 <= end:
        return np.zeros((1, 2)), 0.0
    # blend length: a fraction of the arc of a circle of radius roll from theta0 to end
    L_b = blend * roll * (theta0 - end)
    ds = roll * 1e-3
    pts, th, s = [np.zeros(2)], theta0, 0.0
    while th > end and s < 50 * roll:
        t = min(s / L_b, 1.0) if L_b > 0 else 1.0
        w = np.sin(0.5 * np.pi * t) ** 2
        k = (1 - w) * kappa_p + w * kr
        pts.append(pts[-1] + ds * np.array([np.cos(th), np.sin(th)]))
        th += k * ds
        s += ds
    P = np.array(pts)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    a = np.linspace(0, arc[-1], n_pts)
    return np.c_[np.interp(a, arc, P[:, 0]), np.interp(a, arc, P[:, 1])], float(arc[-1])


def mesh_reflector(r: OffsetReflector, h: float) -> ReflectorMesh:
    """Triangle mesh with edges of about ``h`` (use lambda/8 or less for PO)."""
    a, b, F = r.half_x, r.half_y, r.focal
    rmax = max(a, b) + r.roll * 1.5
    n_phi = max(32, int(np.ceil(2 * np.pi * rmax / h)))
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    rj = superellipse_radius(phi, a, b, r.n)
    K = max(4, int(np.ceil(rj.max() / h)))
    c2 = np.array([0.0, r.centre_y])
    rows = []
    # paraboloid part: K + 1 rings from the centre (ring 0 collapses to the centre) to the junction
    for k in range(K + 1):
        xy = c2 + (k / K) * rj[:, None] * np.c_[np.cos(phi), np.sin(phi)]
        rows.append(np.c_[xy, (xy ** 2).sum(1) / (4 * F)])
    n_par = len(rows)
    if r.roll > 0:
        u = np.c_[np.cos(phi), np.sin(phi)]
        Pj = rows[-1]
        # tangent angle and curvature of the paraboloid along each meridian at the junction
        dzds = (Pj[:, :2] * u).sum(1) / (2 * F)
        th0 = np.arctan(dzds)
        kp = (1 / (2 * F)) / (1 + dzds ** 2) ** 1.5
        prof, arcs = zip(*[_roll_profile(t0, k0, r.roll, r.end_angle, r.blend, 400) for t0, k0 in zip(th0, kp, strict=True)],
                         strict=True)
        M = max(2, int(np.ceil(max(arcs) / h)))
        for m in range(1, M + 1):
            pts = []
            for i in range(n_phi):
                P = prof[i]
                j = min(len(P) - 1, int(round(m / M * (len(P) - 1))))
                du, dw = P[j]
                t0 = np.tan(th0[i])
                dw = du * t0 + r.roll_aspect * (dw - du * t0)
                pts.append([Pj[i, 0] + du * u[i, 0], Pj[i, 1] + du * u[i, 1], Pj[i, 2] + dw])
            rows.append(np.array(pts))
    V = np.concatenate(rows)
    faces, is_roll = [], []
    for k in range(len(rows) - 1):
        i0, i1 = k * n_phi, (k + 1) * n_phi
        for i in range(n_phi):
            j = (i + 1) % n_phi
            faces += [(i0 + i, i1 + i, i1 + j), (i0 + i, i1 + j, i0 + j)]
            is_roll += [k >= n_par - 1] * 2
    Fc = np.array(faces)
    A, B, C = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
    cr = np.cross(B - A, C - A)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    ok = area > 1e-14 * h * h
    Fc, cr, area, is_roll = Fc[ok], cr[ok], area[ok], np.array(is_roll)[ok]
    Nn = cr / (2 * area[:, None])
    cen = V[Fc].mean(1)
    par = ~is_roll
    if np.mean(np.einsum("ij,ij->i", Nn[par], r.focus - cen[par])) < 0:     # concave side towards the focus
        Nn, Fc = -Nn, Fc[:, ::-1]
    return ReflectorMesh(V, Fc, cen, Nn, area, is_roll)


def flat_disk(radius: float, h: float) -> ReflectorMesh:
    """A flat disk of ``radius`` in z = 0 (normal +z), for the PO validation against the closed form."""
    n_phi = max(32, int(np.ceil(2 * np.pi * radius / h)))
    K = max(4, int(np.ceil(radius / h)))
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    rows = [np.c_[(k / K) * radius * np.cos(phi), (k / K) * radius * np.sin(phi), np.zeros(n_phi)]
            for k in range(K + 1)]
    V = np.concatenate(rows)
    faces = []
    for k in range(K):
        for i in range(n_phi):
            j = (i + 1) % n_phi
            faces += [(k * n_phi + i, (k + 1) * n_phi + i, (k + 1) * n_phi + j),
                      (k * n_phi + i, (k + 1) * n_phi + j, k * n_phi + j)]
    Fc = np.array(faces)
    A, B, C = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
    cr = np.cross(B - A, C - A)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    ok = area > 0
    Fc, cr, area = Fc[ok], cr[ok], area[ok]
    Nn = cr / (2 * area[:, None])
    Nn = np.where(Nn[:, 2:3] < 0, -Nn, Nn)
    return ReflectorMesh(V, Fc, V[Fc].mean(1), Nn, area, np.zeros(len(Fc), bool))
