"""Closed triangulated bodies for external-flow studies: the Ahmed body, a sphere, a cylinder, a box, an annulus.

Each returns ``(vertices, faces)`` with outward-facing triangles, in metres, x along the flow, z up. Use them with
``pinneapple_simulation.numerical_solvers.external_flow.ExternalFlow`` or ``pinneapple_tools.visualization.studio``.
"""
from __future__ import annotations

import math

import numpy as np

Mesh = tuple[np.ndarray, np.ndarray]


def _loft(rings: np.ndarray, close_ends: bool = True) -> Mesh:
    """Triangulate a stack of closed rings (n_rings, n_pts, 3), ordered along +x, each counter-clockwise seen from
    +x; caps the two ends."""
    nr, n, _ = rings.shape
    V = rings.reshape(-1, 3)
    F = []
    for i in range(nr - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = a + n, b + n
            F += [[a, c, b], [b, c, d]]
    V = list(V)
    if close_ends:                                          # cap normals point away from the neighbouring ring
        for i, nb in ((0, 1), (nr - 1, nr - 2)):
            ctr = len(V)
            c = rings[i].mean(0)
            V.append(c)
            out_dir = c - rings[nb].mean(0)
            for j in range(n):
                a, b = i * n + j, i * n + (j + 1) % n
                nrm = np.cross(rings[i][j] - c, rings[i][(j + 1) % n] - c)
                F.append([ctr, a, b] if nrm @ out_dir > 0 else [ctr, b, a])
    # sides: orient like the caps (outward)
    V, F = np.array(V, float), np.array(F, np.int64)
    ns = 2 * (nr - 1) * n
    if close_ends and _signed_volume(V, F[ns:]) * _signed_volume(V, F[:ns]) < 0:
        F[:ns] = F[:ns, ::-1]
    if _signed_volume(V, F) < 0:
        F = F[:, ::-1]
    return V, F


def _signed_volume(V, F) -> float:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def _rounded_rect(w: float, h: float, r: float, n_corner: int = 8, side: float = 0.0) -> np.ndarray:
    """Counter-clockwise (y, z) outline of a w x h rectangle centred at 0 with corner radius r; ``side`` > 0 adds
    points along the straight edges at most ``side`` apart (otherwise each edge is one segment)."""
    pts = []
    corners = ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270))
    for i, (cy, cz, a0) in enumerate(corners):
        for k in range(n_corner + 1):
            a = math.radians(a0 + 90 * k / n_corner)
            pts.append((cy + r * math.cos(a), cz + r * math.sin(a)))
        if side > 0:                                        # the straight edge to the next corner's first point
            ny, nz, b0 = corners[(i + 1) % 4]
            q = (ny + r * math.cos(math.radians(b0)), nz + r * math.sin(math.radians(b0)))
            n = int(math.ceil(math.hypot(q[0] - pts[-1][0], q[1] - pts[-1][1]) / side))
            p0 = pts[-1]
            pts += [(p0[0] + (q[0] - p0[0]) * j / n, p0[1] + (q[1] - p0[1]) * j / n) for j in range(1, n)]
    return np.array(pts)


def ahmed_body(slant_deg: float = 25.0, scale: float = 1.0, clearance: float = 0.05, n_nose: int = 10,
               n_body: int = 5, n_slant: int = 6, n_corner: int = 4, slant_length: float = 0.222,
               side: float = 0.0) -> Mesh:
    """The Ahmed body (Ahmed, Ramm & Faltin 1984): 1.044 x 0.389 x 0.288 m, front corners rounded with 0.1 m radius,
    a 0.222 m long rear slant at ``slant_deg``. Its lower face sits ``clearance`` above z = 0 (the road; the stilts
    are left out). Nose at x = 0. The classic drag coefficient at 25 degrees is about 0.285 (with ground, Re 4.3e6).

    ``n_body`` / ``n_slant`` sections along the straight part and the slant, ``n_corner`` points per rounded long
    edge, ``side`` > 0 extra points along the flat faces at most that far apart: the vertex count depends only on these (and ``n_nose``), never on the slant angle, so vertex ``i`` is the
    same material point on every variant (fields on the body can be compared vertex by vertex)."""
    L, W, H, R, Ls = 1.044, 0.389, 0.288, 0.100, float(slant_length)
    if not 0.0 < Ls < L - R:
        raise ValueError(f"slant_length must be in (0, {L - R:g}) m, got {Ls:g}")
    if Ls * math.tan(math.radians(slant_deg)) >= H:
        raise ValueError(f"a {Ls:g} m slant at {slant_deg:g} deg cuts through the {H:g} m high body")
    zc = clearance + H / 2
    out = _rounded_rect(W, H, 0.02, n_corner, side)        # the long edges are slightly rounded in the original
    rings = []
    for k in range(n_nose + 1):                             # front edges rounded with radius R: theta 0 -> 90 deg
        th = math.pi / 2 * k / n_nose
        x, inset = R * (1 - math.cos(th)), R * (1 - math.sin(th))
        sc = np.c_[out[:, 0] * (W / 2 - inset) / (W / 2), out[:, 1] * (H / 2 - inset) / (H / 2)]
        rings.append(np.c_[np.full(len(sc), x), sc[:, 0], sc[:, 1] + zc])
    t = math.tan(math.radians(slant_deg))
    for x in np.r_[np.linspace(R, L - Ls, n_body + 1)[1:], np.linspace(L - Ls, L, n_slant + 1)[1:]]:
        top = H / 2 - max(0.0, x - (L - Ls)) * t
        s = out.copy()
        s[:, 1] = np.minimum(s[:, 1], top)
        rings.append(np.c_[np.full(len(s), x), s[:, 0], s[:, 1] + zc])
    V, F = _loft(np.array(rings))
    return V * scale, F


def sphere(radius: float = 0.5, centre=(0.0, 0.0, 0.0), n: int = 48) -> Mesh:
    """UV sphere."""
    th = np.linspace(0, math.pi, n // 2 + 1)[1:-1]
    rings = [np.c_[radius * np.cos(t) * np.ones(n), radius * np.sin(t) * np.cos(np.linspace(0, 2 * math.pi, n, endpoint=False)),
                   radius * np.sin(t) * np.sin(np.linspace(0, 2 * math.pi, n, endpoint=False))] for t in th[::-1]]
    V, F = _loft(np.array(rings))
    V[np.argsort(V[:, 0])[0]] = (-radius, 0, 0)            # cap centres onto the poles
    V[np.argsort(V[:, 0])[-1]] = (radius, 0, 0)
    return V + np.asarray(centre, float), F


def cylinder(radius: float = 0.5, length: float = 2.0, axis: str = "y", n: int = 48) -> Mesh:
    """Closed cylinder centred at the origin; axis "y" is the classic cross-flow cylinder, "x" an axial one."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rings = np.array([np.c_[np.full(n, s), radius * np.cos(a), radius * np.sin(a)] for s in np.linspace(-length / 2, length / 2, 9)])
    V, F = _loft(rings)
    if axis == "y":
        V = V[:, [1, 0, 2]]
        F = F[:, ::-1]
    elif axis == "z":
        V = V[:, [1, 2, 0]]
    return V, F


def box(size=(1.0, 1.0, 1.0), centre=(0.0, 0.0, 0.0)) -> Mesh:
    """Axis-aligned box (e.g. a building block)."""
    sx, sy, sz = np.asarray(size, float) / 2
    V = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]) + np.asarray(centre, float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    if _signed_volume(V, F) < 0:
        F = F[:, ::-1]
    return V, F


def annulus(r_inner: float, r_outer: float, height: float, n: int = 72, z0: float = 0.0) -> Mesh:
    """Closed annular ring (a tube with flat ends) around the z axis, from ``z0`` to ``z0 + height``: outer and inner
    walls plus the two ring faces, outward-facing triangles. For ``r_inner = 0`` use :func:`cylinder`."""
    if not 0.0 < r_inner < r_outer or height <= 0:
        raise ValueError("need 0 < r_inner < r_outer and height > 0")
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    # four rings of n points (outer-bottom, outer-top, inner-top, inner-bottom) joined in a closed loop of bands:
    # outer wall, top face, inner wall, bottom face
    V = np.vstack([np.c_[r * c, r * s, np.full(n, z)] for r, z in
                   ((r_outer, z0), (r_outer, z0 + height), (r_inner, z0 + height), (r_inner, z0))])
    F = []
    for i in range(4):
        p, q = i * n, ((i + 1) % 4) * n
        for j in range(n):
            j1 = (j + 1) % n
            F += [[p + j, p + j1, q + j], [q + j, p + j1, q + j1]]
    F = np.array(F, np.int64)
    if _signed_volume(V, F) < 0:
        F = F[:, ::-1]
    return V, F
