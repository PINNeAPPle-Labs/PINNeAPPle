"""Mesh primitives for building twin parts without a CAD file: revolve, sweep, loft, box, cylinder.

Every function returns ``(vertices (V, 3) float, faces (F, 3) int)`` with y up, in metres. Vertices are shared
within a primitive (smooth shading); combine primitives with :func:`merge`. Winding is consistent but the exported
glTF material is double sided, so orientation never hides a face.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

Mesh = Tuple[np.ndarray, np.ndarray]


def _grid_faces(n_rows: int, n_cols: int, closed_cols: bool) -> np.ndarray:
    """Triangles between consecutive rows of a (n_rows, n_cols) vertex grid (row-major)."""
    ncol_links = n_cols if closed_cols else n_cols - 1
    f = []
    for i in range(n_rows - 1):
        for j in range(ncol_links):
            a, b = i * n_cols + j, i * n_cols + (j + 1) % n_cols
            c, d = a + n_cols, b + n_cols
            f += [[a, b, c], [b, d, c]]
    return np.asarray(f, dtype=np.int64)


def _area(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    tri = v[f]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def revolve(profile: Sequence[Tuple[float, float]], n_theta: int = 48, *, cap: bool = False) -> Mesh:
    """Surface of revolution of ``(r, y)`` points around the y axis (``cap`` closes the ends with a fan)."""
    p = np.asarray(profile, dtype=float)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    v = np.stack([p[:, 0, None] * np.cos(th), np.repeat(p[:, 1, None], n_theta, 1), -p[:, 0, None] * np.sin(th)], -1)
    v = v.reshape(-1, 3)
    f = _grid_faces(len(p), n_theta, True)
    f = f[_area(v, f) > 1e-14]  # a profile point on the axis (r = 0) collapses a ring: drop the zero-area triangles
    if cap:
        extra_v, extra_f = [], []
        for row, flip in ((0, False), (len(p) - 1, True)):
            c = len(v) + len(extra_v)
            extra_v.append([0.0, p[row, 1], 0.0])
            for j in range(n_theta):
                a, b = row * n_theta + j, row * n_theta + (j + 1) % n_theta
                extra_f.append([c, b, a] if flip else [c, a, b])
        v = np.vstack([v, extra_v])
        f = np.vstack([f, extra_f])
    return v, f


def _frames(path: np.ndarray, up: Sequence[float]):
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    up = np.asarray(up, float)
    n = up - np.dot(up, t[0]) * t[0]
    if np.linalg.norm(n) < 1e-9:  # path parallel to `up`
        n = np.cross(t[0], [1.0, 0.0, 0.0])
    n /= np.linalg.norm(n)
    N, B = [], []
    for ti in t:
        n = n - np.dot(n, ti) * ti  # parallel transport: keep the section from twisting
        n /= np.linalg.norm(n)
        N.append(n.copy()); B.append(np.cross(ti, n))
    return np.asarray(N), np.asarray(B)


def sweep(section: Sequence[Tuple[float, float]], path: Sequence[Sequence[float]], *, closed: bool = True,
          up: Sequence[float] = (0, 1, 0)) -> Mesh:
    """Extrude a 2D section ``(u, v)`` along a 3D path. ``u`` runs along the frame normal (closest to ``up``),
    ``v`` along the binormal. ``closed=False`` leaves an open profile (e.g. a U-channel)."""
    P = np.asarray(path, float)
    S = np.asarray(section, float)
    N, B = _frames(P, up)
    v = P[:, None, :] + S[None, :, 0, None] * N[:, None, :] + S[None, :, 1, None] * B[:, None, :]
    return v.reshape(-1, 3), _grid_faces(len(P), len(S), closed)


def airfoil(chord: float = 1.0, thickness: float = 0.12, n: int = 24) -> np.ndarray:
    """Closed symmetric NACA 00xx outline, ``(2n, 2)``: x along the chord from the leading edge, y thickness."""
    b = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(b))  # cosine spacing: dense at the leading edge
    yt = 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    upper = np.stack([x, yt], 1)
    lower = np.stack([x[::-1], -yt[::-1]], 1)[1:-1]
    return np.vstack([upper, lower]) * chord


def loft(rings: Sequence[np.ndarray]) -> Mesh:
    """Skin through rings of equal point count ``(K, 3)`` (closed rings): wings, blades, fuselage sections."""
    R = [np.asarray(r, float) for r in rings]
    if len({len(r) for r in R}) != 1:
        raise ValueError("all rings need the same number of points")
    v = np.vstack(R)
    return v, _grid_faces(len(R), len(R[0]), True)


def wing(span: float, root_chord: float, tip_chord: float, *, sweep_deg: float = 0.0, dihedral_deg: float = 0.0,
         twist_deg: float = 0.0, thickness: float = 0.12, n_sections: int = 12, n_pts: int = 20) -> Mesh:
    """Half wing / blade along +z from the root (z=0): chord along x (leading edge at -x), thickness along y."""
    rings = []
    for k in range(n_sections):
        s = k / (n_sections - 1)
        chord = root_chord + (tip_chord - root_chord) * s
        a = airfoil(chord, thickness, n_pts)
        a[:, 0] -= 0.25 * chord  # quarter-chord at the reference line
        tw = np.radians(twist_deg * s)
        x = a[:, 0] * np.cos(tw) - a[:, 1] * np.sin(tw)
        y = a[:, 0] * np.sin(tw) + a[:, 1] * np.cos(tw)
        z = s * span
        rings.append(np.stack([x + z * np.tan(np.radians(sweep_deg)), y + z * np.tan(np.radians(dihedral_deg)),
                               np.full_like(x, z)], 1))
    return loft(rings)


def box(size: Sequence[float], center: Sequence[float] = (0, 0, 0)) -> Mesh:
    sx, sy, sz = (np.asarray(size, float) / 2)
    c = np.asarray(center, float)
    v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]) + c
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = [t for a, b, c_, d in q for t in ([a, b, c_], [a, c_, d])]
    return v, np.asarray(f, dtype=np.int64)


def cylinder(p0: Sequence[float], p1: Sequence[float], radius: float, n_theta: int = 32, *, cap: bool = True) -> Mesh:
    """Cylinder between two points (``radius`` may be a pair for a frustum)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    r0, r1 = (radius, radius) if np.isscalar(radius) else radius
    d = p1 - p0
    L = np.linalg.norm(d)
    ax = d / L
    seed = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(ax, seed); u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ring = lambda c, r: c + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)
    v = np.vstack([ring(p0, r0), ring(p1, r1)])
    f = _grid_faces(2, n_theta, True)
    if cap:
        c0, c1 = len(v), len(v) + 1
        v = np.vstack([v, p0, p1])
        fan = [[c0, (j + 1) % n_theta, j] for j in range(n_theta)] + [[c1, n_theta + j, n_theta + (j + 1) % n_theta] for j in range(n_theta)]
        f = np.vstack([f, fan])
    return v, f


def rotation(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    t = np.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def transform(mesh: Mesh, *, R: np.ndarray | None = None, t: Sequence[float] = (0, 0, 0), scale: float = 1.0) -> Mesh:
    v, f = mesh
    v = np.asarray(v, float) * scale
    if R is not None:
        v = v @ np.asarray(R).T
    return v + np.asarray(t, float), f


def mirror_z(mesh: Mesh) -> Mesh:
    v, f = mesh
    return v * np.array([1, 1, -1]), f[:, ::-1]


def merge(meshes: Sequence[Mesh]) -> Mesh:
    vs, fs, off = [], [], 0
    for v, f in meshes:
        vs.append(np.asarray(v, float)); fs.append(np.asarray(f) + off); off += len(v)
    return np.vstack(vs), np.vstack(fs)


def mesh_quality(mesh: Mesh) -> dict:
    """Sanity numbers used by the tests: finite coordinates, indices in range, degenerate-triangle share, bounds."""
    v, f = mesh
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return {"finite": bool(np.isfinite(v).all()), "in_range": bool(f.min() >= 0 and f.max() < len(v)),
            "degenerate_share": float(np.mean(area < 1e-12)), "bounds": (v.min(0), v.max(0)), "area": float(area.sum())}
