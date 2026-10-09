"""Cell meshes for surfaces of revolution.

Each cell gets its own 4 vertices (two triangles), so a *per-cell* value becomes a per-vertex field with
no interpolation: what the viewer colours is exactly the cell's measurement. Cell ``(i, j)`` (row i along
the profile, sector j around the axis) owns vertices ``4*(i*M + j) .. +3``.

Convention (same as the reference apps): y is up, ``x = r cos(th)``, ``z = -r sin(th)``, so angles run
counter-clockwise seen from above; row 0 is the first profile point.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .model import ZoneSpec


def sample_profile(profile, n_rows: int) -> np.ndarray:
    """(n_rows + 1, 2) points (r, y) equally spaced in arc length along the polyline."""
    p = np.asarray(profile, dtype=float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    if np.any(seg <= 0):
        raise ValueError("profile has repeated points")
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n_rows + 1)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def cell_mesh(zone: ZoneSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Vertices (N*M*4, 3) and faces (N*M*2, 3) of the zone, one quad per cell."""
    N, M = zone.shape
    rows = sample_profile(zone.profile, N)
    th = zone.theta0 + np.linspace(0.0, 2 * np.pi, M + 1)
    cx, cz = zone.center

    def pt(r, y, a):
        return np.array([cx + r * np.cos(a), y, cz - r * np.sin(a)])

    v = np.empty((N, M, 4, 3))
    for i in range(N):
        (r0, y0), (r1, y1) = rows[i], rows[i + 1]
        for j in range(M):
            v[i, j] = [pt(r0, y0, th[j]), pt(r0, y0, th[j + 1]), pt(r1, y1, th[j]), pt(r1, y1, th[j + 1])]
    base = (np.arange(N * M) * 4)[:, None]
    # the two triangles of a cell stay adjacent in the face list
    f = np.stack([base + [0, 1, 2], base + [1, 3, 2]], axis=1).reshape(-1, 3)
    return v.reshape(-1, 3), f.astype(np.int64)


def cell_to_vertex(values: np.ndarray) -> np.ndarray:
    """Per-cell values (..., N, M) -> per-vertex (..., N*M*4) following ``cell_mesh``'s ordering."""
    a = np.asarray(values, dtype=float)
    flat = a.reshape(a.shape[:-2] + (-1,))
    return np.repeat(flat, 4, axis=-1)


def cell_centers(zone: ZoneSpec) -> np.ndarray:
    """(N, M, 3) world position of each cell centre (used to place sensors / hotspots)."""
    v, _ = cell_mesh(zone)
    return v.reshape(zone.n_rows, zone.n_sectors, 4, 3).mean(axis=2)
