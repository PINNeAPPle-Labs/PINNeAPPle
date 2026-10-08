"""Sampling points for conditions whose region is given by a selector function.

Boundary and initial conditions live on a lower-dimensional set (``t == t0``, ``x == x_min``...). Drawing points
uniformly inside the box and keeping the ones a selector accepts almost never lands on that set: with
``np.isclose``-style selectors it returns zero points, and the condition is silently dropped. Here candidates are
drawn inside the box **and on every face of it** (one coordinate pinned to its lower or upper bound), and on the
plane where a coordinate is 0 when 0 lies inside its range (an initial condition at t = 0 on t in [-T, T]), so face-
and edge-type selectors find points, while interior selectors (data regions, sub-domains) still do.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence, Tuple

import numpy as np

__all__ = ["sample_condition_points", "box_face_normals"]


def sample_condition_points(select: Callable[[np.ndarray], np.ndarray], domain_bounds: Mapping[str, Tuple[float, float]],
                            coords: Sequence[str], n: int, rng: np.random.Generator) -> np.ndarray:
    """Up to ``n`` points (float32, shape ``(m, len(coords))``) accepted by ``select(X) -> bool mask``.

    Returns an empty ``(0, d)`` array when the selector accepts none of the candidates; callers must treat that as
    an error or skip explicitly, never pad with made-up points.
    """
    d = len(coords)
    lo = np.array([domain_bounds[c][0] for c in coords], dtype=np.float64)
    hi = np.array([domain_bounds[c][1] for c in coords], dtype=np.float64)
    parts = [lo + (hi - lo) * rng.random((n, d))]
    for i in range(d):
        # faces, plus the plane x_i = 0 when it is inside the range (t = 0 on a symmetric interval, x = 0 axes)
        planes = (lo[i], hi[i]) + ((0.0,) if lo[i] < 0.0 < hi[i] else ())
        for v in planes:
            p = lo + (hi - lo) * rng.random((n, d))
            p[:, i] = v
            parts.append(p)
    cand = np.concatenate(parts, axis=0).astype(np.float32)
    mask = np.asarray(select(cand), dtype=bool)
    if mask.shape != (len(cand),):
        raise ValueError(f"selector returned a mask of shape {mask.shape}, expected ({len(cand)},)")
    sel = cand[mask]
    if len(sel) > n:
        sel = sel[rng.choice(len(sel), n, replace=False)]
    return sel


def box_face_normals(points: np.ndarray, domain_bounds: Mapping[str, Tuple[float, float]], coords: Sequence[str],
                     time_coord: str = "t") -> Tuple[np.ndarray, np.ndarray]:
    """Outward unit normals of the bounding box at ``points`` (one column per coordinate, 0 for time).

    Returns ``(normals, on_face)``: a point on one face gets that face's normal, a point on an edge or corner the
    normalised sum of its faces' normals, and a point on no face (a curved boundary inside the box) gets zeros and
    ``on_face=False``; its normal cannot be known from a selector, so callers must not use it as if it were one.
    """
    pts = np.asarray(points, dtype=np.float64)
    normals = np.zeros_like(pts)
    for i, c in enumerate(coords):
        if c == time_coord:
            continue
        lo, hi = domain_bounds[c]
        tol = 1e-6 * max(abs(hi - lo), 1e-12)
        normals[np.abs(pts[:, i] - lo) <= tol, i] -= 1.0
        normals[np.abs(pts[:, i] - hi) <= tol, i] += 1.0
    norm = np.linalg.norm(normals, axis=1)
    on_face = norm > 0
    normals[on_face] /= norm[on_face, None]
    return normals.astype(np.float32), on_face
