"""Sampling points for conditions whose region is given by a selector function.

Boundary and initial conditions live on a lower-dimensional set (``t == t0``, ``x == x_min``...). Drawing points
uniformly inside the box and keeping the ones a selector accepts almost never lands on that set: with
``np.isclose``-style selectors it returns zero points, and the condition is silently dropped. Here candidates are
drawn inside the box **and on every face of it** (one coordinate pinned to its lower or upper bound), so face-
and edge-type selectors find points, while interior selectors (data regions, sub-domains) still do.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence, Tuple

import numpy as np

__all__ = ["sample_condition_points"]


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
        for v in (lo[i], hi[i]):
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
