"""Domain: the region of space a problem lives on.

A :class:`Domain` is an axis-aligned bounding box, optionally cut down by a
signed distance function (negative inside). It answers the three questions a
solver asks of its region: is this point inside, give me points inside, give me
points on the boundary. Boxes work in any dimension; irregular domains come from
the CSG shapes of :mod:`pinneapple_design.geometry.csg` (2D).

>>> from pinneapple_core import Domain
>>> d = Domain.box([0, 0], [2, 1])
>>> d.volume()
2.0
>>> d.sample_interior(128, seed=0).shape
(128, 2)
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

SdfFn = Callable[[np.ndarray], np.ndarray]

_MC_VOLUME_SAMPLES = 200_000


class Domain:
    """A region of R^d: a box, or a box cut by a signed distance function."""

    def __init__(
        self,
        lo: Sequence[float],
        hi: Sequence[float],
        *,
        sdf: Optional[SdfFn] = None,
        shape=None,
    ) -> None:
        self.lo = np.atleast_1d(np.asarray(lo, dtype=np.float64))
        self.hi = np.atleast_1d(np.asarray(hi, dtype=np.float64))
        if self.lo.ndim != 1 or self.lo.shape != self.hi.shape:
            raise ValueError("lo and hi must be 1D arrays of the same length")
        if np.any(self.hi <= self.lo):
            raise ValueError("every hi must be greater than lo")
        self._sdf = sdf
        self._shape = shape  # optional CSG shape, used for boundary sampling
        self._volume: Optional[float] = None

    # -- constructors -------------------------------------------------------
    @classmethod
    def box(cls, lo: Sequence[float], hi: Sequence[float]) -> "Domain":
        """Axis-aligned box in any dimension."""
        return cls(lo, hi)

    @classmethod
    def interval(cls, a: float, b: float) -> "Domain":
        return cls([a], [b])

    @classmethod
    def from_sdf(cls, shape) -> "Domain":
        """Wrap a 2D CSG shape (``pinneapple_design.geometry.csg``) as a Domain."""
        lo = np.asarray(shape.bounds_min, dtype=np.float64)
        hi = np.asarray(shape.bounds_max, dtype=np.float64)
        return cls(lo, hi, sdf=lambda x: shape.sdf(np.asarray(x, dtype=np.float64)), shape=shape)

    # -- basic properties ---------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.lo.shape[0])

    @property
    def is_box(self) -> bool:
        return self._sdf is None

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lo, self.hi

    def volume(self) -> float:
        """Length / area / volume. Exact for boxes, Monte Carlo (fixed seed) otherwise."""
        if self._volume is None:
            box_vol = float(np.prod(self.hi - self.lo))
            if self.is_box:
                self._volume = box_vol
            else:
                rng = np.random.default_rng(0)
                pts = self.lo + rng.random((_MC_VOLUME_SAMPLES, self.dim)) * (self.hi - self.lo)
                self._volume = box_vol * float(np.mean(self.contains(pts)))
        return self._volume

    def boundary_measure(self) -> float:
        """Length / area of the boundary (point count in 1D). Exact for boxes; for
        SDF domains estimated from the volume of a thin shell around the boundary."""
        if self.is_box:
            span = self.hi - self.lo
            if self.dim == 1:
                return 2.0
            return float(sum(2.0 * np.prod(np.delete(span, a)) for a in range(self.dim)))
        if self.dim != 2:
            raise NotImplementedError("boundary measure of non-box domains is only supported in 2D")
        rng = np.random.default_rng(0)
        span = self.hi - self.lo
        eps = 1e-3 * float(span.min())
        pts = self.lo + rng.random((400_000, 2)) * span
        return float(np.prod(span) * np.mean(np.abs(self.sdf(pts)) < eps) / (2 * eps))

    # -- queries ------------------------------------------------------------
    def sdf(self, x: np.ndarray) -> np.ndarray:
        """Signed distance, negative inside. Exact for boxes."""
        x = self._as_points(x)
        if self._sdf is not None:
            return np.asarray(self._sdf(x), dtype=np.float64)
        c = (self.lo + self.hi) / 2.0
        q = np.abs(x - c) - (self.hi - self.lo) / 2.0
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside

    def contains(self, x: np.ndarray, tol: float = 0.0) -> np.ndarray:
        """Boolean mask of points inside the domain (boundary included up to ``tol``)."""
        return self.sdf(x) <= tol

    # -- sampling -----------------------------------------------------------
    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        """``(n, dim)`` points uniformly distributed inside the domain."""
        rng = np.random.default_rng(seed)
        span = self.hi - self.lo
        if self.is_box:
            return self.lo + rng.random((n, self.dim)) * span
        out = np.empty((0, self.dim))
        while out.shape[0] < n:
            cand = self.lo + rng.random((max(2 * n, 1024), self.dim)) * span
            out = np.concatenate([out, cand[self.contains(cand)]])
        return out[:n]

    def sample_boundary(
        self, n: int, *, seed: int = 0, return_normals: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """``(n, dim)`` points on the boundary, uniform in boundary measure.

        Boxes also return outward unit normals with ``return_normals=True``;
        SDF domains estimate them from the gradient of the signed distance.
        """
        rng = np.random.default_rng(seed)
        if self.is_box:
            pts, nrm = self._box_boundary(n, rng)
        elif self._shape is not None:
            pts = np.asarray(self._shape.sample_boundary(n, seed=seed), dtype=np.float64)
            # CSG samplers keep points within a tolerance band; project them onto the
            # zero level set with Newton steps along the SDF gradient.
            for _ in range(3):
                pts = pts - self.sdf(pts)[:, None] * self._sdf_normals(pts)
            nrm = self._sdf_normals(pts)
        else:
            raise NotImplementedError("boundary sampling needs a box or a CSG shape")
        return (pts, nrm) if return_normals else pts

    def _box_boundary(self, n: int, rng) -> Tuple[np.ndarray, np.ndarray]:
        d = self.dim
        span = self.hi - self.lo
        # face (axis, side) has measure prod of the other extents (1 for d == 1)
        areas = np.array([np.prod(np.delete(span, a)) for a in range(d) for _ in (0, 1)])
        face = rng.choice(2 * d, size=n, p=areas / areas.sum())
        pts = self.lo + rng.random((n, d)) * span
        nrm = np.zeros((n, d))
        axis, side = face // 2, face % 2
        pts[np.arange(n), axis] = np.where(side == 1, self.hi[axis], self.lo[axis])
        nrm[np.arange(n), axis] = np.where(side == 1, 1.0, -1.0)
        return pts, nrm

    def _sdf_normals(self, pts: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        g = np.empty_like(pts)
        for a in range(self.dim):
            e = np.zeros(self.dim)
            e[a] = eps
            g[:, a] = (self.sdf(pts + e) - self.sdf(pts - e)) / (2 * eps)
        norm = np.linalg.norm(g, axis=1, keepdims=True)
        return g / np.where(norm == 0, 1.0, norm)

    def _as_points(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, self.dim) if self.dim > 1 else x[:, None]
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"points must have shape (N, {self.dim}), got {x.shape}")
        return x

    def __repr__(self) -> str:
        kind = "box" if self.is_box else "sdf"
        return f"Domain({kind}, dim={self.dim}, lo={self.lo.tolist()}, hi={self.hi.tolist()})"
