"""The set of vortex domains: their positions, circulations and core radii, plus the bookkeeping VVD needs.

Insertion, the core radius every domain carries (the distance to its second nearest neighbour, with a floor of a
third of the local panel length), the merge that keeps the number of domains in check, and removal with the
circulation accounted for -- a domain that leaves the field is vortex momentum that has to be reported, not
discarded silently.

The numbers are the ones the method is built on (Dynnikova's VVD scheme as documented in the reference
implementation): ``eps`` from the second neighbour, merge distance ``0.4 eps`` with the "own among strangers" rule,
and the diffusion drift clamped so a domain never jumps further than its own core in one step.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["Domains"]


class Domains:
    """Vortex domains of a 2-D field: ``x``, ``y``, circulation ``gamma`` and core radius ``eps``."""

    def __init__(self, x: np.ndarray | None = None, y: np.ndarray | None = None,
                 gamma: np.ndarray | None = None):
        x = np.zeros(0) if x is None else np.asarray(x, dtype=float).ravel()
        y = np.zeros(0) if y is None else np.asarray(y, dtype=float).ravel()
        gamma = np.zeros(0) if gamma is None else np.asarray(gamma, dtype=float).ravel()
        if not (x.size == y.size == gamma.size):
            raise ValueError("x, y and gamma must have the same length")
        self.x = x
        self.y = y
        self.gamma = gamma
        self.eps = np.ones_like(x)
        self.velocity = np.zeros((x.size, 2))

    # -- basics -----------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.gamma.size)

    def copy(self) -> Domains:
        other = Domains(self.x.copy(), self.y.copy(), self.gamma.copy())
        other.eps = self.eps.copy()
        other.velocity = self.velocity.copy()
        return other

    @property
    def positions(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])

    def subsets(self, mask: np.ndarray) -> Domains:
        mask = np.asarray(mask, dtype=bool)
        other = Domains(self.x[mask], self.y[mask], self.gamma[mask])
        other.eps = self.eps[mask]
        other.velocity = self.velocity[mask]
        return other

    def keep(self, mask: np.ndarray) -> None:
        """Drop every domain where ``mask`` is false (positions, circulations, cores and velocities together)."""
        mask = np.asarray(mask, dtype=bool)
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.gamma = self.gamma[mask]
        self.eps = self.eps[mask]
        self.velocity = self.velocity[mask]

    @property
    def total_circulation(self) -> float:
        return float(self.gamma.sum())

    @property
    def total_vorticity_moment(self) -> np.ndarray:
        """``sum_j gamma_j (y_j, -x_j)`` -- the vorticity impulse of the field (what a force changes)."""
        return np.array([(self.gamma * self.y).sum(), -(self.gamma * self.x).sum()])

    @property
    def enstrophy(self) -> float:
        """``sum_j gamma_j^2 / (2 pi eps_j^2)``, the discrete enstrophy."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.nansum(self.gamma ** 2 / (2 * np.pi * self.eps ** 2)))

    # -- insertion --------------------------------------------------------------------------------------
    def extend(self, points: np.ndarray, gamma: np.ndarray) -> None:
        """Append domains at ``points`` (n, 2) with the given circulations."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        gamma = np.asarray(gamma, dtype=float).ravel()
        if points.shape[0] != gamma.size:
            raise ValueError("points and gamma must have the same length")
        self.x = np.concatenate([self.x, points[:, 0]])
        self.y = np.concatenate([self.y, points[:, 1]])
        self.gamma = np.concatenate([self.gamma, gamma])
        self.eps = np.concatenate([self.eps, np.ones(gamma.size)])
        self.velocity = np.concatenate([self.velocity, np.zeros((gamma.size, 2))])

    def append(self, domain: Domains) -> None:
        self.extend(domain.positions, domain.gamma)

    def remove(self, mask: np.ndarray) -> np.ndarray:
        """Drop the domains where ``mask`` is true; returns their circulations."""
        mask = np.asarray(mask, dtype=bool)
        removed = self.gamma[mask].copy()
        self.x = self.x[~mask]
        self.y = self.y[~mask]
        self.gamma = self.gamma[~mask]
        self.eps = self.eps[~mask]
        self.velocity = self.velocity[~mask]
        return removed

    # -- core radius ------------------------------------------------------------------------------------
    def update_epsilon(self, floor: float = 0.0, floor_of_points=None) -> None:
        """Core radius = distance to the second nearest neighbour, but never below ``floor``.

        ``floor_of_points`` (optional callable) gives a per-domain floor, which the solver uses for the panels:
        a domain sitting on a panel of length ``L`` must not be thinner than ``L/3``.
        """
        n = len(self)
        if n == 0:
            return
        if n == 1:
            eps = np.full(1, max(float(floor), 1e-12))
        else:
            pts = self.positions
            dist, _ = cKDTree(pts).query(pts, k=min(3, n))
            dist = np.atleast_2d(dist)
            eps = dist[:, -1]                                    # second neighbour when n >= 3, else the nearest
        if floor_of_points is not None:
            low = np.asarray(floor_of_points(self.positions), dtype=float)
        else:
            low = np.full(n, float(floor))
        self.eps = np.maximum(eps, low)

    # -- merging ----------------------------------------------------------------------------------------
    def merge(self, min_distance: float, factor: float = 0.4, strangers: bool = True) -> int:
        """Merge domains closer than ``factor * eps`` (and the "own among strangers" triple rule).

        Two domains of the same sign merge at their circulation-weighted centroid and add their circulations;
        opposite signs keep the position of the larger one. Returns the number of merged domains.
        """
        n = len(self)
        if n < 2 or min_distance <= 0.0:
            return 0
        pts = self.positions
        tree = cKDTree(pts)
        # candidate pairs: within the largest merge radius allowed
        radii = np.maximum(factor * self.eps, 0.5 * min_distance)
        pairs = tree.query_pairs(float(np.max(radii)), output_type="ndarray")
        merged = 0
        if len(pairs):
            dist = np.hypot(pts[pairs[:, 0], 0] - pts[pairs[:, 1], 0], pts[pairs[:, 0], 1] - pts[pairs[:, 1], 1])
            limit = np.maximum(radii[pairs[:, 0]], radii[pairs[:, 1]]) * factor
            limit = np.maximum(limit, min_distance)
            order = np.argsort(dist)
            alive = np.ones(n, dtype=bool)
            for k in order:
                i, j = int(pairs[k, 0]), int(pairs[k, 1])
                if not (alive[i] and alive[j]) or dist[k] > limit[k]:
                    continue
                self._merge_into(i, j)
                alive[j] = False
                merged += 1
            if merged:
                self.keep(alive)
        if strangers and len(self) >= 3:
            merged += self._merge_own_among_strangers()
        return merged

    def _merge_into(self, i: int, j: int) -> None:
        gi, gj = self.gamma[i], self.gamma[j]
        if gi * gj > 0:
            total = gi + gj
            self.x[i] = (self.x[i] * gi + self.x[j] * gj) / total
            self.y[i] = (self.y[i] * gi + self.y[j] * gj) / total
        elif abs(gj) > abs(gi):
            self.x[i] = self.x[j]
            self.y[i] = self.y[j]
        self.gamma[i] = gi + gj
        self.gamma[j] = 0.0

    def _merge_own_among_strangers(self) -> int:
        """A domain whose two nearest neighbours share a sign opposite to its own is absorbed by them."""
        pts = self.positions
        k = min(3, len(self))
        if k < 3:
            return 0
        _, idx = cKDTree(pts).query(pts, k=3)
        sign = np.sign(self.gamma)
        n1 = idx[:, 1]
        n2 = idx[:, 2]
        odd = (sign[n1] == sign[n2]) & (sign[n1] != sign) & (sign != 0.0) & (sign[n1] != 0.0)
        if not np.any(odd):
            return 0
        dropped = 0
        alive = np.ones(len(self), dtype=bool)
        for i in np.flatnonzero(odd):
            j = int(n1[i])
            if not (alive[i] and alive[j]):
                continue
            self._merge_into(j, int(i))
            alive[i] = False
            dropped += 1
        if dropped:
            self.keep(alive)
        return dropped

    # -- diagnostics ------------------------------------------------------------------------------------
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self) == 0:
            return np.zeros(2), np.zeros(2)
        return np.array([self.x.min(), self.y.min()]), np.array([self.x.max(), self.y.max()])

    def __repr__(self) -> str:                                         # pragma: no cover - debugging aid
        return f"Domains(n={len(self)}, gamma={self.total_circulation:+.4g})"
