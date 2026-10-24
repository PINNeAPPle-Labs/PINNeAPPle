"""Where the surrogate has evidence: the validated domain of a set of simulated designs.

A prediction is only as good as the simulations near it. :class:`ValidatedDomain` measures, for any design in the
unit cube, the distance to the nearest simulated design in units of the typical spacing of the simulated designs
(the median nearest-neighbour distance among them). A design is inside the validated domain when

* that ratio is at most ``max_ratio`` (default 1.5: no farther from the data than the data are from each other,
  with a margin), and
* every variable lies within the range the simulations covered, widened by ``margin`` of the full range.

The reasons a design is outside are returned in words with the numbers ("rear_flap_aoa = 15.1 is below the
simulated range 17.4 to 39.0"), so a report or an agent can say why it asked for another simulation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["ValidatedDomain", "DomainVerdict"]


@dataclass
class DomainVerdict:
    inside: bool
    ratio: float                 # distance to the nearest simulated design / typical spacing
    nearest: int                 # index of the nearest simulated design
    reasons: list[str]


class ValidatedDomain:
    def __init__(self, U: np.ndarray, names: Sequence[str] | None = None, bounds: np.ndarray | None = None,
                 max_ratio: float = 1.5, margin: float = 0.05):
        self.U = np.atleast_2d(np.asarray(U, float))
        if len(self.U) < 2:
            raise ValueError("a validated domain needs at least 2 simulated designs")
        self.names = list(names or [f"x{i}" for i in range(self.U.shape[1])])
        self.bounds = None if bounds is None else np.asarray(bounds, float)
        self.max_ratio, self.margin = float(max_ratio), float(margin)
        D = np.linalg.norm(self.U[:, None] - self.U[None], axis=-1)
        np.fill_diagonal(D, np.inf)
        self.spacing = float(np.median(D.min(1)))
        self.lo, self.hi = self.U.min(0), self.U.max(0)

    def ratio(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        D = np.linalg.norm(np.atleast_2d(U)[:, None] - self.U[None], axis=-1)
        j = D.argmin(1)
        return D[np.arange(len(D)), j] / max(self.spacing, 1e-12), j

    def assess(self, U: np.ndarray) -> list[DomainVerdict]:
        U = np.atleast_2d(np.asarray(U, float))
        r, j = self.ratio(U)
        out = []
        for u, ri, ji in zip(U, r, j, strict=True):
            reasons = []
            for i, nm in enumerate(self.names):
                if u[i] < self.lo[i] - self.margin or u[i] > self.hi[i] + self.margin:
                    side = "below" if u[i] < self.lo[i] else "above"
                    reasons.append(f"{nm} = {self._phys(i, u[i]):.4g} is {side} the simulated range "
                                   f"{self._phys(i, self.lo[i]):.4g} to {self._phys(i, self.hi[i]):.4g}")
            if ri > self.max_ratio:
                reasons.append(f"nearest simulated design is {ri:.2f} x the typical spacing away "
                               f"(limit {self.max_ratio:g})")
            out.append(DomainVerdict(not reasons, float(ri), int(ji), reasons))
        return out

    def _phys(self, i: int, u: float) -> float:
        if self.bounds is None:
            return float(u)
        return float(self.bounds[i, 0] + u * (self.bounds[i, 1] - self.bounds[i, 0]))
