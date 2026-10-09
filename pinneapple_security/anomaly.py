"""Physics-based detection of manipulated sensor data (false data injection, spoofing, replay) in industrial streams.

An attacker who changes a sensor reading keeps it plausible on its own; what breaks is consistency with the physics
linking several measurements (mass and energy balances, a pump curve, a heat-exchanger model, a digital-twin
surrogate). The detector watches the *physics residual* r_t = measured − predicted (or a balance that should be zero):

1. :meth:`ResidualCUSUM.fit` learns the residual's normal mean and spread from trusted data;
2. two-sided CUSUM (Page 1954) accumulates standardised deviations beyond a slack ``k`` and raises an alarm when the
   sum crosses ``h`` — small persistent biases (a slow ramp attack) are caught, not only big jumps;
3. :func:`replay_score` flags a window that repeats an earlier window too exactly (a replayed recording has the
   same noise; a live sensor does not).

The defaults (k = 0.5, h = 5 in units of σ) give an in-control average run length of roughly 465 samples for a
standard normal residual (Montgomery, Introduction to Statistical Quality Control, Table 9.3; it varies with the
edition); tune ``h`` to the false-alarm rate the plant accepts.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ResidualCUSUM", "balance_residual", "replay_score"]


def balance_residual(inflows: Sequence[Sequence[float]], outflows: Sequence[Sequence[float]],
                     accumulation: Sequence[float] | None = None) -> np.ndarray:
    """Conservation residual Σin − Σout − d(storage)/dt per sample (mass, energy or charge balance)."""
    r = np.sum(np.atleast_2d(inflows), axis=0) - np.sum(np.atleast_2d(outflows), axis=0)
    if accumulation is not None:
        r = r - np.asarray(accumulation, dtype=float)
    return r


@dataclass
class ResidualCUSUM:
    k: float = 0.5
    h: float = 5.0
    mu: float = 0.0
    sigma: float = 1.0
    fitted: bool = False
    s_hi: float = 0.0
    s_lo: float = 0.0
    t: int = 0
    alarms: list[dict[str, Any]] = field(default_factory=list)

    def fit(self, residuals: Sequence[float]) -> ResidualCUSUM:
        r = np.asarray(residuals, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 30:
            raise ValueError("fit on at least 30 trusted residuals")
        med = np.median(r)
        mad = 1.4826 * np.median(np.abs(r - med))                  # robust: a few bad points in "trusted" data are ok
        self.mu, self.sigma, self.fitted = float(med), float(mad if mad > 0 else r.std() or 1.0), True
        self.reset()
        return self

    def reset(self) -> None:
        self.s_hi = self.s_lo = 0.0
        self.t = 0
        self.alarms = []

    def update(self, r: float) -> dict[str, Any] | None:
        if not self.fitted:
            raise RuntimeError("call fit() on trusted residuals first")
        z = (float(r) - self.mu) / self.sigma
        self.s_hi = max(0.0, self.s_hi + z - self.k)
        self.s_lo = max(0.0, self.s_lo - z - self.k)
        alarm = None
        if self.s_hi > self.h or self.s_lo > self.h:
            alarm = {"t": self.t, "direction": "high" if self.s_hi > self.h else "low",
                     "statistic": max(self.s_hi, self.s_lo), "z": z}
            self.alarms.append(alarm)
            self.s_hi = self.s_lo = 0.0                            # restart after an alarm
        self.t += 1
        return alarm

    def run(self, residuals: Sequence[float]) -> dict[str, Any]:
        self.reset()
        for r in residuals:
            self.update(r)
        first = self.alarms[0]["t"] if self.alarms else None
        return {"n": self.t, "n_alarms": len(self.alarms), "first_alarm": first, "alarms": list(self.alarms)}


def replay_score(x: Sequence[float], window: int = 50) -> dict[str, Any]:
    """Smallest normalised distance between the last ``window`` samples and any earlier window. Near 0 → the latest
    data repeats an old recording sample for sample (replay attack) — live noise never matches that closely."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2 * window:
        raise ValueError("need at least two windows of data")
    last = x[-window:]
    scale = np.std(x) or 1.0
    best, at = np.inf, -1
    for s in range(0, len(x) - 2 * window + 1):
        d = np.sqrt(np.mean((x[s:s + window] - last) ** 2)) / scale
        if d < best:
            best, at = d, s
    return {"min_distance": float(best), "matches_window_start": int(at), "suspicious": bool(best < 1e-3)}
