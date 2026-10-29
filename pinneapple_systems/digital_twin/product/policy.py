"""What the twin does when it stops agreeing with the plant.

A twin that only draws a line is a dashboard. A twin that knows it has drifted is a product:
the operator needs to be told *before* the prediction is worthless, and the system needs a
defined thing to do about it.

The signal is **band consistency**, not raw error. The filter's own 95 % band is the twin's
claim about itself; when held-out measurements fall outside it far more often than 5 % of the
time, the twin's claim is false and no threshold on the error would have said so — the error
can be within tolerance while the band is a lie.

The ladder
----------
``NORMAL``      the band contains reality at the declared rate.
``WATCH``       consistency has been off for ``max_sustained`` steps; keep running, tell someone.
``ALARM``       consistency broke hard; the estimate is not to be used for decisions.
``RECALIBRATE`` the state has been unusable for ``recalibrate_after`` steps; re-run
                commissioning on recent data before trusting it again.

Limits
------
This detects that the twin is wrong, not why. A fouled surface, a drifting instrument and a
genuinely new operating regime all look the same here; only a human or a second model can
tell them apart. The policy therefore never silently retrains — it states the mode and stops.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Mapping, Sequence

__all__ = ["TwinMode", "DriftPolicy", "DriftMonitor"]


class TwinMode(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    ALARM = "ALARM"
    RECALIBRATE = "RECALIBRATE"


_RANK = {TwinMode.NORMAL: 0, TwinMode.WATCH: 1, TwinMode.ALARM: 2, TwinMode.RECALIBRATE: 3}


@dataclass(frozen=True)
class DriftPolicy:
    """Thresholds for the consistency ladder.

    ``target_coverage`` should match the band the estimator produces (0.90 for a 90 %
    interval, 0.95 for 95 %). ``hard_break`` is the coverage that ends the argument: below it
    the twin is not mildly off, it is describing a different machine.
    """

    window: int = 60
    target_coverage: float = 0.90
    hard_break: float = 0.50
    max_sustained: int = 5
    recalibrate_after: int = 30
    hysteresis: int = 10

    def __post_init__(self) -> None:
        if self.window < 5:
            raise ValueError("window must be at least 5 steps to mean anything")
        if not 0.0 < self.hard_break <= self.target_coverage <= 1.0:
            raise ValueError("need 0 < hard_break <= target_coverage <= 1")
        if min(self.max_sustained, self.recalibrate_after, self.hysteresis) < 1:
            raise ValueError("run lengths must be at least 1")


@dataclass
class DriftMonitor:
    """Rolling consistency of the twin's band, converted into a mode with hysteresis.

    Feed it one boolean per step per held-out sensor — ``in_band`` — and read
    :attr:`mode`. Every transition is appended to :attr:`decisions` so the evidence file says
    what the twin thought of itself, and when.
    """

    policy: DriftPolicy = field(default_factory=DriftPolicy)
    _window: Deque[bool] = field(default_factory=deque, repr=False)
    _mode: TwinMode = TwinMode.NORMAL
    _off_run: int = 0
    _alarm_run: int = 0
    _ok_run: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._window = deque(maxlen=self.policy.window)

    @property
    def mode(self) -> TwinMode:
        return self._mode

    @property
    def coverage(self) -> float | None:
        if not self._window:
            return None
        return sum(self._window) / len(self._window)

    def observe(self, in_band: Sequence[bool], *, t: float | None = None) -> TwinMode:
        """Record one instant's band consistency and return the resulting mode."""
        if len(in_band) == 0:
            return self._mode
        hit_rate = sum(bool(b) for b in in_band) / len(in_band)
        self._window.append(hit_rate >= 0.5)

        p = self.policy
        cov = self.coverage
        assert cov is not None

        if cov < p.target_coverage:
            self._off_run += 1
            self._ok_run = 0
        else:
            self._ok_run += 1
            self._off_run = 0

        proposed = self._mode
        if cov < p.hard_break:
            proposed = TwinMode.ALARM
            self._alarm_run += 1
        elif self._off_run >= p.max_sustained:
            proposed = TwinMode.WATCH
            self._alarm_run = 0
        elif self._ok_run >= p.hysteresis:
            proposed = TwinMode.NORMAL
            self._alarm_run = 0
        elif self._mode in (TwinMode.ALARM, TwinMode.RECALIBRATE):
            proposed = self._mode  # do not fall straight out of an alarm on one good run

        if proposed is TwinMode.ALARM and self._alarm_run >= p.recalibrate_after:
            proposed = TwinMode.RECALIBRATE
        if proposed is not TwinMode.ALARM:
            if self._mode is TwinMode.RECALIBRATE and proposed is not TwinMode.RECALIBRATE:
                self._alarm_run = 0

        if proposed is not self._mode and _RANK[proposed] != _RANK[self._mode]:
            self.decisions.append(
                {
                    "t": t,
                    "from": self._mode.value,
                    "to": proposed.value,
                    "coverage": round(cov, 4),
                    "step": len(self.decisions),
                }
            )
            self._mode = proposed
        return self._mode

    def summary(self) -> dict[str, Any]:
        return {
            "policy": {
                "window": self.policy.window,
                "target_coverage": self.policy.target_coverage,
                "hard_break": self.policy.hard_break,
                "max_sustained": self.policy.max_sustained,
                "recalibrate_after": self.policy.recalibrate_after,
                "hysteresis": self.policy.hysteresis,
            },
            "mode": self._mode.value,
            "coverage": self.coverage,
            "transitions": len(self.decisions),
            "decisions": list(self.decisions),
        }
