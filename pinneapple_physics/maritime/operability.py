"""The operability window: the criterion, the hours it opens and closes, and the confidence of the verdict.

The criterion is the operator's, not the physics: a significant height the crane is certified to work in, an
acceleration the lifted load is allowed to see, and -- where a hull can slam -- the relative velocity at the bow. Each
of the three is a number somebody is responsible for, so they are parameters with defaults that are illustrative and
say so: 1.5 m and 0.2 g are the kind of limit a 40 m support vessel works to, and they are not a standard.

What the package adds is the shape of the answer over a horizon. A forecast is good for a while and then it is not, so
a window is not "yes" or "no": it is "yes, until hour 14, with the confidence the forecast was measured to have at hour
14". That confidence is a measured number, not a curve somebody chose -- :func:`decision_skill` counts, on a record the
model was not fitted on, how often the verdict the forecast gave matched the verdict the observation would have given,
at each lead time. A window is then quoted with the probability that it is right.

One criterion failing closes the window; the report says which one, because "wait for the height to drop" and "wait for
the period to lengthen" are different instructions to a deck crew.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spectrum as sp

__all__ = ["Criteria", "Window", "windows", "decision_skill", "verdict", "opening"]

G = sp.G


@dataclass(frozen=True)
class Criteria:
    """The operating limits, in the units the deck uses.

    ``hs_max`` [m] and ``acceleration_max`` [g] are illustrative defaults, not a classification rule: they are the order
    of magnitude a crane on a 40 m support vessel works to. ``relative_velocity_max`` [m/s] defaults to Ochi's slamming
    threshold for a 60 m vessel; set it to ``None`` to drop the slamming criterion.
    """

    hs_max: float = 1.5
    acceleration_max: float = 0.2          # g
    relative_velocity_max: float | None = None
    length: float = 60.0                   # m, for the slamming threshold when none is given

    def __post_init__(self) -> None:
        if self.hs_max <= 0 or self.acceleration_max <= 0:
            raise ValueError("the limits have to be positive")

    @property
    def slam_limit(self) -> float | None:
        if self.relative_velocity_max is None and self.length <= 0:
            return None
        return self.relative_velocity_max if self.relative_velocity_max is not None else \
            0.093 * np.sqrt(G * self.length)

    def allows(self, hs, acceleration, relative_velocity=None) -> np.ndarray:
        """The verdict at each hour: ``True`` where every criterion is inside its limit."""
        hs = np.atleast_1d(np.asarray(hs, dtype=float))
        acceleration = np.atleast_1d(np.asarray(acceleration, dtype=float))
        ok = (hs <= self.hs_max) & (np.abs(acceleration) / G <= self.acceleration_max)
        limit = self.slam_limit
        if limit is not None and relative_velocity is not None:
            ok = ok & (np.abs(np.atleast_1d(np.asarray(relative_velocity, dtype=float))) <= limit)
        return ok

    def reasons(self, hs, acceleration, relative_velocity=None) -> list[str]:
        """Which criterion is the one that fails, per hour -- ``""`` when none does."""
        hs = np.atleast_1d(np.asarray(hs, dtype=float))
        acceleration = np.atleast_1d(np.asarray(acceleration, dtype=float)) / G
        out = []
        limit = self.slam_limit
        velocities = None if relative_velocity is None else np.abs(np.atleast_1d(np.asarray(relative_velocity)))
        for i in range(hs.size):
            breaking = []
            if hs[i] > self.hs_max:
                breaking.append(f"Hs {hs[i]:.1f} m > {self.hs_max:.1f} m")
            if acceleration[i] > self.acceleration_max:
                breaking.append(f"acceleration {acceleration[i]:.2f} g > {self.acceleration_max:.2f} g")
            if limit is not None and velocities is not None and velocities[i] > limit:
                breaking.append(f"relative velocity {velocities[i]:.2f} m/s > {limit:.2f} m/s")
            out.append("; ".join(breaking))
        return out


@dataclass(frozen=True)
class Window:
    """One contiguous run of hours with the same verdict, and how much the forecast behind it is worth."""

    start: float
    end: float
    allowed: bool
    confidence: float
    reason: str = ""

    @property
    def hours(self) -> float:
        return self.end - self.start

    def text(self) -> str:
        if self.allowed:
            return f"allowed from +{self.start:.0f} h to +{self.end:.0f} h (confidence {self.confidence:.0%})"
        return (f"stand down from +{self.start:.0f} h to +{self.end:.0f} h (confidence {self.confidence:.0%}):"
                f" {self.reason}")


def windows(criteria: Criteria, hours, hs, acceleration, confidence=None, relative_velocity=None) -> list[Window]:
    """Cut a forecast into runs of the same verdict.

    ``confidence`` is either a scalar, an array with one value per hour, or a callable of the lead time -- the last is
    what :func:`decision_skill` produces. A window is quoted with the confidence of its **last** hour, because that is
    the hour the crew is relying on it for.
    """
    hours = np.atleast_1d(np.asarray(hours, dtype=float))
    verdicts = criteria.allows(hs, acceleration, relative_velocity)
    reasons = criteria.reasons(hs, acceleration, relative_velocity)
    if confidence is None:
        level = np.ones_like(hours)
    elif callable(confidence):
        level = np.asarray([confidence(float(h)) for h in hours], dtype=float)
    else:
        level = np.broadcast_to(np.asarray(confidence, dtype=float), hours.shape)

    out: list[Window] = []
    start = 0
    for i in range(1, hours.size + 1):
        if i == hours.size or verdicts[i] != verdicts[start]:
            out.append(Window(start=float(hours[start]), end=float(hours[i - 1]), allowed=bool(verdicts[start]),
                              confidence=float(level[i - 1]), reason=reasons[i - 1] if not verdicts[start] else ""))
            start = i
    return out


def opening(criteria: Criteria, hours, hs, acceleration, confidence=None, relative_velocity=None) -> dict[str, object]:
    """The one sentence the deck needs: how long the window stays open, and how much to believe it."""
    found = windows(criteria, hours, hs, acceleration, confidence=confidence, relative_velocity=relative_velocity)
    open_now = found[0].allowed if found else False
    first_closed = next((w for w in found if not w.allowed), None)
    return {"open_now": open_now, "until_hours": first_closed.start if first_closed else float(hours[-1]),
            "confidence": (found[0].confidence if found else 0.0) if first_closed is None else first_closed.confidence,
            "reason": first_closed.reason if first_closed else "", "windows": found,
            "sentence": _sentence(open_now, first_closed, hours, found)}


def _sentence(open_now: bool, first_closed, hours, found) -> str:
    if not found:
        return "no forecast"
    horizon = float(hours[-1])
    if open_now and first_closed is None:
        return f"work is allowed for the whole {horizon:.0f} h forecast (confidence {found[0].confidence:.0%})"
    if open_now and first_closed is not None:
        return (f"work is allowed until +{first_closed.start:.0f} h, then stand down ({first_closed.reason}); "
                f"confidence at that hour {first_closed.confidence:.0%}")
    return f"stand down now: {found[0].reason} (confidence {found[0].confidence:.0%})"


def decision_skill(verdicts: np.ndarray, truth: np.ndarray) -> float:
    """The fraction of hours the forecast's verdict matched the observed one -- the confidence of a window."""
    verdicts = np.atleast_1d(np.asarray(verdicts, dtype=bool))
    truth = np.atleast_1d(np.asarray(truth, dtype=bool))
    return float(np.mean(verdicts == truth)) if verdicts.size else float("nan")


def verdict(criteria: Criteria, hs, acceleration, relative_velocity=None) -> np.ndarray:
    """``Criteria.allows`` as a function, for the records that want a one-liner."""
    return criteria.allows(hs, acceleration, relative_velocity)
