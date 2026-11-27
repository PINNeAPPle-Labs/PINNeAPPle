"""Whether the model's residual is a new regime or a sensor that is lying.

The residual is the arithmetic first: what the model of the motion predicts the instruments should have read, minus what
they read, in units of the instrument's own noise. A residual inside a few sigma is a sea the model describes; a
residual outside it is one of three things, and they are told apart by *which* channels disagree and by what the
physics says was happening at the time:

* **slamming.** The hull at the bow is coming down faster than the surface is rising. That is not a residual at all --
  it is a kinematic quantity, the relative vertical velocity ``v_rel``, and the criterion is the classical one,
  ``v_rel`` above Ochi's ``0.093 sqrt(g L)``. The signature is an acceleration residual that lasts a fraction of a wave
  period, appears on every accelerometer at once (it is the whole hull being hit, not one instrument), and coincides
  with ``v_rel`` over the threshold;
* **a mooring line changing.** A line parts, a tether is shortened, a pretension is re-set. The signature is a *step* in
  the tension residual that stays, while the accelerometers stay inside their noise: the vessel is not being hit, the
  load path changed;
* **a sensor fault.** One instrument disagrees with the model *and* with the instrument that measures the same motion,
  or fails its own statistics -- a spike, a flat line, a drift with no physics behind it. The distinction that matters
  is redundancy: two accelerometers on one hull have to agree with each other, so when one disagrees with the model and
  the other does not, the vessel is fine and the instrument is not.

Everything here is measured on the recorded series; nothing is a label read from the simulation that produced them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Event", "Diagnosis", "evaluate", "sensor_checks", "normalised_residual", "storm_flag"]


@dataclass(frozen=True)
class Event:
    """One thing that happened: when, what, on which channel, and against which limit."""

    time: float
    kind: str                     # "slamming", "mooring", "sensor"
    channel: str
    value: float
    threshold: float
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.time:8.1f} s  {self.kind:<9} {self.channel:<22} {self.value:8.3f} against {self.threshold:8.3f}"


@dataclass(frozen=True)
class Diagnosis:
    """What the residuals and the sensor checks add up to over one record."""

    events: tuple[Event, ...] = ()
    worst_residual: dict[str, float] = field(default_factory=dict)
    sensor_faults: tuple[str, ...] = ()
    windows: int = 0

    def of_kind(self, kind: str) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.kind == kind)

    @property
    def real_events(self) -> tuple[Event, ...]:
        """The events the vessel had, as opposed to the instruments."""
        return tuple(e for e in self.events if e.kind != "sensor")

    def summary(self) -> dict[str, object]:
        return {"events": len(self.events), "slamming": len(self.of_kind("slamming")),
                "mooring": len(self.of_kind("mooring")), "sensor": len(self.of_kind("sensor")),
                "sensor_faults": list(self.sensor_faults), "worst_residual": self.worst_residual}


def normalised_residual(predicted, measured, noise: float) -> np.ndarray:
    """``(measured - predicted) / noise`` -- the residual in units of the instrument's own standard deviation."""
    predicted = np.asarray(predicted, dtype=float)
    measured = np.asarray(measured, dtype=float)
    return (measured - predicted) / float(noise)


def sensor_checks(series, noise: float, *, spike_sigma: float = 6.0, flat_m: float = 1e-6) -> dict[str, object]:
    """The checks an instrument has to pass on its own, without the model: spikes, a flat line, a drift.

    A stuck channel has no variance; a spiking one has a second difference far outside its noise; a drifting one has a
    mean that walks away from the model's over the record. The first two need no model at all, which is what makes them
    usable when the model is the thing in doubt.
    """
    values = np.asarray(series, dtype=float)
    if values.size < 4:
        return {"spikes": 0, "flat": False, "spike_rate": 0.0, "variance": 0.0}
    second = np.diff(values, 2)
    spikes = int(np.sum(np.abs(second) > spike_sigma * noise * math.sqrt(6.0)))
    flat = bool(np.std(values) < flat_m)
    return {"spikes": spikes, "flat": flat, "spike_rate": spikes / max(values.size - 2, 1),
            "variance": float(np.var(values))}


def _partner(pair: tuple[str, str], name: str) -> str:
    """The other instrument of a redundant pair."""
    return pair[1] if name == pair[0] else pair[0]


def storm_flag(relative_velocity, threshold: float) -> np.ndarray:
    """``True`` where the hull is closing on the water faster than the slamming threshold."""
    return np.abs(np.asarray(relative_velocity, dtype=float)) > float(threshold)


def evaluate(*, time, predicted: dict, measured: dict, noise: dict, relative_velocity=None,
             slamming_threshold: float | None = None, threshold_sigma: float = 4.0, share: float = 0.005,
             redundant: tuple[str, str] | None = None, window: float = 60.0) -> Diagnosis:
    """Read one record and decide, per window, whether what happened was the sea, the mooring, or an instrument.

    Parameters
    ----------
    time : array
        The record's time stamps, seconds.
    predicted, measured : dict of arrays
        The same channels from the motion model and from the instruments.
    noise : dict
        One standard deviation per channel, in the channel's unit.
    relative_velocity : array, optional
        ``v_rel`` from the motion model, for the slamming classification.
    slamming_threshold : float, optional
        The relative velocity that counts as a slam; Ochi's value for the vessel if not given.
    redundant : (str, str), optional
        Two channels that measure the same motion. When one disagrees with the model and the other does not, the one
        that disagrees is the fault.
    share : float
        The fraction of a window's samples that has to sit outside the threshold before it counts as an event. A single
        sample beyond four sigma happens by definition once in 30 000, and a window of a few hundred samples would fire
        on noise alone: the share is what keeps the quiet sea quiet.
    window : float
        Length of the window an event is reported for, seconds.
    """
    time = np.asarray(time, dtype=float)
    events: list[Event] = []
    worst: dict[str, float] = {}
    faults: list[str] = []
    edges = np.arange(time[0], time[-1] + window, window)

    residuals = {name: normalised_residual(predicted[name], measured[name], noise[name]) for name in predicted}
    for name, residual in residuals.items():
        worst[name] = float(np.max(np.abs(residual))) if residual.size else 0.0

    own = {name: sensor_checks(measured[name], noise[name]) for name in measured}
    for name, check in own.items():
        if check["flat"]:
            faults.append(name)
            events.append(Event(float(time[0]), "sensor", name, 0.0, 0.0, "the channel is flat: the instrument is stuck"))
        if check["spike_rate"] > 0.01:
            faults.append(name)
            events.append(Event(float(time[0]), "sensor", name, float(check["spikes"]), 0.0,
                                f"{check['spikes']} spikes on its own second difference"))

    for each in edges[:-1]:
        inside = (time >= each) & (time < each + window)
        if not np.any(inside):
            continue
        count = max(3, int(share * np.sum(inside)))
        outside = {name: np.abs(residual[inside]) > threshold_sigma for name, residual in residuals.items()}
        hitting = [name for name, flag in outside.items() if int(np.sum(flag)) >= count]
        if not hitting:
            continue
        slammed = bool(relative_velocity is not None and np.any(storm_flag(np.asarray(relative_velocity)[inside],
                                                                         slamming_threshold or 0.0)))
        for name in hitting:
            value = float(np.max(np.abs(residuals[name][inside])))
            stepped = bool(np.mean(residuals[name][inside]) != 0.0 and
                           np.sign(np.mean(residuals[name][inside][: inside.sum() // 2])) ==
                           np.sign(np.mean(residuals[name][inside][inside.sum() // 2:])))
            if name.lower().startswith(("tension", "mooring", "load")):
                kind, detail = "mooring", "the load path changed; the hull was not hit"
            elif redundant and name in redundant and _partner(redundant, name) not in hitting:
                kind, detail = "sensor", f"disagrees with the model while {_partner(redundant, name)} agrees with it"
                if name not in faults:
                    faults.append(name)
            elif slammed:
                kind, detail = "slamming", "the hull closed on the surface above the threshold"
            elif stepped:
                kind, detail = "mooring", "a persistent step in the residual"
            else:
                kind, detail = "sensor", "a residual with no physical event behind it"
            events.append(Event(float(each + window / 2.0), kind, name, value, threshold_sigma, detail))
    return Diagnosis(events=tuple(events), worst_residual=worst, sensor_faults=tuple(sorted(set(faults))),
                     windows=int(edges.size - 1))
