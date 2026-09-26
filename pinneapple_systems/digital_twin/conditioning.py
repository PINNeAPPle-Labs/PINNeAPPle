"""Causal, sample-by-sample telemetry conditioning for production digital twins.

``signal_reconstruction`` cleans a signal *after the fact* (FFT, SSA, wavelets, interpolation over
the whole history). A running twin cannot wait for the future: every reading must be conditioned
the moment it arrives, before it reaches the model or the state estimator. This module does that,
causally, per signal, with settings that depend on what the signal is (a thermocouple is slow and
smooth, a pressure transmitter fast and spiky, a valve command must not be filtered at all).

Pipeline per reading (each step flags what it did, nothing is changed silently):

1. **range check**: outside ``valid_range`` -> treated as missing (``out_of_range``).
2. **outlier rejection**: Hampel test made causal. A plain median of *past* readings lags a
   trending signal by half a window and flags good readings, so the reading is compared with a
   one-step prediction instead: last accepted value + median of the recent rates of change x dt.
   Scale: 1.4826 * MAD of those rates x dt, floored by the median |rate| x dt. Beyond
   ``hampel_k`` scales -> replaced by the prediction (``outlier``). Spikes are short: after
   ``max_consecutive_outliers`` rejections in a row the reading is accepted (a real step change
   in the process, ``level_shift``) and the rate history restarts.
3. **rate limit**: change per second capped at ``max_rate`` (``rate_limited``).
4. **low-pass**: first-order filter with time constant ``tau_s``; the gain uses the actual
   time step (``alpha = 1 - exp(-dt / tau)``), so irregular sampling is handled correctly.
5. **gaps**: ``fill(t)`` when no reading arrives: holds the last value up to ``max_gap_s``
   (``gap_filled``), then returns NaN (``stale``) so downstream code knows the sensor is gone.

Motivation: R. Belbachir, "a deployable industrial AI pipeline begins before the model" (2026,
see ``ROADMAP.md`` §11). Methods: Hampel identifier (Hampel, J. Am. Stat. Assoc. 69 (1974) 383;
Pearson, "Outliers in process modeling and identification", IEEE TCST 10(1) (2002) 55-63) and
the first-order low-pass filter.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SignalProfile:
    """How to condition one kind of signal. Use a preset (``PROFILES``) or build your own."""

    kind: str = "generic"
    tau_s: float = 0.0  # low-pass time constant; 0 disables filtering
    hampel_window: int = 9  # recent accepted readings kept for the outlier test; 0 disables it
    hampel_k: float = 3.0
    max_consecutive_outliers: int = 3
    max_rate: Optional[float] = None  # units per second
    valid_range: Optional[Tuple[float, float]] = None
    max_gap_s: float = 60.0


PROFILES: Dict[str, SignalProfile] = {
    # slow, smooth: strong smoothing, tight rate limit, gaps tolerated for minutes
    "thermal": SignalProfile("thermal", tau_s=30.0, hampel_window=11, hampel_k=3.0, max_gap_s=300.0),
    # fast, spiky: light smoothing so transients survive, outliers removed
    "pressure": SignalProfile("pressure", tau_s=1.0, hampel_window=9, hampel_k=3.5, max_gap_s=10.0),
    "flow": SignalProfile("flow", tau_s=3.0, hampel_window=9, hampel_k=3.0, max_gap_s=30.0),
    "vibration": SignalProfile("vibration", tau_s=0.0, hampel_window=0, max_gap_s=2.0),
    # commands and setpoints are exact by definition: never filtered, never "corrected"
    "control": SignalProfile("control", tau_s=0.0, hampel_window=0, max_gap_s=60.0),
}


@dataclass
class ConditionedSample:
    t: float
    raw: float
    value: float
    flags: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return math.isfinite(self.value) and "stale" not in self.flags


class StreamingConditioner:
    """Condition one signal, one reading at a time (causal; O(window) per reading)."""

    def __init__(self, profile: SignalProfile | str = "generic", **overrides):
        base = PROFILES.get(profile, SignalProfile()) if isinstance(profile, str) else profile
        self.profile = replace(base, **overrides) if overrides else base
        self._rates: Deque[float] = deque(maxlen=max(self.profile.hampel_window, 1))
        self._acc: Optional[Tuple[float, float]] = None  # last accepted (t, value) for the outlier test
        self._n_out = 0
        self._t: Optional[float] = None
        self._y: Optional[float] = None
        self._last_real_t: Optional[float] = None

    def reset(self) -> None:
        self._rates.clear()
        self._acc = None
        self._n_out = 0
        self._t = self._y = self._last_real_t = None

    def update(self, t: float, raw: Optional[float]) -> ConditionedSample:
        p = self.profile
        flags: List[str] = []
        x = float("nan") if raw is None else float(raw)
        if math.isfinite(x) and p.valid_range is not None and not (p.valid_range[0] <= x <= p.valid_range[1]):
            flags.append("out_of_range")
            x = float("nan")
        if not math.isfinite(x):
            s = self.fill(t)
            s.raw = float("nan") if raw is None else float(raw)
            s.flags = flags + s.flags
            return s

        if p.hampel_window and self._acc is not None and t > self._acc[0]:
            t0, v0 = self._acc
            h = t - t0
            if len(self._rates) >= min(4, p.hampel_window):
                r = np.fromiter(self._rates, float)
                med = float(np.median(r))
                scale = max(1.4826 * float(np.median(np.abs(r - med))), float(np.median(np.abs(r)))) * h
                scale = max(scale, 1e-9 * (1.0 + abs(v0)))
                pred = v0 + med * h
                if abs(x - pred) > p.hampel_k * scale and self._n_out < p.max_consecutive_outliers:
                    flags.append("outlier")
                    self._n_out += 1
                    x = pred
                elif abs(x - pred) > p.hampel_k * scale:
                    flags.append("level_shift")
                    self._rates.clear()
                    self._n_out = 0
                else:
                    self._n_out = 0
            if "outlier" not in flags and "level_shift" not in flags:
                self._rates.append((x - v0) / h)
        self._acc = (t, x)

        dt = None if self._t is None else max(t - self._t, 0.0)
        if self._y is not None and p.max_rate is not None and dt is not None:
            lim = p.max_rate * dt
            if abs(x - self._y) > lim:
                flags.append("rate_limited")
                x = self._y + math.copysign(lim, x - self._y)
        if self._y is None or not p.tau_s or dt is None:
            y = x
        else:
            alpha = 1.0 - math.exp(-dt / p.tau_s)
            y = self._y + alpha * (x - self._y)
        self._t, self._y, self._last_real_t = t, y, t
        return ConditionedSample(t, float(raw), y, flags)

    def fill(self, t: float) -> ConditionedSample:
        """Value to use at time ``t`` when no valid reading arrived."""
        if self._y is None or self._last_real_t is None:
            return ConditionedSample(t, float("nan"), float("nan"), ["stale"])
        if t - self._last_real_t <= self.profile.max_gap_s:
            return ConditionedSample(t, float("nan"), self._y, ["gap_filled"])
        return ConditionedSample(t, float("nan"), float("nan"), ["stale"])


class TelemetryConditioner:
    """Per-sensor, per-field conditioners for a whole twin.

    ``profiles`` maps ``"sensor_id.field"``, ``"field"`` or ``"sensor_id"`` to a profile name or
    ``SignalProfile`` (most specific wins); anything unmapped uses ``default``.
    """

    def __init__(self, profiles: Optional[Dict[str, SignalProfile | str]] = None,
                 default: SignalProfile | str = "generic"):
        self.profiles = dict(profiles or {})
        self.default = default
        self._cond: Dict[Tuple[str, str], StreamingConditioner] = {}
        self.counts: Dict[str, int] = {}

    def _get(self, sensor_id: str, name: str) -> StreamingConditioner:
        key = (sensor_id, name)
        if key not in self._cond:
            prof = self.profiles.get(f"{sensor_id}.{name}", self.profiles.get(name, self.profiles.get(sensor_id, self.default)))
            self._cond[key] = StreamingConditioner(prof)
        return self._cond[key]

    def process(self, sensor_id: str, t: float, values: Dict[str, float]) -> Dict[str, ConditionedSample]:
        out = {}
        for name, v in values.items():
            s = self._get(sensor_id, name).update(t, v)
            for f in s.flags:
                self.counts[f] = self.counts.get(f, 0) + 1
            out[name] = s
        return out

    def condition_observation(self, obs):
        """Return a copy of an ``Observation`` with conditioned values; flags go to ``metadata``."""
        from .state import Observation
        res = self.process(obs.sensor_id, obs.timestamp, obs.values)
        meta = dict(obs.metadata)
        meta["conditioning"] = {k: s.flags for k, s in res.items() if s.flags}
        meta["raw_values"] = dict(obs.values)
        return Observation(timestamp=obs.timestamp, sensor_id=obs.sensor_id,
                           values={k: s.value for k, s in res.items()}, coords=obs.coords, metadata=meta)
