"""Deterministic timestamp alignment for digital twin predictions and sensors.

All timestamps are Unix epoch seconds (UTC). The aligner does not interpolate
measurements, and refuses to label stale observations as synchronized.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .inference_monitor import DigitalTwinMonitor


@dataclass(frozen=True)
class TwinSample:
    timestamp: float
    values: Mapping[str, float]

    def __post_init__(self):
        if not math.isfinite(self.timestamp):
            raise ValueError("invalid UTC timestamp")
        if any(not isinstance(v, (float, int)) or isinstance(v, bool)
               or not math.isfinite(v) for v in self.values.values()):
            raise ValueError("non-finite sensor or prediction")


class TwinSynchronizer:
    """Align a predicted state with the nearest measured sample in tolerance."""

    def __init__(self, twin_id: str, publish, *, tolerance_seconds: float = 0.25,
                 max_buffer: int = 1000):
        if not 0 < tolerance_seconds <= 3600 or not 1 <= max_buffer <= 100_000:
            raise ValueError("invalid alignment bounds")
        self.monitor = DigitalTwinMonitor(twin_id, publish)
        self.tolerance = tolerance_seconds
        self.max_buffer = max_buffer
        self.samples: list[TwinSample] = []
        self.step = 0

    def observe(self, timestamp: float, values: Mapping[str, float]) -> None:
        self.samples.append(TwinSample(timestamp, dict(values)))
        self.samples.sort(key=lambda sample: sample.timestamp)
        if len(self.samples) > self.max_buffer:
            del self.samples[:len(self.samples) - self.max_buffer]

    def predict(self, timestamp: float, values: Mapping[str, float]) -> bool:
        prediction = TwinSample(timestamp, dict(values))
        nearest = min(self.samples, key=lambda s: abs(s.timestamp - timestamp),
                      default=None)
        matched = nearest is not None and abs(nearest.timestamp - timestamp) <= self.tolerance
        self.monitor.state(self.step, prediction.values,
                           nearest.values if matched else None)
        self.monitor.emitter.emit("sync", self.step,
                                  prediction_timestamp=timestamp,
                                  observation_timestamp=nearest.timestamp if matched else -1.0,
                                  synchronized=1 if matched else 0,
                                  lag_seconds=abs(nearest.timestamp - timestamp) if matched else -1.0)
        self.step += 1
        return matched
