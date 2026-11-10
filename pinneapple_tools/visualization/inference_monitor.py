"""Model inference and digital-twin observability adapters.

This module publishes measurements supplied by the caller. It does not infer
physical correctness, fabricate ground truth, or execute an untrusted model.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .telemetry import TelemetryEmitter


def _finite(values: Mapping[str, float]) -> dict[str, float]:
    result = {}
    for key, value in values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("model telemetry must be finite numeric values")
        result[str(key)] = float(value)
    return result


class InferenceMonitor:
    """Publish predictions and measured values to a model-scoped dashboard."""

    def __init__(self, model_id: str, publish):
        self.emitter = TelemetryEmitter("model", model_id, publish)

    def prediction(self, step: int, predictions: Mapping[str, float], *,
                   latency_ms: float | None = None,
                   uncertainty: Mapping[str, float] | None = None,
                   observed: Mapping[str, float] | None = None):
        self.emitter.emit("inference", step, **_finite(predictions))
        if latency_ms is not None:
            self.emitter.emit("latency", step, latency_ms=_finite({"latency_ms": latency_ms})["latency_ms"])
        if uncertainty is not None:
            self.emitter.emit("uncertainty", step, **_finite(uncertainty))
        if observed is not None:
            # Prefixes distinguish ground truth/measurements from predictions.
            state = {f"pred_{k}": v for k, v in _finite(predictions).items()}
            state.update({f"obs_{k}": v for k, v in _finite(observed).items()})
            self.emitter.emit("state", step, **state)

    def physical_checks(self, step: int, residuals: Mapping[str, float]):
        self.emitter.emit("checks", step, **_finite(residuals))

    def drift(self, step: int, statistics: Mapping[str, float]):
        self.emitter.emit("drift", step, **_finite(statistics))


class DigitalTwinMonitor:
    """Track simulated/estimated twin state and independently measured state."""

    def __init__(self, twin_id: str, publish):
        self.emitter = TelemetryEmitter("model", twin_id, publish)

    def state(self, step: int, predicted: Mapping[str, float],
              measured: Mapping[str, float] | None = None):
        values = {f"pred_{k}": v for k, v in _finite(predicted).items()}
        if measured is not None:
            values.update({f"obs_{k}": v for k, v in _finite(measured).items()})
        self.emitter.emit("state", step, **values)
