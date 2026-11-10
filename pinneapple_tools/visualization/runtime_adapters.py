"""Opt-in adapters for real training loops, FEM fields and model serving.

Adapters consume actual producer values; they never fabricate physics outputs.
They do not change the underlying training/solver/model execution.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .telemetry import TelemetryEmitter
from .inference_monitor import InferenceMonitor
from .scientific_artifacts import from_meshio, publish_surface


def _scalar(value: Any) -> float:
    """Accept a detached PyTorch scalar, NumPy scalar or native number."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected a finite scalar metric")
    return float(value)


class PINNTrainingCallback:
    """Call on_epoch_end from an existing PyTorch PINN trainer."""

    def __init__(self, run_id: str, publish: Callable[[dict], None]):
        self.emitter = TelemetryEmitter("training", run_id, publish)

    def on_epoch_end(self, epoch: int, *, losses: Mapping[str, Any],
                     optimizer: Any | None = None,
                     gradients: Mapping[str, Any] | None = None) -> None:
        self.emitter.emit("loss", epoch, **{k: _scalar(v) for k, v in losses.items()})
        if optimizer is not None:
            for group in optimizer.param_groups[:1]:
                self.emitter.emit("lr", epoch, learning_rate=_scalar(group["lr"]))
        if gradients is not None:
            self.emitter.emit("gradients", epoch, **{k: _scalar(v) for k, v in gradients.items()})


class FEMResultPublisher:
    """Publish an actual meshio surface and a named nodal scalar field."""

    def __init__(self, endpoint: str, api_key: str, case_id: str):
        self.endpoint, self.api_key, self.case_id = endpoint, api_key, case_id

    def publish_mesh(self, mesh: Any, *, field_name: str | None = None,
                     channel: str = "mesh", coordinate_unit: str = "m",
                     field_unit: str | None = None) -> dict:
        artifact = from_meshio(mesh, field_name=field_name, coordinate_unit=coordinate_unit)
        artifact["field_unit"] = field_unit
        return publish_surface(artifact, endpoint=self.endpoint, api_key=self.api_key,
                               source="simulation", resource_id=self.case_id, channel=channel)


class ModelServingObserver:
    """Explicit inference wrapper; execution stays in the trusted caller process."""

    def __init__(self, model_id: str, predict: Callable[[Any], Mapping[str, float]],
                 publish: Callable[[dict], None]):
        self.predict = predict
        self.monitor = InferenceMonitor(model_id, publish)
        self.step = 0

    def infer(self, inputs: Any, *, observed: Mapping[str, float] | None = None,
              uncertainty: Mapping[str, float] | None = None) -> Mapping[str, float]:
        start = time.perf_counter()
        result = self.predict(inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Validate before publishing, but never silently substitute a prediction.
        if not isinstance(result, Mapping):
            raise TypeError("predict must return named scalar predictions")
        self.monitor.prediction(self.step, result, observed=observed,
                                uncertainty=uncertainty, latency_ms=elapsed_ms)
        self.step += 1
        return result
