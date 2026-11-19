"""Optional dashboard telemetry producers for training and solver monitoring.

Network publication is opt-in, with a caller-supplied authenticated transport.
Never print or persist credentials. This module has no third-party dependencies.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable, Mapping

_METRIC = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.:-]{0,127}$")


class TelemetryEmitter:
    def __init__(self, source: str, resource_id: str, publish: Callable[[dict], None]):
        if source not in {"training", "simulation", "pipeline", "model", "rover", "job", "sensor"}:
            raise ValueError("unsupported telemetry source")
        if not _METRIC.fullmatch(resource_id):
            raise ValueError("invalid resource ID")
        self.source, self.resource_id, self.publish = source, resource_id, publish
        self.sequence = 0

    def emit(self, channel: str, step: int, **values: float | int | str) -> dict:
        if not _METRIC.fullmatch(channel) or isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("invalid telemetry channel/step")
        for key, value in values.items():
            if not _METRIC.fullmatch(key) or isinstance(value, bool) or not isinstance(value, (float, int, str)):
                raise ValueError("invalid metric")
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError("non-finite metric")
            if isinstance(value, str) and len(value) > 160:
                raise ValueError("metric label too long")
        self.sequence += 1
        event = {"source": self.source, "resource_id": self.resource_id, "sequence": self.sequence,
                 "channel": channel, "step": step, "values": values}
        self.publish(event)
        return event

    def training_step(self, epoch: int, losses: Mapping[str, float], lr: float | None = None):
        self.emit("loss", epoch, **{str(k): float(v) for k, v in losses.items()})
        if lr is not None:
            self.emit("lr", epoch, learning_rate=float(lr))

    def simulation_step(self, iteration: int, residuals: Mapping[str, float],
                        conservation: Mapping[str, float] | None = None):
        self.emit("residuals", iteration, **{str(k): float(v) for k, v in residuals.items()})
        if conservation:
            self.emit("conservation", iteration, **{str(k): float(v) for k, v in conservation.items()})


def http_publisher(endpoint: str, api_key: str, *, timeout: float = 5.0) -> Callable[[dict], None]:
    """Opt-in transport to Engineering API; use HTTPS and an ingest-only credential."""
    from urllib.parse import urlsplit
    from urllib.request import Request, urlopen
    parts = urlsplit(endpoint)
    if parts.scheme != "https" or not parts.netloc or parts.username or parts.password:
        raise ValueError("telemetry endpoint must use HTTPS without URL credentials")
    if not api_key:
        raise ValueError("API key required")

    def publish(event: dict):
        req = Request(endpoint.rstrip("/") + "/v1/dashboards/events",
                      data=json.dumps(event, allow_nan=False).encode("utf-8"),
                      headers={"Content-Type": "application/json", "X-API-Key": api_key},
                      method="POST")
        with urlopen(req, timeout=timeout) as response:
            if response.status != 201:
                raise RuntimeError("telemetry publication failed")
    return publish


def openfoam_residuals(log_text: str) -> list[dict[str, float]]:
    """Parse standard OpenFOAM 'Solving for ... Initial residual' log lines."""
    pattern = re.compile(r"Solving for\s+([A-Za-z][A-Za-z0-9_]*)\s*,\s*Initial residual\s*=\s*([0-9.eE+-]+)")
    records = []
    for field, value in pattern.findall(log_text):
        number = float(value)
        if math.isfinite(number):
            records.append({"field": field, "residual": number})
    return records


def emit_openfoam_log(path: str | Path, emitter: TelemetryEmitter, *, max_bytes: int = 5_000_000) -> int:
    """Read an existing bounded solver log and publish residuals (not a live tail)."""
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("solver log exceeds max_bytes; use streaming producer")
    records = openfoam_residuals(path.read_text(errors="replace"))
    for i, row in enumerate(records):
        emitter.emit("residuals", i, **{row["field"]: row["residual"]})
    return len(records)
