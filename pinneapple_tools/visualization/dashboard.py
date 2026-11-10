"""Framework-independent Physics Dashboard contracts and reusable visualization registry.

Dashboard layouts contain declarative visualization specs, never executable Python,
SQL, HTML, JavaScript, or user-supplied file paths. Rendering belongs to the app.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import math
import re

KINDS = {
    "metric": ("scalar",), "status": ("event", "scalar"), "line": ("timeseries",),
    "multi_line": ("timeseries",), "scatter": ("points",), "histogram": ("distribution",),
    "heatmap": ("matrix", "field"), "contour": ("field",), "surface3d": ("field",),
    "volume3d": ("volume",), "vector_field": ("vector",), "streamlines": ("vector",),
    "mesh3d": ("mesh",), "point_cloud3d": ("points",), "trajectory3d": ("trajectory",),
    "blender_scene": ("artifact",), "digital_twin": ("scene",),
    "image": ("artifact",), "video": ("artifact",), "table": ("table",),
    "training_losses": ("timeseries",), "training_gradients": ("timeseries",),
    "learning_rate": ("timeseries",), "solver_residuals": ("timeseries",),
    "solver_health": ("event", "timeseries"), "resource_usage": ("timeseries",),
    "drift": ("timeseries", "distribution"), "ood": ("event", "scalar"),
    "uncertainty": ("field", "distribution"), "physics_checks": ("event", "table"),
    "conservation": ("timeseries",), "boundary_conditions": ("timeseries",),
    "pipeline_graph": ("graph",), "pipeline_progress": ("event",),
    "rover_pose": ("trajectory",), "rover_terrain": ("mesh", "field"),
    "rover_telemetry": ("timeseries",), "comparison": ("field", "table"),
    "spectrum": ("timeseries",), "phase_portrait": ("points",),
    "timeseries": ("timeseries",), "map": ("geo",),
}
SOURCES = ("job", "training", "simulation", "pipeline", "model", "sensor", "artifact", "rover")
KEY = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.:-]{0,127}$")


def visualization_catalog() -> list[dict[str, Any]]:
    return [{"kind": name, "accepts": list(kinds), "realtime": name not in
             {"blender_scene", "image", "video"}} for name, kinds in sorted(KINDS.items())]


@dataclass
class DataBinding:
    source: str
    resource_id: str
    channel: str
    unit: str | None = None

    def __post_init__(self):
        if self.source not in SOURCES or not KEY.fullmatch(self.resource_id) or not KEY.fullmatch(self.channel):
            raise ValueError("invalid source, resource identifier or channel")
        if self.unit is not None and len(self.unit) > 48:
            raise ValueError("unit label too long")


@dataclass
class Widget:
    id: str
    kind: str
    title: str
    binding: DataBinding
    x: int = 0
    y: int = 0
    w: int = 4
    h: int = 3
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not KEY.fullmatch(self.id) or self.kind not in KINDS:
            raise ValueError("invalid widget identifier or kind")
        if not self.title or len(self.title) > 120:
            raise ValueError("invalid widget title")
        if not (0 <= self.x < 12 and 0 <= self.y <= 200 and 1 <= self.w <= 12 - self.x and 1 <= self.h <= 12):
            raise ValueError("widget outside the 12-column layout")
        if len(str(self.options)) > 4096 or any(k not in ("color", "min", "max", "threshold",
                "palette", "axis", "field", "camera", "show_legend", "log_scale", "refresh_ms")
                for k in self.options):
            raise ValueError("unsupported widget options")


@dataclass
class Dashboard:
    title: str
    widgets: list[Widget]
    description: str = ""
    version: int = 1

    def __post_init__(self):
        if not self.title or len(self.title) > 120 or len(self.widgets) > 80:
            raise ValueError("invalid dashboard title or widget count")
        ids = [w.id for w in self.widgets]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate widget IDs")
        if self.version != 1:
            raise ValueError("unsupported dashboard schema version")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def template(name: str, resource_id: str) -> Dashboard:
    """Reusable presets; bindings are resolved by the authenticated app."""
    definitions = {
        "rover": [("rover_terrain", "Terrain / wheel contact", "rover", "terrain"),
                  ("rover_pose", "Pose and path", "rover", "pose"),
                  ("rover_telemetry", "Velocity and slip", "rover", "telemetry"),
                  ("physics_checks", "Traction / physical validity", "rover", "checks")],
        "training": [("training_losses", "Loss components", "training", "loss"),
                     ("training_gradients", "Gradient norms", "training", "gradients"),
                     ("learning_rate", "Learning rate", "training", "lr"),
                     ("resource_usage", "GPU / CPU / memory", "training", "resources")],
        "cfd": [("solver_residuals", "Solver convergence", "simulation", "residuals"),
                ("solver_health", "Solver health", "simulation", "health"),
                ("streamlines", "Flow field", "simulation", "velocity"),
                ("conservation", "Mass / energy balance", "simulation", "conservation")],
        "drift": [("drift", "Input and prediction drift", "model", "drift"),
                  ("ood", "Out-of-domain events", "model", "ood"),
                  ("uncertainty", "Uncertainty", "model", "uncertainty"),
                  ("physics_checks", "Physics trust", "model", "checks")],
        "pipeline": [("pipeline_graph", "Execution graph", "pipeline", "graph"),
                     ("pipeline_progress", "Node progress", "pipeline", "progress"),
                     ("resource_usage", "Compute usage", "pipeline", "resources"),
                     ("table", "Produced datasets", "pipeline", "datasets")],
        "blender": [("blender_scene", "Rendered simulation", "artifact", "scene"),
                    ("mesh3d", "Interactive geometry", "simulation", "mesh"),
                    ("contour", "Physical field", "simulation", "field"),
                    ("physics_checks", "Verification", "simulation", "checks")],
    }
    if name not in definitions:
        raise ValueError(f"unknown dashboard template: {name}")
    widgets = [Widget(f"{name}_{i}", kind, title, DataBinding(source, resource_id, channel),
                      x=(i % 2) * 6, y=(i // 2) * 4, w=6, h=4)
               for i, (kind, title, source, channel) in enumerate(definitions[name])]
    return Dashboard(name.title(), widgets)


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """Validate bounded numerical/event telemetry before sending to browser subscribers."""
    if not isinstance(event, dict):
        raise ValueError("event must be an object")
    channel, step, values = event.get("channel"), event.get("step"), event.get("values")
    if not isinstance(channel, str) or not KEY.fullmatch(channel):
        raise ValueError("invalid channel")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("invalid step")
    if not isinstance(values, dict) or len(values) > 64:
        raise ValueError("values must be a small object")
    for k, v in values.items():
        if not isinstance(k, str) or not KEY.fullmatch(k):
            raise ValueError("invalid metric key")
        if isinstance(v, bool) or not isinstance(v, (float, int, str)):
            raise ValueError("metric must be numeric or a bounded label")
        if isinstance(v, (float, int)) and not math.isfinite(v):
            raise ValueError("non-finite telemetry")
        if isinstance(v, str) and len(v) > 160:
            raise ValueError("metric label too long")
    return {"channel": channel, "step": step, "values": values}
