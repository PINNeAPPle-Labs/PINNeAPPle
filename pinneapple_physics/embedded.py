"""Embedded execution: the model, its physics residual and its measurement check as **one** ONNX graph, with a
runtime a flight computer calls at tens to hundreds of hertz (#350).

The residual is cheap, which is what lets trust fly. A robot cannot afford a Python interpreter to tell it that the
prediction it is about to act on left the physics, so the whole decision goes into the graph:

* the **model** -- any ``torch.nn.Module`` mapping a fixed-size state ``(frames, H, W)`` to the next frame;
* the **residual** -- a finite-difference stencil written as convolutions, so it is arithmetic and not control flow:
  ``advection-diffusion`` (``u_t + c . grad u - alpha lap u``), ``advection``, or ``heat``, with periodic boundaries;
* the **measurement check** -- the sensor reading against the prediction, relative to its scale;
* the **decision** -- ``off_physics`` where the residual is above its tolerance, ``suspect`` where the measurement
  disagrees (which is what ``PhysicsEnsemble`` already knows how to discount), and ``trust`` as the residual judged
  against its tolerance, so a caller can threshold it at any level.

Nothing in the graph branches: ``torch.where``/comparisons export as arithmetic. :class:`EmbeddedRuntime` is the other
half -- it loads the graph, runs one step, and *measures* the latency, because "tens of hertz" is a claim only a
measurement supports.

    >>> import torch
    >>> from pinneapple_physics.embedded import EmbeddedSpec, export, EmbeddedRuntime, graph_summary
    >>> model = torch.nn.Conv2d(2, 1, 3, padding=1, padding_mode="circular")     # (2, H, W) -> (1, H, W)
    >>> spec = EmbeddedSpec(state_shape=(2, 32, 64), tolerance=0.05)
    >>> graph_summary(export(model, spec, "flight.onnx")).is_straight_line
    True
    >>> rt = EmbeddedRuntime("flight.onnx", threads=1)
    >>> out = rt.step(state, measurement)          # prediction, residual, trust, off_physics, suspect, error
    >>> rt.latency(state, measurement, n=200)["hz"]

What this is not: a quantised or pruned graph (the arithmetic is the model's own), a CUDA graph, or the PX4/ROS 2
bridge itself -- that needs a board with the flight stack on it. This ships the graph and the runtime it calls.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["EmbeddedSpec", "EmbeddedOutput", "GraphSummary", "EQUATIONS", "CONTROL_FLOW", "Composite",
           "export", "graph_summary", "EmbeddedRuntime", "residual_fn"]

#: The equations the stencil knows, and the parameters each one needs.
EQUATIONS = {"advection-diffusion": ("c_x", "c_y", "alpha"), "advection": ("c_x", "c_y"), "heat": ("alpha",)}

#: Node types that mean the graph carries control flow -- exactly what an embedded path cannot have.
CONTROL_FLOW = ("If", "Loop", "Scan", "SequenceAt", "OptionalGetElement")


@dataclass(frozen=True)
class EmbeddedSpec:
    """The fixed contract of the graph: what goes in, what comes out, and what each check is judged against.

    ``state_shape`` is ``(frames, height, width)``. The exported graph is fixed to this size, because a flight
    controller allocates its buffers once. ``sensor_channels`` are the state channels a sensor measures (the
    measurement check compares those), ``parameters`` are the equation's numbers, ``tolerance`` is the residual limit
    and ``measurement_tolerance`` the relative disagreement a reading is allowed to have.
    """

    state_shape: tuple[int, int, int]
    equation: str = "advection-diffusion"
    parameters: dict[str, float] = field(default_factory=lambda: {"c_x": 1.0, "c_y": 0.0, "alpha": 0.01})
    dt: float = 0.01
    dx: float = 1.0
    tolerance: float = 0.05
    sensor_channels: tuple[int, ...] = (0,)
    measurement_tolerance: float = 0.1
    measurement_scale: float = 1.0

    def __post_init__(self):
        frames, h, w = self.state_shape
        if frames < 2:
            raise ValueError("the state needs at least two frames: the residual is a time derivative")
        if h < 3 or w < 3:
            raise ValueError("the stencil needs a 3x3 neighbourhood at least")
        if self.equation not in EQUATIONS:
            raise ValueError(f"unknown equation {self.equation!r}; this stencil knows {sorted(EQUATIONS)}")
        missing = [p for p in EQUATIONS[self.equation] if p not in self.parameters]
        if missing:
            raise ValueError(f"{self.equation} needs {missing} in parameters")
        if not self.sensor_channels:
            raise ValueError("sensor_channels cannot be empty: the measurement check compares one of them")
        if max(self.sensor_channels) >= frames:
            raise ValueError(f"sensor channel {max(self.sensor_channels)} is outside a {frames}-frame state")


@dataclass(frozen=True)
class EmbeddedOutput:
    """One step of the embedded path: what the model says, what the physics says about it, what the sensor says."""

    prediction: np.ndarray
    residual: float
    trust: float                 # 1 - residual/tolerance, clipped to [0, 1]: 1 is fully inside the physics
    off_physics: bool
    suspect: bool
    measurement_error: float

    def to_dict(self) -> dict[str, Any]:
        return {"residual": self.residual, "trust": self.trust, "off_physics": self.off_physics,
                "suspect": self.suspect, "measurement_error": self.measurement_error,
                "prediction_shape": list(self.prediction.shape)}


@dataclass(frozen=True)
class GraphSummary:
    """What the exported file contains: the node types, whether any can branch, and its size."""

    path: str
    bytes: int
    op_types: tuple[str, ...]
    control_flow: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    @property
    def is_straight_line(self) -> bool:
        return not self.control_flow

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "op_types": list(self.op_types),
                "control_flow": list(self.control_flow), "inputs": list(self.inputs),
                "outputs": list(self.outputs), "straight_line": self.is_straight_line}


# -------------------------------------------------------------------------------------------------- the graph
def _kernels(spec: EmbeddedSpec, channels: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The centred-difference and Laplacian kernels of a periodic grid, one per channel (depthwise convolution)."""
    dx = float(spec.dx)
    kx = torch.zeros(1, 1, 3, 3)
    ky = torch.zeros(1, 1, 3, 3)
    lap = torch.zeros(1, 1, 3, 3)
    kx[0, 0, 1, 0], kx[0, 0, 1, 2] = -0.5 / dx, 0.5 / dx           # d/dx: along the last axis of the field
    ky[0, 0, 0, 1], ky[0, 0, 2, 1] = -0.5 / dx, 0.5 / dx           # d/dy: along the rows
    lap[0, 0, 1, 1] = -4.0 / dx ** 2
    lap[0, 0, 0, 1] = lap[0, 0, 2, 1] = lap[0, 0, 1, 0] = lap[0, 0, 1, 2] = 1.0 / dx ** 2
    return (kx.expand(channels, 1, 3, 3).contiguous(), ky.expand(channels, 1, 3, 3).contiguous(),
            lap.expand(channels, 1, 3, 3).contiguous())


def residual_fn(spec: EmbeddedSpec) -> Any:
    """The residual as a torch function of ``(prediction, previous_frame)`` -- the same arithmetic the graph carries.

    Kept public so the embedded number can be compared with the one computed in Python on the same arrays: a graph
    that disagrees with its own reference is a graph nobody should fly.
    """
    kx, ky, lap = _kernels(spec, 1)
    p = spec.parameters

    def fn(prediction: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        # keep four dimensions everywhere: the circular pad of the stencil is lowered to Slice+Concat over the last
        # two axes, and that lowering is validated for a rank-4 tensor, not a rank-3 one
        u = prediction.reshape(1, -1, prediction.shape[-2], prediction.shape[-1])
        pad = (1, 1, 1, 1)
        ux = F.conv2d(F.pad(u, pad, mode="circular"), kx)
        uy = F.conv2d(F.pad(u, pad, mode="circular"), ky)
        ul = F.conv2d(F.pad(u, pad, mode="circular"), lap)
        ut = (u - previous.reshape(1, -1, u.shape[-2], u.shape[-1])) / float(spec.dt)
        if spec.equation == "heat":
            eq = ut - p["alpha"] * ul
        elif spec.equation == "advection":
            eq = ut + p["c_x"] * ux + p.get("c_y", 0.0) * uy
        else:
            eq = ut + p["c_x"] * ux + p.get("c_y", 0.0) * uy - p.get("alpha", 0.0) * ul
        scale = u.abs().mean() + 1e-6
        return eq.pow(2).mean().sqrt() / scale

    return fn


class Composite(torch.nn.Module):
    """The whole decision as one straight-line graph: prediction, residual, trust, the two flags and the error."""

    def __init__(self, model: torch.nn.Module, spec: EmbeddedSpec):
        super().__init__()
        self.model = model
        self.spec = spec
        kx, ky, lap = _kernels(spec, spec.state_shape[0])
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("lap", lap)
        self.register_buffer("sensor_index", torch.tensor(list(spec.sensor_channels), dtype=torch.long))
        self.register_buffer("tol", torch.tensor(float(spec.tolerance)))
        self.register_buffer("mtol", torch.tensor(float(spec.measurement_tolerance)))
        self.register_buffer("mscale", torch.tensor(float(spec.measurement_scale)))
        self.register_buffer("dt", torch.tensor(float(spec.dt)))
        self.register_buffer("c_x", torch.tensor(float(spec.parameters.get("c_x", 0.0))))
        self.register_buffer("c_y", torch.tensor(float(spec.parameters.get("c_y", 0.0))))
        self.register_buffer("alpha", torch.tensor(float(spec.parameters.get("alpha", 0.0))))

    def forward(self, state, measurement):
        pred = self.model(state)                                  # (1, 1, H, W), (1, H, W) or (H, W)
        # four dimensions from here on: the circular pad exports as Slice+Concat over the last two axes
        u = pred.reshape(1, -1, pred.shape[-2], pred.shape[-1])   # (1, C, H, W), the batch is fixed to one
        pad = (1, 1, 1, 1)
        ux = F.conv2d(F.pad(u, pad, mode="circular"), self.kx, groups=u.shape[0])
        uy = F.conv2d(F.pad(u, pad, mode="circular"), self.ky, groups=u.shape[0])
        lap = F.conv2d(F.pad(u, pad, mode="circular"), self.lap, groups=u.shape[0])
        previous = state[:, -1:, :, :]                   # (1, 1, H, W): same rank as u
        ut = (u - previous) / self.dt
        if self.spec.equation == "heat":
            eq = ut - self.alpha * lap
        elif self.spec.equation == "advection":
            eq = ut + self.c_x * ux + self.c_y * uy
        else:
            eq = ut + self.c_x * ux + self.c_y * uy - self.alpha * lap
        scale = u.abs().mean() + 1e-6
        residual = eq.pow(2).mean().sqrt() / scale
        predicted_sensors = u.reshape(-1)[self.sensor_index]
        error = (measurement.reshape(-1) - predicted_sensors).abs().mean() / self.mscale
        off = (residual > self.tol).to(residual.dtype)
        suspect = (error > self.mtol).to(error.dtype)
        trust = (1.0 - residual / self.tol).clamp(0.0, 1.0)
        return pred, residual, trust, off, suspect, error


def export(model: torch.nn.Module, spec: EmbeddedSpec, path: Any, *, opset: int = 17) -> Path:
    """Write the composite graph: ``(state, measurement) -> (prediction, residual, trust, off_physics, suspect,
    error)``.

    Fixed shapes, no Python in the hot path: the exported file is arithmetic from the first node to the last.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = Composite(model.eval(), spec).eval()
    frames, h, w = spec.state_shape
    state = torch.zeros(1, frames, h, w)
    measurement = torch.zeros(1, len(spec.sensor_channels))
    names = ["prediction", "residual", "trust", "off_physics", "suspect", "measurement_error"]
    with torch.no_grad():
        torch.onnx.export(graph, (state, measurement), str(path), input_names=["state", "measurement"],
                          output_names=names, opset_version=opset, do_constant_folding=True, dynamo=False)
    return path


def graph_summary(path: Any) -> GraphSummary:
    """What the file contains, and whether it is a straight line (no ``If``/``Loop``/``Scan`` anywhere)."""
    import onnx

    model = onnx.load(str(path))
    ops: list[str] = []
    flow: list[str] = []
    for node in model.graph.node:
        if node.op_type not in ops:
            ops.append(node.op_type)
        if node.op_type in CONTROL_FLOW:
            flow.append(node.op_type)
    return GraphSummary(str(path), Path(path).stat().st_size, tuple(sorted(ops)), tuple(flow),
                        tuple(i.name for i in model.graph.input), tuple(o.name for o in model.graph.output))


# ------------------------------------------------------------------------------------------------ the runtime
class EmbeddedRuntime:
    """One ONNX session, one step, and the latency measured -- the half a flight computer actually calls.

    ``threads`` is 1 by default: an embedded core is one core, and a latency measured on eight threads is not a
    budget anybody can use.
    """

    def __init__(self, path: Any, *, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = int(threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.path = Path(path)
        self.threads = int(threads)
        self.session = ort.InferenceSession(str(self.path), sess_options=so, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.outputs = [o.name for o in self.session.get_outputs()]

    def step(self, state: np.ndarray, measurement: np.ndarray) -> EmbeddedOutput:
        """One decision for one fixed-size state."""
        feed = {"state": np.asarray(state, np.float32)[None], "measurement": np.asarray(measurement, np.float32)[None]}
        pred, residual, trust, off, suspect, error = (np.asarray(v) for v in self.session.run(self.outputs, feed))
        return EmbeddedOutput(pred[0], float(residual.ravel()[0]), float(trust.ravel()[0]),
                              bool(off.ravel()[0] > 0.5), bool(suspect.ravel()[0] > 0.5),
                              float(error.ravel()[0]))

    def latency(self, state: np.ndarray, measurement: np.ndarray, *, n: int = 200) -> dict[str, float]:
        """Run ``n`` steps and report what the machine did: median, p95, worst, and the rate they imply."""
        feed = {"state": np.asarray(state, np.float32)[None],
                "measurement": np.asarray(measurement, np.float32)[None]}
        self.session.run(self.outputs, feed)                          # warm up: the first call pays for init
        samples = np.empty(int(n))
        for k in range(int(n)):
            t0 = time.perf_counter()
            self.session.run(self.outputs, feed)
            samples[k] = time.perf_counter() - t0
        median = float(np.median(samples))
        p95 = float(np.percentile(samples, 95))
        return {"n": float(n), "median_ms": median * 1e3, "p95_ms": p95 * 1e3, "max_ms": float(samples.max()) * 1e3,
                "hz": (1.0 / median) if median > 0 else float("inf"),
                "hz_p95": (1.0 / p95) if p95 > 0 else float("inf"), "threads": float(self.threads)}
