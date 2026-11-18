"""Serving benchmarks: what a trained surrogate costs per call, in PyTorch and as an ONNX graph. INT-013.

A surrogate that is accurate and slow is not a product. The number that decides whether a model can sit behind an
API is latency at a realistic batch size and how it scales with it, so this module measures:

* **PyTorch** on CPU (and CUDA when it is there), after a warm-up, reporting p50 / p95 milliseconds and throughput;
* **ONNX** through ``onnxruntime``, when the package is installed, on the same inputs -- and the agreement between
  the two outputs, because a silently different ONNX graph is a serving bug that a latency table alone would hide;
* the **export** itself, when ``onnx`` can be written, by whichever of PyTorch's two exporters can lower the model.

Every missing piece is reported as a note in the result rather than crashing: a machine without ``onnxruntime`` still
gets the PyTorch numbers and a line saying what is not measured. The benchmark is CPU-first and single-stream, which
is the shape of the API this platform serves; a throughput number from a batched GPU run would answer a different
question.

**Three export attempts, in this order.** The TorchScript exporter (``dynamo=False``) is tried first: it is quick and
it covers the pointwise architectures most of this platform's surrogates have. When it refuses an operator, the graph
exporter (``dynamo=True``, opset 18, needs ``onnxscript``) is tried, because the two do not support the same set. When
both refuse -- which is what a spectral model does, since the FNO's convolution keeps **complex tensors** and neither
exporter lowers them (the graph exporter reports ``No decompositions registered for the complex-valued input``) -- the
model is rewritten in real arithmetic by
:func:`pinneapple_neural.architectures.neural_operators.fno_export.to_exportable` and exported like any other module.
That rewrite is a dense cosine/sine transform instead of an FFT, so the graph it produces is not the graph PyTorch
runs; ``max_abs_difference`` is what says whether the substitution was faithful, and it is reported for exactly that
reason. Which attempt produced the graph is in ``ServingReport.exporter``: "it exported" and "which exporter" are
different facts about a serving stack.

    >>> from pinneapple_arena.serving_bench import benchmark_model
    >>> report = benchmark_model(model, example_input, path="surrogate.onnx")
    >>> print(report.to_markdown())
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Latency", "ServingReport", "benchmark_model", "benchmark_callable", "export_onnx"]


@dataclass
class Latency:
    """Timings of one backend: percentiles in milliseconds, and inputs per second."""

    backend: str
    p50_ms: float = float("nan")
    p95_ms: float = float("nan")
    throughput_per_s: float = float("nan")
    n: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ServingReport:
    model: str = ""
    batch_size: int = 1
    latencies: list[Latency] = _dc_field(default_factory=list)
    onnx_path: str = ""
    exporter: str = ""
    max_abs_difference: float | None = None
    notes: list[str] = _dc_field(default_factory=list)

    def get(self, backend: str) -> Latency | None:
        return next((latency for latency in self.latencies if latency.backend == backend), None)

    @property
    def speedup(self) -> float | None:
        """``p50(torch) / p50(onnxruntime)``: how many times faster serving is, when both ran."""
        torch_row, onnx_row = self.get("torch"), self.get("onnxruntime")
        if not torch_row or not onnx_row or not np.isfinite(onnx_row.p50_ms) or onnx_row.p50_ms == 0:
            return None
        return torch_row.p50_ms / onnx_row.p50_ms

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "batch_size": self.batch_size,
                "latencies": [latency.to_dict() for latency in self.latencies], "onnx_path": self.onnx_path,
                "exporter": self.exporter, "max_abs_difference": self.max_abs_difference, "speedup": self.speedup,
                "notes": self.notes}

    def to_markdown(self) -> str:
        lines = [f"# Serving benchmark: {self.model or 'a model'} (batch {self.batch_size})", "",
                 "| backend | p50 (ms) | p95 (ms) | calls/s |", "|---|---:|---:|---:|"]
        for latency in self.latencies:
            lines.append(f"| {latency.backend} | {latency.p50_ms:.3f} | {latency.p95_ms:.3f} | "
                         f"{latency.throughput_per_s:.1f} |")
        if self.exporter:
            lines += ["", f"The ONNX graph came from the {self.exporter} exporter."]
        if self.max_abs_difference is not None:
            lines += ["", f"ONNX and PyTorch agree to {self.max_abs_difference:.2e} (max abs difference) on the "
                          "same input."]
        if self.speedup is not None:
            lines.append(f"Serving is {self.speedup:.2f}x faster on ONNX runtime (p50).")
        if self.notes:
            lines += ["", "Notes:"] + [f"- {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


def benchmark_callable(fn, make_input, *, n: int = 30, warmup: int = 5, backend: str = "",
                       note: str = "") -> Latency:
    """Time ``fn`` on fresh inputs of ``make_input()``; percentiles, not a mean that hides the tail."""
    for _ in range(max(0, warmup)):
        fn(make_input())
    timings = []
    for _ in range(max(1, n)):
        x = make_input()
        started = time.perf_counter()
        fn(x)
        timings.append((time.perf_counter() - started) * 1e3)
    timings = np.asarray(timings, dtype=np.float64)
    return Latency(backend=backend, p50_ms=float(np.percentile(timings, 50)),
                   p95_ms=float(np.percentile(timings, 95)),
                   throughput_per_s=float(1000.0 / max(timings.mean(), 1e-9)), n=int(n), note=note)


def export_onnx(model: Any, example_input: Any, path: str | os.PathLike[str], *, opset: int = 17,
                input_name: str = "x", output_name: str = "y",
                exporter: str = "auto") -> Path:
    """Write the model as an ONNX graph and return the path.

    ``exporter="auto"`` tries the TorchScript exporter at ``opset`` and, if it refuses an operator, the graph
    (dynamo) exporter at ``opset + 1``, which needs ``onnxscript``. ``"torchscript"`` and ``"graph"`` pin one. Raises
    whatever the last attempt raised, so the caller reports the real reason instead of a generic failure -- and never
    leaves a truncated file behind."""
    return _export(model, example_input, path, opset=opset, input_name=input_name, output_name=output_name,
                   exporter=exporter)[0]


def _export(model: Any, example_input: Any, path: str | os.PathLike[str], *, opset: int = 17,
            input_name: str = "x", output_name: str = "y",
            exporter: str = "auto") -> tuple[Path, str]:
    """``(path, which exporter produced it)`` -- the name is what :meth:`ServingReport.exporter` records."""
    import torch
    from torch import nn

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    class _Serve(nn.Module):
        """The model behind a one-argument callable: the library's operators take keyword extras (``y_true``,
        ``return_loss``) that a graph export neither needs nor tolerates positionally."""

        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x):                                  # noqa: D102 - the graph is the documentation
            out = self.inner(x)
            return out.y if hasattr(out, "y") else out

    wrapped = _Serve(model)
    attempts = {"torchscript": (False, opset), "graph": (True, max(opset, 18)), "real-arithmetic": (False, opset)}
    if exporter == "auto":
        order = ["torchscript", "graph", "real-arithmetic"]
    elif exporter in attempts:
        order = [exporter]
    else:
        raise ValueError(f"unknown exporter '{exporter}'; use 'auto', 'torchscript', 'graph' or 'real-arithmetic'")

    first: Exception | None = None
    for name in order:
        dynamo, version = attempts[name]
        try:
            with torch.no_grad():
                if name == "real-arithmetic":
                    # a spectral model keeps complex tensors, which neither exporter lowers; the real-arithmetic
                    # copy of the same layer does, and its agreement with the original is checked by the caller
                    from pinneapple_neural.architectures.neural_operators.fno_export import (
                        spatial_shape,
                        to_exportable,
                    )
                    grid = spatial_shape(model, example_input)
                    if grid is None:
                        raise ValueError("no spectral convolution to rewrite in real arithmetic")
                    torch.onnx.export(to_exportable(wrapped, grid), (example_input,), str(target),
                                      opset_version=version, input_names=[input_name],
                                      output_names=[output_name], dynamic_axes={input_name: {0: "batch"}},
                                      dynamo=dynamo)
                else:
                    torch.onnx.export(wrapped, (example_input,), str(target), opset_version=version,
                                      input_names=[input_name], output_names=[output_name], dynamo=dynamo)
            return target, name
        except ImportError as exc:
            if first is None:
                first = exc
            if name == order[-1]:
                break
        except Exception as exc:                               # try the other exporter, remember the reason
            first = first or exc
            if target.exists():                                # torch writes as it traces: a failed export leaves
                target.unlink()                                # a truncated graph, worse than no graph at all
    assert first is not None
    raise first


def benchmark_model(model: Any, example_input: Any, *, path: str | os.PathLike[str] | None = None,
                    n: int = 30, warmup: int = 5, check: bool = True) -> ServingReport:
    """PyTorch latency, then ONNX export + onnxruntime latency and agreement, when the packages are there.

    ``example_input`` is the tensor an inference call receives (``(batch, channels, length)`` for an FNO). The same
    input is reused for every backend so the comparison is of the runtimes, not of two different problems."""
    import torch

    report = ServingReport(model=type(model).__name__, batch_size=int(example_input.shape[0]))
    model.eval()

    def infer_torch(x):
        with torch.no_grad():
            out = model(x)
            return getattr(out, "y", out)

    report.latencies.append(benchmark_callable(infer_torch, lambda: example_input.clone(), n=n, warmup=warmup,
                                               backend="torch (cpu)", note="eager inference, no compile"))

    if path is None:
        report.notes.append("no --onnx path given: only PyTorch was measured")
        return report
    target = Path(path)
    try:
        target, report.exporter = _export(model, example_input, path)
    except ImportError as exc:
        report.notes.append(f"ONNX export skipped: {exc}")
        return report
    except Exception as exc:                                  # an unsupported op is a real answer, not a crash
        if target.exists():                                   # torch writes as it traces: a failed export leaves
            target.unlink()                                   # a truncated graph, which is worse than no graph
        report.notes.append(f"ONNX export failed for {type(model).__name__}: {type(exc).__name__}: {exc}")
        return report
    report.onnx_path = str(target)
    if report.exporter == "graph":
        report.notes.append("the TorchScript exporter refused this model; the graph exporter (dynamo, opset 18) did "
                            "it, and onnxscript is needed to re-export, not to serve")
    elif report.exporter == "real-arithmetic":
        report.notes.append("both exporters refused the complex tensors of this spectral model, so the graph is the "
                            "real-arithmetic copy of its spectral convolutions on the grid of this example: same "
                            "weights, dense cos/sin transforms instead of FFTs, fixed input size")

    try:
        import onnxruntime as ort
    except ImportError:
        report.notes.append("onnxruntime is not installed: export was written, runtime latency was not measured")
        return report

    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    def infer_onnx(x):
        return session.run(None, {name: x.numpy()})[0]

    report.latencies.append(benchmark_callable(infer_onnx, lambda: example_input.clone(), n=n, warmup=warmup,
                                               backend="onnxruntime (cpu)", note="CPUExecutionProvider"))
    if check:
        with torch.no_grad():
            reference = np.asarray(getattr(model(example_input), "y", model(example_input)))
        served = np.asarray(infer_onnx(example_input))
        report.max_abs_difference = float(np.max(np.abs(reference - served)))
        report.notes.append("the agreement is a max absolute difference on one input: a smoke check, not a "
                            "tolerance on the model's accuracy")
    return report
