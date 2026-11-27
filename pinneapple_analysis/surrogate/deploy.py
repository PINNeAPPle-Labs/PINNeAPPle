"""A surrogate made smaller for deployment, with both numbers measured: error lost and latency gained (#112).

"The model is smaller" is not a result. A deployment change costs accuracy and buys speed, and neither side of that
trade is worth believing unless it is measured on the same inputs and reported together — which is what every
function here returns: a :class:`DeploymentReport` carrying the relative L2 the compression cost, the p50 latency
before and after, the bytes before and after, and whether the error stayed inside the tolerance the caller set.

The paths, and the reason each is where it is:

``onnx-int8`` (:func:`compress_onnx_int8`) — **runs on any CPU.**
    The model is exported to ONNX (with a dynamic batch axis, so one graph serves a row or a batch) and quantized
    with ONNX Runtime's dynamic int8 pass: weights become int8, activations are quantized at run time, and the
    arithmetic is int8 where the kernel exists. It needs the ``onnx`` and ``onnxruntime`` extras and no GPU.

    **Where the gain is, measured:** int8 wins on *batched* inference and loses on a single row, because the
    quantize/dequantize of the activations is per call. On the 16→256→256→8 stack of the test, one thread: batch 1
    is 0.90x (slower), batch 8 is 1.45x, batch 4000 is 1.74x. The report therefore records the batch it measured —
    a "3x faster" claim with the batch hidden is exactly the kind of number this module exists to prevent — and the
    single-row latency goes in ``details`` so the losing side is visible too.

``structured-prune`` (:func:`prune_structured`, :func:`measure_pruned`) — **needs nothing but torch.**
    Whole hidden units whose incoming and outgoing weights are small are removed from a feed-forward stack, so the
    dense matrices really do get smaller — unlike magnitude pruning of individual weights, which leaves the latency
    exactly where it was. Also measured, also against the tolerance, and the numbers say what a reader needs to
    know before choosing it: on the trained 16→64→64→8 stack of the test, keeping 95 % of the units costs 11.6 %
    relative L2 for 1.16x, and keeping 50 % costs 81 % for 2.15x (50 000 rows). **Pruning without fine-tuning is
    not a deployment path on a surrogate this size** — it is fast and it destroys the answer — so the report's
    ``within_tolerance`` is False at any tolerance a surrogate would be deployed under, and that verdict is the
    deliverable, not a footnote.

``torch-int8`` (:func:`quantize_torch_dynamic`, :func:`measure_torch_dynamic`) — **works, and does not pay off here.**
    PyTorch's dynamic int8 runs once a quantized engine is selected — the macOS arm64 wheel ships ``qnnpack`` but
    leaves the engine unset, so the first call raises ``NoQEngine`` until it is set, which
    :func:`quantize_torch_dynamic` does. Measured on the same stack: relative L2 1.5 % (inside tolerance) and
    **0.88x** — *slower* than float32, because the per-call quantize/dequantize costs more than the int8 GEMM
    saves at this size. The number is reported as measured; a path that is available is not the same as a path that
    is worth taking, and this module is built so that the difference is visible.

``TensorRT`` — **declared, not executed.**
    :func:`tensorrt_status` says whether a TensorRT runtime and a CUDA device are present. There is no NVIDIA GPU
    here, so the TensorRT path is not taken and the report says so in ``tensorrt`` rather than leaving a gap a
    reader would take for an omission. The CPU number above is what a reader can reproduce.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_arena.serving_bench import benchmark_callable

DEFAULT_TOLERANCE = 2e-2


class CompressionUnavailable(RuntimeError):
    """A path that cannot run in this environment, with the reason in the message."""


@dataclass(frozen=True)
class DeploymentReport:
    """One compression, measured: what it cost in accuracy and what it bought in latency and size."""

    method: str
    tolerance: float
    relative_l2: float
    max_abs: float
    latency_float_ms: float
    latency_compressed_ms: float
    bytes_float: int
    bytes_compressed: int
    rows: int
    backend: str = ""
    tensorrt: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def speedup(self) -> float:
        """How many times faster the compressed model is, p50 against p50, same backend, same inputs."""
        if not np.isfinite(self.latency_compressed_ms) or self.latency_compressed_ms <= 0:
            return float("nan")
        return self.latency_float_ms / self.latency_compressed_ms

    @property
    def size_ratio(self) -> float:
        """How many times smaller, in bytes on disk — ``nan`` when the two are not comparable (see the notes)."""
        return self.bytes_float / self.bytes_compressed if self.bytes_compressed else float("nan")

    @property
    def within_tolerance(self) -> bool:
        return bool(self.relative_l2 <= self.tolerance)

    def to_dict(self) -> dict[str, Any]:
        return {**{k: v for k, v in self.__dict__.items()}, "speedup": self.speedup,
                "size_ratio": self.size_ratio, "within_tolerance": self.within_tolerance}

    def to_markdown(self) -> str:
        verdict = "inside" if self.within_tolerance else "**outside**"
        lines = [f"# {self.method}: {self.rows} rows, {self.backend or 'torch'}", "",
                 "| | float | compressed |", "|---|---:|---:|",
                 f"| p50 latency (ms) | {self.latency_float_ms:.2f} | {self.latency_compressed_ms:.2f} |",
                 f"| bytes | {self.bytes_float} | {self.bytes_compressed} |", "",
                 f"Error lost: relative L2 **{self.relative_l2:.3%}** (max abs {self.max_abs:.3g}), {verdict} the "
                 f"tolerance of {self.tolerance:.1%}.",
                 (f"Latency gained: **{self.speedup:.2f}x**, size **{self.size_ratio:.2f}x** smaller."
                  if np.isfinite(self.speedup) and np.isfinite(self.size_ratio) else
                  f"Latency: **{self.speedup:.2f}x**." if np.isfinite(self.speedup) else "Latency: not measured.")]
        if self.tensorrt:
            lines += ["", f"TensorRT: {self.tensorrt}"]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- what can run here
def quantization_status() -> dict[str, str]:
    """Which compression paths this environment can take, and the sentence for each one it cannot."""
    status: dict[str, str] = {}
    status["onnx-int8"] = "ok" if _has("onnxruntime") and _has("onnx") else "needs onnxruntime and onnx (pip install 'pinneapple[export]')"
    status["structured-prune"] = "ok"
    import torch

    engines = list(getattr(torch.backends.quantized, "supported_engines", []))
    status["torch-int8"] = ("ok" if engines else
                            "this PyTorch build has no quantized engine, so torch's dynamic int8 cannot run")
    status["tensorrt"] = tensorrt_status()["reason"]
    return status


def _has(module: str) -> bool:
    from importlib.util import find_spec

    return find_spec(module) is not None


def tensorrt_status() -> dict[str, Any]:
    """Whether a TensorRT path is possible here: the runtime and a CUDA device, both required."""
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if cuda else ""
    except Exception:                                          # pragma: no cover - torch is a core dependency
        cuda, device = False, ""
    if not cuda:
        return {"available": False, "reason": "no NVIDIA GPU on this machine, so the TensorRT path is not taken; "
                                             "the CPU path above is the reproducible number"}
    if not _has("tensorrt"):
        return {"available": False, "reason": f"a CUDA device ({device}) is present but the tensorrt package is "
                                              "not installed"}
    return {"available": True, "reason": f"TensorRT is installed and {device} is available; not exercised by this "
                                         "module's tests, which run on CPU"}


# ---------------------------------------------------------------- the ONNX int8 path
def compress_onnx_int8(module: Any, example: Any, *, n_rows: int = 4000, tolerance: float = DEFAULT_TOLERANCE,
                       out_dir: str | Path | None = None, timing_runs: int = 20) -> DeploymentReport:
    """Export, quantize with ONNX Runtime's dynamic int8, and measure both sides of the trade.

    ``example`` is one input row (or a batch); ``n_rows`` rows are sampled from it for the error and the timings.
    The two sessions share the backend and the input, so the latency comparison is not "ONNX against torch" but
    "the same graph, int8 weights against float32 weights" — the difference a deployment would actually see.

    Raises :class:`CompressionUnavailable` when the extras are missing, and nothing is written when ``out_dir`` is
    None (a temporary directory is used and removed).
    """
    if not (_has("onnx") and _has("onnxruntime")):
        raise CompressionUnavailable("the int8 path runs on ONNX Runtime: pip install 'pinneapple[export]' "
                                     "(onnx + onnxruntime)")
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    module.eval()
    x = _as_torch_batch(example, n_rows)
    directory = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="pinneapple-int8-"))
    directory.mkdir(parents=True, exist_ok=True)
    float_path = _export_dynamic_batch(module, x[:1], directory / "model.onnx")
    quantized_path = directory / "model.int8.onnx"
    quantize_dynamic(str(float_path), str(quantized_path), weight_type=QuantType.QInt8)

    providers = ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1                # one thread: the comparison is the graph, not the scheduler
    float_session = ort.InferenceSession(str(float_path), options, providers=providers)
    int8_session = ort.InferenceSession(str(quantized_path), options, providers=providers)
    name_in = float_session.get_inputs()[0].name
    name_out = float_session.get_outputs()[0].name

    xx = x.numpy().astype(np.float32)

    def run(session):
        return session.run([name_out], {name_in: xx})[0]

    reference, quantized = run(float_session), run(int8_session)
    relative_l2 = _relative_l2(quantized, reference)
    max_abs = float(np.max(np.abs(quantized - reference)))

    latency_float = benchmark_callable(lambda _: run(float_session), lambda: xx, n=timing_runs, warmup=3,
                                       backend="onnxruntime-float")
    latency_int8 = benchmark_callable(lambda _: run(int8_session), lambda: xx, n=timing_runs, warmup=3,
                                      backend="onnxruntime-int8")
    single = xx[:1]
    single_float = benchmark_callable(lambda _: float_session.run([name_out], {name_in: single}), lambda: single,
                                      n=timing_runs, warmup=3, backend="onnxruntime-float")
    single_int8 = benchmark_callable(lambda _: int8_session.run([name_out], {name_in: single}), lambda: single,
                                     n=timing_runs, warmup=3, backend="onnxruntime-int8")
    return DeploymentReport(
        method="onnx-int8", tolerance=float(tolerance), relative_l2=relative_l2, max_abs=max_abs,
        latency_float_ms=latency_float.p50_ms, latency_compressed_ms=latency_int8.p50_ms,
        bytes_float=os.path.getsize(float_path), bytes_compressed=os.path.getsize(quantized_path),
        rows=int(len(xx)), backend=", ".join(providers), tensorrt=tensorrt_status()["reason"],
        details={"float_graph": str(float_path), "int8_graph": str(quantized_path),
                 "p95_float_ms": latency_float.p95_ms, "p95_int8_ms": latency_int8.p95_ms,
                 "batch": int(len(xx)), "single_row_float_ms": single_float.p50_ms,
                 "single_row_int8_ms": single_int8.p50_ms,
                 "single_row_speedup": single_float.p50_ms / max(single_int8.p50_ms, 1e-12),
                 "threads": 1, "weight_type": "int8", "quantization": "dynamic (ONNX Runtime)"})


# ---------------------------------------------------------------- the pruning path
def prune_structured(module: Any, keep: float = 0.5) -> Any:
    """Drop the least important hidden units of a feed-forward stack, keeping ``keep`` of them.

    Importance is the product of the incoming and outgoing weight norms of a unit — a unit that is small on both
    sides is the one that costs latency and contributes least. Only the ``out_features`` dimension of a ``Linear``
    that feeds another ``Linear`` (with an activation in between) is pruned, so the next layer's ``in_features`` is
    reduced with it: the dense matrices really shrink, which is the only reason latency would move at all.

    Raises ``ValueError`` for a model that is not a stack of ``Linear`` layers — a convolution, a residual block or
    an operator with a custom forward has no "hidden unit" this can remove without changing what it computes.
    """
    import torch
    from torch import nn

    if not 0.0 < keep <= 1.0:
        raise ValueError(f"keep must be in (0, 1], got {keep}")
    if not isinstance(module, nn.Sequential):
        raise ValueError("structured pruning here works on an nn.Sequential of Linear layers; for another "
                         "architecture prune by its own structure, or use the onnx-int8 path, which does not care")
    linears = [i for i, layer in enumerate(module) if isinstance(layer, nn.Linear)]
    if len(linears) < 2:
        raise ValueError("a stack with one Linear layer has no hidden unit between two layers to prune")
    clone = _deepcopy(module)
    clone.eval()
    pruned: dict[int, Any] = {}
    for previous, current in zip(linears, linears[1:], strict=False):
        before, after = clone[previous], clone[current]
        if previous in pruned:
            before = pruned[previous]
        if before.out_features != after.in_features:
            continue
        units = before.out_features
        n_keep = max(1, int(round(units * keep)))
        if n_keep >= units:
            continue
        with torch.no_grad():
            incoming = before.weight.norm(dim=1)                    # (units,)
            outgoing = after.weight.norm(dim=0)                     # (units,)
            importance = incoming * outgoing
            index = torch.argsort(importance, descending=True)[:n_keep].sort().values
            new_before = nn.Linear(before.in_features, n_keep, bias=before.bias is not None)
            new_before.weight.copy_(before.weight[index])
            if before.bias is not None:
                new_before.bias.copy_(before.bias[index])
            new_after = nn.Linear(n_keep, after.out_features, bias=after.bias is not None)
            new_after.weight.copy_(after.weight[:, index])
            if after.bias is not None:
                new_after.bias.copy_(after.bias)
        pruned[previous], pruned[current] = new_before, new_after
    if not pruned:
        raise ValueError("nothing was pruned: the stack has no Linear layer feeding another one")
    layers = [pruned.get(i, layer) for i, layer in enumerate(clone)]
    return nn.Sequential(*layers)


def measure_pruned(module: Any, pruned: Any, example: Any, *, n_rows: int = 4000,
                   tolerance: float = DEFAULT_TOLERANCE, timing_runs: int = 25) -> DeploymentReport:
    """The pruning, measured against the model it came from: error on the same rows, p50 latency on each."""
    return _measure_torch_pair(module, pruned, example, method="structured-prune", n_rows=n_rows,
                               tolerance=tolerance, timing_runs=timing_runs)


def _measure_torch_pair(module: Any, candidate: Any, example: Any, *, method: str, n_rows: int,
                        tolerance: float, timing_runs: int) -> DeploymentReport:
    """Error and p50 latency of a torch module against the torch module it came from — one batch, both sides."""
    import torch

    x = _as_torch_batch(example, n_rows)
    module.eval()
    candidate.eval()
    with torch.inference_mode():
        reference = module(x).detach().cpu().numpy()
        produced = candidate(x).detach().cpu().numpy()
    latency_float = benchmark_callable(lambda _: module(x), lambda: x, n=timing_runs, warmup=5, backend="torch")
    latency_candidate = benchmark_callable(lambda _: candidate(x), lambda: x, n=timing_runs, warmup=5,
                                           backend="torch")
    return DeploymentReport(
        method=method, tolerance=float(tolerance), relative_l2=_relative_l2(produced, reference),
        max_abs=float(np.max(np.abs(produced - reference))),
        latency_float_ms=latency_float.p50_ms, latency_compressed_ms=latency_candidate.p50_ms,
        bytes_float=_parameter_bytes(module), bytes_compressed=_parameter_bytes(candidate), rows=int(len(x)),
        backend="torch (CPU)", tensorrt=tensorrt_status()["reason"],
        details={"parameters_float": _parameter_count(module), "parameters_compressed": _parameter_count(candidate),
                 "batch": int(len(x)), "p95_float_ms": latency_float.p95_ms,
                 "p95_compressed_ms": latency_candidate.p95_ms,
                 "size_note": "" if _parameter_count(candidate) else
                 "the int8 weights are packed quantized tensors, not parameters, so bytes_compressed is 0 and the "
                 "size ratio is not comparable with the float model's"})


# ---------------------------------------------------------------- the declared paths
def quantize_torch_dynamic(module: Any) -> Any:
    """``torch.ao.quantization.quantize_dynamic`` on the Linear layers, with a quantized engine selected first.

    The wheel ships ``qnnpack`` (or ``fbgemm``) but leaves ``torch.backends.quantized.engine`` unset, and
    ``quantize_dynamic`` then raises ``NoQEngine`` — which reads like "unsupported" when it is only "not
    selected". The engine is set here, and only when it is unset, so a caller that chose one keeps it.

    Raises :class:`CompressionUnavailable` when the build really has no quantized engine.
    """
    import torch

    engines = list(getattr(torch.backends.quantized, "supported_engines", []))
    if not engines:
        raise CompressionUnavailable("this PyTorch build has no quantized engine, so torch's dynamic int8 cannot "
                                     "run here; compress_onnx_int8 measures the same pair of numbers on CPU with "
                                     "ONNX Runtime")
    if getattr(torch.backends.quantized, "engine", "none") == "none":
        torch.backends.quantized.engine = engines[0]
    return torch.ao.quantization.quantize_dynamic(module.eval(), {torch.nn.Linear}, dtype=torch.qint8)


def measure_torch_dynamic(module: Any, example: Any, *, n_rows: int = 4000,
                          tolerance: float = DEFAULT_TOLERANCE, timing_runs: int = 25) -> DeploymentReport:
    """Torch's dynamic int8, measured against the float model it came from: same rows, error and p50 both."""
    quantized = quantize_torch_dynamic(module)
    return _measure_torch_pair(module, quantized, example, method="torch-int8", n_rows=n_rows,
                               tolerance=tolerance, timing_runs=timing_runs)


# ---------------------------------------------------------------- small helpers
def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    """``||candidate - reference|| / ||reference||`` over every element: one number for the whole output tensor."""
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(candidate - reference) / denominator) if denominator else float("inf")


def _export_dynamic_batch(module: Any, example: Any, path: Path) -> Path:
    """torch -> ONNX with the batch axis dynamic, so the same graph serves one row or four thousand.

    ``serving_bench.export_onnx`` exports at the batch of the example it is given, which would pin this comparison
    to that batch; a deployment graph is the one asked for a batch of any size, so the axis is declared here.
    """
    import torch

    with torch.no_grad():
        try:
            torch.onnx.export(module.eval(), (example,), str(path), input_names=["x"], output_names=["y"],
                              dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, opset_version=17,
                              dynamo=False)
        except TypeError:                                     # torch < 2.5 has no dynamo flag
            torch.onnx.export(module.eval(), (example,), str(path), input_names=["x"], output_names=["y"],
                              dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, opset_version=17)
    return path

def _as_torch_batch(example: Any, n_rows: int) -> Any:
    """``n_rows`` rows shaped like ``example`` (one row or a batch), as float32 on CPU."""
    import torch

    x = torch.as_tensor(example, dtype=torch.float32).detach().clone()
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if len(x) < n_rows:
        repeats = int(np.ceil(n_rows / len(x)))
        x = x.repeat(repeats, *([1] * (x.ndim - 1)))[:n_rows]
    return x[:n_rows]


def _parameter_count(module: Any) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _parameter_bytes(module: Any) -> int:
    return int(sum(p.numel() * p.element_size() for p in module.parameters()))


def _deepcopy(module: Any) -> Any:
    import copy

    return copy.deepcopy(module)
