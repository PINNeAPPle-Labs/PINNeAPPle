"""Serving benchmarks (INT-013): PyTorch latency, the ONNX export, and what is honestly not measured.

``onnxruntime`` is not a dependency of this library and is absent on the machine this was written on, so the tests
cover both legs: the export (``onnx`` is present) and the graceful "written, not measured" report (runtime absent).
The end-to-end ONNX latency path was exercised separately against ``onnxruntime`` in a throwaway environment; what is
pinned here is the contract the normal environment can check.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_arena.serving_bench import (
    Latency,
    ServingReport,
    benchmark_callable,
    benchmark_model,
    export_onnx,
)

torch = pytest.importorskip("torch")


class MLP(torch.nn.Module):
    """The shape most of this platform's surrogates have: a pointwise map, which ONNX lowers without trouble."""

    def __init__(self, n_in: int = 4, n_out: int = 2) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, 16), torch.nn.Tanh(), torch.nn.Linear(16, n_out))

    def forward(self, x):
        return self.net(x)


class Spectral(torch.nn.Module):
    """One real FFT, which is the op that makes an FNO unexportable at opset 17."""

    def forward(self, x):
        return torch.fft.rfft(x, dim=-1).real


def test_the_timer_reports_percentiles_and_throughput():
    latency = benchmark_callable(lambda x: x.sum(), lambda: torch.zeros(8), n=10, warmup=1, backend="test")
    assert latency.backend == "test" and latency.n == 10
    assert 0 <= latency.p50_ms <= latency.p95_ms
    assert latency.throughput_per_s > 0


def test_an_mlp_is_measured_in_torch_and_exported_to_onnx(tmp_path):
    model = MLP()
    report = benchmark_model(model, torch.randn(2, 4), path=tmp_path / "mlp.onnx", n=5, warmup=1)
    torch_row = report.get("torch (cpu)")
    assert torch_row is not None and torch_row.p50_ms > 0
    assert report.onnx_path.endswith("mlp.onnx") and (tmp_path / "mlp.onnx").exists()
    assert any("onnxruntime" in note for note in report.notes) or report.get("onnxruntime (cpu)") is not None
    assert "| torch (cpu) |" in report.to_markdown()


def test_without_a_path_only_torch_is_measured():
    report = benchmark_model(MLP(), torch.randn(1, 4), n=3, warmup=1)
    assert [latency.backend for latency in report.latencies] == ["torch (cpu)"]
    assert report.onnx_path == "" and report.speedup is None
    assert any("only PyTorch" in note for note in report.notes)


def test_an_unsupported_op_leaves_no_partial_file(tmp_path):
    """An FNO's rfft is not lowerable at opset 17: the export must fail loudly in the report, not on disk."""
    target = tmp_path / "spectral.onnx"
    report = benchmark_model(Spectral(), torch.randn(1, 8), path=target, n=2, warmup=0)
    assert report.onnx_path == ""
    assert any("ONNX export failed" in note for note in report.notes)
    assert not target.exists(), "torch writes as it traces: a truncated graph must not survive the failure"
    assert np.isfinite([[1.0]]).all()
    assert report.get("torch (cpu)") is not None


def test_the_export_wrapper_unpacks_an_output_object(tmp_path):
    """The library's operators return an object with ``.y``; the graph takes tensors, so the wrapper unwraps."""

    class Wrapped(torch.nn.Module):
        def forward(self, x):
            return type("Out", (), {"y": x * 2})()

    path = export_onnx(Wrapped(), torch.ones(1, 3), tmp_path / "wrapped.onnx")
    assert path.exists()


def test_the_speedup_needs_both_backends():
    report = ServingReport(model="m", batch_size=1, latencies=[
        Latency(backend="torch", p50_ms=2.0), Latency(backend="onnxruntime", p50_ms=0.5)])
    assert report.speedup == pytest.approx(4.0)
    only_torch = ServingReport(model="m", latencies=[Latency(backend="torch", p50_ms=2.0)])
    assert only_torch.speedup is None


def test_the_markdown_says_what_was_not_measured():
    report = ServingReport(model="m", batch_size=1, latencies=[Latency(backend="torch", p50_ms=1.234)],
                           notes=["onnxruntime is not installed"])
    body = report.to_markdown()
    assert "| torch | 1.234 |" in body
    assert "Notes:" in body and "onnxruntime is not installed" in body
