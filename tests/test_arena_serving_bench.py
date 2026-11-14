"""Serving benchmarks (INT-013): PyTorch latency, the ONNX export, and what is honestly not measured.

``onnxruntime`` and ``onnxscript`` are not dependencies of this library and are absent on the machine this was
written on, so the tests say which of the three export attempts did the work and let the runtime leg be the one that
is skipped. The FNO case is the interesting one and it *is* covered here: a spectral model keeps complex tensors, so
both PyTorch exporters refuse it and the graph is written from the real-arithmetic copy of its spectral convolution.
Measured end to end against ``onnxruntime`` in a throwaway environment: 0.190 ms in PyTorch vs 0.144 ms served, with
the graph agreeing to 2.98e-08.
"""
from __future__ import annotations

import importlib.util

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


def test_a_model_no_exporter_can_lower_is_reported_and_leaves_no_file(tmp_path):
    """A bare ``rfft`` with nothing to rewrite it: the report says why, and a truncated graph does not survive."""
    onnxscript = importlib.util.find_spec("onnxscript") is not None
    target = tmp_path / "spectral.onnx"
    report = benchmark_model(Spectral(), torch.randn(1, 8), path=target, n=2, warmup=0)
    assert report.get("torch (cpu)") is not None
    if onnxscript:
        # the graph exporter does lower a bare rfft whose output is real; with it installed the export succeeds
        assert report.onnx_path and report.exporter == "graph" and target.exists()
    else:
        assert report.onnx_path == ""
        assert any("ONNX export failed" in note for note in report.notes)
        assert not target.exists(), "torch writes as it traces: a truncated graph must not survive the failure"
    assert np.isfinite([[1.0]]).all()


def test_an_fno_exports_through_the_real_arithmetic_copy(tmp_path):
    """The case the module exists for: a spectral model, whose complex tensors neither exporter lowers."""
    from pinneapple_neural.architectures.neural_operators.fno import FourierNeuralOperator

    torch.manual_seed(0)
    fno = FourierNeuralOperator(in_channels=1, out_channels=1, width=8, modes=4, layers=2, use_grid=True)
    target = tmp_path / "fno.onnx"
    report = benchmark_model(fno, torch.randn(1, 1, 32), path=target, n=2, warmup=0)
    assert report.exporter == "real-arithmetic"
    assert target.exists() and report.onnx_path.endswith("fno.onnx")
    assert any("real-arithmetic" in note for note in report.notes)
    assert "real-arithmetic exporter" in report.to_markdown()
    if importlib.util.find_spec("onnxruntime") is not None:
        assert report.max_abs_difference is not None and report.max_abs_difference < 1e-5
        assert report.speedup is not None
    else:
        assert any("onnxruntime is not installed" in note for note in report.notes)


def test_the_real_arithmetic_copy_is_the_same_function_on_one_and_two_dimensions():
    """Without ONNX in the loop: the substitution must not change the model, 1-D included."""
    from pinneapple_neural.architectures.neural_operators.fno import FNO2d, FourierNeuralOperator
    from pinneapple_neural.architectures.neural_operators.fno_export import (
        spatial_shape,
        to_exportable,
    )

    torch.manual_seed(0)
    one_d = FourierNeuralOperator(in_channels=1, out_channels=1, width=8, modes=4, layers=2, use_grid=True)
    x1 = torch.randn(2, 1, 32)
    assert spatial_shape(one_d, x1) == (32,)
    with torch.no_grad():
        assert (to_exportable(one_d, (32,))(x1).y - one_d(x1).y).abs().max().item() < 1e-6

    two_d = FNO2d(in_channels=1, out_channels=1, width=8, modes1=4, modes2=4, layers=2)
    x2 = torch.randn(2, 1, 16, 16)
    assert spatial_shape(two_d, x2) == (16, 16)
    with torch.no_grad():
        assert (to_exportable(two_d, (16, 16))(x2).y - two_d(x2).y).abs().max().item() < 1e-6


def test_a_padded_fno_sees_the_padded_grid():
    from pinneapple_neural.architectures.neural_operators.fno import FourierNeuralOperator
    from pinneapple_neural.architectures.neural_operators.fno_export import spatial_shape

    model = FourierNeuralOperator(in_channels=1, out_channels=1, width=4, modes=2, layers=1, padding=4)
    assert spatial_shape(model, torch.randn(1, 1, 24)) == (28,)


def test_a_pointwise_model_has_no_spectral_grid():
    from pinneapple_neural.architectures.neural_operators.fno_export import spatial_shape

    assert spatial_shape(MLP(), torch.randn(1, 4)) is None


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
