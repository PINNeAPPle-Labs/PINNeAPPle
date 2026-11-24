"""Compression and deployment: the error loss and the latency gain, both measured (#112).

The point of these tests is that a deployment change is a trade and that this module reports both sides of it on
the same rows: the relative L2 the compression cost and the p50 latency it bought, in one :class:`DeploymentReport`.
A test that only checked "the int8 file is smaller" would pass for a model that got three times slower.

The numbers quoted in the comments are what this machine measured when the test was written; the assertions are on
the relationship (inside tolerance, faster than the float model), not on the exact value.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from pinneapple_analysis.surrogate import deploy

N_ROWS = 4000            # the batch the int8 comparison runs on
PRUNE_ROWS = 50000       # pruning removes work, so it needs a batch where the work dominates the call overhead


def _truth(x: torch.Tensor) -> torch.Tensor:
    """A smooth function of 16 inputs into 8 outputs, cheap to learn and rich enough to have something to prune."""
    columns = [torch.sin(np.pi * x[:, :1]) + 0.5 * x[:, 1:2]]
    columns += [torch.sin(2 * np.pi * x[:, 2 * k:2 * k + 1]) * x[:, 2 * k + 1:2 * k + 2] for k in range(4)]
    columns += [(x ** 2).sum(1, keepdim=True), torch.cos(np.pi * x[:, -1:]), torch.tanh(x[:, :1] * x[:, 1:2])]
    return torch.cat(columns[:8], dim=1)


def _train(model: torch.nn.Module, inputs: int, outputs: int, steps: int = 400) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        batch = torch.rand(256, inputs)
        loss = ((model(batch) - _truth(batch)[:, :outputs]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


@pytest.fixture(scope="module")
def trained():
    """A surrogate that actually learned something: pruning or quantizing a random net measures nothing readable."""
    torch.manual_seed(0)
    model = _train(nn.Sequential(nn.Linear(16, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                                 nn.Linear(256, 8)), inputs=16, outputs=8)
    return model, torch.rand(N_ROWS, 16)


@pytest.fixture(scope="module")
def small():
    """The stack the pruning tests use: two 64-unit hidden layers, the shape a surrogate study produces."""
    torch.manual_seed(0)
    model = _train(nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 8)), inputs=16, outputs=8)
    return model, torch.rand(PRUNE_ROWS, 16)


# ---------------------------------------------------------------- the int8 path (ONNX Runtime, CPU)
def test_int8_reports_the_error_it_lost_and_the_latency_it_gained(trained):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model, x = trained
    report = deploy.compress_onnx_int8(model, x, tolerance=0.02, timing_runs=30)

    assert report.relative_l2 < report.tolerance, report.to_markdown()
    assert report.within_tolerance
    assert report.bytes_compressed < report.bytes_float            # measured ~3.7x smaller
    assert report.speedup > 1.0, report.to_markdown()              # measured ~1.3-1.9x faster, batched
    assert report.rows == N_ROWS and report.details["batch"] == N_ROWS
    assert report.details["single_row_float_ms"] > 0               # the losing side is recorded too
    assert report.tensorrt                                         # declared, with the reason


def test_the_verdict_follows_the_tolerance_the_caller_set(trained):
    """The same compression, judged against a tolerance no lossy path can meet: the report says so instead of passing."""
    pytest.importorskip("onnxruntime")
    model, x = trained
    strict = deploy.compress_onnx_int8(model, x, tolerance=1e-6, timing_runs=5)
    assert not strict.within_tolerance
    assert strict.relative_l2 > 0


def test_the_extras_declare_themselves_missing(monkeypatch, trained):
    model, x = trained
    monkeypatch.setattr(deploy, "_has", lambda module: False)
    with pytest.raises(deploy.CompressionUnavailable, match="onnxruntime"):
        deploy.compress_onnx_int8(model, x)
    assert "needs onnxruntime" in deploy.quantization_status()["onnx-int8"]


# ---------------------------------------------------------------- the pruning path (torch only)
def test_structured_pruning_shrinks_the_matrices_and_reports_what_it_cost(small):
    """Pruning without fine-tuning is **not** deployable here, and the report is what says so.

    Measured at 50 000 rows: keeping 95 % of the units costs 11.6 % error for 1.16x, keeping 50 % costs 81.5 %
    for 2.15x. The latency gain is real — the matrices genuinely get smaller — and the accuracy loss is measured
    in the same report, which is the whole point: `within_tolerance` is False at the 5 % tolerance a surrogate
    would be deployed under, so nobody reads "prunable" as "free".
    """
    model, x = small
    pruned = deploy.prune_structured(model, keep=0.5)
    report = deploy.measure_pruned(model, pruned, x, n_rows=PRUNE_ROWS, tolerance=0.05, timing_runs=15)

    assert report.details["parameters_compressed"] < 0.6 * report.details["parameters_float"]
    assert report.speedup > 1.2, report.to_markdown()               # measured ~2x at 50k rows
    assert report.relative_l2 > 0.05, report.to_markdown()          # and it is not inside the tolerance
    assert not report.within_tolerance


def test_pruning_harder_costs_more_accuracy(small):
    """The monotone relation is the evidence that the error is measured and not a number copied from a docstring."""
    model, x = small
    gentle = deploy.measure_pruned(model, deploy.prune_structured(model, keep=0.75), x, n_rows=N_ROWS,
                                   timing_runs=5)
    harsh = deploy.measure_pruned(model, deploy.prune_structured(model, keep=0.25), x, n_rows=N_ROWS, timing_runs=5)
    assert harsh.relative_l2 > gentle.relative_l2
    assert harsh.details["parameters_compressed"] < gentle.details["parameters_compressed"]


def test_pruning_refuses_a_model_with_no_hidden_units_to_remove():
    with pytest.raises(ValueError, match="nn.Sequential"):
        deploy.prune_structured(nn.Conv1d(1, 1, 1), keep=0.5)


# ---------------------------------------------------------------- torch's own int8, measured honestly
def test_torch_int8_runs_and_reports_what_it_measured(trained):
    """Available is not the same as worth taking: the latency is reported whether it went up or down."""
    model, x = trained
    report = deploy.measure_torch_dynamic(model, x, timing_runs=20)
    assert report.relative_l2 < 0.05, report.to_markdown()
    assert np.isfinite(report.speedup) and report.speedup > 0
    assert np.isnan(report.size_ratio)                              # packed int8 weights are not parameters
    assert "not comparable" in report.details["size_note"]


# ---------------------------------------------------------------- what is declared
def test_tensorrt_is_declared_with_its_reason_instead_of_left_out():
    status = deploy.tensorrt_status()
    assert set(status) == {"available", "reason"}
    if not status["available"]:
        assert "GPU" in status["reason"] or "tensorrt" in status["reason"]


def test_the_status_lists_every_path_and_says_which_ones_run():
    status = deploy.quantization_status()
    assert set(status) == {"onnx-int8", "structured-prune", "torch-int8", "tensorrt"}
    assert status["structured-prune"] == "ok"
    assert all(isinstance(sentence, str) and sentence for sentence in status.values())
