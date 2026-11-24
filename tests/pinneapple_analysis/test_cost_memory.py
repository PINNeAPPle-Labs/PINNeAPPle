"""Peak memory measured, not guessed (issue #54): each method against an allocation of known size.

The reference is the size the operation allocates and keeps alive, so a measurement can be compared with it
directly and the 10% requirement is a real bound rather than a label. The tests also pin the *limits* of each
method, because those limits are the reason ``measure_peak`` defaults to what it does: tracemalloc is blind to
tensors, the PyTorch profiler is blind to NumPy, and the forked-child method cannot run autograd.
"""
import os
import sys

import numpy as np
import pytest

from pinneapple_analysis.cost import Budget, measure, measure_peak
from pinneapple_analysis.cost.memory import (
    MB,
    cuda_peak,
    process_peak,
    rss_peak,
    torch_peak,
    traced_peak,
)

torch = pytest.importorskip("torch")

TOL = 0.10   # the issue's bar: within 10% of the reference


def within_ten_percent(measured, reference):
    return measured is not None and abs(measured - reference) <= TOL * reference


# --------------------------------------------------------------------------- NumPy and Python
def test_tracemalloc_measures_a_numpy_array_within_ten_percent_of_its_size():
    target = 64 * MB

    def op():
        a = np.ones(target // 8, dtype=np.float64)
        return float(a[0])

    peak = measure_peak(op, method="tracemalloc")
    assert peak.source == "tracemalloc" and peak.exact
    assert within_ten_percent(peak.bytes, target), f"{peak.bytes} vs {target}"


def test_torch_profiler_is_blind_to_numpy_and_says_so():
    def op():
        return float(np.ones(32 * MB // 8).sum())

    assert torch_peak(op) == 0           # zero tensor bytes, not "no measurement"
    assert measure_peak(op, method="auto").components.get("torch", 0) == 0


# --------------------------------------------------------------------------- PyTorch CPU
def test_torch_profiler_measures_live_tensor_peak_within_ten_percent():
    """Two tensors alive together: the peak is their sum, not the last one allocated."""
    target = 64 * MB

    def op():
        a = torch.ones(target // 2 // 4, dtype=torch.float32)
        b = torch.ones(target // 2 // 4, dtype=torch.float32)
        return torch.stack([a.sum(), b.sum()])

    peak = torch_peak(op)
    assert within_ten_percent(peak, target), f"{peak} vs {target}"
    assert within_ten_percent(measure_peak(op, method="auto").bytes, target)


def test_torch_peak_tracks_the_high_water_mark_when_tensors_are_freed_and_reallocated():
    """Sequential 48 MiB then 48 MiB is 48 MiB, not 96: the profiler's running sum has to go down on free."""
    target = 48 * MB

    def op():
        a = torch.ones(target // 4, dtype=torch.float32)
        a.sum()
        del a
        b = torch.ones(target // 4, dtype=torch.float32)
        return b.sum()

    peak = torch_peak(op)
    assert within_ten_percent(peak, target), f"{peak} vs {target}"


def test_mixed_numpy_and_torch_are_summed_and_the_parts_are_reported():
    """NumPy visible only to tracemalloc, tensors only to the profiler: the two add up to the live peak."""
    numpy_bytes, torch_bytes = 32 * MB, 32 * MB

    def op():
        a = np.ones(numpy_bytes // 8)
        t = torch.ones(torch_bytes // 4, dtype=torch.float32)
        return float(a[0]) + float(t.sum())

    peak = measure_peak(op)
    assert peak.source == "torch+trace" and peak.exact
    assert within_ten_percent(peak.components["torch"], torch_bytes)
    assert within_ten_percent(peak.components["python"], numpy_bytes)
    assert within_ten_percent(peak.bytes, numpy_bytes + torch_bytes), f"{peak.bytes} vs {numpy_bytes + torch_bytes}"


# --------------------------------------------------------------------------- a real PINN pass
def _pinn_step(n_points=1024, width=64, workspace_bytes=0):
    """One PINN training step (forward, first and second derivatives, loss backward, Adam step).

    ``workspace_bytes`` keeps a tensor of exactly that size alive for the whole step, which is how a known
    reference gets inside a real pass: the extra footprint is known to the byte, so the difference between two
    measurements has a reference to be compared against.
    """
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, width), torch.nn.Tanh(),
        torch.nn.Linear(width, width), torch.nn.Tanh(),
        torch.nn.Linear(width, 1),
    )
    x = torch.randn(n_points, 2, requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    keep = {}                                            # holds the workspace across the whole step

    def step():
        if workspace_bytes:
            keep["w"] = torch.ones(workspace_bytes // 4, dtype=torch.float32)
        xx = x.detach().clone().requires_grad_(True)
        u = model(xx)
        g = torch.autograd.grad(u.sum(), xx, create_graph=True)[0]
        lap = sum(torch.autograd.grad(g[:, i].sum(), xx, create_graph=True)[0][:, i] for i in range(xx.shape[1]))
        loss = (lap ** 2).mean() + (u ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        keep.clear()

    step()                                   # Adam state exists before it is measured
    return step


def test_pinn_training_step_peak_matches_a_known_injected_workspace_within_ten_percent():
    """The issue's own case: one PINN step, measured within 10% of a reference of known size.

    The reference is a workspace of exactly 96 MiB held for the whole step. The measured peak has to move by
    that much and no more, which is what pins the scale of the measurement inside a real autograd pass — the
    case where ``process`` cannot be used and sampled RSS is useless (measured: re-running the same step after a
    warm-up grows RSS by 45 % of the peak at one size and 12 % at another, because the allocator reuses the
    arena; the live-allocation peak does not move at all, correctly).
    """
    workspace = 96 * MB
    peak = measure_peak(_pinn_step(workspace_bytes=workspace))
    assert peak.exact and peak.source in ("torch", "torch+trace")
    smaller = measure_peak(_pinn_step())
    delta = peak.bytes - smaller.bytes
    assert within_ten_percent(delta, workspace), f"delta {delta} vs workspace {workspace}"


def test_pinn_training_step_counts_its_widest_activation():
    """A lower bound that needs no model of the autograd graph: the widest activation is resident at one point."""
    n_points, width = 1024, 64
    peak = measure_peak(_pinn_step(n_points=n_points, width=width))
    assert peak.bytes >= n_points * width * 4                      # one float32 activation row
    assert peak.bytes < 64 * MB                                    # and it is nowhere near a whole gigabyte


def test_measure_records_the_peak_with_its_source_and_parts():
    rec = measure(_pinn_step(), repeats=2, flops=False)
    assert rec.mem_exact and rec.mem_source in ("torch", "torch+trace")
    assert rec.peak_mem_bytes and rec.peak_mem_bytes > 0
    assert rec.mem_components.get("torch", 0) > 0
    assert rec.to_dict()["mem_exact"] is True


# --------------------------------------------------------------------------- method limits
def test_tracemalloc_is_blind_to_torch_tensors_which_is_why_auto_sums_two_methods():
    def op():
        return torch.ones(32 * MB // 4).sum()

    assert (traced_peak(op) or 0) < 1000
    assert within_ten_percent(torch_peak(op), 32 * MB)


@pytest.mark.skipif(not hasattr(os, "fork") or sys.platform.startswith("win"), reason="needs fork")
def test_forked_child_watermark_is_exact_for_forward_work():
    target = 64 * MB

    def op():
        return float(np.ones(target // 8).sum())

    peak = process_peak(op)
    assert peak is not None
    assert within_ten_percent(peak, target), f"{peak} vs {target}"


@pytest.mark.skipif(not hasattr(os, "fork") or sys.platform.startswith("win"), reason="needs fork")
def test_forked_child_cannot_run_autograd_and_returns_none_instead_of_a_number():
    """The limit that keeps ``process`` out of the default: PyTorch refuses fork with autograd's threads."""
    model = torch.nn.Linear(2, 1)
    x = torch.ones(8, 2)

    def op():
        model(x).sum().backward()

    assert process_peak(op) is None
    assert measure_peak(op, method="process").source == "unknown"
    assert measure_peak(op, method="process").bytes is None


def test_rss_sampling_is_labelled_a_lower_bound_and_missing_psutil_is_not_a_zero():
    try:
        import psutil  # noqa: F401
    except ImportError:
        peak = measure_peak(lambda: 1, method="rss")
        assert peak.bytes is None and peak.source == "unknown" and "psutil" in peak.notes
        assert rss_peak(lambda: 1) is None
    else:
        peak = measure_peak(lambda: np.ones(32 * MB // 8), method="rss")
        assert peak.source == "rss" and not peak.exact


def test_unknown_method_raises_and_no_cuda_reports_unknown_not_zero():
    with pytest.raises(ValueError, match="unknown method"):
        measure_peak(lambda: 1, method="telepathy")
    if not torch.cuda.is_available():
        assert cuda_peak(lambda: 1) is None
        assert measure_peak(lambda: 1, method="cuda").bytes is None
        assert measure_peak(lambda: 1, method="cuda").source == "unknown"


def test_budget_still_checks_a_none_peak_as_absent():
    """A peak that could not be measured must not pass a budget silently or fail it spuriously."""
    rec = measure(lambda: sum(range(1000)), repeats=1, flops=False, mem_method="cuda")
    assert rec.peak_mem_bytes is None
    assert Budget(max_mem_bytes=1).check(rec) == []
