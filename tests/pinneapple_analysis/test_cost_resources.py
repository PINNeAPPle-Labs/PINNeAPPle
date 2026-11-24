"""The measured cost decides the resources (issue #131): kind, cores and memory, each with the rule it follows.

Two kinds of test here, on purpose:

* the **rule** is tested on records built by hand, because a rule has to be checked at its boundaries (a rate just
  under 10 GFLOP/s is not a GPU job) and a measured 10 GFLOP/s would take a minute to produce;
* the **path** is tested once from a real measurement (``measure`` on a real operation) through ``CostLedger`` to
  the saved table, because the point of the issue is that the number comes from measurement rather than a label.
"""
import json

import numpy as np
import pytest

from pinneapple_analysis.cost import (
    JobProfileTable,
    MachineRecord,
    classify,
    measure,
    plan,
    recommend_cores,
)
from pinneapple_analysis.cost.profiler import CostLedger, CostRecord
from pinneapple_analysis.cost.resources import (
    CPU_HEAVY,
    CPU_LIGHT,
    GPU_FLOPS_PER_S,
    GPU_PREFERRED,
    HEAVY_FLOPS_PER_S,
    HEAVY_MEMORY_BYTES,
    HEAVY_TIME_S,
    IO_BOUND,
    IO_BYTES_PER_S,
)

MIB = 1024 * 1024


def record(**kw) -> CostRecord:
    """A record with only the fields that matter here; the rest at neutral values."""
    base = dict(name="op", n_runs=3, time_s=0.1, time_min_s=0.1, time_iqr_s=0.0)
    base.update(kw)
    return CostRecord(**base)


# --------------------------------------------------------------------------- kind
def test_the_measured_work_rate_decides_the_kind():
    assert classify(record(time_s=0.5, flops=int(GPU_FLOPS_PER_S * 0.5 * 2), peak_mem_bytes=MIB)) == GPU_PREFERRED
    assert classify(record(time_s=0.5, flops=int(HEAVY_FLOPS_PER_S * 0.5 * 2), peak_mem_bytes=MIB)) == CPU_HEAVY
    assert classify(record(time_s=0.5, flops=1000, peak_mem_bytes=MIB)) == CPU_LIGHT


def test_a_cuda_peak_is_a_gpu_job_whatever_the_flops_say():
    """The allocator that reported the peak is evidence stronger than the FLOP counter (which sees no CUDA work)."""
    assert classify(record(mem_source="cuda", peak_mem_bytes=2 * 1024 ** 3, time_s=10.0)) == GPU_PREFERRED


def test_an_operation_that_neither_computes_nor_holds_memory_is_waiting_not_light():
    """No countable FLOPs and under 1 MiB/s held: the time goes to disk or to a subprocess, so more cores are not
    the answer. This is the kind a static label gets wrong most often."""
    waiting = record(time_s=2.0, peak_mem_bytes=int(IO_BYTES_PER_S * 2 * 0.5), flops=None)
    assert classify(waiting) == IO_BOUND
    light = record(time_s=2.0, peak_mem_bytes=int(IO_BYTES_PER_S * 2 * 4), flops=None)
    assert classify(light) == CPU_LIGHT


def test_long_and_big_is_heavy_even_without_a_flop_count():
    assert classify(record(time_s=HEAVY_TIME_S * 2, peak_mem_bytes=HEAVY_MEMORY_BYTES * 2, flops=None)) == CPU_HEAVY
    # one of the two is not enough: long but small holds nothing, so it is light; big but quick passes through
    assert classify(record(time_s=0.5, peak_mem_bytes=4 * MIB, flops=None)) == CPU_LIGHT
    assert classify(record(time_s=0.01, peak_mem_bytes=HEAVY_MEMORY_BYTES * 2, flops=None)) == CPU_LIGHT


# --------------------------------------------------------------------------- cores
def _curve(times):
    return {n: record(time_s=t, name=f"op@{n}") for n, t in times.items()}


def test_cores_are_the_largest_count_that_still_uses_the_cores_it_is_given():
    # 4 workers give 3.2x: efficiency 0.8; 8 workers give 4x: efficiency 0.5, below the floor
    workers, efficiency, basis = recommend_cores(_curve({1: 8.0, 2: 4.4, 4: 2.5, 8: 2.0}))
    assert workers == 4 and efficiency == pytest.approx(0.8, abs=0.01)
    assert basis["speedup"][-1] == pytest.approx(4.0, abs=0.01)
    assert basis["measured_workers"] == [1, 2, 4, 8]


def test_cores_never_exceed_what_was_measured_and_say_when_the_deadline_is_out_of_reach():
    """No extrapolation: an unreachable deadline returns the largest measured count, with the reason recorded."""
    workers, _, basis = recommend_cores(_curve({1: 8.0, 4: 2.5}), deadline_s=0.5)
    assert workers == 4
    assert "deadline" in basis["note"]


def test_a_deadline_pulls_the_count_down_to_the_fewest_cores_that_is_on_time():
    """8 workers are the throughput choice, but 2 already finish inside the deadline: ask for 2."""
    workers, efficiency, _ = recommend_cores(_curve({1: 8.0, 2: 3.0, 4: 2.6, 8: 2.4}), deadline_s=3.5)
    assert workers == 2 and efficiency == pytest.approx(1.0 / 3.0 / 2 * 8, abs=0.01)


def test_a_scale_that_gets_faster_per_core_is_not_penalised():
    """Efficiency above 1 (a smaller problem finishing faster per core) keeps the largest measured count."""
    assert recommend_cores(_curve({1: 4.0, 2: 1.0}))[0] == 2


def test_recommend_cores_refuses_an_empty_or_timeless_measurement():
    with pytest.raises(ValueError, match="at least one measurement"):
        recommend_cores({})
    with pytest.raises(ValueError, match="no time"):
        recommend_cores({1: record(time_s=0.0)})


# --------------------------------------------------------------------------- memory and the profile
def test_memory_is_the_peak_times_the_headroom_and_the_campaign_total_is_reported_separately():
    prof = plan(record(peak_mem_bytes=100 * MIB, mem_exact=True, time_s=1.5), workers=4, headroom=1.25)
    assert prof.memory_bytes == 125 * MIB
    assert prof.campaign_memory_bytes == 500 * MIB
    assert prof.cores == 4 and not prof.memory_is_floor


def test_a_lower_bound_peak_asks_for_a_floor_and_says_so():
    """Sampled RSS is not a measurement of what the code asked for: the request must not pretend otherwise."""
    prof = plan(record(peak_mem_bytes=64 * MIB, mem_exact=False, mem_source="rss", time_s=1.0))
    assert prof.memory_is_floor and "floor" in prof.notes
    unmeasured = plan(record(peak_mem_bytes=None, mem_exact=False, mem_source="unknown", time_s=1.0))
    assert unmeasured.memory_is_floor and "no peak was measurable" in unmeasured.notes


def test_a_headroom_below_one_is_refused():
    with pytest.raises(ValueError, match="headroom"):
        plan(record(peak_mem_bytes=MIB), headroom=0.9)


def test_the_profile_records_the_machine_it_was_measured_on():
    here = MachineRecord.here()
    prof = plan(record(peak_mem_bytes=MIB, mem_exact=True), measured_on=here)
    assert prof.to_dict()["measured_on"] == here.to_dict()
    assert here.cpu_count >= 1 and here.system and here.python


# --------------------------------------------------------------------------- the path from a measurement
def test_the_table_is_generated_from_a_real_measurement_and_carries_the_machine(tmp_path):
    ledger = CostLedger()
    ledger.measure("matmul", lambda: float(np.ones(256 * 256).sum()), repeats=2, flops=False)

    table = JobProfileTable.from_ledger(ledger, workers=2)
    prof = table.profiles["matmul"]
    assert prof.measured_on == table.measured_on and prof.measured_on.cpu_count >= 1
    assert prof.memory_bytes >= (ledger.records["matmul"].peak_mem_bytes or 0)
    assert prof.kind in (CPU_LIGHT, CPU_HEAVY, IO_BOUND)

    path = table.save(str(tmp_path / "job_profiles.json"))
    saved = json.loads(open(path).read())
    assert saved["schema"] == "pinneapple.job_profiles/1"
    assert saved["measured_on"]["cpu_count"] >= 1
    assert saved["profiles"]["matmul"]["workers"] == 2
    assert "Measured on" in table.to_markdown()


def test_a_cores_curve_in_the_table_sets_the_worker_count_and_keeps_its_basis():
    table = JobProfileTable()
    prof = table.add(record(name="solver", time_s=8.0, peak_mem_bytes=256 * MIB, mem_exact=True),
                     cores=_curve({1: 8.0, 2: 4.4, 4: 2.5, 8: 2.0}))
    assert prof.workers == 4 and prof.efficiency == pytest.approx(0.8, abs=0.01)
    assert prof.basis["cores"]["speedup"] == [1.0, 1.818, 3.2, 4.0]
    assert prof.to_dict()["cores"] == 4


def test_measured_pinn_step_gets_a_heavy_profile():
    """A real training step is CPU heavy by measurement, and asks for more memory than it alone holds."""
    import torch
    model = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
    x = torch.randn(2048, 2)

    def step():
        model(x).sum().backward()
        for p in model.parameters():
            p.grad = None

    rec = measure(step, repeats=2, flops=False)
    prof = plan(rec, workers=1)
    assert prof.measured_on.cpu_count >= 1
    assert prof.memory_bytes >= (rec.peak_mem_bytes or 0)
    assert classify(rec) in (CPU_LIGHT, CPU_HEAVY)
