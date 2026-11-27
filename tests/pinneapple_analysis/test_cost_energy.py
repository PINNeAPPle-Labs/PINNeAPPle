"""Energy in the cost report (roadmap item K20, issue #119): joules from a named, sourced method, never a guess.

Two kinds of test, on purpose:

* the **arithmetic** is checked by hand on numbers chosen so the answer is exact (1e12 FLOPs at 2.0 pJ/FLOP is 2 J;
  65 W for 2 s is 130 J), because the point of the issue is that a reader can redo the sum from the report;
* the **honesty** is checked on what the library does when a number cannot be had: no FLOP count, no declared
  hardware, no readable counters. Each of those must print what was assumed or why there is no joule count, so no
  report can carry a joule figure whose provenance is missing.
"""
import numpy as np
import pytest

from pinneapple_analysis.cost import (
    DEFAULT_HARDWARE,
    ENERGY_PER_FLOP,
    CostLedger,
    RAPLMeter,
    cost_report,
    estimate_energy,
    fmt_joules,
    measure,
    measure_energy,
    rapl_packages,
)
from pinneapple_analysis.cost.energy import _JOULES_PER_KWH
from pinneapple_core.errors import BadValue

torch = pytest.importorskip("torch")

PJ_PER_FLOP = 2.0        # the declared cpu_45nm entry: (3.1 + 0.9) pJ per multiply-add / 2 FLOPs


# --------------------------------------------------------------------------- the arithmetic
def test_one_teraflop_at_the_declared_entry_is_two_joules():
    estimate = estimate_energy(flops=1e12, hardware="cpu_45nm")
    assert estimate.joules == pytest.approx(1e12 * PJ_PER_FLOP * 1e-12) == pytest.approx(2.0)
    assert estimate.method == "flops" and estimate.hardware == "cpu_45nm" and estimate.hardware_declared


def test_a_matmul_counted_by_torch_is_priced_from_its_own_flop_count():
    # 2 * 64 * 32 * 16 = 65 536 FLOPs; at 2 pJ/FLOP that is 1.31072e-7 J, which is the number in the report
    estimate = estimate_energy(flops=2 * 64 * 32 * 16, hardware="cpu_45nm")
    assert estimate.joules == pytest.approx(1.31072e-7, rel=1e-12)
    assert "2 pJ/FLOP" in estimate.derivation() and "131 nJ" in estimate.derivation()


def test_the_table_entry_is_the_arithmetic_of_the_source_it_cites():
    entry = ENERGY_PER_FLOP["cpu_45nm"]
    assert entry.pj_per_flop == pytest.approx((3.1 + 0.9) / 2)     # add + multiply, over 2 FLOPs per multiply-add
    assert "Horowitz" in entry.source and "ISSCC 2014" in entry.source and "45 nm" in entry.source


def test_power_times_time_is_the_other_method_and_needs_no_flop_count():
    estimate = estimate_energy(time_s=2.0, power_w=65.0, power_source="wall meter on this host")
    assert estimate.method == "power_time" and estimate.joules == pytest.approx(130.0)
    assert estimate.source == "wall meter on this host"


def test_a_declared_energy_per_flop_replaces_the_table_row_only_with_its_source():
    estimate = estimate_energy(flops=1e9, energy_per_flop_pj=10.0, energy_per_flop_source="measured on this NPU")
    assert estimate.joules == pytest.approx(1e9 * 10.0 * 1e-12) and estimate.hardware == "caller-declared"
    with pytest.raises(BadValue) as excinfo:            # a pJ/FLOP with no source is an invented number
        estimate_energy(flops=1e9, energy_per_flop_pj=10.0)
    assert "energy_per_flop_source" in str(excinfo.value)


def test_emissions_need_a_declared_intensity_and_follow_the_joules():
    estimate = estimate_energy(flops=1e12, hardware="cpu_45nm", carbon_g_per_kwh=150.0,
                               carbon_source="grid operator, 2025 average")
    assert estimate.co2_g == pytest.approx(2.0 / _JOULES_PER_KWH * 150.0)     # 8.33e-5 gCO2eq for 2 J
    with pytest.raises(BadValue):
        estimate_energy(flops=1e12, carbon_g_per_kwh=150.0)                  # an intensity with no provenance


# --------------------------------------------------------------------------- the honesty
def test_without_a_declared_hardware_the_report_says_what_it_assumed():
    estimate = estimate_energy(flops=1e9)
    assert estimate.hardware == DEFAULT_HARDWARE and not estimate.hardware_declared
    assert estimate.joules == pytest.approx(1e9 * PJ_PER_FLOP * 1e-12)        # the default entry's arithmetic
    assert estimate.assumptions and "no hardware was declared" in estimate.assumptions[0]
    assert "assumed, not declared" in estimate.report() and "Horowitz" in estimate.report()


def test_an_unknown_hardware_is_refused_and_the_known_ones_are_named():
    with pytest.raises(BadValue) as excinfo:
        estimate_energy(flops=1e9, hardware="cpu_7nm")
    message = str(excinfo.value)
    assert "cpu_45nm" in message and "energy_per_flop_pj" in message


def test_no_flop_count_and_no_declared_draw_gives_no_joule_count():
    estimate = estimate_energy(time_s=1.0)
    assert estimate.joules is None and estimate.method == "none"
    assert "no FLOP count" in estimate.reason and estimate.fix
    assert "not reported" in estimate.report()


def test_a_watt_without_a_source_is_refused():
    with pytest.raises(BadValue) as excinfo:
        estimate_energy(time_s=1.0, power_w=65.0)
    assert "power_source" in str(excinfo.value)


def test_the_report_prices_a_numpy_operation_as_not_estimable_rather_than_inventing_it():
    rec = measure(lambda: np.random.rand(120, 120) @ np.random.rand(120, 120), repeats=2, memory=False)
    assert rec.flops is None                                    # numpy work has no countable FLOPs
    text = cost_report(rec, hardware="cpu_45nm")
    assert "| not estimable |" in text and "no FLOP count" in text
    assert "Joules for" not in text                             # no priced block, so no joule figure at all


def test_the_report_prints_the_method_the_source_and_the_hardware_of_a_measured_matmul():
    rec = measure(lambda: torch.rand(64, 32) @ torch.rand(32, 16), repeats=3)
    assert rec.flops == 2 * 64 * 32 * 16 and rec.flops_source == "flop_counter"
    text = cost_report(rec, hardware="cpu_45nm", title="matmul 64x32x16")
    expected = fmt_joules(rec.flops * PJ_PER_FLOP * 1e-12)
    assert expected in text                                     # the row and the footer agree
    assert "method: flops" in text and "Horowitz" in text and "cpu_45nm (declared)" in text
    assert "derivation: 6.554e+04 FLOPs x 2 pJ/FLOP" in text
    assert "emissions: not reported" in text


def test_the_report_prices_a_ledger_row_by_row_and_groups_the_provenance():
    ledger = CostLedger()
    ledger.measure("matmul_256", lambda: torch.rand(256, 256) @ torch.rand(256, 256), repeats=3, memory=False)
    ledger.measure("numpy_gemm", lambda: np.random.rand(64, 64) @ np.random.rand(64, 64), repeats=2,
                   memory=False)
    text = cost_report(ledger, hardware="cpu_45nm")
    assert "2 operations" in text
    assert "Joules for 1 of 2 operations (`matmul_256`)" in text
    assert "No joules for 1 of 2 operations (`numpy_gemm`)" in text


def test_a_training_run_is_priced_from_its_wall_time_and_declared_draw():
    class Run:                                                  # the shape of equal_budget.BudgetRun
        name, wall_s, flops = "training", 4.0, 24e9
    # auto prefers the countable FLOPs; the declared draw prices the whole run when it is asked for by name
    text = cost_report(Run(), method="power_time", power_w=65.0, power_source="PSU readout of the host")
    assert "260 J" in text and "method: power_time" in text and "PSU readout" in text
    by_flops = cost_report(Run(), hardware="cpu_45nm")
    assert "48 mJ" in by_flops and "method: flops" in by_flops     # 24e9 FLOPs x 2 pJ/FLOP = 48 mJ


# --------------------------------------------------------------------------- RAPL, the measured path
def _fake_rapl(root, before_uj, after_uj, max_uj=262_143_328_850):
    package = root / "intel-rapl:0"
    package.mkdir(parents=True, exist_ok=True)
    (package / "max_energy_range_uj").write_text(str(max_uj))
    (package / "energy_uj").write_text(str(before_uj))

    def write_after():
        (package / "energy_uj").write_text(str(after_uj))
    return write_after


def test_rapl_delta_across_a_counter_wrap_is_the_same_energy(tmp_path):
    max_uj = 262_143_328_850
    write_after = _fake_rapl(tmp_path, max_uj - 500, 500, max_uj)     # wrapped: 1000 uJ were spent
    estimate = measure_energy(write_after, repeats=1, root=tmp_path)
    assert estimate.joules == pytest.approx(1000e-6)                  # 1000 uJ = 1 mJ
    assert estimate.method == "rapl" and estimate.measured and estimate.hardware_declared


def test_a_rapl_read_prices_the_block_between_the_counters(tmp_path):
    package = tmp_path / "intel-rapl:0"
    package.mkdir()
    (package / "max_energy_range_uj").write_text("262143328850")
    (package / "energy_uj").write_text("1000000")
    with RAPLMeter(root=tmp_path) as meter:
        (package / "energy_uj").write_text("3250000")                 # 2.25 J were spent inside the block
    assert meter.available and meter.joules == pytest.approx(2.25)


def test_without_readable_counters_the_measured_path_reports_the_absence(tmp_path):
    estimate = measure_energy(lambda: None, root=tmp_path)
    assert estimate.joules is None and estimate.method == "rapl" and not estimate.available
    assert "no RAPL counter" in estimate.reason and estimate.fix
    assert "not reported" in estimate.report()
    assert rapl_packages(tmp_path) == []


def test_the_package_counters_are_summed_but_not_their_subzones(tmp_path):
    for name, value in (("intel-rapl:0", 1000), ("intel-rapl:0:0", 400), ("intel-rapl:1", 2500)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "energy_uj").write_text(str(value))
    assert [p.name for p in rapl_packages(tmp_path)] == ["intel-rapl:0", "intel-rapl:1"]
