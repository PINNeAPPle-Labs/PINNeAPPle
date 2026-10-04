"""pinneapple_analysis.cost: per-operation cost measurement, scaling fits, budgets, regression ledger."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_analysis.cost import (
    Budget, BudgetExceeded, CostLedger, CostRecord, expected_exponent, measure, pinn_operation_costs,
    profile_ops, scaling_study,
)
from pinneapple_analysis.cost.profiler import _fit_exponent, _verdict


def test_measure_reports_exact_matmul_flops_and_none_for_numpy():
    a, b = torch.rand(64, 32), torch.rand(32, 16)
    rec = measure(lambda: a @ b, repeats=3)
    assert rec.flops == 2 * 64 * 32 * 16 and rec.flops_source == "flop_counter"
    assert rec.time_s > 0 and rec.time_min_s <= rec.time_s and len(rec.times_s) == 3
    x = np.random.rand(200, 200)
    assert measure(lambda: x @ x, repeats=2).flops is None  # numpy work is never given a made-up count


def test_flops_scale_exactly_as_n_cubed_for_matmul():
    study = scaling_study(lambda n: (lambda a=torch.rand(n, n): a @ a), [32, 64, 128, 256], repeats=3,
                          declared_flops="n^3")
    f = study.fits["flops"]
    assert f.exponent == pytest.approx(3.0, abs=1e-9) and f.verdict == "consistent"


def test_time_scaling_detects_quadratic_work_and_flags_a_wrong_declaration():
    def quad(n):
        def run():
            s = 0
            for i in range(n):
                for j in range(n):
                    s += i ^ j
            return s
        return run
    sizes = [60, 120, 240, 480]
    ok = scaling_study(quad, sizes, repeats=5, declared="n^2", name="quad")
    # wide bounds: this runs in parallel test workers, where load noise bends a time fit
    assert 1.5 < ok.fits["time"].exponent < 3.0
    wrong = scaling_study(quad, sizes, repeats=5, declared="n", name="quad")
    assert wrong.fits["time"].verdict == "worse than declared"
    assert "worse than declared" in wrong.summary()


def test_expected_exponent_for_n_log_n_is_between_one_and_two_and_unknown_raises():
    k = expected_exponent("n log n", [10 ** 4, 10 ** 5, 10 ** 6])
    assert 1.0 < k < 1.2
    assert expected_exponent("n^2", [10, 100, 1000]) == pytest.approx(2.0)
    assert expected_exponent(1.5, [1, 2, 3]) == 1.5 and expected_exponent("1", [1, 2, 3]) == 0.0
    with pytest.raises(ValueError):
        expected_exponent("n^7", [1, 2, 3])


def test_fit_and_verdict_logic_on_exact_power_laws():
    n = np.array([10.0, 100.0, 1000.0, 10000.0])
    k, b, r2 = _fit_exponent(n, 3e-9 * n ** 2)
    assert k == pytest.approx(2.0) and r2 == pytest.approx(1.0) and np.exp(b) == pytest.approx(3e-9)
    assert _verdict((1.9, 2.1), 2.0, 0.15) == "consistent"
    assert _verdict((2.6, 2.9), 2.0, 0.15) == "worse than declared"
    assert _verdict((0.9, 1.1), 2.0, 0.15) == "better than declared"


def test_scaling_study_needs_three_distinct_sizes():
    with pytest.raises(ValueError):
        scaling_study(lambda n: (lambda: n), [10, 20])


def test_budget_reports_and_raises_and_exponent_uses_the_ci_lower_bound():
    rec = CostRecord("op", 3, time_s=2.0, time_min_s=1.9, time_iqr_s=0.1, peak_mem_bytes=10_000, flops=5_000)
    assert Budget(max_time_s=5, max_mem_bytes=20_000, max_flops=10_000).check(rec) == []
    v = Budget(max_time_s=1.0, max_flops=1_000).check(rec)
    assert len(v) == 2 and "time" in v[0] and "FLOPs" in v[1]
    with pytest.raises(BudgetExceeded):
        Budget(max_time_s=1.0).validate(rec)
    study = scaling_study(lambda n: (lambda a=torch.rand(n, n): a @ a), [32, 64, 128], repeats=3)
    study.fits["time"].ci = (0.5, 2.4)  # wide CI that straddles the limit: noise alone must not fail the gate
    assert Budget(max_time_exponent=2.0).check(study) == []
    study.fits["time"].ci = (2.5, 2.9)
    assert Budget(max_time_exponent=2.0).check(study)


def test_ledger_round_trip_and_regression_gate(tmp_path):
    a = torch.rand(64, 64)
    base = CostLedger()
    base.measure("matmul", lambda: a @ a, repeats=3)
    path = base.save(str(tmp_path / "cost_baseline.json"))
    loaded = CostLedger.load(path)
    assert loaded.records["matmul"].flops == base.records["matmul"].flops

    bigger = torch.rand(96, 96)
    now = CostLedger()
    now.measure("matmul", lambda: bigger @ bigger, repeats=3)
    regs = now.compare(loaded, time_factor=1e9)  # ignore noisy time; FLOPs are exact
    assert [r.quantity for r in regs] == ["flops"] and regs[0].ratio == pytest.approx((96 / 64) ** 3)
    assert now.compare(now) == []
    assert str(regs[0]).startswith("matmul: flops")


def test_ledger_track_decorator_stores_the_record_and_returns_the_result():
    ledger = CostLedger()

    @ledger.track("square", repeats=2)
    def square(x):
        return x * x

    assert square(torch.tensor([3.0])).item() == 9.0
    assert ledger.records["square"].n_runs == 2


def test_pinn_operation_costs_forward_flops_are_exact_and_derivatives_cost_more():
    model = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 16),
                                torch.nn.Tanh(), torch.nn.Linear(16, 1))
    n = 256
    costs = pinn_operation_costs(model, torch.rand(n, 2), repeats=2)
    assert set(costs) == {"forward", "grad1", "grad2", "backward", "step"}
    assert costs["forward"].flops == 2 * n * (2 * 16 + 16 * 16 + 16 * 1)
    assert costs["forward"].flops < costs["grad1"].flops < costs["grad2"].flops < costs["backward"].flops
    assert costs["grad1"].flops_source == "profiler"  # flop_counter cannot run around autograd.grad


def test_profile_ops_lists_the_matmul_with_its_flops():
    lin = torch.nn.Linear(8, 4)
    x = torch.rand(32, 8)
    rows = profile_ops(lambda: lin(x), top=10)
    mm = [r for r in rows if r["flops"]]
    assert mm and mm[0]["flops"] == 2 * 32 * 8 * 4
