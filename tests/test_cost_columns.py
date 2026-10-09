"""Cost columns of the benchmark protocol (#55): training time, inference per point, reference simulations."""
import math

import torch.nn as nn

import pinneapple as pp
from pinneapple_tools.benchmark_suite.benchmark import BenchmarkConfig, ModelSpec, PINNArenaBenchmark
from pinneapple_tools.benchmark_suite.tasks.heat_1d import Heat1DTask


def _mlp(i, o):
    return nn.Sequential(nn.Linear(i, 16), nn.Tanh(), nn.Linear(16, o))


def test_compare_reports_the_three_cost_columns():
    c = pp.compare("burgers_1d", ["analytic"], reference="analytic", n_points=300)
    row = c.rows[0]
    assert row["wall_time_s"] >= 0 and row["infer_us_per_point"] > 0 and row["n_reference_sims"] == 0
    head = str(c).splitlines()[1]
    assert "train [s]" in head and "infer [us/pt]" in head and "ref sims" in head


def test_benchmark_metrics_and_leaderboards_carry_the_cost_columns():
    task = Heat1DTask()
    task.n_reference_simulations = 3
    cfg = BenchmarkConfig(n_col=200, n_bc=50, n_ic=50, epochs=20, n_eval=200, log_interval=1000, device="cpu")
    bench = PINNArenaBenchmark([task], [ModelSpec("tiny_mlp", _mlp)], cfg)
    (r,) = bench.run(verbose=False)
    assert r.metrics["train_time_s"] > 0
    assert r.metrics["infer_us_per_point"] > 0 and not math.isnan(r.metrics["infer_us_per_point"])
    assert r.metrics["n_reference_sims"] == 3
    for table in (bench.leaderboard(by_problem=True), bench.leaderboard()):
        assert "train [s]" in table and "infer [us/pt]" in table and "ref sims" in table
