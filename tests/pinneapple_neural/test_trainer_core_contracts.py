"""Core coverage for the trainer's metrics and parallel helpers (#1129, toward 1.0), on CPU.

The metrics are compared with values computed by hand; the helpers with what they promise on a machine without a GPU
(the CI case): the CPU device, no-op wrappers, gradient accumulation equal to one large batch, a sweep over a grid.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from pinneapple_neural.trainer.metrics import (  # noqa: E402
    MAE,
    MSE,
    R2,
    RMSE,
    MaxError,
    MetricBundle,
    RegressionMetrics,
    RelL2,
    default_metrics,
    regression_metrics_bundle,
)
from pinneapple_neural.trainer.parallel import (  # noqa: E402
    AMPContext,
    CUDAPrefetcher,
    GradAccumConfig,
    GradAccumTrainer,
    SweepConfig,
    ThroughputMonitor,
    best_device,
    count_gpus,
    enable_gradient_checkpointing,
    gpu_info,
    maybe_compile,
    run_parallel_sweep,
    unwrap_model,
    wrap_data_parallel,
)

Y = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
YH = torch.tensor([[1.5], [2.0], [2.0], [4.0]])          # errors 0.5, 0, -1, 0


def test_each_metric_against_a_hand_computed_reference():
    assert MSE()(YH, Y) == pytest.approx((0.25 + 1.0) / 4)
    assert MAE()(YH, Y) == pytest.approx((0.5 + 1.0) / 4)
    assert RMSE()(YH, Y) == pytest.approx(math.sqrt(1.25 / 4))
    assert R2()(YH, Y) == pytest.approx(1 - 1.25 / 5.0)                    # ss_tot = sum (y - 2.5)^2 = 5
    assert RelL2()(YH, Y) == pytest.approx(math.sqrt(1.25) / math.sqrt(30.0))
    assert MaxError()(YH, Y) == pytest.approx(1.0)
    assert MSE()(Y, Y) == 0.0 and R2()(Y, Y) == pytest.approx(1.0)


def test_bundles_agree_with_the_single_metrics():
    full = MetricBundle(default_metrics()).compute(YH, Y)
    assert set(full) == {"mse", "mae", "rmse", "r2", "rel_l2", "max_error"}
    reg = RegressionMetrics().compute(YH, Y)
    small = regression_metrics_bundle().compute(YH, Y)
    for k in ("mse", "mae", "rmse", "r2"):
        assert reg[k] == pytest.approx(full[k]) == pytest.approx(small[k])


def test_device_helpers_on_a_cpu_only_machine():
    if torch.cuda.is_available():
        pytest.skip("this test describes the CPU-only case")
    assert best_device(allow_mps=False).type == "cpu"
    assert count_gpus() == 0 and gpu_info() == []
    m = torch.nn.Linear(3, 2)
    assert wrap_data_parallel(m) is m and unwrap_model(m) is m
    assert unwrap_model(torch.nn.DataParallel(m)) is m
    amp = AMPContext(device="cpu")
    assert not amp.enabled
    with amp.autocast():
        assert m(torch.ones(1, 3)).dtype == torch.float32                   # no mixed precision on CPU
    assert enable_gradient_checkpointing(m) is m


def test_maybe_compile_wraps_the_same_model():
    """Without running the compiled graph (that needs a C compiler on CPU): the wrapper holds the very same model,
    or the model itself when compilation is unavailable."""
    m = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    c = maybe_compile(m)
    assert getattr(c, "_orig_mod", c) is m


def test_prefetcher_yields_every_batch_in_order_on_cpu():
    data = [torch.full((2,), float(i)) for i in range(5)]
    got = list(CUDAPrefetcher(data, device="cpu"))
    assert [float(b[0]) for b in got] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_gradient_accumulation_equals_one_large_batch():
    """Four micro-batches of 2 accumulated give the same update as one batch of 8 (mean loss, no clipping)."""
    torch.manual_seed(0)
    x, y = torch.randn(8, 3), torch.randn(8, 1)
    a, b = torch.nn.Linear(3, 1), torch.nn.Linear(3, 1)
    b.load_state_dict(a.state_dict())
    oa, ob = torch.optim.SGD(a.parameters(), lr=0.1), torch.optim.SGD(b.parameters(), lr=0.1)
    gat = GradAccumTrainer(a, oa, GradAccumConfig(accumulate_steps=4, clip_grad_norm=0.0))

    def loss_fn(model, batch):
        return torch.mean((model(batch[0]) - batch[1]) ** 2)
    outs = [gat.step((x[i:i + 2], y[i:i + 2]), loss_fn) for i in range(0, 8, 2)]
    assert outs[:3] == [None, None, None] and outs[3] is not None
    ob.zero_grad()
    loss_fn(b, (x, y)).backward()
    ob.step()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.allclose(pa, pb, atol=1e-6)


def test_parallel_sweep_runs_the_grid_and_sorts_by_the_first_metric():
    cfg = SweepConfig(param_grid={"a": [1, 2, 3], "b": [10, 20]}, n_jobs=2)
    assert len(cfg.grid_points()) == 6
    res = run_parallel_sweep(lambda p: {"loss": abs(p["a"] - 2) + p["b"] / 100}, cfg)
    assert len(res) == 6 and res[0]["params"] == {"a": 2, "b": 10}
    assert [r["loss"] for r in res] == sorted(r["loss"] for r in res)


def test_throughput_monitor_counts_samples_per_second():
    mon = ThroughputMonitor()
    mon.start_epoch()
    info = mon.end_epoch(n_samples=1000)
    assert info["epoch"] == 1 and info["samples_sec"] > 0
    s = mon.summary()
    assert s["n_epochs"] == 1 and s["avg_samples_sec"] == info["samples_sec"]
