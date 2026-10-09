"""pp.metrics: the pooled leaderboard metrics and their use by the benchmark suite (#32)."""
import numpy as np
import pytest

from pinneapple_physics import metrics


def test_pooled_matches_the_benchmark_definition():
    import torch
    rng = np.random.default_rng(1)
    t = rng.normal(size=(50, 2))
    p = t + 0.1 * rng.normal(size=(50, 2))
    m = metrics.pooled(torch.tensor(p), t)
    d = p - t
    assert m["relative_l2"] == pytest.approx(np.sqrt((d ** 2).sum() / (t ** 2).sum()))
    assert m["max_abs"] == pytest.approx(np.abs(d).max()) and m["mse"] == pytest.approx((d ** 2).mean())
    assert metrics.pooled(t[:, 0], t[:, :1])["relative_l2"] == 0.0          # (N,) against (N, 1)
    assert np.isnan(metrics.pooled([1.0, 2.0], [0.0, 0.0])["relative_l2"])   # undefined, never a huge number


def test_benchmark_suite_uses_pp_metrics():
    import inspect
    from pinneapple_tools.benchmark_suite import api, benchmark
    from pinneapple_tools.benchmark_suite.tasks import base
    for mod in (benchmark, api, base):
        src = inspect.getsource(mod)
        assert "pooled(" in src and "+ 1e-10)).sqrt()" not in src, mod.__name__
