"""The neutral evaluator and status rule of pinneapple_analysis.reproduction, against hand-computed values."""
import math

import numpy as np
import pytest

from pinneapple_analysis import reproduction as R


def test_metrics_by_hand():
    ref = np.array([1.0, -2.0, 2.0])
    pred = np.array([1.0, -1.0, 2.0])
    assert R.rel_l2(pred, ref) == pytest.approx(1.0 / 3.0)
    assert R.rmae(pred, ref) == pytest.approx(1.0 / 5.0)
    assert R.max_abs(pred, ref) == 1.0
    assert R.mean_abs(pred, ref) == pytest.approx(1.0 / 3.0)
    assert R.mse(pred, ref) == pytest.approx(1.0 / 3.0)
    assert R.rmse(pred, ref) == pytest.approx(math.sqrt(1.0 / 3.0))
    assert R.evaluate(pred, ref, ["rrmse"])["rrmse"] == R.rel_l2(pred, ref)


def test_pinnsformer_definitions_match_official_formula():
    rng = np.random.default_rng(0)
    u, p = rng.normal(size=(11, 7)), rng.normal(size=(11, 7))
    assert R.rmae(p, u) == pytest.approx(np.sum(np.abs(u - p)) / np.sum(np.abs(u)))
    assert R.rel_l2(p, u) == pytest.approx(np.sqrt(np.sum((u - p) ** 2) / np.sum(u ** 2)))


def test_diverged_prediction_and_shapes():
    assert R.rel_l2([np.nan, 1.0], [1.0, 1.0]) == math.inf
    with pytest.raises(ValueError):
        R.rel_l2([1.0], [1.0, 2.0])


def test_seed_stats_keeps_failures():
    s = R.seed_stats([1e-3, float("nan"), 2e-3])
    assert s.n_failed == 1 and s.median == 2e-3 and s.max == math.inf and len(s.values) == 3


def test_status_rule():
    assert R.status(1.5e-3, 1e-3) == "reproduced"
    assert R.status(4e-4, 1e-3) == "not reproduced"            # 2.5x better is outside the factor-2 band too
    assert R.status(3e-3, 1e-3, ordering_holds=True) == "partially reproduced"
    assert R.status(3e-3, 1e-3, spread=(1e-3, 4e-3)) == "reproduced"
    assert R.status(5e-8, 1e-8) == "reproduced"                # below 1e-6: same order of magnitude
    assert R.status(5e-6, 1e-8) == "not reproduced"
    assert R.status(1e-3, 1e-3, reduced_scale=True) == "reduced-scale"
    assert R.status(math.inf, 1e-3, ordering_holds=False) == "not reproduced"
