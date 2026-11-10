import numpy as np
import pytest
from pinneapple_veriphysics.design_evaluation import (
    evaluate_arm, paired_bootstrap, summarize_paired_runs,
)


def test_arm_uses_only_frozen_selection():
    m = evaluate_arm([0, 2], [1.3, 1.9, 1.2], [1.2, .95],
                     predicted_baseline=1., reference_baseline=1., threshold=.1)
    assert m.selected == 2
    assert m.promised == 2
    assert m.confirmed == 1
    assert m.confirmation_rate == .5
    assert m.best_actual_gain == pytest.approx(.2)


def test_empty_arm_is_not_a_zero_success_rate():
    m = evaluate_arm([], [1.], [], predicted_baseline=1.,
                     reference_baseline=1., threshold=.1)
    assert m.confirmation_rate is None
    assert m.best_actual_gain is None


def test_duplicate_selection_is_rejected():
    with pytest.raises(ValueError):
        evaluate_arm([0, 0], [1.], [1., 1.], predicted_baseline=0.,
                     reference_baseline=0., threshold=.1)


def test_bootstrap_is_deterministic_and_paired():
    x = paired_bootstrap([.1, .2, .3], [.2, .3, .4], seed=10)
    y = paired_bootstrap([.1, .2, .3], [.2, .3, .4], seed=10)
    assert x == y
    assert x["mean_difference"] == pytest.approx(.1)
    assert x["interval"][0] == pytest.approx(.1)


def test_missing_seed_is_reported_not_silently_zero_filled():
    rows = [{"seed": 0, "baseline_confirmation_rate": .5,
             "screened_confirmation_rate": .6},
            {"seed": 1, "baseline_confirmation_rate": .5,
             "screened_confirmation_rate": None}]
    s = summarize_paired_runs(rows)
    assert s["paired_seeds"] == 1
    assert s["excluded_seeds"] == [1]
