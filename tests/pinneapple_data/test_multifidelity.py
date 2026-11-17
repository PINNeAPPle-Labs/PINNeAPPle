"""The multi-fidelity curve, on problems whose answer is known before it is computed.

The whole point of the curve is that it says something about a *cheap* model and an *expensive* one; so the tests
build the two fidelities from a function, and the known relation between them is what the strategies have to recover.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.multifidelity import (
    STRATEGIES,
    FidelityPair,
    KernelRidge,
    LearningCurve,
    learning_curve,
    r2,
    relative_l2,
)


def _two_fidelities(n_low=120, n_high=24, seed=0, *, offset=0.3, gain=1.5):
    """``y_high = gain * y_low + offset`` over a smooth cheap model on a unit box."""
    rng = np.random.default_rng(seed)
    design = rng.uniform(-1.0, 1.0, size=(n_low, 2))
    low = np.sin(2.0 * design[:, 0]) + np.cos(design[:, 1])
    keep = rng.choice(n_low, size=n_high, replace=False)
    return FidelityPair(design_low=design, values_low=low, design_high=design[keep],
                        values_high=gain * low[keep] + offset, low_at_high=low[keep])


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------

def test_the_scores_are_the_ones_they_say():
    measured = np.array([1.0, 2.0, 3.0])
    assert relative_l2(measured, measured) == pytest.approx(0.0)
    assert r2(measured, measured) == pytest.approx(1.0)
    assert r2(np.full(3, measured.mean()), measured) == pytest.approx(0.0)      # the mean: R^2 = 0
    assert r2(np.zeros(3), measured) < 0.0                                      # worse than the mean: negative
    assert relative_l2(measured + 0.5, measured) == pytest.approx(np.linalg.norm(np.full(3, 0.5)) / np.linalg.norm(measured))


# ---------------------------------------------------------------------------
# the pair
# ---------------------------------------------------------------------------

def test_pairing_finds_the_cheap_run_of_every_expensive_design():
    design = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    low = np.array([10.0, 20.0, 30.0, 40.0])
    pair = FidelityPair.pair(design, low, design[[2, 0]], np.array([300.0, 100.0]))
    assert pair.low_at_high.tolist() == [30.0, 10.0]
    assert pair.n_low == 4 and pair.n_high == 2 and pair.dimension == 2


def test_pairing_refuses_a_design_the_cheap_model_never_saw():
    design = np.array([[0.0, 1.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="not among the cheap ones"):
        FidelityPair.pair(design, np.array([1.0, 2.0]), np.array([[9.0, 9.0]]), np.array([1.0]))


def test_a_pair_refuses_arrays_that_do_not_line_up():
    with pytest.raises(ValueError, match="values_low has 1 rows"):
        FidelityPair(design_low=[[0.0, 1.0], [1.0, 0.0]], values_low=[1.0], design_high=[[0.0, 1.0]],
                     values_high=[1.0], low_at_high=[1.0])
    with pytest.raises(ValueError, match="same d"):
        FidelityPair(design_low=[[0.0, 1.0]], values_low=[1.0], design_high=[[0.0, 1.0, 2.0]],
                     values_high=[1.0], low_at_high=[1.0])


def test_the_summary_states_how_far_the_cheap_model_is():
    pair = _two_fidelities()
    assert "off the expensive one" in pair.summary()


# ---------------------------------------------------------------------------
# the regressor
# ---------------------------------------------------------------------------

def test_kernel_ridge_reproduces_a_smooth_function_from_a_few_points():
    design = np.linspace(-1.0, 1.0, 12)[:, None]
    target = np.sin(3.0 * design[:, 0])
    model = KernelRidge().fit(design, target)
    assert relative_l2(model.predict(design), target) < 1e-3          # the training points come back
    middle = np.linspace(-0.9, 0.9, 7)[:, None]
    assert relative_l2(model.predict(middle), np.sin(3.0 * middle[:, 0])) < 1e-2


def test_kernel_ridge_refuses_one_point_and_predicting_before_fitting():
    with pytest.raises(ValueError, match="one design is not enough"):
        KernelRidge().fit([[0.0]], [1.0])
    with pytest.raises(RuntimeError, match="predict before fit"):
        KernelRidge().predict([[0.0]])


# ---------------------------------------------------------------------------
# the curve
# ---------------------------------------------------------------------------

def test_the_difference_strategy_recovers_an_affine_relation_the_cheap_model_misses():
    """``y_high = 1.5 y_low + 0.3``: the cheap model alone is 1.5x off, and the correction is learnable from two
    designs. Nothing here needs 64 expensive runs to be right, which is the claim the curve makes."""
    pair = _two_fidelities()
    curve = learning_curve(pair, budgets=(2, 4, 8, 24), repeats=3)
    cheap_error = float(np.median(curve.scores["low_fidelity"]["relative_l2"][0]))
    corrected = float(np.median(curve.scores["difference"]["relative_l2"][0]))
    assert cheap_error > 0.1                       # the cheap model is far off the expensive one
    assert corrected < cheap_error                 # and the correction already helps at two designs
    assert float(np.median(curve.scores["difference"]["relative_l2"][-1])) < 0.02
    geometry = float(np.median(curve.scores["geometry"]["relative_l2"][-1]))
    assert geometry > float(np.median(curve.scores["difference"]["relative_l2"][-1]))


def test_the_cheap_model_row_is_the_cheap_model_with_no_fit_at_all():
    pair = _two_fidelities()
    curve = learning_curve(pair, budgets=(24,), repeats=1)
    assert float(curve.scores["low_fidelity"]["relative_l2"][0, 0]) == pytest.approx(
        relative_l2(pair.low_at_high, pair.values_high))


def test_the_full_budget_is_leave_one_out_and_has_no_spread_over_repeats():
    pair = _two_fidelities(n_high=12)
    curve = learning_curve(pair, budgets=(12,), repeats=4)
    for strategy in STRATEGIES:
        values = curve.scores[strategy]["relative_l2"][0]
        assert np.allclose(values, values[0])              # the same leave-one-out, four times


def test_every_strategy_is_scored_on_the_same_held_out_designs():
    pair = _two_fidelities()
    curve = learning_curve(pair, budgets=(6,), repeats=2)
    assert set(curve.scores) == set(STRATEGIES)
    for strategy in STRATEGIES:
        assert curve.scores[strategy]["relative_l2"].shape == (1, 2)


def test_the_residual_feature_strategy_beats_the_design_only_one():
    pair = _two_fidelities(n_high=20)
    curve = learning_curve(pair, budgets=(20,), repeats=1)
    assert (float(np.median(curve.scores["residual_feature"]["relative_l2"]))
            < float(np.median(curve.scores["geometry"]["relative_l2"])))


def test_a_budget_the_curve_cannot_score_is_refused():
    pair = _two_fidelities(n_high=6)
    for bad in (1, 7, 0, -3):
        with pytest.raises(ValueError, match="outside"):
            learning_curve(pair, budgets=(bad,))
    with pytest.raises(ValueError, match="repeats"):
        learning_curve(pair, budgets=(4,), repeats=0)


def test_the_table_names_every_strategy_and_budget():
    pair = _two_fidelities()
    table = learning_curve(pair, budgets=(4, 8), repeats=2).table()
    lines = table.splitlines()
    assert len(lines) == 2 + 2
    for strategy in STRATEGIES:
        assert strategy in lines[0]
    assert lines[2].startswith("   4") and lines[3].startswith("   8")
    assert "%" in lines[2] and "%" in lines[3]


def test_a_curve_carries_the_spread_of_its_repeats():
    curve = LearningCurve(budgets=(4,), repeats=3,
                          scores={"low_fidelity": {"relative_l2": np.array([[1.0, 2.0, 3.0]]), "r2": np.ones((1, 3))}})
    assert curve.median("low_fidelity")[0] == pytest.approx(2.0)
    assert curve.spread("low_fidelity")[0] == pytest.approx(2.0)
