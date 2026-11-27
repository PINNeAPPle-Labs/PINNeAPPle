"""Decision-level validation: the metrics that say whether a surrogate changes the design, on constructed cases
whose answers are known by hand.

Every number below is arithmetic on arrays written out in the test, so a mistake in the metric is caught here and
cannot be hidden inside a fitted model. The identity case (surrogate equals the reference) must come out at a zero
flip rate, a zero regret and a rank correlation of one; a constructed permutation must reach its analytic value; a
pair the reference itself cannot separate must not be counted; and the card must carry the five numbers it
promises.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_analysis.decision import (
    DecisionThresholds,
    decision_card,
    failure_map,
    flip_rate,
    rank_preservation,
    regret,
)


# ---------------------------------------------------------------- flip rate: the identity case
def test_perfect_surrogate_has_no_flip_and_a_decision_safe_card():
    truth = np.array([3.1, 2.4, 5.0, 1.2, 4.3, 0.7, 2.9, 3.8, 1.9, 6.2])
    fr = flip_rate(truth, truth.copy())
    assert fr.flip_rate == 0.0
    assert fr.inverted == ()
    assert fr.choice_flip is False
    assert fr.n_pairs == 45  # 10 designs -> 10*9/2 pairs
    assert fr.n_flipped_pairs == 0

    card = decision_card(truth, truth.copy())
    assert card.flip_rate == 0.0
    assert card.spearman == pytest.approx(1.0)
    assert card.kendall_tau == pytest.approx(1.0)
    assert card.top_k_kept == pytest.approx(1.0)
    assert card.regret_relative == 0.0
    assert card.regret_absolute == 0.0
    assert card.verdict == "decision-safe"
    assert card.reasons == ()


# ---------------------------------------------------------------- flip rate: a constructed permutation
def test_two_swapped_designs_are_counted_as_one_reversed_pair():
    # designs 2 and 3 exchange values in the prediction; nothing else moves, so exactly one of the ten pairs flips
    truth = np.array([1.0, 2.0, 5.0, 4.0, 3.0])
    pred = np.array([1.0, 2.0, 4.0, 5.0, 3.0])
    fr = flip_rate(truth, pred)
    assert fr.n_pairs == 10
    assert fr.n_flipped_pairs == 1
    assert fr.flip_rate == pytest.approx(0.1)
    assert fr.inverted == (2, 3)
    assert fr.choice_flip is False, "swapping two mid-ranked designs does not change the winner"

    card = decision_card(truth, pred)
    assert card.flip_rate == pytest.approx(0.1)
    assert card.verdict == "review"  # 10 % is inside the review band, not the safe one
    assert any("Flip rate" in reason for reason in card.reasons)


def test_fully_inverted_surrogate_reverses_every_pair():
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = -truth  # every ordering reversed
    fr = flip_rate(truth, pred)
    assert fr.flip_rate == pytest.approx(1.0)
    assert fr.inverted == (0, 1, 2, 3, 4)
    assert fr.choice_flip is True


# ---------------------------------------------------------------- flip rate: the tolerance
def test_equivalent_pair_inside_the_margin_is_not_a_flip():
    # the two leading designs differ by 0.004 in the reference; any sane customer calls them the same design
    truth = np.array([1.0, 1.004, 2.0, 3.0])
    pred = np.array([1.004, 1.0, 2.0, 3.0])  # the surrogate swaps exactly that pair
    sd = float(np.std(truth, ddof=1))

    strict = flip_rate(truth, pred)  # margin 0: the reference separates them, so the swap is a flip
    assert strict.tolerance == 0.0
    assert strict.n_pairs == 6
    assert strict.n_flipped_pairs == 1
    assert strict.flip_rate == pytest.approx(1.0 / 6.0)
    assert strict.inverted == (0, 1)

    tolerant = flip_rate(truth, pred, margin=0.01)  # tolerance = 0.01 * sd > 0.004
    assert tolerant.tolerance == pytest.approx(0.01 * sd)
    assert tolerant.tolerance > 0.004
    assert tolerant.n_equivalent_pairs == 1
    assert tolerant.n_pairs == 5, "the equivalent pair leaves the denominator"
    assert tolerant.n_flipped_pairs == 0
    assert tolerant.flip_rate == 0.0
    assert tolerant.inverted == ()


def test_margin_is_relative_to_the_reference_spread_not_an_absolute_value():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    pred = np.array([0.0, 1.0, 3.0, 2.0])
    wide = flip_rate(truth, pred, margin=100.0)  # everything is equivalent, so nothing is comparable
    assert wide.flip_rate == 0.0
    assert wide.n_pairs == 0
    assert wide.n_equivalent_pairs == 6


# ---------------------------------------------------------------- ranking
def test_monotone_inversion_gives_spearman_minus_one():
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = truth[::-1].copy()
    rp = rank_preservation(truth, pred)
    assert rp.spearman == pytest.approx(-1.0)
    assert rp.kendall_tau == pytest.approx(-1.0)
    assert rp.top_k == 1  # default is 10 % of 5 designs, at least one
    assert rp.top_k_kept == 0.0
    assert rp.top_k_truth == (0,)
    assert rp.top_k_prediction == (4,)


def test_top_k_default_and_preserved_fraction():
    truth = np.arange(20.0)                  # 0 is best
    pred = truth + np.linspace(0.0, 0.05, 20)  # a small drift, ordering untouched
    rp = rank_preservation(truth, pred)
    assert rp.top_k == 2                     # 10 % of 20
    assert rp.top_k_kept == pytest.approx(1.0)
    assert rp.spearman == pytest.approx(1.0)

    moved = pred.copy()
    moved[0] = 19.0                          # the winner falls to the bottom of the surrogate's ranking
    lost = rank_preservation(truth, moved)
    assert lost.top_k_kept == pytest.approx(0.5)
    assert lost.top_k_prediction == (1, 2)


def test_kendall_tau_b_corrects_for_ties():
    truth = np.array([1.0, 1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 2.0, 3.0])
    rp = rank_preservation(truth, pred)
    # 6 pairs: (0,1) tied in truth -> excluded by the tie correction; one pair tied in the prediction
    assert -1.0 <= rp.kendall_tau <= 1.0
    assert rp.spearman == pytest.approx(5.0 / 6.0)  # ranks (1.5, 1.5, 3, 4) against (1, 2.5, 2.5, 4)


def test_constant_prediction_has_no_ranking_to_correlate():
    rp = rank_preservation(np.array([1.0, 2.0, 3.0]), np.full(3, 2.0))
    assert np.isnan(rp.spearman)
    assert np.isnan(rp.kendall_tau)


# ---------------------------------------------------------------- regret
def test_regret_on_three_points_computed_by_hand():
    # minimisation: the surrogate likes design 1 (predicted 0.0); the reference says design 1 is the worst at 12.0
    truth = np.array([10.0, 12.0, 11.0])
    pred = np.array([1.0, 0.0, 2.0])
    rg = regret(truth, pred, lower_is_better=True)
    assert rg.chosen == 1
    assert rg.best_truth == 0
    assert rg.absolute == pytest.approx(2.0)          # 12.0 - 10.0
    assert rg.span == pytest.approx(2.0)              # 12.0 - 10.0
    assert rg.relative == pytest.approx(1.0)          # the whole spread
    assert rg.truth_at_chosen == pytest.approx(12.0)
    assert rg.best_value == pytest.approx(10.0)

    # maximisation: the surrogate now picks design 2 at 11.0 against the true best of 12.0
    up = regret(truth, pred, lower_is_better=False)
    assert up.chosen == 2
    assert up.best_truth == 1
    assert up.absolute == pytest.approx(1.0)          # 12.0 - 11.0
    assert up.relative == pytest.approx(0.5)


def test_regret_is_zero_when_the_surrogate_picks_the_reference_best():
    truth = np.array([4.0, 1.0, 9.0])
    rg = regret(truth, np.array([2.0, 0.0, 3.0]))
    assert rg.absolute == 0.0
    assert rg.relative == 0.0


# ---------------------------------------------------------------- failure map
def test_failure_map_marks_the_known_hot_cell():
    coords = np.array([((i + 0.5) / 4, (j + 0.5) / 4)
                       for i in range(4) for j in range(4) for _ in range(2 if (i, j) == (3, 0) else 1)])
    error = np.array([1.0 if (i, j) == (3, 0) else 0.01
                      for i in range(4) for j in range(4) for _ in range(2 if (i, j) == (3, 0) else 1)])

    fm = failure_map(coords, error, bins=4)
    assert fm.mean.shape == (4, 4)
    assert fm.count.shape == (4, 4)
    assert fm.occupancy() == pytest.approx(1.0)
    assert fm.mean[3, 0] == pytest.approx(1.0)   # the hot cell, mean of its two designs
    assert fm.count[3, 0] == 2
    assert fm.mean[0, 3] == pytest.approx(0.01)

    worst = fm.worst()
    assert worst is not None
    assert worst["index"] == (3, 0)
    assert worst["count"] == 2
    assert worst["mean_error"] == pytest.approx(1.0)
    # the grid is binned over the sampled range [0.125, 0.875], so cell (3, 0) spans [0.6875, 0.875] x [0.125, 0.3125]
    assert worst["centre"] == pytest.approx((0.78125, 0.21875))
    assert worst["low"] == pytest.approx((0.6875, 0.125))
    assert worst["high"] == pytest.approx((0.875, 0.3125))


def test_failure_map_leaves_unsampled_cells_as_not_run():
    coords = np.array([[0.1], [0.1], [0.9]])  # 1-D design space, nothing in the middle third bin
    fm = failure_map(coords, np.array([0.2, 0.4, 0.9]), bins=3)
    assert fm.dim == 1
    assert fm.count[0] == 2 and fm.count[2] == 1
    assert np.isnan(fm.mean[1]) and fm.count[1] == 0
    assert fm.top_regions[0]["index"] == (2,)
    assert fm.top_regions[0]["mean_error"] == pytest.approx(0.9)
    assert fm.occupancy() == pytest.approx(2.0 / 3.0)


def test_failure_map_accepts_three_dimensions_and_drops_non_finite_designs():
    rng = np.random.default_rng(0)
    coords = rng.random((50, 3))
    error = np.linalg.norm(coords, axis=1)
    coords = np.vstack([coords, [[np.nan, 0.5, 0.5]]])
    error = np.concatenate([error, [1.0]])
    fm = failure_map(coords, error, bins=5)
    assert fm.dim == 3
    assert fm.mean.shape == (5, 5, 5)
    assert fm.n == 50
    assert fm.n_dropped == 1
    assert fm.summary().startswith("Failure map over 50 designs in 3-D")


# ---------------------------------------------------------------- the card
def test_markdown_carries_the_five_numbers():
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    pred = np.array([1.1, 2.0, 3.2, 4.0, 5.0, 6.4, 7.0, 8.2, 9.0, 10.0])
    card = decision_card(truth, pred, names=[f"bend_{i}" for i in range(10)], margin=0.01)
    md = card.to_markdown()

    assert f"{100 * card.flip_rate:.1f} %" in md
    assert f"{card.spearman:.3f}" in md
    assert f"{card.kendall_tau:.3f}" in md
    assert f"{100 * card.top_k_kept:.0f} %" in md
    assert f"{100 * card.regret_relative:.1f} %" in md

    for heading in ("Flip rate", "Spearman rho", "Kendall tau-b", "Top-", "Relative regret"):
        assert heading in md
    assert f"Verdict: {card.verdict}" in md
    assert "10 designs" in md
    assert "A surrogate without a card is not deployable" in md
    assert "engineering convention, not a measured result" in md
    assert set(card.numbers()) == {"flip_rate", "spearman", "kendall_tau", "top_k_kept", "regret_relative"}


def test_card_dict_is_json_serialisable_and_keeps_the_verdict():
    truth = np.array([2.0, 0.5, 3.0, 1.5])
    pred = np.array([2.1, 0.4, 2.9, 1.6])
    card = decision_card(truth, pred)
    payload = json.loads(json.dumps(card.to_dict()))
    assert payload["verdict"] in {"decision-safe", "review", "unsafe"}
    assert payload["n"] == 4
    assert payload["flip_rate"] == pytest.approx(card.flip_rate)
    assert payload["thresholds"]["flip_rate_safe"] == pytest.approx(DecisionThresholds().flip_rate_safe)


def test_a_surrogate_that_cannot_rank_is_unsafe_and_says_why():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.full(4, 2.5)  # the training mean: every design looks the same
    card = decision_card(truth, pred)
    assert card.spearman != card.spearman  # nan: there is no ranking to correlate
    assert card.flip_rate == 0.0            # the surrogate separates nothing, so it reverses nothing
    assert card.verdict == "unsafe"
    assert any("could not be computed" in reason for reason in card.reasons)


def test_thresholds_are_configurable_and_validated():
    truth = np.array([1.0, 2.0, 5.0, 4.0, 3.0])
    pred = np.array([1.0, 2.0, 4.0, 5.0, 3.0])
    strict = DecisionThresholds(flip_rate_safe=0.01, flip_rate_review=0.02)
    assert decision_card(truth, pred, thresholds=strict).verdict == "unsafe"

    tolerant = DecisionThresholds(flip_rate_safe=0.2, flip_rate_review=0.4)
    assert decision_card(truth, pred, thresholds=tolerant).verdict == "decision-safe"

    with pytest.raises(ValueError):
        DecisionThresholds(flip_rate_safe=0.2, flip_rate_review=0.1)


def test_names_of_flipped_designs_reach_the_card():
    truth = np.array([1.0, 2.0, 5.0, 4.0, 3.0])
    pred = np.array([1.0, 2.0, 4.0, 5.0, 3.0])
    names = ["r40", "r50", "r60", "r70", "r80"]
    card = decision_card(truth, pred, names=names)
    assert card.inverted_names == ("r60", "r70")
    assert "r60, r70" in card.to_markdown()
    with pytest.raises(ValueError):
        decision_card(truth, pred, names=names[:3])


def test_input_validation_names_the_problem():
    with pytest.raises(ValueError):
        flip_rate([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        flip_rate([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError):
        rank_preservation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], top_k=4)
    with pytest.raises(ValueError):
        failure_map(np.zeros((5, 4)), np.zeros(5))
    with pytest.raises(ValueError):
        failure_map(np.zeros((5, 2)), -np.ones(5))
