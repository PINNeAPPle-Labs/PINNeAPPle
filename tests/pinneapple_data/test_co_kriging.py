"""Co-kriging on problems whose two fidelities are known before anything is fitted.

The claim the module exists to support -- two levels beat the best single level at equal expensive budget -- is
measured here on the Forrester pair and on a 2-D pair, and so is the case where it does not hold: a cheap model that
knows nothing about the expensive one. Both directions are asserted, because a study that only reports the wins is a
brochure.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.co_kriging import (
    MODELS,
    CoKriging,
    GaussianProcess,
    _nearest,
    forrester_high,
    forrester_low,
    study,
)
from pinneapple_data.multifidelity import relative_l2


def _two_d():
    """A cheap model that is a scaled copy of the expensive one plus nothing: the structure the budget cannot afford
    is already in the cheap model, which is the case co-kriging is for."""
    def high(x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return np.sin(4.0 * x[:, 0]) * np.cos(3.0 * x[:, 1]) + 0.15 * x[:, 0]

    def low(x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return 0.85 * np.sin(4.0 * x[:, 0]) * np.cos(3.0 * x[:, 1])

    return low, high


def _uninformative(x):
    """A cheap model with structure of its own and no relation to the expensive one."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.sin(7.0 * x) * 3.0 + 1.0


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------

def test_the_forrester_pair_carries_the_relation_it_is_known_for():
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(forrester_low(x), 0.5 * forrester_high(x) + 10.0 * (x - 0.5) - 5.0)
    assert np.allclose(forrester_high(x), (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0))


def test_the_gaussian_process_learns_a_smooth_function_and_its_interval_is_calibrated():
    """The 1-sigma and 2-sigma coverage of a GP fitted on 20 designs of a function it can see -- measured, and it is
    not exactly 68/95: the hyperparameters come from the same 20 points."""
    rng = np.random.default_rng(5)
    test = rng.uniform(0.0, 1.0, (200, 1))
    truth = forrester_high(test)
    one = []
    two = []
    errors = []
    for _ in range(20):
        x = rng.uniform(0.0, 1.0, (20, 1))
        mean, std = GaussianProcess().fit(x, forrester_high(x)).predict(test, return_std=True)
        one.append(np.mean(np.abs(mean - truth) <= std))
        two.append(np.mean(np.abs(mean - truth) <= 2.0 * std))
        errors.append(relative_l2(mean, truth))
    assert 0.6 <= np.mean(one) <= 0.95 and 0.85 <= np.mean(two) <= 1.0
    assert np.median(errors) < 0.02


def test_the_fit_is_the_same_every_run():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, (15, 1))
    y = forrester_high(x)
    first, second = GaussianProcess().fit(x, y), GaussianProcess().fit(x, y)
    assert (first.length_scale, first.signal, first.noise) == (second.length_scale, second.signal, second.noise)
    assert np.array_equal(first.predict(x), second.predict(x))


def test_a_process_that_never_saw_a_point_is_refused():
    with pytest.raises(ValueError, match="needs at least two"):
        GaussianProcess().fit([[0.0]], [1.0])
    with pytest.raises(RuntimeError, match="predict before fit"):
        GaussianProcess().predict([[0.0]])


def test_the_scale_between_the_levels_is_recovered():
    """``high = 1.5 * low + 3``: the offset goes to the discrepancy and ``rho`` comes back as 1.5."""
    rng = np.random.default_rng(1)
    design = rng.uniform(0.0, 1.0, (30, 1))
    cheap = forrester_low(design)
    cr = CoKriging.fit(design, cheap, design[:8], 1.5 * cheap[:8] + 3.0)
    assert cr.rho == pytest.approx(1.5, rel=0.02)
    test = rng.uniform(0.0, 1.0, (100, 1))
    prediction = cr.predict(test, low_values=forrester_low(test), return_std=False)
    assert relative_l2(prediction, 1.5 * forrester_low(test) + 3.0) < 0.01


# ---------------------------------------------------------------------------
# the comparison the issue asks for: two levels against one, same budget
# ---------------------------------------------------------------------------

def test_two_levels_beat_the_best_single_level_at_the_same_expensive_budget():
    """Forrester, 60 cheap runs, the expensive budget swept from 5 to 20, five repeats each.

    The single-fidelity GP is the best single level by construction (the cheap model alone is 176 % off), so it is
    the one to beat -- and the co-kriging model beats it at every budget, by the factors printed in the table."""
    result = study(forrester_low, forrester_high, n_low=60, budgets=(5, 10, 20), repeats=5, seed=0)
    assert tuple(result.budgets) == (5, 10, 20) and result.repeats == 5
    factors = result.better_by()
    assert np.all(factors > 1.5), factors                      # 2.97x at N=5, 428x at N=10, 44x at N=20
    for i, budget in enumerate(result.budgets):
        assert np.median(result.scores["co_kriging"][i]) < np.median(result.scores["single_fidelity"][i]) / 1.5, budget
    assert result.beats_single_fidelity(minimum_factor=1.5)
    # the scale the fit recovers is the one the pair was built with: cheap = high / 2 + drift, so rho = 2
    assert np.median(result.rho[1:]) == pytest.approx(2.0, abs=0.05)
    # the single-fidelity GP is learning what the budget buys -- and the cheap model alone is over 100 % off
    assert np.median(result.scores["single_fidelity"][-1]) < np.median(result.scores["single_fidelity"][0])
    assert np.median(result.scores["low_fidelity"]) > 1.0


def test_the_table_names_the_three_models_and_their_errors():
    result = study(forrester_low, forrester_high, n_low=40, budgets=(6, 12), repeats=3, seed=11)
    table = result.table()
    assert all(model in table for model in MODELS)
    assert table.splitlines()[0].split()[0] == "N" and len(table.splitlines()) == 4
    assert set(result.scores) == set(MODELS)
    # the cheap model alone does not use the expensive runs, so its row is the same at every budget
    assert np.ptp(result.scores["low_fidelity"][:, 0]) == 0.0
    assert np.ptp(result.scores["low_fidelity"][:, 1]) == 0.0


def test_the_study_is_deterministic_and_respects_its_seed():
    first = study(forrester_low, forrester_high, n_low=30, budgets=(6,), repeats=2, seed=4)
    second = study(forrester_low, forrester_high, n_low=30, budgets=(6,), repeats=2, seed=4)
    third = study(forrester_low, forrester_high, n_low=30, budgets=(6,), repeats=2, seed=5)
    assert np.array_equal(first.scores["co_kriging"], second.scores["co_kriging"])
    assert not np.array_equal(first.scores["co_kriging"], third.scores["co_kriging"])


def test_a_second_input_dimension_behaves_the_same_way():
    low, high = _two_d()
    result = study(low, high, n_low=80, budgets=(8, 16), repeats=5, seed=3, bounds=((0.0, 1.0), (0.0, 1.0)))
    assert result.beats_single_fidelity(minimum_factor=2.0)
    assert np.median(result.rho) == pytest.approx(1.0 / 0.85, rel=0.05)     # cheap = 0.85 * high
    assert np.all(result.median("single_fidelity") > 10.0 * result.median("co_kriging"))


def test_a_cheap_model_that_knows_nothing_buys_nothing_and_the_study_says_so():
    """The honest half: an uninformative cheap model is not turned into an improvement by the machinery.

    Same budget, same seeds, same function: the factors ``better_by`` reports are around one, and
    ``beats_single_fidelity`` is false -- which is the answer a study is supposed to be able to give."""
    result = study(_uninformative, forrester_high, n_low=60, budgets=(5, 10, 20), repeats=5, seed=0)
    assert not result.beats_single_fidelity()
    assert np.all(result.better_by() < 1.1) and np.all(result.better_by() > 0.85)
    assert np.median(result.scores["co_kriging"][-1]) > 0.5 * np.median(result.scores["single_fidelity"][-1])


# ---------------------------------------------------------------------------
# the two ways of knowing the cheap model
# ---------------------------------------------------------------------------

def test_the_cheap_model_as_a_process_answers_what_its_exact_values_answer():
    """With the cheap values at the prediction designs the model uses them; without, the low level's posterior mean
    stands in. On Forrester the two agree to well under a hundredth of the target's norm."""
    rng = np.random.default_rng(7)
    test = rng.uniform(0.0, 1.0, (200, 1))
    design = rng.uniform(0.0, 1.0, (40, 1))
    cr = CoKriging.fit(design, forrester_low(design), design[:10], forrester_high(design[:10]),
                       low_at_high=forrester_low(design[:10]))
    exact = cr.predict(test, low_values=forrester_low(test), return_std=False)
    through_process = cr.predict(test, return_std=False)
    truth = forrester_high(test)
    assert relative_l2(exact, truth) < 0.05
    assert relative_l2(exact, through_process) < 0.01


def test_the_interval_is_approximate_and_the_module_says_by_how_much():
    """The variance adds the two levels without their cross-covariance. Measured over ten Forrester designs with ten
    expensive runs: 1-sigma coverage 0.72, 2-sigma 0.88, and one design where the 2-sigma band held 26 % of the
    points -- optimistic, and not uniformly so. The numbers are asserted here so the docstring cannot drift from
    them; the mean is what the module claims, not the band."""
    rng = np.random.default_rng(5)
    test = rng.uniform(0.0, 1.0, (200, 1))
    truth = forrester_high(test)
    one, two = [], []
    for _ in range(10):
        design = rng.uniform(0.0, 1.0, (30, 1))
        cr = CoKriging.fit(design, forrester_low(design), design[:10], forrester_high(design[:10]))
        mean, std = cr.predict(test, low_values=forrester_low(test))
        one.append(float(np.mean(np.abs(mean - truth) <= std)))
        two.append(float(np.mean(np.abs(mean - truth) <= 2.0 * std)))
    assert np.mean(one) == pytest.approx(0.72, abs=0.02)
    assert np.mean(two) == pytest.approx(0.88, abs=0.02)
    assert min(two) < 0.5                                       # the band is not a guarantee, and says so
    assert np.median(one) > 0.5                                 # ... but it is not empty either


def test_a_cheap_model_with_no_spread_cannot_scale_anything():
    design = np.linspace(0.0, 1.0, 10)[:, None]
    with pytest.raises(ValueError, match="no spread"):
        CoKriging.fit(design, np.full(10, 3.0), design[:5], design[:5, 0] ** 2)


def test_the_tables_have_to_agree():
    design = np.linspace(0.0, 1.0, 10)[:, None]
    with pytest.raises(ValueError, match="different input counts"):
        CoKriging.fit(design, np.ones(10), np.column_stack([design, design]), np.ones(10))
    with pytest.raises(ValueError, match="disagree on the number of runs"):
        CoKriging.fit(design, np.ones(9), design[:5], np.ones(5))
    with pytest.raises(ValueError, match="expensive run"):
        CoKriging.fit(design, np.ones(10), design[:1], np.ones(1))
    with pytest.raises(ValueError, match="cheap values against"):
        CoKriging.fit(design, np.ones(10), design[:5], np.ones(5), low_at_high=np.ones(3))


def test_the_study_refuses_a_budget_it_cannot_fit():
    with pytest.raises(ValueError, match="at least two expensive"):
        study(forrester_low, forrester_high, budgets=(1,))
    with pytest.raises(ValueError, match="campaign"):
        study(forrester_low, forrester_high, budgets=(500,))


def test_the_cheap_baseline_is_the_nearest_cheap_run():
    design = np.array([[0.0], [1.0], [10.0]])
    values = np.array([1.0, 2.0, 3.0])
    assert _nearest(design, values, np.array([[0.2], [9.0]])).tolist() == [1.0, 3.0]


def test_a_model_of_the_old_shape_is_not_a_lowering():
    """``summary`` of a fitted model names the scale and the discrepancy, so a log line a study prints is checkable."""
    design = np.linspace(0.0, 1.0, 12)[:, None]
    cr = CoKriging.fit(design, forrester_low(design), design[:6], forrester_high(design[:6]))
    text = cr.summary()
    assert "co-kriging: rho" in text and "12 cheap run(s)" in text and "6 expensive one(s)" in text
