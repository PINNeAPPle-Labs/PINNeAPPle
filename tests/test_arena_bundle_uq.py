"""Uncertainty and design search over a surrogate (INT-010).

The design half is deterministic and is tested exactly. The ensemble half trains three small FNOs on a tiny bundle:
what it pins is the *contract* -- one score per seed, a spread normalised by the data, and the wording that says the
spread is a disagreement between seeds and not a posterior.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_arena.bundle_bench import BenchConfig
from pinneapple_arena.bundle_uq import EnsembleReport, design_optimize, seed_ensemble, spearman
from pinneapple_data.schema import AxisSpec, Field, Physics, PhysicsDataBundle

NT, NX = 8, 24


def wave_bundle() -> PhysicsDataBundle:
    x = np.linspace(0.0, 1.0, NX)
    t = np.linspace(0.0, 1.0, NT)
    u = np.array([np.sin(2 * np.pi * (x - 0.2 * ti)) for ti in t], dtype=np.float32)
    return PhysicsDataBundle.from_arrays(
        {"axis:t": t, "axis:x": x, "field:u": u}, fields=[Field.from_array("u", u, axes=("t", "x"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")], physics=Physics(), source="uq test",
        license="CC0-1.0", kind="synthetic")


# ── the ensemble ───────────────────────────────────────────────────────────────────────────────────────────────
def test_three_seeds_report_a_score_each_and_a_normalised_spread():
    report = seed_ensemble(wave_bundle(), BenchConfig(field="u", epochs=2, width=8, modes=4, layers=2, seed=0),
                           seeds=(0, 1))
    assert report.seeds == (0, 1) and len(report.rel_l2) == 2
    assert all(np.isfinite(score) for score in report.rel_l2)
    assert 0.0 <= report.spread < 1.0, "the spread is relative to the signal's own standard deviation"
    assert report.predictions.shape[0] == 2
    assert "not a posterior" in report.note


def test_one_seed_is_not_an_ensemble():
    with pytest.raises(ValueError, match="at least two seeds"):
        seed_ensemble(wave_bundle(), BenchConfig(field="u", epochs=1), seeds=(0,))


def test_the_report_serialises_without_the_prediction_cube():
    report = EnsembleReport(model="fno", field="u", seeds=(0, 1), rel_l2=[0.1, 0.2], spread=0.05,
                            predictions=np.zeros((2, 3, 4)))
    data = report.to_dict()
    assert "predictions" not in data
    assert data["mean"] == pytest.approx(0.15) and data["std"] > 0
    assert data["coefficient_of_variation"] == pytest.approx(report.std / report.mean)
    assert "rel-L2 mean" in report.summary()


def test_a_single_score_has_no_standard_deviation():
    assert EnsembleReport(seeds=(0,), rel_l2=[0.1]).std == 0.0


# ── the design search ─────────────────────────────────────────────────────────────────────────────────────────
def test_the_search_finds_the_minimum_of_a_quadratic_in_the_box():
    result = design_optimize(lambda d: (d["thickness"] - 0.004) ** 2, {"thickness": (0.001, 0.01)}, n=128, seed=0)
    assert 0.001 <= result.best["thickness"] <= 0.01
    assert abs(result.best["thickness"] - 0.004) < 5e-4
    assert len(result.evaluations) == 128
    # the optimum sits near the low end of the box, so most designs are on the rising branch and the ranks are
    # mostly monotone: a positive correlation is the right answer here, not zero
    assert result.sensitivity["thickness"] > 0.5
    assert "not a global analysis" in result.note


def test_maximize_flips_the_direction():
    objective = lambda d: -((d["k"] - 0.2) ** 2)                            # noqa: E731
    low = design_optimize(objective, {"k": (0.0, 1.0)}, n=64, seed=1)
    high = design_optimize(objective, {"k": (0.0, 1.0)}, n=64, seed=1, maximize=True)
    assert abs(high.best["k"] - 0.2) < 0.05
    assert high.value > low.value


def test_an_input_the_objective_ignores_has_no_rank_correlation():
    """This also pins the design: with one permutation shared across the columns, every input would move with
    every other and a column the objective ignores would still come back correlated."""
    result = design_optimize(lambda d: d["a"], {"a": (0.0, 1.0), "b": (0.0, 1.0)}, n=64, seed=2)
    assert result.sensitivity["a"] == pytest.approx(1.0)
    assert abs(result.sensitivity["b"]) < 0.3
    assert "rank correlation" in result.summary()


def test_a_non_monotone_dependence_shows_up_as_a_negative_rank():
    result = design_optimize(lambda d: d["a"] + 2.0 * np.sin(6.0 * d["b"]), {"a": (0.0, 1.0), "b": (0.0, 1.0)},
                             n=64, seed=2)
    assert result.sensitivity["b"] < -0.3
    assert result.sensitivity["a"] > result.sensitivity["b"]


def test_a_latin_hypercube_fills_every_stratum():
    result = design_optimize(lambda d: d["a"], {"a": (0.0, 1.0)}, n=16, seed=3)
    values = np.sort([p["a"] for p, _ in result.evaluations])
    assert np.all(values >= 0.0) and np.all(values <= 1.0)
    strata = np.floor(values * 16).astype(int)
    assert len(set(strata.tolist())) == 16, "one design per stratum is what makes it an LHS and not a random sample"


def test_empty_or_inverted_bounds_are_refused():
    with pytest.raises(ValueError, match="bounds is empty"):
        design_optimize(lambda d: 0.0, {})
    with pytest.raises(ValueError, match="high >= low"):
        design_optimize(lambda d: 0.0, {"x": (1.0, 0.0)})


def test_spearman_is_nan_when_there_is_nothing_to_correlate():
    assert np.isnan(spearman([1.0, 2.0], [1.0, 2.0]))          # fewer than three points
    assert np.isnan(spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))  # a constant column
    assert spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
