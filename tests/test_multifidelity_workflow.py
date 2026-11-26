"""The measured checks of the two-level workflow: what the cheap level buys, and when it stops paying (#35).

#35 asks for "a workflow with two data levels" whose benchmark "shows a gain over using high-fidelity data only". The
gain exists and is counted here, on a **real pair of solvers** -- the same one-dimensional Poisson problem at two
``fem`` mesh resolutions, so the two levels are the same physics and the discrepancy is the discretization error, not
two unrelated functions.

The other half of the test is the one a workflow is worth having for: the equal-cost comparison. Handing the cheap
level to a study and comparing against a high-fidelity-only model with the *same number of expensive runs* flatters
co-kriging, because the cheap runs are free in that comparison. The benchmark therefore also asks how many expensive
runs the cheap ones could have bought instead, and the answer here is that the cheap level pays while that number is
one and stops paying at two.

Measured on this machine (``2026-11-25``), 5 repeats, ``n_high = 10``, ``n_low = 64``, 120 held-out designs against a
``fem`` solve twenty times finer than the cheap level and four times finer than the expensive one:

======================  ==================
quantity                 value
======================  ==================
cheap model alone        9.1e-2
high-fidelity only, 10   1.8e-1
co-kriging, 10 + 64      7.8e-2
gain at equal count      2.28 x (worst 1.79 x)
gain with 1 extra run    1.36 x
gain with 2 extra runs   0.74 x
break-even               1 extra run, cost ratio 1/64
======================  ==================
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_analysis.fidelity import FidelityPolicy
from pinneapple_analysis.multifidelity_workflow import (
    FidelityPlan,
    MultiFidelityRun,
    MultiFidelityWorkflow,
    benchmark_multifidelity_workflow,
    rod_fidelities,
)

BOX = [(0.15, 0.85), (0.2, 2.0)]


@pytest.fixture(scope="module")
def rod_pair():
    """The two solver levels and the finer solve used as truth, built once for the module."""
    low, high = rod_fidelities(8, 32)
    oracle = rod_fidelities(128, 128)[0]
    return low, high, oracle


# ------------------------------------------------------------------ the plan: nested designs, one currency
def test_the_expensive_designs_are_a_subset_of_the_cheap_ones():
    plan = FidelityPlan(n_high=6, n_low=24, cost_ratio=0.05, seed=3)
    low, high = plan.designs(np.array(BOX))
    assert low.shape == (24, 2) and high.shape == (6, 2)
    # nested: the discrepancy needs the cheap model at the expensive designs, not an interpolation of it
    assert np.allclose(high, low[:6])
    assert plan.total_cost == pytest.approx(6 + 0.05 * 24)
    assert json.dumps(plan.to_dict())


def test_the_plan_refuses_a_split_it_cannot_fit():
    with pytest.raises(ValueError, match="at least 3"):
        FidelityPlan(n_high=2, n_low=10)
    with pytest.raises(ValueError, match="smaller than n_high"):
        FidelityPlan(n_high=10, n_low=4)
    with pytest.raises(ValueError, match="fraction between 0 and 1"):
        FidelityPlan(n_high=5, n_low=10, cost_ratio=0.0)


# ------------------------------------------------------------------ the run: two models, one comparison
def test_the_workflow_fits_both_models_and_scores_them_on_held_out_designs(rod_pair):
    low, high, oracle = rod_pair
    run = MultiFidelityWorkflow(low, high, BOX, plan=FidelityPlan(n_high=10, n_low=48, cost_ratio=0.02),
                                oracle=oracle, n_test=60).run()
    assert isinstance(run, MultiFidelityRun)
    assert run.design_low.shape == (48, 2) and run.design_high.shape == (10, 2)
    assert run.values_low.shape == (48,) and run.values_high.shape == (10,)
    assert run.error_low and run.error_high_only and run.error_co_kriging      # all three measured
    # the co-kriging model is the one that used both levels: it is ahead of the high-fidelity-only model on the same
    # ten expensive runs, which is the claim #35 makes
    assert run.error_co_kriging < run.error_high_only
    assert run.gain is not None and run.gain > 1.0
    assert run.verdict() == "pays"
    assert run.rho != 0 and np.isfinite(run.rho)
    assert json.dumps(run.to_dict())
    assert "co_kriging" in run.table() and run.report()["n_low"] == 48


def test_without_an_oracle_the_workflow_claims_nothing(rod_pair):
    low, high, _ = rod_pair
    run = MultiFidelityWorkflow(low, high, BOX, plan=FidelityPlan(n_high=6, n_low=20)).run()
    assert run.error_co_kriging is None and run.gain is None
    assert run.verdict() == "undecided"                    # not "pays", not "does not pay": undecided
    assert run.predict(np.array([[0.5, 1.0]])).shape == (1,)
    mean, lower, upper = run.interval(np.array([[0.5, 1.0]]))
    assert np.all(lower <= mean) and np.all(mean <= upper)


def test_the_fitted_model_answers_through_the_fidelity_router(rod_pair):
    """The two levels used as what they were bought for: the model answers, the expensive level is the escalation."""
    low, high, oracle = rod_pair
    run = MultiFidelityWorkflow(low, high, BOX, plan=FidelityPlan(n_high=8, n_low=32), oracle=oracle, n_test=40).run()
    requests = np.array([[0.3, 1.0], [0.6, 1.5], [0.45, 0.5]])
    report = run.serves(requests, oracle, policy=FidelityPolicy(tolerance=None, unknown_trust="surrogate"))
    assert report.n == 3
    assert all(r["value"] is not None for r in report.records)
    assert all(r["route"] in ("surrogate", "high_fidelity") for r in report.records)
    # the escalated requests carry the solver's own number, the surrogate ones carry the model's
    for record in report.records:
        if record["route"] == "surrogate":
            assert record["solver_value"] is None and record["interval"][0] <= record["value"] <= record["interval"][1]
    # a verdict the policy escalates sends every request to the expensive level, and the solver's number is recorded
    escalated = run.serves(requests, oracle, trust="REJECT")
    assert [r["route"] for r in escalated.records] == ["high_fidelity"] * 3
    assert all(r["solver_value"] is not None and r["value"] == r["solver_value"] for r in escalated.records)


# ------------------------------------------------------------------ the measurement
def test_two_levels_beat_high_fidelity_only_at_equal_expensive_count_and_the_gain_is_bounded():
    """The claim of #35, counted, with the equal-cost half that says when it stops being true."""
    result = benchmark_multifidelity_workflow(repeats=5, n_test=120)
    # the measured numbers of the documented case
    assert result["median_error_low"] == pytest.approx(0.0907, abs=2e-3)
    assert result["median_error_high_only"] == pytest.approx(0.179, abs=2e-2)
    assert result["median_error_co_kriging"] == pytest.approx(0.0776, abs=5e-3)
    # the cheap level buys accuracy at the same number of expensive runs, in every repeat
    assert result["median_gain"] == pytest.approx(2.28, abs=0.3)
    assert result["worst_gain"] > 1.5
    for row in result["rows"]:
        assert row["error_co_kriging"] < row["error_high_only"]
    # ... and the equal-cost comparison is what bounds the claim: one extra expensive run still loses to the cheap
    # level, two do not
    assert result["equal_cost_gain"][1] > 1.0
    assert result["equal_cost_gain"][2] < 1.0
    assert result["break_even_extra_runs"] == 1
    assert result["break_even_cost_ratio"] == pytest.approx(1 / 64)
    json.dumps(result)
