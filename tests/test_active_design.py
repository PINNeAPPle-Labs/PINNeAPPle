"""The measured checks of the active design on ``Experiment``: fewer simulations for the same error (#34, #153).

The two issues ask for the same thing from two sides -- #153 for "the loop reaches a target error with fewer
simulations than a fixed design, measured and not assumed", #34 for "an ``active=True`` option on ``Experiment`` with
the error gain per simulation measured in the benchmark". The tests here are therefore in two halves: the **interface**
(one solve per design point, the budget spent exactly, the record readable and JSON-safe, the option off leaving every
old behaviour alone), and the **measurement** (:func:`benchmark_active_design` on the documented case, whose numbers
the module's docstring quotes and which are asserted here).

Measured on this machine (``2026-11-25``), ``source_rod_1d``, ``budget = 40``, ``n_initial = 8``, ``batch = 4``,
5 repeats, 300 held-out designs, signals ``uncertainty + curvature + diversity``:

==================  =========================  ========================  ==========================
target              Sobol' 40: simulations     adaptive 40: simulations  final relative error, medians
==================  =========================  ========================  ==========================
0.5 %               28 (24 - 32)               20 (20 - 24)              3.1e-3 vs 1.9e-3 (ratio 0.61)
0.3 %               40, reached in 2 of 5      28, reached in 5 of 5     --
==================  =========================  ========================  ==========================

The case is a real one-dimensional Poisson solve per evaluation with a kink in the design box, so the count is a count
of simulations and the error is the surrogate's against the simulator's own values.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pinneapple_physics.active_design import (  # noqa: E402
    ActivePolicy,
    ActiveRun,
    active_design,
    benchmark_active_design,
    design_box,
    probe,
    source_rod,
)
from pinneapple_physics.experiment import Experiment  # noqa: E402
from pinneapple_physics.physical_problem import Parameter  # noqa: E402
from pinneapple_physics.solving import solve  # noqa: E402

ROD, ROD_CTX = source_rod()


# ------------------------------------------------------------------ the interface: the box, the quantity, one solve
def test_the_design_box_is_the_params_the_problem_declares_as_design_variables():
    box = design_box(ROD)
    assert box == {"x0": (0.15, 0.85), "kappa": (0.2, 2.0)}
    assert design_box(ROD, names=["kappa"]) == {"kappa": (0.2, 2.0)}
    with pytest.raises(KeyError):
        design_box(ROD, names=["nu"])
    # a parameter with bounds but a fixed role is a value the user pinned: a study must not move it
    from dataclasses import replace
    pinned = replace(ROD, parameters={**ROD.parameters, "kappa": Parameter("kappa", value=1.0, role="fixed",
                                                                          bounds=(0.2, 2.0))})
    assert design_box(pinned) == {"x0": (0.15, 0.85)}
    # nothing to vary: refused with the reason, not with an empty study
    nothing = replace(ROD, parameters={})
    with pytest.raises(ValueError, match="declares no parameter to vary"):
        design_box(nothing)


def test_probe_reduces_one_solution_to_one_number_and_refuses_a_name_it_does_not_know():
    solution = solve(ROD.with_parameters(x0=0.3, kappa=1.5), "fem", n=32,
                     ctx={**ROD_CTX, "x0": 0.3, "kappa": 1.5})
    peak = probe(solution, "abs_max")
    assert peak > 0 and peak == pytest.approx(probe(solution, "min") * -1)  # the field is negative throughout
    assert probe(solution, "max") == pytest.approx(0.0)        # the field vanishes on the two Dirichlet ends
    assert probe(solution, "mean") < 0
    assert probe(solution, "integral") == pytest.approx(probe(solution, "mean"), rel=1e-12)  # the rod is unit long
    assert probe(solution, lambda sol: 7.0) == pytest.approx(7.0)          # a QoI the library does not know
    with pytest.raises(ValueError, match="unknown quantity"):
        probe(solution, "drag")


def test_active_design_spends_exactly_the_budget_one_solve_per_point():
    run = active_design(ROD, method="fem", options={"n": 16}, context=ROD_CTX,
                        policy=ActivePolicy(budget=16, n_initial=6, batch=5))
    assert isinstance(run, ActiveRun)
    assert run.n_simulations == 16
    assert run.X.shape == (16, 2) and run.y.shape == (16,)
    assert sum(r["n"] for r in run.design.rounds) == 16 and run.design.rounds[0]["n"] == 6
    assert run.names == ["x0", "kappa"] and run.box == {"x0": (0.15, 0.85), "kappa": (0.2, 2.0)}
    # every simulation sits inside the box the problem declares
    lo = np.array([0.15, 0.2])
    hi = np.array([0.85, 2.0])
    assert np.all(run.X >= lo) and np.all(run.X <= hi)
    # the values are the simulator's, not the surrogate's: the peak of the field is positive and varies over the box
    assert np.all(run.y > 0) and np.ptp(run.y) > 0
    report = run.report()
    json.dumps(run.to_dict())                                   # the record is JSON-safe
    assert report["n_simulations"] == 16 and report["quantity"] == "abs_max"
    assert report["signals"] == ["uncertainty", "curvature", "diversity"]
    assert report["error"] is None and report["reached_target"] is None   # no reference, no claim
    assert "round" in run.table() and run.table().count("\n") >= 16


def test_with_a_reference_the_error_is_measured_and_the_target_is_decided():
    run = active_design(ROD, method="fem", options={"n": 16}, context=ROD_CTX, reference="fem",
                        reference_options={"n": 64}, n_test=80,
                        policy=ActivePolicy(budget=16, n_initial=8, batch=4, target=0.05))
    assert run.reference == "fem"
    assert run.error is not None and np.isfinite(run.error)
    assert run.error == pytest.approx(run.error)               # not NaN
    assert run.reached_target() == bool(run.error <= 0.05)
    json.dumps(run.to_dict())


def test_the_policy_refuses_a_budget_it_cannot_adapt_in():
    with pytest.raises(ValueError, match="nothing to adapt"):
        ActivePolicy(budget=8, n_initial=8)
    with pytest.raises(ValueError, match="at least 2"):
        ActivePolicy(n_initial=1)
    with pytest.raises(ValueError, match="at least 2 simulations"):
        ActivePolicy(budget=1)


# ------------------------------------------------------------------ the option on Experiment
def test_active_on_Experiment_returns_the_same_result_type_with_the_study_recorded():
    experiment = Experiment(ROD, method="fem", options={"n": 16}, reference="fem", reference_options={"n": 64},
                            active={"budget": 16, "n_initial": 8, "batch": 4, "target": 0.10},
                            active_context=ROD_CTX)
    assert experiment.config()["active"] == {"budget": 16, "n_initial": 8, "batch": 4, "target": 0.10}
    result = experiment.run()
    record = result.active
    assert record is not None
    assert record["n_simulations"] == 16 and record["method"] == "fem"
    assert record["error"] is not None and record["reached_target"] in (True, False)
    assert set(record["best_design"]) == {"x0", "kappa"}
    assert "design" in record and len(record["design"]["X"]) == 16
    # the solution is a real one at the best measured design, with the ordinary predict contract
    predicted = result.solution.predict(np.array([[0.5]]))
    assert predicted.shape == (1, 1) and np.isfinite(predicted[0, 0])
    assert np.all(ROD.with_parameters(**record["best_design"]).parameter_values().values())
    json.dumps(result.to_dict())


def test_active_is_saved_and_reloaded_and_off_changes_nothing():
    experiment = Experiment(ROD, method="fem", options={"n": 16}, active={"budget": 12, "n_initial": 6, "batch": 3})
    reloaded = Experiment.from_dict(json.loads(json.dumps(experiment.to_dict())))
    assert reloaded.active == {"budget": 12, "n_initial": 6, "batch": 3}
    assert json.loads(json.dumps(experiment.to_dict()))["active"]["budget"] == 12

    plain = Experiment(ROD, method="fem", options={"n": 16}, active_context=ROD_CTX)
    assert plain.active is False
    result = plain.run()                                        # one solve, exactly as before
    assert result.active is None and "active" not in result.to_dict()
    assert result.solution.info["n_cells_per_axis"] == [16]
    assert "active" not in experiment.to_dict()["detached"] if "detached" in experiment.to_dict() else True
    with pytest.raises(TypeError, match="active must be"):
        Experiment(ROD, method="fem", active="yes")
    with pytest.raises(TypeError, match="unexpected keyword"):
        Experiment(ROD, method="fem", active={"budget": 12, "n_init": 6})


# ------------------------------------------------------------------ the measurement
def test_the_active_rule_reaches_the_target_error_with_fewer_simulations_than_sobol():
    """The acceptance of #153, counted: same target, fewer simulations, and it is not one lucky repeat."""
    result = benchmark_active_design(target=0.005, budget=40, repeats=5)
    assert result["median_fixed_simulations"] is not None and result["median_active_simulations"] is not None
    # the measured numbers: 28 simulations for Sobol', 20 for the active design
    assert result["median_fixed_simulations"] == pytest.approx(28.0)
    assert result["median_active_simulations"] == pytest.approx(20.0)
    assert result["simulations_saved"] == pytest.approx(8.0)
    # every repeat of both rules reached it, so the medians are counts and not survivors of a failure
    assert result["fixed_reached_target"] == result["repeats"] == result["active_reached_target"]
    # the active design never loses badly on a repeat: the worst per-repeat error ratio stays under one
    assert result["median_error_ratio"] < 0.8
    assert result["worst_error_ratio"] < 0.9
    # and the error at equal budget is the other way to state the same gain
    for row in result["rows"]:
        assert row["active_error"] < row["fixed_error"] or row["error_ratio"] < 1.0
        assert row["active_simulations"] <= row["fixed_simulations"]
    json.dumps(result)


def test_at_a_tighter_target_the_gap_stops_being_a_count():
    """Where Sobol' does not get there at all inside the budget, the active design still does -- 5 of 5 against 2."""
    result = benchmark_active_design(target=0.003, budget=40, repeats=5)
    assert result["active_reached_target"] == result["repeats"]
    assert result["fixed_reached_target"] < result["repeats"]
    assert result["median_active_simulations"] is not None
    assert result["median_active_simulations"] <= 32
