"""The battle scoreboard (#373): two models on one stream, and the step at which each one leaves its domain.

The number that matters is not a mean error -- it is the step where a model stops being usable, and a table of those
steps is only worth publishing if it can be wrong and be caught. So the tests here check the arithmetic of the exit,
the total order the scoreboard sorts by, and that a table whose order was changed by hand fails the gate; plus one
small battle trained end to end, so the pipeline (both families, the same split, the same budget) is known to run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pinneapple_arena import battle

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ the exit criterion
def test_the_residual_alarm_is_the_first_step_past_the_factor_calibrated_on_the_windows_edge():
    in_window = [True] * 5 + [False] * 5
    curve = [0.4, 0.5, 0.2, 0.3, 0.45, 0.6, 1.9, 2.1, 3.0, 4.0]
    record = battle.domain_exit(curve, in_window, factor=4.0)
    assert record["calibrated_on"] == pytest.approx(0.45), "the last step inside the window, not the worst one"
    assert record["threshold"] == pytest.approx(1.8)
    assert record["step"] == 6, "step 5 is below the threshold; the exit is the first step above it"
    assert record["value"] == pytest.approx(1.9)


def test_the_error_exit_is_absolute_so_it_reads_the_same_for_a_good_and_a_mediocre_model():
    in_window = [True, True, False, False]
    tight = battle.domain_exit([1e-4, 2e-4, 3e-2, 6e-2], in_window, absolute=5e-2)
    loose = battle.domain_exit([1e-2, 4e-2, 5.1e-2, 9e-2], in_window, absolute=5e-2)
    assert tight["step"] == 3 and loose["step"] == 2
    assert tight["criterion"] == "the published tolerance"
    relative = battle.domain_exit([1e-4, 2e-4, 3e-2, 6e-2], in_window, factor=4.0)
    assert relative["step"] == 2, "a relative criterion would declare the better model out first"


def test_a_spike_inside_the_window_is_never_an_exit():
    in_window = [False, True, True, False, False]
    curve = [9.0, 0.1, 5.0, 9.0, 1.0]
    record = battle.domain_exit(curve, in_window, factor=4.0)
    assert record["step"] is None, "an exit can only be outside the window the model was validated on"
    assert record["reason"] == "never crossed the threshold"


def test_calibrating_on_the_worst_step_inside_is_the_stricter_bar():
    in_window = [True, True, True, False, False]
    curve = [0.01, 0.9, 0.05, 1.0, 2.0]
    strict = battle.domain_exit(curve, in_window, factor=4.0, calibrate="max")
    edge = battle.domain_exit(curve, in_window, factor=4.0)
    assert strict["step"] is None and edge["step"] == 3


def test_the_two_thresholds_have_to_be_given_one_at_a_time():
    with pytest.raises(ValueError, match="exactly one"):
        battle.domain_exit([1.0, 2.0], [True, False])
    with pytest.raises(ValueError, match="exactly one"):
        battle.domain_exit([1.0, 2.0], [True, False], factor=4.0, absolute=0.1)


def test_a_curve_with_no_finite_value_inside_the_window_has_nothing_to_calibrate_on():
    record = battle.domain_exit([np.nan, 1.0, 2.0], [True, False, False], factor=4.0)
    assert record["step"] is None and record["threshold"] is None
    assert "calibrate" in record["reason"]


# ------------------------------------------------------------------ the measurements
def test_the_error_curve_is_the_relative_l2_of_each_step():
    ref = np.array([[1.0, 0.0], [0.0, 2.0]])
    pred = np.array([[0.0, 0.0], [0.0, 3.0]])
    curve = battle.relative_l2_curve(pred, ref)
    assert curve[0] == pytest.approx(1.0)
    assert curve[1] == pytest.approx(0.5)


def test_the_residual_operator_finds_the_exact_solution_almost_residual_free():
    field = battle._heat_field(nx=129, nt=41, alpha=0.05, split_at=0.5)
    exact = battle.pde_residual_curve(field.exact, field.residual)
    bumped = battle.pde_residual_curve(field.exact * 1.05, field.residual)
    finite = exact[np.isfinite(exact)]
    assert np.all(np.isfinite(finite)) and finite.max() < 1e-3, "on a grid that resolves it, the closed form " \
                                                               "satisfies its own equation up to rounding"
    assert np.nanmean(bumped) > np.nanmean(exact), "a perturbed field must not look healthier than the reference"


def test_the_burgers_floor_is_the_grid_and_the_alarm_still_does_not_fire_on_the_truth():
    """Burgers at nu = 0.01 forms a shock at t = 1/pi, and past it the layer is thinner than a cell, so the operator's
    floor there is *not* a rounding error: measured 0.89 / 0.35 / 0.096 on 65 / 129 / 257 points in x at 41 in t. That
    is why the alarm is calibrated on the model's own value at the edge of its window and not on an absolute number --
    and why the property the monitor actually needs is the one asserted here: the truth, scored with the same
    criterion the models are, never leaves its domain."""
    floors = [float(np.nanmax(battle.pde_residual_curve(f.exact, f.residual)))
              for f in (battle._burgers_field(nx=nx, nt=41, nu=0.01, split_at=0.5) for nx in (65, 129, 257))]
    assert floors[0] > floors[1] > floors[2], f"the floor has to fall with refinement, got {floors}"
    assert floors[0] / floors[2] > 4.0, "and it has to fall by more than the factor the alarm uses, or refining " \
                                        "the grid would be the only way to pass"
    field = battle._burgers_field(nx=129, nt=41, nu=0.01, split_at=0.5)
    truth = battle.pde_residual_curve(field.exact, field.residual)
    alarm = battle.domain_exit(truth, field.t <= 0.5, factor=battle.PROTOCOL["exit"]["residual_factor"])
    assert alarm["step"] is None, "a monitor that fires on the exact solution measures the grid, not the model"


def test_the_residual_operator_keeps_the_edges_out_instead_of_inventing_a_value():
    field = battle._burgers_field(nx=33, nt=17, nu=0.01, split_at=0.5)
    r = field.residual(field.exact)
    assert np.isnan(r[:, 0]).all() and np.isnan(r[:, -1]).all(), "one-sided differences would hide a wall error"


# ------------------------------------------------------------------ the scoreboard
def _block(rows, *, in_window=6, n_steps=10):
    """A minimal case block, shaped exactly like the one ``run_case`` records, so the board can be tested alone.

    A row may give ``exits`` -- one per seed, ``None`` for a seed that survived the stream -- instead of the single
    ``exit`` every run of the single-seed fixtures shares; the summary is then the same ``_summarise`` the record is
    built with, so the test cannot agree with the board while disagreeing with the run."""
    times = np.linspace(0.0, 1.0, n_steps)
    models = {}
    for row in rows:
        curve = list(row["curve"])
        exits = list(row["exits"]) if "exits" in row else [None if row["exit"] >= n_steps else row["exit"]]
        models[row["model"]] = {
            "kind": "test", "supervision": "nothing", "parameters": 1,
            "runs": [{"seed": i, "train_s": row["train_s"], "predict_s": 0.0, "parameters": 1,
                      "relative_l2": curve, "pde_residual": [0.0] * n_steps,
                      "residual_alarm": {"step": None}, "error_exit": {"step": step, "threshold": 0.1}}
                     for i, step in enumerate(exits)],
            "summary": {
                "rel_l2_in_window_max": {"mean": 0.01},
                "rel_l2_out_window_mean": {"mean": 0.5},
                "rel_l2_final": {"mean": row["final"], "min": row["final"], "max": row["final"]},
                "error_exit_step": battle._summarise(exits, never=float(n_steps)),
                "residual_alarm_step": battle._summarise([None] * len(exits), never=float(n_steps)),
                "train_s": {"mean": row["train_s"], "min": row["train_s"], "max": row["train_s"]},
                "predict_s": {"mean": 0.0},
                "exit_threshold": {"mean": 0.1, "alarm_threshold": 0.1},
            },
        }
    return {"case": {"id": "synthetic", "name": "synthetic", "equation": "", "reference": "", "split_at": 0.5},
            "stream": {"n_steps": n_steps, "t": [float(v) for v in times],
                       "in_window": [bool(i < in_window) for i in range(n_steps)]},
            "models": models, "model_order": [r["model"] for r in rows]}


def test_the_scoreboard_is_ordered_by_the_step_each_model_left_capital_first():
    block = _block([{"model": "fno", "curve": [0.0] * 9, "exit": 6, "final": 0.2, "train_s": 1.0},
                    {"model": "pinn", "curve": [0.0] * 9, "exit": 8, "final": 0.1, "train_s": 1.0}])
    rows = battle.scoreboard(block)
    assert [r["model"] for r in rows] == ["fno", "pinn"], "who left first ranks first, whatever the final error"
    assert [r["rank"] for r in rows] == [1, 2]
    assert rows[0]["left_at_t"] == pytest.approx(block["stream"]["t"][6])


def test_a_model_that_survived_the_stream_is_last_and_says_so():
    block = _block([{"model": "fno", "curve": [0.0] * 9, "exit": 6, "final": 0.2, "train_s": 1.0},
                    {"model": "pinn", "curve": [0.0] * 9, "exit": 10, "final": 0.01, "train_s": 1.0}])
    rows = battle.scoreboard(block)
    assert [r["model"] for r in rows] == ["fno", "pinn"]
    assert rows[1]["left_within_stream"] is False


def test_the_order_is_total_so_a_tie_is_still_a_decision():
    same_exit = [{"model": "fno", "curve": [0.0] * 9, "exit": 7, "final": 0.4, "train_s": 2.0},
                 {"model": "pinn", "curve": [0.0] * 9, "exit": 7, "final": 0.1, "train_s": 9.0}]
    rows = battle.scoreboard(_block(same_exit))
    assert [r["model"] for r in rows] == ["pinn", "fno"], "a tie goes to the smaller final error"
    same_both = [{"model": "fno", "curve": [0.0] * 9, "exit": 7, "final": 0.2, "train_s": 9.0},
                 {"model": "pinn", "curve": [0.0] * 9, "exit": 7, "final": 0.2, "train_s": 1.0}]
    rows = battle.scoreboard(_block(same_both))
    assert [r["model"] for r in rows] == ["pinn", "fno"], "then to the cheaper training"


def test_a_model_that_left_in_one_seed_has_left_and_the_mean_does_not_get_to_say_otherwise():
    """The defect this test exists for: two seeds, one model that left at step 3 in the first and survived the second.

    The mean of a real exit and the end of the stream substituted for a missing one is 6.5, and printing the row from
    it says "never" and ranks the model as if it had survived. It left -- and it is the model that left first."""
    block = _block([{"model": "fno", "curve": [0.0] * 9, "exits": [None, None], "final": 0.30, "train_s": 1.0},
                    {"model": "pinn", "curve": [0.0] * 9, "exits": [3, None], "final": 0.10, "train_s": 1.0}])
    rows = {r["model"]: r for r in battle.scoreboard(block)}
    assert rows["pinn"]["left_within_stream"] is True, "one seed of two left; the row must not say never"
    assert rows["pinn"]["left_in_seeds"] == 1 and rows["pinn"]["n_seeds"] == 2
    assert rows["pinn"]["left_at_step"] == pytest.approx(3.0), "the first seed to leave, not the mean of 3 and the " \
                                                              "end of the stream"
    assert rows["pinn"]["left_at_step_mean"] == pytest.approx(6.5), "and the mean is still published, as a mean"
    assert rows["fno"]["left_within_stream"] is False and rows["fno"]["left_in_seeds"] == 0
    assert [r["model"] for r in battle.scoreboard(block)] == ["pinn", "fno"], "the model that left is the one that " \
                                                                            "left, and it ranks first"
    block["scoreboard"] = battle.scoreboard(block)
    text = battle.table({"cases": {"synthetic": block}})
    assert "in 1 of 2 seeds" in text, text
    assert "never (survived the whole stream)" in text, "and the model that did survive still says so"


def test_between_two_models_first_seen_to_leave_together_the_one_that_left_every_time_is_worse():
    block = _block([{"model": "fno", "curve": [0.0] * 9, "exits": [5, 5], "final": 0.1, "train_s": 1.0},
                    {"model": "pinn", "curve": [0.0] * 9, "exits": [5, None], "final": 0.1, "train_s": 1.0}])
    rows = battle.scoreboard(block)
    assert [r["model"] for r in rows] == ["fno", "pinn"], "same first step: the one that left in both seeds is worse"


def test_the_reference_is_what_decides_a_disagreement():
    block = _block([{"model": "fno", "curve": [0.0] * 5 + [0.9, 0.1, 0.1, 0.1], "exit": 6, "final": 0.1,
                     "train_s": 1.0},
                    {"model": "pinn", "curve": [0.0] * 5 + [0.05, 0.1, 0.1, 0.1], "exit": 6, "final": 0.1,
                     "train_s": 1.0}], n_steps=9, in_window=5)
    spread = [0.0] * 5 + [0.8, 0.1, 0.1, 0.1]
    verdict = battle.disagreement(block, spread)
    assert verdict["available"] is True
    assert verdict["step"] == 5, "the step where the two fields are furthest apart"
    assert verdict["reference_favours"] == "pinn", "the smaller error at that step is the one the reference agrees with"
    block["models"]["fno"]["runs"][0]["relative_l2"][5] = 0.01
    assert battle.disagreement(block, spread)["reference_favours"] == "fno", "and the verdict follows the numbers"


def test_a_disagreement_inside_the_window_is_not_the_one_that_counts():
    block = _block([{"model": "fno", "curve": [0.0] * 9, "exit": 6, "final": 0.1, "train_s": 1.0},
                    {"model": "pinn", "curve": [0.0] * 9, "exit": 6, "final": 0.1, "train_s": 1.0}],
                   n_steps=9, in_window=5)
    spread = [9.0, 9.0, 9.0, 9.0, 9.0, 0.1, 0.2, 0.3, 0.4]
    assert battle.disagreement(block, spread)["step"] == 8, "both models were fitted inside the window; a gap there " \
                                                           "is a training artefact"


# ------------------------------------------------------------------ the record and the gate
@pytest.fixture(scope="module")
def small_battle():
    """One real battle at a test budget: the smallest grid the residual operator accepts, a handful of epochs.

    The protocol is shrunk only for the duration of the run and restored before this fixture yields, not at module
    teardown: a module-scoped teardown runs *after* the last test in the file, so a global left at the test budget
    would make ``test_the_protocol_declares_what_a_reproduction_has_to_keep`` read one seed where the module declares
    two -- a leak that only shows up depending on where the test sits in the file."""
    saved_grid, saved_seeds = battle.PROTOCOL["grid"], battle.PROTOCOL["seeds"]
    saved_epochs = battle.PROTOCOL["budget"]["epochs"]
    try:
        battle.PROTOCOL["grid"] = {"nx": 33, "nt": 17}
        battle.PROTOCOL["seeds"] = (0,)
        battle.PROTOCOL["budget"]["epochs"] = 5
        results = battle.run_all()
    finally:
        battle.PROTOCOL["grid"], battle.PROTOCOL["seeds"] = saved_grid, saved_seeds
        battle.PROTOCOL["budget"]["epochs"] = saved_epochs
    return results


def test_the_battle_trains_both_families_on_the_same_stream(small_battle):
    block = small_battle["cases"]["burgers"]
    assert block["model_order"] == list(battle.MODELS)
    n_steps = block["stream"]["n_steps"]
    for model in battle.MODELS:
        runs = block["models"][model]["runs"]
        assert len(runs) == 1 and runs[0]["seed"] == 0
        assert len(runs[0]["relative_l2"]) == n_steps, "every step of the stream is scored, not only the held-out tail"
        assert np.isfinite(runs[0]["relative_l2"]).all()
        assert runs[0]["train_s"] > 0 and runs[0]["parameters"] > 0
    assert len(block["scoreboard"]) == 2
    assert [r["rank"] for r in block["scoreboard"]] == [1, 2]


def test_the_pinn_sees_the_equation_and_the_operator_does_not(small_battle):
    block = small_battle["cases"]["burgers"]
    assert "equation" in block["models"]["pinn"]["supervision"]
    assert "equation" not in block["models"]["fno"]["supervision"]


def test_the_page_is_generated_from_the_record(small_battle):
    text = battle.page(small_battle)
    assert "| rank | model |" in text
    for model in battle.MODELS:
        assert model in text
    assert "operator floor" in text, "the alarm column is read next to the operator's own floor, or it is read as " \
                                     "an absolute number it is not"
    assert "burgers" in text


def test_an_inverted_scoreboard_is_refused(tmp_path, small_battle):
    results = json.loads(json.dumps(small_battle))
    results["cases"]["burgers"]["scoreboard"].reverse()
    battle.write(results, root=tmp_path)
    problems = battle.check(tmp_path)
    assert any("the published scoreboard is not the one the measurements give" in p for p in problems), problems


def test_a_scoreboard_whose_ranks_were_renumbered_is_refused(tmp_path, small_battle):
    results = json.loads(json.dumps(small_battle))
    rows = results["cases"]["burgers"]["scoreboard"]
    rows[0]["rank"], rows[1]["rank"] = rows[1]["rank"], rows[0]["rank"]
    battle.write(results, root=tmp_path)
    assert battle.check(tmp_path), "renumbering without reordering is still a table that disagrees with the run"


def test_a_page_that_drifted_from_the_record_is_refused(tmp_path, small_battle):
    battle.write(small_battle, root=tmp_path)
    page = tmp_path / battle.PAGE
    page.write_text(page.read_text(encoding="utf-8").replace("burgers_1d", "burgers_first"), encoding="utf-8")
    problems = battle.check(tmp_path)
    assert any("is not what the results file generates" in p for p in problems)


def test_the_record_published_in_the_repository_is_the_one_the_measurements_give():
    assert (ROOT / battle.RESULTS).is_file(), "run python scripts/battle.py --run"
    problems = battle.check(ROOT)
    assert problems == [], "\n".join(problems)


def test_the_protocol_declares_what_a_reproduction_has_to_keep():
    for field in battle.REQUIRED_PROTOCOL_FIELDS:
        assert field in battle.PROTOCOL, field
    assert len(battle.PROTOCOL["seeds"]) >= 2, "one seed has no spread"
    assert battle.PROTOCOL["exit"]["residual_factor"] > 1
