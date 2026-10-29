"""pinneapple_design.design_optimizer.airfoil_drag (ported from the tests of PINNeAPPle-apps
poc_design_optimization; shorter runs)."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
from pinneapple_design.design_optimizer import airfoil_drag as ad  # noqa: E402


def test_outline_spans_the_chord():
    o = ad.airfoil_outline(0.02, 0.15, n_pts=160)
    assert len(o["x"]) == len(o["y"]) == 160
    assert min(o["x"]) >= -0.01 and max(o["x"]) <= 1.01


def test_surrogate_fits_the_closed_form_on_held_out_designs():
    info = ad.train_surrogate(seed=7)
    assert info["val_rel_error"]["cl"] < 0.01 and info["val_rel_error"]["cd"] < 0.01
    assert info["n_train"] == 2000 and info["n_val"] == 400


@pytest.fixture(scope="module")
def run():
    return ad.run_design_optimization(n_iterations=150, seed=1)


def test_run_cuts_closed_form_drag_and_holds_lift(run):
    assert run["final"]["cd"] < run["baseline"]["cd"]
    assert run["improvement"]["drag_reduction_pct"] > 5.0
    assert abs(run["improvement"]["lift_delta_pct"]) < 10.0
    assert run["within_bounds"]


def test_trajectory_and_json(run):
    traj = run["trajectory"]
    assert run["n_iterations_run"] == len(traj) <= 150
    assert [p["iteration"] for p in traj] == list(range(len(traj)))
    assert traj[-1]["objective"] <= traj[0]["objective"] + 1e-9
    json.dumps(run)


def test_seed_reproduces_and_none_draws_fresh():
    a = ad.run_design_optimization(n_iterations=30, seed=123, surrogate_epochs=60)
    b = ad.run_design_optimization(n_iterations=30, seed=123, surrogate_epochs=60)
    assert a["trajectory"] == b["trajectory"]
    c = ad.run_design_optimization(n_iterations=5, seed=None, surrogate_epochs=5)
    d = ad.run_design_optimization(n_iterations=5, seed=None, surrogate_epochs=5)
    assert c["seed"] != d["seed"]
