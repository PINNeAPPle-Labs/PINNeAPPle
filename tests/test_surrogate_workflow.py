"""The standard surrogate-modeling workflow (#1186): none when the tool is cheap for its use, otherwise the
recommended family checked on new tool calls, and an exported model that refuses to extrapolate."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.surrogate.workflow import load_surrogate, surrogate_modeling

PFR = "pinneapple_physics.closed_form.ideal_reactors:pfr_conversion"
BOUNDS = {"D": (0.02, 0.1), "L": (1.0, 10.0), "k": (0.01, 0.5), "Q": (1e-4, 1e-3)}


def test_a_cheap_tool_needs_no_surrogate_and_says_why():
    r = surrogate_modeling(tool=PFR, bounds=BOUNDS, outputs=["X"], n_doe=16)
    assert r["trained"] is False and r["recommendation"]["recommended"][0] == "none"
    assert r["checks"][0]["code"] == "surrogate.not_needed" and "per solve" in r["checks"][0]["detail"]


def test_forced_training_is_checked_on_new_calls_and_the_export_refuses_extrapolation(tmp_path):
    r = surrogate_modeling(tool=PFR, bounds=BOUNDS, outputs=["X"], n_doe=80, n_fresh=128, force=True,
                           out_dir=str(tmp_path))
    st = {c["code"]: c["status"] for c in r["checks"]}
    assert st["surrogate.X.held_out_error"] == st["surrogate.X.coverage"] == st["surrogate.X.beats_baseline"] == "pass"
    assert r["results"]["X"]["rel_rmse"] < 0.05
    pred = load_surrogate(r["model_path"])
    mean, hw = pred({"D": 0.05, "L": 5.0, "k": 0.1, "Q": 5e-4})["X"]
    tau = np.pi * 0.05 ** 2 / 4 * 5.0 / 5e-4
    assert abs(mean[0] - (1 - np.exp(-0.1 * tau))) < max(3 * hw[0], 0.05)
    with pytest.raises(ValueError, match="outside the trained range"):
        pred({"D": 0.5, "L": 5.0, "k": 0.1, "Q": 5e-4})
    assert (tmp_path / "model_card.json").exists()


def test_too_few_samples_fail_the_held_out_check_instead_of_passing_silently():
    r = surrogate_modeling(tool=PFR, bounds=BOUNDS, outputs=["X"], n_doe=12, n_fresh=64, force=True,
                           accuracy_target=0.01)
    held = next(c for c in r["checks"] if c["code"] == "surrogate.X.held_out_error")
    assert held["status"] == "fail" and held["value"] > 0.01


def test_a_table_uses_its_held_out_rows():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 120)
    y = rng.uniform(0, 1, 120)
    table = {"x": x, "y": y, "f": np.sin(3 * x) + y ** 2}
    r = surrogate_modeling(table=table, inputs=["x", "y"], outputs=["f"], force=True)
    assert r["source"] == {"kind": "table", "rows": 120, "dropped_rows": 0}
    assert r["results"]["f"]["rel_rmse"] < 0.05


def test_flow_and_planner():
    from pinneapple_flow.planner.extensions import match
    from pinneapple_flow.registry import REGISTRY
    assert "surrogate.modeling" in REGISTRY.types()
    assert match("quero um modelo substituto (surrogate) do arrasto da asa")[0][1].id == "surrogate.modeling"
