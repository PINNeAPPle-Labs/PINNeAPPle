"""Geometry-conditioned surrogate case (#83): the field error on a body that was not in the training set.

The acceptance of the roadmap item is a case, not a training script: a geometry outside the training set and the
error reported on it. So the tests here check the three things that could make the reported number meaningless --
the geometry frame, the split, and the report -- and run the pipeline at the quick budget instead of the published
one (``benchmarks/_out/geometry_conditioned_surrogate.json`` holds the run of the full protocol).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.geometry_conditioned_surrogate import (
    OUT,
    PROTOCOL,
    QUICK,
    _outside,
    _train_box,
    body,
    lift_from_cp,
    run,
    split,
)
from pinneapple_design.aero.geometry import REFERENCE

ROOT = Path(__file__).resolve().parent.parent


def test_the_panel_frame_is_the_solvers_own_and_points_out_of_the_body():
    """A predicted Cp and a solved Cp must be integrated over the same panels, with the same outward normals.

    If the normals were flipped or built on other nodes, the lift of the predicted field would not be the lift the
    solver reports for its own field -- and every derived number of the report would be quietly wrong.
    """
    solved = body(REFERENCE["NACA 2412"], n_per_side=40)
    assert lift_from_cp(solved["cp"], solved["normals"], solved["length"]) == pytest.approx(solved["cl"], abs=1e-12)
    outward = np.einsum("ij,ij->i", solved["normals"], solved["mid"] - solved["mid"].mean(0))
    assert (outward > 0).all(), "the normals of the body must point away from its centre"
    assert len(solved["cp"]) == len(solved["mid"]) == len(solved["length"]) == 80


def test_new_geometries_are_outside_the_training_box_by_construction():
    """The training box is the documented ranges pulled back; the new bodies have a weight outside it.

    Without this the "outside the training set" claim would rest on the sampling seed.
    """
    sets = split(QUICK)
    box = _train_box(PROTOCOL["box_shrink"])
    assert QUICK["train_shapes"] == len(sets["train"])
    assert all(_outside(shape, box) == [] for _, shape in sets["train"])
    assert all(_outside(shape, box) == [] for _, shape in sets["unseen"])
    drawn = [(name, shape) for name, shape in sets["new"] if not name.startswith("NACA")]
    assert drawn, "at least one new geometry has to be drawn outside the training box"
    assert all(_outside(shape, box) for _, shape in drawn)


def test_the_error_is_reported_on_a_body_it_never_saw():
    """The number the issue asks for: a body outside the training set and its error, next to the trained-on ones."""
    r = run(QUICK)
    assert r["train"]["n"] == QUICK["train_shapes"]
    assert r["new_geometry"]["n"] >= 1
    for group in ("unseen", "new_geometry"):
        block = r[group]
        assert np.isfinite(block["relative_l2_mean"])
        assert 0.0 < block["relative_l2_mean"] < 1.0, f"{group}: a reported error, not a placeholder"
        assert block["relative_l2_max"] >= block["relative_l2_mean"]
        for row in block["rows"]:
            assert np.isfinite(row["relative_l2"]) and row["relative_l2"] > 0
            assert np.isfinite(row["cl_surrogate"])
    # an untrained body carries the parameters that make it new, so the report says why it is new
    assert any(row["outside_training_box"] or row["name"].startswith("NACA")
               for row in r["new_geometry"]["rows"])
    # the decision level is measured on the held-out bodies only
    assert r["decision"]["n_designs"] == QUICK["unseen_shapes"] + r["new_geometry"]["n"]
    assert 0.0 <= r["decision"]["flip_rate"] <= 1.0
    assert np.isfinite(r["decision"]["regret_relative"])


def test_the_published_result_is_the_protocol_it_says_it_is():
    """The committed JSON is a full-protocol run: the same protocol, and a number for the new geometry."""
    published = Path(OUT)
    if not published.exists():
        pytest.skip("no published result; run python -m benchmarks.geometry_conditioned_surrogate")
    result = json.loads(published.read_text(encoding="utf-8"))
    expected = {k: (list(v) if isinstance(v, tuple) else v) for k, v in PROTOCOL.items()}
    assert result["protocol"] == expected
    assert result["train"]["n"] == PROTOCOL["train_shapes"]
    assert result["new_geometry"]["n"] >= PROTOCOL["new_geometries"]
    assert 0.0 < result["new_geometry"]["relative_l2_mean"] < 1.0
    assert result["new_geometry"]["rows"][0]["relative_l2"] > 0
    # the published run is the full budget, not the quick one
    assert result["cost"]["train_s"] > 1.0
