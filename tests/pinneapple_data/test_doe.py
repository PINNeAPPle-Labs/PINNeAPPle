"""pinneapple_data.doe: designs, coverage, redundancy, priorities, campaign plans."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pinneapple_data.contracts import DatasetSpec, Parameter, ParameterSpace, StudyResult
from pinneapple_data.doe import (
    coverage,
    design,
    fractional_factorial_unit,
    greedy_maximin,
    plan_campaign,
    prioritize,
    redundancy,
)


def _space(constraints=()):
    return ParameterSpace(name="s", parameters=[
        Parameter(name="a", low=0.0, high=1.0),
        Parameter(name="b", low=1.0, high=100.0, log=True),
        Parameter(name="c", kind="integer", low=1, high=4),
        Parameter(name="m", kind="categorical", values=["x", "y"]),
        Parameter(name="k", kind="constant", value=2.0)], constraints=list(constraints))


@pytest.mark.parametrize("method", ["lhs", "maximin", "sobol", "halton", "random"])
def test_designs_have_n_feasible_cases(method):
    space = _space(["a + log10(b) / 2 <= 1.2"])
    d = design(space, 24, method, seed=1)
    assert len(d) == 24 and list(d.cases.columns) == space.names
    assert space.feasible(d.cases).all()
    assert d.unit.shape == (24, 4) and ((d.unit >= 0) & (d.unit <= 1)).all()


def test_lhs_without_constraints_is_stratified():
    space = ParameterSpace(name="s", parameters=[Parameter(name=n, low=0, high=1) for n in "abc"])
    d = design(space, 10, "lhs", seed=3)
    for j in range(3):                                     # one point in each of the 10 strata, every dimension
        assert sorted((d.unit[:, j] * 10).astype(int)) == list(range(10))


def test_maximin_beats_random_on_min_distance():
    space = ParameterSpace(name="s", parameters=[Parameter(name=n, low=0, high=1) for n in "abcd"])
    md = lambda m: coverage(design(space, 30, m, seed=0).unit)["min_distance"]  # noqa: E731
    assert md("maximin") > md("lhs") and md("maximin") > md("random")


def test_sobol_beats_random_on_discrepancy():
    space = ParameterSpace(name="s", parameters=[Parameter(name=n, low=0, high=1) for n in "abc"])
    disc = lambda m: coverage(design(space, 64, m, seed=0).unit)["discrepancy"]  # noqa: E731
    assert disc("sobol") < disc("random")


def test_full_factorial_levels_and_categories():
    d = design(_space(), 0, "full_factorial", levels={"a": 3, "b": 2, "c": 4})
    assert len(d) == 3 * 2 * 4 * 2 and d.info["removed_by_constraints"] == 0   # c: all 4 integers, m: both
    assert sorted(design(_space(), 0, "full_factorial").cases["c"].unique()) == [1, 2, 4]   # 3 levels of 1..4
    assert sorted(d.cases["a"].unique()) == [0.0, 0.5, 1.0] and set(d.cases["m"]) == {"x", "y"}
    assert sorted(d.cases["b"].unique()) == pytest.approx([1.0, 100.0])


@pytest.mark.parametrize("k,n,runs,resolution", [(3, 4, 4, 3), (4, 8, 8, 4), (5, 16, 16, 5), (7, 16, 16, 4),
                                                 (8, 16, 16, 4), (10, 32, 32, 4)])
def test_fractional_factorial_known_resolutions(k, n, runs, resolution):
    unit, info = fractional_factorial_unit(k, n)
    x = 2 * unit - 1
    assert info["runs"] == runs and info["resolution"] == resolution
    np.testing.assert_allclose(x.T @ x, runs * np.eye(k))  # orthogonal, balanced main effects


def test_infill_fills_the_hole_and_adaptive_weights_outputs():
    space = ParameterSpace(name="s", parameters=[Parameter(name="a", low=0, high=1), Parameter(name="b", low=0, high=1)])
    rng = np.random.default_rng(0)
    pts = rng.random((200, 2))
    pts = pts[~((pts[:, 0] > 0.6) & (pts[:, 1] > 0.6))]       # existing runs leave the top-right corner empty
    existing = pd.DataFrame({"a": pts[:, 0], "b": pts[:, 1], "y": np.where(pts[:, 0] < 0.3, 10 * pts[:, 1], 0.0)})
    d = design(space, 5, "infill", existing=existing, seed=0)
    assert (d.unit[:, 0] > 0.6).sum() >= 3 and (d.unit[:, 1] > 0.6).sum() >= 3
    before = coverage(pts)["fill_distance"]
    after = coverage(np.vstack([pts, d.unit]))["fill_distance"]
    assert after < before
    adaptive = design(space, 5, "infill", existing=existing, outputs=["y"], seed=0)
    assert adaptive.info["adaptive"] and len(adaptive) == 5


def test_coverage_metrics_on_known_points():
    grid = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    c = coverage(grid, ["x", "y"])
    assert c["min_distance"] == pytest.approx(0.5) and c["ideal_spacing"] == pytest.approx(0.5)
    assert c["projection_1d"] == {"x": 0.5, "y": 0.5}         # 2 of 4 bins per axis
    assert c["projection_2d"]["x|y"] == pytest.approx(1.0)    # 2 x 2 cells all filled
    assert c["fill_distance"] == pytest.approx(np.hypot(0.25, 0.25), abs=0.01)


def test_redundancy_flags_repeats():
    existing = np.array([[0.1, 0.1], [0.9, 0.9], [0.1001, 0.1]])
    planned = np.array([[0.1, 0.1005], [0.5, 0.5], [0.5, 0.5004]])
    r = redundancy(planned, existing, existing_ids=["r1", "r2", "r3"])
    flags = [(c["repeats_existing"], c["repeats_planned"]) for c in r["cases"]]
    assert flags == [(True, False), (False, False), (False, True)]
    assert r["existing_duplicate_pairs"] == 1 and set(r["existing_redundant"]) == {"r1", "r3"}
    assert r["cases"][0]["nearest_existing"] in ("r1", "r3")


def test_priority_prefix_is_spread_out():
    pts = np.random.default_rng(2).random((40, 2))
    p = prioritize(pts)
    first = pts[p["order"][:4]]
    assert coverage(first)["min_distance"] > 0.3
    assert sorted(p["rank"]) == list(range(1, 41))
    scores = [p["score"][i] for i in p["order"][1:]]
    assert all(scores[i] >= scores[i + 1] - 1e-12 for i in range(len(scores) - 1))  # non-increasing information


def test_greedy_maximin_with_weights_prefers_weighted_neighbourhood():
    fixed = np.array([[0.0, 0.0], [1.0, 1.0]])
    pts = np.array([[0.2, 0.2], [0.8, 0.8]])
    assert greedy_maximin(pts, 1, fixed=fixed, weights_fixed=np.array([1.0, 3.0]))[0] == 1


def test_plan_campaign_drops_repeats_and_exports_contracts():
    space = ParameterSpace(name="s", budget=12, parameters=[Parameter(name="a", low=0, high=1, unit="m"),
                                                            Parameter(name="b", low=0, high=1)])
    pre = design(space, 12, "lhs", seed=5).cases
    existing = pre.assign(run_id=[f"r{i}" for i in range(12)], y=np.arange(12.0))
    plan = plan_campaign(space, method="lhs", seed=5, existing=existing, existing_id_column="run_id", tol=0.01)
    assert len(plan.dropped) == 12 and len(plan.cases) == 0   # the same seed repeats every existing run
    plan = plan_campaign(space, method="maximin", seed=1, existing=existing, existing_id_column="run_id")
    assert len(plan.cases) == 12 and list(plan.cases["priority"]) == list(range(1, 13))
    assert plan.cases["case_id"].iloc[0] == "case_001"
    s = plan.summary()
    assert s["coverage"]["combined"]["fill_distance"] < s["coverage"]["existing"]["fill_distance"]
    res = plan.study_result()
    assert isinstance(res, StudyResult) and res.passed and res.status == "planned"
    assert StudyResult.from_json(res.to_json()) == res
    skel = plan.dataset_skeleton(outputs=["y"], units={"y": "K"})
    assert isinstance(skel, DatasetSpec) and skel.status == "skeleton" and skel.outputs == ["y"]
    assert skel.inputs == ["a", "b"] and skel.parameter_space == space


def test_infeasible_space_is_reported():
    space = _space(["a > 2"])
    with pytest.raises(ValueError, match="constraints"):
        design(space, 5, "lhs", max_candidates=2000)


def test_coverage_of_a_categorical_counts_its_values():
    space = ParameterSpace(name="s", parameters=[Parameter(name="a", low=0, high=1),
                                                 Parameter(name="m", kind="categorical", values=["x", "y", "z"])])
    plan = plan_campaign(space, 30, "lhs", seed=0)
    assert plan.coverage_planned["projection_1d"]["m"] == 1.0          # all 3 categories used
    assert plan.coverage_planned["projection_1d"]["a"] == 1.0          # LHS: every one of the 30 bins
