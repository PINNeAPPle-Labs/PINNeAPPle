"""pinneapple_design.design_optimizer.quality_diversity: the archive, the search, the failure map (#354).

The tests use analytic evaluators, so the assertions are about the machinery -- a cell keeps its own best, the
archive does not collapse into a point, the budget is spent exactly, out-of-range behaviour is refused instead of
clipped -- and not about a surrogate's accuracy, which the example measures.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_design.design_optimizer.quality_diversity import (
    BehaviourAxis,
    BehaviourSpace,
    CollapsedArchiveError,
    EliteArchive,
    Evaluation,
    FailureRegion,
    QDConfig,
    QualityDiversitySearch,
    descriptor_axes,
    descriptor_vector,
    find_failure_region,
    latin_hypercube,
    plot_archive,
    uniform_search,
)

BOX = {"x": (0.0, 1.0), "y": (0.0, 1.0)}


def space(bins: int = 8) -> BehaviourSpace:
    """A two-axis grid over the whole (0, 1) square, so a descriptor is placeable by construction."""
    return BehaviourSpace((BehaviourAxis("a", 0.0, 1.0, why="first behaviour"),
                           BehaviourAxis("b", 0.0, 1.0, why="second behaviour")), bins=bins)


def quadratic(row: np.ndarray) -> Evaluation:
    """A cheap smooth objective whose descriptors are the design itself: the archive should fill the map."""
    x, y = float(row[0]), float(row[1])
    return Evaluation(value=(x - 0.5) ** 2 + (y - 0.5) ** 2, descriptors=(x, y))


# -- the behaviour axes -------------------------------------------------------------------------------------------

def test_axis_bins_clamp_the_top_edge_and_refuse_what_is_outside():
    axis = BehaviourAxis("t", 0.0, 10.0)
    assert axis.index(0.0, 10) == 0
    assert axis.index(10.0, 10) == 9                 # the top edge belongs to the last cell, not to a tenth
    assert axis.index(-0.001, 10) is None            # outside is refused, never clipped into an edge cell
    assert axis.index(10.001, 10) is None
    assert axis.index(float("nan"), 10) is None
    assert axis.index(float("inf"), 10) is None


def test_log_axis_spends_its_cells_on_the_low_end_and_needs_a_positive_range():
    axis = BehaviourAxis("Rth", 0.01, 1.0, scale="log")
    assert axis.index(0.01, 4) == 0
    assert axis.index(1.0, 4) == 3
    assert axis.index(0.1, 4) == 2                   # a decade is half the positions on a log axis
    assert axis.index(0.0, 4) is None
    with pytest.raises(ValueError):
        BehaviourAxis("bad", 0.0, 1.0, scale="log")
    with pytest.raises(ValueError):
        BehaviourAxis("bad", 0.0, 1.0, scale="sqrt")


def test_axis_interval_and_centre_agree_with_the_bins():
    axis = BehaviourAxis("t", 0.0, 4.0)
    lo, hi = axis.interval(1, 4)
    assert (lo, hi) == (1.0, 2.0)
    assert lo < axis.centre(1, 4) < hi


def test_behaviour_space_condition_names_the_cell_by_its_intervals():
    s = space(4)
    condition = s.condition((1, 2))
    assert condition["a"] == (0.25, 0.5)
    assert condition["b"] == (0.5, 0.75)
    assert "a in [0.25, 0.5]" in s.describe((1, 2))


def test_behaviour_space_rejects_a_grid_that_cannot_exist():
    with pytest.raises(ValueError):
        BehaviourSpace((), bins=4)
    with pytest.raises(ValueError):
        BehaviourSpace((BehaviourAxis("a", 1.0, 1.0),), bins=4)
    with pytest.raises(ValueError):
        BehaviourSpace((BehaviourAxis("a", 0.0, 1.0),), bins=(4, 4))


def test_descriptor_vector_reads_the_qualitative_descriptors():
    class Descriptors:                                     # the shape of pinneapple_design.qualitative.Descriptors
        def as_dict(self):
            return {"slenderness": 2.5, "sphericity": 0.81, "volume": 0.02}

    assert descriptor_vector(Descriptors(), ("slenderness", "sphericity")) == (2.5, 0.81)
    assert descriptor_vector({"a": 1, "b": 2}, ("b",)) == (2.0,)
    with pytest.raises(KeyError):
        descriptor_vector({"a": 1}, ("nope",))


def test_descriptor_axes_span_what_the_samples_reach():
    samples = [{"s": v, "c": 1.0 + v} for v in (1.0, 2.0, 3.0)]
    s = descriptor_axes(samples, ("s", "c"), bins=5, labels="bodies", why={"s": "flow goes around"}, margin=0.1)
    assert s.shape == (5, 5)
    assert s.axes[0].why == "flow goes around"
    assert s.axes[0].lo < 1.0 and s.axes[0].hi > 3.0        # padded, so a slightly different design is not refused
    assert s.labels == "bodies"


def test_descriptor_axes_refuses_a_descriptor_that_does_not_vary():
    with pytest.raises(ValueError):
        descriptor_axes([{"s": 1.0}, {"s": 1.0}], ("s",))
    with pytest.raises(ValueError):
        descriptor_axes([], ("s",))


# -- the archive --------------------------------------------------------------------------------------------------

def test_archive_keeps_the_best_of_each_cell_under_each_sense():
    cheaper = EliteArchive(space(2), sense="min")
    dearer = EliteArchive(space(2), sense="max")
    for archive in (cheaper, dearer):
        archive.consider((0.0, 0.0), Evaluation(value=1.0, descriptors=(0.1, 0.1)))
        archive.consider((0.0, 0.0), Evaluation(value=2.0, descriptors=(0.1, 0.1)))
    assert cheaper.elites[0].value == 1.0
    assert dearer.elites[0].value == 2.0
    assert cheaper.n_not_better == 1 and dearer.n_not_better == 0


def test_archive_refuses_invalid_and_unplaceable_cases_instead_of_inventing_a_cell():
    archive = EliteArchive(space(4), sense="min")
    assert not archive.consider((0.0, 0.0), Evaluation(value=1.0, descriptors=(0.1, 0.1), valid=False))
    assert not archive.consider((0.0, 0.0), Evaluation(value=1.0, descriptors=None))
    assert not archive.consider((0.0, 0.0), Evaluation(value=1.0, descriptors=(5.0, 0.1)))
    assert not archive.consider((0.0, 0.0), Evaluation(value=float("nan"), descriptors=(0.1, 0.1)))
    assert archive.cells == 0
    assert (archive.n_invalid, archive.n_unplaceable) == (1, 3)


def test_archive_elites_come_out_in_cell_order_so_two_runs_compare_line_by_line():
    archive = EliteArchive(space(3), sense="min", design_names=("x", "y"))
    for value, cell in ((3.0, (2, 2)), (1.0, (0, 0)), (2.0, (1, 0))):
        archive.consider((0.1, 0.1), Evaluation(value=value, descriptors=(cell[0] / 3 + 0.05, cell[1] / 3 + 0.05)))
    assert [e.cell for e in archive.elites] == [(0, 0), (1, 0), (2, 2)]
    rows = archive.table()
    assert rows[0]["design"] == {"x": pytest.approx(0.1), "y": pytest.approx(0.1)}
    assert "a in [" in rows[0]["condition"]


def test_diversity_calls_a_one_point_archive_a_collapse():
    archive = EliteArchive(space(8), sense="min", name="collapsed")
    for _ in range(5):
        archive.consider((0.5, 0.5), Evaluation(value=1.0, descriptors=(0.51, 0.51)))
    report = archive.diversity()
    assert not report.ok
    assert report.cells == 1
    assert any("cells filled" in p for p in report.problems)
    with pytest.raises(CollapsedArchiveError):
        archive.assert_diverse()


def test_diversity_accepts_a_spread_out_map():
    archive = EliteArchive(space(8), sense="min")
    for i in range(8):
        archive.consider((i / 8, i / 8), Evaluation(value=float(i), descriptors=(i / 8 + 0.01, 1 - i / 8 - 0.01)))
    report = archive.diversity()
    assert report.ok, report.text()
    assert report.design_spread > 0.05


def test_archive_json_carries_the_axes_with_their_reasons():
    archive = EliteArchive(space(4), sense="min", name="map")
    archive.consider((0.2, 0.3), Evaluation(value=1.0, descriptors=(0.1, 0.1), error={"a": 0.5}))
    payload = json.loads(json.dumps(archive.to_json()))
    assert payload["axes"][0]["why"] == "first behaviour"
    assert payload["counters"]["considered"] == 1
    assert payload["elites"][0]["error"] == {"a": 0.5}


# -- the search ---------------------------------------------------------------------------------------------------

def test_latin_hypercube_gives_one_point_per_stratum_and_stays_in_the_box():
    box = np.array([[0.0, 1.0], [-2.0, 2.0], [5.0, 6.0]])
    X = latin_hypercube(20, box, np.random.default_rng(0))
    assert X.shape == (20, 3)
    assert np.all(X >= box[:, 0]) and np.all(X <= box[:, 1])
    for j in range(3):
        strata = np.floor(20 * (X[:, j] - box[j, 0]) / (box[j, 1] - box[j, 0])).astype(int)
        assert sorted(strata) == list(range(20))


def test_search_spends_the_budget_exactly_and_is_a_function_of_the_seed():
    first = QualityDiversitySearch(EliteArchive(space(8)), BOX, quadratic, QDConfig(budget=120, population=20,
                                                                                    seed=3)).run()
    second = QualityDiversitySearch(EliteArchive(space(8)), BOX, quadratic, QDConfig(budget=120, population=20,
                                                                                     seed=3)).run()
    assert first.evaluations == 120
    assert first.seed == 3
    assert [e.design.tolist() for e in first.archive.elites] == [e.design.tolist() for e in second.archive.elites]
    other = QualityDiversitySearch(EliteArchive(space(8)), BOX, quadratic, QDConfig(budget=120, population=20,
                                                                                     seed=4)).run()
    assert [e.design.tolist() for e in other.archive.elites] != [e.design.tolist() for e in first.archive.elites]


def test_search_keeps_its_designs_inside_the_box():
    result = QualityDiversitySearch(EliteArchive(space(6)), BOX, quadratic, QDConfig(budget=150, population=25,
                                                                                     seed=1, mutation_std=0.4,
                                                                                     immigrant_share=0.2)).run()
    X = np.stack([e.design for e in result.archive.elites])
    assert np.all(X >= 0.0) and np.all(X <= 1.0)
    assert result.generations >= 2


def test_the_line_mutation_is_reachable_and_the_run_still_covers_the_map():
    result = QualityDiversitySearch(EliteArchive(space(8)), BOX, quadratic,
                                    QDConfig(budget=200, population=25, seed=0, line_share=1.0)).run()
    assert result.archive.cells >= 10
    assert result.archive.diversity().design_spread > 0.05


def test_search_config_refuses_numbers_that_are_not_a_run():
    for bad in ({"budget": 0}, {"population": 1}, {"line_share": 1.5}, {"immigrant_share": -0.1},
                {"init_fraction": 2.0}, {"mutation_std": 0.0}, {"init": "sobol"}):
        with pytest.raises(ValueError):
            QDConfig(**bad)


def test_baselines_spend_the_same_budget_and_land_in_the_same_grid():
    for kind in ("random", "latin"):
        base = uniform_search(BOX, quadratic, space(8), budget=64, seed=0, kind=kind)
        assert base.evaluations == 64
        assert base.archive.cells > 0
        assert np.all(np.isfinite(base.best.value))
    with pytest.raises(ValueError):
        uniform_search(BOX, quadratic, space(8), budget=8, kind="sobol")


def test_every_sample_invalid_is_reported_not_silently_empty():
    def nothing_is_valid(row):
        return Evaluation(value=float("nan"), descriptors=None, valid=False)

    with pytest.raises(RuntimeError):
        uniform_search(BOX, nothing_is_valid, space(4), budget=8)


# -- the failure map ----------------------------------------------------------------------------------------------

def test_failure_region_lists_the_cells_above_the_tolerance_with_their_condition():
    archive = EliteArchive(space(4), sense="max", name="model error", design_names=("x", "y"))

    def error(row):
        e = 4.0 * float(row[0])                       # error grows towards the right of the map
        return Evaluation(value=e, descriptors=(float(row[0]), float(row[1])), error={"cl": e})

    need = sum(1 for i in range(4) for _ in range(8) if 4.0 * (i / 4 + 0.1) > 0.5)
    QualityDiversitySearch(archive, BOX, error, QDConfig(budget=256, population=32, seed=0)).run()
    region = FailureRegion(archive, tolerance=0.5)
    cells = region.cells()
    assert cells and len(cells) >= need // 2                       # the right-hand columns are all above it
    assert [c.error for c in cells] == sorted((c.error for c in cells), reverse=True)
    assert all(0.0 < c.error for c in cells)
    assert set(cells[0].design) == {"x", "y"}
    assert cells[0].condition["a"][1] > cells[0].condition["a"][0]
    assert 0.0 < region.failing_fraction() <= 1.0
    assert region.map_max == max(e.value for e in archive.elites)
    assert "above 50.0%" in region.text()


def test_failure_region_json_round_trips():
    archive = EliteArchive(space(4), sense="max", design_names=("x", "y"))
    archive.consider((0.9, 0.5), Evaluation(value=0.42, descriptors=(0.9, 0.5), error={"cl": 0.42}))
    payload = json.loads(json.dumps(FailureRegion(archive, tolerance=0.1).to_json()))
    assert payload["tolerance"] == 0.1
    assert payload["failing"][0]["channels"] == {"cl": 0.42}
    assert payload["archive"]["sense"] == "max"


def test_find_failure_region_searches_with_the_error_as_the_fitness():
    def error(row):
        return Evaluation(value=float(row[0]), descriptors=(float(row[0]), float(row[1])))

    region, result = find_failure_region(BOX, error, space(8), budget=120, seed=0, tolerance=0.5)
    assert result.evaluations == 120
    assert region.archive.sense == "max"
    assert region.map_max > 0.5
    assert region.cells()


# -- the picture --------------------------------------------------------------------------------------------------

def test_plot_archive_writes_a_file_and_needs_two_axes(tmp_path):
    pytest.importorskip("matplotlib")
    archive = EliteArchive(space(5), sense="max", name="error")
    for i in range(5):
        archive.consider((i / 5, 0.5), Evaluation(value=float(i), descriptors=(i / 5 + 0.05, 0.5)))
    out = plot_archive(archive, tmp_path / "map.png", tolerance=2.5, title="map", unit="rel")
    assert out.exists() and out.stat().st_size > 1000
    with pytest.raises(ValueError):
        plot_archive(EliteArchive(BehaviourSpace((BehaviourAxis("a", 0.0, 1.0),), bins=4)), tmp_path / "x.png")
