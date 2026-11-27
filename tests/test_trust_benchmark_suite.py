"""The trust benchmark (#349): four injected failure kinds through one API, and the ranking they produce.

A benchmark is only worth publishing if it can fail. So these tests check that each injection did what its row says
(the regime change is invisible in the inputs, the sensor faults move the sensor and not the plant, the
out-of-domain readings leave the box), that the ranking is the one the measurements give -- including a model that is
exact inside the box and wrong outside it, which is the case the whole suite exists to rank -- and that the record
published in the repository still carries the numbers the module docstring quotes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pinneapple_tools.benchmark_suite import trust

ROOT = Path(__file__).resolve().parent.parent

#: A tenth of the declared stream and a smaller bottom: enough readings to detect a change, cheap enough for a test.
SMALL = {"samples": {"train": 120, "validation": 100, "calibration": 150},
         "stream": {"n_clean": 40, "n_after": 40},
         "monitor": {"window": 10, "calibration_windows": 60}}


# ------------------------------------------------------------------ the plant and the injections
def test_the_plant_is_the_cited_closed_form_and_not_a_stand_in():
    from pinneapple_physics.closed_form.fin_array_conduction import (
        fin_array_base_temperature_c,
        fin_array_profile_params,
    )
    params = fin_array_profile_params(k_material_w_mk=205.0, h_conv_w_m2k=float(trust.FAN_H_REF), n_fins=12.0,
                                      fin_thickness_m=2e-3, fin_length_m=0.03, transverse_extent_m=0.09,
                                      spacing_extent_m=0.09, base_thickness_m=4e-3, power_w=40.0)
    expected = fin_array_base_temperature_c(params, 25.0, 40.0)
    assert trust.plant_response(40.0, 25.0, trust.FAN_RPM_REF) == pytest.approx([expected])
    # and the hidden parameters are real parameters of that model, not offsets added to its answer
    derated = trust.plant_response(40.0, 25.0, 1200.0, eta_fan=0.70)
    intact = trust.plant_response(40.0, 25.0, 1200.0)
    assert derated[0] - intact[0] == pytest.approx(19.660, abs=5e-3), \
        "a 30 % derating is worth 19.66 K at 1200 rpm and 40 W -- measured, and the kind's whole signal"


def test_the_four_kinds_each_have_injected_cases_and_the_same_ground_truth_step():
    kinds = {c.kind for c in trust.CASE_DECLARATIONS}
    assert kinds == set(trust.KINDS), "every kind of the issue has at least one injected stream"
    for case in trust.CASE_DECLARATIONS:
        readings = trust.build_case(case.id)
        assert len(readings) == 200 and readings.changed == 100, f"{case.id}: the ground truth is a declared step"
        assert np.all(np.isfinite(readings.x)) and np.all(np.isfinite(readings.y))


def test_a_hidden_regime_change_is_not_visible_in_the_inputs_and_moves_the_response():
    from scipy.stats import ks_2samp

    for case_id in ("fan_derate", "fins_blocked"):
        readings = trust.build_case(case_id)
        before, after = slice(0, 100), slice(100, 200)
        for column in range(readings.x.shape[1]):
            stat = ks_2samp(readings.x[before, column], readings.x[after, column]).statistic
            assert stat < 0.20, (f"{case_id}: input {trust.INPUT_NAMES[column]} moved (KS {stat:.3f}); a regime "
                                 "change has to be hidden from anything that reads the inputs")
        shift = np.abs(readings.y[after] - readings.y[before]).mean()
        assert shift > 5.0, f"{case_id}: the response moved {shift:.2f} K, which is what the model has to notice"
        assert np.array_equal(readings.measured, readings.y), "a regime change is not a sensor fault"


def test_the_sensor_faults_move_the_sensor_and_leave_the_plant_alone():
    for case_id in ("power_spike", "fan_stuck", "ambient_bias"):
        readings = trust.build_case(case_id)
        faulted = np.abs(readings.x[100:] - _clean_inputs(case_id)[100:]).max()
        assert faulted > 0.0, f"{case_id}: the input the model is fed is not the operating condition any more"
        assert np.array_equal(readings.measured, readings.y), f"{case_id}: the response sensor is not the fault"
    drift = trust.build_case("thermocouple_drift")
    gap = drift.measured[100:] - drift.y[100:]
    assert gap.min() == pytest.approx(0.0) and gap.max() > 30.0, "the response sensor ramps away from the plant"
    assert np.array_equal(drift.x, _clean_inputs("thermocouple_drift")), "and the model's inputs stay clean"


def _clean_inputs(case_id: str) -> np.ndarray:
    """The stream a case starts from: the same draw, before its injection touched anything."""
    p = trust.resolve_protocol(None)
    return trust._in_domain_readings(int(p["stream"]["n_clean"]), int(p["stream"]["n_after"]),
                                     trust._seed_of(p["stream"]["seed"], case_id))[0]


def test_the_out_of_domain_cases_leave_the_box_the_candidates_were_fitted_on():
    box = trust.BOX
    for case_id, name, low, high in (("fan_below_range", "fan_rpm", 480.0, 700.0),
                                     ("power_above_range", "power_w", 90.0, 140.0)):
        readings = trust.build_case(case_id)
        column = trust.INPUT_NAMES.index(name)
        after = readings.x[100:, column]
        inside = (after >= box[name][0]) & (after <= box[name][1])
        assert not inside.any(), f"{case_id}: every reading after the change is outside the {name} range"
        assert after.min() >= low - 1e-9 and after.max() <= high + 1e-9
        assert readings.y[100:].size == 100 and np.all(np.isfinite(readings.y[100:])), \
            "the plant is fine outside the box: it is the model that is out of its domain, not the physics"


def test_a_slow_drift_ramps_instead_of_stepping():
    readings = trust.build_case("dust_ramp")
    intact = trust.plant_response(readings.x[:, 0], readings.x[:, 1], readings.x[:, 2])
    effect = readings.y[100:] - intact[100:]
    assert effect[0] < 0.25 * effect[-1], "the first reading under the ramp is a fraction of the last: it is a ramp"
    assert effect[:10].mean() < 0.3 * effect[-10:].mean()
    quarters = np.array_split(effect, 4)
    means = np.array([q.mean() for q in quarters])
    assert np.all(np.diff(means) > 0.0), f"the ramp's effect grows quarter by quarter, measured {means}"
    assert effect[-1] > 5.0
    slower = trust.build_case("slow_fouling")
    intact = trust.plant_response(slower.x[:, 0], slower.x[:, 1], slower.x[:, 2])
    assert (slower.y[100:] - intact[100:])[-1] < 0.5 * effect[-1], "the second ramp is the near-threshold one"


# ------------------------------------------------------------------ one API, four kinds
def _clamped(predict_inside):
    """A model that is exact where it was fitted and stops extrapolating: it evaluates the plant at the box's edge."""
    low = np.array([trust.BOX[name][0] for name in trust.INPUT_NAMES])
    high = np.array([trust.BOX[name][1] for name in trust.INPUT_NAMES])

    def predict(X: np.ndarray) -> np.ndarray:
        Z = np.clip(np.asarray(X, dtype=np.float64), low, high)
        return predict_inside(Z)
    return predict


def test_the_protocol_catches_a_model_that_is_good_in_domain_and_bad_outside_it():
    """The case the suite exists for: same model, same in-domain accuracy, and only one of them extrapolates."""
    truth = lambda X: trust.plant_response(X[:, 0], X[:, 1], X[:, 2])   # noqa: E731
    candidates = [
        trust.Candidate(id="extrapolating", title="extrapolating", family="closed form", predict=truth),
        trust.Candidate(id="clamped", title="clamped at the edge", family="closed form",
                        predict=_clamped(truth)),
    ]
    result = trust.run_suite(candidates, protocol=SMALL, kinds=["out_of_domain"])
    rows = {r["id"]: r for r in result.kinds["out_of_domain"]}
    assert rows["clamped"]["error_before"] < 0.02 and rows["extrapolating"]["error_before"] < 0.02, \
        "both are right inside the box -- that is the premise of the test"
    assert rows["clamped"]["error_after"] > 0.25, "clamping is a large error once the readings leave the box"
    assert rows["extrapolating"]["error_after"] < 0.05, "the model that extrapolates is still right"
    assert rows["clamped"]["rank"] > rows["extrapolating"]["rank"], "and the suite ranks it behind, by that column"
    assert rows["clamped"]["error_after"] > rows["extrapolating"]["error_after"]


def test_all_four_kinds_run_through_the_one_api_and_return_the_declared_columns():
    truth = lambda X: trust.plant_response(X[:, 0], X[:, 1], X[:, 2])   # noqa: E731
    result = trust.run_suite([trust.Candidate(id="exact", title="exact", predict=truth)], protocol=SMALL)
    assert list(result.kinds) == list(trust.KINDS), "the four kinds come back from one call, in the declared order"
    columns = {"missed", "detect_delay", "false_alarms_per_1000", "error_before", "error_after",
               "error_exit_delay", "coverage_before", "coverage_after", "alarmed_checks", "rank"}
    for kind, rows in result.kinds.items():
        assert len(rows) == 1
        assert columns <= set(rows[0]), f"{kind}: a declared column is missing"
        assert rows[0]["rank"] == 1
    assert len(result.cases) == len(trust.CASE_DECLARATIONS)
    assert {c for c in result.cases} == {c.id for c in trust.CASE_DECLARATIONS}
    for kind, rows in result.kinds.items():
        for row in rows:
            assert row["cases"] == sum(1 for c in trust.CASE_DECLARATIONS if c.kind == kind)


def test_an_exact_model_is_not_wrong_and_its_monitor_says_so_on_the_hidden_regime_change():
    """The regime change is hidden from the inputs, so the model *is* wrong -- and only the labeled check sees it."""
    truth = lambda X: trust.plant_response(X[:, 0], X[:, 1], X[:, 2])   # noqa: E731
    result = trust.run_suite([trust.Candidate(id="exact", title="exact", predict=truth)], protocol=SMALL,
                             kinds=["regime_change"])
    row = result.kinds["regime_change"][0]
    assert row["error_before"] < 0.02, "inside the box the exact model is exact"
    assert row["error_after"] > 0.5, "and the hidden change is not something the inputs could have told it about"
    assert "error_drift" in row["alarmed_checks"], "the labeled check is what catches a change the inputs cannot hide"
    assert row["missed"] == 0 and row["detect_delay"] < 40


def test_a_candidate_without_an_uncertainty_reports_no_coverage_instead_of_a_number():
    truth = lambda X: trust.plant_response(X[:, 0], X[:, 1], X[:, 2])   # noqa: E731
    banded = lambda X: (truth(X).reshape(-1, 1), np.full((len(X), 1), 1.0))   # noqa: E731
    result = trust.run_suite([trust.Candidate(id="bare", title="bare", predict=truth),
                              trust.Candidate(id="banded", title="banded", predict=banded)],
                             protocol=SMALL, kinds=["slow_drift"])
    rows = {r["id"]: r for r in result.kinds["slow_drift"]}
    assert rows["bare"]["coverage_before"] is None and rows["bare"]["coverage_after"] is None
    assert rows["banded"]["coverage_before"] is not None
    assert rows["banded"]["coverage_after"] < rows["banded"]["coverage_before"], \
        "a fixed band stops covering when the plant moves"


def test_the_streams_and_the_in_domain_sample_are_deterministic_in_the_declared_seeds():
    first, second = trust.build_case("dust_ramp"), trust.build_case("dust_ramp")
    assert np.array_equal(first.x, second.x) and np.array_equal(first.y, second.y)
    X1, y1 = trust.in_domain_data()
    X2, y2 = trust.in_domain_data()
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert not np.array_equal(trust.build_case("dust_ramp", protocol=SMALL).x, first.x), \
        "a different stream length is a different stream"


# ------------------------------------------------------------------ the ranking
def _row(**kwargs):
    base = {"id": "m", "title": "m", "kind": "slow_drift", "missed": 0, "detect_delay": 10.0,
            "false_alarms_per_1000": 10.0, "error_after": 0.5}
    return {**base, **kwargs}


def test_a_missed_case_outranks_every_other_column():
    rows = [_row(id="fast", detect_delay=1.0), _row(id="blind", missed=1, detect_delay=0.0, error_after=0.01)]
    assert [r["id"] for r in trust.ranked(rows)] == ["fast", "blind"], \
        "a model that never noticed is the failure this suite exists to find"


def test_a_model_that_is_far_more_wrong_is_ranked_behind_even_if_it_alarmed_first():
    rows = [_row(id="wrong", detect_delay=0.0, error_after=1.944), _row(id="right", detect_delay=0.0,
                                                                      error_after=0.125)]
    assert [r["id"] for r in trust.ranked(rows)] == ["right", "wrong"]


def test_a_difference_under_the_declared_resolution_is_not_a_difference():
    resolution = trust.PROTOCOL["ranking"]["resolution"]["error_after"]
    near = _row(id="a", error_after=0.8332, detect_delay=4.5)
    other = _row(id="b", error_after=0.8259, detect_delay=6.0)
    assert (other["error_after"] - near["error_after"]) / near["error_after"] < resolution
    assert [r["id"] for r in trust.ranked([near, other])] == ["a", "b"], \
        "the errors tie at this resolution, so the delay decides -- not the seventh decimal"
    far = _row(id="c", error_after=0.30, detect_delay=99.0)
    assert [r["id"] for r in trust.ranked([near, far])] == ["c", "a"]


def test_false_alarms_break_the_tie_the_delay_cannot():
    quiet = _row(id="quiet", false_alarms_per_1000=0.0)
    noisy = _row(id="noisy", false_alarms_per_1000=900.0)
    assert [r["id"] for r in trust.ranked([quiet, noisy])] == ["quiet", "noisy"]


def test_a_missed_case_is_censored_and_counted_instead_of_dropped():
    p = trust.resolve_protocol(SMALL)
    caught = {"detected_at": 110, "detect_delay": 10, "error_exit_delay": 12, "false_alarms": 1,
              "n_clean_checked": 30, "error_before": 0.0, "error_after": 0.5, "error_after_max": 0.9,
              "coverage_before": None, "coverage_after": None, "alarmed_checks": {"error_drift": 3}}
    missed = {**caught, "detected_at": None, "detect_delay": None}
    row = trust._aggregate("slow_drift", {"a": caught, "b": missed}, p)
    assert row["missed"] == 1 and row["missed_cases"] == ["b"] and row["detected"] == "1/2"
    assert row["detect_delay"] == pytest.approx((10 + p["stream"]["n_after"]) / 2), \
        "a miss is censored at the readings that remained, and only the missed column forgives it"


def test_the_leaderboard_is_the_mean_of_the_ranks_and_a_total_order():
    kinds = {"regime_change": [{"id": "a", "title": "a", "kind": "regime_change", "rank": 1, "error_after": 0.2},
                               {"id": "b", "title": "b", "kind": "regime_change", "rank": 2, "error_after": 0.4}],
             "slow_drift": [{"id": "b", "title": "b", "kind": "slow_drift", "rank": 1, "error_after": 0.1},
                            {"id": "a", "title": "a", "kind": "slow_drift", "rank": 2, "error_after": 0.2}]}
    board = trust.leaderboard(kinds)
    assert [r["id"] for r in board] == ["a", "b"], "equal mean rank goes to the smaller mean error after the change"
    assert board[0]["mean_rank"] == pytest.approx(1.5) and board[0]["rank"] == 1
    assert board[0]["worst_kind"] == "slow_drift"


def test_a_candidate_that_holds_the_same_rank_everywhere_has_no_worst_kind():
    kinds = {kind: [{"id": "a", "title": "a", "kind": kind, "rank": 1, "error_after": 0.1}] for kind in trust.KINDS}
    board = trust.leaderboard(kinds)
    assert board[0]["worst_kind"] is None and board[0]["same_rank_on_every_kind"] is True


# ------------------------------------------------------------------ the suite as run, end to end
@pytest.fixture(scope="module")
def small_run():
    truth = lambda X: trust.plant_response(X[:, 0], X[:, 1], X[:, 2])   # noqa: E731
    candidates = [trust.Candidate(id="exact", title="exact", family="closed form", predict=truth),
                  trust.Candidate(id="clamped", title="clamped", family="closed form", predict=_clamped(truth))]
    return trust.run_suite(candidates, protocol=SMALL)


def test_the_run_carries_its_ground_truth_and_its_protocol(small_run):
    record = small_run.to_dict()
    assert record["schema"] == trust.SCHEMA
    assert record["protocol"]["declared"] is False, "a reduced protocol says so, and never becomes the record"
    assert record["ground_truth"]["fan_derate"]["changed"] == 40 and record["ground_truth"]["fan_derate"]["readings"] == 80
    assert set(record["ground_truth"]) == {c.id for c in trust.CASE_DECLARATIONS}
    assert record["scale_k"] > 10.0, "the error is normalised by the spread of the in-domain responses"


def test_the_page_is_generated_from_the_record(small_run):
    record = small_run.to_dict()
    page = trust.markdown(record)
    assert "| rank | candidate | mean rank |" in page
    assert "the same rank on every kind" in page or "(#" in page
    for kind in trust.KINDS:
        assert kind.replace("_", " ") in page
    for case in trust.CASE_DECLARATIONS:
        assert f"`{case.id}`" in page


def test_the_gate_refuses_a_record_whose_order_is_not_the_measurements(tmp_path, small_run):
    record = small_run.to_dict()
    record["protocol"]["declared"] = True
    trust.write(record, root=tmp_path)
    assert trust.check(tmp_path) == []
    record["kinds"]["out_of_domain"] = list(reversed(record["kinds"]["out_of_domain"]))
    trust.write(record, root=tmp_path)
    problems = trust.check(tmp_path)
    assert any("published order is not the one the measurements give" in p for p in problems)


def test_the_gate_refuses_a_page_that_drifted_from_the_record(tmp_path, small_run):
    record = small_run.to_dict()
    record["protocol"]["declared"] = True
    trust.write(record, root=tmp_path)
    (tmp_path / trust.PAGE).write_text("a hand-written table\n", encoding="utf-8")
    assert any("is not what the record generates" in p for p in trust.check(tmp_path))


def test_the_gate_refuses_a_record_from_a_reduced_protocol(tmp_path, small_run):
    trust.write(small_run.to_dict(), root=tmp_path)
    assert any("reduced protocol" in p for p in trust.check(tmp_path))


# ------------------------------------------------------------------ the numbers published in the repository
@pytest.fixture(scope="module")
def published():
    path = ROOT / trust.RESULTS
    if not path.is_file():                                  # pragma: no cover - the record ships with the repo
        pytest.skip(f"{trust.RESULTS.as_posix()} is not in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_published_record_is_declared_recorded_and_in_sync():
    assert trust.check(ROOT) == [], "the published record and page are the ones the measurements give"


def test_the_published_leaderboard_is_the_one_the_docstring_quotes(published):
    board = {row["id"]: row for row in published["leaderboard"]}
    assert [row["id"] for row in published["leaderboard"]] == ["gp", "ridge", "ridge_band", "gbt"]
    assert board["gp"]["mean_rank"] == pytest.approx(1.0)
    assert board["ridge"]["mean_rank"] == pytest.approx(2.0)
    assert board["ridge_band"]["mean_rank"] == pytest.approx(3.0)
    assert board["gbt"]["mean_rank"] == pytest.approx(4.0)


def test_the_published_headline_numbers_are_the_ones_the_module_quotes(published):
    """Every number in the module docstring's "Measured, not estimated" section, read back from the record."""
    kinds = published["kinds"]

    def row(kind, cid):
        return next(r for r in kinds[kind] if r["id"] == cid)

    # in domain: the GP is exact, the ridge is a few per cent of the spread, the trees are an order worse
    assert 1.0e-5 < row("regime_change", "gp")["error_before"] < 3.0e-5
    assert row("regime_change", "ridge")["error_before"] == pytest.approx(0.036, abs=5e-4)
    assert row("regime_change", "gbt")["error_before"] == pytest.approx(0.118, abs=5e-4)
    assert row("out_of_domain", "gbt")["error_before"] == pytest.approx(0.130, abs=5e-4)

    # the detection delays the docstring claims, kind by kind
    assert [row("regime_change", c)["detect_delay"] for c in ("gp", "ridge", "ridge_band", "gbt")] == \
        pytest.approx([4.5, 5.5, 5.5, 6.5])
    assert [row("sensor_fault", c)["detect_delay"] for c in ("gp", "gbt")] == pytest.approx([4.0, 14.75])
    assert [row("slow_drift", c)["detect_delay"] for c in ("gp", "ridge")] == pytest.approx([6.5, 20.5])
    assert row("out_of_domain", "gp")["detect_delay"] == pytest.approx(0.0)
    assert row("out_of_domain", "gbt")["detect_delay"] == pytest.approx(0.0)
    assert "envelope" in row("out_of_domain", "gbt")["alarmed_checks"]

    # the factor the docstring quotes between the best and the worst error outside the box
    assert row("out_of_domain", "gbt")["error_after"] / row("out_of_domain", "gp")["error_after"] == \
        pytest.approx(15.6, abs=0.1)
    # the extreme false-alarm rates it quotes on the two sides
    assert row("out_of_domain", "gbt")["false_alarms_per_1000"] == pytest.approx(0.0)
    assert row("out_of_domain", "gp")["false_alarms_per_1000"] == pytest.approx(98.8, abs=0.1)
    assert max(r["false_alarms_per_1000"] for rows in kinds.values() for r in rows) == pytest.approx(216.0, abs=0.1)


def test_the_published_numbers_per_kind_are_the_measured_ones(published):
    kinds = published["kinds"]

    def row(kind, cid):
        return next(r for r in kinds[kind] if r["id"] == cid)

    # the hidden regime change: every candidate is wrong by almost exactly the same amount (0.8333 to 0.8345, well
    # inside the declared resolution), and the monitor catches it within a handful of readings, from the labels --
    # nothing else sees it
    assert row("regime_change", "gp")["detect_delay"] == pytest.approx(4.5)
    assert row("regime_change", "gbt")["error_after"] == pytest.approx(0.8333, abs=1e-3)
    assert row("regime_change", "gp")["error_after"] == pytest.approx(0.8345, abs=1e-3)
    assert row("regime_change", "gp")["missed"] == 0
    # and the declared rule is what ranks it: gbt has the fewest false alarms of the four (86.4 per 1000 against
    # 216.0) and is still last, because the errors tie at the resolution and its delay is two readings worse
    assert row("regime_change", "gbt")["false_alarms_per_1000"] < row("regime_change", "gp")["false_alarms_per_1000"]
    assert row("regime_change", "gbt")["rank"] > row("regime_change", "gp")["rank"]

    # out of domain: an order of magnitude of difference in what the model gets wrong, and the ranking follows it
    assert row("out_of_domain", "gp")["error_after"] == pytest.approx(0.1246, abs=1e-3)
    assert row("out_of_domain", "gbt")["error_after"] == pytest.approx(1.9440, abs=1e-3)
    assert row("out_of_domain", "gp")["rank"] == 1 and row("out_of_domain", "gbt")["rank"] == 4

    # slow drift: one candidate never notices one of the two ramps, and the missed column says so
    assert row("slow_drift", "gbt")["missed"] == 1
    assert row("slow_drift", "gbt")["error_exit_delay"] == pytest.approx(11.0)
    assert row("slow_drift", "gp")["detect_delay"] == pytest.approx(6.5)

    # the fixed band covers in domain and stops covering under a regime change
    band = row("regime_change", "ridge_band")
    assert band["coverage_before"] >= 0.94 and band["coverage_after"] <= 0.03


def test_the_published_cases_are_the_injected_streams_they_are_named_after(published):
    assert published["ground_truth"]["fan_derate"]["changed"] == 100
    assert set(published["cases"]) == {c.id for c in trust.CASE_DECLARATIONS}

    def case(name, cid):
        return published["cases"][name][cid]

    # the model is right and the sensor is not: the error against the plant is zero and the monitor still fires
    assert case("thermocouple_drift", "gp")["error_after"] < 0.01
    assert case("thermocouple_drift", "gp")["detect_delay"] is not None
    # the exact model never crosses the tolerance in the power range it was fitted on, and it does not claim to have
    assert case("power_above_range", "gp")["error_exit_delay"] is None
    assert case("fan_below_range", "gbt")["error_after"] > 1.0


def test_the_published_protocol_declares_what_a_reproduction_has_to_keep(published):
    protocol = published["protocol"]
    for field in trust.REQUIRED_PROTOCOL_FIELDS:
        assert field in protocol, f"the record does not carry {field!r}"
    assert protocol["declared"] is True
    assert protocol["ranking"]["resolution"]["error_after"] > 0
    assert set(protocol["box"]) == set(trust.INPUT_NAMES)
