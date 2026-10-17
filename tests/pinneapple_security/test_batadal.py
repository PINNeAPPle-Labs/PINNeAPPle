"""BATADAL adapter (pinneapple_security.batadal): loader, grey-box components, scores. No network: the real data are
only used by the last test, and only when they are already in the cache."""
import math

import numpy as np
import pytest

from pinneapple_security import batadal, cps_bench

HEADER = "DATETIME, L_T1, F_PU1, S_PU1, P_J14, ATT_FLAG\n"


def test_load_parses_columns_hour_of_day_and_attack_flag(tmp_path):
    f = tmp_path / "mini.csv"
    f.write_text(HEADER + "04/07/16 00, 2.44, 93.63, 1.00, 28.2, -999\n"
                          "04/07/16 23, 2.50, 90.00, 0.00, 27.9, 1\n")
    D = batadal.load(f)
    np.testing.assert_allclose(D["L_T1"], [2.44, 2.50])
    np.testing.assert_allclose(D["S_PU1"], [1.0, 0.0])
    np.testing.assert_array_equal(D["hour"], [0, 23])
    np.testing.assert_array_equal(D["att_flag"], [-999, 1])
    assert "ATT_FLAG" not in D and "DATETIME" not in D


def test_episodes_are_the_runs_of_ones_and_unlabelled_rows_are_not_attacks():
    flag = np.array([-999, -999, 1, 1, -999, 1, 0, 1, 1])
    assert batadal.episodes(flag) == [(2, 4), (5, 6), (7, 9)]
    assert batadal.episodes(np.zeros(5)) == []


def _tank(n=24 * 40, area=100.0, seed=0):
    """A tank filled by one pump that switches on and off and drained by an hour-of-day demand profile."""
    rng = np.random.default_rng(seed)
    hour = np.arange(n) % 24
    demand = 40.0 + 30.0 * np.sin(2 * np.pi * np.arange(24) / 24)           # m^3/h
    flow = np.where(rng.random(n) < 0.5, 50.0, 0.0)                        # L/s
    level = np.empty(n)
    level[0] = 3.0
    for t in range(n - 1):
        level[t + 1] = level[t] + (batadal.FLOW_TO_M3_PER_H * flow[t] - demand[hour[t]]) / area
    return {"L": level, "F": flow, "hour": hour.astype(float)}, demand


def test_tank_balance_identifies_the_inflow_and_the_hourly_demand_and_sees_a_level_sensor_attack():
    D, demand = _tank()
    comp = batadal.TankBalance("L", ["F"], area=100.0).fit(D)
    assert comp.coef_[0] == pytest.approx(1.0, abs=1e-6)                    # the pump flow enters one for one
    np.testing.assert_allclose(comp.coef_[1:], -demand, atol=1e-6)         # demand = the 24 hourly terms
    r = comp.residual(D)
    assert np.isnan(r[0]) and np.nanmax(np.abs(r)) < 1e-6
    attacked = {**D, "L": D["L"] + np.where(np.arange(len(D["L"])) >= 500, 0.05, 0.0)}   # +5 cm from hour 500
    r = comp.residual(attacked)
    assert r[500] == pytest.approx(100.0 * 0.05, rel=1e-6)                  # the jump, in m^3
    assert abs(r[501]) < 1e-6                                               # then the level just stays offset


def test_pump_law_recovers_the_head_curve_and_is_silent_while_the_pump_is_off():
    rng = np.random.default_rng(0)
    n = 600
    suction = 20.0 + rng.random(n)
    gain = 20.0 + 20.0 * rng.random(n)
    on = (np.arange(n) % 3) != 0
    q = np.where(on, np.sqrt(2500.0 - 40.0 * gain), 0.0)
    D = {"q": q, "s": on.astype(float), "pd": suction + gain, "ps": suction}
    law = batadal.PumpLaw("q", "s", "pd", "ps").fit(D)
    assert law.coef == pytest.approx((2500.0, -40.0), rel=1e-6)
    r = law.residual(D)
    assert np.all(np.isnan(r[~on])) and np.nanmax(np.abs(r[on])) < 1e-6
    D["q"] = D["q"] * np.where(on, 1.1, 1.0)                                # flow meter reads 10 % high while running
    assert np.nanmean(law.residual(D)[on]) > 3.0


def test_time_to_detection_score():
    eps = [(10, 20), (30, 40)]
    assert batadal.ttd_score([10, 30], eps) == pytest.approx(1.0)           # caught at once
    assert batadal.ttd_score([12], eps) == pytest.approx(1 - (0.2 + 1.0) / 2)   # second attack never caught: ratio 1
    assert batadal.ttd_score([], eps) == pytest.approx(0.0)
    assert batadal.ttd_score([5, 25], eps) == pytest.approx(0.0)            # alarms outside the attacks do not count


def test_classification_score_uses_a_hold_and_a_clean_trace_or_the_unlabelled_hours():
    eps = [(10, 20)]
    # one alarm at 12 flags hours 12..16: half the attack hours; nothing flagged outside the attack
    assert batadal.classification_score([12], eps, 100, hold=5) == pytest.approx(0.5 * (0.5 + 1.0))
    # a clean trace of 40 hours with one alarm flags 5 of them
    assert batadal.classification_score([12], eps, 100, hold=5, clean_alarms=[3], n_clean=40) == pytest.approx(
        0.5 * (0.5 + 1 - 5 / 40))
    # without a clean trace, flagged hours outside the attack and its grace hours count against the TNR
    outside = 100 - (10 + 6)
    assert batadal.classification_score([12, 50], eps, 100, hold=5, grace=6) == pytest.approx(
        0.5 * (0.5 + 1 - 5 / outside))


def test_batadal_score_blends_equally_and_the_floor_exposes_a_detector_that_always_alarms():
    eps = [(10, 20), (30, 40)]
    storm = list(range(100))                                                # an alarm every hour of the attacked trace
    s = batadal.batadal_score(storm, [], eps, n_attack=100, n_clean=50, hold=12)
    assert s["S_TTD"] == pytest.approx(1.0)
    assert s["S"] == pytest.approx(0.5 * (s["S_TTD"] + s["S_CLF"]))
    assert s["S_CLF"] == pytest.approx(0.5 * (1.0 + 1.0))                   # the clean trace had no alarms
    assert s["S_CLF_floor"] == pytest.approx(0.5)                           # but it flags every unlabelled hour
    assert s["S_floor"] < s["S"]


def test_fetch_verifies_the_sha256_and_removes_a_file_that_does_not_match(tmp_path, monkeypatch):
    (tmp_path / "CTOWN.INP").write_bytes(b"not the file the checksum was recorded for")
    with pytest.raises(ValueError, match="SHA-256"):
        batadal.fetch(tmp_path, ["CTOWN.INP"])
    assert not (tmp_path / "CTOWN.INP").exists()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"bytes from a host that changed the file"

    seen = []
    monkeypatch.setattr(batadal.urllib.request, "urlopen", lambda url, timeout=0: seen.append(url) or FakeResponse())
    with pytest.raises(ValueError, match="SHA-256"):
        batadal.fetch(tmp_path, ["CTOWN.INP"])
    assert seen == ["https://www.batadal.net/data/CTOWN.INP"]
    assert not (tmp_path / "CTOWN.INP").exists()


def test_cache_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PINNEAPPLE_BATADAL_CACHE", str(tmp_path / "b"))
    assert batadal.cache_dir() == tmp_path / "b"


def test_ctown_topology_geometry_and_detector_layout():
    assert batadal.TANK_AREA["T1"] == pytest.approx(math.pi * 31.3 ** 2 / 4)
    assert len(batadal.SENSORS) == len(set(batadal.SENSORS)) == 31         # 7 levels, 12 flows, 12 pressures
    for ins, outs in batadal.TANK_FLOWS.values():
        assert set(ins + outs) <= set(batadal.SENSORS)
    det = batadal.ctown_detector()
    assert det.channels == [f"balance[T{i}]" for i in range(1, 8)] + ["law[PU2]", "law[PU10]"]


def test_baselines_read_the_sensor_columns_they_are_given():
    rng = np.random.default_rng(0)
    n = 2000
    base = np.sin(np.arange(n) / 20.0)
    D = {"a": base + 0.01 * rng.normal(size=n), "b": 2 * base + 0.01 * rng.normal(size=n),
         "c": -base + 0.01 * rng.normal(size=n)}
    sensors = ("a", "b", "c")
    spike = {k: v.copy() for k, v in D.items()}
    spike["b"][1500:1520] += 1.0                                            # a sensor that stops following the others
    for det in (cps_bench.PCAResidualDetector(sensors=sensors), cps_bench.ARResidualDetector(sensors=sensors)):
        det.fit(D)
        assert det.score(spike)[1500:1520].max() > 3 * det.score(D).max()


@pytest.mark.skipif(not (batadal.cache_dir() / "BATADAL_dataset04.csv").exists(),
                    reason="BATADAL data are not cached (batadal.fetch() downloads them)")
def test_real_data_have_the_documented_shape_and_the_benchmark_keeps_its_false_alarm_budget():
    d = batadal.fetch()
    clean, att = batadal.load(d / "BATADAL_dataset03.csv"), batadal.load(d / "BATADAL_dataset04.csv")
    assert len(clean["hour"]) == 8761 and np.all(clean["att_flag"] == 0)
    assert len(att["hour"]) == 4177 and len(batadal.episodes(att["att_flag"])) == 5
    out = batadal.run_benchmark(d, false_alarms_per_day=0.25, detectors={"physics": batadal.ctown_detector()})
    row = out["results"]["physics"]
    assert row["clean_false_alarms_per_day"] <= 3 * 0.25                    # calibrated budget, generous margin
    assert row["episodes"] == 5
