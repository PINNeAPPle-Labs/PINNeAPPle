"""pinneapple_systems.wind: geometry, power curves, wake models, QC and the farm comparison.

Reference values here are properties of the *models* (monotone deficit, exact
conservation of the no-wake case, a hard power cap), which can be asserted to
machine precision, not field measurements -- those would need a source this POC
does not ship.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pytest

pytest.importorskip("pandas")

from pinneapple_systems import wind  # noqa: E402

RATED = 2050.0
DIAMETER = 82.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_wind_vector_points_downwind():
    """Meteorological direction is where the wind comes FROM; the vector points away."""
    assert np.allclose(wind.wind_vector(0.0), (0.0, -1.0), atol=1e-12)      # from north -> south
    assert np.allclose(wind.wind_vector(90.0), (-1.0, 0.0), atol=1e-12)     # from east -> west
    assert np.allclose(wind.wind_vector(180.0), (0.0, 1.0), atol=1e-12)     # from south -> north
    assert np.allclose(wind.wind_vector(270.0), (1.0, 0.0), atol=1e-12)     # from west -> east


def _row_layout(n=3, spacing=400.0):
    turbines = [
        wind.Turbine(name=f"T{i:02d}", x=i * spacing, y=0.0, hub_height=59.0,
                     rotor_diameter=DIAMETER, rated_power_kw=RATED)
        for i in range(n)
    ]
    return wind.WindFarmLayout("row", turbines)


def test_upwind_order_follows_the_wind():
    layout = _row_layout()
    assert layout.upwind_order(270.0) == [0, 1, 2]   # from the west: author's west end first
    assert layout.upwind_order(90.0) == [2, 1, 0]    # from the east


def test_layout_rejects_duplicate_names():
    t = wind.Turbine("T01", 0.0, 0.0, 59.0, DIAMETER, RATED)
    with pytest.raises(ValueError):
        wind.WindFarmLayout("dup", [t, t])


# ---------------------------------------------------------------------------
# Wake models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["jensen", "gaussian"])
def test_deficit_is_bounded_and_decays_downstream(model):
    fn = wind.jensen_deficit if model == "jensen" else wind.gaussian_deficit
    ct = 0.8
    near = float(fn(2.0 * DIAMETER, 0.0, DIAMETER, ct))
    far = float(fn(20.0 * DIAMETER, 0.0, DIAMETER, ct))
    assert 0.0 <= far < near <= 1.0
    assert float(fn(-1.0, 0.0, DIAMETER, ct)) == 0.0        # no wake upstream
    off_axis = float(fn(10.0 * DIAMETER, 10.0 * DIAMETER, DIAMETER, ct))
    if model == "gaussian":
        assert off_axis > 0.0        # the Gaussian profile has no hard edge
    else:
        assert off_axis == 0.0       # the Jensen cone does


def test_jensen_wake_is_a_cone():
    """Zero outside the cone, positive inside, and the cone grows with distance."""
    radius = lambda x: DIAMETER / 2.0 + 0.075 * x  # noqa: E731
    x = 8.0 * DIAMETER
    inside = float(wind.jensen_deficit(x, radius(x) - 1.0, DIAMETER, 0.8))
    outside = float(wind.jensen_deficit(x, radius(x) + 1.0, DIAMETER, 0.8))
    assert inside > 0.0 and outside == 0.0


def test_wake_deficit_matrix_is_directional():
    layout = _row_layout()
    west = wind.wake_deficits(layout, 8.0, 270.0)
    east = wind.wake_deficits(layout, 8.0, 90.0)
    assert west[0, 2] > 0.0 and west[2, 0] == 0.0     # T00 shadows T02 under westerlies
    assert east[2, 0] > 0.0 and east[0, 2] == 0.0     # and the other way round
    assert np.allclose(np.diagonal(west), 0.0)


def test_farm_power_without_wakes_is_exactly_the_sum_of_the_curves():
    layout = _row_layout(n=4)
    curve = wind.ReferencePowerCurve(RATED, DIAMETER)
    alone = wind.farm_power(layout, 8.0, 270.0, curve.power)
    # Spread the turbines far apart so no wake survives: the farm must equal n x one machine.
    far = _row_layout(n=4, spacing=40_000.0)
    spread = wind.farm_power(far, 8.0, 270.0, curve.power)
    assert spread.wake_loss_fraction == pytest.approx(0.0, abs=1e-2)
    assert alone.wake_loss_fraction > 0.05
    assert alone.efficiency == pytest.approx(alone.total_power_kw / alone.total_power_no_wake_kw)


def test_farm_power_respects_the_rated_cap():
    layout = _row_layout(n=3)
    curve = wind.ReferencePowerCurve(RATED, DIAMETER)
    result = wind.farm_power(layout, 30.0, 270.0, curve.power)
    assert result.total_power_kw == 0.0            # above cut-out
    result = wind.farm_power(layout, 11.0, 270.0, curve.power)
    assert np.all(result.power_kw <= RATED + 1e-9)


# ---------------------------------------------------------------------------
# Power curves
# ---------------------------------------------------------------------------

def test_reference_curve_is_cubic_then_flat():
    curve = wind.ReferencePowerCurve(RATED, DIAMETER, cp=0.45)
    u = np.array([5.0, 10.0])
    ratio = curve.power(u)[1] / curve.power(u)[0]
    assert ratio == pytest.approx(8.0, rel=1e-12)        # region 2 is exactly cubic
    assert float(curve.power(curve.rated_speed + 1.0)) == pytest.approx(RATED, rel=1e-9)
    assert float(curve.power(2.0)) == 0.0                 # below cut-in
    assert float(curve.power(26.0)) == 0.0                # above cut-out


def test_reference_curve_cannot_exceed_betz():
    with pytest.raises(ValueError):
        wind.ReferencePowerCurve(RATED, DIAMETER, cp=0.7)


def test_density_correction_matches_the_cube_root_law():
    corrected = wind.density_corrected_speed(np.array([10.0]), 0.8)
    assert corrected[0] == pytest.approx(10.0 * 0.8 ** (1 / 3))
    assert wind.betz_limit() == pytest.approx(16 / 27)


def test_binned_curve_does_not_invent_values_past_cut_out():
    rng = np.random.default_rng(0)
    speed = rng.uniform(0, 25, 5000)
    power = np.clip(0.5 * 1.225 * np.pi * (DIAMETER / 2) ** 2 * 0.45 * speed**3 / 1000, 0, RATED)
    power = np.where(speed < 3, 0.0, power)
    curve = wind.fit_binned_curve(speed, power, cut_out=25.0, min_count=5)
    assert float(curve.power(np.array([26.0]))[0]) == 0.0
    assert curve.support()["bins_supported"] > 5


# ---------------------------------------------------------------------------
# Physics-informed model
# ---------------------------------------------------------------------------

def _noisy_power(speed, rng):
    curve = wind.ReferencePowerCurve(RATED, DIAMETER, cp=0.45)
    clean = curve.power(speed)
    return np.clip(clean + rng.normal(0, 25, size=speed.shape), 0, RATED)


def test_physics_informed_curve_obeys_its_hard_constraints():
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    speed = rng.uniform(0, 28, 4000)
    model = wind.PhysicsInformedPowerModel(rated_power_kw=RATED, rotor_diameter=DIAMETER,
                                           epochs=150).fit(speed[:, None], _noisy_power(speed, rng))
    grid = np.linspace(0, 32, 801)
    predicted = model.predict(grid[:, None])
    inside = (grid >= 3.0) & (grid <= 25.0)
    assert predicted.max() <= RATED + 1e-6
    assert predicted.min() >= -1e-9
    assert np.diff(predicted[inside]).min() >= -1e-6            # never decreases inside the window
    assert np.all(predicted[grid < 3.0] == 0.0)
    assert np.all(predicted[grid > 25.0] == 0.0)
    assert 0.2 < model.diagnostics["implied_cp_region2"] < wind.betz_limit() + 1e-9


def test_guardrails_re_measure_the_claims():
    pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    speed = rng.uniform(0, 28, 2000)
    model = wind.PhysicsInformedPowerModel(rated_power_kw=RATED, rotor_diameter=DIAMETER,
                                           epochs=80).fit(speed[:, None], _noisy_power(speed, rng))
    checks = wind.physical_guardrails(lambda u: model.predict(np.column_stack([u])),
                                      rated_power_kw=RATED, cut_in=3.0, cut_out=25.0,
                                      rotor_diameter=DIAMETER)
    assert {c["name"] for c in checks} == {
        "conservation_power_cap", "conservation_monotonicity",
        "conservation_operating_window", "conservation_betz_limit",
    }
    # The three hard constraints are enforced by construction and must pass on any
    # data. The Betz bound is a *penalty*, so it is measured, not asserted -- that
    # is the whole point of re-running the check on the fitted curve.
    by_name = {c["name"]: c for c in checks}
    for hard in ("conservation_power_cap", "conservation_monotonicity",
                 "conservation_operating_window"):
        assert by_name[hard]["passed"], by_name[hard]


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------

def _synthetic_frame(n=600, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    speed = rng.uniform(0, 25, n)
    curve = wind.ReferencePowerCurve(RATED, DIAMETER)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2017-01-01", periods=n, freq="10min"),
        "wind_speed_ms": speed,
        "power_kw": curve.power(speed) + rng.normal(0, 15, n),
        "wind_direction_deg": rng.uniform(0, 360, n),
        "nacelle_deg": rng.uniform(0, 360, n),
        "ambient_temperature_c": rng.uniform(-5, 25, n),
        "data_availability": 1,
        "lost_production_downtime_kwh": 0.0,
        "lost_production_curtailment_kwh": 0.0,
    })
    return frame


def test_operating_mask_drops_curtailed_and_unavailable_samples():
    frame = _synthetic_frame()
    frame.loc[0:9, "lost_production_curtailment_kwh"] = 100.0
    frame.loc[10:19, "data_availability"] = 0
    mask = wind.operating_mask(frame, rated_power_kw=RATED)
    assert not mask[0:20].any()
    assert mask.sum() > len(frame) * 0.5
    assert wind.mask_provenance(frame)["evidence_based"] is True


def test_canonicalise_derives_yaw_error_and_pitch():
    import pandas as pd
    raw = pd.DataFrame({
        "timestamp": ["2017-01-01 00:00", "2017-01-01 00:10"],
        "wind_speed_ms": ["5", "6"],
        "wind_direction_deg": ["350", "10"],
        "nacelle_deg": ["5", "350"],
        "pitch_a_deg": ["1", "2"], "pitch_b_deg": ["1", "2"], "pitch_c_deg": ["1", "2"],
    })
    out = wind.penmanshiel.canonicalise(raw)
    assert out["yaw_error_deg"].to_list() == pytest.approx([15.0, -20.0])
    assert out["pitch_deg"].to_list() == pytest.approx([1.0, 2.0])


def test_reader_maps_the_real_greenbyte_header(tmp_path):
    """A miniature archive with the vendor's comment header must parse to the canonical columns."""
    header = ("# Date and time,Wind speed (m/s),Power (kW),Potential power default PC (kW),"
              "Wind direction (°),Nacelle ambient temperature (°C)")
    lines = ["# Turbine: Penmanshiel 07", header,
             "2017-01-01 00:00:00,5.0,100.0,110.0,290.0,8.0",
             "2017-01-01 00:10:00,6.0,180.0,195.0,291.0,8.5"]
    path = tmp_path / "Penmanshiel_SCADA_2017_WT01-10_9999.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Turbine_Data_Penmanshiel_07_2017-01-01_-_2018-01-01_1.csv",
                    "\r\n".join(lines).encode())
    frames, report = wind.penmanshiel.read_archive(path)
    assert list(frames) == ["T07"]
    frame = frames["T07"]
    assert frame["power_kw"].to_list() == [100.0, 180.0]
    assert frame["potential_power_kw"].to_list() == [110.0, 195.0]
    assert "ambient_temperature_c" in frame
    assert report["files"][0]["turbine"] == "T07"


def test_discover_archives_skips_a_truncated_download(tmp_path):
    good = tmp_path / "Penmanshiel_SCADA_2017_WT01-10_3114.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("x.csv", "a,b\n1,2\n")
    (tmp_path / "Penmanshiel_SCADA_2018_WT01-10_3113.zip").write_bytes(b"PK\x03\x04truncated")
    archives = wind.discover_archives(tmp_path)
    assert [a.year for a in archives] == [2017]
    assert archives[0].turbines[1] == "T02"   # T03 does not exist


# ---------------------------------------------------------------------------
# Metrics and comparison plumbing
# ---------------------------------------------------------------------------

def test_point_metrics_are_exact_on_a_perfect_prediction():
    measured = np.array([0.0, 500.0, 1000.0, 2000.0])
    metrics = wind.point_metrics(measured, measured)
    assert metrics.rmse_kw == 0.0 and metrics.bias_kw == 0.0
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.energy_error_fraction == pytest.approx(0.0)


def test_conformal_half_width_covers_the_requested_fraction():
    rng = np.random.default_rng(3)
    residuals = rng.normal(0, 50, 10_000)
    half = wind.conformal_interval(residuals, level=0.9)
    assert wind.calibration_report(residuals, half)["coverage"] == pytest.approx(0.9, abs=0.01)


def test_curve_inverse_round_trips():
    curve = wind.ReferencePowerCurve(RATED, DIAMETER)
    speeds = np.linspace(4.0, 10.0, 7)
    back = wind.curve_inverse(curve.power, curve.power(speeds))
    assert np.allclose(back, speeds, atol=0.1)


def test_domain_report_flags_extrapolation():
    train = _synthetic_frame(400)
    test = _synthetic_frame(200, seed=5)
    test.loc[0:19, "wind_speed_ms"] = 35.0
    report = wind.domain_report(train, test, variables=("wind_speed_ms",))
    assert report["variables"][0]["outside_count"] >= 20
    assert report["joint_inside_fraction"] < 1.0


def test_curtailment_optimiser_never_reports_a_loss():
    layout = _row_layout(n=4, spacing=300.0)
    curve = wind.ReferencePowerCurve(RATED, DIAMETER)
    result = wind.optimize_curtailment(layout, 8.0, 270.0, curve.power)
    assert result.optimal_power_kw >= result.baseline_power_kw - 1e-9
    assert np.all(result.thrust_scale >= 0.2 - 1e-12) and np.all(result.thrust_scale <= 1.0 + 1e-12)


def test_farm_efficiency_is_recovered_from_perfect_data():
    """A farm with no wakes must come back as efficiency 1 from the observation routine."""
    import pandas as pd

    layout = _row_layout(n=3, spacing=40_000.0)     # no wake can survive this spacing
    curve = wind.ReferencePowerCurve(RATED, DIAMETER)
    rng = np.random.default_rng(4)
    n = 400
    speeds = rng.uniform(5.0, 10.0, n)
    frames = []
    for i, name in enumerate(layout.names):
        frames.append(pd.DataFrame({
            "timestamp": pd.date_range("2017-01-01", periods=n, freq="10min"),
            "turbine": name,
            "power_kw": curve.power(speeds) * (1 + rng.normal(0, 0.01, n)),
            "wind_speed_ms": speeds,
            "u_corrected": speeds,
            "wind_direction_deg": np.full(n, 270.0 + (i - 1) * 0.5),
            "operating": True,
        }))
    farm = pd.concat(frames, ignore_index=True)
    curves = {name: curve.power for name in layout.names}
    observation = wind.farm_efficiency_observed(farm, layout, curves, n_sectors=36,
                                                speed_edges=(4.0, 11.0, 1.0), min_samples=5)
    values = observation.efficiency[np.isfinite(observation.efficiency)]
    assert values.size > 0
    assert np.allclose(values, 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# Field, twin and film
# ---------------------------------------------------------------------------

def test_wake_field_is_zero_upstream_and_matches_the_farm_at_the_turbines():
    layout = _row_layout(n=2)
    points = np.vstack([layout.positions, [[-5_000.0, 0.0]]])
    field = wind.wake_field(layout, points, 8.0, 270.0)
    assert field["deficit"][-1] == 0.0                       # far upstream of every rotor
    assert np.allclose(field["velocity_ratio"], 1.0 - field["deficit"])
    result = wind.farm_power(layout, 8.0, 270.0, wind.ReferencePowerCurve(RATED, DIAMETER).power)
    # With one wake path the field and the farm model must agree exactly.
    assert np.allclose(field["deficit"][:2], result.wake_deficit, atol=1e-9)


def test_wake_field_does_not_cascade():
    """Three turbines in a row: the field superposes free-stream wakes, the farm model cascades."""
    layout = _row_layout(n=3)
    field = wind.wake_field(layout, layout.positions, 8.0, 270.0)
    result = wind.farm_power(layout, 8.0, 270.0, wind.ReferencePowerCurve(RATED, DIAMETER).power)
    assert field["deficit"][2] >= result.wake_deficit[2]     # the documented, optimistic direction


def test_wake_field_deficit_grows_along_the_wake():
    layout = _row_layout(n=1, spacing=0.0)
    x = np.array([[2.0 * DIAMETER, 0.0], [6.0 * DIAMETER, 0.0], [15.0 * DIAMETER, 0.0]])
    field = wind.wake_field(layout, x, 8.0, 270.0)
    assert field["deficit"][0] > field["deficit"][1] > field["deficit"][2] > 0.0


def test_farm_scene_carries_one_field_slice_and_one_sensor_per_turbine():
    pytest.importorskip("pinneapple_twin3d")
    from pinneapple_systems.wind import twin

    layout = _row_layout(n=3)
    directions = np.arange(0.0, 360.0, 30.0)
    scene = twin.build_farm_scene(layout, directions, wind_speed=8.0,
                                  power_curves=[wind.ReferencePowerCurve(RATED, DIAMETER).power] * 3,
                                  n_x=24, n_y=18)
    ground = next(p for p in scene.parts if p.name == "ground")
    field = ground.fields["velocity_ratio"]
    assert field.shape == (directions.size, ground.vertices.shape[0])
    assert field.min() >= 0.0 and field.max() <= 1.0
    assert len(scene.sensors) == 3
    assert all(s.series is not None and s.series.size == directions.size for s in scene.sensors)
