"""Tests for ``pinneapple_physics.domain_checks`` -- the three checks that need no reference field (#348).

A pre-trained model is accurate where it was trained and quiet outside it, so the checks here ask what the domain
itself demands of any field. Each one is asserted twice: on a field that satisfies the structure (it has to come out
near zero) and on a field that breaks it in the specific way the check exists to catch (it has to come out near one).
The distance between those two numbers on the same construction is the check's separating power.

* **weather**, :func:`advection_residual`: a field that is exactly advected by the wind scores 0.0028 (the scheme's
  truncation error, not the field's), and the same field sitting still under that wind scores **1.000** -- the tracer
  that does not move is flagged as hard as the one that moves for no reason.
* **weather**, :func:`global_drift`: a constant total drifts 0.000; a total that grows by 5 % of its spread drifts
  0.079 spreads per day.
* **flood**, :func:`hand` and :func:`flood_mask_score`: on a slope draining to the right, hand is the height above the
  outlet (0 to 10 m); a mask that wets only the low ground is admissible (1.000), one that wets ground far above the
  water is not (0.306), and a mask that wets nothing is admissible **and uninformative** -- the score says 1, which is
  why it is documented as such instead of being read as a pass.
* **video**, :func:`temporal_residual`: frames translated by a constant displacement have a flow residual of 2e-9;
  frames of noise at the same size have 0.974, with 7.5 % of the pixels teleporting.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_physics.domain_checks import (
    advection_check,
    advection_residual,
    as_prediction_check,
    dry_static_energy,
    energy_check,
    flood_mask_score,
    flow_directions,
    global_drift,
    hand,
    hand_check,
    temporal_continuity_check,
    temporal_residual,
)

R_EARTH = 6.371e6
OMEGA = 1.0e-6                                  # rad/s of the zonal phase speed of the test field
HOURS = 6.0
LAT = np.linspace(-60, 60, 25)
LON = np.linspace(0, 360, 49, endpoint=False)
LON_R, LAT_R = np.meshgrid(np.deg2rad(LON), np.deg2rad(LAT))


def _advected(steps: int = 3, speed: float = OMEGA) -> np.ndarray:
    """``cos(lon - OMEGA t)`` with the zonal wind that carries it exactly, at every latitude."""
    return np.stack([np.cos(LON_R - speed * HOURS * 3600.0 * k) for k in range(steps + 1)])


def _wind(speed: float = OMEGA, shape=(25, 49)) -> tuple[np.ndarray, np.ndarray]:
    """The wind that advects the test field's phase speed, in m/s, broadcast over the grid."""
    u = np.broadcast_to((speed * R_EARTH * np.cos(np.deg2rad(LAT)))[:, None], shape)
    return u, np.zeros_like(u)


# --------------------------------------------------------------------------------------------------- weather
def test_a_field_that_is_exactly_advected_has_almost_no_residual():
    u, v = _wind()
    value = advection_residual(_advected(), u, v, HOURS, LAT, LON)

    assert value < 0.01, "what is left is the second-order scheme's truncation error, not the field's"
    assert value > 0.0


def test_a_tracer_that_sits_still_under_the_wind_is_flagged():
    still = _advected()
    still = np.stack([still[0]] * still.shape[0])          # nothing changes while the wind blows
    u, v = _wind()

    assert advection_residual(still, u, v, HOURS, LAT, LON) == pytest.approx(1.0, abs=1e-6)


def test_a_field_that_does_not_move_and_no_wind_to_move_it_is_zero():
    still = np.stack([np.cos(LON_R)] * 3)
    u, v = _wind(speed=0.0)

    assert advection_residual(still, u, v, HOURS, LAT, LON) == 0.0


def test_a_time_change_with_no_advective_structure_scores_about_one():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((2, 25, 49))
    u, v = _wind()

    assert advection_residual(noise, u, v, HOURS, LAT, LON) == pytest.approx(1.0, abs=0.05)


def test_the_advection_residual_needs_a_time_axis():
    with pytest.raises(ValueError, match="steps"):
        advection_residual(np.zeros((25, 49)), *_wind(), HOURS, LAT, LON)


def test_a_conserved_total_does_not_drift():
    constant = np.stack([np.full((25, 49), 280.0), np.full((25, 49), 280.0)])

    assert global_drift(constant, LAT, HOURS) == 0.0


def test_a_total_that_grows_drifts_in_units_of_its_own_spread():
    base = np.cos(LON_R) - np.cos(LON_R).mean()            # the spatial pattern; its spread is the unit
    value = global_drift(np.stack([base, base + 0.05]), LAT, HOURS)

    assert value == pytest.approx(0.040, abs=0.003), "half the spread per day for a 5 % step over 6 h"
    assert global_drift(np.stack([base, base + 0.10]), LAT, HOURS) == pytest.approx(2 * value)


def test_dry_static_energy_is_cp_t_plus_geopotential():
    energy = dry_static_energy(np.full((2, 2), 300.0), np.full((2, 2), 1000.0))

    assert np.allclose(energy, 1004.0 * 300.0 + 1000.0)


def test_the_weather_checks_speak_the_language_of_the_monitor():
    u, v = _wind()
    check = advection_check(lambda q: (u, v), 0.05, LAT, LON)
    energy = energy_check(0.05, LAT)

    assert check.name == "advection_consistency" and check.kind == "residual"
    assert check.fn(_advected(), {}) < 0.01
    assert "tolerance" in check.describe(check.fn(_advected(), {}))
    assert energy.fn(np.stack([np.full((25, 49), 280.0)] * 2), {}) == 0.0


# ----------------------------------------------------------------------------------------------- flood masks
def _slope() -> tuple[np.ndarray, np.ndarray]:
    """A plane draining to the right, with the outlet on the last column."""
    terrain = np.tile(np.linspace(10.0, 0.0, 49), (25, 1))
    drainage = np.zeros_like(terrain, bool)
    drainage[:, -1] = True
    return terrain, drainage


def test_flow_directions_point_downhill_and_stop_at_the_outlet():
    terrain, _ = _slope()
    nxt = flow_directions(terrain).reshape(terrain.shape)
    middle = nxt[12, 24]

    assert middle == 12 * 49 + 25                          # one cell to the right
    assert np.all(nxt[:, -1] == -1), "the outlet has no lower neighbour and is reported as a pit"


def test_hand_is_the_height_above_the_nearest_drainage():
    terrain, drainage = _slope()
    h = hand(terrain, drainage)

    assert not np.isnan(h).any(), "every cell of a filled DEM drains somewhere"
    assert h[:, -1] == pytest.approx(0.0)                  # the drainage itself
    assert h[:, 0] == pytest.approx(10.0)                  # the far end of the slope
    assert np.all(np.diff(h[0]) <= 1e-9), "the height above the outlet only grows uphill"


def test_a_mask_inside_the_water_depth_is_admissible():
    terrain, drainage = _slope()

    # deeper than the whole slope (10 m), so no wet cell is above the water: the mask the terrain admits
    assert flood_mask_score(np.ones_like(terrain, bool), terrain, drainage, depth=12.0) == pytest.approx(1.0)


def test_a_mask_that_wets_ground_far_above_the_water_is_not():
    terrain, drainage = _slope()
    score = flood_mask_score(np.ones_like(terrain, bool), terrain, drainage, depth=3.0)

    assert score == pytest.approx(0.306, abs=0.01), "the cells with 3 m or more above the outlet are inadmissible"
    assert score > 0.0, "the low cells it wets are still admissible"


def test_a_mask_that_wets_the_outlet_and_not_its_donors_is_flagged():
    terrain, drainage = _slope()
    low = hand(terrain, drainage) <= 1.0

    with_donors = flood_mask_score(low, terrain, drainage, depth=12.0)
    without = flood_mask_score(low, terrain, drainage, depth=12.0, check_upstream=False)
    assert with_donors == pytest.approx(0.8, abs=0.02), "one cell in five wets without the cells that feed it"
    assert without == pytest.approx(1.0), "turn the rule off and the same mask is admissible"


def test_a_mask_that_wets_nothing_scores_one_and_says_nothing():
    terrain, drainage = _slope()

    assert flood_mask_score(np.zeros_like(terrain, bool), terrain, drainage, depth=3.0) == 1.0


def test_the_flood_check_reads_like_the_others():
    terrain, drainage = _slope()
    check = hand_check(terrain, drainage, depth=3.0)
    admissible = np.ones_like(terrain, float)                  # the whole slope, with a 12 m water depth parked above
    check_deep = hand_check(terrain, drainage, depth=12.0)

    assert check.kind == "constitutive" and check.tolerance == 0.0
    assert check_deep.fn(admissible, {}) == pytest.approx(0.0), "1 - score: 0 for a mask the terrain admits"
    assert check.fn(admissible, {}) > 0.5, "the same mask with 3 m of water is inadmissible"
    assert "inadmissible" in check.describe(check.fn(admissible, {}))


# --------------------------------------------------------------------------------------------------- video
def _scene() -> np.ndarray:
    """A frame with structure in both axes: a field varying in one axis alone leaves the motion indeterminate."""
    return np.sin(LAT_R) * np.cos(LON_R)


def test_frames_that_translate_are_explained_by_their_own_motion():
    frame = _scene()
    frames = np.stack([np.roll(frame, 2 * k, axis=1) for k in range(3)])
    out = temporal_residual(frames, max_speed=3.0)

    assert out["flow_residual"] < 1e-6
    assert out["teleport_fraction"] == 0.0
    assert out["mean_speed"] == pytest.approx(2.0, abs=0.2)


def test_frames_of_noise_are_flagged_and_the_teleport_fraction_names_it():
    frames = np.random.default_rng(0).standard_normal((3, 25, 49))
    out = temporal_residual(frames, max_speed=1.0)

    assert out["flow_residual"] > 0.5
    assert out["teleport_fraction"] > 0.0
    assert set(out) == {"flow_residual", "mean_speed", "teleport_fraction", "jerk"}


def test_the_video_check_turns_the_flow_residual_into_a_verdict():
    frame = _scene()
    frames = np.stack([np.roll(frame, 2 * k, axis=1) for k in range(3)])
    check = temporal_continuity_check(tolerance=0.25, max_speed=3.0)

    assert check.fn(frames, {}) < 0.01
    assert "tolerance" in check.describe(check.fn(frames, {}))


# ---------------------------------------------------------------------------------------- the ensemble hook
def test_a_check_becomes_a_prediction_check_with_a_score_in_the_unit_interval():
    u, v = _wind()
    check = advection_check(lambda q: (u, v), 0.05, LAT, LON)
    prediction_check = as_prediction_check(check)

    score, reason = prediction_check(_advected(), {})
    assert 0.0 <= score <= 1.0 and score > 0.9
    assert reason.startswith("advection_consistency:")

    bad_score, bad_reason = prediction_check(np.stack([np.cos(LON_R)] * 4), {})   # nothing changes
    assert bad_score < score and "unexplained by advection" in bad_reason
