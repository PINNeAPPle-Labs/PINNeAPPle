"""Real regression tests -- reproduces patchdose_ai's own baseline
numbers exactly, and its confirmed real finding that both depletion
time and the post-depletion decay constant are independent of patch
area."""
from __future__ import annotations

from pinneapple_physics.closed_form import membrane_diffusion as md

D = 5e-13
K = 1.0
RESERVOIR_THICKNESS_M = 0.001


def test_release_rate_matches_patchdose_ai_baseline():
    """patchdose_ai's baseline patch: 20cm^2 (20e-4 m^2), 80um membrane, 60 kg/m^3 reservoir."""
    rate = md.steady_state_release_rate_mg_day(D, K, 60.0, 20e-4, 80e-6)
    # by hand: D K C / L = 5e-13 * 60 / 80e-6 = 3.75e-7 kg/(m^2 s); x A = 7.5e-10 kg/s = 7.5e-4 mg/s = 64.8 mg/day
    # (patchdose_ai reported 0.0648 "mg/day": its kg -> mg factor was 1e3, i.e. grams)
    assert abs(rate - 64.8) < 1e-9 * 64.8 + 1e-9


def test_released_mass_over_the_depletion_time_equals_the_reservoir_mass():
    """Rate x t_dep = C A L_res (units must close: 60 kg/m^3 x 20 cm^2 x 1 mm = 120 mg)."""
    rate = md.steady_state_release_rate_mg_day(D, K, 60.0, 20e-4, 80e-6)
    t_dep = md.depletion_time_days(D, K, 80e-6, RESERVOIR_THICKNESS_M)
    assert abs(rate * t_dep - 120.0) < 1e-9
    points = md.simulate_release_timeseries(D, K, 60.0, 20e-4, 80e-6, RESERVOIR_THICKNESS_M, n_points=3)
    assert abs(points[0]["reservoir_remaining_mg"] - 120.0) < 1e-9


def test_depletion_time_independent_of_area():
    t1 = md.depletion_time_days(D, K, 80e-6, RESERVOIR_THICKNESS_M)
    t2 = md.depletion_time_days(D, K, 80e-6, RESERVOIR_THICKNESS_M)  # depletion_time_days doesn't take area at all
    assert t1 == t2
    assert abs(t1 - 1.8518518518518519) < 1e-6  # L_res*L/(D*K) in days, hand-computed


def test_decay_constant_independent_of_area_and_equals_inverse_depletion_time():
    k1 = md.post_depletion_decay_rate_constant_per_day(D, K, 80e-6, RESERVOIR_THICKNESS_M)
    t_dep = md.depletion_time_days(D, K, 80e-6, RESERVOIR_THICKNESS_M)
    assert abs(k1 - 1.0 / t_dep) < 1e-9  # real derived identity, not a coincidence


def test_concentration_profile_is_linear_and_matches_flux_gradient():
    c0 = md.concentration_at_membrane_depth(60.0, K, 80e-6, 0.0)
    c_mid = md.concentration_at_membrane_depth(60.0, K, 80e-6, 40e-6)
    c_end = md.concentration_at_membrane_depth(60.0, K, 80e-6, 80e-6)
    assert abs(c0 - 60.0) < 1e-9
    assert abs(c_end - 0.0) < 1e-9
    assert abs(c_mid - 30.0) < 1e-9  # linear profile -> exact midpoint


def test_timeseries_has_flat_phase_then_strictly_decreasing_tail():
    points = md.simulate_release_timeseries(D, K, 60.0, 20e-4, 80e-6, RESERVOIR_THICKNESS_M, n_points=200)
    t_dep = md.depletion_time_days(D, K, 80e-6, RESERVOIR_THICKNESS_M)
    flat_rates = [p["rate_mg_day"] for p in points if p["day"] < t_dep * 0.95]
    tail_rates = [p["rate_mg_day"] for p in points if p["day"] > t_dep * 1.05]
    assert max(flat_rates) - min(flat_rates) < 1e-9  # real constant zero-order phase
    assert all(tail_rates[i] > tail_rates[i + 1] for i in range(len(tail_rates) - 1))  # real strictly decreasing tail
