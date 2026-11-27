"""Orbit determination and space traffic (``pinneapple_physics.astro.orbit_determination``), offline and without the
network: the propagator against the analytic two-body solution and its invariants, the decay against the orbit-averaged
formula of ``astro.decay``, the covariance against Monte Carlo and against the ``3 dv t`` along-track law, the
collision probability against its closed form and against a Monte Carlo of the encounter, the manoeuvre and storm flags
on an injected event, the TLE check on a real element set and on a broken one, and the learned correction on a
controlled drag mismatch.

Nothing here downloads a TLE or a space-weather index: what needs CelesTrak and the geomagnetic indices is listed in
``STARLINK_FEB_2022`` and in ``docs/core_concepts/orbit_determination.md``, and is not claimed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sgp4")

from pinneapple_physics.astro import orbit_determination as od  # noqa: E402
from pinneapple_physics.astro.conjunction import collision_probability_2d  # noqa: E402

MU = od.MU_EARTH
# ISS (ZARYA), the element set of pinneapple_physics/astro/data/reference/iss_passes_reference.json (epoch 2018-07-03,
# from CelesTrak and used there against skyfield 1.55)
ISS1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993"
ISS2 = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106"


def circular_state(a_km=od.R_E + 420.0, inc_deg=51.6):
    """A circular state of radius ``a_km`` at the ascending node: position on +x, velocity in the y-z plane."""
    v = math.sqrt(MU / a_km)
    i = math.radians(inc_deg)
    return np.array([a_km, 0.0, 0.0, 0.0, v * math.cos(i), v * math.sin(i)])


def period_s(a_km):
    return 2.0 * math.pi * math.sqrt(a_km**3 / MU)


def _checksum_fixed(line):
    """A TLE line with its checksum recomputed, so that an element corruption is not caught by the checksum alone."""
    body = line[:68]
    return body + str((sum(int(c) for c in body if c.isdigit()) + body.count("-")) % 10)


def test_two_body_propagation_conserves_energy_and_angular_momentum():
    """The numerical propagator on the two-body problem: energy and angular momentum are invariants, one period
    closes the orbit, and the analytic Kepler solution of the same state is reproduced."""
    s0 = circular_state(7000.0)
    T = period_s(7000.0)
    t = np.linspace(0.0, T, 200)
    S = od.propagate(s0, t, j2=False)
    energy = 0.5 * np.sum(S[:, 3:] ** 2, axis=1) - MU / np.linalg.norm(S[:, :3], axis=1)
    angmom = np.linalg.norm(np.cross(S[:, :3], S[:, 3:]), axis=1)
    period = period_s(float(od.semi_major_axis(S[:1])[0]))
    assert np.ptp(energy) / abs(energy[0]) < 1e-9            # measured 2e-11
    assert np.ptp(angmom) / angmom[0] < 1e-9                 # measured 1e-11
    assert abs(period - T) < 1e-6 * T                        # the orbit's own period, from the energy
    assert np.abs(S[-1, :3] - s0[:3]).max() < 1e-3           # measured 9e-8 km: it closes after one period


def test_numerical_propagator_matches_the_analytic_kepler_solution():
    """The universal-variable closed form against DOP853 on the same state: the analytic solution is the reference,
    and the J2 term is present (it moves the orbit by tens of km in one revolution, so its absence would show)."""
    s0 = circular_state(7000.0)
    t = np.linspace(0.0, period_s(7000.0), 40)
    S = od.propagate(s0, t, j2=False)
    K = od.propagate_kepler(s0, t)
    assert np.abs(S[:, :3] - K[:, :3]).max() < 1e-3          # measured 1e-7 km
    assert np.abs(S[:, 3:] - K[:, 3:]).max() < 1e-6
    Sj = od.propagate(s0, t)
    assert 10.0 < np.abs(Sj[:, :3] - K[:, :3]).max() < 1000.0   # measured 66 km: J2 is in the model
    # the analytic side is the exact two-body flow, so a state taken mid-arc reaches the same end point
    k = len(K) // 2
    back = od.propagate_kepler(K[k], [t[-1] - t[k]])
    assert np.abs(back[0, :3] - K[-1, :3]).max() < 1e-4


def test_drag_decay_matches_the_orbit_averaged_rate_and_a_storm_multiplies_it():
    """The measured secular decay of the semi-major axis against ``da/dt = -B rho sqrt(mu a)`` (the orbit-averaged
    formula of ``astro.decay``, an independent implementation), and a density jump injected at day 1: the decay rate
    must follow the factor, which is the mechanism of the February 2022 Starlink loss."""
    s0 = circular_state()
    B = 0.005
    t = np.arange(0.0, 2.0 * 86400.0 + 1.0, 120.0)
    quiet = od.propagate(s0, t, drag=od.DragModel(B), rtol=1e-9, atol=1e-8)
    t_d, a_d = od.mean_semi_major_axis(quiet, t)
    rate_meas = float(np.polyfit(t_d, a_d, 1)[0]) * 1000.0 / 86400.0        # m/s
    a0 = float(a_d[0])
    rho0 = float(np.atleast_1d(od.default_density(a0 - od.R_E))[0])
    rate_analytic = -B * rho0 * math.sqrt(MU * 1e9 * a0 * 1e3)
    assert rate_meas < 0.0
    assert abs(rate_meas / rate_analytic - 1.0) < 0.12        # measured 3.7 % apart
    storm = lambda tt: 1.0 if tt < 86400.0 else 3.0           # noqa: E731
    S = od.propagate(s0, t, drag=od.DragModel(B, storm=storm), rtol=1e-9, atol=1e-8)
    t_s, a_s = od.mean_semi_major_axis(S, t)
    pre = np.polyfit(t_s[t_s < 0.9], a_s[t_s < 0.9], 1)[0]
    post = np.polyfit(t_s[t_s > 1.5], a_s[t_s > 1.5], 1)[0]
    assert 2.5 < post / pre < 3.6                             # measured 3.15 for a factor-3 density jump


def test_covariance_growth_matches_monte_carlo_and_the_three_sigma_v_t_law():
    """The linear covariance (state transition matrix) against Monte Carlo on the same two-body orbit, and the
    along-track growth against ``3 dv t``, the closed form of an in-plane velocity error. The radial direction is
    where linearity is expected to fail once the along-track error is large: that is measured, not hidden."""
    s0 = circular_state(7000.0)
    sig_r, sig_v = 0.1, 1e-3
    P0 = np.diag([sig_r**2] * 3 + [sig_v**2] * 3)
    t = np.array([3600.0, 6 * 3600.0, 12 * 3600.0, 24 * 3600.0])
    lin = od.position_covariance(od.propagate_kepler(s0, t), od.propagate_covariance(s0, P0, t))
    mc = od.position_covariance(od.propagate_kepler(s0, t), od.monte_carlo_covariance(s0, P0, t, n=300, seed=1))
    ratio = mc["sigma_along_km"] / lin["sigma_along_km"]
    assert np.all(np.abs(ratio - 1.0) < 0.15)                 # measured 1.02-1.04 with 300 samples
    assert np.all(np.diff(lin["sigma_along_km"]) > 0)
    expected = 3.0 * sig_v * t                                # km, the along-track drift of an in-plane dv
    # asymptotic: valid once the secular -3 dv t overtakes the radial-velocity term (2/n)(1 - cos nt), which is what
    # the 1 h point still carries
    assert np.all(np.abs(lin["sigma_along_km"][1:] / expected[1:] - 1.0) < 0.10)   # measured +6 %, -1 %, +2 %
    assert 1.15 < lin["sigma_along_km"][0] / expected[0] < 1.40                     # measured 1.27 at 1 h
    # the radial direction departs from the linearisation at 24 h, where the along-track error is ~260 km and the
    # second-order term is O(ds^2 / a) ~ 10 km
    assert mc["sigma_radial_km"][-1] > 3.0 * lin["sigma_radial_km"][-1]


def test_trust_horizon_is_a_number_with_a_stated_tolerance():
    """The lead time at which the covariance reaches a tolerance: linear in the tolerance (the growth is linear in
    time while the velocity error dominates), and consistent with the closed form tolerance / (3 sigma_v)."""
    s0 = circular_state(7000.0)
    sig_v = 1e-3
    P0 = np.diag([0.1**2] * 3 + [sig_v**2] * 3)
    hours = {}
    for tol in (1.0, 5.0, 10.0):
        r = od.trust_horizon(s0, P0, tolerance_km=tol, days=3.0)
        assert r["reached"] and r["criterion"].startswith("covariance")
        assert r["sigma_km_at_horizon"] == pytest.approx(tol, rel=0.02)
        hours[tol] = r["t_hours"]
    assert hours[1.0] < hours[5.0] < hours[10.0]
    assert 0.8 < (hours[10.0] / hours[1.0]) / 10.0 < 1.3      # measured 11.9
    closed_form = 1.0 / (3.0 * sig_v) / 3600.0                # hours at which 3 sigma_v t = 1 km
    assert 0.5 < hours[1.0] / closed_form < 2.0               # measured 0.07 h vs 0.09 h


def test_encounter_plane_pc_matches_the_closed_form():
    """The 2-D probability of collision of a projected encounter against the closed form of the isotropic case:
    ``Pc = F_noncentral_chi2(r^2 / sigma^2; 2, m^2 / sigma^2)`` (Marcum Q, the same function as the Rayleigh form
    ``1 - exp(-r^2 / 2 sigma^2)`` at zero miss distance)."""
    from scipy.stats import ncx2

    for miss, sigma, hard_body in ((0.0, 0.05, 0.010), (0.05, 0.05, 0.010), (0.2, 0.1, 0.005), (0.5, 0.05, 0.02)):
        xm, zm, sx, sz, _ = od.encounter_plane([miss, 0.0, 0.0], [0.0, 7.5, 0.0], np.diag([sigma**2] * 3))
        assert sx == pytest.approx(sigma, rel=1e-9) and sz == pytest.approx(sigma, rel=1e-9)
        assert math.hypot(xm, zm) == pytest.approx(miss, rel=1e-9)
        pc = collision_probability_2d(xm, zm, sx, sz, hard_body)
        closed = float(ncx2.cdf((hard_body / sigma) ** 2, df=2, nc=(miss / sigma) ** 2))
        assert pc == pytest.approx(closed, rel=2e-3)


def test_propagated_conjunction_pc_matches_a_monte_carlo_of_the_encounter():
    """The whole conjunction path -- TCA from the propagated states, the covariances propagated to it, the encounter
    plane -- against a Monte Carlo: states drawn from both covariances, propagated to the same TCA, and counted inside
    the hard-body disk in the encounter plane."""
    chief = np.array([7000.0, 0.0, 0.0, 0.0, math.sqrt(MU / 7000.0), 0.0])
    tilt = 0.3
    deputy = np.array([7000.0 + 0.02, 0.0, 0.0, 0.0,
                       math.sqrt(MU / 7000.0) * math.cos(tilt), math.sqrt(MU / 7000.0) * math.sin(tilt)])
    cov = np.diag([0.1**2] * 3 + [1e-3**2] * 3)
    out = od.conjunction_probability(chief, deputy, cov, cov, hard_body_radius_km=0.02, n_scan=1200)
    assert out["miss_distance_km"] > 0.0 and out["relative_speed_km_s"] > 0.5   # a real crossing, not a drift
    assert 1e-3 < out["collision_probability"] < 0.1
    rng = np.random.default_rng(3)
    n, t_tca = 4000, out["t_tca_s"]
    L = np.linalg.cholesky(cov)
    sc = chief + rng.normal(size=(n, 6)) @ L.T
    sd = deputy + rng.normal(size=(n, 6)) @ L.T
    rc = od.propagate_kepler(chief, [t_tca])[0]
    rd = od.propagate_kepler(deputy, [t_tca])[0]
    n_hat = rc[3:] - rd[3:]
    n_hat = n_hat / np.linalg.norm(n_hat)
    e1 = np.cross(n_hat, [0.0, 0.0, 1.0])
    e1 = e1 / np.linalg.norm(e1)
    basis = np.stack([e1, np.cross(n_hat, e1)])
    rel = np.array([od.propagate_kepler(sc[i], [t_tca])[0, :3] - od.propagate_kepler(sd[i], [t_tca])[0, :3]
                    for i in range(n)])
    hits = float(np.mean(np.linalg.norm(rel @ basis.T, axis=1) < out["hard_body_radius_km"]))
    sigma = math.sqrt(hits * (1.0 - hits) / n)
    assert abs(hits - out["collision_probability"]) < 4.0 * sigma + 1e-9
    # a tighter miss is a larger probability, up to the hard-body radius
    closer = deputy.copy()
    closer[0] = 7000.0 + 0.005
    out2 = od.conjunction_probability(chief, closer, cov, cov, hard_body_radius_km=0.02, n_scan=1200)
    assert out2["collision_probability"] > out["collision_probability"]


def test_regime_changes_flags_a_manoeuvre_and_a_storm_but_not_a_clean_arc():
    """The residual that is not explained by the model: a clean arc flags nothing, an along-track manoeuvre of 5 m/s
    at day 1 is a step in the semi-major axis, and a density jump of a factor 3 at day 1 is a step in its rate. The
    quiet model is what the detector is given: the residual is the decay the model did not predict."""
    s0 = circular_state()
    B = 0.005
    t = np.arange(0.0, 2.0 * 86400.0 + 1.0, 120.0)
    quiet = od.propagate(s0, t, drag=od.DragModel(B), rtol=1e-9, atol=1e-8)
    t_q, a_q = od.mean_semi_major_axis(quiet, t)
    assert od.regime_changes(t_q, a_q) == []                   # ... with and without the model
    assert od.regime_changes(t_q, a_q, model_km=(t_q, a_q)) == []
    storm = od.propagate(s0, t, drag=od.DragModel(B, storm=lambda tt: 1.0 if tt < 86400.0 else 3.0),
                         rtol=1e-9, atol=1e-8)
    t_s, a_s = od.mean_semi_major_axis(storm, t)
    events = od.regime_changes(t_s, a_s, model_km=(t_q, a_q), min_magnitude=1e-3)
    assert len(events) == 1 and events[0]["kind"] == "drag"
    assert abs(events[0]["t_days"] - 1.0) < 0.15 and abs(events[0]["z"]) > 10.0
    assert -0.25 < events[0]["magnitude"] < -0.08              # km/day; the truth is -0.124 (a factor 3 in the rate)
    # a manoeuvre: 5 m/s along-track at day 1, then the same drag on both sides of it
    at_one_day = od.propagate(s0, [86400.0], drag=od.DragModel(B), rtol=1e-9, atol=1e-8)[0]
    r_hat = at_one_day[:3] / np.linalg.norm(at_one_day[:3])
    h_hat = np.cross(at_one_day[:3], at_one_day[3:])
    t_hat = np.cross(h_hat / np.linalg.norm(h_hat), r_hat)
    burnt = at_one_day.copy()
    burnt[3:] += 0.005 * t_hat                                 # 5 m/s along-track
    t2 = np.arange(0.0, 86400.0 + 1.0, 120.0)
    after = od.propagate(burnt, t2, drag=od.DragModel(B), rtol=1e-9, atol=1e-8)
    left = t_q <= 1.0
    t_after, a_after = od.mean_semi_major_axis(after, t2)
    t_m = np.concatenate([t_q[left] * 86400.0, 86400.0 + t_after * 86400.0])
    a_m = np.concatenate([a_q[left], a_after])
    jump = float(od.semi_major_axis(after[:1])[0] - od.semi_major_axis(at_one_day[None, :])[0])
    assert jump > 5.0                                          # km: a 5 m/s burn raises a by ~9 km
    events = od.regime_changes(t_m / 86400.0, a_m, model_km=(t_q, a_q), min_magnitude=0.5)
    assert len(events) == 1 and events[0]["kind"] == "manoeuvre"
    assert abs(events[0]["t_days"] - 1.0) < 0.15               # one-period averaging lags the break
    assert abs(events[0]["magnitude"] - jump) < 0.25 * jump


def test_tle_check_accepts_the_reference_element_set_and_rejects_a_broken_one():
    """A real element set (the ISS TLE of the reference directory) passes every check; a corrupted copy fails the
    one it should: the checksum, the satellite number, and -- with the checksum repaired, so the elements are judged
    on their own -- a perigee below the surface and a propagation that does not reproduce the element set."""
    good = od.tle_check(ISS1, ISS2)
    assert good["suspect"] is False and good["flags"] == []
    ids = {c["id"] for c in good["checks"]}
    assert {"checksum", "satnum", "perigee", "reproduces_mean_motion", "osculating_semi_major_axis"} <= ids
    sc = good["self_consistency"]
    assert 0.0 < abs(sc["revolutions_rel"]) < 1e-3            # measured 9.2e-5: it makes its own mean motion
    assert 1.0 < abs(sc["da_km"]) < 10.0                      # measured 4.5 km: the Brouwer short-period term
    assert od.tle_measurement_check(ISS1, ISS2) is False
    # a retyped last digit: the checksum catches it
    bad = od.tle_check(ISS1, ISS2[:68] + ("0" if ISS2[68] != "0" else "1"))
    assert bad["suspect"] and bad["flags"] == ["checksum"]
    # a satellite number that does not match the other line
    bad = od.tle_check(ISS1, "2 25545" + ISS2[7:])
    assert "satnum" in bad["flags"]
    # an eccentricity of 0.9 with the checksum repaired: the perigee is below the ground and the propagation no
    # longer reproduces the element set
    bad = od.tle_check(ISS1, _checksum_fixed(ISS2[:26] + "9000000" + ISS2[33:]))
    assert bad["suspect"] and "perigee" in bad["flags"]
    assert "reproduces_mean_motion" in bad["flags"]
    assert od.tle_measurement_check(ISS1, _checksum_fixed(ISS2[:26] + "9000000" + ISS2[33:])) is True


def test_learned_correction_reduces_the_error_by_lead_time():
    """The learned correction of the TLE propagation, in the controlled experiment: the "TLE" has a ballistic
    coefficient 30 % off the truth, the correction is fitted on the first day and scored on the second, which it
    never saw. The target carries the ~0.05 km noise of a TLE as truth, so the reduction is not credited with more
    than the fit can know."""
    s0 = circular_state()
    t = np.arange(0.0, 2.0 * 86400.0 + 1.0, 300.0)
    truth = od.propagate(s0, t, drag=od.DragModel(0.005), rtol=1e-9, atol=1e-8)
    model = od.propagate(s0, t, drag=od.DragModel(0.0065), rtol=1e-9, atol=1e-8)
    rtn = np.array([od.rtn_frame(truth[k, :3], truth[k, 3:]) @ (truth[k, :3] - model[k, :3])
                    for k in range(len(t))])
    rng = np.random.default_rng(0)
    rtn_noisy = rtn + rng.normal(0.0, 0.05, rtn.shape)
    train = t <= 86400.0
    fit = od.LearnedTleCorrection(ridge=1e-3).fit(t[train], rtn_noisy[train])
    pred = fit.predict(t[~train])
    before = np.linalg.norm(rtn[~train], axis=1)
    after = np.linalg.norm(rtn[~train] - pred, axis=1)
    assert before.max() > 3.0                                  # measured 5.4 km at day 2
    assert after.max() < 0.15 * before.max()                   # measured 0.003 of it
    # every feature vanishes at the epoch, so the correction leaves the TLE's own state where it is
    assert np.abs(fit.predict([0.0])).max() < 1e-9
    # ... and it explains what it was fitted on
    assert np.linalg.norm(rtn[train][-1] - fit.predict(t[train])[-1]) < 0.2
    # the correction moves a trajectory by the predicted offset in its own RTN frame
    idx = np.nonzero(~train)[0]
    moved = fit.apply(model[~train], t[~train])
    manual = np.array([model[k, :3] + pred[j] @ od.rtn_frame(model[k, :3], model[k, 3:])
                       for j, k in enumerate(idx)])
    assert np.abs(moved[:, :3] - manual).max() < 1e-12


def test_traffic_ensemble_combines_the_experts_and_refuses_a_bad_element_set():
    """The three witnesses in ``PhysicsEnsemble``: SGP4, the numerical propagator and SGP4 plus the learned correction,
    each scored by the two-body residual of its own trajectory (no reference needed), and the TLE check wired in as
    the ensemble's ``measurement_check``: a broken element set is flagged and teaches nothing."""
    t = np.arange(0.0, 4.0 * 3600.0, 20.0)
    correction = od.LearnedTleCorrection(ridge=1e-3).fit([86400.0], np.zeros((1, 3)))
    ens = od.traffic_ensemble(correction)
    assert ens.names == ["sgp4", "numerical", "sgp4+correction"]
    out = ens.predict({"tle": (ISS1, ISS2), "t_s": t})
    assert out["suspect"] is False
    assert out["prediction"].shape == (len(t), 3)
    assert set(out["expert_predictions"]) == set(ens.names)
    assert all(np.isfinite(v) for v in out["residuals"].values())
    assert max(out["residuals"].values()) < 5e-2              # measured 1.5e-3: the J2 term a two-body residual sees
    assert sum(out["weights"].values()) > 0.0
    spread = np.linalg.norm(out["expert_predictions"]["numerical"] - out["expert_predictions"]["sgp4"], axis=1)
    assert 1e-3 < spread.max() < 5.0                           # measured 0.23 km over 4 h, two drag models
    # the corrupted element set: the ensemble still predicts, and marks the case as suspect
    out_sgp4 = out["expert_predictions"]["sgp4"]
    broken = (ISS1, _checksum_fixed(ISS2[:26] + "9000000" + ISS2[33:]))
    out = ens.predict({"tle": broken, "t_s": t})
    assert out["suspect"] is True
    info = ens.update(reference=out_sgp4)
    assert info.get("suspect") is True and "errors" in info     # scored, but the weights did not learn from it
    # ... while SGP4 itself parses the broken set and propagates it happily: the check is what catches it
    assert int(od.sgp4_states(*broken, [0.0])[2][0]) == 0
    assert "perigee" in od.tle_check(*broken)["flags"]


def test_the_february_2022_starlink_case_is_declared_with_what_it_needs():
    """The declared real-world case is in the module as data: the event, what it reports (which is not measured here),
    and the inputs a reproduction needs (CelesTrak's TLE archive, the storm indices, the ballistic coefficient). The
    mechanism is what the tests measure: at the 210 km deployment altitude a density jump of a factor 2 halves the
    remaining life, and at 550 km it does not matter inside the horizon."""
    case = od.STARLINK_FEB_2022
    assert case["reported"]["lost"] == 38 and case["reported"]["launched"] == 49
    assert set(case["needed_to_reproduce"]) >= {"tles", "space_weather", "ballistic_coefficient"}
    low = od.reentry_warning(210.0, 0.005, storm_factor=2.0, days=60.0)
    high = od.reentry_warning(550.0, 0.005, storm_factor=2.0, days=60.0)
    assert low["warning"] and not high["warning"]
    assert low["censored_quiet"] is False
    assert 0.3 < low["shortening_fraction"] < 0.7              # measured 0.51
    assert 1.5 < low["lifetime_days_quiet"] / low["lifetime_days_storm"] < 3.0    # measured 2.0
    assert high["altitude_loss_km_storm"] < 5.0


def test_the_training_set_and_the_residual_by_lead_time():
    """The two pieces that consume real data without needing it to be here: the residual of a TLE against a later one
    (``tle_pair_residuals``, whose training set is an archive that only the network can give, and which therefore
    says so instead of inventing one), and the ratio of the measured error to the model's sigma by lead time."""
    empty = od.tle_pair_residuals([(ISS1, ISS2)])               # one element set: no pairs, no invented data
    assert empty["t_s"].size == 0 and empty["rtn_km"].shape == (0, 3)
    # the features vanish at the epoch: the correction cannot move a TLE's own state
    assert np.allclose(od.correction_features([0.0]), 0.0)
    assert od.correction_features([86400.0]).shape == (1, 4)
    r = od.residual_by_lead_time([1.0, 4.0, 9.0], [1.0, 6.0, 12.0], [0.5, 2.0, 3.0])
    assert np.allclose(r["ratio"], [2.0, 2.0, 3.0]) and r["max_ratio"] == 3.0
    assert r["median_ratio"] == 2.0
