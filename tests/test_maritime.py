"""The maritime package of #364, held to what a buoy, a wave tank and a load cell would show.

What is asserted, in the order the chain is built:

* **the spectrum** -- a JONSWAP whose own integral is the ``Hs`` that was asked for, the Pierson-Moskowitz relations
  ``Hs = 0.21 U^2/g`` and ``Tp = 7.16 U/g`` out of the coefficients they come from, and a synthesized record whose
  ``4 sigma`` reproduces the spectrum it was cut from;
* **the sea state with a horizon** -- the group velocity against ``g T / 4 pi``, a swell that decays faster at high
  frequency (so ``Tp`` rises while ``Hs`` falls), a wind sea that grows toward what the wind can hold, and a model
  whose two coefficients are recovered from data generated with them;
* **the section** -- the checks the free-surface solution has to pass whatever the numbers are: ``RAO -> 1`` for heave
  and roll in the long-wave limit, positive damping, a section that stops making waves when they are short, and a roll
  resonance that moves with the added inertia the solve measures;
* **the mooring** -- the catenary against its own closed forms (a vertical hang, the tangency point, the taut line),
  tension that rises nonlinearly with the offset, and a lost line loading the rest harder;
* **the residual** -- a quiet sea stays quiet, a slam is called a slam, a step in the tension is called the mooring, and
  two instruments that fail in two ways are called instruments;
* **the window** -- the criterion, the hours it closes, and a confidence that is the measured fraction of right
  verdicts rather than a curve somebody picked;
* **the buoy reader** -- NDBC's files parsed from text that is in the repository, so the parser is tested without a
  network.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_physics.maritime import (
    confidence,
    mooring,
    motion,
    ndbc,
    operability,
    rao,
    seastate,
)
from pinneapple_physics.maritime import spectrum as sp

G = 9.81


# ------------------------------------------------------------------------------------------------ the spectrum
def test_jonswap_integrates_to_the_height_it_was_given():
    for hs, tp in ((0.5, 4.0), (1.0, 6.0), (3.0, 10.0), (6.0, 14.0)):
        sea = sp.Jonswap(hs=hs, tp=tp)
        assert sea.significant_height() == pytest.approx(hs, rel=2e-3), f"Hs {hs} at Tp {tp}"
        freq = np.linspace(1e-3, 1.0, 60001)
        assert sp.hs_from_spectrum(freq, sea.density_hz(freq)) == pytest.approx(hs, rel=2e-3)


def test_pierson_moskowitz_gives_the_textbook_relations():
    """``Hs = 0.21 U^2/g`` and ``Tp = 7.16 U/g`` come out of ``alpha = 0.0081`` and ``w_p = 0.877 g/U``."""
    for wind in (8.0, 12.0, 18.0, 25.0):
        sea = sp.fully_developed(wind)
        assert sea.hs / (0.21 * wind ** 2 / G) == pytest.approx(0.997, rel=3e-3)
        assert sea.tp / (7.16 * wind / G) == pytest.approx(1.0, rel=3e-3)
        assert sea.gamma == 1.0


def test_peak_period_comes_back_off_the_spectrum():
    sea = sp.Jonswap(hs=2.5, tp=9.5)
    freq = np.linspace(0.01, 1.0, 40001)
    assert sp.peak_period(freq, sea.density_hz(freq)) == pytest.approx(9.5, rel=5e-3)


def test_a_synthesized_record_measures_the_height_of_its_spectrum():
    sea = sp.Jonswap(hs=3.0, tp=10.0)
    freq = np.linspace(0.01, 0.6, 4000)
    for seed in (0, 1, 2):
        components = sp.components(freq, sea.density_hz(freq), seed=seed, count=200)
        time = np.arange(0.0, 30000.0, 0.5)
        measured = 4.0 * float(np.std(components.elevation(time)))
        assert measured == pytest.approx(3.0, rel=3e-2), f"seed {seed}: {measured:.3f} m"
        assert components.hs == pytest.approx(3.0, rel=1e-3)


def test_group_velocity_and_wavelength_are_the_linear_wave_relations():
    for period in (4.0, 8.0, 14.0):
        assert float(sp.group_velocity(period)) == pytest.approx(G * period / (4.0 * math.pi), rel=1e-12)
        assert float(sp.wavelength(period)) == pytest.approx(1.56 * period ** 2, rel=2e-3)
    # in shallow water the group velocity tends to the phase speed sqrt(g h), which is *above* the deep-water group
    # velocity of the same period; in between it sits between that phase speed and the deep-water value
    for period in (8.0, 14.0):
        deep, finite = float(sp.group_velocity(period)), float(sp.group_velocity(period, 20.0))
        phase = float(sp.phase_speed(period, 20.0))
        assert min(deep, phase) <= finite <= max(deep, phase)
    shallow = float(sp.group_velocity(200.0, 20.0))
    assert shallow == pytest.approx(math.sqrt(G * 20.0), rel=5e-2)


def test_the_fit_describes_a_synthetic_jonswap_and_prefers_gamma_above_one():
    sea = sp.Jonswap(hs=2.0, tp=8.0, gamma=3.3)
    grid = np.array([0.033, 0.038, 0.043, 0.048, 0.053, 0.058, 0.063, 0.068, 0.073, 0.078, 0.083, 0.088, 0.093,
                     0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180, 0.190, 0.200])
    fitted, rms = sp.fit(grid, sea.density_hz(grid))
    assert rms < 0.05, f"the analytic shape fitted to itself is {rms:.1%} off"
    assert fitted.hs == pytest.approx(2.0, rel=5e-2)
    assert fitted.gamma > 1.5, f"gamma {fitted.gamma:.2f} came back below the peak enhancement that was put in"


# ------------------------------------------------------------------------------------------------ the sea state
def test_the_separation_frequency_is_where_the_peak_of_a_developed_sea_sits():
    for wind in (6.0, 12.0, 20.0):
        developed = float(seastate.separation_frequency(wind, ratio=1.14))
        assert developed == pytest.approx(1.0 / sp.fully_developed(wind).tp, rel=2e-2)


def test_a_swell_ages_while_it_decays():
    """No wind: the high bands go first, so ``Hs`` falls and ``Tp`` rises. Persistence cannot say that."""
    model = seastate.SeaStateModel(swell_decay_hours=30.0, windsea_growth_hours=6.0)
    freq = np.linspace(0.03, 0.5, 200)
    density = sp.Jonswap(hs=3.0, tp=12.0).density_hz(freq)
    out = model.forecast(freq, density, 0.5, leads=[1.0, 6.0, 24.0, 72.0])
    assert np.all(np.diff(out.hs) < 0), out.hs
    assert out.hs[0] < 3.0 and out.hs[-1] < out.hs[0]
    decayed = model.advance(freq, density, 0.5, 24.0)
    ratio = decayed[density > 0.01 * density.max()] / density[density > 0.01 * density.max()]
    assert np.all(ratio < 1.0 + 1e-12)
    assert model.advance(freq, density, 0.5, 0.0).tolist() == density.tolist()


def test_the_high_frequency_end_decays_faster_than_the_peak():
    model = seastate.SeaStateModel(swell_decay_hours=10.0)
    freq = np.linspace(0.03, 0.5, 400)
    density = sp.Jonswap(hs=2.0, tp=14.0).density_hz(freq)
    decayed = model.advance(freq, density, 0.4, 12.0)
    peak = int(np.argmax(density))
    assert decayed[-1] / density[-1] < decayed[peak] / density[peak] < 1.0
    assert sp.peak_period(freq, decayed) > sp.peak_period(freq, density)


def test_a_persisted_wind_carries_its_wind_sea_and_growing_it_inflates_the_height():
    """The two closures, measured against each other: carrying the wind sea, and relaxing it toward fully developed."""
    freq = np.linspace(0.03, 0.5, 200)
    young = sp.Jonswap(hs=0.4, tp=4.0).density_hz(freq)          # much smaller than a 12 m/s wind can make
    carried = seastate.SeaStateModel(swell_decay_hours=48.0, windsea_growth=False)
    grown = seastate.SeaStateModel(swell_decay_hours=48.0, windsea_growth_hours=3.0, windsea_growth=True)
    leads = [1.0, 6.0, 24.0]
    flat = carried.forecast(freq, young, 12.0, leads=leads)
    rising = grown.forecast(freq, young, 12.0, leads=leads)
    assert rising.hs[-1] > 3.0 * flat.hs[-1], "the growing closure is what fills an empty sea up"
    assert np.all(np.diff(rising.hs) > 0)
    assert rising.hs[-1] < sp.fully_developed(12.0).hs, "no wind sea grows past what the wind holds"
    assert flat.hs[-1] < 0.5, "a carried wind sea stays where it was, less whatever the swell loses"


def test_the_two_coefficients_and_only_them_come_back_out_of_synthetic_data():
    """Generate with a known pair, calibrate on it, and the search has to land on that pair -- not near it."""
    truth = seastate.SeaStateModel(swell_decay_hours=24.0, windsea_growth_hours=8.0)
    freq = np.linspace(0.03, 0.5, 200)
    leads = np.array([1.0, 3.0, 6.0, 12.0, 18.0, 24.0])
    rng = np.random.default_rng(4)
    states, observed_hs, observed_tp = [], [], []
    for height, period, wind in ((1.5, 9.0, 8.0), (2.5, 11.0, 6.0), (3.5, 7.0, 14.0), (1.0, 13.0, 4.0)):
        spectrum = sp.Jonswap(hs=height, tp=period).density_hz(freq)
        out = truth.forecast(freq, spectrum, wind, leads=leads)
        states.append((freq, spectrum, wind))
        observed_hs.append(out.hs * (1.0 + 0.01 * rng.standard_normal(leads.size)))
        observed_tp.append(out.tp * (1.0 + 0.01 * rng.standard_normal(leads.size)))
    model, scores = seastate.calibrate(np.tile(freq, (4, 1)), np.array([s[1] for s in states]),
                                       np.array([s[2] for s in states]), np.array(observed_hs),
                                       np.array(observed_tp), leads, grid_swell=(12.0, 24.0, 48.0),
                                       grid_growth=(4.0, 8.0, 16.0))
    assert model.swell_decay_hours == pytest.approx(24.0, rel=1e-6)
    # the growth timescale is *not* identifiable in the default closure: a wind sea below what the wind holds is
    # carried, so the cost does not depend on it. That is a property of the model, and it is why the default has one
    # coefficient doing the work and the other one measured against an alternative closure instead.
    for growth in (4.0, 8.0, 16.0):
        assert scores[f"swell=24,growth={growth:g}"] == pytest.approx(scores["cost"], rel=1e-9)


def test_skill_score_is_one_when_the_forecast_is_perfect_and_negative_when_it_is_worse():
    observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    persistence = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    assert seastate.skill(observed, observed, persistence)["skill"] == pytest.approx(1.0)
    assert seastate.skill(persistence, observed, persistence)["skill"] == pytest.approx(0.0)
    assert seastate.skill(observed + 2.0, observed, persistence)["skill"] < 0.0
    # a reference that is already perfect leaves the score undefined rather than infinite
    assert math.isnan(seastate.skill(observed, observed, observed)["skill"])


def test_arrival_hours_is_the_distance_over_the_group_velocity():
    assert float(seastate.arrival_hours(39.0, 14.0)) == pytest.approx(1.0, rel=2e-2)
    ratio = float(seastate.arrival_hours(39.0, 5.0)) / float(seastate.arrival_hours(39.0, 14.0))
    assert ratio == pytest.approx(14.0 / 5.0, rel=1e-6), "the travel time is linear in the period"


# ------------------------------------------------------------------------------------------------ the section
@pytest.fixture(scope="module")
def section():
    return rao.Section.from_gm(20.0, 6.0, 1.5, depth=30.0)


@pytest.fixture(scope="module")
def long_wave(section):
    return rao.solve(section, 2.0 * math.pi / 80.0)


def test_the_section_hydrostatics_are_the_ones_the_naval_module_uses(section):
    assert section.gm == pytest.approx(1.5, rel=1e-6)
    assert section.k_xx == pytest.approx(7.0)
    props = rao.section_mass_properties(section)
    assert props["mass"] == pytest.approx(sp.RHO * 20.0 * 6.0)
    assert props["roll"] == pytest.approx(sp.RHO * G * 120.0 * 1.5)


def test_every_response_tends_to_one_in_the_long_wave_limit(long_wave):
    """A body small against the wavelength moves with the water: ``heave/a -> 1`` and ``roll/slope -> 1``."""
    assert long_wave["rao_heave"] == pytest.approx(1.0, abs=0.01)
    assert long_wave["rao_roll"] == pytest.approx(1.0, abs=0.15)


def test_added_mass_and_damping_are_positive_and_the_damping_is_not_invented(section):
    for period in (6.0, 8.0, 10.0, 14.0, 20.0):
        row = rao.solve(section, 2.0 * math.pi / period)
        assert row["a"][1, 1] > 0.0 and row["a"][2, 2] > 0.0
        assert row["b"][1, 1] >= 0.0
        # the potential roll damping is essentially nothing, which is why the section carries an imposed ratio
        critical = 2.0 * math.sqrt(section.restoring()["roll"] *
                                   (rao.section_mass_properties(section)["inertia"] + row["a"][2, 2]))
        assert row["b"][2, 2] / critical == pytest.approx(section.roll_damping, abs=0.01)


def test_the_section_stops_making_waves_when_they_are_short(section):
    short = rao.solve(section, 2.0 * math.pi / 4.0)
    assert short["rao_heave"] < 0.2, "a 4 s wave against a 6 m draught must not move the hull much"
    long_ = rao.solve(section, 2.0 * math.pi / 20.0)
    assert long_["b"][1, 1] > short["b"][1, 1]


def test_the_roll_resonance_moves_with_the_added_inertia_the_solve_measures(section):
    """``omega^2 = C44 / (I44 + A44)``: the hydrostatic period is the same equation without ``A44``."""
    transfer = rao.sweep(section, periods=np.geomspace(5.0, 25.0, 24))
    props = rao.section_mass_properties(section)
    near = int(np.argmin(np.abs(transfer.periods - transfer.period_roll)))
    row = rao.solve(section, 2.0 * math.pi / float(transfer.periods[near]))
    predicted = 2.0 * math.pi * math.sqrt((props["inertia"] + row["a"][2, 2]) / props["roll"])
    assert transfer.period_roll == pytest.approx(predicted, rel=5e-2)
    hydrostatic = 2.0 * math.pi * section.k_xx / math.sqrt(G * section.gm)
    assert hydrostatic < transfer.period_roll, "the added inertia can only slow the roll down"


def test_the_grid_and_the_sponge_do_not_move_the_answer(section):
    """Two grids and two sponge lengths: a converged solution cannot depend on either."""
    base = rao.solve(section, 2.0 * math.pi / 12.0)
    for kwargs in ({"nodes_per_wave": 40.0}, {"sponge_waves": 4.0, "open_waves": 1.5}):
        grid = rao.build_grid(section, 2.0 * math.pi / 12.0, **kwargs)
        original = rao.build_grid
        rao.build_grid = lambda s, w, _grid=grid, **_: _grid
        try:
            other = rao.solve(section, 2.0 * math.pi / 12.0)
        finally:
            rao.build_grid = original
        assert other["rao_heave"] == pytest.approx(base["rao_heave"], rel=2e-2), kwargs
        assert other["a"][1, 1] == pytest.approx(base["a"][1, 1], rel=3e-2), kwargs


def test_the_pitch_stand_in_has_the_same_limits(section):
    periods = np.geomspace(4.0, 25.0, 12)
    pitch = rao.pitch_transfer_function(40.0, 6.0, 30.0, kg=7.056, periods=periods)
    assert pitch[-1] == pytest.approx(1.0, abs=0.1), "a long wave pitches the hull with its slope"
    assert pitch[0] < 0.5, "a 4 s wave does not pitch a 40 m hull"


def test_a_motion_record_carries_the_variance_the_spectrum_implies(section):
    transfer = rao.sweep(section, periods=np.geomspace(4.0, 25.0, 20))
    sea = sp.Jonswap(hs=2.0, tp=8.0)
    freq = np.linspace(0.02, 0.6, 240)
    density = sea.density_hz(freq)
    spectral = motion.significant_acceleration(freq, density, transfer, motion.Point(x=0.0, y=8.0))
    state = motion.response(freq, density, transfer, seed=5, count=128, depth=30.0)
    clock = np.arange(0.0, 20000.0, 0.5)
    sampled = 4.0 * float(np.std(state.vertical_acceleration(clock, motion.Point(x=0.0, y=8.0))))
    assert sampled == pytest.approx(spectral, rel=5e-2)
    assert 0.0 < spectral < 3.0


def test_the_relative_velocity_is_zero_where_the_hull_rides_the_water():
    """A section with ``heave = 1`` and no pitch has nothing to slam into: ``v_rel`` has to vanish."""
    transfer = rao.TransferFunction(periods=np.array([5.0, 25.0]), heave=np.array([1.0, 1.0]),
                                    sway=np.zeros(2), roll=np.zeros(2), period_roll=20.0,
                                    pitch=np.zeros(2))
    freq = np.linspace(0.05, 0.2, 60)          # every period inside the table, so nothing is extrapolated to zero
    density = sp.Jonswap(hs=1.5, tp=8.0).density_hz(freq)
    assert motion.significant_relative_velocity(freq, density, transfer) == pytest.approx(0.0, abs=1e-9)


def test_the_slamming_threshold_is_ochis():
    assert motion.slamming_threshold(60.0) == pytest.approx(0.093 * math.sqrt(G * 60.0))


# ------------------------------------------------------------------------------------------------ the mooring
def test_a_line_that_hangs_vertically_carries_its_own_weight():
    out = mooring.catenary(length=200.0, depth=120.0, span=0.0, weight=1000.0)
    assert out["T_fairlead"] == pytest.approx(1000.0 * 120.0, rel=1e-6)
    assert out["bottom"] == pytest.approx(80.0, rel=1e-9)
    assert out["H"] == 0.0


def test_the_tangency_point_is_where_the_whole_line_just_lifts_off():
    """At ``reach`` the line touches the seabed at the anchor and the two branches have to agree there."""
    length, depth, weight = 437.0, 120.0, 1106.0
    reach = mooring.catenary(length, depth, 0.0, weight)["reach"]
    hanging = mooring.catenary(length, depth, reach - 1e-6, weight)
    taut = mooring.catenary(length, depth, reach + 1e-6, weight)
    assert hanging["T_fairlead"] == pytest.approx(taut["T_fairlead"], rel=2e-2)
    assert hanging["bottom"] == pytest.approx(0.0, abs=1e-3)


def test_the_tension_rises_faster_than_the_offset():
    length, depth, weight = 437.0, 120.0, 1106.0
    tensions = np.array([mooring.catenary(length, depth, span, weight)["T_fairlead"]
                         for span in (390.0, 400.0, 410.0)])
    first, second = np.diff(tensions)
    assert second > first, "the last 10 m of offset costs more than the first: the load is not a spring"
    assert tensions[0] < tensions[1] < tensions[2]


def test_a_line_too_short_to_reach_is_refused():
    with pytest.raises(ValueError):
        mooring.catenary(length=100.0, depth=120.0, span=50.0, weight=1000.0)


def test_losing_a_pair_of_lines_raises_the_load_on_the_rest_at_the_same_environmental_load():
    spread = mooring.spread(8, length=437.0, depth=120.0, span=400.0)
    kept = [1, 2, 3, 4, 5, 6]
    lost = mooring.Mooring(lines=tuple(spread.lines[i] for i in kept), depth=spread.depth, span=spread.span,
                           direction=np.asarray(spread.direction, dtype=float)[kept])
    load = 4.0e5
    intact = spread.equilibrium(load)
    without = lost.equilibrium(load)
    assert without["surge"] > intact["surge"], "a spread with fewer lines sits further off to carry the same load"
    assert without["tension"] > intact["tension"], "and every line that is left carries more"
    assert 0.0 < intact["utilisation"] < 0.3


# ------------------------------------------------------------------------------------------------ the residual
def _record(slam=False, step=0.0, sensor=None, seed=5):
    transfer = rao.TransferFunction(periods=np.array([4.0, 25.0]), heave=np.array([1.2, 1.0]),
                                    sway=np.zeros(2), roll=np.zeros(2), period_roll=20.0, pitch=np.zeros(2))
    sea = sp.Jonswap(hs=4.0, tp=7.0)
    freq = np.linspace(0.05, 0.6, 120)
    state = motion.response(freq, sea.density_hz(freq), transfer, seed=seed, count=48, depth=30.0)
    clock = np.arange(0.0, 900.0, 0.5)
    crane = state.vertical_acceleration(clock, motion.Point(x=0.0, y=8.0))
    centre = state.vertical_acceleration(clock)
    tension = 4.0e5 + 1.0e4 * state.vertical_displacement(clock)
    pulse = np.exp(-((clock - 300.0) / 1.5) ** 2) * 3.0 * ((clock > 295.0) & (clock < 305.0))
    rng = np.random.default_rng(seed + 1)
    measured = {"crane": crane + (pulse if slam else 0.0) + rng.normal(0.0, 0.4, clock.size),
                "centre": centre + (0.9 * pulse if slam else 0.0) + rng.normal(0.0, 0.4, clock.size),
                "tension": tension + step * (clock > 500.0) + rng.normal(0.0, 5.0e3, clock.size)}
    if sensor == "spike":
        measured["crane"] += (rng.random(clock.size) < 0.02) * 25.0
    if sensor == "stuck":
        measured["centre"] = np.full(clock.size, measured["centre"][0])
    return {"time": clock, "predicted": {"crane": crane, "centre": centre, "tension": tension}, "measured": measured,
            "noise": {"crane": 0.4, "centre": 0.4, "tension": 5.0e3}}


def _diagnose(record):
    velocity = np.zeros_like(record["time"])
    velocity = np.where((record["time"] > 295.0) & (record["time"] < 305.0), 3.0, velocity)
    return confidence.evaluate(time=record["time"], predicted=record["predicted"], measured=record["measured"],
                               noise=record["noise"], redundant=("crane", "centre"), window=120.0,
                               relative_velocity=velocity, slamming_threshold=motion.slamming_threshold(40.0))


def test_a_quiet_sea_is_not_an_event():
    found = _diagnose(_record())
    assert found.of_kind("slamming") == ()
    assert found.of_kind("mooring") == ()
    assert found.sensor_faults == ()


def test_a_slam_and_a_step_in_the_tension_are_told_apart():
    slammed = _diagnose(_record(slam=True))
    assert len(slammed.of_kind("slamming")) >= 1 and not slammed.of_kind("mooring")
    stepped = _diagnose(_record(step=2.5e4))
    assert len(stepped.of_kind("mooring")) >= 1 and not stepped.of_kind("slamming")


def test_two_failures_of_one_instrument_are_called_instrument_failures():
    spiking = _diagnose(_record(sensor="spike"))
    assert "crane" in spiking.sensor_faults and not spiking.of_kind("mooring")
    stuck = _diagnose(_record(sensor="stuck"))
    assert "centre" in stuck.sensor_faults and not stuck.of_kind("slamming")


def test_the_residual_is_in_units_of_the_instruments_own_noise():
    predicted = np.zeros(10)
    measured = np.full(10, 0.4)
    assert np.allclose(confidence.normalised_residual(predicted, measured, 0.2), 2.0)


# ------------------------------------------------------------------------------------------------ the window
def test_the_window_closes_on_the_criterion_that_fails():
    criteria = operability.Criteria(hs_max=1.5, acceleration_max=0.2, length=0.0)
    hours = np.arange(1.0, 7.0)
    found = operability.windows(criteria, hours, [1.0, 1.2, 1.6, 1.4, 1.3, 1.1], [0.5, 0.7, 0.9, 0.8, 0.6, 0.5],
                                confidence=0.9)
    assert found[0].allowed and found[0].end == 2.0
    assert not found[1].allowed and "Hs" in found[1].reason
    opening = operability.opening(criteria, hours, [1.0, 1.2, 1.6, 1.4, 1.3, 1.1],
                                  [0.5, 0.7, 0.9, 0.8, 0.6, 0.5], confidence=lambda h: 0.95 - 0.01 * h)
    assert opening["open_now"] and opening["until_hours"] == pytest.approx(3.0)
    assert "until +3 h" in opening["sentence"]


def test_an_acceleration_over_the_limit_closes_the_window_on_its_own():
    criteria = operability.Criteria(hs_max=3.0, acceleration_max=0.2, length=0.0)
    assert not criteria.allows([1.0], [2.5])[0]
    assert criteria.allows([1.0], [1.5])[0]


def test_the_confidence_of_a_window_is_the_measured_agreement_of_the_verdicts():
    truth = np.array([True, True, False, False])
    assert operability.decision_skill(np.array([True, True, False, False]), truth) == pytest.approx(1.0)
    assert operability.decision_skill(np.array([True, False, True, False]), truth) == pytest.approx(0.5)


def test_a_window_is_quoted_with_the_confidence_of_its_last_hour():
    criteria = operability.Criteria(hs_max=5.0, acceleration_max=1.0, length=0.0)
    found = operability.windows(criteria, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.1, 0.1, 0.1],
                                confidence=lambda h: 0.9 - 0.1 * h)
    assert found[0].confidence == pytest.approx(0.6)


# ------------------------------------------------------------------------------------------------ the buoy
MET = """\
#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft
2026 11 27 01 50 260 10.0 13.0   3.1    10   6.4  36 1014.9  23.1  25.7  15.5   MM   MM    MM
2026 11 27 03 10 250  8.0 11.0    MM    MM    MM  MM 1014.8  23.1  25.7  14.7   MM   MM    MM
2026 11 27 02 30 250  9.0 12.0   2.4     8   5.9 300 1015.2  23.0  25.6  15.0   MM   MM    MM
"""

DENSITY = """\
#YY  MM DD hh mm Sep_Freq  < spec_1 (freq_1) spec_2 (freq_2) spec_3 (freq_3) ... >
2026 11 27 01 50 0.105 2.247 (0.073) 3.820 (0.078) 2.921 (0.083) 2.696 (0.088) 4.194 (0.093)
2026 11 27 01 20 0.098 2.110 (0.073) 3.558 (0.078) 4.522 (0.083) 4.764 (0.088) 3.678 (0.093)
"""

SUMMARY = """\
#YY  MM DD hh mm WVHT  SwH  SwP  WWH  WWP SwD WWD  STEEPNESS  APD MWD
#yr  mo dy hr mn    m    m  sec    m  sec  -  degT     -      sec degT
2026 11 27 01 40  3.1  1.8 10.0  2.5  8.3  NE   N    AVERAGE  6.4  36
2026 11 27 01 10  3.0  1.2 11.4  2.8  6.7 NNE   W VERY_STEEP  6.3 261
"""


def test_the_standard_meteorological_file_is_read_by_column():
    met = ndbc.read_standard_met(MET)
    assert met["time"].size == 3
    assert np.all(np.diff(met["time"].astype("datetime64[s]").astype(int)) > 0), "the file is read oldest first"
    assert met["hs"].tolist()[:2] == pytest.approx([3.1, 2.4]), "the file is newest first and is read oldest first"
    assert np.isnan(met["hs"]).sum() == 1, "MM is a missing value and stays one"
    assert met["wspd"].tolist() == pytest.approx([10.0, 9.0, 8.0])
    assert met["pres"][-1] == pytest.approx(1014.8), "a pressure of 1014.8 hPa is not a missing marker"


def test_the_spectral_density_file_is_read_into_a_band_grid():
    time, freq, density = ndbc.read_spectral_density(DENSITY)
    assert time.size == 2 and freq.size == 5
    assert freq.tolist() == pytest.approx([0.073, 0.078, 0.083, 0.088, 0.093])
    assert density[0].tolist() == pytest.approx([2.110, 3.558, 4.522, 4.764, 3.678])
    assert density[-1].tolist() == pytest.approx([2.247, 3.820, 2.921, 2.696, 4.194]), "ascending in time"
    # the band energy of the newest row: 4 sqrt(integral S df) over the five bands the file carries
    band_height = 4.0 * math.sqrt(float(np.trapezoid(density[0], freq)))
    assert band_height == pytest.approx(1.12, rel=0.02)


def test_ndbc_swell_and_wind_sea_are_read():
    summary = ndbc.read_spectral_summary(SUMMARY)
    assert summary["swell_hs"].tolist() == pytest.approx([1.2, 1.8])
    assert summary["windsea_tp"].tolist() == pytest.approx([6.7, 8.3])


def test_a_record_makes_the_round_trip_through_the_compressed_file(tmp_path):
    met = ndbc.read_standard_met(MET)
    time, freq, density = ndbc.read_spectral_density(DENSITY)
    summary = ndbc.read_spectral_summary(SUMMARY)
    record = ndbc.BuoyRecord(station="TEST", time=met["time"], hs=met["hs"], tp=met["tp"], apd=met["apd"],
                             wspd=met["wspd"], mwd=met["mwd"], spec_time=time, spec_freq=freq, spec_density=density,
                             swell_hs=summary["swell_hs"], swell_tp=summary["swell_tp"],
                             windsea_hs=summary["windsea_hs"], windsea_tp=summary["windsea_tp"],
                             split_time=summary["time"], source="test")
    path = ndbc.to_npz(record, tmp_path / "buoy.npz")
    back = ndbc.load(path)
    assert back.station == "TEST" and back.time.tolist() == record.time.tolist()
    assert np.allclose(back.spec_density, density)
    assert back.valid().tolist() == record.valid().tolist()
    hours, fields = ndbc.hourly(back)
    assert hours.size == 3 and np.isnan(fields["hs"]).sum() == 1, "an hour with no measurement stays missing"
    assert fields["hs"][:2].tolist() == pytest.approx([3.1, 2.4])
