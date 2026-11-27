"""pinneapple_physics.terraform: the Mars constants and their sources, the insolation against the solar constant,
the frost point against the phase diagram, the energy and the CO2 conserved, the cap thermostat, the ensemble's
spread, and the verdict the whole package exists to check.

Every tolerance below is a measurement first, then a number with room around it: the comment above each says what the
run gave when the tolerance was written.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinneapple_physics.terraform import co2, ebm, globe, mars
from pinneapple_physics.terraform import terraforming as tf


# --------------------------------------------------------------------------------------------- the constants
def test_every_constant_carries_its_source():
    fields = set(mars.Mars.__dataclass_fields__)                       # noqa: SLF001 -- the point of the test
    assert set(mars.SOURCES) == fields
    for name, text in mars.SOURCES.items():
        assert len(text) > 20, f"{name} has a source field that says nothing"
    assert mars.source("co2_triple_point_k").startswith("NIST")
    assert mars.source("bond_albedo").startswith("NASA Mars Fact Sheet")


def test_a_constant_without_a_source_is_refused():
    with pytest.raises(TypeError):
        mars.Mars(albedo_of_the_sky=0.5)                               # not a field: the dataclass refuses it
    saved = dict(mars.SOURCES)
    try:
        del mars.SOURCES["mass_kg"]
        with pytest.raises(ValueError, match="mass_kg"):
            mars.Mars()
    finally:
        mars.SOURCES.update(saved)
    assert mars.Mars()                                                    # and the guard is back where it was


def test_the_atmosphere_of_the_fact_sheet_comes_out_of_the_pressure():
    # p A / g = 636 Pa * 1.4496e14 m2 / 3.73 = 2.47e16 kg, against the fact sheet's "~2.5e16 kg" (1.2 %)
    mass = mars.MARS.mass_from_pressure(mars.MARS.surface_pressure_pa)
    assert mass == pytest.approx(mars.MARS.atmosphere_mass_kg, rel=0.02)


def test_the_column_of_the_present_atmosphere_is_the_published_15_g_per_cm2():
    # Jakosky & Edwards: the present atmosphere is "about 15 g CO2 cm-2"; p/g here gives 17.1, the 12 % being the
    # 0.951 CO2 mole fraction and their rounded pressure
    column = mars.column_mass(mars.MARS.surface_pressure_pa) / 10.0
    assert column == pytest.approx(15.0, rel=0.20)


def test_the_solar_longitude_of_perihelion_is_the_published_250_99():
    # derived from the pole (RA 317.681, Dec 52.887) and the J2000 elements, not remembered: 250.99246
    assert mars.perihelion_solar_longitude() == pytest.approx(250.99, abs=0.05)


def test_the_bundled_facts_card_agrees_with_the_constants():
    card = mars.fact_sheet_numbers()
    assert card["diameter_km"]["value"] == pytest.approx(2.0 * mars.MARS.radius_m / 1000.0, rel=0.01)
    assert card["distance_sun_km"]["value"] / mars.AU_KM == pytest.approx(mars.MARS.semimajor_au, rel=0.01)


# --------------------------------------------------------------------------------------------- the insolation
def test_the_annual_insolation_conserves_the_solar_constant():
    # (1/4 pi) int S d(area) / S_mean = 1 / sqrt(1 - e^2); the 720-sample orbit gives 147.22 against 147.17 W/m2,
    # 4e-4 relative -- the error of the orbit quadrature, not of the physics
    grid = ebm.Grid(72)
    mean = float(np.sum(ebm.annual_insolation(grid.latitude_deg) * grid.weights))
    expected = mars.MARS.solar_irradiance_w_m2 / (4.0 * np.sqrt(1.0 - mars.MARS.eccentricity ** 2))
    assert mean == pytest.approx(expected, rel=2e-3)


def test_the_daily_insolation_at_an_equinox_is_the_beam_over_pi():
    # at Ls = 0 the Sun is over the equator: the daily mean at the equator is S/pi exactly, 186.59 W/m2 at 1.5237 au
    value = ebm.daily_insolation(0.0, 0.0, mars.MARS.semimajor_au)
    beam = float(mars.MARS.solar_irradiance_at(mars.MARS.semimajor_au))
    assert float(value) == pytest.approx(beam / np.pi, rel=1e-12)
    # and the fact sheet's own 586.2 W/m2 at Mars is the same beam to its rounding: 586.27 computed, 1e-4 apart
    assert beam == pytest.approx(mars.MARS.solar_irradiance_w_m2, rel=1e-3)


def test_the_winter_pole_gets_nothing_at_the_solstice():
    assert float(ebm.daily_insolation(-87.5, 90.0, mars.MARS.semimajor_au)) == 0.0
    summer = float(ebm.daily_insolation(87.5, 90.0, mars.MARS.semimajor_au))
    assert summer == pytest.approx(mars.MARS.solar_irradiance_w_m2 * np.sin(np.radians(25.19)), rel=2e-3)


def test_the_southern_summer_gets_more_because_perihelion_is_near_it():
    # eccentricity 0.0934 with perihelion at Ls = 251 deg: the south polar annual mean beats the north
    grid = ebm.Grid(72)
    annual = ebm.annual_insolation(grid.latitude_deg)
    south, north = float(annual[0]), float(annual[-1])
    assert south > north
    assert south / north == pytest.approx(1.14, abs=0.03)              # measured 1.138


# --------------------------------------------------------------------------------------------- the frost point
def test_the_frost_point_at_the_present_pressure_is_the_winter_cap_temperature():
    # 146.50 K at 6.36 mbar: the temperature the seasonal cap is observed to form at
    assert float(co2.frost_point(636.0)) == pytest.approx(146.5, abs=0.4)


def test_the_sublimation_curve_reproduces_the_boiling_point_of_dry_ice():
    # Clausius-Clapeyron from the triple point gives 1.073e5 Pa at 194.7 K against the measured 1.013e5: 6 %
    assert float(co2.saturation_pressure(194.7)) == pytest.approx(101_325.0, rel=0.08)


def test_the_frost_point_round_trips_and_rises_with_pressure():
    pressure = np.array([1.0, 100.0, 636.0, 5_000.0, 1.0e5])
    assert co2.saturation_pressure(co2.frost_point(pressure)) == pytest.approx(pressure, rel=1e-9)
    frost = co2.frost_point(pressure)
    assert np.all(np.diff(frost) > 0.0)
    # dT/dlnp = R T^2 / dH ~ 7.0 K per e-fold at 146 K: the thermostat's stiffness, and the reason a greenhouse
    # warming does not melt the cap
    slope = (float(co2.frost_point(636.0 * np.e)) - float(co2.frost_point(636.0)))
    assert slope == pytest.approx(7.4, abs=0.3)                        # 7.43 K per e-fold at 146 K


def test_the_reservoirs_are_the_published_ones():
    assert co2.RESERVOIRS["polar_cap"].pressure_pa == pytest.approx(900.0)
    assert co2.RESERVOIRS["regolith"].pressure_pa == pytest.approx(4000.0)          # "< 0.04 bar"
    assert co2.RESERVOIRS["clathrate"].pressure_pa == pytest.approx(15_000.0)       # "<< 0.150 bar"
    assert co2.RESERVOIRS["carbonate_deep"].accessible is False
    # the present atmosphere plus the polar cap is the paper's "less than 15 mbar total"
    assert (co2.RESERVOIRS["atmosphere"].pressure_pa + co2.RESERVOIRS["polar_cap"].pressure_pa) < 1600.0
    # and the representative maximum a scenario could reach is 20 mbar
    assert co2.inventory("mobilizable") == pytest.approx(2000.0)


# --------------------------------------------------------------------------------------------- the greenhouse
def test_the_greenhouse_passes_through_the_two_published_ends():
    # Jakosky & Edwards: 20 mbar warms by "less than 10 K" and 1 bar by the ~60 K liquid water needs
    assert float(ebm.greenhouse_warming(2_000.0)) == pytest.approx(9.0, abs=0.1)
    assert float(ebm.greenhouse_warming(1.0e5)) == pytest.approx(60.0, abs=0.1)
    # the third point is a prediction, and it is the greenhouse Mars is observed to have: 5.16 K at 6.36 mbar
    assert float(ebm.greenhouse_warming(636.0)) == pytest.approx(5.2, abs=0.2)


def test_the_greenhouse_saturates_rather_than_compounding():
    # the optical depth is a fraction of the outgoing radiation, so 10x the pressure is far less than 10x the kelvin
    assert float(ebm.greenhouse_warming(6_360.0)) < 3.5 * float(ebm.greenhouse_warming(636.0))
    tau = ebm.greenhouse_optical_depth(np.array([636.0, 2_000.0, 1.0e5]))
    assert np.all(np.diff(tau) > 0.0)
    assert float(tau[2]) == pytest.approx(1.735, abs=0.01)


# --------------------------------------------------------------------------------------------- the energy balance
def test_the_transport_conserves_the_global_mean_exactly():
    grid = ebm.Grid(72)
    model = ebm.ZonalEBM(grid=grid)
    rng = np.random.default_rng(0)
    profile = 210.0 + 40.0 * rng.standard_normal((3, grid.bands))
    divergence = model.divergence(profile)
    assert np.allclose(np.sum(divergence * grid.weights, axis=-1), 0.0, atol=1e-9)


def test_without_transport_or_greenhouse_the_profile_is_the_radiative_equilibrium():
    grid = ebm.Grid(72)
    model = ebm.ZonalEBM(ebm.Climate(transport=0.0, greenhouse_one_bar_k=0.0), grid)
    temperature = model.settle(636.0, np.zeros(grid.bands), years=60.0, steps_per_year=12).reshape(-1)
    expected = ((1.0 - 0.250) * model.insolation / ebm.STEFAN_BOLTZMANN) ** 0.25
    assert temperature == pytest.approx(expected, rel=2e-3)


def test_the_model_reproduces_the_mean_temperature_of_the_fact_sheet():
    # 214.29 K at 6.36 mbar with no forcing, against the fact sheet's "~214 K": 0.3 K, and nothing in the greenhouse
    # law was fitted to this point
    grid = ebm.Grid(72)
    model = tf.TerraformModel(ebm.Climate(), grid, atmosphere_pa=636.0, cap_pa=0.0, regolith_pa=0.0)
    run = model.run(tf.Scenario(years=400.0, ramp_years=1e9))
    assert float(run.mean_temperature[-1, 0]) == pytest.approx(214.0, abs=2.0)


def test_the_grid_is_equal_area():
    grid = ebm.Grid(72)
    assert grid.weights.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.allclose(grid.weights, grid.weights[0])
    assert np.allclose(grid.faces()[[0, -1]], 0.0)


# --------------------------------------------------------------------------------------------- the coupled model
def test_no_co2_is_created_or_destroyed():
    model = tf.TerraformModel(ebm.Climate(), ebm.Grid(36), regolith_pa=2_000.0, regolith_years=500.0)
    run = model.run(tf.Scenario(years=200.0, ramp_years=50.0, greenhouse_flux_w_m2=4.0, cap_darkening=0.1))
    total = run.pressure_pa + run.cap_pa + run.regolith_pa
    assert np.allclose(total, model.total_pa[0], rtol=1e-9)
    assert np.all(run.pressure_pa >= 0.0)


def test_the_cap_holds_the_pole_at_the_frost_point_while_it_melts():
    # the thermostat: a band with ice on it cannot rise above the frost point, the energy goes into latent heat
    model = tf.TerraformModel(ebm.Climate(), ebm.Grid(72))
    run = model.run(tf.Scenario(years=200.0, ramp_years=50.0, greenhouse_flux_w_m2=6.0, cap_darkening=0.2))
    pole = run.temperature[:, 0, 0]
    frost = co2.frost_point(run.pressure_pa[:, 0])
    band = run.cap_bands[:, 0, 0]                                      # the ice in the pole's own band
    melting = (band[1:] > 1.0) & (np.diff(band) < -0.1)                # the years that band is giving ice up
    assert melting.sum() > 10
    assert np.all(np.abs(pole[1:][melting] - frost[1:][melting]) < 1.5)  # measured: within 0.05 K


def test_the_cap_retreats_and_the_pressure_rises_with_it():
    model = tf.TerraformModel(ebm.Climate(), ebm.Grid(72))
    run = model.run(tf.TERRAFORMING_SCENARIO)
    assert run.cap_pa[-1, 0] < run.cap_pa[0, 0]                        # 9.00 -> 2.25 mbar
    assert run.pressure_pa[-1, 0] > run.pressure_pa[0, 0]              # 6.05 -> 12.80 mbar
    assert run.cover[-1, 0] < 0.5 * run.cover[0, 0]                    # 1.00 -> 0.25
    assert run.mean_temperature[-1, 0] > run.mean_temperature[0, 0]    # 210.96 -> 217.26 K


def test_the_pressure_cannot_exceed_the_inventory():
    model = tf.TerraformModel(ebm.Climate(), ebm.Grid(36))
    run = model.run(tf.Scenario(years=200.0, ramp_years=20.0, greenhouse_flux_w_m2=40.0, cap_darkening=0.4))
    assert np.all(run.pressure_pa <= model.total_pa[0] + 1e-9)


def test_a_planet_that_lost_its_caps_does_not_get_them_back():
    # the model is bistable: starting warm melts the caps in a decade, and the dark polar ground keeps it that way
    model = tf.TerraformModel(ebm.Climate(), ebm.Grid(72))
    cold = model.run(tf.Scenario(years=300.0, ramp_years=1e9))
    bare = model.initial_state(caps_full=False)
    bare.temperature = np.full_like(bare.temperature, 260.0)          # a warm planet with no ice on it
    warm = model.run(tf.Scenario(years=300.0, ramp_years=1e9), state=bare)
    assert cold.cap_pa[-1, 0] > 500.0                                  # 6.75 mbar of the 9 mbar cap
    assert warm.cap_pa[:, 0].max() < 1.0                               # and a warm Mars never grows one back


def test_the_cap_edge_follows_the_area_that_is_frosted():
    assert float(globe.cap_edge_latitude(1.0)) == pytest.approx(70.0, abs=1e-6)
    assert float(globe.cap_edge_latitude(0.0)) == pytest.approx(90.0, abs=1e-6)
    assert float(globe.cap_edge_latitude(0.25)) == pytest.approx(80.0, abs=0.05)      # measured 80.04


# --------------------------------------------------------------------------------------------- the ensemble
def test_the_ensemble_reproduces_itself():
    first = tf.run_ensemble(members=16, seed=3, grid=ebm.Grid(36))
    second = tf.run_ensemble(members=16, seed=3, grid=ebm.Grid(36))
    assert np.array_equal(first.trajectory.mean_temperature, second.trajectory.mean_temperature)
    other = tf.run_ensemble(members=16, seed=4, grid=ebm.Grid(36))
    assert not np.array_equal(first.trajectory.mean_temperature, other.trajectory.mean_temperature)


def test_the_spread_grows_from_nothing_and_the_horizon_is_a_century():
    ensemble = tf.run_ensemble(members=64, seed=0)
    spread = ensemble.spread
    assert spread[0] == pytest.approx(0.0, abs=1e-9)                   # every member starts from the same planet
    assert spread[-1] > 10.0 * max(spread[0], 1e-3)
    horizon = ensemble.horizon()
    assert np.isfinite(horizon)
    assert 20.0 < horizon < 120.0                                     # measured 43.96 years at the 2 K threshold
    assert spread[np.argmin(np.abs(ensemble.trajectory.years - horizon))] == pytest.approx(2.0, abs=0.15)


def test_the_horizon_moves_the_way_a_threshold_should():
    years = np.array([0.0, 10.0, 20.0, 30.0])
    spread = np.array([0.0, 1.0, 2.0, 3.0])
    assert tf.trust_horizon(years, spread, 1.0) == pytest.approx(10.0)
    assert tf.trust_horizon(years, spread, 1.5) == pytest.approx(15.0)          # interpolated
    assert np.isnan(tf.trust_horizon(years, spread, 4.0))                       # never reached
    assert tf.trust_horizon(years, spread, 0.0) == pytest.approx(0.0)


def test_the_confidence_bar_empties_exactly_at_the_threshold():
    assert float(globe.trust(0.0, 2.0)) == 1.0
    assert float(globe.trust(1.0, 2.0)) == pytest.approx(0.5)
    assert float(globe.trust(2.0, 2.0)) == 0.0
    assert float(globe.trust(9.0, 2.0)) == 0.0


# --------------------------------------------------------------------------------------------- the verdict
def test_terraforming_is_not_possible_with_present_day_co2():
    """The conclusion Jakosky & Edwards publish, which the model has to reproduce rather than the other way round.

    Every accessible reservoir at its published upper bound is 67.4 mbar, and it warms Mars by 11 K of the ~60 K
    liquid water needs. The only inventory that reaches the melting point is the ~1 bar of deep carbonates their
    figure 3 lists as inaccessible -- which is their point, not a way around it.
    """
    accessible = co2.inventory("accessible_top")
    model = tf.TerraformModel(ebm.Climate(), ebm.Grid(36), atmosphere_pa=accessible, cap_pa=0.0, regolith_pa=0.0)
    run = model.run(tf.Scenario(years=800.0, ramp_years=1e9))
    warming = float(run.mean_temperature[-1, 0] - run.mean_temperature[0, 0])
    assert warming < 0.35 * tf.TARGET_WARMING_K                       # measured 11.0 K of 60
    assert float(ebm.greenhouse_warming(accessible)) < 0.5 * tf.TARGET_WARMING_K


def test_the_representative_maximum_is_twenty_millibars_and_a_few_kelvin():
    model = tf.TerraformModel(ebm.Climate(), ebm.Grid(36), atmosphere_pa=co2.inventory("mobilizable"),
                              cap_pa=0.0, regolith_pa=0.0)
    run = model.run(tf.Scenario(years=800.0, ramp_years=1e9))
    assert float(run.pressure_pa[-1, 0]) / 100.0 == pytest.approx(20.0, rel=1e-6)
    assert float(run.mean_temperature[-1, 0]) < 273.15 - 40.0


def test_the_deep_carbonates_would_do_it_and_the_paper_says_they_are_out_of_reach():
    one_bar = co2.inventory("deep_carbonates")
    assert one_bar > 1.0e5                                            # 1.22 bar
    assert co2.RESERVOIRS["carbonate_deep"].accessible is False
    assert float(ebm.greenhouse_warming(one_bar)) > 0.9 * tf.TARGET_WARMING_K      # 66.0 K
    mobilizable = {k for k, v in co2.RESERVOIRS.items() if v.accessible}
    assert "carbonate_deep" not in mobilizable


# --------------------------------------------------------------------------------------------- the picture
def test_the_synthetic_topography_says_it_is_synthetic():
    lat = np.linspace(-90.0, 90.0, 46)
    lon = np.linspace(-180.0, 180.0, 91)
    relief = globe.synthetic_topography(*np.meshgrid(lat, lon, indexing="ij"))
    assert relief.shape == (len(lat), len(lon))
    assert relief.max() > 15.0 and relief.min() < -5.0                 # a volcano positive, a basin negative
    assert "synthetic" in (globe.synthetic_topography.__doc__ or "").lower()


def test_the_assumptions_are_written_down():
    text = " ".join(ebm.ASSUMPTIONS).lower()
    for subject in ("season", "dust", "greenhouse", "regolith", "transport"):
        assert subject in text
    assert len(ebm.ASSUMPTIONS) >= 7
