"""The ascent model against its references: the 1976 standard atmosphere, the rotation of the pad, the two-body
bookkeeping, and one flight that has to end on the target orbit.

    pytest tests/test_launch.py -q
"""
import math

import numpy as np
import pytest

from pinneapple_simulation.launch import Site, Target, atmosphere, vehicle, verification
from pinneapple_simulation.launch.ascent import (
    Earth,
    circular_speed,
    launch_azimuth_deg,
    orbit_elements,
    simulate_ascent,
)

SITE = Site("Tanegashima", 30.4, 130.97)
TARGET = Target(500.0)


@pytest.fixture(scope="module")
def flight():
    return simulate_ascent(vehicle.small_launcher(), SITE, TARGET, dt=0.25, t_max=3000.0)


# ------------------------------------------------------------------ atmosphere
def test_atmosphere_matches_the_published_table():
    rep = verification.atmosphere_checks()
    assert rep["pass"], [c["name"] for c in rep["checks"] if not c["pass"]]
    assert rep["worst_rel_error"] < 2e-3


def test_atmosphere_is_monotone_and_empty_above_orbit():
    h = np.array([0.0, 1e3, 11e3, 20e3, 50e3, 86e3, 200e3])
    rho = atmosphere.density(h)
    assert np.all(np.diff(rho) < 0.0)
    assert atmosphere.density(200e3) < 1e-8          # above the insertion altitude drag is irrelevant
    assert atmosphere.geopotential(atmosphere.geometric(50e3)) == pytest.approx(50e3, rel=1e-12)


def test_sound_speed_at_sea_level_and_tropopause():
    assert float(atmosphere.speed_of_sound(0.0)) == pytest.approx(340.294, abs=0.02)
    assert float(atmosphere.speed_of_sound(atmosphere.geometric(11e3))) == pytest.approx(295.07, abs=0.02)


# ------------------------------------------------------------------ geometry and the pad
def test_pad_moves_east_at_the_rotation_speed():
    v = SITE.velocity(Earth())
    assert np.linalg.norm(v) == pytest.approx(Earth().omega * Earth().radius * math.cos(math.radians(30.4)),
                                              rel=1e-12)
    lam = math.radians(SITE.lon_deg)                  # east at 130.97 deg E is (-sin, cos) in ECI x/y
    assert v[0] == pytest.approx(-Earth().omega * Earth().radius * math.cos(math.radians(30.4)) * math.sin(lam), rel=1e-12)
    assert v[1] == pytest.approx(Earth().omega * Earth().radius * math.cos(math.radians(30.4)) * math.cos(lam), rel=1e-12)


def test_eastward_launch_reaches_the_site_latitude():
    assert launch_azimuth_deg(30.4, 30.4) == pytest.approx(90.0, abs=1e-9)
    assert launch_azimuth_deg(30.4, 45.0) < 90.0      # a higher inclination needs a northward azimuth
    assert launch_azimuth_deg(5.2, 97.0) > 180.0      # a retrograde inclination heads south-east
    with pytest.raises(ValueError):
        launch_azimuth_deg(60.0, 20.0)                # an inclination below the latitude is not reachable


def test_circular_orbit_at_the_target_altitude():
    r = Earth().radius + 500e3
    v = circular_speed(r)
    assert v == pytest.approx(7616.6, abs=1.0)        # sqrt(mu / r), mu = 3.986004418e14
    orb = orbit_elements(np.array([r, 0.0, 0.0]), np.array([0.0, v, 0.0]))
    assert orb["e"] < 1e-12 and orb["perigee_km"] == pytest.approx(500.0, abs=1e-6)
    assert orb["period_s"] == pytest.approx(5668.0, abs=2.0)      # 94.5 min


def test_orbit_elements_agree_with_the_astro_module():
    from pinneapple_physics.astro.orbits import elements_from_state

    r = np.array([Earth().radius + 500e3, 0.0, 0.0])
    v = np.array([0.0, circular_speed(r[0]) * math.cos(math.radians(30.4)), circular_speed(r[0]) * math.sin(math.radians(30.4))])
    mine = orbit_elements(r, v, Earth().mu, Earth().radius)
    ref = elements_from_state(r / 1e3, v / 1e3, Earth().mu / 1e9)
    assert mine["a_m"] / 1e3 == pytest.approx(ref["a_km"], rel=1e-12)
    assert mine["inclination_deg"] == pytest.approx(ref["i_deg"], abs=1e-9)


# ------------------------------------------------------------------ the vehicle
def test_vehicle_is_a_concept_that_can_leave_the_pad():
    veh = vehicle.small_launcher()
    assert veh.thrust_to_weight_liftoff > 1.2
    assert veh.payload_kg == 200.0
    assert 0.01 < veh.payload_fraction < 0.05
    assert 8_000.0 < veh.delta_v_vac_m_s < 12_000.0
    scaled = veh.with_propellant_scale(1.1)
    assert scaled.liftoff_mass_kg > veh.liftoff_mass_kg
    assert scaled.stages[0].dry_mass_kg == veh.stages[0].dry_mass_kg


def test_thrust_falls_with_ambient_pressure():
    st = vehicle.small_launcher().stages[0]
    assert st.thrust_n(101325.0) == pytest.approx(st.thrust_sl_n)
    assert st.thrust_n(0.0) == pytest.approx(st.thrust_vac_n)
    assert st.mass_flow_kg_s(101325.0) == pytest.approx(st.mass_flow_vac_kg_s)


def test_drag_coefficient_peaks_transonically():
    cd = vehicle.cd_of_mach
    assert cd(0.3) < cd(1.05)
    assert cd(1.05) > cd(3.0) > cd(8.0)
    assert vehicle.cd_of_mach(np.array([0.3, 1.05, 3.0])).shape == (3,)


# ------------------------------------------------------------------ the flight
def test_the_flight_reaches_the_target_orbit(flight):
    orb = flight.final_orbit()
    assert orb["perigee_km"] == pytest.approx(TARGET.altitude_km, abs=30.0)
    assert orb["apogee_km"] == pytest.approx(TARGET.altitude_km, abs=30.0)
    assert orb["e"] < 5e-3
    assert orb["inclination_deg"] == pytest.approx(SITE.lat_deg, abs=0.2)


def test_the_velocity_bookkeeping_closes(flight):
    assert abs(flight.losses["closure_m_s"]) < 2e-3 * flight.losses["dv_thrust_m_s"]   # quadrature at dt=0.25
    assert flight.losses["dv_thrust_m_s"] > 8_000.0
    assert flight.losses["dv_gravity_m_s"] > flight.losses["dv_drag_m_s"]


def test_the_stages_are_flown_in_order(flight):
    assert len(flight.events["separation_s"]) == 2
    assert np.all(np.diff(flight.mass) <= 1e-9)
    # -2 marks the coast to apogee in the record, -1 the states after cut-off
    assert flight.stage[0] == 0 and flight.stage[-1] == 2
    assert set(np.unique(flight.stage)) <= {-2, -1, 0, 1, 2} and -2 in set(np.unique(flight.stage))
    assert flight.events["cutoff_s"] > flight.events["separation_s"][-1]
    assert 10e3 < flight.summary()["max_q_pa"] < 60e3


def test_the_checks_pass_on_the_flight(flight):
    rep = verification.run(flight, site=SITE, target=TARGET)
    assert rep["pass"], rep["failed"]
    assert rep["n_checks"] > 20


def test_the_scene_and_the_animation_have_one_frame_per_sample(flight):
    from pinneapple_simulation.launch.visuals import flight_scene

    sc = flight_scene(flight, view="globe", n_frames=40)
    assert sc.animation is not None
    assert len(sc.animation.times) == 40
    assert "vehicle" in sc.animation.tracks
    assert sc.n_faces() > 1000                     # the globe
