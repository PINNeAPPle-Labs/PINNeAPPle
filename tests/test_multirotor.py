"""The multirotor of #335, held to what a bench and a flight would show.

What is asserted, in the order the physics builds up:

* **the rotor** -- BEMT's figure of merit lands in the published 0.5-0.8 band, thrust follows the propeller law
  ``T ~ omega^2`` to 2 %, and a chipped tip costs thrust while costing *more* torque per newton, which is the mechanism
  the fault detection rides on;
* **the vehicle** -- at the rotor speed BEMT says it hovers at, the thrust is the weight; the closed loop holds a
  hover, a circle and a climb, so a fault is measured on a flying vehicle and not on a divergence;
* **the faults** -- each one moves the physics it should and an impossible one is refused;
* **the residual and the detectors** -- a payload shows up in the vertical force channel as ``g (m_true/m_nom - 1)``,
  the detectors find it, and a healthy flight is not a fault;
* **the expert** -- the mass is identified from a hover that started two times too heavy, the inertia only from a
  maneuver, the ground effect transfers to an altitude it never saw, the causally estimated wind to a speed it never
  saw, and the confidence horizon is a number of seconds that shortens as the fault grows.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_physics.multirotor.control import Controller, Trajectory, quaternion_error
from pinneapple_physics.multirotor.detect import (
    Calibration,
    CusumDetector,
    Detection,
    KalmanDetector,
    ThresholdDetector,
    calibrate,
    diagnose,
    tune,
)
from pinneapple_physics.multirotor.estimate import (
    EstimateHistory,
    OnlineExpert,
    ResidualCorrection,
    confidence_horizon,
)
from pinneapple_physics.multirotor.faults import Fault, apply_faults, chip_factors
from pinneapple_physics.multirotor.model import Airframe, Battery, Multirotor, RotorTable, State
from pinneapple_physics.multirotor.residuals import CHANNELS
from pinneapple_physics.multirotor.rotor import PROPS, Rotor
from pinneapple_physics.multirotor.sim import Flight, fly

RPM = 2.0 * math.pi / 60.0


# ---------------------------------------------------------------------------------------------------- the rotor
def test_the_figure_of_merit_lands_in_the_published_band():
    """An actuator disk has ``P_ideal/P = 1``; a real propeller of this size is 0.5-0.8, and BEMT has to say so."""
    for name in PROPS:
        rotor = Rotor.from_prop(name)
        for rpm in (3000.0, 5000.0, 7000.0):
            state = rotor.state(rpm * RPM)
            assert 0.5 < state.figure_of_merit < 0.8, f"{name} at {rpm} rpm: FM {state.figure_of_merit:.3f}"
            ideal = state.thrust_n ** 1.5 / math.sqrt(2.0 * 1.225 * math.pi * rotor.radius_m ** 2)
            assert state.power_w >= ideal * 0.99, "a propeller cannot beat the actuator disk"


def test_thrust_follows_the_propeller_law():
    """``T ~ omega^2``: doubling the speed quadruples the thrust, to within 2 % over the whole bench range."""
    rotor = Rotor.from_prop("apc-10x4.7")
    low, high = rotor.state(3000 * RPM), rotor.state(6000 * RPM)
    exponent = math.log(high.thrust_n / low.thrust_n) / math.log(2.0)
    assert exponent == pytest.approx(2.0, abs=0.02)


def test_a_bigger_or_higher_pitched_propeller_makes_more_thrust():
    small = Rotor.from_prop("apc-9x4.5").state(6000 * RPM).thrust_n
    large = Rotor.from_prop("apc-10x4.7").state(6000 * RPM).thrust_n
    assert large > small > 0.0

    coarse = Rotor(radius_m=0.127, pitch_m=0.18, chord_m=0.022).state(6000 * RPM).thrust_n
    assert coarse > large


def test_a_chipped_tip_costs_thrust_and_buys_efficiency():
    """The measurement that corrected the guess: the tip is the least efficient part of the disc, so a chip costs
    thrust (to 82 % of the span intact) and *reduces* the torque per newton (to 94 %), raising the figure of merit.

    It is worth asserting because the obvious story -- a damaged prop works harder -- is the wrong one, and because it
    is why a blade fault cannot be told from a weak motor by the power it draws.
    """
    rotor = Rotor.from_prop("apc-10x4.7")
    omega = rotor.omega_for_thrust(3.68)
    thrust_ratio, torque_ratio = chip_factors(rotor, omega, tip_loss=0.12)
    assert 0.5 < torque_ratio < thrust_ratio < 1.0, "the chip costs more torque than thrust"
    assert torque_ratio / thrust_ratio < 1.0, "so the torque per newton falls"

    efficiencies = [Rotor(**{**rotor.__dict__, "tip_loss": loss}).state(omega).figure_of_merit
                    for loss in (0.0, 0.12, 0.3)]
    assert efficiencies[0] < efficiencies[1] < efficiencies[2], "less span, better efficiency, monotonically"


def test_a_stopped_rotor_makes_nothing_and_a_backwards_one_is_refused():
    rotor = Rotor.from_prop("apc-10x4.7")
    still = rotor.state(0.0)
    assert (still.thrust_n, still.torque_nm, still.power_w, still.figure_of_merit) == (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="backwards"):
        rotor.state(-10.0)


def test_the_table_carries_the_bemt_it_was_built_from():
    rotor = Rotor.from_prop("apc-9x4.5")
    table = RotorTable(rotor, omega_max=900.0)
    for omega in (200.0, 450.0, 700.0, 880.0):
        exact, sampled = rotor.state(omega).thrust_n, table.thrust(omega)
        assert sampled == pytest.approx(exact, rel=0.03), f"at {omega} rad/s"

    assert table.thrust(0.0) == 0.0 and table.omega_for_thrust(0.0) == 0.0
    asked = 3.0
    assert table.thrust(table.omega_for_thrust(asked)) == pytest.approx(asked, rel=1e-3)


# ---------------------------------------------------------------------------------------------------- the battery
def test_the_pack_sags_under_load_and_reports_its_own_losses():
    pack = Battery()
    assert pack.open_circuit(1.0) == pytest.approx(4.2 * 4)
    assert pack.open_circuit(0.0) == pytest.approx(3.5 * 4)
    assert pack.voltage(1.0, 0.0) > pack.voltage(1.0, 20.0) > pack.voltage(1.0, 60.0)

    current = pack.current(200.0, 1.0)
    assert current > 0.0
    # the current solves V_oc I - I^2 R = P, so the pack supplies the rotor power plus its own loss
    assert pack.open_circuit(1.0) * current - current ** 2 * pack.resistance_ohm == pytest.approx(200.0, rel=1e-6)

    worse = Battery(resistance_ohm=3 * pack.resistance_ohm)
    assert worse.current(200.0, 1.0) ** 2 * worse.resistance_ohm > current ** 2 * pack.resistance_ohm


# --------------------------------------------------------------------------------------------------- the vehicle
def test_the_hover_speed_of_the_rotor_holds_the_vehicle_up():
    airframe = Airframe()
    model = Multirotor(airframe)
    state = State(position=np.array([0.0, 0.0, -5.0]), rotors=np.full(4, airframe.hover_omega()))

    thrust, _ = model.thrust_and_torque(state)
    assert float(thrust.sum()) == pytest.approx(airframe.weight_n, rel=0.01)
    telemetry = model.measure(state)
    assert 100.0 < telemetry.power_w < 220.0, "a 1.5 kg quad at hover draws 100-220 W"
    grams_per_watt = airframe.mass_kg * 1000.0 / telemetry.power_w
    assert 7.0 < grams_per_watt < 13.0, f"{grams_per_watt:.1f} g/W is what a quad of this class does"


def test_the_ground_effect_amplifies_the_thrust_only_near_the_floor():
    airframe = Airframe()
    model = Multirotor(airframe)
    rotors = np.full(4, airframe.hover_omega())
    low = float(model.thrust_and_torque(State(position=np.zeros(3), rotors=rotors))[0].sum())
    high = float(model.thrust_and_torque(State(position=np.array([0.0, 0.0, -5.0]), rotors=rotors))[0].sum())
    assert low > high * 1.05


def test_the_closed_loop_holds_a_hover_a_circle_and_a_climb():
    airframe = Airframe()
    model = Multirotor(airframe)
    controller = Controller(model)
    dt = 0.002
    for trajectory, seconds, tolerance in ((Trajectory("hover", altitude=5.0), 8.0, 0.05),
                                           (Trajectory("circle", altitude=5.0, radius=4.0, period_s=20.0), 12.0, 0.9),
                                           (Trajectory("climb", altitude=2.0, climb_rate=0.5), 8.0, 0.35)):
        state = State(position=np.asarray(trajectory.at(0.0)[0], float).copy(),
                      rotors=np.full(4, airframe.hover_omega()))
        errors = []
        for k in range(int(seconds / dt)):
            t = k * dt
            target, velocity, yaw = trajectory.at(t)
            state, _ = model.step(state, controller.command(state, target, velocity, yaw, dt), dt)
            if t > seconds * 0.6:
                errors.append(float(np.linalg.norm(state.position - target)))
        assert float(np.mean(errors)) < tolerance, f"{trajectory.kind}: {np.mean(errors):.3f} m off"


def test_the_mixer_makes_roll_pitch_and_yaw_the_way_it_was_asked():
    airframe = Airframe()
    model = Multirotor(airframe)

    level = model.allocate(airframe.weight_n)
    assert np.allclose(level, level[0], rtol=1e-6), "a symmetric demand is a symmetric command"

    roll = model.allocate(airframe.weight_n, (0.1, 0.0, 0.0))
    assert roll[0] != pytest.approx(roll[1]), "roll is a differential across the arms"
    assert roll[0] * roll[1] > 0.0, "the two motors on each side move together"

    yaw = model.allocate(airframe.weight_n, (0.0, 0.0, 0.05))
    assert not np.allclose(yaw, level, rtol=1e-9), "yaw needs the pairs to split"
    assert np.mean(yaw) == pytest.approx(np.mean(level), rel=0.02), "and it should not cost collective thrust"


def test_the_attitude_error_is_in_the_body_frame():
    """``conj(current) * target``: with a pure yaw offset the error must be a yaw, not a roll and a pitch."""
    half = math.radians(25.0) / 2.0
    target = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    error = quaternion_error(target, np.array([1.0, 0.0, 0.0, 0.0]))
    assert abs(error[0]) < 1e-9 and abs(error[1]) < 1e-9
    # the error is a rotation *vector*: 2 sin(theta/2) for a half-angle of 12.5 degrees
    assert error[2] == pytest.approx(2.0 * math.sin(half), rel=1e-3)


# ---------------------------------------------------------------------------------------------------- the faults
@pytest.mark.parametrize("bad", [
    dict(kind="alien", start_s=1.0),
    dict(kind="payload", start_s=1.0, magnitude=0.0),
    dict(kind="payload", start_s=-1.0, magnitude=0.4),
    dict(kind="motor", start_s=1.0, magnitude=1.5),
    dict(kind="motor", start_s=1.0, magnitude=0.5, motor=7),
    dict(kind="blade", start_s=1.0, magnitude=1.2),
    dict(kind="gust", start_s=2.0, end_s=1.0),
])
def test_an_impossible_fault_is_refused(bad):
    with pytest.raises(ValueError):
        Fault(**bad)


def test_a_fault_is_active_only_inside_its_window():
    fault = Fault("gust", 3.0, 4.0, end_s=6.0)
    assert not fault.active(2.9) and fault.active(3.0) and fault.active(5.9) and not fault.active(6.0)
    assert "gust 4.0 m/s" == fault.label()


def test_a_payload_adds_mass_and_a_motor_takes_thrust_away():
    airframe = Airframe()
    heavy = apply_faults([Fault("payload", 0.0, 0.4)], 1.0, base_airframe=airframe,
                         base_resistance_ohm=airframe.battery.resistance_ohm, rotor=airframe.rotor,
                         omega_hover=airframe.hover_omega())
    assert heavy.mass_kg == pytest.approx(airframe.mass_kg + 0.4)
    assert np.all(heavy.thrust_gain == 1.0)

    weak = apply_faults([Fault("motor", 0.0, 0.6, motor=1)], 1.0, base_airframe=airframe,
                        base_resistance_ohm=airframe.battery.resistance_ohm, rotor=airframe.rotor,
                        omega_hover=airframe.hover_omega())
    assert weak.thrust_gain[1] == pytest.approx(0.6) and weak.thrust_gain[0] == 1.0
    assert weak.mass_kg == airframe.mass_kg


# ----------------------------------------------------------------------------------------- the residual, flying
def _flight(fault=None, *, duration: float = 7.0, dt: float = 0.005, seed: int = 0) -> Flight:
    return fly(() if fault is None else (fault,), duration=duration, dt=dt, seed=seed)


def _before_after(flight: Flight, onset: float):
    before = flight.residual[(flight.time > 1.5) & (flight.time < onset)]
    after = flight.residual[flight.time > onset + 1.5]
    return before.mean(axis=0), after.mean(axis=0)


def test_a_payload_is_the_mass_ratio_in_the_vertical_force():
    """In a hover the residual along body ``-z`` is ``g (m_true / m_nominal - 1)``: 0.4 kg on 1.5 kg is 2.6 m/s^2."""
    airframe = Airframe()
    flight = _flight(Fault("payload", 3.0, 0.4))
    before, after = _before_after(flight, 3.0)
    vertical = CHANNELS.index("force_z")
    assert abs(before[vertical]) < 0.4
    assert after[vertical] == pytest.approx(-airframe.gravity * 0.4 / airframe.mass_kg, abs=0.6)


def test_a_healthy_flight_leaves_the_residual_at_the_noise():
    flight = _flight(None)
    calibration = calibrate([flight])
    settled = flight.residual[flight.time > 1.5]
    assert np.all(np.abs(calibration.sigmas(settled.mean(axis=0))) < 3.0)
    assert calibration.std[CHANNELS.index("force_z")] < 0.6


def test_the_detectors_find_the_payload_and_a_healthy_flight_is_not_a_fault():
    healthy = [_flight(None, seed=s) for s in (0, 1)]
    calibration = calibrate(healthy)
    faulted = _flight(Fault("payload", 3.0, 0.4))

    for detector_class in (ThresholdDetector, KalmanDetector, CusumDetector):
        detector = tune(detector_class, calibration, healthy)
        assert not any(detector.run(f).detected for f in healthy), f"{detector_class.__name__} false-alarmed"
        detection = detector.run(faulted)
        assert detection.detected, f"{detector_class.__name__} missed a 0.4 kg payload"
        # the fault is a step, so the earliest honest answer is the first sample at or after it
        assert detection.time is not None and detection.time >= 3.0
        assert "force_z" in detection.channels, detection.channels


def test_the_diagnosis_names_the_payload_and_its_evidence():
    healthy = [_flight(None, seed=s) for s in (0, 1)]
    calibration = calibrate(healthy)
    detector = tune(ThresholdDetector, calibration, healthy)
    faulted = _flight(Fault("payload", 3.0, 0.4))

    verdict = diagnose(faulted, detector.run(faulted), calibration)
    assert verdict["label"] == "payload"
    assert verdict["evidence"]["force_z"] < -5.0
    assert verdict["confidence"] > 0.2
    assert diagnose(faulted, type(detector.run(faulted))(False, None), calibration)["label"] == "no fault detected"


def test_calibration_subtracts_the_offset_the_flight_really_has():
    time = np.arange(0.0, 10.0, 0.01)
    biased = np.zeros((len(time), len(CHANNELS)))
    rng = np.random.default_rng(0)
    biased[:, CHANNELS.index("force_z")] = -0.3 + rng.normal(0.0, 0.05, len(time))
    flight = Flight(time, biased, np.zeros((len(time), 3)), np.zeros(len(time)), np.zeros(len(time)))

    calibration = calibrate([flight])
    assert calibration.mean[CHANNELS.index("force_z")] == pytest.approx(-0.3, abs=0.02)
    # the calibrated mean maps to zero sigma, whatever the offset was
    assert abs(calibration.sigmas(calibration.mean)[CHANNELS.index("force_z")]) < 1e-6
    assert set(calibration.to_dict()) == set(CHANNELS)


# ------------------------------------------------------------------------------- the maneuver and the ramp
def test_the_weave_moves_the_way_its_analytic_velocity_says():
    """The trajectory hands the controller a position and a velocity; if they disagree the loop is chasing a ghost."""
    trajectory = Trajectory("weave", altitude=5.0, radius=3.0, period_s=8.0)
    h = 1e-5
    for t in (0.0, 1.3, 2.7, 5.1, 7.4):
        position, velocity, _ = trajectory.at(t)
        ahead, behind = trajectory.at(t + h)[0], trajectory.at(t - h)[0]
        assert velocity == pytest.approx((ahead - behind) / (2 * h), abs=1e-5), f"at t = {t}"
    # and it is a maneuver: the heading moves, and the vehicle does not stay in one place
    yaws = [trajectory.at(t)[2] for t in np.linspace(0.0, 8.0, 40)]
    assert max(yaws) > 0.4 > min(yaws)
    assert np.ptp([trajectory.at(t)[0][0] for t in np.linspace(0.0, 8.0, 40)]) > 4.0


def test_a_ramping_fault_reaches_its_magnitude_and_starts_from_the_identity():
    """A motor ramps from full efficiency *down*, everything else ramps from nothing *up*; a step has no ramp."""
    gust = Fault("gust", 2.0, 6.0, ramp_s=4.0)
    assert gust.magnitude_at(2.0) == pytest.approx(0.0)
    assert gust.magnitude_at(4.0) == pytest.approx(3.0)
    assert gust.magnitude_at(6.0) == pytest.approx(6.0)
    assert gust.magnitude_at(9.0) == pytest.approx(6.0)

    motor = Fault("motor", 2.0, 0.7, ramp_s=2.0)
    assert motor.magnitude_at(2.0) == pytest.approx(1.0), "a degraded motor is intact before it degrades"
    assert motor.magnitude_at(3.0) == pytest.approx(0.85)
    assert Fault("gust", 2.0, 6.0).magnitude_at(2.0) == pytest.approx(6.0), "no ramp is a step"

    with pytest.raises(ValueError, match="negative"):
        Fault("gust", 2.0, 6.0, ramp_s=-1.0)


def test_a_ramping_fault_is_not_a_step_in_the_flight():
    """The loop used to rebuild the fault state only at its edges, so a ramp arrived all at once."""
    ramping = fly((Fault("payload", 2.0, 0.5, ramp_s=4.0),), duration=8.0, dt=0.01, seed=0)
    vertical = CHANNELS.index("force_z")
    early = ramping.residual[(ramping.time > 2.5) & (ramping.time < 3.0), vertical].mean()
    late = ramping.residual[ramping.time > 7.0, vertical].mean()
    assert abs(late) > 3.0 * abs(early), f"early {early:.2f}, late {late:.2f} m/s^2"


# ---------------------------------------------------------------------------------------------------- the expert
def _expert_flight(trajectory, *, duration: float = 12.0, dt: float = 0.01, seed: int = 0) -> Flight:
    return fly(duration=duration, dt=dt, seed=seed, trajectory=trajectory)


def test_the_expert_recovers_a_mass_it_started_wrong_about_by_a_factor_of_two():
    """Two times too heavy to within 2 %, and it says so with an uncertainty far smaller than the error it had."""
    flight = _expert_flight(Trajectory("hover", altitude=5.0))
    expert = OnlineExpert(mass_factor=2.0)
    assert expert.rls.theta[0] == pytest.approx(0.5), "the prior says the vehicle is twice as heavy"
    history = expert.run(flight)

    # in a hover the accelerometer gives ``T/m`` outright, so one sample is the whole answer: the mass does not
    # converge over seconds, it converges on the first reading, and what takes seconds is the drag and the inertia
    assert history.at(0.0).mass_kg == pytest.approx(1.5, rel=0.01), "identified from the first sample"
    assert history.settle_time(tolerance=0.01) == 0.0
    error = history.at(11.99).mass_kg / 1.5 - 1.0
    assert abs(error) < 0.005, f"{100 * error:+.2f} % off the true mass after twelve seconds"
    assert history.at(11.99).mass_std_kg < 0.05 * 1.5, "and it is not claiming that from a wide posterior"


def test_both_sides_of_the_equation_are_filtered_at_the_motor_lag():
    """Filtering the command at the motor's own time constant is what makes it the flown rotor speed: without it the
    thrust the expert divides by is one lag behind the accelerometer, and the mass pays for it."""
    flight = _expert_flight(Trajectory("weave", altitude=5.0, radius=3.0, period_s=8.0), duration=12.0)
    raw = OnlineExpert(angular_window_s=0.04)
    raw.lowpass.tau = 0.0
    unfiltered = raw.run(flight).at(11.99).mass_kg / 1.5 - 1.0
    filtered = OnlineExpert().run(flight).at(11.99).mass_kg / 1.5 - 1.0
    assert abs(filtered) <= abs(unfiltered) + 1e-3, f"filtered {100 * filtered:+.2f} %, raw {100 * unfiltered:+.2f} %"


def test_the_inertia_is_identified_by_a_maneuver_and_unknown_after_a_hover():
    """A hover leaves the torque equation at ``0 = 0``: nothing moves, so nothing says what it costs to move. The
    recursive covariance is what says so -- 20 % or more of the prior becomes the answer, and the maneuver's does not.
    """
    hover = OnlineExpert().run(_expert_flight(Trajectory("hover", altitude=5.0)))
    weave = OnlineExpert().run(_expert_flight(Trajectory("weave", altitude=5.0, radius=3.0, period_s=8.0)))

    hovers_uncertainty = hover.at(11.99).inertia_std[2] / hover.at(11.99).inertia[2]
    weaves_uncertainty = weave.at(11.99).inertia_std[2] / weave.at(11.99).inertia[2]
    assert hovers_uncertainty > 0.2, f"the hover claims {100 * hovers_uncertainty:.0f} % -- it should be unsure"
    assert weaves_uncertainty < hovers_uncertainty / 4.0, "the maneuver has to be the one that knows"

    for axis, truth in enumerate((0.011, 0.011, 0.021)):
        assert weave.at(11.99).inertia[axis] == pytest.approx(truth, rel=0.15), f"axis {axis}"


def test_an_estimate_needs_a_flight_with_a_sensor_stream():
    bare = Flight(np.arange(3.0), np.zeros((3, len(CHANNELS))), np.zeros((3, 3)), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="sensor stream"):
        OnlineExpert().run(bare)


# ------------------------------------------------------------------------------------------ the learned residual
def test_the_learned_ground_effect_corrects_a_flight_it_never_saw():
    """Fit on two flights through the skirt, measured on a third: the force residual has to come down, and a flight at
    5 m -- where there is no ground effect -- has to be left alone."""
    class Bob(Trajectory):
        def at(self, t):
            position, velocity, yaw = super().at(t)
            w = 2 * math.pi / self.period_s
            position, velocity = position.copy(), velocity.copy()
            position[2] += -self.vertical * math.sin(w * t)
            velocity[2] = -self.vertical * w * math.cos(w * t)
            return position, velocity, yaw

    def bob(vertical, altitude, period, seed):
        trajectory = Bob("hover", altitude=altitude, period_s=period)
        trajectory.vertical = vertical
        return _expert_flight(trajectory, duration=8.0, seed=seed)

    training = [bob(0.20, 0.25, 6.0, 10), bob(0.30, 0.30, 5.0, 11)]
    model = ResidualCorrection.fit(training)

    held_out = bob(0.22, 0.22, 5.5, 20)
    before = _force_rms(held_out, held_out.residual)
    after = _force_rms(held_out, model.apply(held_out, wind=None))
    assert after < 0.8 * before, f"{before:.3f} -> {after:.3f} m/s^2"

    calm = _expert_flight(Trajectory("hover", altitude=5.0), duration=8.0, seed=21)
    assert _force_rms(calm, model.apply(calm, wind=None)) == pytest.approx(_force_rms(calm, calm.residual),
                                                                         rel=0.05)


def test_the_wind_is_a_state_that_transfers_and_a_coefficient_that_does_not():
    """The same correction on a gust it never saw, and the version of it fitted across flights: one works because it
    estimates the wind of the day, the other cannot, and it damages a flight with no wind at all."""
    from pinneapple_physics.multirotor.faults import Fault

    def gust(speed, seed):
        return fly((Fault("gust", 1.0, speed, direction=(1.0, 0.3, 0.0)),), duration=8.0, dt=0.01, seed=seed,
                   trajectory=Trajectory("hover", altitude=5.0))

    model = ResidualCorrection.fit([gust(6.0, 20), gust(10.0, 21)], pool_wind=True)
    unseen = gust(8.0, 30)
    calm = fly(duration=8.0, dt=0.01, seed=31, trajectory=Trajectory("hover", altitude=5.0))

    before = _force_rms(unseen, unseen.residual)
    assert _force_rms(unseen, model.apply(unseen)) < 0.5 * before, "the causal wind has to transfer"

    calm_before = _force_rms(calm, calm.residual)
    pooled = _force_rms(calm, model.apply(calm, wind=model.pooled_wind))
    assert pooled > 1.5 * calm_before, "a wind fitted as a coefficient makes a calm flight worse, and that is the point"
    assert _force_rms(calm, model.apply(calm)) == pytest.approx(calm_before, rel=0.10)


def _force_rms(flight, residual, *, after: float = 2.0) -> float:
    mask = flight.time > after
    return float(np.sqrt(np.mean(np.sum(residual[mask][:, :3] ** 2, axis=1))))


# --------------------------------------------------------------------------------------------- the horizon
def _horizon_flight(level_now: float, rate: float, *, duration: float = 5.0, dt: float = 0.01) -> Flight:
    """A flight whose force residual ramps at ``rate`` and sits at ``level_now`` at its last sample: a synthetic
    stream, so the horizon is tested on the arithmetic it is made of instead of on a flight that takes ten seconds."""
    n = int(round(duration / dt))
    time = np.arange(n) * dt
    residual = np.zeros((n, len(CHANNELS)))
    residual[:, CHANNELS.index("force_z")] = -(level_now + rate * (time - time[-1]))
    return Flight(time, residual, np.zeros((n, 3)), np.zeros(n), np.zeros(n))


def _zero_calibration() -> Calibration:
    """A calibration that subtracts nothing: the horizon reads the residual, and here the residual is synthetic."""
    return Calibration(np.zeros(len(CHANNELS)), np.ones(len(CHANNELS)))


def _no_sigma_history(flight, sigma: float) -> EstimateHistory:
    n = len(flight.time)
    return EstimateHistory(flight.time.copy(), np.full(n, 1.5), np.zeros((n, 3)), np.zeros((n, 3)),
                           np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3)), np.full(n, sigma))


def test_the_horizon_is_the_time_the_residual_takes_to_reach_the_threshold():
    """``(threshold - level - sigma) / rate``, in seconds, with the threshold supplied in m/s^2."""
    flight = _horizon_flight(0.50, 0.05)
    history = _no_sigma_history(flight, 0.0)
    horizon = confidence_horizon(flight, history, _zero_calibration(), threshold_m_s2=1.00)
    assert horizon.rate_m_s2_s == pytest.approx(0.05, rel=0.15), horizon.rate_m_s2_s
    assert horizon.level_m_s2 == pytest.approx(0.50, abs=0.03)
    assert horizon.seconds == pytest.approx((1.00 - horizon.level_m_s2) / 0.05, rel=0.25)


def test_the_horizon_shortens_when_the_fault_grows_and_the_model_uncertainty_is_a_margin():
    """A bigger fault is a faster level, so fewer seconds; a wider model is less headroom, so fewer seconds still."""
    def judge(level, rate, sigma):
        flight = _horizon_flight(level, rate)
        return confidence_horizon(flight, _no_sigma_history(flight, sigma), _zero_calibration(),
                                  threshold_m_s2=1.00, sigma_m_s2=sigma)

    small, large, wide = judge(0.50, 0.02, 0.02), judge(0.50, 0.08, 0.02), judge(0.50, 0.02, 0.30)
    assert small.seconds > large.seconds
    assert wide.seconds < small.seconds, "a model that is unsure has to answer for less time"


def test_a_model_already_past_the_threshold_answers_for_no_time_and_a_still_one_forever():
    over = _horizon_flight(1.20, 0.05)
    assert confidence_horizon(over, _no_sigma_history(over, 0.0), _zero_calibration(), threshold_m_s2=1.00).seconds == 0.0

    still = _horizon_flight(0.50, 0.0)
    horizon = confidence_horizon(still, _no_sigma_history(still, 0.0), _zero_calibration(), threshold_m_s2=1.00)
    assert horizon.capped and horizon.seconds == pytest.approx(60.0)


def test_the_diagnosis_carries_the_horizon_it_was_given():
    flight = _horizon_flight(0.50, 0.05)
    calibration = _zero_calibration()
    horizon = confidence_horizon(flight, _no_sigma_history(flight, 0.0), calibration, threshold_m_s2=1.00)
    verdict = diagnose(flight, Detection(False, None), calibration, horizon=horizon)
    assert verdict["label"] == "no fault detected"
    assert verdict["horizon_s"] == pytest.approx(horizon.seconds)
    assert verdict["horizon"]["threshold_m_s2"] == pytest.approx(1.00)
