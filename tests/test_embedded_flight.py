"""Tests for ``pinneapple_physics.embedded_flight`` -- the multirotor of #335, Newton-Euler and the sensor check as
one ONNX graph (#350).

What is asserted, in the order a flight computer cares about it:

* **it is the same physics**: one RK4 step of the graph against ``Multirotor.step``, and the seven residual channels
  against ``ResidualModel`` on sensors the library's own ``SensorObservatory`` produced -- not a made-up state;
* **it decides**: a hover with a healthy accelerometer is inside the physics and not suspect, a payload sets
  ``off_physics``, a loose IMU sets ``suspect``;
* **it is a straight line**: no ``If``/``Loop``/``Scan``, seven fixed inputs, thirteen fixed outputs;
* **the parts that do not export are named**: :func:`export_probe` is run, and its three findings are asserted --
  the lookup is refused, the rotor and the branch export *and freeze*, which is worse than a refusal;
* **it is measured**: the latency is reported per call, on one thread.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from pinneapple_physics.embedded_flight import (  # noqa: E402 - after the importorskip
    CHAINS,
    STATE_LAYOUT,
    FlightRuntime,
    FlightSpec,
    export_flight,
    export_probe,
    graph_summary,
)
from pinneapple_physics.multirotor import (  # noqa: E402
    Airframe,
    Multirotor,
    ResidualModel,
    RotorTable,
    SensorObservatory,
    State,
    apply_faults,
)


def _state_vector(state: State) -> np.ndarray:
    return np.concatenate([state.position, state.velocity, state.attitude, state.rates, state.rotors,
                           [state.soc]]).astype(np.float32)


def _airframe(mass_kg: float | None = None) -> Airframe:
    return Airframe() if mass_kg is None else Airframe(mass_kg=mass_kg)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    spec = FlightSpec(dt=0.002)
    path = export_flight(spec, tmp_path_factory.mktemp("onnx") / "flight.onnx")
    return path, spec


@pytest.fixture(scope="module")
def runtime(exported):
    path, _ = exported
    return FlightRuntime(path, threads=1)


# ------------------------------------------------------------------------------------------------ the contract
def test_the_graph_is_a_straight_line_with_the_contract_the_layout_declares(exported):
    path, _ = exported
    summary = graph_summary(path)

    assert summary.is_straight_line is True and summary.control_flow == ()
    assert summary.inputs == ("state", "command", "accel", "gyro", "gyro_previous", "power", "accel_history")
    assert summary.outputs == ("next_state", "thrust_predicted", "pack_power", "current", "voltage", "residual_force",
                               "residual_torque", "residual_power", "residual", "trust", "off_physics", "vibration",
                               "suspect")
    assert summary.bytes > 0 and "Conv" not in summary.op_types, "the rotor table is weights, not a convolution"
    assert len(STATE_LAYOUT) == 6 and len(CHAINS) == 7


@pytest.mark.parametrize("spec", [dict(dt=0.0), dict(dt=-1.0), dict(history=2)])
def test_an_impossible_contract_is_refused(spec):
    with pytest.raises(ValueError):
        FlightSpec(**spec)


def test_the_tolerances_come_from_the_declared_noise_or_from_the_caller():
    spec = FlightSpec(dt=0.002)
    assert spec.force_limit == pytest.approx(3.0 * spec.sensors.accel_noise_m_s2)
    assert spec.torque_limit == pytest.approx(3.0 * spec.sensors.gyro_noise_rad_s / 0.002)
    calibrated = FlightSpec(dt=0.002, force_tolerance=0.01, vibration_tolerance=2.0)
    assert calibrated.force_limit == pytest.approx(0.01) and calibrated.vibration_limit == pytest.approx(2.0)


# ------------------------------------------------------------------------------------------------ the physics
def test_one_step_of_the_dynamics_matches_the_numpy_airframe(exported, runtime):
    path, spec = exported
    af = _airframe()
    model = Multirotor(af, table=RotorTable(af.rotor, omega_max=af.omega_max * 1.2))
    state = State(position=np.array([0.0, 0.0, -5.0]), rotors=np.full(4, af.hover_omega()))
    command = np.full(4, 0.55)

    reference, telemetry = model.step(state, command, spec.dt)
    out = runtime.step(_state_vector(state), command, np.zeros(3), np.zeros(3), np.zeros(3), np.array([0.0]),
                       np.zeros((spec.history, 3)))

    expected = np.concatenate([reference.position, reference.velocity, reference.attitude, reference.rates,
                               reference.rotors, [reference.soc]])
    scale = np.maximum(np.abs(expected), 1.0)
    assert float(np.max(np.abs(out.next_state - expected) / scale)) < 1e-6, "float32 against float64 on 18 numbers"
    assert out.pack_power == pytest.approx(telemetry.power_w, rel=1e-5)
    assert out.current == pytest.approx(telemetry.current_a, rel=1e-5)
    assert out.voltage == pytest.approx(telemetry.voltage_v, rel=1e-5)


def test_the_ground_effect_is_in_the_graph_not_in_a_branch(exported, runtime):
    """At 10 cm the thrust is amplified by ``1 + k exp(-h/h0)``; a trace-time ``if`` would lose it (see the probe)."""
    _, spec = exported
    af = _airframe()
    model = Multirotor(af)
    near = State(position=np.array([0.0, 0.0, -0.1]), rotors=np.full(4, af.hover_omega()))
    far = State(position=np.array([0.0, 0.0, -5.0]), rotors=np.full(4, af.hover_omega()))

    low, _ = model.step(near, np.full(4, 0.55), spec.dt)
    high, _ = model.step(far, np.full(4, 0.55), spec.dt)
    from_graph = runtime.step(_state_vector(near), np.full(4, 0.55), np.zeros(3), np.zeros(3), np.zeros(3),
                              np.array([0.0]), np.zeros((spec.history, 3))).next_state

    assert abs(float(from_graph[5]) - float(low.velocity[2])) < abs(float(from_graph[5]) - float(high.velocity[2]))
    assert float(low.velocity[2]) < float(high.velocity[2]), "the ground pushes up: the near vehicle accelerates less"


def test_the_residual_channels_match_the_library_residual_model(exported, runtime):
    path, spec = exported
    af = _airframe()
    model = Multirotor(af)
    observatory = SensorObservatory(seed=3)
    residual_model = ResidualModel(af)
    state = State(position=np.array([0.0, 0.0, -5.0]), rotors=np.full(4, af.hover_omega()))
    command = np.full(4, 0.52)
    telemetry = model.measure(state)
    nominal = apply_faults((), 0.0, base_airframe=af, base_resistance_ohm=af.battery.resistance_ohm, rotor=af.rotor,
                           omega_hover=af.hover_omega())

    reading = observatory.measure(state, model, command, telemetry, nominal)
    residual_model.reset(reading)
    reference = residual_model(reading, spec.dt)
    out = runtime.step(_state_vector(state), command, reading.accel, reading.gyro, reading.gyro,
                       np.array([reading.power]), np.repeat(reading.accel[None, :], spec.history, axis=0))

    force = np.array([reference.values[c] for c in CHAINS[:3]])
    torque = np.array([reference.values[c] for c in CHAINS[3:6]])
    assert np.allclose(out.residual_force, force, atol=1e-4)
    assert np.allclose(out.residual_torque, torque, atol=1e-3)
    assert out.residual_power == pytest.approx(reference.values["power"], abs=1e-3)


# ------------------------------------------------------------------------------------------------ the decision
def test_a_healthy_hover_is_inside_the_physics_and_not_suspect(runtime, exported):
    _, spec = exported
    af = _airframe()
    model = Multirotor(af)
    state = State(position=np.array([0.0, 0.0, -5.0]), rotors=np.full(4, af.hover_omega()))
    telemetry = model.measure(state)
    nominal = apply_faults((), 0.0, base_airframe=af, base_resistance_ohm=af.battery.resistance_ohm, rotor=af.rotor,
                           omega_hover=af.hover_omega())
    hover = np.full(4, af.hover_omega() / af.omega_max)          # the command that holds this state
    reading = SensorObservatory(seed=5).measure(state, model, hover, telemetry, nominal)

    out = runtime.step(_state_vector(state), hover, reading.accel, reading.gyro, reading.gyro,
                       np.array([reading.power]), np.repeat(reading.accel[None, :], spec.history, axis=0))

    assert out.off_physics is False and out.suspect is False
    assert 0.0 <= out.trust <= 1.0 and out.residual < 1.0


def test_a_payload_is_off_physics_and_a_loose_imu_is_suspect(runtime, exported):
    """The two flags read different evidence: the vehicle's mass, and the accelerometer's own high-frequency content."""
    _, spec = exported
    af = _airframe()
    model = Multirotor(af)
    state = State(position=np.array([0.0, 0.0, -5.0]), rotors=np.full(4, af.hover_omega()))
    telemetry = model.measure(state)
    hover = np.full(4, af.hover_omega() / af.omega_max)

    def fly(faults):
        current = apply_faults(faults, 0.0, base_airframe=af, base_resistance_ohm=af.battery.resistance_ohm,
                               rotor=af.rotor, omega_hover=af.hover_omega())
        heavy = Multirotor(Airframe(mass_kg=current.mass_kg), table=model.table, thrust_gain=current.thrust_gain,
                           torque_gain=current.torque_gain)
        observatory = SensorObservatory(seed=7)
        reading = observatory.measure(state, heavy, hover, telemetry, current)
        history = np.array([observatory.measure(State(position=state.position, rotors=state.rotors,
                                                      time=k * spec.dt), heavy, hover, telemetry, current).accel
                            for k in range(spec.history)])
        return runtime.step(_state_vector(state), hover, reading.accel, reading.gyro, reading.gyro,
                            np.array([reading.power]), history)

    from pinneapple_physics.multirotor import Fault

    payload = fly((Fault("payload", 0.0, magnitude=0.6),))
    loose = fly((Fault("vibration", 0.0, magnitude=2.0),))

    assert payload.off_physics is True and abs(payload.residual_force[2]) > 0.1
    assert loose.suspect is True and loose.vibration > 0.0


# ------------------------------------------------------------------------------------------------ the measurement
def test_the_latency_is_measured_on_one_thread(runtime, exported):
    _, spec = exported
    latency = runtime.latency(*spec.hover(), n=100)

    assert latency["threads"] == 1.0
    assert latency["median_ms"] > 0.0 and latency["p95_ms"] >= latency["median_ms"]
    assert latency["hz"] == pytest.approx(1000.0 / latency["median_ms"], rel=1e-6)


# ------------------------------------------------------------------------------------------ the parts that do not
def test_the_parts_that_do_not_export_are_named_and_their_cost_measured(tmp_path):
    findings = export_probe(tmp_path / "probe")
    by_piece = {f["piece"]: f for f in findings}

    assert len(findings) == 3
    lookup = next(f for f in findings if "lookup by index" in f["piece"])
    assert lookup["result"] == "refused" and "searchsorted" in lookup["detail"]
    for frozen in findings:
        if frozen["result"] == "exported and frozen":
            assert any(word in frozen["detail"] for word in ("missing", "not one of them"))
    assert len(by_piece) == 3


def test_a_frozen_graph_is_worse_than_a_refused_one(tmp_path):
    """The rotor inside the graph exports and loses its input: the file answers one number for any vehicle state."""
    findings = export_probe(tmp_path / "probe")
    rotor = next(f for f in findings if "BEMT" in f["piece"])

    assert rotor["result"] == "exported and frozen"
    assert "no input" in rotor["detail"] or "not one of them" in rotor["detail"]
