"""What the vehicle's sensors report, and what the physics says they should have reported.

The residual is the second half: with the *commanded* rotor speeds (an ESC reports them), the nominal airframe and the
measured specific force, Newton-Euler predicts what the accelerometer and the power draw should be. A fault is then a
disagreement between two numbers that should be equal -- not a machine-learning label.

The channels, and why each one is there:

* ``force_x/y/z`` -- the body specific force the nominal model predicts from the commanded rotors, minus what the
  accelerometer measured. In a hover this is ``g (m_true / m_nominal - 1)`` along body ``-z``: a payload makes the
  rotors work harder than the nominal mass explains. A degraded motor or a chipped blade points along *that motor's*
  arm instead, and a gust is horizontal in the world frame -- three different directions, which is the diagnosis.
* ``torque_x/y/z`` -- the body torque the commanded rotors imply, over the inertia, minus the angular acceleration the
  gyro measured. A payload does not make a torque; a motor does.
* ``power`` -- the electrical power the nominal rotor and pack predict, minus what the ESC and the pack report. A
  sagging pack and a chipped blade live here while leaving the force channel alone.
* ``gps``, ``heading``, ``vibration`` -- the three *measurement* checks: the fix against dead reckoning, the
  magnetometer against the integrated gyro, and the accelerometer's high-frequency content against the declared noise.
  A fault in these must not be diagnosed as a change of the vehicle, which is exactly what they are separated for.

Everything is computed from quantities a real flight controller has: the commands it sent, the IMU, the GPS, the pack
telemetry. It never sees the fault, and it never sees the true state.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .model import Airframe, Multirotor, State

__all__ = ["Sensors", "SensorModel", "Residual", "ResidualModel", "CHANNELS", "FORCE", "TORQUE"]

#: The residual channels, in the order the diagnosis reads them.
CHANNELS = ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z", "power", "gps", "heading",
            "vibration")
FORCE = ("force_x", "force_y", "force_z")
TORQUE = ("torque_x", "torque_y", "torque_z")


@dataclass(frozen=True)
class SensorModel:
    """The noise of each sensor, declared. Nothing here is fitted: these are the numbers a datasheet would give."""

    accel_noise_m_s2: float = 0.06
    gyro_noise_rad_s: float = 0.004
    gps_noise_m: float = 0.25
    gps_velocity_noise_m_s: float = 0.06
    magnetometer_noise_rad: float = 0.02
    power_noise_w: float = 0.8
    voltage_noise_v: float = 0.05
    vibration_hz: float = 90.0

    def drift(self) -> dict[str, float]:
        return {"accel_m_s2": self.accel_noise_m_s2, "gyro_rad_s": self.gyro_noise_rad_s,
                "gps_m": self.gps_noise_m, "power_w": self.power_noise_w}


@dataclass
class Sensors:
    """One sample of everything the flight controller has."""

    time: float
    accel: np.ndarray            # body specific force, m/s^2, accelerometer
    gyro: np.ndarray             # body rates, rad/s
    position: np.ndarray         # GPS/baro fused fix, world
    velocity: np.ndarray         # GPS velocity, world
    heading: float               # magnetometer yaw, rad
    power: float                 # ESC + pack telemetry, W
    voltage: float               # pack terminal voltage, V
    rotors: np.ndarray           # commanded rotor speeds, rad/s (what the ESC was told)
    altitude: float


class SensorObservatory:
    """Turns the truth into what the sensors report, including what the fault makes them lie about.

    A physics fault never touches the sensors: a payload changes the truth, not the noise. The sensor faults
    (:mod:`.faults`) are the ones that land here -- the fix jumps, the heading is wrong, the accelerometer vibrates.
    """

    def __init__(self, sensors: SensorModel | None = None, seed: int = 0):
        self.sensors = sensors or SensorModel()
        self.rng = np.random.default_rng(seed)

    def measure(self, state: State, model: Multirotor, command: np.ndarray, telemetry, faults) -> Sensors:
        s = self.sensors
        rotation = state.rotation()
        v_body = rotation.T @ (state.velocity - faults.wind)
        drag = -0.5 * 1.225 * np.array(model.airframe.drag_area_m2) * v_body * np.abs(v_body)
        thrust, _ = model.thrust_and_torque(state, faults.wind)
        specific = (np.array([0.0, 0.0, -float(np.sum(thrust * model.thrust_gain))]) + drag) / model.airframe.mass_kg
        accel = specific + self.rng.normal(0.0, s.accel_noise_m_s2, 3)
        if faults.vibration_m_s2 > 0.0:
            accel = accel + faults.vibration_m_s2 * math.sin(2 * math.pi * s.vibration_hz * state.time) * \
                np.array([0.35, 0.2, 1.0])
        return Sensors(
            time=state.time,
            accel=accel,
            gyro=state.rates + self.rng.normal(0.0, s.gyro_noise_rad_s, 3),
            position=state.position + faults.gps_offset + self.rng.normal(0.0, s.gps_noise_m, 3),
            velocity=state.velocity + self.rng.normal(0.0, s.gps_velocity_noise_m_s, 3),
            heading=state.euler[2] + faults.heading_error_rad + self.rng.normal(0.0, s.magnetometer_noise_rad),
            power=telemetry.power_w + self.rng.normal(0.0, s.power_noise_w),
            voltage=telemetry.voltage_v + self.rng.normal(0.0, s.voltage_noise_v),
            rotors=np.asarray(command, float) * model.airframe.omega_max,
            altitude=state.altitude_m,
        )


@dataclass
class Residual:
    """The named disagreements at one instant."""

    time: float
    values: dict[str, float] = field(default_factory=dict)

    def vector(self) -> np.ndarray:
        return np.array([self.values[c] for c in CHANNELS])

    def force_norm(self) -> float:
        return float(np.linalg.norm([self.values[c] for c in FORCE]))

    def torque_norm(self) -> float:
        return float(np.linalg.norm([self.values[c] for c in TORQUE]))

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


CACHE: dict[tuple, Multirotor] = {}


def _nominal(airframe: Airframe) -> Multirotor:
    """The nominal model, cached: the residual is evaluated every step and rebuilding the rotor table each time is
    half a second of waste per call."""
    key = (airframe.mass_kg, airframe.rotor.radius_m, airframe.rotor.chord_m, airframe.omega_max)
    if key not in CACHE:
        CACHE[key] = Multirotor(airframe)
    return CACHE[key]


class ResidualModel:
    """Newton-Euler on the nominal vehicle, evaluated against the sensors, every step."""

    def __init__(self, airframe: Airframe | None = None, *, window: int = 25, noise_floor: float = 0.06):
        self.airframe = airframe or Airframe()
        self.noise_floor = float(noise_floor)
        self.window = int(window)
        #: the accelerometer's *high-frequency* content: the second difference kills a step (a payload!) and keeps a
        #: 90 Hz oscillation, which is the difference between a sensor that is loose and a vehicle that got heavier
        self._accel_history: deque[np.ndarray] = deque(maxlen=max(3, self.window))
        self._previous_gyro: np.ndarray | None = None
        self._dead_reckoned: np.ndarray | None = None
        self._heading = 0.0

    def reset(self, sensors: Sensors | None = None) -> None:
        self._accel_history.clear()
        self._previous_gyro = None if sensors is None else np.asarray(sensors.gyro, float).copy()
        self._dead_reckoned = None if sensors is None else np.asarray(sensors.position, float).copy()
        self._heading = 0.0 if sensors is None else float(sensors.heading)

    def __call__(self, sensors: Sensors, dt: float) -> Residual:
        af = self.airframe
        nominal = _nominal(af)
        command = np.asarray(sensors.rotors, float)

        # --- force: what the commanded rotors should have done, against what the accelerometer felt
        thrust = nominal.table.thrust_many(command)
        predicted = np.array([0.0, 0.0, -float(thrust.sum())]) / af.mass_kg
        force = predicted - np.asarray(sensors.accel, float)

        # --- torque: the couple the commanded rotors make, over the inertia, against the gyro's angular acceleration
        spin_torque = nominal.table.torque_many(command)
        magnitude = thrust
        tau = np.array([-float(nominal.arms[:, 1] @ magnitude), float(nominal.arms[:, 0] @ magnitude),
                        -float(nominal.spins @ spin_torque)])
        inertia = np.array(af.inertia)
        if self._previous_gyro is None:
            angular = np.zeros(3)
        else:
            angular = (np.asarray(sensors.gyro, float) - self._previous_gyro) / max(dt, 1e-9)
        self._previous_gyro = np.asarray(sensors.gyro, float).copy()
        torque = tau / inertia - np.cross(sensors.gyro, inertia * sensors.gyro) / inertia - angular

        # --- power: what the nominal rotor and pack should have drawn
        power = nominal.table.total_power(command) / af.rotor.drivetrain_efficiency - float(sensors.power)

        # --- the three measurement checks
        if self._dead_reckoned is None:
            self._dead_reckoned = np.asarray(sensors.position, float).copy()
        self._dead_reckoned = self._dead_reckoned + np.asarray(sensors.velocity, float) * dt
        gps = float(np.linalg.norm(np.asarray(sensors.position, float) - self._dead_reckoned))
        self._dead_reckoned = 0.999 * self._dead_reckoned + 0.001 * np.asarray(sensors.position, float)

        self._heading = _wrap(self._heading + float(sensors.gyro[2]) * dt)
        heading = _wrap(float(sensors.heading) - self._heading)
        self._heading = _wrap(self._heading + 0.001 * heading)         # slow trim, so the check does not drift away

        self._accel_history.append(np.asarray(sensors.accel, float).copy())
        vibration = 0.0
        if len(self._accel_history) == self._accel_history.maxlen:
            history = np.array(self._accel_history)
            second_difference = history[2:] - 2.0 * history[1:-1] + history[:-2]
            vibration = float(np.sqrt(np.mean(np.sum(second_difference ** 2, axis=1))))
            # white noise alone gives sqrt(18) sigma_0 of that over three axes (6 sigma^2 per axis, three axes); what
            # is left is the vehicle and its props
            vibration = max(vibration - math.sqrt(18.0) * self.noise_floor, 0.0)

        return Residual(sensors.time, {
            "force_x": float(force[0]), "force_y": float(force[1]), "force_z": float(force[2]),
            "torque_x": float(torque[0]), "torque_y": float(torque[1]), "torque_z": float(torque[2]),
            "power": float(power), "gps": gps, "heading": heading, "vibration": max(vibration, 0.0)})


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
