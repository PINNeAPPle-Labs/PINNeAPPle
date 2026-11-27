"""The 6-DoF airframe: four BEMT rotors, a battery that sags, drag, wind and ground effect.

The state is the one a flight controller estimates -- position, velocity, attitude, body rates -- plus what the vehicle
knows about itself: the four rotor speeds and how much charge is left. Every fault in :mod:`.faults` enters here and
nowhere else, which is what lets the residual of :mod:`.residuals` name it.

Frame conventions: body ``x`` forward, ``y`` right, ``z`` down; thrust along body ``-z``; attitude is a unit quaternion
``(w, x, y, z)`` rotating body to world; ``position[2] = -altitude``.

The rotor model is BEMT (:class:`~pinneapple_physics.multirotor.rotor.Rotor`), which costs milliseconds per call and
cannot live inside a 500 Hz loop, so :class:`RotorTable` tabulates it once and interpolates -- the same physics,
sampled: BEMT thrust follows ``T ~ omega^2`` to within 2 %, so the sampling is not an approximation of the answer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .rotor import Rotor

__all__ = ["Battery", "Airframe", "Multirotor", "State", "Telemetry", "RotorTable", "QUAD_X", "G"]

G = 9.80665

#: A quad-X: motor -> (x forward, y right, spin). ``x``/``y`` are the signs of the arm; the spin signs are what give
#: the airframe yaw authority.
QUAD_X = {"FR": (1.0, 1.0, 1.0), "FL": (1.0, -1.0, -1.0), "RR": (-1.0, 1.0, -1.0), "RL": (-1.0, -1.0, 1.0)}
MOTORS = tuple(QUAD_X)


class RotorTable:
    """The BEMT rotor, tabulated on ``(omega, climb)`` and interpolated: the same numbers, cheap enough for 500 Hz."""

    def __init__(self, rotor: Rotor, *, omega_max: float = 1300.0, climb=(-8.0, 8.0), n_omega: int = 27,
                 n_climb: int = 9):
        self.rotor = rotor
        self.omega_max = float(omega_max)
        self.omegas = np.linspace(0.0, omega_max, n_omega)
        self.climbs = np.linspace(climb[0], climb[1], n_climb)
        self._thrust = np.empty((n_climb, n_omega))
        self._torque = np.empty((n_climb, n_omega))
        for i, c in enumerate(self.climbs):
            for j, w in enumerate(self.omegas):
                s = rotor.state(float(w), float(c))
                self._thrust[i, j] = s.thrust_n
                self._torque[i, j] = s.torque_nm

    def _interp(self, grid: np.ndarray, omega: float, climb: float) -> float:
        w = min(max(omega, 0.0), self.omegas[-1])
        c = min(max(climb, self.climbs[0]), self.climbs[-1])
        j = min(max(int(np.searchsorted(self.omegas, w)) - 1, 0), len(self.omegas) - 2)
        i = min(max(int(np.searchsorted(self.climbs, c)) - 1, 0), len(self.climbs) - 2)
        tw = (w - self.omegas[j]) / (self.omegas[j + 1] - self.omegas[j])
        tc = (c - self.climbs[i]) / (self.climbs[i + 1] - self.climbs[i])
        return float((1 - tc) * ((1 - tw) * grid[i, j] + tw * grid[i, j + 1])
                     + tc * ((1 - tw) * grid[i + 1, j] + tw * grid[i + 1, j + 1]))

    def thrust(self, omega: float, climb: float = 0.0) -> float:
        return self._interp(self._thrust, omega, climb)

    def thrust_many(self, omegas, climb: float = 0.0) -> np.ndarray:
        """The same interpolation for an array of rotor speeds, in one numpy pass (the flight loop calls it 4x a step)."""
        w = np.clip(np.asarray(omegas, float), 0.0, self.omegas[-1])
        c = min(max(climb, self.climbs[0]), self.climbs[-1])
        i = min(max(int(np.searchsorted(self.climbs, c)) - 1, 0), len(self.climbs) - 2)
        tc = (c - self.climbs[i]) / (self.climbs[i + 1] - self.climbs[i])
        row = (1 - tc) * self._thrust[i] + tc * self._thrust[i + 1]
        return np.interp(w, self.omegas, row)

    def torque_many(self, omegas, climb: float = 0.0) -> np.ndarray:
        w = np.clip(np.asarray(omegas, float), 0.0, self.omegas[-1])
        c = min(max(climb, self.climbs[0]), self.climbs[-1])
        i = min(max(int(np.searchsorted(self.climbs, c)) - 1, 0), len(self.climbs) - 2)
        tc = (c - self.climbs[i]) / (self.climbs[i + 1] - self.climbs[i])
        row = (1 - tc) * self._torque[i] + tc * self._torque[i + 1]
        return np.interp(w, self.omegas, row)

    def torque(self, omega: float, climb: float = 0.0) -> float:
        return self._interp(self._torque, omega, climb)

    def power(self, omega: float, climb: float = 0.0) -> float:
        return self.torque(omega, climb) * max(omega, 0.0)

    def total_power(self, omegas, climb: float = 0.0) -> float:
        """Aerodynamic power of every rotor, summed."""
        return float(sum(self.power(float(w), climb) for w in omegas))

    def omega_for_thrust(self, thrust_n: float, climb: float = 0.0) -> float:
        """The rotor speed that produces ``thrust_n`` on this table, by bisection (the mixer's inverse)."""
        if thrust_n <= 0.0:
            return 0.0
        row = self._row(self._thrust, climb)
        if row[-1] <= thrust_n:
            return float(self.omegas[-1])
        # inverting the table by interpolating T -> omega: the forward map is monotone and piecewise linear, so this is
        # the inverse to the table's own resolution and costs one lookup instead of a 24-step bisection per rotor per
        # step -- which is what the controller does, four times, ten thousand times a flight
        return float(np.interp(thrust_n, row, self.omegas))

    def _row(self, grid: np.ndarray, climb: float) -> np.ndarray:
        """The line of the table at ``climb``, interpolated between the two nearest rows."""
        c = min(max(climb, self.climbs[0]), self.climbs[-1])
        i = min(max(int(np.searchsorted(self.climbs, c)) - 1, 0), len(self.climbs) - 2)
        tc = (c - self.climbs[i]) / (self.climbs[i + 1] - self.climbs[i])
        return (1 - tc) * grid[i] + tc * grid[i + 1]

    def dq_dt(self, omega: float) -> float:
        """``dQ/dT`` at ``omega``: how much yaw torque a newton of differential thrust buys."""
        h = 1.0
        dt = self.thrust(omega + h) - self.thrust(omega - h)
        dq = self.torque(omega + h) - self.torque(omega - h)
        return dq / dt if abs(dt) > 1e-12 else 0.0


@dataclass(frozen=True)
class Battery:
    """A pack that sags: ``V = V_oc(SoC) - I R``, with the open-circuit voltage falling as the charge goes."""

    cells: int = 4
    capacity_ah: float = 5.0
    resistance_ohm: float = 0.040           # the whole pack, not per cell
    v_full: float = 4.2
    v_empty: float = 3.5

    @property
    def energy_wh(self) -> float:
        return self.cells * 3.7 * self.capacity_ah

    def open_circuit(self, soc: float) -> float:
        soc = min(max(soc, 0.0), 1.0)
        return self.cells * (self.v_empty + (self.v_full - self.v_empty) * soc)

    def voltage(self, soc: float, current_a: float) -> float:
        """The terminal voltage under load -- the sag a detector reads as "the pack is unhappy"."""
        return max(self.open_circuit(soc) - abs(current_a) * self.resistance_ohm, 0.5 * self.cells * self.v_empty)

    def current(self, power_w: float, soc: float) -> float:
        """The current that draws ``power_w`` at the terminal voltage of this state of charge."""
        if power_w <= 0.0:
            return 0.0
        v = self.open_circuit(soc)
        return 0.5 * (v - math.sqrt(max(v * v - 4.0 * self.resistance_ohm * power_w, 0.0))) / self.resistance_ohm


@dataclass
class State:
    """The state a flight controller estimates, plus the rotor speeds and the charge."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))   # (w, x, y, z)
    rates: np.ndarray = field(default_factory=lambda: np.zeros(3))                          # body, rad/s
    rotors: np.ndarray = field(default_factory=lambda: np.zeros(4))                         # rad/s
    soc: float = 1.0
    time: float = 0.0

    def copy(self) -> State:
        return State(self.position.copy(), self.velocity.copy(), self.attitude.copy(), self.rates.copy(),
                     self.rotors.copy(), float(self.soc), float(self.time))

    @property
    def altitude_m(self) -> float:
        return float(-self.position[2])

    def rotation(self) -> np.ndarray:
        """Body-to-world rotation matrix from the quaternion."""
        w, x, y, z = self.attitude
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    @property
    def euler(self) -> tuple[float, float, float]:
        """``(roll, pitch, yaw)`` in radians, for reporting only -- the dynamics never needs it."""
        r = self.rotation()
        pitch = math.asin(min(max(-r[2, 0], -1.0), 1.0))
        return math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0])


@dataclass(frozen=True)
class Telemetry:
    """What the battery and the ESCs report at one instant."""

    power_w: float = 0.0
    current_a: float = 0.0
    voltage_v: float = 0.0
    thrust_n: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"power_w": self.power_w, "current_a": self.current_a, "voltage_v": self.voltage_v,
                "thrust_n": self.thrust_n}


@dataclass(frozen=True)
class Airframe:
    """The vehicle: mass, inertia, arm geometry, rotors, drag and the motor lag."""

    mass_kg: float = 1.5
    inertia: tuple[float, float, float] = (0.011, 0.011, 0.021)       # kg m^2, body axes
    arm_m: float = 0.225                                              # centre to motor
    rotor: Rotor = field(default_factory=lambda: Rotor.from_prop("apc-10x4.7"))
    drag_area_m2: tuple[float, float, float] = (0.030, 0.030, 0.080)   # Cd * A per body axis
    motor_tau_s: float = 0.06
    omega_max: float = 1100.0
    battery: Battery = field(default_factory=Battery)
    #: ground effect: thrust is amplified by ``1 + k exp(-h / h0)`` below ``h0`` metres
    ground_effect: tuple[float, float] = (0.15, 0.35)
    gravity: float = G

    def with_(self, **changes) -> Airframe:
        return replace(self, **changes)

    @property
    def weight_n(self) -> float:
        return self.mass_kg * self.gravity

    def hover_thrust_n(self) -> float:
        return self.weight_n / 4.0

    def hover_omega(self) -> float:
        """The rotor speed that holds this vehicle up, straight from the BEMT rotor."""
        return self.rotor.omega_for_thrust(self.hover_thrust_n())


class Multirotor:
    """The airframe, integrated: thrust from each rotor, the moments it makes, the drag it fights and the charge it
    spends. :meth:`step` is one RK4 step of the whole thing, motors included."""

    def __init__(self, airframe: Airframe | None = None, *, table: RotorTable | None = None,
                 motor_efficiency: np.ndarray | None = None, thrust_gain: np.ndarray | None = None,
                 torque_gain: np.ndarray | None = None):
        self.airframe = airframe or Airframe()
        self.table = table or RotorTable(self.airframe.rotor, omega_max=self.airframe.omega_max * 1.2)
        #: a degraded motor scales thrust and torque together; a chipped blade scales them apart -- which is why they
        #: are two gains and not one
        efficiency = np.ones(4) if motor_efficiency is None else np.asarray(motor_efficiency, float).reshape(4)
        self.thrust_gain = efficiency if thrust_gain is None else np.asarray(thrust_gain, float).reshape(4)
        self.torque_gain = efficiency if torque_gain is None else np.asarray(torque_gain, float).reshape(4)
        self.arms = np.array([[v[0], v[1]] for v in QUAD_X.values()]) * self.airframe.arm_m / math.sqrt(2.0)
        self.spins = np.array([v[2] for v in QUAD_X.values()])

    # ---------------------------------------------------------------------------------------------- the dynamics
    def thrust_and_torque(self, state: State, wind=None) -> tuple[np.ndarray, np.ndarray]:
        """Thrust per rotor (body ``-z``) and its aerodynamic torque, ground effect included."""
        wind = np.zeros(3) if wind is None else np.asarray(wind, float)
        v_body = state.rotation().T @ (np.asarray(state.velocity, float) - wind)
        climb = -float(v_body[2])
        thrust = self.table.thrust_many(state.rotors, climb)
        torque = self.table.torque_many(state.rotors, climb)
        k, h0 = self.airframe.ground_effect
        h = max(state.altitude_m, 0.0)
        if h < h0:
            thrust = thrust * (1.0 + k * math.exp(-h / h0))
        return thrust, torque

    def derivatives(self, state: State, command: np.ndarray, wind=None):
        """``(d_position, d_velocity, d_attitude, d_rates, d_rotors)`` -- the right-hand side RK4 integrates."""
        af = self.airframe
        rotation = state.rotation()
        wind = np.zeros(3) if wind is None else np.asarray(wind, float)
        thrust, torque = self.thrust_and_torque(state, wind)
        effective = thrust * self.thrust_gain
        torque = torque * self.torque_gain

        force_body = np.array([0.0, 0.0, -effective.sum()])
        v_body = rotation.T @ (state.velocity - wind)
        drag_body = -0.5 * 1.225 * np.array(af.drag_area_m2) * v_body * np.abs(v_body)
        d_velocity = rotation @ (force_body + drag_body) / af.mass_kg + np.array([0.0, 0.0, af.gravity])

        tau = np.array([-float(self.arms[:, 1] @ effective),        # r x F with F along -z
                        float(self.arms[:, 0] @ effective),
                        -float(self.spins @ torque)])
        inertia = np.array(af.inertia)
        d_rates = (tau - np.cross(state.rates, inertia * state.rates)) / inertia

        w, x, y, z = state.attitude
        p, q, r = state.rates
        d_attitude = 0.5 * np.array([-(p * x + q * y + r * z), p * w + r * y - q * z,
                                     q * w - r * x + p * z, r * w + q * x - p * y])

        d_rotors = (np.asarray(command, float) * af.omega_max - state.rotors) / af.motor_tau_s
        return state.velocity.copy(), d_velocity, d_attitude, d_rates, d_rotors

    def step(self, state: State, command: np.ndarray, dt: float, *, wind=(0.0, 0.0, 0.0),
             disturbance=None) -> tuple[State, Telemetry]:
        """One RK4 step forward, then the charge it spent. ``disturbance`` is an external body force in newtons."""
        command = np.clip(np.asarray(command, float), 0.0, 1.0)
        wind = np.asarray(wind, float)

        def rhs(s: State):
            dpos, dvel, datt, drate, drot = self.derivatives(s, command, wind)
            if disturbance is not None:
                dvel = dvel + state.rotation() @ np.asarray(disturbance, float) / self.airframe.mass_kg
            return dpos, dvel, datt, drate, drot

        def advance(s: State, k: tuple, factor: float) -> State:
            dpos, dvel, datt, drate, drot = k
            return State(s.position + factor * dt * dpos, s.velocity + factor * dt * dvel,
                         s.attitude + factor * dt * datt, s.rates + factor * dt * drate,
                         s.rotors + factor * dt * drot, s.soc, s.time)

        k1 = rhs(state)
        k2 = rhs(advance(state, k1, 0.5))
        k3 = rhs(advance(state, k2, 0.5))
        k4 = rhs(advance(state, k3, 1.0))

        def blend(a, b, c, d):
            return (np.asarray(a) + 2 * np.asarray(b) + 2 * np.asarray(c) + np.asarray(d)) / 6.0
        attitude = state.attitude + dt * blend(k1[2], k2[2], k3[2], k4[2])
        new = State(state.position + dt * blend(k1[0], k2[0], k3[0], k4[0]),
                    state.velocity + dt * blend(k1[1], k2[1], k3[1], k4[1]),
                    attitude / np.linalg.norm(attitude),
                    state.rates + dt * blend(k1[3], k2[3], k3[3], k4[3]),
                    np.clip(state.rotors + dt * blend(k1[4], k2[4], k3[4], k4[4]), 0.0, self.airframe.omega_max),
                    state.soc, state.time + dt)

        thrust, _ = self.thrust_and_torque(new, wind)
        battery = self.airframe.battery
        rotor_power = self.table.total_power(new.rotors) / self.airframe.rotor.drivetrain_efficiency
        current = battery.current(rotor_power, state.soc)
        # the pack supplies the rotor power *plus* its own I^2 R loss: a pack whose resistance has doubled reports more
        # watts for the same flight, which is the whole signature of that fault
        pack_power = rotor_power + current * current * battery.resistance_ohm
        new.soc = max(state.soc - pack_power * dt / 3600.0 / battery.energy_wh, 0.0)
        telemetry = Telemetry(pack_power, current, battery.voltage(state.soc, current),
                              float(np.sum(thrust * self.thrust_gain)))
        return new, telemetry

    def measure(self, state: State) -> Telemetry:
        """The power, current and voltage of a state, without stepping it."""
        thrust, _ = self.thrust_and_torque(state)
        battery = self.airframe.battery
        rotor_power = self.table.total_power(state.rotors) / self.airframe.rotor.drivetrain_efficiency
        current = battery.current(rotor_power, state.soc)
        return Telemetry(rotor_power + current * current * battery.resistance_ohm, current,
                         battery.voltage(state.soc, current), float(thrust.sum()))

    # ------------------------------------------------------------------------------------------------ the mixer
    def allocate(self, thrust_n: float, torque=(0.0, 0.0, 0.0)) -> np.ndarray:
        """``(collective thrust, body torque)`` to four motor commands in ``[0, 1]`` -- the mixer.

        Thrust is split evenly and the roll/pitch torque is made by the differential across the arms
        (``dT_i = (tau_y x_i - tau_x y_i) / (4 a^2)``); yaw comes from spinning the two pairs against each other, scaled
        by ``dQ/dT`` of the rotor so that the couple asked for is the couple delivered.
        """
        arm = self.airframe.arm_m / math.sqrt(2.0)
        tau = np.asarray(torque, float)
        dx = self.arms[:, 0] / arm
        dy = self.arms[:, 1] / arm
        per_motor = thrust_n / 4.0 + (tau[1] * dx - tau[0] * dy) / (4.0 * arm)
        omega_hover = min(max(self.table.omega_for_thrust(thrust_n / 4.0), 1.0), self.airframe.omega_max)
        slope = self.table.dq_dt(omega_hover)
        if abs(slope) > 1e-12:
            per_motor = per_motor - self.spins * tau[2] / (4.0 * slope)
        per_motor = np.maximum(per_motor, 0.0)
        omega = np.array([self.table.omega_for_thrust(float(t)) for t in per_motor])
        return np.clip(omega, 0.0, self.airframe.omega_max) / self.airframe.omega_max      # normalised, as step takes
