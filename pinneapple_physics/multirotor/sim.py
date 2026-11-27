"""The flight loop: the truth with its faults, the sensors reading it, and the residual comparing the two.

One call is one flight. The controller flies the *nominal* model -- it is never told about the fault -- while the
dynamics of the flight carry the fault, so the vehicle drifts, the controller works harder, and the residual records by
how much. That is the whole experiment, and :func:`campaign` repeats it across faults, seeds and detectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .control import Controller, Trajectory
from .faults import Fault, FaultState, apply_faults, faulty_airframe
from .model import Airframe, Multirotor, State
from .residuals import CHANNELS, ResidualModel, SensorModel, SensorObservatory

__all__ = ["Flight", "fly", "campaign", "DURATION_S"]

DURATION_S = 20.0
DT = 0.002
#: the instant a fault is injected in every flight -- after the vehicle has settled, so a detection is the fault's
FAULT_TIME_S = 8.0


@dataclass
class Flight:
    """One flight's record: the residual per step, the truth, and what was flying at each instant."""

    time: np.ndarray
    residual: np.ndarray                       # (n, len(CHANNELS))
    position: np.ndarray                       # (n, 3) truth
    altitude: np.ndarray                       # (n,)
    power: np.ndarray                          # (n,) measured, W
    faults: tuple[Fault, ...] = ()
    active: list[tuple[str, ...]] = field(default_factory=list)
    seed: int = 0
    #: the sensor stream, kept so an estimator can be *replayed* over the flight exactly as it would have run live:
    #: :mod:`.estimate` reads these and nothing else, which is what makes its answer something a flight controller
    #: could have produced. ``None`` on a flight built by hand (the tests do that for the detectors).
    accel: np.ndarray | None = None            # (n, 3) body specific force, m/s^2
    gyro: np.ndarray | None = None             # (n, 3) body rates, rad/s
    velocity: np.ndarray | None = None         # (n, 3) world, m/s
    attitude: np.ndarray | None = None         # (n, 4) (w, x, y, z), body to world
    rotors: np.ndarray | None = None           # (n, 4) commanded rotor speeds, rad/s
    truth_mass_kg: float | None = None         # the mass that was actually flying (faults included)
    truth_inertia: tuple[float, float, float] | None = None
    truth_drag_area: tuple[float, float, float] | None = None

    def channel(self, name: str) -> np.ndarray:
        return self.residual[:, CHANNELS.index(name)]

    def onset(self) -> float | None:
        """When the first fault started, or ``None`` for a fault-free flight."""
        return min((f.start_s for f in self.faults), default=None)

    def labels(self, t: float) -> tuple[str, ...]:
        i = min(np.searchsorted(self.time, t), len(self.time) - 1)
        return self.active[i]

    def rotation(self, k: int) -> np.ndarray:
        """The body-to-world rotation at sample ``k``, from the recorded attitude (identity when there is none)."""
        if self.attitude is None:
            return np.eye(3)
        w, x, y, z = self.attitude[k]
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])


def fly(faults=(), *, duration: float = DURATION_S, seed: int = 0, dt: float = DT, airframe: Airframe | None = None,
        trajectory: Trajectory | None = None, sensors: SensorModel | None = None) -> Flight:
    """Fly once, with ``faults`` injected, and record what the residual saw."""
    base = airframe or Airframe()
    trajectory = trajectory or Trajectory("hover", altitude=5.0)
    controller_model = Multirotor(base)
    controller = Controller(controller_model)
    observatory = SensorObservatory(sensors, seed=seed)
    residual_model = ResidualModel(base)

    start = trajectory.at(0.0)[0]
    state = State(position=np.asarray(start, float).copy(), rotors=np.full(4, base.hover_omega()))
    residual_model.reset()
    n = int(round(duration / dt))
    record = np.zeros((n, len(CHANNELS)))
    position = np.zeros((n, 3))
    altitude = np.zeros(n)
    power = np.zeros(n)
    accel = np.zeros((n, 3))
    gyro = np.zeros((n, 3))
    velocity = np.zeros((n, 3))
    attitude = np.zeros((n, 4))
    rotors = np.zeros((n, 4))
    active: list[tuple[str, ...]] = []

    # the fault state and the model are rebuilt only when a fault starts or ends: building the model every step costs
    # more than the dynamics it integrates, and the hover speed is a full BEMT solve that belongs outside the loop
    hover_omega = base.hover_omega()
    pending = sorted({f.start_s for f in faults} | {f.end_s for f in faults if f.end_s is not None})
    current: FaultState | None = None
    model = controller_model
    for k in range(n):
        t = k * dt
        # rebuild at the first step at or past each event, so a fault whose start is not a multiple of dt still lands
        while pending and t >= pending[0] - 1e-12:
            pending.pop(0)
            current = None
        if current is None:
            current = apply_faults(faults, t, base_airframe=base, base_resistance_ohm=base.battery.resistance_ohm,
                                   rotor=base.rotor, omega_hover=hover_omega)
            model = Multirotor(faulty_airframe(base, mass_kg=current.mass_kg,
                                               battery_resistance_ohm=current.battery_resistance_ohm),
                               table=controller_model.table, thrust_gain=current.thrust_gain,
                               torque_gain=current.torque_gain)
        target, target_velocity, yaw = trajectory.at(t)
        command = controller.command(state, target, target_velocity, yaw, dt)
        state, telemetry = model.step(state, command, dt, wind=current.wind)
        sample = observatory.measure(state, model, command, telemetry, current)
        record[k] = residual_model(sample, dt).vector()
        position[k] = state.position
        altitude[k] = state.altitude_m
        power[k] = sample.power
        accel[k] = sample.accel
        gyro[k] = sample.gyro
        velocity[k] = sample.velocity
        attitude[k] = sample.attitude
        rotors[k] = sample.rotors
        active.append(current.active)

    return Flight(np.arange(n) * dt, record, position, altitude, power, tuple(faults), active, seed,
                  accel=accel, gyro=gyro, velocity=velocity, attitude=attitude, rotors=rotors,
                  truth_mass_kg=current.mass_kg, truth_inertia=tuple(base.inertia),
                  truth_drag_area=tuple(base.drag_area_m2))


def campaign(faults_per_case, *, seeds=(0, 1, 2), duration: float = DURATION_S, dt: float = DT,
             airframe: Airframe | None = None, trajectory: Trajectory | None = None) -> dict[str, list[Flight]]:
    """Fly every named case over every seed: ``{case: [Flight, ...]}``."""
    return {name: [fly(case, duration=duration, dt=dt, seed=s, airframe=airframe, trajectory=trajectory)
                   for s in seeds]
            for name, case in faults_per_case.items()}
