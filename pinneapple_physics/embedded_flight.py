"""The drone itself in the graph: the 6-DoF airframe of #335, Newton-Euler on its rotors and the sensor check, as one
ONNX file (#350).

:mod:`pinneapple_physics.embedded` exports a *field* model: a state of two frames in, the next frame out. This module
exports the vehicle: the airframe of :mod:`pinneapple_physics.multirotor`, whose state is the one a flight controller
estimates, whose thrust comes from a BEMT rotor table, and whose residual is Newton-Euler evaluated against what the
accelerometer, the gyro and the pack reported.

The graph
---------
Seven inputs, thirteen outputs, one straight line:

* **in** -- ``state`` (18: position, velocity, quaternion, body rates, four rotor speeds, state of charge),
  ``command`` (4 motor commands in ``[0, 1]``, what the ESC was told), ``accel``, ``gyro``, ``gyro_previous``,
  ``power`` and ``accel_history`` (the last 25 accelerometer samples: the sensor check is a second difference, so it
  needs a buffer, and a flight computer has one);
* **out** -- ``next_state`` (18, one RK4 step of the whole dynamics), ``thrust`` per rotor, the pack's
  ``pack_power``/``current``/``voltage``, the seven Newton-Euler channels (``residual_force`` x3,
  ``residual_torque`` x3, ``residual_power``), the decision (``residual``, ``trust``, ``off_physics``) and the sensor
  check (``vibration``, ``suspect``).

The two flags read different things, which is the whole point of exporting both. ``off_physics`` is the vehicle
disagreeing with Newton-Euler -- a payload, a degraded motor, a chipped blade. ``suspect`` is the *accelerometer*
disagreeing with its own declared noise -- a loose IMU, a failing sensor. A vibration fault sets both, because a
loose IMU is also a physics disagreement over a single sample; what separates them is the channel the disagreement
lands in, which is why all seven channels are outputs and not just the norm.

Where it breaks, and what that costs
------------------------------------
The multirotor of #335 is numpy with Python control flow. Not everything in it becomes arithmetic, and
:func:`export_probe` is the executable version of that sentence -- it tries the three pieces that cannot go in and
records what actually happened to each:

* ``RotorTable.omega_for_thrust`` (the mixer's inverse, and anything else that looks a table up by index) -- the
  lookup is ``np.interp``/``torch.searchsorted``, and the exporter refuses ``searchsorted`` at opset 17. The forward
  table (thrust from a rotor speed) *is* exportable, as the knot weights :meth:`_RotorTable.forward` writes: one
  weight per knot, piecewise linear, no index into the tensor;
* the BEMT rotor itself (:meth:`~pinneapple_physics.multirotor.rotor.Rotor.state`, a per-annulus solve with an inflow
  bisection) -- a numpy call inside a traced graph does not raise, it *freezes*: the file keeps the number the tracer
  computed. The rotor is therefore solved once, before the export, and its table is baked into the file: change a
  blade and you ship a new graph;
* a Python ``if`` on a tensor (the ground-effect branch is written that way in the numpy model) -- it is evaluated at
  trace time and the branch is gone. This module writes every branch as :func:`torch.where`, and
  :func:`export_probe` measures what the naive version costs at low altitude.

The integration is in the graph as exactly four unrolled RK4 evaluations. A flight loop of ``n`` steps is not: the
step is unrolled, the loop is the caller's.

    >>> from pinneapple_physics.embedded_flight import FlightSpec, export_flight, FlightRuntime
    >>> spec = FlightSpec(dt=0.002)
    >>> export_flight(spec, "flight.onnx").exists()
    True
    >>> rt = FlightRuntime("flight.onnx", threads=1)
    >>> rt.step(*FlightRuntime.feed(*spec.hover(), np.full(4, 0.5)))["residual"]   # doctest: +SKIP
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .embedded import EmbeddedRuntime, GraphSummary, graph_summary
from .multirotor import QUAD_X, Airframe, RotorTable, SensorModel

__all__ = ["FlightSpec", "FlightGraph", "FlightOutput", "FlightRuntime", "export_flight", "graph_summary",
           "CHAINS", "STATE_LAYOUT", "export_probe"]

#: The 18 numbers of the state, in order. A flight controller allocates this once, so the order is the contract.
STATE_LAYOUT = ("position", "velocity", "attitude", "rates", "rotors", "soc")
#: The seven Newton-Euler channels the graph exports, in the order of :data:`pinneapple_physics.multirotor.CHANNELS`.
CHAINS = ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z", "power")


@dataclass(frozen=True)
class FlightSpec:
    """The contract of the flight graph: whose vehicle, at what step, judged against what limits.

    The four tolerances are not calibrated numbers -- each one is derived from a *declared* datasheet number of
    :class:`~pinneapple_physics.multirotor.SensorModel` (three sigma of the accelerometer or gyro noise, at this
    ``dt``), and pass ``None`` to keep that derivation. A flight that calibrates them instead says so and passes the
    measured values; the graph does not know the difference, which is what makes the tolerance an input and not a
    part of the file.
    """

    airframe: Airframe = field(default_factory=Airframe)
    dt: float = 0.002
    sensors: SensorModel = field(default_factory=SensorModel)
    force_tolerance: float | None = None            # m/s^2
    torque_tolerance: float | None = None           # rad/s^2
    vibration_tolerance: float | None = None        # m/s^2
    history: int = 25
    n_omega: int = 27
    n_climb: int = 9
    climb_range: tuple[float, float] = (-8.0, 8.0)

    def __post_init__(self):
        if self.dt <= 0.0:
            raise ValueError("dt must be positive: the residual divides by it")
        if self.history < 3:
            raise ValueError("the sensor check is a second difference: it needs at least three samples")

    @property
    def force_limit(self) -> float:
        return 3.0 * self.sensors.accel_noise_m_s2 if self.force_tolerance is None else float(self.force_tolerance)

    @property
    def torque_limit(self) -> float:
        """Three sigma of the gyro noise, differenced at ``dt``: the floor an angular acceleration has to clear."""
        if self.torque_tolerance is None:
            return 3.0 * self.sensors.gyro_noise_rad_s / self.dt
        return float(self.torque_tolerance)

    @property
    def vibration_limit(self) -> float:
        if self.vibration_tolerance is None:
            return 3.0 * self.sensors.accel_noise_m_s2
        return float(self.vibration_tolerance)

    def table(self) -> RotorTable:
        """The BEMT rotor, tabulated: built once, baked into the graph, never rebuilt inside a loop."""
        return RotorTable(self.airframe.rotor, omega_max=self.airframe.omega_max * 1.2,
                          climb=self.climb_range, n_omega=self.n_omega, n_climb=self.n_climb)

    def hover(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """A vehicle holding station: the seven inputs of the graph, at hover, with a healthy accelerometer."""
        af = self.airframe
        state = np.zeros(18)
        state[2] = -5.0                                          # five metres up (z is down)
        state[6] = 1.0                                            # identity quaternion
        state[14:18] = af.hover_omega()
        state[17] = 1.0                                           # full pack
        thrust = af.weight_n / 4.0
        accel = np.array([0.0, 0.0, -4.0 * thrust / af.mass_kg])
        history = np.repeat(accel[None, :], self.history, axis=0)
        # the command is the one that holds this state: a healthy vehicle commands the rotor speed it is flying at
        command = np.full(4, af.hover_omega() / af.omega_max)
        return (state, command, accel, np.zeros(3), np.zeros(3), np.array([0.0]), history)


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a x b`` written out: ONNX has no Cross, and a decomposition is arithmetic the exporter keeps."""
    return torch.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], dim=-1)


def _rotation(q: torch.Tensor) -> torch.Tensor:
    """Body-to-world rotation from a ``(batch, 4)`` quaternion, the same matrix ``State.rotation()`` builds."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    one, two = torch.ones_like(w), 2.0 * torch.ones_like(w)
    rows = [
        [one - two * (y * y + z * z), two * (x * y - w * z), two * (x * z + w * y)],
        [two * (x * y + w * z), one - two * (x * x + z * z), two * (y * z - w * x)],
        [two * (x * z - w * y), two * (y * z + w * x), one - two * (x * x + y * y)],
    ]
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


class _RotorTable(torch.nn.Module):
    """The BEMT rotor table as arithmetic: piecewise-linear in ``(omega, climb)``, one weight per knot.

    ``np.interp`` finds the cell with a binary search and reads two numbers; ONNX has no such reader that the
    exporter accepts (``searchsorted`` is refused at opset 17) and a ``Gather`` on a computed index is a data-
    dependent address an embedded toolchain may not accept. The weights cost ``n_climb * n_omega`` multiplies -- 243
    here, against four rotors -- and sum to one, so the answer is the interpolated knot with no index in the graph.
    """

    def __init__(self, table: RotorTable):
        super().__init__()
        self.register_buffer("omegas", torch.tensor(table.omegas, dtype=torch.float32))
        self.register_buffer("climbs", torch.tensor(table.climbs, dtype=torch.float32))
        self.register_buffer("d_omega", torch.tensor(float(table.omegas[1] - table.omegas[0])))
        self.register_buffer("d_climb", torch.tensor(float(table.climbs[1] - table.climbs[0])))
        thrust = np.array([[table.thrust(float(w), float(c)) for w in table.omegas] for c in table.climbs])
        torque = np.array([[table.torque(float(w), float(c)) for w in table.omegas] for c in table.climbs])
        self.register_buffer("thrust_grid", torch.tensor(thrust.ravel(), dtype=torch.float32))
        self.register_buffer("torque_grid", torch.tensor(torque.ravel(), dtype=torch.float32))
        self.n_omega = int(len(table.omegas))
        # the clamp bounds as Python numbers: the exporter folds them either way, and a float() on a buffer at trace
        # time is a "the trace might be incorrect" warning on every export
        self.omega_lo, self.omega_hi = float(table.omegas[0]), float(table.omegas[-1])
        self.climb_lo, self.climb_hi = float(table.climbs[0]), float(table.climbs[-1])

    def _weights(self, omega: torch.Tensor, climb: torch.Tensor) -> torch.Tensor:
        w = omega.clamp(self.omega_lo, self.omega_hi)
        c = climb.clamp(self.climb_lo, self.climb_hi)
        dw = 1.0 - ((w.unsqueeze(-1) - self.omegas) / self.d_omega).abs().clamp(0.0, 1.0)
        dc = 1.0 - ((c.unsqueeze(-1) - self.climbs) / self.d_climb).abs().clamp(0.0, 1.0)
        return (dc.unsqueeze(-1) * dw.unsqueeze(-2)).reshape(*w.shape, -1)

    def forward(self, omega: torch.Tensor, climb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self._weights(omega, climb)
        return (weight * self.thrust_grid).sum(-1), (weight * self.torque_grid).sum(-1)


class FlightGraph(torch.nn.Module):
    """The whole embedded flight path as one straight line: dynamics, Newton-Euler residual and sensor check.

    Every branch is :func:`torch.where` and every loop is unrolled, because the alternative exports without an error
    and answers with the number the tracer saw (see :func:`export_probe`).
    """

    def __init__(self, spec: FlightSpec):
        super().__init__()
        self.spec = spec
        af = spec.airframe
        self.rotors_table = _RotorTable(spec.table())
        arms = np.array([[v[0], v[1]] for v in QUAD_X.values()]) * af.arm_m / math.sqrt(2.0)
        self.register_buffer("mass", torch.tensor(float(af.mass_kg)))
        self.register_buffer("inertia", torch.tensor(list(af.inertia), dtype=torch.float32))
        self.register_buffer("arms", torch.tensor(arms, dtype=torch.float32))
        self.register_buffer("spins", torch.tensor([v[2] for v in QUAD_X.values()], dtype=torch.float32))
        self.register_buffer("drag", torch.tensor(list(af.drag_area_m2), dtype=torch.float32))
        self.register_buffer("motor_tau", torch.tensor(float(af.motor_tau_s)))
        self.register_buffer("omega_max", torch.tensor(float(af.omega_max)))
        self.register_buffer("thrust_gain", torch.tensor(np.ones(4), dtype=torch.float32))
        self.register_buffer("torque_gain", torch.tensor(np.ones(4), dtype=torch.float32))
        self.register_buffer("gravity", torch.tensor(float(af.gravity)))
        self.register_buffer("ground_k", torch.tensor(float(af.ground_effect[0])))
        self.register_buffer("ground_h0", torch.tensor(float(af.ground_effect[1])))
        self.register_buffer("rho", torch.tensor(1.225))
        self.register_buffer("dt", torch.tensor(float(spec.dt)))
        battery = af.battery
        self.register_buffer("cells", torch.tensor(float(battery.cells)))
        self.register_buffer("resistance", torch.tensor(float(battery.resistance_ohm)))
        self.register_buffer("v_full", torch.tensor(float(battery.v_full)))
        self.register_buffer("v_empty", torch.tensor(float(battery.v_empty)))
        self.register_buffer("energy_wh", torch.tensor(float(battery.energy_wh)))
        self.register_buffer("eta", torch.tensor(float(af.rotor.drivetrain_efficiency)))
        self.register_buffer("force_tol", torch.tensor(float(spec.force_limit)))
        self.register_buffer("torque_tol", torch.tensor(float(spec.torque_limit)))
        self.register_buffer("vibration_tol", torch.tensor(float(spec.vibration_limit)))
        self.register_buffer("noise_floor", torch.tensor(float(spec.sensors.accel_noise_m_s2)))
        self.register_buffer("history_floor", torch.tensor(math.sqrt(18.0)))
        self.omega_limit = float(af.omega_max)

    # ------------------------------------------------------------------------------------------- the dynamics
    def _thrust_and_torque(self, state: torch.Tensor, climb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Thrust per rotor and its aerodynamic torque, ground effect included -- ``Multirotor.thrust_and_torque``.

        The ground-effect branch is the numpy model's ``if h < h0``; here it is a ``where``. Traced, an ``if`` on a
        tensor is evaluated once, at trace time, and the branch is not in the file -- which is why the module keeps
        one measurement of what that costs (:func:`export_probe`).
        """
        thrust, torque = self.rotors_table(state[:, 13:17], climb)
        altitude = (-state[:, 2]).clamp_min(0.0)
        factor = torch.where(altitude < self.ground_h0,
                             torch.ones_like(altitude) + self.ground_k * torch.exp(-altitude / self.ground_h0),
                             torch.ones_like(altitude))
        return thrust * factor.unsqueeze(-1), torque

    def _derivatives(self, state: torch.Tensor, command: torch.Tensor) -> list[torch.Tensor]:
        """``(d_position, d_velocity, d_attitude, d_rates, d_rotors)`` -- the right-hand side RK4 integrates."""
        rotation = _rotation(state[:, 6:10])
        velocity, rates, rotors = state[:, 3:6], state[:, 10:13], state[:, 13:17]
        v_body = velocity.unsqueeze(-2) @ rotation                           # R^T v, as a row vector times R
        v_body = v_body.squeeze(-2)
        climb = -v_body[:, 2]
        thrust, torque = self._thrust_and_torque(state, climb)
        effective = thrust * self.thrust_gain
        torque = torque * self.torque_gain

        force_body = torch.stack([torch.zeros_like(effective[:, 0]), torch.zeros_like(effective[:, 0]),
                                  -effective.sum(-1)], dim=-1)
        drag_body = -0.5 * self.rho * self.drag * v_body * v_body.abs()
        d_velocity = (force_body + drag_body).unsqueeze(-2) @ rotation.transpose(-1, -2)
        gravity = torch.stack([torch.zeros_like(effective[:, 0]), torch.zeros_like(effective[:, 0]),
                               self.gravity * torch.ones_like(effective[:, 0])], dim=-1)
        d_velocity = d_velocity.squeeze(-2) / self.mass + gravity

        tau = torch.stack([-(effective * self.arms[:, 1]).sum(-1), (effective * self.arms[:, 0]).sum(-1),
                           -(torque * self.spins).sum(-1)], dim=-1)
        d_rates = (tau - _cross(rates, self.inertia * rates)) / self.inertia

        w, x, y, z = state[:, 6], state[:, 7], state[:, 8], state[:, 9]
        p, q, r = rates[:, 0], rates[:, 1], rates[:, 2]
        d_attitude = 0.5 * torch.stack([-(p * x + q * y + r * z), p * w + r * y - q * z,
                                        q * w - r * x + p * z, r * w + q * x - p * y], dim=-1)
        d_rotors = (command * self.omega_max - rotors) / self.motor_tau
        return [velocity, d_velocity, d_attitude, d_rates, d_rotors]

    def _advance(self, state: torch.Tensor, k: list[torch.Tensor], factor: float) -> torch.Tensor:
        dt = self.dt * factor
        return torch.cat([state[:, 0:3] + dt * k[0], state[:, 3:6] + dt * k[1], state[:, 6:10] + dt * k[2],
                          state[:, 10:13] + dt * k[3], state[:, 13:17] + dt * k[4], state[:, 17:18]], dim=-1)

    def _step(self, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """One RK4 step, the four evaluations unrolled: the count is the method's, not the data's."""
        k1 = self._derivatives(state, command)
        k2 = self._derivatives(self._advance(state, k1, 0.5), command)
        k3 = self._derivatives(self._advance(state, k2, 0.5), command)
        k4 = self._derivatives(self._advance(state, k3, 1.0), command)
        blended = [(a + 2.0 * b + 2.0 * c + d) / 6.0 for a, b, c, d in zip(k1, k2, k3, k4, strict=True)]
        attitude = state[:, 6:10] + self.dt * blended[2]
        advanced = torch.cat([
            state[:, 0:3] + self.dt * blended[0],
            state[:, 3:6] + self.dt * blended[1],
            attitude / attitude.pow(2).sum(-1, keepdim=True).sqrt(),
            state[:, 10:13] + self.dt * blended[3],
            (state[:, 13:17] + self.dt * blended[4]).clamp(0.0, self.omega_limit),
            state[:, 17:18]], dim=-1)
        return advanced

    def _telemetry(self, state: torch.Tensor, soc: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """What the pack pays for this state: the rotor power, the current through it and the sag -- ``Multirotor``."""
        rotors = state[:, 13:17]
        _, torque = self.rotors_table(rotors, torch.zeros_like(rotors[:, 0]))
        rotor_power = (torque * rotors.clamp_min(0.0)).sum(-1) / self.eta
        open_circuit = self.cells * (self.v_empty + (self.v_full - self.v_empty) * soc.clamp(0.0, 1.0))
        root = (open_circuit * open_circuit - 4.0 * self.resistance * rotor_power).clamp_min(0.0).sqrt()
        current = torch.where(rotor_power > 0.0,
                              0.5 * (open_circuit - root) / self.resistance, torch.zeros_like(rotor_power))
        pack_power = rotor_power + current * current * self.resistance
        voltage = (open_circuit - current.abs() * self.resistance).clamp_min(0.5 * self.cells * self.v_empty)
        return pack_power, current, voltage

    # ---------------------------------------------------------------------------------------------- forward
    def forward(self, state, command, accel, gyro, gyro_previous, power, accel_history):
        advanced = self._step(state, command)
        pack_power, current, voltage = self._telemetry(advanced, advanced[:, 17])
        soc = (state[:, 17] - pack_power * self.dt / 3600.0 / self.energy_wh).clamp(0.0, 1.0)
        next_state = torch.cat([advanced[:, 0:17], soc.unsqueeze(-1)], dim=-1)

        # --- Newton-Euler on the committed rotors against what the sensors reported (ResidualModel, channels 0-6)
        commanded = command * self.omega_max
        zeros = torch.zeros_like(commanded[:, 0])
        thrust, torque = self.rotors_table(commanded, zeros)
        predicted = torch.stack([torch.zeros_like(thrust[:, 0]), torch.zeros_like(thrust[:, 0]),
                                 -thrust.sum(-1)], dim=-1) / self.mass
        residual_force = predicted - accel
        tau = torch.stack([-(thrust * self.arms[:, 1]).sum(-1), (thrust * self.arms[:, 0]).sum(-1),
                           -(torque * self.spins).sum(-1)], dim=-1)
        angular = (gyro - gyro_previous) / self.dt
        residual_torque = tau / self.inertia - _cross(gyro, self.inertia * gyro) / self.inertia - angular
        residual_power = (torque * commanded.clamp_min(0.0)).sum(-1) / self.eta - power

        # --- the decision: the worst channel, normalised by the limit derived from the declared sensor noise
        force_rms = residual_force.pow(2).mean(-1).sqrt() / self.force_tol
        torque_rms = residual_torque.pow(2).mean(-1).sqrt() / self.torque_tol
        residual = torch.maximum(force_rms, torque_rms)
        trust = (1.0 - residual).clamp(0.0, 1.0)
        off_physics = (residual > 1.0).to(residual.dtype)

        # --- the measurement check: the accelerometer's second difference against its own declared noise floor
        second = accel_history[:, 2:] - 2.0 * accel_history[:, 1:-1] + accel_history[:, :-2]
        vibration = (second.pow(2).sum(-1).mean(-1).sqrt() - self.history_floor * self.noise_floor).clamp_min(0.0)
        suspect = (vibration > self.vibration_tol).to(vibration.dtype)

        return (next_state, thrust, pack_power, current, voltage, residual_force, residual_torque, residual_power,
                residual, trust, off_physics, vibration, suspect)


def export_flight(spec: FlightSpec, path: Any, *, opset: int = 17) -> Path:
    """Write the flight graph: ``(7 inputs) -> (13 outputs)``, fixed size, no interpreter in the hot path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = FlightGraph(spec).eval()
    state, command, accel, gyro, gyro_previous, power, history = spec.hover()
    feed = tuple(torch.tensor(np.asarray(v, np.float32)[None]) for v in
                 (state, command, accel, gyro, gyro_previous, power, history))
    names = ["next_state", "thrust_predicted", "pack_power", "current", "voltage", "residual_force", "residual_torque",
             "residual_power", "residual", "trust", "off_physics", "vibration", "suspect"]
    with torch.no_grad():
        torch.onnx.export(graph, feed, str(path),
                          input_names=["state", "command", "accel", "gyro", "gyro_previous", "power", "accel_history"],
                          output_names=names, opset_version=opset, do_constant_folding=True, dynamo=False)
    return path


@dataclass(frozen=True)
class FlightOutput:
    """One step of the flight graph, read back as numbers a flight computer would act on."""

    next_state: np.ndarray
    thrust_predicted: np.ndarray
    pack_power: float
    current: float
    voltage: float
    residual_force: np.ndarray
    residual_torque: np.ndarray
    residual_power: float
    residual: float
    trust: float
    off_physics: bool
    vibration: float
    suspect: bool

    def to_dict(self) -> dict[str, Any]:
        return {"residual": self.residual, "trust": self.trust, "off_physics": self.off_physics,
                "vibration": self.vibration, "suspect": self.suspect, "pack_power": self.pack_power,
                "voltage": self.voltage, "thrust_predicted": float(np.sum(self.thrust_predicted))}


class FlightRuntime(EmbeddedRuntime):
    """The runtime of the flight graph: the same one ONNX session, with the seven-input feed written out.

    Subclassed for one reason: :meth:`feed` is a different function here, so the tests and the example cannot call
    the field graph's two-input feed on a flight graph and get a broadcast instead of an error.
    """

    @staticmethod
    def feed(state, command, accel, gyro, gyro_previous, power, history) -> dict[str, np.ndarray]:
        return {"state": np.asarray(state, np.float32)[None],
                "command": np.asarray(command, np.float32)[None],
                "accel": np.asarray(accel, np.float32)[None],
                "gyro": np.asarray(gyro, np.float32)[None],
                "gyro_previous": np.asarray(gyro_previous, np.float32)[None],
                "power": np.asarray(power, np.float32)[None],
                "accel_history": np.asarray(history, np.float32)[None]}

    def step(self, state, command, accel, gyro, gyro_previous, power, history) -> FlightOutput:  # type: ignore[override]
        out = self.run(self.feed(state, command, accel, gyro, gyro_previous, power, history))
        return FlightOutput(next_state=out["next_state"][0], thrust_predicted=out["thrust_predicted"][0],
                            pack_power=float(out["pack_power"].ravel()[0]), current=float(out["current"].ravel()[0]),
                            voltage=float(out["voltage"].ravel()[0]), residual_force=out["residual_force"][0],
                            residual_torque=out["residual_torque"][0],
                            residual_power=float(out["residual_power"].ravel()[0]),
                            residual=float(out["residual"].ravel()[0]), trust=float(out["trust"].ravel()[0]),
                            off_physics=bool(out["off_physics"].ravel()[0] > 0.5),
                            vibration=float(out["vibration"].ravel()[0]),
                            suspect=bool(out["suspect"].ravel()[0] > 0.5))

    def latency(self, *inputs, n: int = 200) -> dict[str, float]:     # type: ignore[override]
        return self.benchmark(self.feed(*inputs), n=n)


# ------------------------------------------------------------------------------------ the parts that do not export
class _NaiveBranch(torch.nn.Module):
    """The ground-effect branch written the way the numpy model writes it: a Python ``if`` on a tensor."""

    def __init__(self, k: float, h0: float):
        super().__init__()
        self.register_buffer("k", torch.tensor(k))
        self.register_buffer("h0", torch.tensor(h0))

    def forward(self, thrust, altitude):
        h = (-altitude[:, 2]).clamp_min(0.0)
        if bool(h[0] < self.h0):                    # traced once, at the value the tracer saw
            thrust = thrust * (1.0 + self.k * torch.exp(-h / self.h0)).unsqueeze(-1)
        return thrust


class _NaiveLookup(torch.nn.Module):
    """The table read the way the numpy model reads it: a binary search for the cell, then two gathers."""

    def __init__(self, omega: torch.Tensor, thrust: torch.Tensor):
        super().__init__()
        self.register_buffer("omega", omega)
        self.register_buffer("thrust", thrust)

    def forward(self, w):
        i = torch.clamp(torch.searchsorted(self.omega, w) - 1, 0, len(self.omega) - 2)
        a = torch.gather(self.thrust, 0, i)
        b = torch.gather(self.thrust, 0, i + 1)
        wa = torch.gather(self.omega, 0, i)
        wb = torch.gather(self.omega, 0, i + 1)
        return a + (b - a) * (w - wa) / (wb - wa)


class _NaiveBEMT(torch.nn.Module):
    """The rotor solved where it must not be solved: inside the traced graph."""

    def __init__(self, rotor):
        super().__init__()
        self.rotor = rotor

    def forward(self, omega):
        value = self.rotor.state(float(omega.ravel()[0])).thrust_n     # a Python float out of a tensor
        return torch.full_like(omega, value)


def export_probe(workdir: Any, *, airframe: Airframe | None = None) -> list[dict[str, Any]]:
    """Try to export the three pieces of the flight that are not arithmetic, and record what happened to each.

    The point is that two of the three do *not* raise. An export that fails is a bug report; an export that succeeds
    and froze a number is a vehicle that flies on the number. The ``cost`` of each frozen piece is measured against
    the arithmetic version, at a state where the difference shows.
    """
    import onnxruntime as ort

    af = airframe or Airframe()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []

    table = RotorTable(af.rotor, omega_max=af.omega_max * 1.2)
    omegas = torch.tensor(table.omegas, dtype=torch.float32)
    thrust = torch.tensor([table.thrust(float(w)) for w in table.omegas], dtype=torch.float32)

    # 1. the lookup by index
    path = workdir / "lookup.onnx"
    entry: dict[str, Any] = {"piece": "the table lookup by index (np.interp / torch.searchsorted)",
                             "source": "RotorTable.thrust_many and the mixer's omega_for_thrust"}
    try:
        torch.onnx.export(_NaiveLookup(omegas, thrust), (torch.tensor([100.0, 500.0]),), str(path),
                          input_names=["omega"], opset_version=17, dynamo=False)
        entry |= {"result": "exported", "detail": "searchsorted went into the graph"}
    except Exception as exc:                       # noqa: BLE001 - the type and the message are the finding
        entry |= {"result": "refused", "detail": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"}
    out.append(entry)

    # 2. the rotor solved inside the graph
    path = workdir / "bemt.onnx"
    entry = {"piece": "the BEMT rotor inside the graph (Rotor.state, per-annulus solve with an inflow bisection)",
             "source": "Rotor.state / RotorTable.__init__"}
    try:
        torch.onnx.export(_NaiveBEMT(af.rotor), (torch.tensor([700.0]),), str(path), input_names=["omega"],
                          opset_version=17, dynamo=False)
        summary = graph_summary(path)
        frozen = float(ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {})[0].ravel()[0])
        real = af.rotor.state(1100.0).thrust_n
        entry |= {"result": "exported and frozen",
                  "detail": f"the file has the inputs {list(summary.inputs)} -- the rotor speed is not one of them, "
                            f"because the BEMT solved at trace time and only its answer went in. It returns "
                            f"{frozen:.4f} N for any vehicle state; the rotor itself gives {real:.4f} N at 1100 rad/s "
                            f"({abs(real - frozen) / max(real, 1e-9):.0%} of the thrust missing, and no input to "
                            f"correct it with)"}
    except Exception as exc:                       # noqa: BLE001
        entry |= {"result": "refused", "detail": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"}
    out.append(entry)

    # 3. the branch written as a Python `if`
    k, h0 = af.ground_effect
    naive = workdir / "branch.onnx"
    honest = workdir / "branch_where.onnx"
    entry = {"piece": "the ground-effect branch written as a Python if on a tensor",
             "source": "Multirotor.thrust_and_torque"}
    high = torch.tensor([[0.0, 0.0, -5.0]])                    # 5 m up: outside the ground effect
    thrusts = torch.full((1, 4), 3.0)
    torch.onnx.export(_NaiveBranch(k, h0), (thrusts, high), str(naive), input_names=["thrust", "altitude"],
                      opset_version=17, dynamo=False)

    class _WhereBranch(torch.nn.Module):
        def forward(self, thrust, altitude):
            h = (-altitude[:, 2]).clamp_min(0.0)
            return thrust * torch.where(h < h0, 1.0 + k * torch.exp(-h / h0), torch.ones_like(h)).unsqueeze(-1)

    torch.onnx.export(_WhereBranch(), (thrusts, high), str(honest), input_names=["thrust", "altitude"],
                      opset_version=17, dynamo=False)
    frozen = ort.InferenceSession(str(naive), providers=["CPUExecutionProvider"])
    real = ort.InferenceSession(str(honest), providers=["CPUExecutionProvider"])
    naive_inputs = graph_summary(naive).inputs
    feed = {"thrust": np.full((1, 4), 3.0, np.float32), "altitude": np.array([[0.0, 0.0, -0.1]], np.float32)}
    frozen_value = float(frozen.run(None, {"thrust": feed["thrust"]})[0].sum())
    real_value = float(real.run(None, feed)[0].sum())
    entry |= {"result": "exported and frozen",
              "detail": f"the graph built from the if has the inputs {list(naive_inputs)} -- altitude is not among "
                        f"them, because the branch was decided at trace time (5 m up, outside the effect) and the "
                        f"vehicle's height left the file with it. At 10 cm it returns {frozen_value:.3f} N of thrust "
                        f"where the where version returns {real_value:.3f} N: "
                        f"{(real_value - frozen_value) / real_value:.1%} of the lift is missing, and nothing in the "
                        f"file says so"}
    out.append(entry)
    return out
