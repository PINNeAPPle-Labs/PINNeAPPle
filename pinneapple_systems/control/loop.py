"""Closed loops on any plant: actuators, controllers, a fixed-step simulator, and the linearised closed loop.

The plant is ``dx/dt = rhs(x, u, d, t)`` with a measured output ``y = output(x)`` (:class:`Plant`). Between the
controller and the plant sits an :class:`Actuator`:

* range limits;
* a rate limit;
* a first-order lag;
* a dead band;
* a transport delay;
* optionally *integrating*, where the command is the rate of the applied value. A pump flow is integrating with
  respect to the mass in a tank.

What the controller commands is part of the design. The same PID that holds a plant when it commands a position
(e.g. a target tank mass, with the pump as a rate-limited servo) can destabilise it when it commands the rate (the
pump flow): the integrating actuator adds a pole at the origin. :func:`loop_eigenvalues` shows this before any long
simulation. It linearises the complete discrete-time loop around an equilibrium, so the plant, the actuator
dynamics, the controller states and the integration scheme are all included. :func:`compare_architectures` runs
several (controller, actuator) choices on the same plant, disturbance and integrator, and tabulates both views.

Controllers expose ``act(t, x, y, r, ydot, dt) -> command`` and their internal state as a vector
(``get_state`` / ``set_state``), which is what the closed-loop linearisation perturbs:

* :class:`PID` extends :class:`~pinneapple_systems.component_modeling.control.PIDController` with derivative on the
  measurement (or a measured rate), a derivative filter, setpoint weighting, back-calculation anti-windup and
  feed-forward;
* :class:`Cascade`: an outer controller sets the reference of an inner one;
* :class:`StateFeedback`: ``u = u_ref - K (x - x_ref)``, e.g. with gains from :func:`.tuning.lqr`.

Simulation only: nothing here has a path to real equipment.
"""
from __future__ import annotations

import copy
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinneapple_systems.component_modeling.control import PIDController

from .analysis import stability

__all__ = ["Plant", "Actuator", "PID", "Cascade", "StateFeedback", "simulate_loop", "loop_eigenvalues",
           "compare_architectures"]


@dataclass
class Plant:
    """A dynamic model for closed-loop work.

    ``rhs(x, u, d, t) -> dx/dt`` with ``u`` the applied actuator value (scalar) and ``d`` the disturbance (scalar or
    vector). ``output`` is a state index or a callable ``x -> y``. ``rate`` is optional: a callable ``x -> dy/dt``
    (a velocity or rate sensor) that a PID can use for its derivative term instead of differencing ``y``.
    """
    rhs: Callable[[np.ndarray, float, Any, float], np.ndarray]
    x0: Sequence[float]
    output: int | Callable[[np.ndarray], float] = 0
    rate: Callable[[np.ndarray], float] | None = None
    name: str = "plant"
    state_names: Sequence[str] = ()

    def y(self, x: np.ndarray) -> float:
        return float(x[self.output]) if isinstance(self.output, int) else float(self.output(x))


@dataclass
class Actuator:
    """Between the command and the plant: ``applied = actuator(command)``.

    Parameters
    ----------
    lo, hi : float
        Range of the applied value (units of the plant input).
    rate : float
        Largest ``|d applied / dt|`` (plant input units per second). For an integrating actuator this limits the
        commanded rate itself.
    tau : float [s]
        First-order lag of the actuator (0: none).
    deadband : float
        Commands within ``+-deadband`` of the previous target (servo) or of zero (integrating) are ignored
        (command units).
    delay : float [s]
        Transport delay of the command, rounded to whole steps of the simulation.
    integrating : bool
        The command is the rate of the applied value (e.g. a pump flow driving a tank mass).
    initial : float
        Applied value at the start (plant input units).
    """
    lo: float = -math.inf
    hi: float = math.inf
    rate: float = math.inf
    tau: float = 0.0
    deadband: float = 0.0
    delay: float = 0.0
    integrating: bool = False
    initial: float = 0.0
    _applied: float = field(default=float("nan"), init=False, repr=False)
    _lag: float = field(default=float("nan"), init=False, repr=False)
    _target: float = field(default=float("nan"), init=False, repr=False)
    _buffer: list = field(default_factory=list, init=False, repr=False)
    _dt: float = field(default=0.0, init=False, repr=False)

    def reset(self, dt: float, applied: float | None = None, command: float | None = None) -> None:
        a = self.initial if applied is None else float(applied)
        self._applied = float(min(max(a, self.lo), self.hi))
        self._lag = self._applied
        self._target = self._applied
        self._dt = dt
        nd = int(round(self.delay / dt)) if self.delay > 0 else 0
        c = self.equilibrium_command(self._applied) if command is None else float(command)
        self._buffer = [c] * nd

    def equilibrium_command(self, applied: float) -> float:
        """The command that holds ``applied`` still: the value itself for a servo, zero for an integrating one."""
        return 0.0 if self.integrating else float(applied)

    @property
    def applied(self) -> float:
        return self._applied

    def step(self, command: float, dt: float) -> float:
        """Advance one step and return the applied value."""
        if not np.isfinite(self._applied):
            self.reset(dt)
        if self._buffer:
            self._buffer.append(float(command))
            command = self._buffer.pop(0)
        if self.integrating:
            q = 0.0 if abs(command) <= self.deadband else float(command)
            q = min(max(q, -self.rate), self.rate)
            a = self._applied + q * dt
            self._target = self._lag = float(min(max(a, self.lo), self.hi))
        else:
            target = float(command)
            if abs(target - self._target) <= self.deadband:
                target = self._target
            self._target = target
            if self.tau > 0:
                self._lag += (target - self._lag) * (1.0 - math.exp(-dt / self.tau))
                target = self._lag
            else:
                self._lag = target
            move = min(max(target - self._applied, -self.rate * dt), self.rate * dt)
            a = self._applied + move
        self._applied = float(min(max(a, self.lo), self.hi))
        return self._applied

    def get_state(self) -> np.ndarray:
        return np.array([self._applied, self._lag, self._target] + list(self._buffer), float)

    def set_state(self, s: np.ndarray) -> None:
        self._applied, self._lag, self._target = float(s[0]), float(s[1]), float(s[2])
        self._buffer = [float(v) for v in s[3:]]


class PID(PIDController):
    """PID for real loops: ``u = bias + ff(t) + Kp (b r - y) + Ki int(e) + Kd D``, with ``e = r - y``.

    Parameters
    ----------
    kp, ki, kd : float
        Gains (command units per output unit; per output unit-second; per output unit per second).
    setpoint : float
        Default reference (output units); ``act`` takes the reference of each step.
    output_limits : tuple, optional
        ``(lo, hi)`` saturation of the command, with anti-windup.
    bias : float
        Manual reset / feed-forward constant (command units).
    derivative : {'measurement', 'error'}
        Differentiate the measurement (no derivative kick on setpoint steps) or the error. When the loop provides
        a measured rate (``ydot``), ``'measurement'`` uses it directly.
    filter_tau : float [s]
        First-order filter on the derivative term (0: none).
    anti_windup : {'conditional', 'back_calculation', 'none'}
        Conditional integration (the base class), back-calculation with tracking time ``tracking_time``, or none.
    tracking_time : float [s], optional
        Back-calculation time constant (default ``sqrt(Ti Td)`` or ``Ti``).
    setpoint_weight : float [-]
        ``b`` in the proportional term (1: plain PID; 0: I-PD, no proportional kick).
    feedforward : callable, optional
        ``t -> command`` added to the output.
    """

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0, setpoint: float = 0.0,
                 output_limits: tuple[float, float] | None = None, bias: float = 0.0, *,
                 derivative: str = "measurement", filter_tau: float = 0.0, anti_windup: str = "conditional",
                 tracking_time: float | None = None, setpoint_weight: float = 1.0,
                 feedforward: Callable[[float], float] | None = None):
        super().__init__(kp, ki, kd, setpoint, output_limits=output_limits, bias=bias)
        if derivative not in ("measurement", "error"):
            raise ValueError("derivative must be 'measurement' or 'error'")
        if anti_windup not in ("conditional", "back_calculation", "none"):
            raise ValueError("anti_windup must be 'conditional', 'back_calculation' or 'none'")
        self.derivative, self.filter_tau, self.anti_windup = derivative, float(filter_tau), anti_windup
        self.setpoint_weight, self.feedforward = float(setpoint_weight), feedforward
        if tracking_time is None and ki:
            ti = kp / ki if kp else 1.0 / abs(ki)
            td = kd / kp if kp else 0.0
            tracking_time = math.sqrt(ti * td) if td > 0 else ti
        self.tracking_time = float(tracking_time) if tracking_time else 1.0
        self._prev_meas: float | None = None
        self._dfilt = 0.0

    def reset(self) -> None:
        super().reset()
        self._prev_meas = None
        self._dfilt = 0.0

    def update(self, measurement: float, dt: float, rate: float | None = None, setpoint: float | None = None,
               t: float = 0.0) -> float:
        r = self.setpoint if setpoint is None else float(setpoint)
        y = float(measurement)
        e = r - y
        if self.derivative == "measurement":
            if rate is not None:
                raw = -float(rate)
            else:
                raw = 0.0 if self._prev_meas is None else -(y - self._prev_meas) / dt
        else:
            raw = 0.0 if self._prev_error is None else (e - self._prev_error) / dt
        self._prev_meas, self._prev_error = y, e
        if self.filter_tau > 0:
            self._dfilt += (raw - self._dfilt) * dt / (self.filter_tau + dt)
        else:
            self._dfilt = raw
        integral = self._integral + e * dt
        ff = float(self.feedforward(t)) if self.feedforward is not None else 0.0
        v = (self.bias + ff + self.kp * (self.setpoint_weight * r - y) + self.ki * integral
             + self.kd * self._dfilt)
        if self.output_limits is None:
            self._integral = integral
            return float(v)
        lo, hi = self.output_limits
        u = float(min(max(v, lo), hi))
        if self.anti_windup == "none":
            self._integral = integral
        elif self.anti_windup == "back_calculation" and self.ki:
            self._integral = integral + (u - v) / (self.ki * self.tracking_time) * dt
        else:
            saturated = (v > hi and e > 0) or (v < lo and e < 0)
            if not saturated:
                self._integral = integral
        return u

    # closed-loop interface
    def act(self, t: float, x: np.ndarray, y: float, r: float, ydot: float | None, dt: float) -> float:
        return self.update(y, dt, rate=ydot, setpoint=r, t=t)

    def set_equilibrium(self, y: float, r: float, command: float, t: float = 0.0, x=None) -> None:
        """Bumpless start: the internal state that outputs ``command`` at ``y`` with reference ``r``.

        With integral action the integral takes up the difference; without it the bias does."""
        ff = float(self.feedforward(t)) if self.feedforward is not None else 0.0
        e = r - y
        base = self.bias + ff + self.kp * (self.setpoint_weight * r - y)
        self._integral = (command - base) / self.ki if self.ki else 0.0
        if not self.ki and abs(command - base) > 1e-9 * max(1.0, abs(command)):
            self.bias += command - base
        self._prev_meas, self._prev_error, self._dfilt = float(y), float(e), 0.0

    def get_state(self) -> np.ndarray:
        pm = np.nan if self._prev_meas is None else self._prev_meas
        pe = np.nan if self._prev_error is None else self._prev_error
        return np.array([self._integral, pm, pe, self._dfilt], float)

    def set_state(self, s: np.ndarray) -> None:
        self._integral = float(s[0])
        self._prev_meas = None if np.isnan(s[1]) else float(s[1])
        self._prev_error = None if np.isnan(s[2]) else float(s[2])
        self._dfilt = float(s[3])


class Cascade:
    """Outer loop sets the reference of the inner loop.

    ``outer`` acts on the plant output ``y`` against the reference; its command becomes the reference of ``inner``,
    which acts on ``inner_measurement(x)`` (e.g. a flow, a velocity, a valve position).
    ``inner_rate(x)`` optionally provides the inner rate for its derivative term.
    """

    def __init__(self, outer, inner, inner_measurement: Callable[[np.ndarray], float],
                 inner_rate: Callable[[np.ndarray], float] | None = None):
        self.outer, self.inner = outer, inner
        self.inner_measurement, self.inner_rate = inner_measurement, inner_rate

    def act(self, t, x, y, r, ydot, dt):
        r_in = self.outer.act(t, x, y, r, ydot, dt)
        yi = float(self.inner_measurement(x))
        return self.inner.act(t, x, yi, r_in, None if self.inner_rate is None else float(self.inner_rate(x)), dt)

    def set_equilibrium(self, y, r, command, t=0.0, x=None):
        if x is None:
            raise ValueError("a cascade needs the equilibrium state x to set its inner loop")
        yi = float(self.inner_measurement(x))
        self.outer.set_equilibrium(y, r, yi, t)
        self.inner.set_equilibrium(yi, yi, command, t)

    def get_state(self):
        a, b = self.outer.get_state(), self.inner.get_state()
        return np.concatenate([[a.size], a, b])

    def set_state(self, s):
        n = int(s[0])
        self.outer.set_state(s[1:1 + n])
        self.inner.set_state(s[1 + n:])


class StateFeedback:
    """``u = u_ref - K (x - x_ref)``, optionally with integral action on the output error (``ki``).

    Parameters
    ----------
    K : array_like
        Gain row (command units per state unit).
    x_ref : array_like
        State the feedback regulates to (state units).
    u_ref : float
        Command at ``x_ref`` (command units).
    ki : float
        Integral gain on ``r - y`` (command units per output unit-second).
    output_limits : tuple, optional
        Saturation of the command; the integral is frozen while saturated.
    """

    def __init__(self, K, x_ref, u_ref: float = 0.0, ki: float = 0.0,
                 output_limits: tuple[float, float] | None = None):
        self.K = np.atleast_1d(np.asarray(K, float)).ravel()
        self.x_ref = np.asarray(x_ref, float)
        self.u_ref, self.ki, self.output_limits = float(u_ref), float(ki), output_limits
        self._integral = 0.0

    def act(self, t, x, y, r, ydot, dt):
        e = r - y
        integral = self._integral + e * dt
        v = self.u_ref - float(self.K @ (np.asarray(x, float) - self.x_ref)) + self.ki * integral
        if self.output_limits is None:
            self._integral = integral
            return v
        lo, hi = self.output_limits
        u = float(min(max(v, lo), hi))
        if lo <= v <= hi:
            self._integral = integral
        return u

    def set_equilibrium(self, y, r, command, t=0.0, x=None):
        """With integral action, the integral takes up the difference between ``command`` and the feedback at
        ``x``; without it nothing changes (``u_ref`` and ``x_ref`` define the target)."""
        base = self.u_ref - (0.0 if x is None else float(self.K @ (np.asarray(x, float) - self.x_ref)))
        self._integral = (command - base) / self.ki if self.ki else 0.0

    def get_state(self):
        return np.array([self._integral], float)

    def set_state(self, s):
        self._integral = float(s[0])


def _integrate(plant: Plant, x: np.ndarray, u: float, d, t: float, dt: float, method: str) -> np.ndarray:
    f = plant.rhs
    if method == "rk4":
        k1 = np.asarray(f(x, u, d, t), float)
        k2 = np.asarray(f(x + 0.5 * dt * k1, u, d, t + 0.5 * dt), float)
        k3 = np.asarray(f(x + 0.5 * dt * k2, u, d, t + 0.5 * dt), float)
        k4 = np.asarray(f(x + dt * k3, u, d, t + dt), float)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    if method == "euler":
        return x + dt * np.asarray(f(x, u, d, t), float)
    if method == "sequential_euler":
        # semi-implicit (symplectic) Euler generalised: update the components in order, each from the latest values
        # (state [velocity, position] gives v += a(x) dt, then x += v_new dt)
        z = x.copy()
        for i in range(z.size):
            z[i] = z[i] + dt * float(np.asarray(f(z, u, d, t), float)[i])
        return z
    raise ValueError("method must be 'rk4', 'euler' or 'sequential_euler'")


def _set_equilibrium(controller, y: float, r: float, command: float, t: float, x: np.ndarray) -> None:
    fn = controller.set_equilibrium
    if "x" in inspect.signature(fn).parameters:
        fn(y, r, command, t, x=x)
    else:
        fn(y, r, command, t)


def _signal(v, t: float):
    return v(t) if callable(v) else v


def simulate_loop(plant: Plant, controller, actuator: Actuator, reference: float | Callable[[float], float],
                  t_end: float, dt: float, disturbance: Any = 0.0, method: str = "rk4",
                  stop: Callable[[np.ndarray, float], str | None] | None = None, noise_std: float = 0.0,
                  sensor_tau: float = 0.0, seed: int = 0, x0: Sequence[float] | None = None,
                  start_at_equilibrium: bool = True, u0: float | None = None) -> dict[str, Any]:
    """Simulate a closed loop with a fixed step.

    At every step: measure ``y`` (with noise and an optional first-order sensor lag), let the controller compute
    the command from ``y``, the reference and (when the plant has one) the measured rate, pass the command through
    the actuator, and integrate the plant over ``dt`` with the applied value held.

    Parameters
    ----------
    plant : Plant
        The model.
    controller : object
        ``act(t, x, y, r, ydot, dt) -> command`` (e.g. :class:`PID`, :class:`Cascade`, :class:`StateFeedback`).
    actuator : Actuator
        Between the command and the plant.
    reference : float or callable
        Reference of the output, constant or ``t -> r`` (output units).
    t_end, dt : float [s]
        Duration and step.
    disturbance : float, array or callable
        ``d`` passed to the plant, constant or ``t -> d``.
    method : {'rk4', 'euler', 'sequential_euler'}
        Integrator.
    stop : callable, optional
        ``(x, t) -> reason or None``: ends the run when it returns a reason (e.g. a depth limit).
    noise_std : float
        Standard deviation of Gaussian measurement noise (output units).
    sensor_tau : float [s]
        First-order lag of the sensor (0: none).
    x0 : array_like, optional
        Initial state (default ``plant.x0``).
    start_at_equilibrium : bool
        Initialise the controller bumplessly so it outputs the command that holds the actuator at ``u0``.
    u0 : float, optional
        Initial applied value (default ``actuator.initial``).

    Returns
    -------
    t : ndarray [s]
        Times of the recorded samples (``t[0] = 0``).
    x : ndarray
        States (state units), one row per sample.
    y : ndarray
        True outputs (output units).
    applied : ndarray
        Applied actuator values (plant input units).
    command : ndarray
        Controller commands (command units); ``command[k]`` acted between ``t[k]`` and ``t[k+1]``.
    reference : ndarray
        References (output units).
    stopped : str or None
        Reason the run ended early.
    y_min, y_max : float
        Extremes of the output over the run (output units).
    iae : float
        Integral of ``|r - y|`` (output units times seconds).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(plant.x0 if x0 is None else x0, float).copy()
    a0 = actuator.initial if u0 is None else float(u0)
    actuator.reset(dt, a0)
    r0 = float(_signal(reference, 0.0))
    y_meas = plant.y(x)
    if start_at_equilibrium and hasattr(controller, "set_equilibrium"):
        # bumpless at the starting point: as if the reference had been the measurement, so a later (or
        # immediate) reference step acts on the controller as a step, and the command holds the actuator at u0
        _set_equilibrium(controller, plant.y(x), plant.y(x), actuator.equilibrium_command(actuator.applied), 0.0, x)
    n = int(round(t_end / dt))
    T, X, Y, U, C, R = [0.0], [x.copy()], [plant.y(x)], [actuator.applied], [], [r0]
    stopped = None
    for i in range(n):
        t = i * dt
        r = float(_signal(reference, t))
        y_true = plant.y(x)
        y_new = y_true + (rng.normal(0.0, noise_std) if noise_std > 0 else 0.0)
        y_meas = y_new if sensor_tau <= 0 or i == 0 else y_meas + (y_new - y_meas) * dt / (sensor_tau + dt)
        ydot = float(plant.rate(x)) if plant.rate is not None else None
        cmd = float(controller.act(t, x, y_meas, r, ydot, dt))
        u = actuator.step(cmd, dt)
        x = _integrate(plant, x, u, _signal(disturbance, t), t, dt, method)
        T.append((i + 1) * dt)
        X.append(x.copy())
        Y.append(plant.y(x))
        U.append(u)
        C.append(cmd)
        R.append(r)
        if not np.all(np.isfinite(x)):
            stopped = "state is no longer finite"
            break
        if stop is not None:
            why = stop(x, (i + 1) * dt)
            if why:
                stopped = str(why)
                break
    T, Y, R_ = np.array(T), np.array(Y), np.array(R)
    return {"t": T, "x": np.array(X), "y": Y, "applied": np.array(U), "command": np.array(C), "reference": R_,
            "stopped": stopped, "y_min": float(np.min(Y)), "y_max": float(np.max(Y)),
            "iae": float(np.trapezoid(np.abs(R_ - Y), T)) if len(T) > 1 else 0.0}


def loop_eigenvalues(plant: Plant, controller, actuator: Actuator, x_eq: Sequence[float], u_eq: float,
                     reference: float, dt: float, disturbance: Any = 0.0, method: str = "rk4", t: float = 0.0,
                     rel_step: float = 1e-6) -> dict[str, Any]:
    """Stability of the complete discrete-time closed loop around an equilibrium.

    The loop's state is the plant state, the actuator state and the controller state. The one-step map of
    :func:`simulate_loop` is differentiated numerically at the equilibrium, so the eigenvalues include the
    actuator (integrating or not, its lag and delay), the controller's integral and derivative states and the
    integration scheme. Limits must be inactive at the equilibrium; rate limits are, since nothing moves there.

    Parameters
    ----------
    x_eq : array_like
        Plant equilibrium with the actuator at ``u_eq`` and the given disturbance (state units).
    u_eq : float
        Applied actuator value at equilibrium (plant input units).
    reference : float
        Reference at equilibrium (output units).
    dt : float [s]
        Step of the loop.

    Returns
    -------
    stable : bool
        All eigenvalues of the one-step map inside the unit circle.
    spectral_radius : float [-]
        Largest eigenvalue modulus of the one-step map.
    classification : str
        From :func:`.analysis.stability` on the continuous-time equivalents.
    s : ndarray [1/s]
        Continuous-time equivalents ``log(z)/dt`` of the eigenvalues.
    max_real : float [1/s]
        Largest real part of ``s``: positive means the loop diverges, at that e-folding rate.
    doubling_time : float [s]
        ``ln 2 / max_real`` for an unstable loop (``nan`` when stable).
    fixed_point_residual : float [-]
        Relative change of the loop state over one step at the equilibrium (should be ~0).
    ignored_states : int
        Controller or actuator states that influence no other state (their eigenvalue is left out).
    """
    ctrl, act = copy.deepcopy(controller), copy.deepcopy(actuator)
    x_eq = np.asarray(x_eq, float)
    act.reset(dt, u_eq)
    y_eq = plant.y(x_eq)
    cmd_eq = act.equilibrium_command(u_eq)
    _set_equilibrium(ctrl, y_eq, reference, cmd_eq, t, x_eq)
    nx = x_eq.size
    sa, sc = act.get_state(), ctrl.get_state()
    na = sa.size
    z0 = np.concatenate([x_eq, sa, sc])
    valid = np.isfinite(z0)
    z0f = np.where(valid, z0, 0.0)

    def phi(z):
        zz = np.where(valid, z, np.nan)
        c, a = copy.deepcopy(ctrl), copy.deepcopy(act)
        a.set_state(zz[nx:nx + na])
        c.set_state(zz[nx + na:])
        x = zz[:nx]
        y = plant.y(x)
        ydot = float(plant.rate(x)) if plant.rate is not None else None
        cmd = float(c.act(t, x, y, reference, ydot, dt))
        u = a.step(cmd, dt)
        xn = _integrate(plant, x, u, _signal(disturbance, t), t, dt, method)
        out = np.concatenate([xn, a.get_state(), c.get_state()])
        return np.where(valid, out, 0.0)

    f0 = phi(z0f)
    resid = float(np.linalg.norm(f0 - z0f) / max(np.linalg.norm(z0f), 1.0))
    keep = np.where(valid)[0]
    J = np.zeros((keep.size, keep.size))
    for jj, j in enumerate(keep):
        h = rel_step * max(1.0, abs(z0f[j]))
        zp, zm = z0f.copy(), z0f.copy()
        zp[j] += h
        zm[j] -= h
        J[:, jj] = (phi(zp)[keep] - phi(zm)[keep]) / (2 * h)
    # drop states that influence no other state (e.g. the integral of a PID with ki = 0, a stored copy): their
    # eigenvalue is their own diagonal entry and says nothing about the loop
    live = np.ones(J.shape[0], bool)
    while True:
        Jl = J[np.ix_(live, live)]
        off = np.abs(Jl - np.diag(np.diag(Jl))).sum(axis=0)
        dead = np.where(off <= 1e-12)[0]
        dead = [d for d in dead if np.where(live)[0][d] >= nx]       # plant states always stay
        if not dead or live.sum() <= 1:
            break
        live[np.where(live)[0][dead]] = False
    J = J[np.ix_(live, live)]
    st = stability(J, dt=dt)
    zl = np.linalg.eigvals(J)
    rho = float(np.max(np.abs(zl))) if zl.size else 0.0
    mr = st["max_real"]
    return {"stable": bool(rho < 1.0 - 1e-9), "spectral_radius": rho, "classification": st["classification"],
            "s": st["s"], "z": zl, "max_real": mr,
            "doubling_time": float(math.log(2.0) / mr) if mr > 1e-12 else float("nan"),
            "fixed_point_residual": resid, "ignored_states": int((~live).sum())}


def compare_architectures(plant: Plant, architectures: Mapping[str, tuple[Any, Actuator]], reference: float,
                          t_end: float, dt: float, disturbance: Any = 0.0, method: str = "rk4",
                          stop: Callable[[np.ndarray, float], str | None] | None = None,
                          x_eq: Sequence[float] | None = None, u_eq: float | None = None,
                          d_eq: Any = 0.0) -> dict[str, Any]:
    """Run several (controller, actuator) choices on the same plant, disturbance and integrator.

    Each architecture is simulated with :func:`simulate_loop` and, when ``x_eq`` and ``u_eq`` are given, analysed
    with :func:`loop_eigenvalues` at the undisturbed equilibrium (``d_eq``).

    Returns
    -------
    rows : list of dict
        Per architecture:
        ``name``, ``stable_linear`` (bool or None), ``spectral_radius`` [-], ``max_real`` [1/s],
        ``doubling_time`` [s], ``y_min`` and ``y_max`` (output units), ``final_y`` (output units),
        ``iae`` (output units times seconds), ``stopped`` (str or None) and ``t_end`` [s] reached.
    runs : dict
        The full :func:`simulate_loop` result per architecture.
    """
    rows, runs = [], {}
    for name, (ctrl, act) in architectures.items():
        sim = simulate_loop(plant, copy.deepcopy(ctrl), copy.deepcopy(act), reference, t_end, dt,
                            disturbance=disturbance, method=method, stop=stop,
                            u0=u_eq if u_eq is not None else None)
        runs[name] = sim
        row = {"name": name, "y_min": sim["y_min"], "y_max": sim["y_max"], "final_y": float(sim["y"][-1]),
               "iae": sim["iae"], "stopped": sim["stopped"], "t_end": float(sim["t"][-1]),
               "stable_linear": None, "spectral_radius": float("nan"), "max_real": float("nan"),
               "doubling_time": float("nan")}
        if x_eq is not None and u_eq is not None:
            le = loop_eigenvalues(plant, ctrl, act, x_eq, u_eq, reference, dt, disturbance=d_eq, method=method)
            row.update(stable_linear=le["stable"], spectral_radius=le["spectral_radius"], max_real=le["max_real"],
                       doubling_time=le["doubling_time"])
        rows.append(row)
    return {"rows": rows, "runs": runs}
