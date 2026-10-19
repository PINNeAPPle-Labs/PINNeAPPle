"""Heated buffer tank: a lumped (0-D) plant with a mass and an energy balance, its sensors and its faults.

One plant shared by the operations tools (digital twin runtime, physics anomaly detection, physics-based
control), so a twin's state estimate, an anomaly detector's balances and a controller's model all describe the
same equations and can be checked against each other.

Model (states ``h`` liquid level [m], ``T`` liquid temperature [degC]; inputs ``q_in`` pump inflow [m3/s],
``Q`` heater power [W]; disturbances ``T_in`` inlet temperature, ``T_amb`` ambient temperature, ``q_leak`` an
unmeasured outflow)::

    q_out        = k_out * sqrt(h)                                   (gravity drain through a fixed valve)
    A dh/dt      = q_in - q_out - q_leak                             (mass balance, constant density)
    A h dT/dt    = q_in (T_in - T) + (eta Q - UA (T - T_amb)) / (rho cp)
                                                                      (energy balance, perfectly mixed)

The energy equation is the product rule applied to ``d(rho cp A h T)/dt = rho cp (q_in T_in - (q_out + q_leak) T)
+ eta Q - UA (T - T_amb)`` with the mass balance substituted, so both balances in their conservative form hold
exactly for a fault-free plant; :func:`mass_balance_residual` and :func:`energy_balance_residual` evaluate them
from measured signals.

Faults (each with a start and end time) change the *plant* or the *sensors*, never the equations used by the
detectors: ``leak`` (unmeasured outflow), ``heater`` (efficiency ``eta`` drops), ``sensor_bias`` (a constant
offset on one sensor), ``sensor_drift`` (a ramp on one sensor), ``stuck_sensor`` (a sensor freezes) and ``valve``
(the drain valve partially closes: ``k_out`` scaled).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace

import numpy as np

STATE_NAMES = ("h", "T")
INPUT_NAMES = ("q_in", "Q")
SENSOR_NAMES = ("h", "T", "q_in", "q_out", "Q", "T_in", "T_amb")
SENSOR_UNITS = {"h": "m", "T": "degC", "q_in": "m3/s", "q_out": "m3/s", "Q": "W", "T_in": "degC",
                "T_amb": "degC"}
FAULT_KINDS = ("leak", "heater", "sensor_bias", "sensor_drift", "stuck_sensor", "valve")


@dataclass
class TankParams:
    """Physical parameters of the heated tank (SI units, temperatures in degC)."""

    area: float = 1.5            # cross-section A [m2]
    k_out: float = 0.006         # drain coefficient [m^2.5/s]: q_out = k_out sqrt(h)
    ua: float = 500.0            # wall heat-loss coefficient UA [W/K]
    rho: float = 1000.0          # density [kg/m3]
    cp: float = 4186.0           # specific heat [J/(kg K)]
    h_min: float = 0.05          # level below which the model is not valid (heater uncovered) [m]
    h_max: float = 2.0           # overflow level [m]
    q_in_max: float = 0.015      # pump capacity [m3/s]
    q_max_heater: float = 600e3  # heater capacity [W]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fault:
    """A fault active on ``[start, end)`` seconds. ``target`` names the sensor for sensor faults."""

    kind: str
    start: float
    end: float = float("inf")
    magnitude: float = 0.0
    target: str = ""

    def __post_init__(self):
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}; one of {FAULT_KINDS}")
        if self.kind.startswith(("sensor", "stuck")) and self.target not in SENSOR_NAMES:
            raise ValueError(f"fault {self.kind!r} needs a target sensor in {SENSOR_NAMES}")
        if self.end <= self.start:
            raise ValueError("fault end must be after its start")

    def active(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return (t >= self.start) & (t < self.end)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["end"] = None if not np.isfinite(self.end) else self.end
        return d


def tank_rhs(x, u, d, p: TankParams, eta: float = 1.0, valve: float = 1.0, q_leak: float = 0.0) -> np.ndarray:
    """Time derivative ``[dh/dt, dT/dt]`` at state ``x = [h, T]``, inputs ``u = [q_in, Q]`` and disturbances
    ``d = [T_in, T_amb]``. Works on arrays of states with the leading axis being the state component."""
    h, T = x[0], x[1]
    q_in, Q = u[0], u[1]
    T_in, T_amb = d[0], d[1]
    hp = np.maximum(h, 1e-6)
    q_out = valve * p.k_out * np.sqrt(hp)
    dh = (q_in - q_out - q_leak) / p.area
    dT = (q_in * (T_in - T) + (eta * Q - p.ua * (T - T_amb)) / (p.rho * p.cp)) / (p.area * np.maximum(hp, p.h_min))
    return np.array([dh, dT])


def tank_step(x, u, d, p: TankParams, dt: float, substeps: int = 4, **fault_kw) -> np.ndarray:
    """Advance the tank ``dt`` seconds with classical RK4 (inputs held constant over the step)."""
    x = np.asarray(x, dtype=float)
    hsub = dt / substeps
    for _ in range(substeps):
        k1 = tank_rhs(x, u, d, p, **fault_kw)
        k2 = tank_rhs(x + 0.5 * hsub * k1, u, d, p, **fault_kw)
        k3 = tank_rhs(x + 0.5 * hsub * k2, u, d, p, **fault_kw)
        k4 = tank_rhs(x + hsub * k3, u, d, p, **fault_kw)
        x = x + hsub / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[0] = np.maximum(x[0], 0.0)
    return x


def steady_state(p: TankParams, q_in: float, Q: float, T_in: float, T_amb: float) -> np.ndarray:
    """Equilibrium ``[h, T]`` for constant inputs (closed form)."""
    if q_in <= 0:
        raise ValueError("q_in must be positive for a steady level")
    h = (q_in / p.k_out) ** 2
    T = (q_in * T_in * p.rho * p.cp + Q + p.ua * T_amb) / (q_in * p.rho * p.cp + p.ua)
    return np.array([h, T])


def inputs_for_steady_state(p: TankParams, h: float, T: float, T_in: float, T_amb: float) -> np.ndarray:
    """Inputs ``[q_in, Q]`` that hold the tank at ``[h, T]`` (closed form)."""
    q_in = p.k_out * np.sqrt(h)
    Q = p.rho * p.cp * q_in * (T - T_in) + p.ua * (T - T_amb)
    return np.array([q_in, Q])


@dataclass
class Scenario:
    """Operating scenario: how the inputs and disturbances evolve. Callables take the time [s] and return a value.

    ``inputs`` may be ``None`` when a controller computes them (see :mod:`pinneapple_systems.operations.control`).
    """

    duration: float = 3600.0
    dt: float = 10.0
    x0: Sequence[float] = (1.2, 25.0)
    T_in: Callable[[float], float] = lambda t: 15.0
    T_amb: Callable[[float], float] = lambda t: 20.0
    q_in: Callable[[float], float] | None = None
    Q: Callable[[float], float] | None = None


@dataclass
class SensorNoise:
    """Standard deviation of each sensor's white measurement noise (SI units)."""

    sigma: dict[str, float] = field(default_factory=lambda: {
        "h": 0.003, "T": 0.05, "q_in": 5e-5, "q_out": 5e-5, "Q": 500.0, "T_in": 0.05, "T_amb": 0.1})


def _plant_modifiers(faults: Sequence[Fault], t: float) -> dict:
    eta, valve, leak = 1.0, 1.0, 0.0
    for f in faults:
        if not f.active(t):
            continue
        if f.kind == "leak":
            leak += f.magnitude
        elif f.kind == "heater":
            eta *= max(0.0, 1.0 - f.magnitude)
        elif f.kind == "valve":
            valve *= max(0.0, 1.0 - f.magnitude)
    return {"eta": eta, "valve": valve, "q_leak": leak}


def measure(truth: dict[str, float], t: float, faults: Sequence[Fault], noise: SensorNoise,
            rng: np.random.Generator, frozen: dict[str, float]) -> dict[str, float]:
    """Sensor readings for one time step: true value + noise, then sensor faults."""
    out = {}
    for name in SENSOR_NAMES:
        v = truth[name] + rng.normal(0.0, noise.sigma.get(name, 0.0))
        for f in faults:
            if f.target != name or not f.active(t):
                continue
            if f.kind == "sensor_bias":
                v += f.magnitude
            elif f.kind == "sensor_drift":
                v += f.magnitude * (t - f.start) / 3600.0       # magnitude per hour
            elif f.kind == "stuck_sensor":
                v = frozen.setdefault(name, v)          # holds the first reading after the fault starts
        out[name] = float(v)
    return out


class TankSimulator:
    """Step-by-step simulator of the heated tank with sensors and faults. Used open loop (scenario inputs),
    closed loop (a controller sets the inputs each step) or as the live source of a digital twin demo."""

    def __init__(self, params: TankParams | None = None, x0: Sequence[float] = (1.2, 25.0),
                 faults: Sequence[Fault] = (), noise: SensorNoise | None = None, seed: int = 0):
        self.p = params or TankParams()
        self.x = np.asarray(x0, dtype=float).copy()
        self.t = 0.0
        self.faults = list(faults)
        self.noise = noise or SensorNoise()
        self.rng = np.random.default_rng(seed)
        self._frozen: dict[str, float] = {}

    def truth(self, u: Sequence[float], d: Sequence[float]) -> dict[str, float]:
        mods = _plant_modifiers(self.faults, self.t)
        h, T = float(self.x[0]), float(self.x[1])
        return {"h": h, "T": T, "q_in": float(u[0]), "q_out": mods["valve"] * self.p.k_out * np.sqrt(max(h, 0.0)),
                "Q": float(u[1]), "T_in": float(d[0]), "T_amb": float(d[1]), "q_leak": mods["q_leak"],
                "eta": mods["eta"]}

    def read(self, u: Sequence[float], d: Sequence[float]) -> dict[str, float]:
        """Sensor readings at the current time (before stepping)."""
        stuck_now = {f.target for f in self.faults if f.kind == "stuck_sensor" and f.active(self.t)}
        self._frozen = {k: v for k, v in self._frozen.items() if k in stuck_now}
        return measure(self.truth(u, d), self.t, self.faults, self.noise, self.rng, self._frozen)

    def step(self, u: Sequence[float], d: Sequence[float], dt: float) -> None:
        u = np.clip(np.asarray(u, dtype=float), [0.0, 0.0], [self.p.q_in_max, self.p.q_max_heater])
        self.x = tank_step(self.x, u, d, self.p, dt, **_plant_modifiers(self.faults, self.t))
        self.t += dt


def simulate(scenario: Scenario, params: TankParams | None = None, faults: Sequence[Fault] = (),
             noise: SensorNoise | None = None, seed: int = 0) -> dict[str, np.ndarray]:
    """Open-loop run of ``scenario`` (its ``q_in`` and ``Q`` must be set). Returns columns: ``t``, the measured
    sensors, ``true_<state>``, ``true_q_leak``, ``true_eta`` and ``fault`` (name of the active fault or "")."""
    if scenario.q_in is None or scenario.Q is None:
        raise ValueError("an open-loop scenario needs q_in and Q schedules")
    sim = TankSimulator(params, scenario.x0, faults, noise, seed)
    n = int(round(scenario.duration / scenario.dt)) + 1
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    for _ in range(n):
        t = sim.t
        u = (scenario.q_in(t), scenario.Q(t))
        d = (scenario.T_in(t), scenario.T_amb(t))
        meas = sim.read(u, d)
        tr = sim.truth(u, d)
        meas.update({"t": t, "true_h": tr["h"], "true_T": tr["T"], "true_q_leak": tr["q_leak"],
                     "true_eta": tr["eta"]})
        rows.append(meas)
        labels.append(",".join(f.kind for f in faults if f.active(t)))
        sim.step(u, d, scenario.dt)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["fault"] = np.array(labels, dtype=object)
    return out


def demo_scenario(duration: float = 3600.0, dt: float = 10.0, seed: int = 0) -> Scenario:
    """A realistic day-shift profile: pump flow steps between production rates, heater follows a schedule,
    inlet water temperature wanders slowly. Deterministic for a given ``seed``."""
    rng = np.random.default_rng(seed)
    p = TankParams()
    q_nom = inputs_for_steady_state(p, 1.2, 25.0, 15.0, 20.0)
    knots = np.arange(0.0, duration + 900.0, 900.0)
    q_lvls = q_nom[0] * (1.0 + 0.15 * rng.uniform(-1, 1, size=knots.size))
    Q_lvls = q_nom[1] * (1.0 + 0.12 * rng.uniform(-1, 1, size=knots.size))
    phase = rng.uniform(0, 2 * np.pi)

    def piecewise(levels):
        return lambda t: float(levels[min(int(t // 900.0), levels.size - 1)])

    return Scenario(duration=duration, dt=dt, x0=(1.2, 25.0),
                    T_in=lambda t: 15.0 + 0.8 * np.sin(2 * np.pi * t / 7200.0 + phase),
                    T_amb=lambda t: 20.0 + 0.5 * np.sin(2 * np.pi * t / 14400.0),
                    q_in=piecewise(q_lvls), Q=piecewise(Q_lvls))


def mass_balance_residual(t, h, q_in, q_out, area: float) -> np.ndarray:
    """``A dh/dt - (q_in - q_out)`` [m3/s] from measured signals (central differences in time, inputs averaged
    over each interval). Zero for a fault-free tank up to noise; a leak makes it equal to minus the leak flow."""
    t, h, q_in, q_out = (np.asarray(a, dtype=float) for a in (t, h, q_in, q_out))
    dhdt = np.gradient(h, t)
    return area * dhdt - (q_in - q_out)


def energy_balance_residual(t, h, T, q_in, q_out, Q, T_in, T_amb, p: TankParams) -> np.ndarray:
    """Energy balance in conservative form [W]::

        rho cp A d(hT)/dt - [rho cp (q_in T_in - q_out T) + Q - UA (T - T_amb)]

    Zero for a fault-free tank up to noise; a heater losing efficiency makes it negative (less energy arrives
    than the measured power says), a leak makes it equal to ``-rho cp q_leak T``."""
    arrs = [np.asarray(a, dtype=float) for a in (t, h, T, q_in, q_out, Q, T_in, T_amb)]
    t, h, T, q_in, q_out, Q, T_in, T_amb = arrs
    rc = p.rho * p.cp
    acc = rc * p.area * np.gradient(h * T, t)
    return acc - (rc * (q_in * T_in - q_out * T) + Q - p.ua * (T - T_amb))


def with_params(p: TankParams, **changes) -> TankParams:
    return replace(p, **changes)
