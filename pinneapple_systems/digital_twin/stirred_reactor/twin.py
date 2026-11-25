"""The twin: a lumped fast model, an extended Kalman filter that carries the cooling it was not
told about, and the alarm decision with its safety margin.

What the twin is
----------------
The reference of :mod:`~.plant` resolves the five cooling tubes one by one. The twin does not: it
carries a **single** conductance, the nominal one of five tubes, multiplied by a latent state
``phi`` that the filter estimates from the measurements. ``phi = 1`` means every tube is in
service; a tube that is isolated (or fouled, or valved off) shows up as ``phi < 1`` without the
twin ever being told which tube it was. That is the structural gap the benchmark exists to
measure: the twin's model is *wrong* the moment the plant changes, and the estimator is what
keeps its state honest.

What the twin is told
---------------------
The four inputs — feed flow, feed temperature, coolant inlet temperature and impeller speed — are
instrumented and reach the twin every sample, because they are what an operator sets. So a
stopped impeller is *known* to the twin (the film coefficient falls with the speed and the twin
computes that), and its alarm for that case is a forecast, not a surprise. A failed tube is not
instrumented: nothing measures the flow through one tube, which is why it is a perturbation and
not an input.

The estimator
-------------
An extended Kalman filter over ``[c_A, c_B, T, phi]``, driven by the two measurements (bulk
temperature and reactant concentration). The process noise on ``phi`` is a random walk: the
cooling capacity is allowed to change slowly and without a stated reason, which is the assumption
an operator makes. The filter is
:class:`pinneapple_analysis.state_estimation.kalman.ExtendedKalmanFilter`, the same one the
pipe-loop twin uses, and it runs one predict-and-fuse per sample period.

The alarm
---------
Two ways to alarm, both reported:

``now``
    the filtered temperature is at or above the alarm setpoint;
``forecast``
    the model, rolled forward from the estimate with ``phi`` frozen and the current inputs held,
    reaches the setpoint inside the horizon. This is the one that gives lead time, and it is why
    a stopped impeller is caught minutes after the trip instead of an hour later.

The **safety margin** the twin reports is ``t_trip - max(forecast temperature)``: how much room
is left before the interlock, predicted from where the plant is now. The forecast holds the inputs
where they are, because a twin has no way to know the operator will open a valve — so a coolant that
keeps drifting is caught later than a step the twin is told about, and the benchmark reports the
difference instead of hiding it: every alarm carries its signed lead time, and the cooling family
(a stopped drive, which is instrumented) leads the margin family (a drifting coolant, which is not).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinneapple_analysis.state_estimation.kalman import ExtendedKalmanFilter

from .plant import T0_K, Inputs, Reactor, ReferenceRun, Scenario, run_reference

__all__ = ["LumpedModel", "LumpedTwin", "OpenLoop", "TwinRun", "alarm_scores", "run_twin"]


class LumpedModel:
    """The twin's physics: the reference equations with the five tubes replaced by one
    conductance times the latent factor ``phi``, integrated by explicit Euler.

    Coarse on purpose. The twin has to answer inside the sample period, so it takes a few Euler
    sub-steps where the reference takes a stiff solver at ``rtol = 1e-10``; the difference is part
    of what the state error measures, and it is separated from the structural difference (one
    conductance against five) by running the same step size in both.
    """

    def __init__(self, reactor: Reactor, sub_steps: int = 3) -> None:
        self.reactor = reactor
        self.sub_steps = max(int(sub_steps), 1)

    def nominal_duty(self, t_k: float, inputs: Inputs, coefficient: float | None = None) -> float:
        """The cooling the twin believes in: five tubes, every one in service."""
        k = self.reactor.duty_coefficient(inputs.n_impeller) if coefficient is None else coefficient
        return k * (t_k - inputs.t_coolant_k)

    def derivative(self, x: np.ndarray, inputs: Inputs, coefficient: float | None = None) -> np.ndarray:
        R = self.reactor
        c_a, c_b, t_k, phi = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        r = R.rate(c_a, c_b, t_k)
        q = max(phi, 0.0) * self.nominal_duty(t_k, inputs, coefficient)
        thermal = R.rho * R.cp * R.volume_m3
        dil = inputs.feed_m3_s / R.volume_m3
        return np.array([dil * (R.c_feed - c_a) - r,
                         -dil * c_b + r,
                         dil * (inputs.t_feed_k - t_k)
                         + (-R.delta_h) * r / (R.rho * R.cp)
                         - q / thermal
                         - R.ua_wall_w_k * (t_k - (R.t_ambient_c + T0_K)) / thermal,
                         0.0])

    def advance(self, x: np.ndarray, inputs: Inputs, dt_s: float,
                coefficient: float | None = None) -> np.ndarray:
        h = float(dt_s) / self.sub_steps
        y = np.asarray(x, dtype=float).copy()
        for _ in range(self.sub_steps):
            y = y + h * self.derivative(y, inputs, coefficient)
        if y[3] < 0.0:
            y[3] = 0.0
        return y

    def settled_temperature(self, x: np.ndarray, inputs: Inputs, bracket: tuple[float, float] = (250.0, 520.0)) -> float:
        """Stationary temperature of the twin's own model for these inputs and this ``phi``.

        Same algebra as the reference's stationary solve, with ``phi`` multiplying the duty: the
        conversion at a trial temperature comes from the mass balance, the energy balance is then
        one equation in one unknown, and the hottest root is taken (an exothermic tank can have
        more than one stationary point, and the operator cares about the hot one).
        """
        R = self.reactor
        phi = max(float(x[3]), 0.0)

        def g(t_k: float) -> float:
            tau = R.volume_m3 / inputs.feed_m3_s
            kf = R.k_forward(t_k)
            xc = kf * tau / (1.0 + kf * tau + kf / R.equilibrium_constant(t_k) * tau)
            generated = (-R.delta_h) * R.c_feed * xc * inputs.feed_m3_s
            feed = R.rho * R.cp * inputs.feed_m3_s * (inputs.t_feed_k - t_k)
            return (generated + feed - phi * self.nominal_duty(t_k, inputs)
                    - R.ua_wall_w_k * (t_k - (R.t_ambient_c + T0_K)))

        lo, hi = bracket
        grid = np.linspace(lo, hi, 200)
        values = np.array([g(t) for t in grid])
        roots = []
        for i in range(len(grid) - 1):
            if values[i] * values[i + 1] < 0.0:
                a, b, fa = float(grid[i]), float(grid[i + 1]), float(values[i])
                for _ in range(70):
                    mid = 0.5 * (a + b)
                    fm = g(mid)
                    if fa * fm <= 0.0:
                        b = mid
                    else:
                        a, fa = mid, fm
                roots.append(0.5 * (a + b))
        return (max(roots) if roots else float(x[2])) - T0_K


@dataclass
class LumpedTwin:
    """The fast model plus the filter: four states, two measurements, one sample at a time."""
    reactor: Reactor
    sub_steps: int = 3
    q_state: tuple[float, float, float] = (1.0, 1.0, 2.5e-3)   # mol/m3, mol/m3, K: per-step sigma
    q_phi: float = 2.0e-3              # random walk on the cooling effectiveness, per-step sigma
    horizon_s: float = 5400.0          # the forecast the alarm is judged on (90 min)
    model: LumpedModel = field(init=False, repr=False)
    ekf: ExtendedKalmanFilter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model = LumpedModel(self.reactor, self.sub_steps)
        self._inputs = None
        self._dt = 1.0
        self._coeff = None
        self._sigma = (1.0, 40.0)
        self.ekf = ExtendedKalmanFilter(
            4, 2, f=self._transition, h=self._observe,
            Q=np.diag([self.q_state[0] ** 2, self.q_state[1] ** 2, self.q_state[2] ** 2, self.q_phi ** 2]),
            R=np.diag([self._sigma[0] ** 2, self._sigma[1] ** 2]),
            F_jac=self._F, H_jac=self._H)

    # -- callables the filter needs
    def _transition(self, x: np.ndarray) -> np.ndarray:
        return self.model.advance(x, self._inputs, self._dt, self._coeff)

    def _observe(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[2] - T0_K, x[0]])

    def _F(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of one sample period, by central differences (four states: cheap and enough)."""
        F = np.empty((4, 4))
        for j in range(4):
            dx = np.zeros(4)
            dx[j] = 1e-6 if j < 3 else 1e-4
            F[:, j] = (self._transition(x + dx) - self._transition(x - dx)) / (2.0 * dx[j])
        return F

    def _H(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((2, 4))
        H[0, 2] = 1.0
        H[1, 0] = 1.0
        return H

    # ------------------------------------------------------------------ lifecycle
    def set_noise(self, sigma: tuple[float, float]) -> None:
        """The instrument noise the filter is told about (it is the plant's own specification)."""
        self._sigma = (float(sigma[0]), float(sigma[1]))
        self.ekf.R = np.diag([self._sigma[0] ** 2, self._sigma[1] ** 2])

    def start(self, x0: np.ndarray, phi_sigma: float = 0.02) -> None:
        """Commission the twin at a state, with a small covariance: it starts where it was designed."""
        self.ekf.initialize(np.asarray(x0, dtype=float).copy(), np.diag([1e-6, 1e-6, 1e-6, phi_sigma ** 2]))

    def step(self, z: np.ndarray, inputs: Inputs, dt_s: float) -> dict[str, Any]:
        """One sample: predict over ``dt_s``, then fuse the two readings. The clock is here."""
        self._inputs, self._dt = inputs, float(dt_s)
        self._coeff = self.reactor.duty_coefficient(inputs.n_impeller)
        t0 = time.perf_counter()
        out = self.ekf.step(np.asarray(z, dtype=float))
        seconds = time.perf_counter() - t0
        return {"x": out["x"].copy(), "innovation": out["innovation"],
                "standard_deviation": np.sqrt(np.maximum(np.diag(out["P"]), 0.0)), "seconds": seconds}

    # ------------------------------------------------------------------ forecast and alarm
    def forecast(self, x: np.ndarray, inputs: Inputs, horizon_s: float | None = None,
                 step_s: float = 60.0) -> dict[str, Any]:
        """Roll the model forward with ``phi`` frozen and the inputs held: where the tank is going.

        The inputs are held where they are, which is the honest assumption of a twin that has no
        way to know the operator will open a valve: a coolant drifting away is *not* in this
        forecast, and the benchmark measures the alarms that costs.
        """
        horizon = self.horizon_s if horizon_s is None else horizon_s
        coefficient = self.reactor.duty_coefficient(inputs.n_impeller)
        y = np.asarray(x, dtype=float).copy()
        n = max(int(round(horizon / step_s)), 1)
        peak, when = float(y[2]) - T0_K, 0.0
        for i in range(1, n + 1):
            y = self.model.advance(y, inputs, step_s, coefficient)
            temp = float(y[2]) - T0_K
            if temp > peak:
                peak, when = temp, i * step_s
        return {"peak_c": peak, "peak_at_s": when,
                "exceeds_alarm": bool(peak >= self.reactor.t_alarm_c),
                "exceeds_trip": bool(peak >= self.reactor.t_trip_c),
                "margin_to_trip_k": self.reactor.t_trip_c - peak}


class OpenLoop:
    """The same lumped model with the tubes at nominal and **no** assimilation: the state error a
    twin would carry if it only believed its own equations. It is the column the filter is
    measured against, and it uses the same step size, so the difference is the data."""

    def __init__(self, reactor: Reactor, sub_steps: int = 3) -> None:
        self.model = LumpedModel(reactor, sub_steps)
        self.x = None

    def start(self, x0: np.ndarray) -> None:
        self.x = np.asarray(x0, dtype=float).copy()
        self.x[3] = 1.0

    def step(self, inputs: Inputs, dt_s: float) -> np.ndarray:
        self.x = self.model.advance(self.x, inputs, float(dt_s),
                                    self.model.reactor.duty_coefficient(inputs.n_impeller))
        return self.x


def _charge(reactor: Reactor, inputs: Inputs) -> np.ndarray:
    """Where the plant starts and where the twin is commissioned: the stationary point."""
    st = reactor.steady_state(inputs)
    return np.array([st["c_a"], st["c_b"], st["t_k"], 1.0])


@dataclass
class TwinRun:
    """What the twin did over one scenario: estimates, errors, forecast, alarms, timings."""
    reference: ReferenceRun
    t: np.ndarray
    estimate: np.ndarray            # (n, 4) c_A, c_B, T[K], phi
    standard_deviation: np.ndarray  # (n, 4)
    open_loop: np.ndarray           # (n, 3) the unfiltered model
    forecast_peak_c: np.ndarray
    forecast_margin_k: np.ndarray
    settled_c_end: float            # where the twin's own model says the tank will end up
    alarm_now: np.ndarray
    alarm_forecast: np.ndarray
    filter_seconds: np.ndarray      # (n,) wall time of one filter step
    forecast_seconds: np.ndarray    # (n,) wall time of the rollout that decides the alarm

    # ---- errors against the reference
    @property
    def temperature_error(self) -> np.ndarray:
        return self.estimate[:, 2] - self.reference.y_true[:, 2]

    @property
    def conversion_error(self) -> np.ndarray:
        R = self.reference.reactor
        return (1.0 - self.estimate[:, 0] / R.c_feed) - self.reference.conversion

    @property
    def open_loop_temperature_error(self) -> np.ndarray:
        return self.open_loop[:, 2] - self.reference.y_true[:, 2]

    @property
    def alarm(self) -> np.ndarray:
        return self.alarm_now | self.alarm_forecast

    def summary(self) -> dict[str, Any]:
        """One row of the results table, in the units the report quotes."""
        measured = self.reference.readings[:, 0]
        step = np.median(np.diff(self.t))
        return {
            "name": self.reference.scenario.name,
            "kind": self.reference.scenario.kind,
            "noise": self.reference.scenario.noise,
            "temperature_error_rms_k": float(np.sqrt(np.mean(self.temperature_error ** 2))),
            "temperature_error_max_k": float(np.abs(self.temperature_error).max()),
            "conversion_error_rms_pp": float(100.0 * np.sqrt(np.mean(self.conversion_error ** 2))),
            "open_loop_temperature_error_rms_k": float(np.sqrt(np.mean(self.open_loop_temperature_error ** 2))),
            "sensor_temperature_error_rms_k": float(np.sqrt(np.mean((measured - self.reference.temperature_c) ** 2))),
            "phi_end": float(self.estimate[-1, 3]),
            "phi_spread_end": float(self.standard_deviation[-1, 3]),
            "phi_truth_end": float(self.reference.effective_phi(len(self.t) - 1)),
            "settled_c_end": float(self.settled_c_end),
            "latency_median_ms": float(np.median(self.filter_seconds + self.forecast_seconds) * 1e3),
            "latency_p95_ms": float(np.quantile(self.filter_seconds + self.forecast_seconds, 0.95) * 1e3),
            "latency_max_ms": float((self.filter_seconds + self.forecast_seconds).max() * 1e3),
            "sample_period_s": float(step),
            "reference_seconds": float(self.reference.seconds),
            "samples": int(len(self.t)),
            **alarm_scores(self),
        }


def alarm_scores(run: TwinRun) -> dict[str, Any]:
    """The alarm column: did the truth cross, did the twin say so, and how early."""
    R = run.reference.reactor
    t = run.t
    truth = run.reference.temperature_c >= R.t_alarm_c
    truth_at = float(t[truth][0]) if truth.any() else None
    trip = run.reference.temperature_c >= R.t_trip_c
    alarm = run.alarm
    first = float(t[alarm][0]) if alarm.any() else None
    return {
        "truth_alarmed": bool(truth.any()), "truth_alarm_at_s": truth_at,
        "truth_tripped": bool(trip.any()),
        "twin_alarmed": bool(alarm.any()), "twin_alarm_at_s": first,
        "twin_alarm_now_at_s": float(t[run.alarm_now][0]) if run.alarm_now.any() else None,
        "twin_alarm_forecast_at_s": float(t[run.alarm_forecast][0]) if run.alarm_forecast.any() else None,
        "detected": bool(truth.any() and alarm.any()),
        "missed": bool(truth.any() and not alarm.any()),
        "false_alarm": bool(alarm.any() and not truth.any()),
        "lead_s": None if (first is None or truth_at is None) else truth_at - first,
        "margin_reported_end_k": float(run.forecast_margin_k[-1]),
        "margin_truth_end_k": float(R.t_trip_c - run.reference.temperature_c[-1]),
        "margin_truth_worst_k": float(R.t_trip_c - run.reference.temperature_c.max()),
    }


def run_twin(reactor: Reactor, sc: Scenario, reference: ReferenceRun | None = None,
             *, twin: LumpedTwin | None = None, open_loop: bool = True) -> TwinRun:
    """Run the twin sample by sample over a reference run and measure what it did at each one.

    The twin sees the noisy readings and the four instrumented inputs, nothing else. Its state
    error is taken against the reference's true state at the same instant, which is the only
    definition that means anything: the reference is what the plant did.
    """
    ref = reference if reference is not None else run_reference(reactor, sc)
    twin = twin or LumpedTwin(reactor)
    twin.set_noise(sc.sigma)
    x0 = _charge(reactor, ref.inputs_at(0))
    twin.start(x0)
    loop = OpenLoop(reactor, sub_steps=twin.sub_steps) if open_loop else None
    if loop is not None:
        loop.start(x0)

    n = len(ref.t)
    estimate = np.empty((n, 4))
    spread = np.empty((n, 4))
    open_state = np.empty((n, 3))
    peak = np.empty(n)
    margin = np.empty(n)
    a_now = np.zeros(n, bool)
    a_fc = np.zeros(n, bool)
    filter_seconds = np.zeros(n)
    forecast_seconds = np.zeros(n)
    estimate[0] = twin.ekf.x.copy()
    spread[0] = np.sqrt(np.diag(twin.ekf.P))
    open_state[0] = x0[:3]
    for i in range(n):
        inputs = ref.inputs_at(i)
        if i > 0:
            out = twin.step(ref.readings[i], inputs, ref.t[i] - ref.t[i - 1])
            estimate[i], spread[i], filter_seconds[i] = out["x"], out["standard_deviation"], out["seconds"]
            if loop is not None:
                open_state[i] = loop.step(inputs, ref.t[i] - ref.t[i - 1])[:3]
        clock = time.perf_counter()
        fc = twin.forecast(estimate[i], inputs)
        forecast_seconds[i] = time.perf_counter() - clock
        peak[i], margin[i] = fc["peak_c"], fc["margin_to_trip_k"]
        a_now[i] = estimate[i, 2] - T0_K >= reactor.t_alarm_c
        a_fc[i] = fc["exceeds_alarm"]
    settled = twin.model.settled_temperature(estimate[-1], ref.inputs_at(len(ref.t) - 1))
    return TwinRun(reference=ref, t=ref.t, estimate=estimate, standard_deviation=spread,
                   open_loop=open_state, forecast_peak_c=peak, forecast_margin_k=margin,
                   settled_c_end=float(settled), alarm_now=a_now, alarm_forecast=a_fc,
                   filter_seconds=filter_seconds, forecast_seconds=forecast_seconds)
