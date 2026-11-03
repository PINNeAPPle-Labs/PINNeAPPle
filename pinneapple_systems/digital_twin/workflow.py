"""The standard digital-twin workflow (#1193): a model, a data source, assimilation of the states and the slow
parameters, monitoring (filter consistency, drift, anomalies), recalibration through the inverse workflow, a
forecast with intervals, what-if runs and a dashboard page.

**Model.** A :class:`StateSpaceModel`: states, slow parameters, known inputs and measured outputs, with a batched
step ``x' = step(x, u, theta, dt)`` and an observation ``y = observe(x, theta)``. Any library solver, formulated
model, surrogate or zoo model wraps into one by writing those two functions; :func:`thermal_plant` is the built-in
one (a lumped heated body, exact zero-order-hold discretisation of ``C dT/dt = Q - UA (T - T_amb)``).

**Data.** A simulated plant (the same model, with process noise, a known parameter drift and an injected sensor
spike: the truth is known, so every check can be scored), a CSV replay (columns ``t``, the inputs, the outputs), or
arrays (``{"t": [...], <input>: [...], <output>: [...]}``) from a live adapter (MQTT, OPC-UA, REST:
:mod:`pinneapple_systems.digital_twin.io.stream`, the ``twin.connect`` node).

**Assimilation.** An ensemble Kalman filter of the states augmented with the logarithms of the slow parameters,
which follow a random walk (positive parameters, slow drift tracked): the deterministic ensemble transform analysis
(:func:`etkf_analysis`, default) or the stochastic one with perturbed observations (Burgers et al. 1998, through
:func:`pinneapple_analysis.state_estimation.kalman.enkf_analysis`).

**Monitoring.**

* Innovation statistics. The normalised innovation squared ``NIS_k = v' S^-1 v`` (``S`` from the forecast ensemble
  plus the sensor variance) of a consistent filter is chi-square with ``m`` degrees of freedom, so the mean over
  ``n`` steps must sit inside the two-sided 95 % band ``chi2_{0.025, 0.975}(n m) / n`` (Bar-Shalom, Li &
  Kirubarajan 2001, *Estimation with Applications to Tracking and Navigation*, sec. 5.4). With a known truth the
  normalised estimation error squared (NEES) of the augmented state (states and log parameters: what the filter
  estimates) is tested over ``nees_runs`` independent simulated plants (estimation errors, unlike innovations, are
  correlated in time, so one run's time average is not chi-square): :func:`nees_monte_carlo`.
* Anomalies. A reading whose normalised innovation exceeds ``anomaly_z`` (default 4: two-sided p = 6e-5 per reading)
  is flagged and not assimilated (innovation gating); ``persist`` such readings in a row are a level change, not
  an outlier, and are assimilated again (the CUSUM sees every reading, its value clipped at ``anomaly_z``).
* Drift. A two-sided Page CUSUM (Page 1954, *Biometrika* 41:100) of the normalised innovations, reference ``k`` and
  threshold ``h``; its in-control average run length (Siegmund 1985 approximation) is reported, so the false-alarm
  rate is stated. An alarm triggers a **recalibration** through
  :func:`pinneapple_analysis.inverse_problems.workflow.inverse_workflow` on the next ``recalibration_window``
  readings (the slow parameters and the initial states of the window estimated, with identifiability, intervals and
  its own checks); the filter restarts from the recalibrated parameters (their interval as prior spread) and from the states that
  the recalibrated run reaches at the window's end. A recalibration whose fit fails (the model does not fit the
  window at the stated sensor noise, or the parameters are not identifiable) is refused and stated.

**Outputs.** A forecast of the outputs ``horizon`` steps ahead with the 90 % interval (members propagated with their
process noise and sensor noise), scored against the withheld future when the truth is known; what-if runs (the inputs
scaled or replaced) with the interval of each output's peak; a self-contained HTML dashboard.
"""
from __future__ import annotations

import base64
import io
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------------------------- the model


@dataclass
class StateSpaceModel:
    """A discrete-time model for the twin. ``step(x (N, nx), u (nu,), theta (N, np), dt) -> (N, nx)`` and
    ``observe(x (N, nx), theta (N, np)) -> (N, ny)`` are batched over ensemble members."""
    name: str
    states: list[str]
    params: list[str]
    inputs: list[str]
    outputs: list[str]
    step: Callable[..., np.ndarray]
    observe: Callable[..., np.ndarray]
    nominal: dict[str, float]
    x0: list[float]
    process_std: list[float]                          # model error per step and per state
    sensor_std: list[float]                           # per output
    state_range: list[tuple[float, float]] = field(default_factory=list)   # bounds of the window's initial states
    units: dict[str, str] = field(default_factory=dict)
    description: str = ""


def thermal_plant(C: float = 2.0e4, T_amb: float = 20.0, UA: float = 50.0) -> StateSpaceModel:
    """Lumped heated body (a tank, a casting, a cooled housing): ``C dT/dt = Q - UA (T - T_amb)``, ``C`` in J/K,
    ``UA`` in W/K, ``Q`` in W. Exact discretisation for ``Q`` held over the step:
    ``T' = T_inf + (T - T_inf) exp(-UA dt / C)`` with ``T_inf = T_amb + Q / UA``."""

    def step(x, u, theta, dt):
        ua = theta[:, 0]
        t_inf = T_amb + float(u[0]) / ua
        a = np.exp(-ua * dt / C)
        return (t_inf + (x[:, 0] - t_inf) * a)[:, None]

    def observe(x, theta):
        return x[:, :1]

    return StateSpaceModel(
        name="thermal_plant", states=["T"], params=["UA"], inputs=["Q"], outputs=["T"], step=step, observe=observe,
        nominal={"UA": float(UA)}, x0=[T_amb], process_std=[0.005], sensor_std=[0.2],
        state_range=[(T_amb - 50.0, T_amb + 300.0)], units={"T": "degC", "UA": "W/K", "Q": "W"},
        description=f"lumped heated body, C = {C:g} J/K, T_amb = {T_amb:g} degC, exact zero-order-hold step")


MODELS: dict[str, Callable[..., StateSpaceModel]] = {"thermal_plant": thermal_plant}


def resolve_model(model: str | StateSpaceModel | Callable[..., StateSpaceModel],
                  model_params: Mapping[str, Any] | None = None) -> StateSpaceModel:
    """A model by name (:data:`MODELS`), a ``module:function`` returning a :class:`StateSpaceModel`, or one."""
    if isinstance(model, StateSpaceModel):
        return model
    kw = dict(model_params or {})
    if callable(model):
        return model(**kw)
    if model in MODELS:
        return MODELS[model](**kw)
    if ":" in str(model):
        import importlib
        mod, fn = str(model).split(":", 1)
        return getattr(importlib.import_module(mod), fn)(**kw)
    raise ValueError(f"unknown twin model {model!r}: one of {sorted(MODELS)} or 'module:function'")


# ---------------------------------------------------------------------------------------------- data sources


def square_wave(n: int, dt: float, low: float, high: float, period: float) -> np.ndarray:
    t = np.arange(n) * dt
    return np.where(np.floor(t / (period / 2)) % 2 == 0, high, low)


def simulate_plant(model: StateSpaceModel, *, n_steps: int = 400, dt: float = 5.0,
                   inputs: Mapping[str, Sequence[float]] | None = None,
                   drift: Mapping[str, Any] | None = None, spike: Mapping[str, Any] | None = None,
                   horizon: int = 0, plant_params: Mapping[str, float] | None = None, seed: int = 0
                   ) -> dict[str, Any]:
    """A simulated sensor stream of ``model`` with its process and sensor noise; the plant's parameters are
    ``plant_params`` (default: the model's nominal values; a twin is never commissioned exactly). ``drift``: ``{param, at (step),
    to (value), ramp (steps, 0: a step change)}``; ``spike``: ``{at, size (in sensor sigmas)}``. ``horizon`` more
    steps are simulated and withheld (the future the forecast is scored on). Returns the record and the truth."""
    rng = np.random.default_rng(seed)
    n = int(n_steps) + int(horizon)
    if inputs is None:
        inputs = {model.inputs[0]: square_wave(n, dt, 1000.0, 2000.0, 1200.0).tolist()}
    U = np.column_stack([np.resize(np.asarray(inputs[k], float), n) for k in model.inputs])
    pp = {**model.nominal, **dict(plant_params or {})}
    theta = np.tile([pp[p] for p in model.params], (n, 1)).astype(float)
    if drift:
        j = model.params.index(drift["param"])
        at, ramp = int(drift["at"]), int(drift.get("ramp", 0))
        v0, v1 = theta[0, j], float(drift["to"])
        for k in range(at, n):
            theta[k, j] = v1 if ramp <= 0 else v0 + (v1 - v0) * min(1.0, (k - at + 1) / ramp)
    X = np.zeros((n, len(model.states)))
    x = np.asarray(model.x0, float)[None, :]
    qs, rs = np.asarray(model.process_std, float), np.asarray(model.sensor_std, float)
    Y = np.zeros((n, len(model.outputs)))
    for k in range(n):
        x = model.step(x, U[k], theta[k:k + 1], dt) + qs * rng.standard_normal(qs.shape)
        X[k] = x[0]
        Y[k] = model.observe(x, theta[k:k + 1])[0] + rs * rng.standard_normal(rs.shape)
    if spike:
        Y[int(spike["at"]), 0] += float(spike.get("size", 12.0)) * rs[0]
    t = (np.arange(n) + 1) * dt
    m = int(n_steps)
    rec = {"t": t[:m], "dt": dt, **{k: U[:m, i] for i, k in enumerate(model.inputs)},
           **{k: Y[:m, i] for i, k in enumerate(model.outputs)}}
    truth = {"states": X, "params": theta, "inputs": U, "outputs": Y, "drift": dict(drift or {}),
             "spike": dict(spike or {}), "n_steps": m, "horizon": int(horizon)}
    return {"record": rec, "truth": truth}


def read_csv_record(path: str, model: StateSpaceModel) -> dict[str, Any]:
    """CSV replay: columns ``t``, the model's inputs and outputs (a header row)."""
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")
    need = ["t", *model.inputs, *model.outputs]
    missing = [c for c in need if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} (needs {need})")
    rec = {c: np.array([float(r[c]) for r in rows]) for c in need}
    rec["dt"] = float(np.median(np.diff(rec["t"]))) if len(rows) > 1 else 1.0
    return rec


# ---------------------------------------------------------------------------------------------- statistics


def chi2_band(dof: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Two-sided band of the mean of ``n`` independent chi-square(dof) values."""
    from scipy.stats import chi2
    a = (1 - level) / 2
    return float(chi2.ppf(a, dof * n) / n), float(chi2.ppf(1 - a, dof * n) / n)


def cusum_arl0(k: float, h: float) -> float:
    """In-control average run length of a two-sided CUSUM of N(0, 1) values (Siegmund 1985 approximation of each
    one-sided chart, combined as 1/ARL = 1/ARL+ + 1/ARL-)."""
    b = h + 1.166
    one = (math.exp(2 * k * b) - 2 * k * b - 1) / (2 * k * k)
    return one / 2


# ---------------------------------------------------------------------------------------------- the filter


def etkf_analysis(ens: np.ndarray, y: np.ndarray, hx: np.ndarray, r_var: np.ndarray, *, inflation: float = 1.0
                  ) -> np.ndarray:
    """Ensemble transform Kalman filter analysis (Bishop, Etherton & Majumdar 2001; the symmetric square root of
    Hunt, Kostelich & Szunyogh 2007, *Physica D* 230:112): deterministic, so the analysis spread carries no
    perturbed-observation sampling noise. ``ens`` (N, n), ``hx`` (N, m), independent sensor variances ``r_var``."""
    N = ens.shape[0]
    xm, hm = ens.mean(0), hx.mean(0)
    A = (ens - xm) * inflation
    HA = (hx - hm) * inflation
    G = HA / np.sqrt(r_var)[None, :]                                 # (N, m) = HA R^-1/2
    # (N-1) I + G G' has the eigenvectors U of G's thin SVD (eigenvalues N-1+s^2) and N-1 on their complement,
    # so the transform costs O(N m^2) instead of an N x N eigenproblem
    U, sv, _ = np.linalg.svd(G, full_matrices=False)
    lam = (N - 1) + sv ** 2
    d = (y - hm) / np.sqrt(r_var)
    gd = G @ d
    w = (gd - U @ ((sv ** 2 / lam) * (U.T @ gd))) / (N - 1)          # Pt G R^-1/2 (y - hm)
    def transform(Z):                                                # sqrt((N-1) Pt) @ Z
        return Z + U @ ((np.sqrt((N - 1) / lam) - 1)[:, None] * (U.T @ Z))
    return xm + transform(A) + w @ A


class TwinFilter:
    """Stochastic EnKF of ``[states, log(params)]``. One :meth:`step` per reading: forecast, innovation statistics,
    gating, analysis."""

    def __init__(self, model: StateSpaceModel, *, members: int = 200, param_walk: float = 2e-4,
                 param_spread: float = 0.1, x0: Sequence[float] | None = None, x0_std: Sequence[float] | None = None,
                 inflation: float = 1.0, scheme: str = "etkf", seed: int = 0):
        self.m = model
        self.scheme = scheme
        self.N = int(members)
        self.nx, self.np_ = len(model.states), len(model.params)
        self.rng = np.random.default_rng(seed)
        self.walk = float(param_walk)
        self.inflation = float(inflation)
        x0 = np.asarray(model.x0 if x0 is None else x0, float)
        sx = np.asarray(x0_std if x0_std is not None else np.maximum(np.asarray(model.sensor_std[:1] * self.nx), 1e-9),
                        float)
        L0 = np.log([model.nominal[p] for p in model.params])
        self.E = np.column_stack([x0 + sx * self.rng.standard_normal((self.N, self.nx)),
                                  L0 + param_spread * self.rng.standard_normal((self.N, self.np_))])
        self.q = np.asarray(model.process_std, float)
        self.r = np.asarray(model.sensor_std, float) ** 2

    @property
    def theta(self) -> np.ndarray:
        return np.exp(self.E[:, self.nx:])

    def reset(self, values: Sequence[float], std_log: Sequence[float], x: Sequence[float] | None = None,
              x_std: Sequence[float] | None = None) -> None:
        """Restart the members at parameters ``values`` (log-normal spread ``std_log``) and, given, states ``x``."""
        self.E[:, self.nx:] = np.log(np.asarray(values, float)) + np.asarray(std_log, float) * \
            self.rng.standard_normal((self.N, self.np_))
        if x is not None:
            self.E[:, :self.nx] = np.asarray(x, float) + np.asarray(x_std, float) * \
                self.rng.standard_normal((self.N, self.nx))

    def forecast(self, u: np.ndarray, dt: float) -> None:
        X = self.m.step(self.E[:, :self.nx], u, self.theta, dt) + self.q * self.rng.standard_normal((self.N, self.nx))
        L = self.E[:, self.nx:] + self.walk * self.rng.standard_normal((self.N, self.np_))
        self.E = np.column_stack([X, L])

    def innovation(self, y: np.ndarray) -> dict[str, Any]:
        hx = self.m.observe(self.E[:, :self.nx], self.theta)
        mu = hx.mean(0)
        S = np.atleast_2d(np.cov(hx.T, ddof=1)) + np.diag(self.r)
        v = y - mu
        nis = float(v @ np.linalg.solve(S, v))
        z = v / np.sqrt(np.diag(S))
        return {"hx": hx, "v": v, "S": S, "nis": nis, "z": z, "mean": mu, "std": np.sqrt(np.diag(S))}

    def analyse(self, y: np.ndarray, hx: np.ndarray) -> None:
        if self.scheme == "stochastic":
            from pinneapple_analysis.state_estimation.kalman import enkf_analysis
            self.E = enkf_analysis(self.E, y, hx, self.r, inflation=self.inflation, rng=self.rng)
            return
        self.E = etkf_analysis(self.E, y, hx, self.r, inflation=self.inflation)

    def summary(self) -> dict[str, np.ndarray]:
        P = np.atleast_2d(np.cov(self.E.T, ddof=1))
        mean = self.E.mean(0)
        th = self.theta
        return {"x": mean[:self.nx], "Px": P[:self.nx, :self.nx], "theta": th.mean(0),
                "theta_lo": np.quantile(th, 0.05, axis=0), "theta_hi": np.quantile(th, 0.95, axis=0)}


def nees_monte_carlo(model: StateSpaceModel, plant_params: Mapping[str, float], U: np.ndarray, dt: float, *,
                     runs: int = 20, members: int = 200, param_walk: float = 2e-4, burn: int = 20, seed: int = 0
                     ) -> dict[str, Any]:
    """NEES consistency over independent plants (Bar-Shalom et al. 2001, sec. 5.4.2): ``runs`` plants with fresh
    noise and the inputs ``U``; at each step the NEES averaged over the runs is chi-square(runs * d) / runs. Returns
    the fraction of the steps inside the two-sided 95 % band (about 95 % for a consistent filter)."""
    d = len(model.states) + len(model.params)
    acc = np.zeros(len(U))
    for r in range(runs):
        sim = simulate_plant(model, n_steps=len(U), dt=dt, inputs={k: U[:, i] for i, k in enumerate(model.inputs)},
                             plant_params=plant_params, seed=seed + r)
        rec, tr = sim["record"], sim["truth"]
        Y = np.column_stack([rec[k] for k in model.outputs])
        f = TwinFilter(model, members=members, param_walk=param_walk, x0=Y[0, :len(model.states)], seed=seed + r)
        for k in range(len(U)):
            f.forecast(U[k], dt)
            inn = f.innovation(Y[k])
            f.analyse(Y[k], inn["hx"])
            e = np.concatenate([tr["states"][k], np.log(tr["params"][k])]) - f.E.mean(0)
            acc[k] += float(e @ np.linalg.solve(np.atleast_2d(np.cov(f.E.T, ddof=1)), e))
    avg = acc[burn:] / runs
    lo, hi = chi2_band(d, runs)
    return {"runs": runs, "steps": len(avg), "dof": d, "band": [lo, hi], "mean": float(avg.mean()) if len(avg) else
            float("nan"), "inside": float(np.mean((avg >= lo) & (avg <= hi))) if len(avg) else 0.0,
            "per_step": avg.tolist()}


# ---------------------------------------------------------------------------------------------- recalibration


def _window_tool(model: StateSpaceModel, U: np.ndarray, dt: float, output: int = 0):
    """A keyword tool for the inverse workflow: the output at window step ``k`` of a deterministic run from initial
    states ``x0_<state>`` with parameters ``<param>`` (the trajectory of one parameter set is cached)."""
    cache: dict[tuple, np.ndarray] = {}

    def tool(k: float, **kw):
        th = np.array([[float(kw[p]) for p in model.params]])
        x = np.array([[float(kw[f"x0_{s}"]) for s in model.states]])
        key = tuple(np.round(np.concatenate([th[0], x[0]]), 12))
        traj = cache.get(key)
        if traj is None:
            out = []
            for j in range(len(U)):
                x = model.step(x, U[j], th, dt)
                out.append(model.observe(x, th)[0, output])
            traj = np.array(out)
            if len(cache) > 4096:
                cache.clear()
            cache[key] = traj
        return {model.outputs[output]: float(traj[int(round(k))])}
    return tool


def recalibrate(model: StateSpaceModel, U: np.ndarray, Y: np.ndarray, dt: float, *, x_start: Sequence[float],
                theta_start: Sequence[float], bootstrap: int = 30, n_recovery: int = 8, seed: int = 0
                ) -> dict[str, Any]:
    """The inverse workflow on one window of readings: the slow parameters and the window's initial states."""
    from pinneapple_analysis.inverse_problems.workflow import inverse_workflow
    tool = _window_tool(model, U, dt)
    pars: dict[str, dict[str, float]] = {}
    for p, v in zip(model.params, theta_start, strict=True):
        pars[p] = {"lower": float(v) / 5, "upper": float(v) * 5, "initial": float(v), "unit": model.units.get(p, "")}
    for i, s in enumerate(model.states):
        lo, hi = model.state_range[i] if i < len(model.state_range) else (-1e9, 1e9)
        x = float(np.clip(x_start[i], lo, hi))
        pars[f"x0_{s}"] = {"lower": lo, "upper": hi, "initial": x, "unit": model.units.get(s, "")}
    meas = {"inputs": {"k": np.arange(len(Y)).astype(float).tolist()}, "values": np.asarray(Y, float)[:, 0].tolist(),
            "sigma": float(model.sensor_std[0])}
    return inverse_workflow(tool=tool, measurements=meas, output=model.outputs[0], parameters=pars,
                            holdout_fraction=0.2, bootstrap=bootstrap, n_recovery=n_recovery, seed=seed)


# ---------------------------------------------------------------------------------------------- forecast, what-if


def _propagate(filt: TwinFilter, U: np.ndarray, dt: float, rng: np.random.Generator, noise: bool = True
               ) -> np.ndarray:
    """Members propagated over the inputs ``U`` (H, nu): outputs (H, N, ny) with sensor noise."""
    X, th = filt.E[:, :filt.nx].copy(), filt.theta.copy()
    out = []
    for u in U:
        X = filt.m.step(X, u, th, dt) + (filt.q * rng.standard_normal(X.shape) if noise else 0.0)
        y = filt.m.observe(X, th)
        out.append(y + (np.sqrt(filt.r) * rng.standard_normal(y.shape) if noise else 0.0))
    return np.array(out)


def _what_if_inputs(model: StateSpaceModel, base: np.ndarray, scenario: Mapping[str, Any]) -> np.ndarray:
    U = base.copy()
    for k, f in dict(scenario.get("scale") or {}).items():
        U[:, model.inputs.index(k)] *= float(f)
    for k, v in dict(scenario.get("set") or {}).items():
        U[:, model.inputs.index(k)] = float(v)
    return U


# ---------------------------------------------------------------------------------------------- the workflow


def digital_twin_workflow(*, model: str | StateSpaceModel = "thermal_plant", model_params: Mapping[str, Any] | None
                          = None, source: str = "simulated", data: Mapping[str, Any] | None = None,
                          csv_path: str | None = None, n_steps: int = 400, dt: float = 5.0,
                          drift: Mapping[str, Any] | None = None, spike: Mapping[str, Any] | None = None,
                          plant_params: Mapping[str, float] | None = None, members: int = 200, param_walk: float = 2e-4, anomaly_z: float = 4.0, persist: int = 3, nees_runs: int = 20,
                          cusum_k: float = 0.5, cusum_h: float = 8.0, max_detection_delay: int = 20,
                          recalibration_window: int = 60, horizon: int = 120,
                          what_if: Sequence[Mapping[str, Any]] | None = None, limits: Mapping[str, float] | None = None,
                          bootstrap: int = 30, n_recovery: int = 8, out_dir: str | None = None, seed: int = 0
                          ) -> dict[str, Any]:
    """Run the twin over the record; returns the traces, the events, the recalibrations, the forecast, the what-if
    runs and the checks (``pass`` / ``fail`` / ``inconclusive``; the truth-scored ones only on a simulated plant)."""
    m = resolve_model(model, model_params)
    truth = None
    plant: dict[str, float] = {}
    if source == "simulated":
        drift = dict(drift) if drift is not None else {"param": m.params[0], "at": int(0.5 * n_steps),
                                                       "to": 0.7 * m.nominal[m.params[0]], "ramp": 0}
        spike = dict(spike) if spike is not None else {"at": int(0.3 * n_steps), "size": 12.0}
        plant = dict(plant_params) if plant_params is not None else {p: 1.05 * v for p, v in m.nominal.items()}
        sim = simulate_plant(m, n_steps=n_steps, dt=dt, drift=drift or None, spike=spike or None,
                             horizon=horizon, plant_params=plant, seed=seed)
        rec, truth = sim["record"], sim["truth"]
    elif source == "csv":
        rec = read_csv_record(str(csv_path), m)
    elif source == "arrays":
        rec = {k: np.asarray(v, float) for k, v in dict(data or {}).items() if k != "dt"}
        rec["dt"] = float((data or {}).get("dt") or (np.median(np.diff(rec["t"])) if len(rec["t"]) > 1 else dt))
    else:
        raise ValueError("source: simulated, csv or arrays")
    dt = float(rec["dt"])
    U = np.column_stack([np.asarray(rec[k], float) for k in m.inputs])
    Y = np.column_stack([np.asarray(rec[k], float) for k in m.outputs])
    n, ny = len(Y), Y.shape[1]
    filt = TwinFilter(m, members=members, param_walk=param_walk, x0=Y[0, :len(m.states)] if ny >= len(m.states)
                      else None, seed=seed)
    rng = np.random.default_rng(seed + 7)
    trace: dict[str, list] = {k: [] for k in ("nis", "z", "y_mean", "y_std", "x", "theta", "theta_lo", "theta_hi",
                                              "cusum_pos", "cusum_neg", "nees", "assimilated")}
    events: list[dict[str, Any]] = []
    recals: list[dict[str, Any]] = []
    gp = gn = 0.0
    run = 0
    pending: int | None = None               # step of the alarm whose recalibration window is being collected
    phase = np.zeros(n, int)                 # 0 before the first alarm, 1 while collecting, 2 after recalibration
    current = 0
    for k in range(n):
        filt.forecast(U[k], dt)
        inn = filt.innovation(Y[k])
        zmax = float(np.max(np.abs(inn["z"])))
        run = run + 1 if zmax > anomaly_z else 0
        # an isolated outlier; a longer run is a level change, and no reading is gated while an alarm's
        # recalibration window is collected (the level is known to be moving)
        anomalous = 0 < run < persist and pending is None
        if anomalous:
            events.append({"step": k, "t": float(rec["t"][k]), "kind": "anomaly",
                           "detail": f"normalised innovation {zmax:.3g} > {anomaly_z:g}: reading flagged and not "
                                     "assimilated"})
        else:
            filt.analyse(Y[k], inn["hx"])
        zc = float(np.clip(inn["z"][0], -anomaly_z, anomaly_z))   # one outlier moves the CUSUM by < anomaly_z - k
        gp, gn = max(0.0, gp + zc - cusum_k), max(0.0, gn - zc - cusum_k)
        if (gp > cusum_h or gn > cusum_h) and pending is None:
            events.append({"step": k, "t": float(rec["t"][k]), "kind": "drift",
                           "detail": f"CUSUM {'up' if gp > cusum_h else 'down'} {max(gp, gn):.3g} > {cusum_h:g}"})
            pending, current = k + 1, 1
            gp = gn = 0.0
        s = filt.summary()
        if truth is not None:                # joint NEES of what the filter estimates: states and log parameters
            e = np.concatenate([truth["states"][k], np.log(truth["params"][k])]) - filt.E.mean(0)
            trace["nees"].append(float(e @ np.linalg.solve(np.atleast_2d(np.cov(filt.E.T, ddof=1)), e)))
        trace["nis"].append(inn["nis"])
        trace["z"].append(float(inn["z"][0]))
        trace["y_mean"].append(float(inn["mean"][0]))
        trace["y_std"].append(float(inn["std"][0]))
        trace["x"].append(s["x"].tolist())
        trace["theta"].append(s["theta"].tolist())
        trace["theta_lo"].append(s["theta_lo"].tolist())
        trace["theta_hi"].append(s["theta_hi"].tolist())
        trace["cusum_pos"].append(gp)
        trace["cusum_neg"].append(gn)
        trace["assimilated"].append(not anomalous)
        phase[k] = current
        if pending is not None and k + 1 - pending >= recalibration_window:
            a, b = pending, k + 1
            x_start = np.asarray(trace["x"][a - 1], float)
            r = recalibrate(m, U[a:b], Y[a:b], dt, x_start=x_start, theta_start=s["theta"],
                            bootstrap=bootstrap, n_recovery=n_recovery, seed=seed)
            item = {"alarm_step": a - 1, "window": [a, b], "result": r}
            trusted = r.get("estimated") and all(c["status"] != "fail" for c in r["checks"]
                                                  if c["code"] in ("identify.structural", "identify.fits_data"))
            if not trusted:
                events.append({"step": k, "t": float(rec["t"][k]), "kind": "recalibration_refused",
                               "detail": "inverse workflow: " + (", ".join(c["code"] for c in r["checks"]
                                                                          if c["status"] == "fail") or "not estimated")
                                         + " failing: the model does not fit the window, kept the parameters"})
            if trusted:
                est = [r["estimate"][p] for p in m.params]
                sd = []
                for p in m.params:
                    lo, hi = r["intervals"].get(p) or (est[m.params.index(p)] * 0.9, est[m.params.index(p)] * 1.1)
                    sd.append(max(math.log(max(hi, 1e-300) / max(lo, 1e-300)) / (2 * 1.96), 1e-4))
                # the states at the window's end from the recalibrated run (the filter's own states lag a drift)
                xe = np.array([[r["estimate"][f"x0_{st}"] for st in m.states]])
                th = np.array([est])
                for j in range(a, b):
                    xe = m.step(xe, U[j], th, dt)
                xs = [r["intervals"].get(f"x0_{st}") for st in m.states]
                xsd = [max((hi - lo) / (2 * 1.96), 1e-9) if (lo, hi) != (None, None) else float(m.sensor_std[0])
                       for lo, hi in (x or (None, None) for x in xs)]
                filt.reset(est, sd, xe[0], xsd)
                item["applied"] = dict(zip(m.params, est, strict=True))
                events.append({"step": k, "t": float(rec["t"][k]), "kind": "recalibration",
                               "detail": "inverse workflow: " + ", ".join(f"{p} = {v:.5g}" for p, v in
                                                                          item["applied"].items())})
            recals.append(item)
            pending, current = None, 2
            gp = gn = 0.0                    # the chart restarts with the recalibrated model
    trace_np = {k: np.asarray(v) for k, v in trace.items()}

    # ------------------------------------------------------------- forecast and what-if
    if truth is not None and truth["horizon"]:
        U_f = truth["inputs"][n:n + horizon]
    else:
        U_f = np.repeat(U[-1:], horizon, 0)
    Yf = _propagate(filt, U_f, dt, rng) if horizon else np.zeros((0, members, ny))
    fc = {"t": (rec["t"][-1] + dt * (np.arange(len(U_f)) + 1)).tolist() if horizon else [],
          "mean": Yf.mean(1)[:, 0].tolist() if horizon else [],
          "lo": np.quantile(Yf, 0.05, axis=1)[:, 0].tolist() if horizon else [],
          "hi": np.quantile(Yf, 0.95, axis=1)[:, 0].tolist() if horizon else []}
    if truth is not None and horizon:
        yt = truth["outputs"][n:n + horizon, 0]
        fc["observed"] = yt.tolist()
        fc["coverage_90"] = float(np.mean((yt >= np.array(fc["lo"])) & (yt <= np.array(fc["hi"]))))
        fc["rmse"] = float(np.sqrt(np.mean((yt - np.array(fc["mean"])) ** 2)))
    scenarios = list(what_if) if what_if is not None else (
        [{"name": f"{m.inputs[0]} +20 %", "scale": {m.inputs[0]: 1.2}},
         {"name": f"{m.inputs[0]} -20 %", "scale": {m.inputs[0]: 0.8}}] if horizon else [])
    whatifs = []
    base_U = U_f if horizon else np.repeat(U[-1:], 60, 0)
    for sc in [{"name": "as forecast"}] + scenarios:
        Yw = _propagate(filt, _what_if_inputs(m, base_U, sc), dt, np.random.default_rng(seed + 11), noise=True)
        peak = Yw[:, :, 0].max(0)
        row = {"name": sc.get("name", "scenario"), "output": m.outputs[0], "peak_mean": float(peak.mean()),
               "peak_lo": float(np.quantile(peak, 0.05)), "peak_hi": float(np.quantile(peak, 0.95))}
        lim = (limits or {}).get(m.outputs[0])
        if lim is not None:
            row["limit"] = float(lim)
            row["p_exceed"] = float(np.mean(peak > float(lim)))
        whatifs.append(row)

    # ------------------------------------------------------------- checks
    checks: list[dict[str, Any]] = []

    def chk(code, ok, detail, value=None):
        checks.append({"code": code, "status": "inconclusive" if ok is None else ("pass" if ok else "fail"),
                       "detail": detail, "value": value})

    burn = min(20, n // 10)
    first_alarm = next((e["step"] for e in events if e["kind"] == "drift"), None)
    assim = trace_np["assimilated"]

    def nis_window(a, b, code, label, sel=None):
        sel = np.arange(a, b) if sel is None else sel
        sel = sel[assim[sel]] if len(sel) else sel
        if len(sel) < 10:
            chk(code, None, f"{label}: {len(sel)} assimilated readings (< 10)")
            return None
        mean = float(np.mean(trace_np["nis"][sel]))
        lo, hi = chi2_band(ny, len(sel))
        chk(code, lo <= mean <= hi, f"{label}: mean NIS {mean:.3g} over {len(sel)} readings, 95 % chi-square band "
                                    f"[{lo:.3g}, {hi:.3g}] ({ny} dof)", mean)
        return mean
    end_pre = first_alarm if first_alarm is not None else n
    if truth is not None and truth["drift"]:
        end_pre = min(end_pre, int(truth["drift"]["at"]))
    if end_pre - burn >= 10:
        nis_pre = nis_window(burn, end_pre, "twin.nis_consistent", "before the drift" if truth is not None and
                             truth["drift"] else "before the first alarm")
    else:                                    # an alarm at once: every reading outside the recalibration windows
        nis_pre = nis_window(0, 0, "twin.nis_consistent", f"first alarm at step {first_alarm}; readings outside "
                             "the recalibration windows", sel=np.nonzero(phase != 1)[0][burn:])
    after = [r["window"][1] for r in recals if r.get("applied")]
    nis_post = nis_window(after[-1], n, "twin.nis_after_recalibration", "after the recalibration") if after else None
    if truth is not None:
        mc = nees_monte_carlo(m, plant, U[:end_pre], dt, runs=nees_runs, members=members, param_walk=param_walk,
                              burn=burn, seed=seed + 101)
        chk("twin.nees_consistent", mc["inside"] >= 0.85,                # 10 points of margin: steps correlate
            f"NEES of the states and log parameters, {mc['runs']} independent plants (truth known), averaged over the "
            f"runs at each of {mc['steps']} steps: {mc['inside']:.0%} of the steps inside the 95 % band "
            f"[{mc['band'][0]:.3g}, {mc['band'][1]:.3g}] ({mc['dof']} dof; >= 85 %); mean {mc['mean']:.3g}", mc["inside"])
        d = truth["drift"]
        if d:
            at = int(d["at"])
            false = [e["step"] for e in events if e["kind"] == "drift" and e["step"] < at]
            hit = [e["step"] for e in events if e["kind"] == "drift" and e["step"] >= at]
            delay = (hit[0] - at) if hit else None
            chk("twin.drift_detected", bool(hit) and not false and delay <= max_detection_delay,
                (f"drift of {d['param']} at step {at} detected at step {hit[0]} (delay {delay} steps, "
                 f"<= {max_detection_delay})" if hit else "drift not detected")
                + (f"; false alarms at {false}" if false else "") +
                f"; in-control ARL of the CUSUM {cusum_arl0(cusum_k, cusum_h):.3g} steps", delay)
            ok_rec = [r for r in recals if r.get("applied")]
            if ok_rec:
                r = ok_rec[-1]
                j = m.params.index(d["param"])
                true_v = float(truth["params"][r["window"][1] - 1, j])
                est = r["applied"][d["param"]]
                lo, hi = r["result"]["intervals"][d["param"]] or (np.nan, np.nan)
                rel = abs(est - true_v) / abs(true_v)
                bad = [c["code"] for c in r["result"]["checks"] if c["status"] == "fail"]
                chk("twin.recalibrated", bool(lo <= true_v <= hi) and not bad,
                    f"inverse workflow on steps {r['window'][0]}-{r['window'][1] - 1}: {d['param']} = {est:.5g} "
                    f"(95 % [{lo:.5g}, {hi:.5g}]) against the true {true_v:.5g} ({rel:.2%})"
                    + (f"; its checks failing: {bad}" if bad else "; its checks pass"), rel)
            else:
                chk("twin.recalibrated", False, "no recalibration was applied"
                    + (f" (identifiability: {recals[-1]['result'].get('identifiability')})" if recals else ""))
        sp = truth["spike"]
        if sp:
            flagged = [e["step"] for e in events if e["kind"] == "anomaly"]
            others = [s for s in flagged if s != int(sp["at"])]
            chk("twin.anomaly_flagged", int(sp["at"]) in flagged,
                f"injected spike of {sp.get('size', 12.0):g} sigma at step {sp['at']}: "
                + ("flagged" if int(sp["at"]) in flagged else "missed")
                + f"; {len(others)} other readings flagged at |z| > {anomaly_z:g}", len(others))
        if horizon:
            c = fc["coverage_90"]
            tol = 2 * math.sqrt(0.9 * 0.1 / horizon) + 0.05
            chk("twin.forecast_coverage", abs(c - 0.9) <= tol, f"{horizon}-step forecast: {c:.0%} of the withheld "
                f"readings inside the 90 % interval (|coverage - 90 %| <= {tol:.0%}, readings are correlated); "
                f"RMSE {fc['rmse']:.3g}", c)
    else:
        chk("twin.truth", None, "no truth for a replayed or live record: drift recovery, NEES and forecast coverage "
                                "are scored only on a simulated plant")
    failed = [c["code"] for c in checks if c["status"] == "fail"]
    out = {"model": {"name": m.name, "states": m.states, "params": m.params, "inputs": m.inputs,
                     "outputs": m.outputs, "nominal": m.nominal, "description": m.description},
           "source": source, "n_steps": n, "dt": dt, "members": members,
           "trace": {k: v.tolist() for k, v in trace_np.items()}, "t": np.asarray(rec["t"]).tolist(),
           "measured": Y[:, 0].tolist(), "inputs": U[:, 0].tolist(), "events": events,
           "recalibrations": [{"alarm_step": r["alarm_step"], "window": r["window"], "applied": r.get("applied"),
                               "estimate": r["result"].get("estimate"), "intervals": r["result"].get("intervals"),
                               "checks": r["result"].get("checks")} for r in recals],
           "forecast": fc, "what_if": whatifs, "checks": checks, "failed": failed,
           "verdict": "fail" if failed else "pass",
           "cusum": {"k": cusum_k, "h": cusum_h, "arl0": cusum_arl0(cusum_k, cusum_h)},
           "nis": {"before": nis_pre, "after": nis_post}}
    if truth is not None:
        out["truth"] = {"params": truth["params"][:n].tolist(), "drift": truth["drift"], "spike": truth["spike"]}
    final = filt.summary()
    out["final"] = {"theta": dict(zip(m.params, final["theta"].tolist(), strict=True)),
                    "theta_90": {p: [float(final["theta_lo"][i]), float(final["theta_hi"][i])]
                                 for i, p in enumerate(m.params)}}
    out["summary"] = (f"{m.name}: {n} readings, {sum(e['kind'] == 'anomaly' for e in events)} anomalies, "
                      f"{sum(e['kind'] == 'drift' for e in events)} drift alarms, {len(after)} recalibrations; "
                      + (f"fail: {', '.join(failed)}" if failed else "every check passes"))
    if out_dir:
        out["dashboard"] = write_dashboard(out, out_dir)
    return out


# ---------------------------------------------------------------------------------------------- dashboard


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def write_dashboard(r: Mapping[str, Any], out_dir: str) -> str:
    """A self-contained HTML page: measured output with the twin's band, the parameter with its 90 % band (and the
    truth), NIS with its band, the CUSUM with its threshold, the forecast and the what-if table, the checks."""
    import html
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    t = np.asarray(r["t"])
    tr = r["trace"]
    mm = r["model"]
    fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    mu, sd = np.asarray(tr["y_mean"]), np.asarray(tr["y_std"])
    ax[0].plot(t, r["measured"], ".", ms=2, color="0.4", label="measured")
    ax[0].fill_between(t, mu - 1.645 * sd, mu + 1.645 * sd, alpha=0.3, label="twin 90 % (forecast)")
    fc = r["forecast"]
    if fc["t"]:
        ax[0].fill_between(fc["t"], fc["lo"], fc["hi"], alpha=0.3, color="C2", label="forecast 90 %")
        if "observed" in fc:
            ax[0].plot(fc["t"], fc["observed"], ".", ms=2, color="C3", label="withheld")
    ax[0].set_ylabel(mm["outputs"][0])
    ax[0].legend(fontsize=7)
    th, lo, hi = (np.asarray(tr[k])[:, 0] for k in ("theta", "theta_lo", "theta_hi"))
    ax[1].plot(t, th, label=f"{mm['params'][0]} (twin)")
    ax[1].fill_between(t, lo, hi, alpha=0.3)
    if "truth" in r:
        ax[1].plot(t, np.asarray(r["truth"]["params"])[:, 0], "k--", lw=1, label="truth")
    ax[1].legend(fontsize=7)
    ax[2].semilogy(t, np.maximum(tr["nis"], 1e-6), ".", ms=2)
    ax[2].axhline(1.0, color="k", lw=0.8)
    ax[2].set_ylabel("NIS")
    ax[3].plot(t, tr["cusum_pos"], label="CUSUM +")
    ax[3].plot(t, tr["cusum_neg"], label="CUSUM -")
    ax[3].axhline(r["cusum"]["h"], color="r", lw=0.8)
    ax[3].legend(fontsize=7)
    for e in r["events"]:
        for a in ax:
            a.axvline(e["t"], color={"anomaly": "orange", "drift": "red", "recalibration": "green"}.get(e["kind"], "k"),
                      lw=0.6, alpha=0.6)
    ax[3].set_xlabel("t")
    img = _png(fig)
    plt.close(fig)
    rows = "".join(f"<tr><td>{html.escape(c['code'])}</td><td class='{c['status']}'>{c['status']}</td>"
                   f"<td>{html.escape(str(c['detail']))}</td></tr>" for c in r["checks"])
    ev = "".join(f"<tr><td>{e['step']}</td><td>{e['t']:.6g}</td><td>{e['kind']}</td>"
                 f"<td>{html.escape(e['detail'])}</td></tr>" for e in r["events"])
    wi = "".join(f"<tr><td>{html.escape(w['name'])}</td><td>{w['peak_mean']:.4g}</td><td>[{w['peak_lo']:.4g}, "
                 f"{w['peak_hi']:.4g}]</td><td>{w.get('p_exceed', '')}</td></tr>" for w in r["what_if"])
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Digital twin</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;margin:16px;max-width:980px;background:#fff;color:#111}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #ccc;padding:4px}}
.pass{{color:#0a0}}.fail{{color:#c00}}.inconclusive{{color:#a60}}img{{max-width:100%}}</style></head><body>
<h1>Digital twin: {html.escape(mm['name'])}</h1><p>{html.escape(r['summary'])}</p>
<img src="data:image/png;base64,{img}" alt="twin traces">
<h2>Checks</h2><table><tr><th>check</th><th>status</th><th>detail</th></tr>{rows}</table>
<h2>Events</h2><table><tr><th>step</th><th>t</th><th>kind</th><th>detail</th></tr>{ev}</table>
<h2>What-if (peak of {html.escape(mm['outputs'][0])} over the horizon)</h2>
<table><tr><th>scenario</th><th>mean</th><th>90 %</th><th>P(exceed)</th></tr>{wi}</table></body></html>"""
    path = os.path.join(out_dir, "dashboard.html")
    with open(path, "w") as f:
        f.write(page)
    return path
