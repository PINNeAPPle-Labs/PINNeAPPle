"""A digital twin built automatically from an identified equation system.

:class:`PDETwinModel` turns the system into a twin model for :class:`pinneapple_systems.operations.twin.TwinRuntime`
(the library's ensemble Kalman filter): the state is the field on a coarse grid (the identified PDE as a fast,
reduced-resolution model) plus the coefficients to calibrate, carried as random-walk states, plus one state per
sensor: the field linearly interpolated at the sensor's position (so a sensor between grid nodes is observed where it
is, not at the nearest node). The runtime then gives state estimation, calibration of the coefficients, alerts when a coefficient leaves its
expected range (an anomaly: the asset no longer behaves like the model) and consistency tests of the innovations.

    model = PDETwinModel(system, bounds, n=21, dt=0.01, estimate={"alpha": 0.1})
    plant = PDETwinModel(system, bounds, n=81, dt=0.01)                 # a finer simulator as the "real" asset
    readings = simulate_sensors(plant, sensors=[0.25, 0.5, 0.75], t_end=2.0, every=5, noise=0.002,
                                coefficients={"alpha": 0.1}, fault={"t": 1.0, "alpha": 0.15})
    twin, result = run_twin(model, readings, limits={"alpha": (0.07, 0.13)})
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


class PDETwinModel:
    """The identified system as a twin model (``state_names``, ``input_names``, ``dt``, ``step``, ``prior``...).
    ``estimate``: coefficient -> prior value; those coefficients become states. First order in time only."""

    kind = "pde_grid"

    def __init__(self, system, bounds: dict[str, tuple[float, float]], *, n: int = 21, dt: float = 0.01,
                 estimate: dict[str, float] | None = None, coefficients: dict[str, float] | None = None):
        from pinneapple_simulation.numerical_solvers.symbolic_fd import (
            FDError,
            compile_system,
            initial_state,
        )
        if not system.time_dependent:
            raise FDError("a twin needs a time-dependent model")
        self.system, self.bounds, self.dt = system, dict(bounds), float(dt)
        self.estimate = dict(estimate or {})
        self.coefficients = {**system.coefficients, **(coefficients or {})}
        keep = tuple(sorted(set(self.estimate) | set(self.coefficients) & set(system.used_coefficients())))
        self.op = compile_system(system, bounds, n=n, coefficients=self.coefficients, keep=keep)
        if any(o != 1 for o in self.op.order.values()):
            raise FDError("the twin model supports first order in time (u_t = ...)")
        self.keep = keep
        g = self.op.g
        self.grid_shape = g.shape
        self.axes = dict(g.axes)
        self.fields = list(self.op.fields)
        self.n_nodes = int(np.prod(g.shape))
        self.node_names = [f"{f}[{i}]" for f in self.fields for i in range(self.n_nodes)]
        self.state_names = self.node_names + list(self.estimate)
        self.input_names: list[str] = []
        self.bounds_states: dict[str, list[float]] = {}
        self.bounds = {}
        self.validation: dict[str, object] = {}
        self.t = float(bounds[system.ctx.time][0])
        init = initial_state(self.op, self.t)
        U0 = {f: init[(f, False)] for f in self.fields}
        env = self.op.environment(U0, self.t, self._coef_env(None))
        for f, sl, tgt in self.op.face_targets(U0, env):
            U0[f][sl] = tgt
        self.x0 = np.concatenate([U0[f].ravel() for f in self.fields] + [np.array(list(self.estimate.values()))])
        self.sensor_names: list[str] = []
        self.W = np.zeros((0, self.n_nodes))
        self._nsub = self._substeps()

    def set_sensors(self, sensors: list) -> None:
        """Add one state per sensor (``sensor<i>``), the first field interpolated at its position."""
        self.W = self.sensor_weights(sensors)
        self.sensor_names = [f"sensor{i}" for i in range(len(self.W))]
        npar = len(self.estimate)
        self.state_names = self.node_names + list(self.estimate) + self.sensor_names
        base = self.x0[:len(self.node_names) + npar]
        self.x0 = np.concatenate([base, self.W @ base[:self.n_nodes]])

    def sensor_weights(self, sensors: list) -> np.ndarray:
        """Multilinear interpolation weights ``(n_sensors, n_nodes)`` of the grid at the sensor positions."""
        import itertools
        W = np.zeros((len(sensors), self.n_nodes))
        names = list(self.axes)
        for k, pos in enumerate(sensors):
            pos = np.atleast_1d(np.asarray(pos, float))
            if len(pos) != len(names):
                raise ValueError(f"sensor {k}: {len(pos)} coordinate(s) for a {len(names)}-D grid")
            per_axis = []
            for c, p in zip(names, pos, strict=True):
                ax = np.asarray(self.axes[c], float)
                if not ax[0] - 1e-12 <= p <= ax[-1] + 1e-12:
                    raise ValueError(f"sensor {k}: {c} = {p:g} is outside [{ax[0]:g}, {ax[-1]:g}]")
                i = int(np.clip(np.searchsorted(ax, p) - 1, 0, len(ax) - 2))
                w = (p - ax[i]) / (ax[i + 1] - ax[i])
                per_axis.append([(i, 1.0 - w), (i + 1, w)])
            for combo in itertools.product(*per_axis):
                idx = tuple(i for i, _ in combo)
                W[k, np.ravel_multi_index(idx, self.grid_shape)] += float(np.prod([w for _, w in combo]))
        return W

    # -- the model interface of TwinRuntime ---------------------------------------------------------------------
    def prior(self) -> dict[str, float]:
        return dict(zip(self.state_names, self.x0.tolist(), strict=True))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "pdes": [e.text for e in self.system.pdes], "grid": list(self.grid_shape),
                "dt": self.dt, "estimate": self.estimate, "coefficients": self.coefficients}

    def _coef_env(self, P: np.ndarray | None, batch: int = 1) -> dict[str, Any]:
        """Coefficient values for the compiled expressions: estimated ones from the state, the rest fixed."""
        nd = len(self.grid_shape)
        env = {k: float(v) for k, v in self.coefficients.items() if k in self.keep and k not in self.estimate}
        for j, k in enumerate(self.estimate):
            if P is None:
                env[k] = float(self.estimate[k])
            else:
                env[k] = P[:, j].reshape((-1,) + (1,) * nd)
        return env

    def _rhs(self, U: np.ndarray, t: float, P: np.ndarray | None) -> np.ndarray:
        """du/dt for a batch of fields ``U`` (B, n_fields, *grid); faces held at their conditions."""
        Ud = {f: U[:, i] for i, f in enumerate(self.fields)}
        env = self.op.environment(Ud, t, self._coef_env(P, len(U)))
        out = np.empty_like(U)
        for i, f in enumerate(self.fields):
            out[:, i] = np.broadcast_to(self.op.compiled[f](env), U[:, i].shape)
        for f, sl, _tgt in self.op.face_targets(Ud, env):
            out[(slice(None), self.fields.index(f)) + sl[1:]] = 0.0
        return out

    def _faces(self, U: np.ndarray, t: float, P: np.ndarray | None) -> np.ndarray:
        Ud = {f: U[:, i] for i, f in enumerate(self.fields)}
        env = self.op.environment(Ud, t, self._coef_env(P, len(U)))
        for f, sl, tgt in self.op.face_targets(Ud, env):
            U[(slice(None), self.fields.index(f)) + sl[1:]] = tgt
        return U

    def _substeps(self, scale: float = 1.0) -> int:
        """RK4 substeps per twin step from the spectral radius of the linearised right-hand side (power iteration
        at the initial state), with a safety factor."""
        U = self.x0[None, :len(self.node_names)].reshape(1, len(self.fields), *self.grid_shape).copy()
        P = np.array([list(self.estimate.values())]) if self.estimate else None
        rng = np.random.default_rng(0)
        v = rng.standard_normal(U.shape)
        eps = 1e-6 * (1 + np.abs(U).max())
        f0 = self._rhs(U, self.t, P)
        lam = 1.0
        for _ in range(30):
            v /= np.linalg.norm(v) + 1e-300
            Jv = (self._rhs(U + eps * v, self.t, P) - f0) / eps
            lam = float(np.linalg.norm(Jv))
            v = Jv
        return max(1, math.ceil(self.dt * lam * scale / 2.0))

    def step(self, x, v=None) -> np.ndarray:
        """One twin step (``dt``) for a state ``(n,)`` or an ensemble ``(N, n)``: RK4 substeps of the identified PDE
        with each member's coefficients; coefficients themselves do not change here (the filter's random walk)."""
        X = np.atleast_2d(np.asarray(x, float))
        N = len(X)
        nf = len(self.node_names)
        U = X[:, :nf].reshape(N, len(self.fields), *self.grid_shape).copy()
        npar = len(self.estimate)
        P = X[:, nf:nf + npar] if self.estimate else None
        if P is not None:
            ratio = float(np.max(np.abs(P) / np.maximum(np.abs(list(self.estimate.values())), 1e-300)))
            nsub = max(self._nsub, math.ceil(self._nsub * max(ratio, 1.0) * 1.1))
            P = np.maximum(P, 0.0) if all(v > 0 for v in self.estimate.values()) else P
        else:
            nsub = self._nsub
        h = self.dt / nsub
        t = self.t
        for _ in range(nsub):
            k1 = self._rhs(U, t, P)
            k2 = self._rhs(U + 0.5 * h * k1, t + 0.5 * h, P)
            k3 = self._rhs(U + 0.5 * h * k2, t + 0.5 * h, P)
            k4 = self._rhs(U + h * k3, t + h, P)
            U = self._faces(U + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), t + h, P)
            t += h
        out = X.copy()
        out[:, :nf] = U.reshape(N, nf)
        if self.sensor_names:
            out[:, nf + npar:] = out[:, :self.n_nodes] @ self.W.T
        return out[0] if np.asarray(x).ndim == 1 else out

    def advance_clock(self) -> None:
        self.t += self.dt

    # -- sensors -----------------------------------------------------------------------------------------------
    def node_of(self, point) -> int:
        """Index of the grid node nearest to ``point`` (a coordinate per spatial axis)."""
        idx = [int(np.argmin(np.abs(self.axes[c] - float(p)))) for c, p in zip(self.axes, np.atleast_1d(point),
                                                                              strict=True)]
        return int(np.ravel_multi_index(idx, self.grid_shape))

    def field_at(self, x: np.ndarray, field: int = 0) -> np.ndarray:
        nf = self.n_nodes
        return np.asarray(x)[field * nf:(field + 1) * nf].reshape(self.grid_shape)


def simulate_sensors(plant: PDETwinModel, *, sensors: list, t_end: float, every: int = 1, noise: float = 0.0,
                     coefficients: dict[str, float] | None = None, fault: dict[str, float] | None = None,
                     seed: int = 0) -> dict[str, np.ndarray]:
    """Readings of a simulated asset (``plant``, usually a finer grid than the twin): sensor values every ``every``
    steps with Gaussian noise; ``fault`` = ``{"t": t_fault, coefficient: new value}`` changes the asset at ``t``.
    Returns columns ``t`` and ``s0, s1, ...`` plus ``truth_<coefficient>``."""
    rng = np.random.default_rng(seed)
    coef = {**plant.estimate, **(coefficients or {})}
    names = list(plant.estimate)
    x = plant.x0.copy()
    nf = len(plant.node_names)
    for j, k in enumerate(names):
        x[nf + j] = coef[k]
    W = plant.sensor_weights(sensors)
    steps = int(round(t_end / plant.dt))
    cols: dict[str, list] = {"t": [], **{f"s{i}": [] for i in range(len(sensors))}, **{f"truth_{k}": [] for k in names}}
    t = plant.t
    for n in range(1, steps + 1):
        if fault and t >= fault.get("t", np.inf):
            for j, k in enumerate(names):
                if k in fault:
                    x[nf + j] = fault[k]
        x = plant.step(x)
        t += plant.dt
        if n % every == 0:
            cols["t"].append(t)
            vals = W @ x[:plant.n_nodes]
            for i, val in enumerate(vals):
                cols[f"s{i}"].append(float(val) + (rng.normal(0, noise) if noise > 0 else 0.0))
            for j, k in enumerate(names):
                cols[f"truth_{k}"].append(float(x[nf + j]))
    return {k: np.asarray(v, float) for k, v in cols.items()}


def run_twin(model: PDETwinModel, readings: dict[str, np.ndarray], *, sensors: list, sigma_obs: float,
             sigma_process: float = 1e-4, sigma_param: dict[str, float] | None = None,
             limits: dict[str, tuple[float | None, float | None]] | None = None, n_ens: int = 48, seed: int = 0,
             progress=None):
    """Assimilate ``readings`` (columns ``t``, ``s0``...) into ``model`` with :class:`TwinRuntime`. Returns the
    runtime and a summary (history of estimates, alerts, final calibrated coefficients)."""
    from pinneapple_systems.operations.twin import AlertRules, TwinRuntime
    model.set_sensors(sensors)
    obs_names = list(model.sensor_names)
    sp = {nm: sigma_process for nm in model.node_names + obs_names}
    sp.update({k: (sigma_param or {}).get(k, 0.01 * max(abs(v), 1e-12)) for k, v in model.estimate.items()})
    T = readings["t"]
    x = model.x0.copy()                               # the model's initial state carried to the first reading
    for _ in range(max(0, int(round((float(T[0]) - model.t) / model.dt)))):
        x = model.step(x)
        model.advance_clock()
    prior = dict(zip(model.state_names, x.tolist(), strict=True))
    rules = AlertRules(limits={}, param_limits=dict(limits or {}))
    rt = TwinRuntime(model, obs_names, {nm: sigma_obs for nm in obs_names}, sp, x0=prior,
                     sigma0={**{nm: max(sigma_obs, 1e-9) for nm in model.node_names + obs_names},
                             **{k: 0.1 * max(abs(v), 1e-12) for k, v in model.estimate.items()}},
                     rules=rules, n_ens=n_ens, asset="pde-twin", seed=seed)
    hist = {"t": [], **{k: [] for k in model.estimate}, **{f"{k}_sd": [] for k in model.estimate},
            "nis": [], **{f"obs{i}": [] for i in range(len(sensors))}, **{f"pred{i}": [] for i in range(len(sensors))}}
    fields = []
    for n, t in enumerate(T):
        r = {"t": float(t)}
        for i, nm in enumerate(obs_names):
            r[nm] = float(readings[f"s{i}"][n])
        steps = 1 if n == 0 else max(1, int(round((t - T[n - 1]) / model.dt)))
        out = rt.ingest(r)
        for _ in range(steps):
            model.advance_clock()
        hist["t"].append(float(t))
        for k in model.estimate:
            hist[k].append(out["state"][k])
            hist[f"{k}_sd"].append(out["state_sd"][k])
        hist["nis"].append(out.get("nis"))
        for i, nm in enumerate(obs_names):
            hist[f"obs{i}"].append(r[nm])
            hist[f"pred{i}"].append((out.get("predicted") or {}).get(nm))
        fields.append([out["state"][nm] for nm in model.node_names])
        if progress and n % 50 == 0:
            progress(n, len(T))
    alerts = [a.to_dict() for a in rt.alerts]
    return rt, {"history": hist, "alerts": alerts, "fields": np.asarray(fields, float),
                "final": {k: hist[k][-1] for k in model.estimate}, "snapshot": rt.snapshot(),
                "obs_names": obs_names}
