"""Digital twin runtime: run a validated model next to live data, estimate its state, forecast, alert and watch the
model's own validity.

One model, one asset, one data stream at a time -- deliberately small:

* **state estimation** with the stochastic ensemble Kalman filter of
  :class:`pinneapple_analysis.state_estimation.EnsembleKalmanFilter` (Evensen 2009). A lumped physical model can
  augment its state with parameters (a leak flow, a heater efficiency) that the filter tracks as random walks;
* **forecasts** by propagating the analysis ensemble with a planned (or held) input, with 5-95 % bands;
* **alerts** with explicit rules: limits on the estimate, limits reached by the forecast, parameters out of range,
  and a filter-consistency test (normalised innovation squared above its chi-square quantile for several
  consecutive samples: the data no longer agree with the model);
* **validity monitoring** with :func:`pinneapple_analysis.uncertainty.drift.assess_drift` on a rolling window of
  one-step-ahead predictions (accuracy, calibration of the predictive spread, residual trend, inputs outside the
  model's training box). A ``review`` / ``retrain`` decision raises a *revalidation required* alert, and
  :meth:`TwinRuntime.revalidate` re-scores the model open loop on the recent history against its validation
  record.

Every update, alert and revalidation can be written to a :class:`TwinStore` (SQLite by default).
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy import stats

from pinneapple_analysis.state_estimation.kalman import EnsembleKalmanFilter
from pinneapple_analysis.uncertainty.drift import DriftThresholds, assess_drift

from .plant import TankParams, tank_step
from .rom import ControlledOpInf

# --------------------------------------------------------------------------------------------------------------
# Models the runtime can run
# --------------------------------------------------------------------------------------------------------------


class LumpedTankModel:
    """The heated tank's mass and energy balances as a twin model, optionally with parameters in the state.

    ``estimate`` may contain ``"q_leak"`` (unaccounted outflow [m3/s], prior 0; not clipped at zero, so a flow
    meter reading high shows as a negative value and the estimate stays unbiased) and ``"eta"`` (heater efficiency,
    prior 1); the filter then tracks them as random walks, which is how the twin *quantifies* a leak or a heater
    losing power instead of only noticing a mismatch.
    """

    kind = "lumped_tank"
    input_names = ["q_in", "Q", "T_in", "T_amb"]

    def __init__(self, params: TankParams | None = None, dt: float = 10.0,
                 estimate: Sequence[str] = ("q_leak", "eta")):
        bad = set(estimate) - {"q_leak", "eta"}
        if bad:
            raise ValueError(f"cannot estimate {sorted(bad)}; choose from q_leak, eta")
        self.p = params or TankParams()
        self.dt = float(dt)
        self.estimate = list(estimate)
        self.state_names = ["h", "T"] + self.estimate
        self.bounds: dict[str, list[float]] = {"h": [self.p.h_min, self.p.h_max]}
        self.validation: dict[str, object] = {}

    def prior(self) -> dict[str, float]:
        return {"q_leak": 0.0, "eta": 1.0}

    def step(self, x, v) -> np.ndarray:
        """One step from state ``x`` (n,) or an ensemble (N, n), inputs ``v`` = [q_in, Q, T_in, T_amb] (m,) or
        (N, m)."""
        x = np.asarray(x, dtype=float)
        X = np.atleast_2d(x)
        kw = {}
        if "q_leak" in self.estimate:
            kw["q_leak"] = X[:, 2 + self.estimate.index("q_leak")]
        if "eta" in self.estimate:
            kw["eta"] = np.clip(X[:, 2 + self.estimate.index("eta")], 0.0, 1.2)
        v = np.asarray(v, dtype=float)
        u, d = (v[:2], v[2:4]) if v.ndim == 1 else (v[:, :2].T, v[:, 2:4].T)    # one input or one per row
        out = X.copy()
        out[:, :2] = tank_step(X[:, :2].T, u, d, self.p, self.dt, substeps=2, **kw).T
        return out[0] if x.ndim == 1 else out

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": self.p.to_dict(), "dt": self.dt, "estimate": self.estimate}


class SurrogateModel:
    """A :class:`ControlledOpInf` reduced model as a twin model (no parameters in the state)."""

    kind = "controlled_opinf"

    def __init__(self, rom: ControlledOpInf):
        self.rom = rom
        self.dt = rom.dt
        self.state_names = list(rom.state_names)
        self.input_names = list(rom.input_names)
        self.estimate: list[str] = []
        self.bounds = rom.bounds
        self.validation = rom.validation

    def prior(self) -> dict[str, float]:
        return {}

    def step(self, x, v) -> np.ndarray:
        return self.rom.step(x, v)

    def to_dict(self) -> dict:
        return self.rom.to_dict()


def model_from_dict(d: dict):
    """Rebuild a twin model from its JSON form (``kind`` = ``lumped_tank`` or ``controlled_opinf``)."""
    if d.get("kind") == "lumped_tank":
        return LumpedTankModel(TankParams(**d.get("params", {})), d.get("dt", 10.0), d.get("estimate", ()))
    if d.get("kind") == "controlled_opinf":
        return SurrogateModel(ControlledOpInf.from_dict(d))
    raise ValueError(f"unknown model kind {d.get('kind')!r}")


# --------------------------------------------------------------------------------------------------------------
# Alerts and storage
# --------------------------------------------------------------------------------------------------------------


@dataclass
class AlertRules:
    """Alert thresholds. ``limits`` and ``param_limits`` map a state name to ``(low, high)`` (``None`` = open)."""

    limits: dict[str, tuple[float | None, float | None]] = field(
        default_factory=lambda: {"h": (0.3, 1.8), "T": (None, 40.0)})
    param_limits: dict[str, tuple[float | None, float | None]] = field(
        default_factory=lambda: {"q_leak": (None, 3e-4), "eta": (0.85, None)})
    param_sigmas: float = 3.0              # raised when the limit is outside the estimate's +- k sigma, cleared
                                           # when the estimate itself is back inside (hysteresis)
    nis_probability: float = 0.999         # chi-square quantile of the consistency test
    nis_consecutive: int = 3               # consecutive samples above it before alerting
    forecast_steps: int = 60               # horizon of the forecast checked against ``limits``
    drift_window: int = 90                 # samples in the validity window
    drift_every: int = 30                  # samples between validity checks
    review_consecutive: int = 2            # "review" alerts after this many checks in a row; "retrain" at once
    hold_window: int = 20                  # readings averaged for the held input of a forecast
    bias_practical: float = 0.5            # a bias below this fraction of the predictive spread is not alerted
    drift: DriftThresholds = field(default_factory=lambda: DriftThresholds(
        nrmse_review=math.inf, nrmse_retrain=math.inf, coverage_z=4.0, trend_alpha=1e-3))


@dataclass
class Alert:
    t: float
    kind: str
    severity: str           # info | warning | critical
    message: str
    data: dict[str, object] = field(default_factory=dict)
    cleared: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class TwinStore:
    """History of readings, estimates, alerts and revalidations.

    ``url`` is a SQLite path (``":memory:"`` by default) or ``sqlite:///path``; ``postgresql://...`` works when
    ``psycopg`` is installed. One table per record type, JSON payloads, indexed by asset and time.
    """

    TABLES = ("readings", "estimates", "alerts", "events")

    def __init__(self, url: str = ":memory:"):
        self.url = url
        self._lock = threading.Lock()
        if url.startswith(("postgresql://", "postgres://")):
            import psycopg  # optional dependency

            self._conn = psycopg.connect(url, autocommit=True)
            self._ph = "%s"
            serial = "BIGSERIAL PRIMARY KEY"
        else:
            path = url[len("sqlite:///"):] if url.startswith("sqlite:///") else url
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._ph = "?"
            serial = "INTEGER PRIMARY KEY AUTOINCREMENT"
        with self._lock:
            cur = self._conn.cursor()
            for tname in self.TABLES:
                cur.execute(f"CREATE TABLE IF NOT EXISTS {tname} (id {serial}, asset TEXT, t DOUBLE PRECISION, "
                            "payload TEXT)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS {tname}_asset_t ON {tname} (asset, t)")
            self._conn.commit()

    def write(self, table: str, asset: str, t: float, payload: dict) -> None:
        if table not in self.TABLES:
            raise ValueError(f"unknown table {table!r}")
        with self._lock:
            self._conn.cursor().execute(
                f"INSERT INTO {table} (asset, t, payload) VALUES ({self._ph}, {self._ph}, {self._ph})",
                (asset, float(t), json.dumps(payload, default=_json_default)))
            self._conn.commit()

    def read(self, table: str, asset: str, since: float = -math.inf, limit: int = 10000) -> list[dict]:
        if table not in self.TABLES:
            raise ValueError(f"unknown table {table!r}")
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(f"SELECT t, payload FROM {table} WHERE asset = {self._ph} AND t >= {self._ph} "
                        f"ORDER BY id DESC LIMIT {int(limit)}", (asset, float(since) if np.isfinite(since) else -1e300))
            rows = cur.fetchall()
        return [json.loads(p) for _, p in reversed(rows)]

    def count(self, table: str, asset: str) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE asset = {self._ph}", (asset,))
            return int(cur.fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# --------------------------------------------------------------------------------------------------------------
# The runtime
# --------------------------------------------------------------------------------------------------------------


class TwinRuntime:
    """Assimilate readings one at a time into ``model`` and keep the twin's state, alerts and validity.

    Parameters
    ----------
    model : :class:`LumpedTankModel`, :class:`SurrogateModel` or anything with ``state_names``, ``input_names``,
        ``dt``, ``step(x, v)``, ``bounds``, ``validation``, ``prior()`` and ``to_dict()``.
    obs_names : measured states (a subset of ``model.state_names``).
    sigma_obs : measurement noise standard deviation per observed state.
    sigma_process : process noise per step per state (parameters: their random-walk step).
    x0 : initial state (defaults: first reading for observed states, ``model.prior()`` for parameters).
    sigma0 : initial spread per state.
    """

    def __init__(self, model, obs_names: Sequence[str], sigma_obs: dict[str, float],
                 sigma_process: dict[str, float], *, x0: dict[str, float] | None = None,
                 sigma0: dict[str, float] | None = None, n_ens: int = 48, inflation: float = 1.0,
                 rules: AlertRules | None = None, asset: str = "asset-1", store: TwinStore | None = None,
                 seed: int = 0):
        self.model = model
        self.obs_names = list(obs_names)
        missing = set(self.obs_names) - set(model.state_names)
        if missing:
            raise ValueError(f"observed {sorted(missing)} are not states of the model {model.state_names}")
        self.obs_idx = [model.state_names.index(n) for n in self.obs_names]
        self.sigma_obs = np.array([float(sigma_obs[n]) for n in self.obs_names])
        self.sigma_process = np.array([float(sigma_process.get(n, 0.0)) for n in model.state_names])
        self.x0 = dict(x0 or {})
        self.sigma0 = dict(sigma0 or {})
        self.rules = rules or AlertRules()
        self.asset = asset
        self.store = store
        self.n = len(model.state_names)
        self.filter = EnsembleKalmanFilter(
            self.n, len(self.obs_names), f=lambda x: x, h=lambda x: x[self.obs_idx],
            Q=np.diag(np.maximum(self.sigma_process, 1e-12) ** 2), R=np.diag(self.sigma_obs ** 2),
            n_ens=n_ens, inflation=inflation, seed=seed)
        self.t: float | None = None
        self.v_last: np.ndarray | None = None
        self.n_updates = 0
        self._nis_run = 0
        self.nis_threshold = float(stats.chi2.ppf(self.rules.nis_probability, len(self.obs_names)))
        self.active: dict[str, Alert] = {}
        self.alerts: list[Alert] = []
        win = self.rules.drift_window
        self._win: dict[str, deque] = {k: deque(maxlen=win) for k in ("y", "yp", "sp", "v")}
        self.history: deque[dict] = deque(maxlen=20000)
        self.last_drift: dict | None = None
        self._review_run = 0
        self.revalidations: list[dict] = []
        self.status = "starting"
        self.timing = {"assimilation_ms": 0.0}

    # ---------------------------------------------------------------------------------------------------- core

    def _inputs(self, reading: dict[str, float]) -> np.ndarray:
        try:
            return np.array([float(reading[n]) for n in self.model.input_names])
        except KeyError as e:
            raise KeyError(f"reading lacks input {e.args[0]!r}; the model needs {self.model.input_names}") from e

    def _initialize(self, reading: dict[str, float]) -> None:
        prior = self.model.prior()
        x0, s0 = np.zeros(self.n), np.zeros(self.n)
        for i, name in enumerate(self.model.state_names):
            if name in self.x0:
                x0[i] = self.x0[name]
            elif name in reading:
                x0[i] = float(reading[name])
            else:
                x0[i] = prior.get(name, 0.0)
            default_s = self.sigma_obs[self.obs_names.index(name)] if name in self.obs_names else \
                10 * max(self.sigma_process[i], 1e-9)
            s0[i] = self.sigma0.get(name, default_s)
        self.filter.initialize(x0, np.diag(s0 ** 2))

    def ingest(self, reading: dict[str, float]) -> dict:
        """Assimilate one reading: a dict with ``t`` [s], the model's inputs and the observed states (a missing or
        NaN observation skips the analysis step: the twin then runs on the model alone)."""
        t0 = time.perf_counter()
        t = float(reading["t"])
        v = self._inputs(reading)
        y = np.array([float(reading.get(n, np.nan)) for n in self.obs_names])
        f = self.filter
        prior_pred = None
        if self.t is None:
            self._initialize(reading)
        else:
            if t <= self.t:
                raise ValueError(f"reading at t={t} is not after the previous one (t={self.t})")
            steps = max(1, int(round((t - self.t) / self.model.dt)))
            for _ in range(steps):
                self._predict(self.v_last)
            HX = f.ensemble[self.obs_idx]
            prior_pred = (HX.mean(1), np.sqrt(HX.var(1, ddof=1) + self.sigma_obs ** 2))
        self._clip_ensemble()
        out: dict[str, object] = {"t": t}
        finite = np.isfinite(y)
        if self.t is not None and finite.all():
            HX = f.ensemble[self.obs_idx]
            innov = y - HX.mean(1)
            S = np.cov(HX).reshape(len(y), len(y)) + np.diag(self.sigma_obs ** 2)
            nis = float(innov @ np.linalg.solve(S, innov))
            f.update(y)
            self._clip_ensemble()
            out["nis"] = nis
            out["innovation"] = dict(zip(self.obs_names, innov.tolist(), strict=False))
            self._win["y"].append(y)
            self._win["yp"].append(prior_pred[0])
            self._win["sp"].append(prior_pred[1])
            self._win["v"].append(self.v_last)
        self.t, self.v_last = t, v
        self.n_updates += 1
        mean, sd = f.mean, np.sqrt(np.maximum(np.diag(f.covariance), 0))
        out["state"] = dict(zip(self.model.state_names, mean.tolist(), strict=False))
        out["state_sd"] = dict(zip(self.model.state_names, sd.tolist(), strict=False))
        out["observed"] = {n: (None if not np.isfinite(val) else float(val)) for n, val in zip(self.obs_names, y, strict=False)}
        out["inputs"] = dict(zip(self.model.input_names, v.tolist(), strict=False))
        if prior_pred is not None:
            out["predicted"] = dict(zip(self.obs_names, prior_pred[0].tolist(), strict=False))
        new_alerts = self._check_alerts(out)
        if self.n_updates % self.rules.drift_every == 0 and len(self._win["y"]) >= 10:
            new_alerts += self._check_validity(t)
        out["alerts"] = [a.to_dict() for a in new_alerts]
        self.status = self._status()
        out["status"] = self.status
        self.timing["assimilation_ms"] = 1000 * (time.perf_counter() - t0)
        self.history.append(out)
        if self.store is not None:
            self.store.write("readings", self.asset, t, dict(reading))
            self.store.write("estimates", self.asset, t, {k: out[k] for k in ("t", "state", "state_sd", "status")
                                                          if k in out} | {"nis": out.get("nis")})
            for a in new_alerts:
                self.store.write("alerts", self.asset, t, a.to_dict())
        return out

    def _predict(self, v: np.ndarray) -> None:
        """EnKF forecast step, vectorised over members (same as ``EnsembleKalmanFilter.predict``)."""
        f = self.filter
        E = self.model.step(f.ensemble.T, v).T
        E += self.sigma_process[:, None] * f.rng.standard_normal(E.shape)
        mu = E.mean(1, keepdims=True)
        f.ensemble = mu + f.inflation * (E - mu)

    def _clip_ensemble(self) -> None:
        names = self.model.state_names
        E = self.filter.ensemble
        if "h" in names:
            E[names.index("h")] = np.maximum(E[names.index("h")], 1e-3)

    # ---------------------------------------------------------------------------------------------------- alerts

    def _raise(self, key: str, t: float, kind: str, severity: str, message: str, data: dict) -> list[Alert]:
        if key in self.active:
            self.active[key].data = data
            return []
        a = Alert(t, kind, severity, message, data)
        self.active[key] = a
        self.alerts.append(a)
        return [a]

    def _clear(self, key: str, t: float) -> list[Alert]:
        a = self.active.pop(key, None)
        if a is None:
            return []
        c = Alert(t, a.kind, "info", f"cleared: {a.message}", {"raised_at": a.t}, cleared=True)
        self.alerts.append(c)
        return [c]

    def _check_alerts(self, out: dict) -> list[Alert]:
        t, st, sd = out["t"], out["state"], out["state_sd"]
        new: list[Alert] = []
        for name, (lo, hi) in self.rules.limits.items():
            if name not in st:
                continue
            key = f"limit:{name}"
            val = st[name]
            margin = 2 * sd[name]                    # hysteresis: clear only once well back inside
            if (lo is not None and val < lo) or (hi is not None and val > hi):
                side = "below" if lo is not None and val < lo else "above"
                lim = lo if side == "below" else hi
                new += self._raise(key, t, "limit", "critical", f"estimated {name} = {val:.4g} is {side} its limit "
                                   f"{lim:.4g}", {"state": name, "value": val, "limit": lim})
            elif (lo is None or val > lo + margin) and (hi is None or val < hi - margin):
                new += self._clear(key, t)
        k = self.rules.param_sigmas
        for name, (lo, hi) in self.rules.param_limits.items():
            if name not in st:
                continue
            key = f"param:{name}"
            val, s = st[name], sd[name]
            if (hi is not None and val - k * s > hi) or (lo is not None and val + k * s < lo):
                side = "above" if hi is not None and val - k * s > hi else "below"
                lim = hi if side == "above" else lo
                new += self._raise(key, t, "parameter", "warning",
                                   f"estimated {name} = {val:.4g} +- {s:.2g} is {side} {lim:.4g} "
                                   f"(beyond {k:g} sigma)", {"parameter": name, "value": val, "sd": s, "limit": lim})
            elif (hi is None or val < hi) and (lo is None or val > lo):   # hysteresis: the mean is back inside
                new += self._clear(key, t)
        nis = out.get("nis")
        if nis is not None:
            self._nis_run = self._nis_run + 1 if nis > self.nis_threshold else 0
            if self._nis_run >= self.rules.nis_consecutive:
                new += self._raise("nis", t, "consistency", "warning",
                                   f"data disagree with the model: normalised innovation {nis:.1f} > "
                                   f"{self.nis_threshold:.1f} for {self._nis_run} samples",
                                   {"nis": nis, "threshold": self.nis_threshold, "run": self._nis_run})
            elif self._nis_run == 0:
                new += self._clear("nis", t)
        if self.rules.forecast_steps and self.n_updates % 5 == 0:
            fc = self.forecast(self.rules.forecast_steps)
            hit = fc["first_limit_crossing"]
            if hit is not None:
                new += self._raise("forecast", t, "forecast", "warning",
                                   f"forecast: {hit['state']} reaches {hit['limit']:.4g} in {hit['in_s']:.0f} s "
                                   "(5-95 % band)", hit)
            else:
                new += self._clear("forecast", t)
        return new

    def _check_validity(self, t: float) -> list[Alert]:
        y, yp, sp, V = (np.array(self._win[k]) for k in ("y", "yp", "sp", "v"))
        bounds = None
        names = self.model.input_names
        if all(n in self.model.bounds for n in names):
            bounds = (np.array([self.model.bounds[n][0] for n in names]),
                      np.array([self.model.bounds[n][1] for n in names]))
        if bounds is not None:                     # an input held constant in training: allow its sensor noise
            flat = bounds[1] - bounds[0] <= 0
            bounds[0][flat] -= np.inf
            bounds[1][flat] += np.inf
        rep = assess_drift(y, yp, sigma=sp, X_new=V if bounds is not None else None, bounds=bounds,
                           thresholds=self.rules.drift)
        # a significant but practically negligible bias (below bias_practical x the predictive spread) is noted,
        # not alerted: with many samples a model bias far inside the noise is "significant"
        rel_bias = np.abs(rep.bias) / np.mean(sp, axis=0)
        if any(r.startswith("bias significant") for r in rep.reasons) and np.all(rel_bias < self.rules.bias_practical):
            rep = assess_drift(y, yp, sigma=sp, X_new=V if bounds is not None else None, bounds=bounds,
                               thresholds=replace(self.rules.drift, bias_z=math.inf))
        self.last_drift = {"t": t, "n": rep.n, "decision": rep.decision, "reasons": rep.reasons,
                           "relative_bias": dict(zip(self.obs_names, rel_bias.tolist(), strict=False)),
                           "nrmse": dict(zip(self.obs_names, rep.nrmse.tolist(), strict=False)),
                           "bias": dict(zip(self.obs_names, rep.bias.tolist(), strict=False)),
                           "coverage": None if rep.coverage is None else dict(zip(self.obs_names,
                                                                                 rep.coverage.tolist(), strict=False)),
                           "out_of_bounds": rep.out_of_bounds}
        if self.store is not None:
            self.store.write("events", self.asset, t, {"type": "drift_check", **self.last_drift})
        if rep.decision == "ok":
            self._review_run = 0
            return self._clear("drift", t)
        self._review_run = self._review_run + 1 if rep.decision == "review" else 0
        if rep.decision == "review" and self._review_run < self.rules.review_consecutive:
            return []                              # one marginal flag is noted (last_drift), not alerted
        return self._raise("drift", t, "revalidation", "critical" if rep.decision == "retrain" else "warning",
                           f"revalidation required ({rep.decision}): " + "; ".join(rep.reasons), self.last_drift)

    def _status(self) -> str:
        if any(a.severity == "critical" for a in self.active.values()):
            return "alarm"
        if "drift" in self.active:
            return "revalidation_required"
        if self.active:
            return "warning"
        return "normal"

    # ---------------------------------------------------------------------------------------------------- outputs

    def forecast(self, steps: int, v_plan: np.ndarray | None = None) -> dict:
        """Propagate the current ensemble ``steps`` model steps with inputs ``v_plan`` (steps, m), or hold the
        inputs: the mean of the last ``rules.hold_window`` readings, each member with a constant offset drawn from
        their spread (a held input is not known exactly: a 1 % flow-meter noise integrates into the level).
        Returns times, mean and 5/95 % bands per state, and the first limit crossing of the band."""
        if self.t is None:
            raise RuntimeError("no reading assimilated yet")
        E = self.filter.ensemble.copy()
        n_ens = E.shape[1]
        rng = np.random.default_rng(self.n_updates)
        if v_plan is None and not self.model.input_names:                # a model without inputs
            Vp = np.zeros((steps, n_ens, 0))
        elif v_plan is None:
            recent = np.array([h["inputs"][n] for h in list(self.history)[-self.rules.hold_window:]
                               for n in self.model.input_names]).reshape(-1, len(self.model.input_names))
            if len(recent) == 0:
                recent = self.v_last[None]
            spread = recent.std(0, ddof=1) if len(recent) > 1 else np.zeros(recent.shape[1])
            Vm = recent.mean(0) + spread * rng.standard_normal((n_ens, recent.shape[1]))
            Vp = np.repeat(Vm[None], steps, 0)                         # (steps, n_ens, m)
        else:
            Vp = np.asarray(v_plan, dtype=float)
        means, lo, hi = [], [], []
        for k in range(steps):
            E = self.model.step(E.T, Vp[k]).T
            E += self.sigma_process[:, None] * rng.standard_normal(E.shape)
            means.append(E.mean(1))
            lo.append(np.percentile(E, 5, axis=1))
            hi.append(np.percentile(E, 95, axis=1))
        means, lo, hi = np.array(means), np.array(lo), np.array(hi)
        ts = self.t + self.model.dt * np.arange(1, steps + 1)
        crossing = None
        for name, (llo, lhi) in self.rules.limits.items():
            if name not in self.model.state_names:
                continue
            j = self.model.state_names.index(name)
            for k in range(steps):
                if (lhi is not None and hi[k, j] > lhi) or (llo is not None and lo[k, j] < llo):
                    lim = lhi if lhi is not None and hi[k, j] > lhi else llo
                    if crossing is None or ts[k] - self.t < crossing["in_s"]:
                        crossing = {"state": name, "limit": lim, "at_t": float(ts[k]), "in_s": float(ts[k] - self.t)}
                    break
        return {"t": ts.tolist(), "state_names": self.model.state_names,
                "mean": {n: means[:, j].tolist() for j, n in enumerate(self.model.state_names)},
                "p05": {n: lo[:, j].tolist() for j, n in enumerate(self.model.state_names)},
                "p95": {n: hi[:, j].tolist() for j, n in enumerate(self.model.state_names)},
                "first_limit_crossing": crossing}

    def revalidate(self, n_last: int = 180, tolerance_factor: float = 3.0) -> dict:
        """Re-score the model *open loop* on the last ``n_last`` assimilated samples: start from the estimate at
        the beginning of the window, replay the recorded inputs, compare with the observations. Passes when every
        observed state's RMSE is within ``tolerance_factor`` x the model's validation rollout RMSE (or, with no
        validation record, within 3 x the measurement noise)."""
        hist = [h for h in list(self.history)[-n_last:] if all(h["observed"].get(n) is not None
                                                               for n in self.obs_names)]
        if len(hist) < 10:
            raise ValueError("need at least 10 assimilated samples to revalidate")
        x = np.array([hist[0]["state"][n] for n in self.model.state_names])
        preds = [x[self.obs_idx]]
        for a, b in zip(hist[:-1], hist[1:], strict=False):
            v = np.array([a["inputs"][n] for n in self.model.input_names])
            for _ in range(max(1, int(round((b["t"] - a["t"]) / self.model.dt)))):
                x = self.model.step(x, v)
            preds.append(x[self.obs_idx])
        P = np.array(preds)
        Y = np.array([[h["observed"][n] for n in self.obs_names] for h in hist])
        rmse = np.sqrt(np.mean((P - Y) ** 2, 0))
        ref = self.model.validation.get("rollout_rmse") if isinstance(self.model.validation, dict) else None
        checks = []
        for j, name in enumerate(self.obs_names):
            base = float(ref[name]) if ref and name in ref else 0.0
            base = max(base, float(self.sigma_obs[j]))
            limit = tolerance_factor * base
            checks.append({"state": name, "rmse": float(rmse[j]), "limit": limit, "reference_rmse": base,
                           "passed": bool(rmse[j] <= limit)})
        rec = {"t": self.t, "n_samples": len(hist), "window_s": hist[-1]["t"] - hist[0]["t"], "checks": checks,
               "passed": all(c["passed"] for c in checks), "model": self.model.kind,
               "method": "open-loop replay of recorded inputs from the window's first estimate"}
        self.revalidations.append(rec)
        if self.store is not None:
            self.store.write("events", self.asset, self.t, {"type": "revalidation", **rec})
        if rec["passed"]:
            self._clear("drift", self.t)
            self.status = self._status()
        return rec

    def snapshot(self) -> dict:
        """Current state, active alerts and validity, JSON-ready (the endpoint other systems read)."""
        last = self.history[-1] if self.history else {}
        return {"asset": self.asset, "t": self.t, "status": self.status, "model": self.model.kind,
                "state": last.get("state"), "state_sd": last.get("state_sd"), "n_updates": self.n_updates,
                "active_alerts": [a.to_dict() for a in self.active.values()], "last_validity_check": self.last_drift,
                "last_revalidation": self.revalidations[-1] if self.revalidations else None,
                "assimilation_ms": self.timing["assimilation_ms"]}
