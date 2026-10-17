"""Open, reproducible benchmark for false-data-injection detection: a three-tank gravity water network.

Public ICS security datasets either need a signed request (SWaT, WADI) or are not reachable from every network, so
this module simulates a small plant whose physics is textbook and whose attacks are the ones reported in the
literature. Every number it produces can be regenerated from a seed.

Plant (SI units, one sample per minute)::

    reservoir --pump Fp--> T1 --valve F12--> T2 --valve F23--> T3 --demand Fd--> consumers

* tanks with floor elevations 3.0 / 1.5 / 0.0 m and areas 2.0 / 1.5 / 1.8 m^2;
* gravity valves ``q = c sqrt(head difference)``; the pump runs under on/off level control of T1 (1.0 to 2.5 m);
* demand: diurnal pattern x random daily factor x AR(1) fluctuation; it is throttled when T3 runs nearly empty;
* sensors: three levels (noise 3 mm) and four flow meters (noise 1.5 % of nominal), flows averaged over the minute.

Attacks (:func:`inject`) rewrite the *reported* readings in ``[start, start + length)``:

=====================  ===========================================================================================
``bias``               constant offset on the F12 flow meter (4 % of its mean, about 3 noise sigmas)
``ramp``               slow drift of the T2 level sensor, 0 → 8 cm over the episode
``stuck``              T3 level sensor frozen at its value at the onset
``delayed``            F23 replaced by its own reading 24 h earlier (plausible value, wrong time)
``scaling``            demand meter Fd under-reads by 8 % (metering fraud / water theft)
``coordinated_replay`` every sensor replaced by its reading 24 h earlier (physically consistent: a residual
                       detector cannot see it — reported to show the limit, caught by ``anomaly.replay_score``)
=====================  ===========================================================================================

Data-driven baselines (no physics): :class:`PCAResidualDetector` (Q statistic of a PCA model of sensors and their
increments, the classic multivariate SPC detector), :class:`ARResidualDetector` (a linear data-driven twin that
predicts each sensor from three lags of all sensors; per-sensor prediction residuals) and
:class:`IsolationForestDetector`. All share :class:`~pinneapple_security.cps.CusumAlarm` with the physics detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cps import CusumAlarm, Data, FlowLaw, MassBalance, PhysicsResidualDetector

__all__ = ["SENSORS", "ATTACKS", "simulate_three_tank", "inject", "physics_detector", "PCAResidualDetector",
           "ARResidualDetector", "IsolationForestDetector", "run_benchmark", "ATTACKED_SENSOR", "SHIFTED_REGIME"]

SENSORS = ("h1", "h2", "h3", "Fp", "F12", "F23", "Fd")
ATTACKS = ("bias", "ramp", "stuck", "delayed", "scaling", "coordinated_replay")
DAY = 1440
_Z = np.array([3.0, 1.5, 0.0])
_A = np.array([2.0, 1.5, 1.8])
_C12, _C23 = 0.0105, 0.0092


def simulate_three_tank(days: float = 7.0, seed: int = 0, dt: float = 60.0, substeps: int = 12,
                        demand_scale: float = 1.0, pump_band: tuple[float, float] = (1.0, 2.5)
                        ) -> dict[str, np.ndarray]:
    """Simulate the plant; returns sensor columns (``SENSORS``) plus the true states ``true_h1..3``.
    ``demand_scale`` and ``pump_band`` (T1 levels that switch the pump on and off) change the operating regime
    without changing the physics: a detector that learned the physics should not care, one that learned the
    regime will."""
    rng = np.random.default_rng(seed)
    n = int(days * DAY)
    h = np.array([1.8, 1.4, 1.2])
    pump_on = True
    out = {k: np.empty(n) for k in SENSORS + ("true_h1", "true_h2", "true_h3")}
    ar, day_factor = 0.0, 1.0
    tau = dt / substeps
    for t in range(n):
        if t % DAY == 0:
            day_factor = rng.uniform(0.85, 1.15)
        ar = 0.97 * ar + rng.normal(0, 0.03)
        phase = 2 * np.pi * (t % DAY) / DAY
        demand = 0.0125 * demand_scale * day_factor * (1 + 0.45 * np.sin(phase - 2.0) + 0.15 * np.sin(2 * phase + 0.5)) * (1 + ar)
        demand = max(demand, 0.0)
        for i, k in enumerate(("true_h1", "true_h2", "true_h3")):
            out[k][t] = h[i]
        fl = np.zeros(4)
        for _ in range(substeps):
            if h[0] < pump_band[0]:
                pump_on = True
            elif h[0] > pump_band[1]:
                pump_on = False
            qp = 0.034 * (1 - 0.08 * h[0]) if pump_on else 0.0
            q12 = _C12 * np.sqrt(max(_Z[0] + h[0] - _Z[1] - h[1], 0.0))
            q23 = _C23 * np.sqrt(max(_Z[1] + h[1] - _Z[2] - h[2], 0.0))
            qd = demand * min(1.0, h[2] / 0.3)
            h = h + tau / _A * np.array([qp - q12, q12 - q23, q23 - qd])
            h = np.maximum(h, 0.0)
            fl += np.array([qp, q12, q23, qd]) / substeps
        for i, k in enumerate(("Fp", "F12", "F23", "Fd")):
            out[k][t] = fl[i] + rng.normal(0, 0.015 * 0.015)
    for i, k in enumerate(("h1", "h2", "h3")):
        out[k] = out[f"true_h{i + 1}"] + rng.normal(0, 0.003, n)
    # level sensors read at the start of the minute, flows are minute averages -> h[t+1] - h[t] = dt/A * q[t]
    return out


def inject(D: Data, kind: str, episodes: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    """Copy of ``D`` with the reported readings rewritten in each ``(start, end)`` episode."""
    if kind not in ATTACKS:
        raise ValueError(f"unknown attack {kind!r}; use one of {ATTACKS}")
    X = {k: np.array(v, dtype=float, copy=True) for k, v in D.items()}
    for s, e in episodes:
        if s < DAY and kind in ("delayed", "coordinated_replay"):
            raise ValueError("replay attacks need a day of history before the episode")
        if kind == "bias":
            X["F12"][s:e] += 0.04 * np.mean(D["F12"])
        elif kind == "ramp":
            X["h2"][s:e] += np.linspace(0.0, 0.08, e - s)
        elif kind == "stuck":
            X["h3"][s:e] = D["h3"][s]
        elif kind == "delayed":
            X["F23"][s:e] = D["F23"][s - DAY:e - DAY]
        elif kind == "scaling":
            X["Fd"][s:e] *= 0.92
        else:
            for k in SENSORS:
                X[k][s:e] = D[k][s - DAY:e - DAY]
    return X


def physics_detector(dt: float = 60.0, window: int = 1, k: float = 0.5) -> PhysicsResidualDetector:
    """Grey-box twin of the plant: three storage balances and two valve laws, constants identified from data."""
    return PhysicsResidualDetector(components=[
        MassBalance("h1", ["Fp"], ["F12"], dt=dt, window=window),
        MassBalance("h2", ["F12"], ["F23"], dt=dt, window=window),
        MassBalance("h3", ["F23"], ["Fd"], dt=dt, window=window),
        FlowLaw("F12", "h1", "h2", midpoint=True),
        FlowLaw("F23", "h2", "h3", midpoint=True),
    ], k=k)


def _matrix(D: Data, sensors: tuple[str, ...] | None = None) -> np.ndarray:
    return np.stack([np.asarray(D[k], dtype=float) for k in (sensors or SENSORS)], axis=1)


@dataclass
class _Scaled(CusumAlarm):
    k: float = 0.5
    h: float = 8.0
    upper_only: bool = True
    sensors: tuple[str, ...] | None = None             # columns to read; the three-tank ``SENSORS`` by default
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)

    def _features(self, D: Data) -> np.ndarray:
        X = (_matrix(D, self.sensors) - self.mean_) / self.std_
        dX = np.vstack([np.zeros((1, X.shape[1])), np.diff(X, axis=0)])
        return np.hstack([X, dX])

    def _fit_scale(self, D: Data) -> None:
        X = _matrix(D, self.sensors)
        self.mean_, self.std_ = X.mean(0), X.std(0) + 1e-12

    def _standardise(self, s: np.ndarray) -> np.ndarray:
        return (s - self._mu) / self._sig

    def _fit_score_scale(self, s: np.ndarray) -> None:
        self._mu = np.median(s, axis=0)
        self._sig = 1.4826 * np.median(np.abs(s - self._mu), axis=0) + 1e-12


@dataclass
class PCAResidualDetector(_Scaled):
    """Squared prediction error (Q statistic) of a PCA model of the sensors and their one-step increments."""
    variance: float = 0.95

    @property
    def channels(self) -> list[str]:
        return ["pca_Q"]

    def fit(self, D: Data) -> PCAResidualDetector:
        self._fit_scale(D)
        F = self._features(D)
        self._fm = F.mean(0)
        _, S, Vt = np.linalg.svd(F - self._fm, full_matrices=False)
        r = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), self.variance) + 1)
        self._P = Vt[:r].T
        self._fit_score_scale(self._q(F))
        return self

    def _q(self, F: np.ndarray) -> np.ndarray:
        Fc = F - self._fm
        R = Fc - Fc @ self._P @ self._P.T
        return np.log((R ** 2).sum(1) + 1e-12)[:, None]

    def z(self, D: Data) -> np.ndarray:
        return self._standardise(self._q(self._features(D)))


@dataclass
class ARResidualDetector(_Scaled):
    """Data-driven linear twin: each sensor predicted from ``lags`` past samples of every sensor (ridge regression);
    per-sensor two-sided CUSUM of the standardised one-step prediction residuals."""
    lags: int = 3
    ridge: float = 1e-3
    upper_only: bool = False

    @property
    def channels(self) -> list[str]:
        return [f"ar[{k}]" for k in (self.sensors or SENSORS)]

    def _design(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        cols = [np.vstack([np.full((j, X.shape[1]), np.nan), X[:n - j]]) for j in range(1, self.lags + 1)]
        return np.hstack(cols + [np.ones((n, 1))])

    def _resid(self, D: Data) -> np.ndarray:
        X = (_matrix(D, self.sensors) - self.mean_) / self.std_
        return X - self._design(X) @ self._W

    def fit(self, D: Data) -> ARResidualDetector:
        self._fit_scale(D)
        X = (_matrix(D, self.sensors) - self.mean_) / self.std_
        Phi = self._design(X)[self.lags:]
        Y = X[self.lags:]
        self._W = np.linalg.solve(Phi.T @ Phi + self.ridge * np.eye(Phi.shape[1]), Phi.T @ Y)
        self._fit_score_scale(self._resid(D)[self.lags:])
        return self

    def z(self, D: Data) -> np.ndarray:
        return self._standardise(self._resid(D))


@dataclass
class IsolationForestDetector(_Scaled):
    """Isolation forest (Liu, Ting & Zhou 2008) on sensors and their increments; needs scikit-learn."""
    n_estimators: int = 200
    seed: int = 0
    max_fit: int = 20000

    @property
    def channels(self) -> list[str]:
        return ["iforest"]

    def fit(self, D: Data) -> IsolationForestDetector:
        from sklearn.ensemble import IsolationForest
        self._fit_scale(D)
        F = self._features(D)
        idx = np.random.default_rng(self.seed).permutation(len(F))[:self.max_fit]
        self._m = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed).fit(F[idx])
        self._fit_score_scale(-self._m.score_samples(F)[:, None])
        return self

    def z(self, D: Data) -> np.ndarray:
        return self._standardise(-self._m.score_samples(self._features(D))[:, None])


def _episodes(n: int, count: int, length: int, rng: np.random.Generator, start: int = DAY) -> list[tuple[int, int]]:
    slots = np.linspace(start, n - length - 60, count + 1).astype(int)[:-1]
    gap = (slots[1] - slots[0]) - length if count > 1 else 0
    offs = rng.integers(0, max(gap // 2, 1), size=len(slots))
    return [(int(s + o), int(s + o) + length) for s, o in zip(slots, offs, strict=True)]


ATTACKED_SENSOR = {"bias": "F12", "ramp": "h2", "stuck": "h3", "delayed": "F23", "scaling": "Fd"}
SHIFTED_REGIME = {"demand_scale": 1.25, "pump_band": (1.4, 2.2)}


def _channel_sensors(det: Any) -> dict[str, set[str]]:
    if isinstance(det, PhysicsResidualDetector):
        out = {}
        for c in det.components:
            if isinstance(c, MassBalance):
                out[c.name] = {c.level, *c.inflows, *c.outflows}
            elif isinstance(c, FlowLaw):
                out[c.name] = {c.flow, c.up} | ({c.down} if c.down else set())
        return out
    if isinstance(det, ARResidualDetector):
        return {f"ar[{k}]": {k} for k in SENSORS}
    return {}


def run_benchmark(seed: int = 0, train_days: float = 7.0, calib_days: float = 7.0, test_days: float = 10.0,
                  episodes: int = 20, length: int = 240, false_alarms_per_day: float = 1.0,
                  attacks: tuple[str, ...] = ATTACKS, detectors: dict[str, Any] | None = None,
                  regimes: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Fit every detector on a trusted trace, calibrate it to ``false_alarms_per_day`` on a second clean trace,
    then, for each operating regime (nominal, and a shifted regime the detectors never saw), score one clean test
    trace and one attacked copy per attack with ``episodes`` episodes of ``length`` minutes.

    Per detector, regime and attack: episode-level ROC AUC (:func:`~pinneapple_security.cps.window_auc`),
    detection rate, median delay (minutes), false alarms per day, and the fraction of detected episodes whose blamed
    channel involves the attacked sensor (localisation; residual-per-channel detectors only)."""
    from .cps import episode_metrics, window_auc
    regimes = regimes if regimes is not None else {"nominal": {}, "shifted": SHIFTED_REGIME}
    train = simulate_three_tank(train_days, seed=seed)
    calib = simulate_three_tank(calib_days, seed=seed + 1)
    if detectors is None:
        detectors = {"physics": physics_detector(), "ar_twin": ARResidualDetector(), "pca": PCAResidualDetector()}
        try:
            import sklearn  # noqa: F401
            detectors["iforest"] = IsolationForestDetector(seed=seed)
        except ImportError:
            pass
    out: dict[str, Any] = {"config": {"seed": seed, "train_days": train_days, "calib_days": calib_days,
                                      "test_days": test_days, "episodes": episodes, "length": length,
                                      "false_alarms_per_day": false_alarms_per_day, "regimes": regimes},
                           "results": {}}
    tests = {}
    for j, (rname, kw) in enumerate(regimes.items()):
        test = simulate_three_tank(test_days, seed=seed + 2 + 10 * j, **kw)
        n = len(test["h1"])
        tests[rname] = (test, _episodes(n, episodes, length, np.random.default_rng(seed + 3 + 10 * j)))
    for name, det in detectors.items():
        det.fit(train)
        det.calibrate(calib, false_alarms_per_day)
        cs = _channel_sensors(det)
        row: dict[str, Any] = {"threshold": det.h}
        for rname, (test, eps) in tests.items():
            n = len(test["h1"])
            clean = det.detect(test)
            stat_clean = det.score(test)
            rr: dict[str, Any] = {"false_alarms_per_day_clean": len(clean["alarms"]) / (n / DAY)}
            for kind in attacks:
                X = inject(test, kind, eps)
                res = det.detect(X)
                m = episode_metrics(res["alarms"], eps, n)
                loc = None
                if cs and kind in ATTACKED_SENSOR:
                    hits = []
                    for s, e in eps:
                        first = [c for a, c in zip(res["alarms"], res["channels"], strict=True) if s <= a < e][:1]
                        if first:
                            hits.append(ATTACKED_SENSOR[kind] in cs.get(first[0], set()))
                    loc = float(np.mean(hits)) if hits else None
                rr[kind] = {"auc": window_auc(det.score(X), eps, stat_clean), "detection_rate": m["detection_rate"],
                            "median_delay_min": m["median_delay"], "false_alarms_per_day": m["false_alarms_per_day"],
                            "localisation": loc}
            row[rname] = rr
        out["results"][name] = row
    return out
