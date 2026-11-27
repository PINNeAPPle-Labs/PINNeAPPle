"""``StateEstimator``: sensors → state → physics → prediction → correction, behind one name (#215).

The four filters existed (or now exist) as separate classes with the same recursion. This puts the choice behind one
object, so a twin can say which filter it uses and a case can change it without being rewritten:

    >>> import numpy as np
    >>> from pinneapple_analysis.state_estimation.estimator import StateEstimator
    >>> model = {"n_state": 2, "n_obs": 1, "f": lambda x: x, "h": lambda x: x[:1]}
    >>> est = StateEstimator("ukf", **model).initialize(np.zeros(2), np.eye(2))
    >>> report = est.run(np.zeros((5, 1)), truth=np.zeros((5, 2)))    # doctest: +SKIP
    >>> report.rmse                                                   # doctest: +SKIP
    array([0., 0.])

What the report carries is the point: the filtered states, the innovations, and — when the truth is known, as it is
in a twin case and never in the field — the error per component. A filter that is not compared with the truth it is
supposed to recover has not been shown to work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .kalman import EnsembleKalmanFilter, ExtendedKalmanFilter
from .unscented import ParticleFilter, UnscentedKalmanFilter

__all__ = ["EstimateReport", "METHODS", "StateEstimator"]

#: the filters a twin can choose from
METHODS = ("ekf", "ukf", "enkf", "particle")


@dataclass
class EstimateReport:
    """What a filtered time series came to: the states, the innovations, and the error against the truth."""

    method: str
    states: np.ndarray                        # (n_steps, n_state)
    innovations: np.ndarray                   # (n_steps, n_obs)
    errors: np.ndarray | None = None          # (n_steps, n_state), only when the truth is known
    rmse: np.ndarray | None = None            # (n_state,)
    max_abs: np.ndarray | None = None
    ess: np.ndarray | None = None             # (n_steps,), particle filter only
    resampled: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(len(self.states))

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "n_steps": self.n_steps,
                "states": self.states.tolist(), "innovations": self.innovations.tolist(),
                "errors": None if self.errors is None else self.errors.tolist(),
                "rmse": None if self.rmse is None else self.rmse.tolist(),
                "max_abs": None if self.max_abs is None else self.max_abs.tolist(),
                "ess": None if self.ess is None else self.ess.tolist(),
                "resampled": None if self.resampled is None else self.resampled.astype(bool).tolist(),
                **self.extra}

    def __str__(self) -> str:
        if self.rmse is None:
            return f"{self.method}: {self.n_steps} steps (no truth to compare against)"
        worst = int(np.argmax(self.rmse))
        return (f"{self.method}: {self.n_steps} steps, rmse {np.array2string(self.rmse, precision=3)}, "
                f"worst component {worst} ({self.rmse[worst]:.3g})")


class StateEstimator:
    """One filter, chosen by name, driven over a stream of observations.

    Parameters
    ----------
    method:
        ``"ekf"`` (linearised, finite-difference or analytic Jacobians), ``"ukf"`` (sigma points, no Jacobian),
        ``"enkf"`` (ensemble, for high dimension) or ``"particle"`` (a full non-Gaussian belief).
    n_state, n_obs:
        Dimensions of the state and the observation.
    f, h:
        State transition and observation operator.
    Q, R:
        Process and observation noise covariances.
    options:
        Handed to the chosen filter: ``n_ens``/``inflation`` (EnKF), ``alpha``/``beta``/``kappa`` (UKF),
        ``n_particles``/``ess_fraction`` (particle).
    """

    def __init__(self, method: str = "ekf", *, n_state: int, n_obs: int, f, h,
                 Q: np.ndarray | None = None, R: np.ndarray | None = None, seed: int = 0, **options: Any) -> None:
        key = str(method).lower()
        if key not in METHODS:
            raise ValueError(f"unknown filter {method!r}; one of {', '.join(METHODS)}")
        self.method = key
        common = {"n_state": n_state, "n_obs": n_obs, "f": f, "h": h, "Q": Q, "R": R}
        if key == "ekf":
            self.filter = ExtendedKalmanFilter(**common, **options)
        elif key == "ukf":
            self.filter = UnscentedKalmanFilter(**common, **options)
        elif key == "enkf":
            self.filter = EnsembleKalmanFilter(**common, seed=seed, **options)
        else:
            self.filter = ParticleFilter(**common, seed=seed, **options)
        self.n_state = int(n_state)
        self.n_obs = int(n_obs)

    def initialize(self, x0: np.ndarray, P0: np.ndarray | None = None) -> StateEstimator:
        self.filter.initialize(np.asarray(x0, dtype=np.float64), P0)
        return self

    @property
    def state(self) -> np.ndarray:
        """The current estimate: the mean of whichever belief the filter keeps."""
        f = self.filter
        return np.asarray(f.mean if hasattr(f, "mean") and not isinstance(f, UnscentedKalmanFilter) else f.x,
                          dtype=np.float64)

    def step(self, y: np.ndarray) -> dict[str, Any]:
        return self.filter.step(np.asarray(y, dtype=np.float64))

    def run(self, observations: np.ndarray, *, truth: np.ndarray | None = None,
            initialize: bool = False) -> EstimateReport:
        """Filter a whole series. ``truth`` (``(n_steps, n_state)``) turns the report into an error measurement."""
        ys = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        if initialize:
            self.initialize(self.state)
        states, innovations, ess, resampled = [], [], [], []
        for y in ys:
            out = self.filter.step(y)
            states.append(np.asarray(out["x"], dtype=np.float64))
            innovations.append(np.asarray(out.get("innovation", np.full(self.n_obs, np.nan)), dtype=np.float64))
            if "ess" in out:
                ess.append(float(out["ess"]))
                resampled.append(bool(np.asarray(out["resampled"]).ravel()[0]))
        report = EstimateReport(method=self.method, states=np.array(states), innovations=np.array(innovations),
                                ess=np.array(ess) if ess else None,
                                resampled=np.array(resampled) if resampled else None)
        if truth is not None:
            errors = np.asarray(truth, dtype=np.float64)[: len(states)] - report.states
            report.errors = errors
            report.rmse = np.sqrt(np.mean(errors ** 2, axis=0))
            report.max_abs = np.max(np.abs(errors), axis=0)
        return report
