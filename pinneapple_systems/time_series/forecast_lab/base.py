"""Common interface of the forecasters in :mod:`pinneapple_systems.time_series.forecast_lab`.

A forecaster sees a :class:`SeriesData` (the observed array, which column is the target, the sampling step) and
makes direct multi-horizon forecasts from *origins*: a forecast from origin ``t`` may use ``obs[: t + 1]`` only and
predicts the target at ``t + h`` for every ``h`` in ``horizons``.

``fit(data, train_origins, val_origins, horizons)`` receives the origins of the training samples and of the
validation block. The labels of a training origin reach ``t + max(horizons)``, so the data a forecaster may use for
fitting end at ``train_end(train_origins, horizons)``; models fitted on a contiguous segment (physical models,
ARIMA, filters) use exactly that segment, window models use the samples. The validation block is used for early
stopping and to calibrate the predictive spread (:meth:`Forecaster.calibrate`): a point forecaster gets a Gaussian
predictive distribution whose standard deviation at each horizon is the root-mean-square error on the validation
block. Nothing after the validation block is ever passed to ``fit``.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SeriesData", "Forecast", "Forecaster", "train_end", "forecast_origins", "targets_at"]


@dataclass
class SeriesData:
    """Observed series (T, d), the forecast target column, the sampling step and free-form metadata."""
    obs: np.ndarray
    target: int = 0
    dt: float = 1.0
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.obs = np.asarray(self.obs, float)
        if self.obs.ndim == 1:
            self.obs = self.obs[:, None]

    @property
    def y(self) -> np.ndarray:
        return self.obs[:, self.target]

    def __len__(self) -> int:
        return len(self.obs)


def train_end(train_origins, horizons) -> int:
    """Exclusive end index of the data that a fit on these origins may use."""
    return int(np.max(train_origins)) + int(np.max(horizons)) + 1


def forecast_origins(lo: int, hi: int, horizons, *, stride: int = 1, min_history: int = 0) -> np.ndarray:
    """Origins t with lo <= t, t >= min_history - 1 and t + max(h) < hi, every ``stride`` samples."""
    start = max(int(lo), int(min_history) - 1, 0)
    stop = int(hi) - int(np.max(horizons))
    return np.arange(start, stop, int(stride))


def targets_at(data: SeriesData, origins, horizons) -> np.ndarray:
    o = np.asarray(origins)[:, None]
    return data.y[o + np.asarray(horizons)[None, :]]


@dataclass
class Forecast:
    """Predictive distribution at (origin, horizon): a mean, and either a Gaussian ``std`` or ``samples``
    (members, origins, horizons)."""
    mean: np.ndarray
    std: np.ndarray | None = None
    samples: np.ndarray | None = None

    def crps(self, y) -> np.ndarray:
        from pinneapple_analysis.uncertainty.posterior_metrics import crps_ensemble, crps_gaussian
        if self.samples is not None:
            return crps_ensemble(self.samples, y, fair=True)
        if self.std is not None:
            return crps_gaussian(self.mean, self.std, y)
        return np.abs(np.asarray(y) - self.mean)

    def pit(self, y, rng=None) -> np.ndarray | None:
        from ..metrics_ext.probabilistic import pit_ensemble, pit_gaussian
        if self.samples is not None:
            return pit_ensemble(self.samples, y, rng)
        if self.std is not None:
            return pit_gaussian(self.mean, self.std, y)
        return None

    def coverage(self, y, level: float) -> np.ndarray | None:
        """Per-horizon fraction of targets inside the central ``level`` interval."""
        from scipy.stats import norm
        y = np.asarray(y)
        if self.samples is not None:
            lo, hi = np.quantile(self.samples, [0.5 - level / 2, 0.5 + level / 2], axis=0)
        elif self.std is not None:
            z = norm.ppf(0.5 + level / 2)
            lo, hi = self.mean - z * self.std, self.mean + z * self.std
        else:
            return None
        return np.mean((y >= lo) & (y <= hi), axis=0)

    def quantiles(self, levels) -> np.ndarray:
        from scipy.stats import norm
        if self.samples is not None:
            return np.quantile(self.samples, levels, axis=0)
        std = self.std if self.std is not None else np.zeros_like(self.mean)
        return np.stack([self.mean + norm.ppf(q) * std for q in levels])


class Forecaster(ABC):
    """Base class: subclasses implement :meth:`_fit` and :meth:`_predict_mean` (or override :meth:`predict`)."""

    name: str = "forecaster"
    family: str = "baseline"
    probabilistic: bool = False   # True when predict() returns its own samples / std

    def __init__(self):
        self.horizons: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None
        self.fit_seconds: float = float("nan")
        self.info: dict[str, Any] = {}

    # -- public -------------------------------------------------------------
    def fit(self, data: SeriesData, train_origins, val_origins=None, horizons: Sequence[int] = (1,)
            ) -> Forecaster:
        t0 = time.perf_counter()
        self.horizons = np.asarray(horizons, int)
        train_origins = np.asarray(train_origins, int)
        val_origins = None if val_origins is None or len(val_origins) == 0 else np.asarray(val_origins, int)
        self._fit(data, train_origins, val_origins)
        self.calibrate(data, val_origins if val_origins is not None else train_origins)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict(self, data: SeriesData, origins) -> Forecast:
        origins = np.asarray(origins, int)
        mean = self._predict_mean(data, origins)
        std = None if self.sigma_ is None else np.broadcast_to(self.sigma_, mean.shape).copy()
        return Forecast(mean, std)

    def calibrate(self, data: SeriesData, origins) -> None:
        """Gaussian spread per horizon = RMSE on ``origins`` (the validation block when there is one)."""
        if self.probabilistic:
            return
        origins = np.asarray(origins, int)
        if len(origins) > 1500:
            origins = origins[:: int(np.ceil(len(origins) / 1500))]
        mean = self._predict_mean(data, origins)
        err = targets_at(data, origins, self.horizons) - mean
        self.sigma_ = np.sqrt(np.nanmean(err ** 2, axis=0))

    def n_params(self) -> int:
        return 0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "family": self.family, "n_params": int(self.n_params()),
                "fit_seconds": float(self.fit_seconds), **self.info}

    # -- subclasses ---------------------------------------------------------
    @abstractmethod
    def _fit(self, data: SeriesData, train_origins: np.ndarray, val_origins: np.ndarray | None) -> None:
        ...

    def _predict_mean(self, data: SeriesData, origins: np.ndarray) -> np.ndarray:
        raise NotImplementedError
