"""The twin's estimator: an ensemble Kalman filter over the state the model cannot know.

The baseline model predicts the world as designed. The twin's job is the other half: find
the value of the *latent* state — the fouling resistance, the wall thickness, an instrument
bias — for which the model reproduces what the sensors are actually reading. That value is
what an operator can act on, and it is never in the model.

Model inputs
------------
The model is evaluated as ``model(X, theta)`` where each row of ``X`` is
``[coordinate…, driver…]``: the sensor's fixed position, then the driver values of the
current frame (inlet temperature, mass flow, ambient). Carrying the drivers in the input
matrix is what lets one twin cover an operating envelope instead of one steady point. A
driver missing from a frame is forward-filled and counted in :attr:`LatentStateTwin.n_driver_gaps`,
because silently reusing a stale flow would bias the estimate without a trace.

The filter is the platform's :class:`~pinneapple_analysis.state_estimation.kalman.EnsembleKalmanFilter`
with a persistence transition and the model itself as the observation operator, so every
assimilated sensor contributes one equation and the latent state is over-determined when
there are more sensors than unknowns — the condition for an estimate to mean anything.

Uncertainty
-----------
The 95 % band comes from the *ensemble spread* around each predicted sensor value. It is a
band on the twin's state, not a prediction interval for a change of regime: if the plant
starts failing in a way the model cannot express, the band stays tight and wrong. Only the
held-out coverage in the backtest detects that, which is why the spec requires a holdout.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Mapping, Sequence

import numpy as np

from pinneapple_analysis.state_estimation.kalman import EnsembleKalmanFilter

from .model import ensure_twin_model
from .spec import SensorSpec, TwinSpec

__all__ = ["TwinEstimate", "LatentStateTwin"]

Z95 = 1.959963984540054


@dataclass
class TwinEstimate:
    """One analysis instant: the latent state, its spread, and the innovation."""

    theta: np.ndarray
    theta_std: np.ndarray
    innovation: np.ndarray
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta": self.theta.tolist(),
            "theta_std": self.theta_std.tolist(),
            "innovation": self.innovation.tolist(),
            "n_obs": self.n_obs,
        }


class LatentStateTwin:
    """Estimates the latent state of one asset from a spec and the model behind it.

    Parameters
    ----------
    model :
        ``model(X, theta) -> {field: array}``. A model that takes only ``X`` is wrapped in
        :class:`~.model.BiasModel`; see :func:`~.model.ensure_twin_model`.
    spec :
        The :class:`~.spec.TwinSpec`. It supplies the sensor layout, the units, the sensor
        noise used for ``R``, which sensors are held out, and which are drivers.
    n_ens, inflation, seed :
        Passed to the filter. The seed makes a commissioning run reproducible.

    Notes
    -----
    Latent dimension: one entry per field declared ``kind="parameter"`` when the spec has
    any, otherwise one bias per predicted field. The two are different instruments — a
    parameter is a physical quantity you can report; a bias absorbs everything, including
    model error, and hides it.
    """

    def __init__(
        self,
        model: Callable[..., Any],
        spec: TwinSpec,
        *,
        n_ens: int = 64,
        inflation: float = 1.02,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        # The model predicts the measurable fields; parameters are its *inputs*, estimated
        # rather than returned, so they never appear in the output contract.
        self.field_names = [f.name for f in spec.fields if f.kind != "parameter"]

        params = [f.name for f in spec.fields if f.kind == "parameter"]
        self.latent_names = list(params) if params else list(self.field_names)
        self.n_latent = len(self.latent_names)

        self.model = ensure_twin_model(model, self.field_names, self.n_latent)

        self.obs_sensors: tuple[SensorSpec, ...] = spec.assimilated_sensors()
        self.holdout_sensors: tuple[SensorSpec, ...] = spec.holdout_sensors()
        self.driver_sensors: tuple[SensorSpec, ...] = spec.driver_sensors
        if not self.obs_sensors:
            raise ValueError(
                "every sensor is a holdout: the twin would have nothing to assimilate. "
                "Mark at least one sensor as observed."
            )

        self._coords_obs = np.array([s.at for s in self.obs_sensors], dtype=np.float32).reshape(
            len(self.obs_sensors), -1
        )
        self._coords_hold = np.array(
            [s.at for s in self.holdout_sensors], dtype=np.float32
        ).reshape(len(self.holdout_sensors), -1)

        self._driver_values: np.ndarray | None = None
        self.n_driver_gaps = 0

        obs_var = np.array(
            [
                (s.noise_sigma if s.noise_sigma is not None else self._default_sigma(s)) ** 2
                for s in self.obs_sensors
            ],
            dtype=np.float64,
        )
        prior = self._prior_variances()
        self._prior = prior
        # Process noise: the state may drift a little between frames, not jump its range.
        Q = np.diag(prior * 1e-4)
        self.filter = EnsembleKalmanFilter(
            n_state=self.n_latent,
            n_obs=len(self.obs_sensors),
            f=lambda x: x,                      # persistence: the state is piecewise constant
            h=self._observe,
            Q=Q,
            R=np.diag(obs_var),
            n_ens=n_ens,
            inflation=inflation,
            seed=seed,
        )
        self._R_full = np.diag(obs_var).copy()
        self._X_obs: np.ndarray | None = None
        # Latent-trajectory log, kept for the projection: a twin that can only say "now" cannot
        # warn anybody in time.
        self._log: Deque[tuple[float, np.ndarray]] = deque(maxlen=240)
        self.reset()

    # ------------------------------------------------------------------
    # prior from the spec
    # ------------------------------------------------------------------

    def _default_sigma(self, s: SensorSpec) -> float:
        """Noise prior when the spec does not declare one: 0.5 % of the sensor envelope."""
        return max(1e-9, 0.005 * (s.hi - s.lo))

    def _prior_variances(self) -> np.ndarray:
        """Prior variance **per latent entry**, from that entry's own declared range.

        Scaling every latent entry by the widest field in the spec is the mistake that
        silently destroys a filter: a fouling resistance declared over 0–0.0025 m2K/W would
        be given a prior standard deviation of 4 in its own units, 2400× its range, and the
        ensemble would wander until the model's sensitivity, not the data, set the answer.
        A parameter gets a quarter of its range; a bias gets 2 % of the field span.
        """
        by_name = {f.name: f for f in self.spec.fields}
        out = np.empty(self.n_latent, dtype=np.float64)
        for i, name in enumerate(self.latent_names):
            f = by_name.get(name)
            if f is None or not f.span:
                out[i] = 1e-6
            elif f.kind == "parameter":
                out[i] = (0.25 * f.span) ** 2
            else:
                out[i] = (0.02 * f.span) ** 2
        return out

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------

    def _take_drivers(self, observations: Mapping[str, float]) -> np.ndarray:
        """Current driver values, forward-filled when a channel is missing from the frame."""
        values = np.zeros(len(self.driver_sensors), dtype=np.float32)
        prev = self._driver_values
        for i, s in enumerate(self.driver_sensors):
            v = observations.get(s.sensor_id)
            if v is None:
                self.n_driver_gaps += 1
                values[i] = prev[i] if prev is not None else 0.0
            else:
                values[i] = float(v)
        self._driver_values = values
        return values

    def _inputs(self, coords: np.ndarray, drivers: np.ndarray) -> np.ndarray:
        """Assemble the model input matrix: coordinates then drivers, per row."""
        n = coords.shape[0]
        if drivers.size == 0:
            return coords
        drv = np.tile(drivers, (n, 1))
        return np.hstack([coords, drv]).astype(np.float32)

    # ------------------------------------------------------------------
    # observation operator
    # ------------------------------------------------------------------

    def _observe(self, theta: np.ndarray) -> np.ndarray:
        """Model prediction at every assimilated sensor, as a 1-D observation vector."""
        if self._X_obs is None:
            self._refresh_inputs(np.zeros(len(self.driver_sensors), dtype=np.float32))
        pred = self.model(self._X_obs, theta)  # type: ignore[arg-type]
        return np.array([pred[s.field][i] for i, s in enumerate(self.obs_sensors)], dtype=np.float64)

    def _refresh_inputs(self, drivers: np.ndarray) -> None:
        self._X_obs = self._inputs(self._coords_obs, drivers)

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------

    def reset(self, theta0: Sequence[float] | None = None) -> None:
        """Restart the filter. ``theta0`` defaults to zero for each latent entry."""
        x0 = np.asarray(theta0, dtype=np.float64) if theta0 is not None else np.zeros(self.n_latent)
        if x0.size != self.n_latent:
            raise ValueError(f"theta0 must have {self.n_latent} entries, got {x0.size}")
        self.filter.initialize(x0, np.diag(self._prior))
        self._driver_values = None
        self.n_driver_gaps = 0
        self._log.clear()

    def step(self, observations: Mapping[str, float], *, t: float | None = None) -> TwinEstimate:
        """Assimilate one instant.

        ``observations`` maps ``sensor_id -> value``. Sensors absent from the dict are
        skipped, which keeps the filter usable with a partial frame; the innovation is only
        over the sensors actually supplied. A missing driver is forward-filled and counted.

        ``t`` is the frame timestamp. Supply it: without it the projection cannot know the
        sample interval and falls back to the spec's ``sampling_s``.
        """
        drivers = self._take_drivers(observations)
        self._refresh_inputs(drivers)
        self._log.append((float(t) if t is not None else float(len(self._log)), self.filter.mean.copy()))

        present_idx = [i for i, s in enumerate(self.obs_sensors) if s.sensor_id in observations]
        if not present_idx:
            return TwinEstimate(
                theta=self.filter.mean.copy(),
                theta_std=self.filter.ensemble.std(axis=1, ddof=1),
                innovation=np.zeros(0),
                n_obs=0,
            )

        y = np.array(
            [observations[self.obs_sensors[i].sensor_id] for i in present_idx], dtype=np.float64
        )

        if len(present_idx) == len(self.obs_sensors):
            out = self.filter.step(y)
        else:
            saved = (self.filter.h, self.filter.R, self.filter.n_obs)
            self.filter.h = lambda th, _idx=present_idx: self._observe(th)[_idx]
            self.filter.R = self._R_full[np.ix_(present_idx, present_idx)]
            self.filter.n_obs = len(present_idx)
            try:
                out = self.filter.step(y)
            finally:
                self.filter.h, self.filter.R, self.filter.n_obs = saved

        return TwinEstimate(
            theta=np.asarray(out["x"], dtype=np.float64),
            theta_std=self.filter.ensemble.std(axis=1, ddof=1),
            innovation=np.asarray(out["innovation"], dtype=np.float64),
            n_obs=len(present_idx),
        )

    # ------------------------------------------------------------------
    # reading it out
    # ------------------------------------------------------------------

    def predict_fields(
        self,
        X: np.ndarray,
        *,
        theta: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Best estimate and ensemble spread of each field at the input rows ``X``."""
        theta = self.filter.mean if theta is None else np.asarray(theta, dtype=np.float64)
        X = np.asarray(X, dtype=np.float32)
        pred = self.model(X, theta)
        mean = {f: np.asarray(v, dtype=np.float64) for f, v in pred.items()}

        spread = {f: np.zeros_like(mean[f]) for f in mean}
        for i in range(self.filter.n_ens):
            member = self.model(X, self.filter.ensemble[:, i])
            for f in mean:
                spread[f] += (np.asarray(member[f], dtype=np.float64) - mean[f]) ** 2
        for f in spread:
            spread[f] = np.sqrt(spread[f] / max(1, self.filter.n_ens - 1))
        return mean, spread

    def _holdout_inputs(self) -> np.ndarray:
        drivers = self._driver_values
        if drivers is None:
            drivers = np.zeros(len(self.driver_sensors), dtype=np.float32)
        return self._inputs(self._coords_hold, drivers)

    def band_at_holdout(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicted value and 95 % band at every held-out sensor, as three (n_hold,) arrays.

        The band is ``pred ± Z95 · ensemble spread``. It describes the twin's confidence in
        its own state, not the plant's future — see the module docstring.
        """
        if not self.holdout_sensors:
            empty = np.zeros(0)
            return empty, empty, empty
        mean, spread = self.predict_fields(self._holdout_inputs())
        values = np.array([mean[s.field][i] for i, s in enumerate(self.holdout_sensors)])
        state_sigma = np.array([spread[s.field][i] for i, s in enumerate(self.holdout_sensors)])
        # The band has to be a *predictive* interval for a reading, so it carries the
        # instrument noise as well as the state uncertainty. Leaving the noise out narrows the
        # band by a factor of ~√2 on a clean channel and makes coverage fail for a reason that
        # has nothing to do with the twin being wrong.
        obs_sigma = np.array([self._sigma_of(s) for s in self.holdout_sensors])
        sigmas = np.sqrt(state_sigma ** 2 + obs_sigma ** 2)
        return values, values - Z95 * sigmas, values + Z95 * sigmas

    def _sigma_of(self, s: SensorSpec) -> float:
        return s.noise_sigma if s.noise_sigma is not None else self._default_sigma(s)

    def mean_at_holdout(self) -> dict[str, float]:
        """The twin's field estimate at each held-out sensor, keyed by sensor id."""
        if not self.holdout_sensors:
            return {}
        mean, _ = self.predict_fields(self._holdout_inputs())
        return {s.sensor_id: float(mean[s.field][i]) for i, s in enumerate(self.holdout_sensors)}

    def latent_trend(self) -> np.ndarray | None:
        """Least-squares drift of the latent state per second, or ``None`` with too short a log."""
        if len(self._log) < 6:
            return None
        ts = np.array([t for t, _ in self._log], dtype=float)
        span = ts[-1] - ts[0]
        if span <= 0:
            return None
        thetas = np.column_stack([th for _, th in self._log])
        slope = np.array([np.polyfit(ts, thetas[i], 1)[0] for i in range(self.n_latent)])
        return slope

    def project_crossing(
        self,
        field_name: str,
        limit: float,
        *,
        above: bool = True,
        horizon_s: float,
        max_steps: int = 400,
    ) -> float | None:
        """Seconds until the field is expected to cross ``limit``, or ``None`` if not within ``horizon_s``.

        The latent state is extrapolated linearly from the logged trajectory and the model is
        re-evaluated at the current operating point with each extrapolated value. This is a
        *trend* projection, not a simulation: it cannot anticipate a change of regime, a
        shutdown or a cleaning, which is why the forecast error is measured and reported
        instead of the projection being presented as a prediction.
        """
        slope = self.latent_trend()
        sensors = [s for s in self.holdout_sensors if s.field == field_name] or [
            s for s in self.obs_sensors if s.field == field_name
        ]
        if slope is None or not sensors:
            return None

        if len(self._log) >= 2:
            dt = float(np.median(np.diff([t for t, _ in self._log])))
        else:
            dt = float(self.spec.sampling_s)
        if dt <= 0:
            dt = float(self.spec.sampling_s)

        coords = np.array([s.at for s in sensors], dtype=np.float32).reshape(len(sensors), -1)
        drivers = self._driver_values
        if drivers is None:
            drivers = np.zeros(len(self.driver_sensors), dtype=np.float32)
        X = self._inputs(coords, drivers)
        theta0 = self.filter.mean

        for k in range(1, max_steps + 1):
            ahead = k * dt
            if ahead > horizon_s:
                return None
            pred = self.model(X, theta0 + slope * ahead)
            value = float(np.asarray(pred[field_name], dtype=float).mean())
            if (above and value >= limit) or (not above and value <= limit):
                return ahead
        return None

    def latent_dict(self) -> dict[str, dict[str, float | str]]:
        """Latent state with units, ready to be published to AAS/DTDL/NGSI-LD."""
        mean = self.filter.mean
        std = self.filter.ensemble.std(axis=1, ddof=1)
        units = {f.name: f.unit for f in self.spec.fields}
        out: dict[str, dict[str, float | str]] = {}
        for i, name in enumerate(self.latent_names):
            out[name] = {
                "value": float(mean[i]),
                "std": float(std[i]),
                "unit": units.get(name, ""),
            }
        return out
