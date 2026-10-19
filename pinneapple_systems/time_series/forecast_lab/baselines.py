"""Baselines every other forecaster must beat: persistence, climatology, AR(p), ARIMA, exponential smoothing, and
for (near-)periodic series the seasonal-naive forecast and a phase-aligned mean cycle."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..predictability import fit_ar
from .base import Forecaster, train_end

__all__ = ["Persistence", "Climatology", "ARForecaster", "ARIMAForecaster", "ExponentialSmoothing", "dominant_period",
           "SeasonalNaive", "PhaseClimatology"]


class Persistence(Forecaster):
    """y_{t+h} = y_t (the random-walk forecast)."""
    name, family = "persistence", "baseline"

    def _fit(self, data, train_origins, val_origins):
        pass

    def _predict_mean(self, data, origins):
        return np.repeat(data.y[origins][:, None], len(self.horizons), axis=1)


class Climatology(Forecaster):
    """y_{t+h} = mean of the training segment, spread = its standard deviation."""
    name, family = "climatology", "baseline"
    probabilistic = True

    def _fit(self, data, train_origins, val_origins):
        seg = data.y[: train_end(train_origins, self.horizons)]
        self.mu_, self.sd_ = float(seg.mean()), float(seg.std())

    def _predict_mean(self, data, origins):
        return np.full((len(origins), len(self.horizons)), self.mu_)

    def predict(self, data, origins):
        from .base import Forecast
        m = self._predict_mean(data, np.asarray(origins))
        return Forecast(m, np.full_like(m, self.sd_))

    def n_params(self):
        return 2


class ARForecaster(Forecaster):
    """Least-squares AR(p) on the target, order by AIC up to ``max_order``, iterated to each horizon."""
    name, family = "ar", "baseline"

    def __init__(self, max_order: int = 20, criterion: str = "aic", difference: bool = False):
        super().__init__()
        self.max_order, self.criterion, self.difference = max_order, criterion, difference

    def _fit(self, data, train_origins, val_origins):
        seg = data.y[: train_end(train_origins, self.horizons)]
        z = np.diff(seg) if self.difference else seg
        self.ar_ = fit_ar(z, self.max_order, self.criterion)
        self.info = {"order": self.ar_.order, "difference": self.difference}

    def _predict_mean(self, data, origins):
        p, c, phi = self.ar_.order, self.ar_.intercept, self.ar_.coef
        y = data.y
        H = int(self.horizons.max())
        if self.difference:
            z = np.diff(y, prepend=y[0])
        else:
            z = y
        hist = np.stack([z[np.maximum(origins - k, 0)] for k in range(p)], axis=1) if p else np.zeros((len(origins), 0))
        path = np.empty((len(origins), H))
        for h in range(H):
            nxt = c + hist @ phi if p else np.full(len(origins), c)
            path[:, h] = nxt
            if p:
                hist = np.concatenate([nxt[:, None], hist[:, :-1]], axis=1)
        if self.difference:
            path = y[origins][:, None] + np.cumsum(path, axis=1)
        return path[:, self.horizons - 1]

    def n_params(self):
        return self.ar_.order + 2


class ARIMAForecaster(Forecaster):
    """ARIMA(p, d, q) (statsmodels SARIMAX), order by AIC over a small grid on the training segment; forecasts
    from every origin with the fitted parameters fixed, through the Kalman filter state at that origin (so a
    forecast uses data up to its origin only). The Gaussian predictive variance comes from the state space."""
    name, family = "arima", "baseline"
    probabilistic = True

    def __init__(self, p: Sequence[int] = (0, 1, 2), d: Sequence[int] = (0, 1), q: Sequence[int] = (0, 1),
                 max_fit_length: int = 2000):
        super().__init__()
        self.grid = [(a, b, c) for a in p for b in d for c in q]
        self.max_fit_length = max_fit_length

    def _fit(self, data, train_origins, val_origins):
        import warnings

        from statsmodels.tsa.statespace.sarimax import SARIMAX
        seg = data.y[: train_end(train_origins, self.horizons)][-self.max_fit_length:]
        best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for order in self.grid:
                trend = "c" if order[1] == 0 else "n"
                try:
                    res = SARIMAX(seg, order=order, trend=trend).fit(disp=False)
                except Exception:  # noqa: BLE001 -- an order that does not converge is skipped
                    continue
                if np.isfinite(res.aic) and (best is None or res.aic < best[0]):
                    best = (res.aic, order, trend, res.params)
        self.aic_, self.order_, self.trend_, self.params_ = best
        self.info = {"order": list(self.order_), "aic": float(self.aic_)}

    def predict(self, data, origins):
        import warnings

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        from .base import Forecast
        origins = np.asarray(origins, int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(data.y[: int(origins.max()) + 1], order=self.order_, trend=self.trend_)
            res = mod.filter(self.params_)
        fr = res.filter_results
        a = fr.filtered_state[:, origins].T                     # (n, k)
        P = fr.filtered_state_cov[:, :, origins].transpose(2, 0, 1)
        T, Z = fr.transition[:, :, 0], fr.design[0, :, 0]
        c = fr.state_intercept[:, 0]
        RQR = fr.selection[:, :, 0] @ fr.state_cov[:, :, 0] @ fr.selection[:, :, 0].T
        H = int(self.horizons.max())
        mean = np.empty((len(origins), H))
        var = np.empty((len(origins), H))
        for h in range(H):
            a = a @ T.T + c
            P = T @ P @ T.T + RQR
            mean[:, h] = a @ Z + fr.obs_intercept[0, 0]
            var[:, h] = np.einsum("i,nij,j->n", Z, P, Z) + fr.obs_cov[0, 0, 0]
        k = self.horizons - 1
        return Forecast(mean[:, k], np.sqrt(np.maximum(var[:, k], 0)))

    def _predict_mean(self, data, origins):
        return self.predict(data, origins).mean

    def n_params(self):
        return int(len(self.params_))


class ExponentialSmoothing(Forecaster):
    """Simple exponential smoothing l_t = alpha y_t + (1 - alpha) l_{t-1}, flat forecast l_t; alpha minimises the
    one-step squared error on the training segment."""
    name, family = "exp_smoothing", "baseline"

    def _levels(self, y, alpha):
        lev = np.empty_like(y)
        cur = y[0]
        for i, v in enumerate(y):
            cur = alpha * v + (1 - alpha) * cur
            lev[i] = cur
        return lev

    def _fit(self, data, train_origins, val_origins):
        from scipy.optimize import minimize_scalar
        seg = data.y[: train_end(train_origins, self.horizons)]

        def sse(a):
            lev = self._levels(seg, a)
            return float(np.sum((seg[1:] - lev[:-1]) ** 2))

        self.alpha_ = float(minimize_scalar(sse, bounds=(1e-3, 1.0), method="bounded").x)
        self.info = {"alpha": self.alpha_}

    def _predict_mean(self, data, origins):
        lev = self._levels(data.y[: int(np.max(origins)) + 1], self.alpha_)
        return np.repeat(lev[origins][:, None], len(self.horizons), axis=1)

    def n_params(self):
        return 1


def dominant_period(y, min_period: float = 2.0, max_frac: float = 0.25) -> float:
    """Period (in samples) of the highest periodogram peak of ``y`` after removing the mean and a linear trend,
    searched between ``min_period`` samples and ``max_frac`` of the series length."""
    y = np.asarray(y, float)
    t = np.arange(len(y))
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    f = np.fft.rfftfreq(len(y))
    p = np.abs(np.fft.rfft(y)) ** 2
    ok = (f > 1.0 / (max_frac * len(y))) & (f <= 1.0 / min_period)
    if not ok.any():
        raise ValueError("series too short to estimate a period")
    k = np.nonzero(ok)[0][np.argmax(p[ok])]
    if 0 < k < len(p) - 1:                       # parabolic interpolation of the peak in log power
        a, b, c = np.log(p[k - 1:k + 2] + 1e-300)
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return float(1.0 / (k / len(y)))


class SeasonalNaive(Forecaster):
    """y_{t+h} = y_{t+h-kP} with k = ceil(h / P): the value one period (or the fewest whole periods) before the
    target. ``period`` in samples; by default the dominant period of the training segment (rounded to an integer
    number of samples)."""
    name, family = "seasonal_naive", "baseline"

    def __init__(self, period: float | None = None):
        super().__init__()
        self.period = period

    def _fit(self, data, train_origins, val_origins):
        P = self.period if self.period is not None else dominant_period(data.y[: train_end(train_origins,
                                                                                            self.horizons)])
        self.P_ = max(1, int(round(P)))
        self.info = {"period": self.P_, "period_estimated": self.period is None}

    def _predict_mean(self, data, origins):
        h = self.horizons[None, :]
        k = np.ceil(h / self.P_).astype(int)
        idx = np.asarray(origins)[:, None] + h - k * self.P_
        return data.y[np.maximum(idx, 0)]

    def n_params(self):
        return 1


class PhaseClimatology(Forecaster):
    """The mean cycle, aligned to the phase of each origin.

    The training segment is folded at the (possibly non-integer) period P into ``round(P)`` phase bins to give a
    mean cycle. At each origin the phase is re-estimated: the cycle offset that best fits the last
    ``fit_cycles`` periods of observations in least squares (with a free mean) is taken, and the forecast is the
    mean cycle continued from there. A near-periodic series whose phase drifts is thus followed; the amplitude and
    shape are those of the average training cycle.
    """
    name, family = "phase_climatology", "baseline"

    def __init__(self, period: float | None = None, fit_cycles: float = 1.0):
        super().__init__()
        self.period, self.fit_cycles = period, float(fit_cycles)

    def _fit(self, data, train_origins, val_origins):
        seg = data.y[: train_end(train_origins, self.horizons)]
        self.P_ = float(self.period if self.period is not None else dominant_period(seg))
        nb = max(2, int(round(self.P_)))
        bins = np.floor((np.arange(len(seg)) % self.P_) / self.P_ * nb).astype(int) % nb
        cyc = np.array([seg[bins == b].mean() if np.any(bins == b) else seg.mean() for b in range(nb)])
        self.mean_ = float(seg.mean())
        self.cycle_ = cyc - cyc.mean()
        self.nb_ = nb
        self.info = {"period": self.P_, "period_estimated": self.period is None, "n_bins": nb}

    def _template(self, phase):
        """Mean cycle at fractional phase (in cycles), periodic linear interpolation."""
        x = (np.asarray(phase) % 1.0) * self.nb_
        i0 = np.floor(x).astype(int) % self.nb_
        w = x - np.floor(x)
        return (1 - w) * self.cycle_[i0] + w * self.cycle_[(i0 + 1) % self.nb_]

    def _predict_mean(self, data, origins):
        origins = np.asarray(origins)
        L = max(2, int(round(self.fit_cycles * self.P_)))
        back = np.arange(-L + 1, 1)                               # last L samples, oldest first
        idx = np.maximum(origins[:, None] + back[None, :], 0)
        obs = data.y[idx]                                         # (n, L)
        offs = np.linspace(0, 1, 4 * self.nb_, endpoint=False)    # candidate phases of the origin
        T = self._template(offs[:, None] + back[None, :] / self.P_)   # (n_off, L)
        Tc = T - T.mean(axis=1, keepdims=True)
        oc = obs - obs.mean(axis=1, keepdims=True)
        sse = (oc ** 2).sum(1)[:, None] - 2 * oc @ Tc.T + (Tc ** 2).sum(1)[None, :]
        best = offs[np.argmin(sse, axis=1)]                       # (n,)
        level = obs.mean(axis=1) - T[np.argmin(sse, axis=1)].mean(axis=1)
        return level[:, None] + self._template(best[:, None] + self.horizons[None, :] / self.P_)

    def n_params(self):
        return self.nb_ + 1
