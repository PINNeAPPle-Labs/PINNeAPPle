"""Physical forecasters: the governing equations with parameters estimated from the training data only.

* :class:`ODEForecaster` -- an ODE whose state is fully observed; parameters by least squares on short
  (multiple-shooting) predictions from the observed training states; forecasts integrate from the observed state
  at the origin. Pass a wrong vector field to measure a misspecified model.
* :class:`DelayODEForecaster` -- a delay differential equation dx/dt = f(x, x(t - tau)) of a scalar series
  (Mackey-Glass), integrated by the method of steps from the interpolated observed history.
* :class:`LinearSSMForecaster` -- linear stochastic models sampled exactly: the Ornstein-Uhlenbeck process and the
  noise-driven damped oscillator, with observation noise. Parameters by maximum likelihood through the Kalman
  filter; forecasts are the exact Gaussian predictive distribution from the filtered state at the origin (data
  assimilation of a linear model).
* :class:`SDEDriftForecaster` -- dx = f(x) dt + sigma dW with f in a polynomial basis, estimated by regressing the
  increments on the basis (first Kramers-Moyal coefficient; Friedrich et al. 2011, Phys. Rep. 506:87);
  Euler-Maruyama ensemble forecasts. An optional neural closure of the drift makes it a hybrid.
* :class:`EnKFForecaster` -- ensemble Kalman filter (:class:`pinneapple_analysis.state_estimation.kalman.
  EnsembleKalmanFilter`) with a fitted :class:`ODEForecaster` as the model; forecast ensembles start from the
  analysis ensemble at each origin.
* :class:`SINDyForecaster` -- equations discovered by sparse regression (:class:`pinneapple_neural.architectures.
  rom.sindy.SINDy`) on the training trajectory.

Forecasts that leave the range of the training data by more than ``divergence_factor`` training standard
deviations (or are not finite) are counted in ``info["diverged_fraction"]`` and replaced by the training mean; the
fraction is reported with the scores, so a model that blows up is visible rather than silently clipped.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..synthetic import oscillator_transition, rk4_step
from .base import Forecast, Forecaster, SeriesData, train_end

__all__ = ["ODEForecaster", "DelayODEForecaster", "LinearSSMForecaster", "SDEDriftForecaster", "EnKFForecaster",
           "SINDyForecaster", "PhysicsFeatures"]


def _sample_origins(origins, k, seed=0):
    origins = np.asarray(origins)
    if len(origins) <= k:
        return origins
    return np.sort(np.random.default_rng(seed).choice(origins, k, replace=False))


class _DivergenceGuard:
    def _guard_fit(self, seg: np.ndarray, factor: float):
        self.g_mu_, self.g_sd_ = seg.mean(axis=0), seg.std(axis=0) + 1e-12
        self.g_lo_ = seg.min(axis=0) - factor * self.g_sd_
        self.g_hi_ = seg.max(axis=0) + factor * self.g_sd_

    def _guard(self, pred: np.ndarray, col: int) -> np.ndarray:
        bad = ~np.isfinite(pred) | (pred < self.g_lo_[col]) | (pred > self.g_hi_[col])
        self.info["diverged_fraction"] = float(bad.mean())
        out = pred.copy()
        out[bad] = self.g_mu_[col]
        return out


class ODEForecaster(_DivergenceGuard, Forecaster):
    """dx/dt = rhs(x, **params) with every state variable observed (the columns of ``data.obs``).

    ``params`` are the initial guesses; the names in ``fit_params`` are estimated (the rest stay fixed) by
    minimising the standardised error of RK4 predictions ``fit_steps`` samples ahead from the observed states at
    up to ``max_fit_origins`` training origins.
    """
    name, family = "ode_fitted", "physical"

    def __init__(self, rhs: Callable, params: dict[str, float], fit_params: Sequence[str] | None = None,
                 sub: int = 5, fit_steps: int = 2, max_fit_origins: int = 1500, divergence_factor: float = 5.0,
                 name: str | None = None, family: str | None = None):
        super().__init__()
        self.rhs, self.params0 = rhs, dict(params)
        self.fit_params = list(fit_params) if fit_params is not None else list(params)
        self.sub, self.fit_steps, self.max_fit_origins = int(sub), int(fit_steps), int(max_fit_origins)
        self.divergence_factor = divergence_factor
        self.name, self.family = name or self.name, family or self.family
        self.params_ = dict(params)

    def rhs_numpy(self, x, params=None):
        return self.rhs(x, **(params or self.params_))

    def torch_rhs(self):
        p = {k: float(v) for k, v in self.params_.items()}
        return lambda x: self.rhs(x, **p)

    def integrate(self, x0: np.ndarray, n_samples: int, dt: float, params=None) -> np.ndarray:
        """States after 1..n_samples sampling intervals from x0 (B, d): returns (B, n_samples, d)."""
        p = params or self.params_
        f = lambda z: self.rhs(z, **p)  # noqa: E731
        h = dt / self.sub
        x = np.array(x0, float)
        out = np.empty((x.shape[0], n_samples, x.shape[1]))
        with np.errstate(all="ignore"):
            for k in range(n_samples):
                for _ in range(self.sub):
                    x = rk4_step(f, x, h)
                out[:, k] = x
        return out

    def _fit(self, data, train_origins, val_origins):
        from scipy.optimize import least_squares
        self.d_, self.dt_ = data.obs.shape[1], float(data.dt)
        end = train_end(train_origins, self.horizons)
        seg = data.obs[:end]
        self._guard_fit(seg, self.divergence_factor)
        o = _sample_origins(np.arange(0, end - self.fit_steps), self.max_fit_origins)
        x0 = data.obs[o]
        target = np.stack([data.obs[o + k] for k in range(1, self.fit_steps + 1)], axis=1)
        sd = seg.std(axis=0)

        def resid(theta):
            p = dict(self.params0)
            p.update(dict(zip(self.fit_params, theta, strict=False)))
            pred = self.integrate(x0, self.fit_steps, self.dt_, p)
            r = ((pred - target) / sd).ravel()
            return np.where(np.isfinite(r), r, 1e3)

        theta0 = np.array([self.params0[k] for k in self.fit_params], float)
        sol = least_squares(resid, theta0, x_scale=np.maximum(np.abs(theta0), 1e-3), max_nfev=200)
        self.params_ = dict(self.params0)
        self.params_.update({k: float(v) for k, v in zip(self.fit_params, sol.x, strict=False)})
        r = resid(sol.x).reshape(len(o), self.fit_steps, self.d_)[:, 0] * sd
        self.one_step_rmse_ = np.sqrt(np.mean(r ** 2, axis=0))
        self.info = {"params": self.params_, "one_step_rmse": self.one_step_rmse_.tolist()}

    def _predict_mean(self, data, origins):
        H = int(self.horizons.max())
        path = self.integrate(data.obs[origins], H, self.dt_)
        return self._guard(path[:, self.horizons - 1, data.target], data.target)

    def physics_features(self, data, origins):
        """The fitted vector field at the observed state at the origin, times dt."""
        return self.rhs_numpy(data.obs[np.asarray(origins)]) * self.dt_

    def n_params(self):
        return len(self.fit_params)


class DelayODEForecaster(_DivergenceGuard, Forecaster):
    """dx/dt = rhs(x, x(t - tau), **params) for a scalar series sampled every dt; integrated with RK4 at
    ``dt_int`` (dt and tau must be multiples of it), the history taken from the observations by linear
    interpolation."""
    name, family = "dde_fitted", "physical"

    def __init__(self, rhs: Callable, params: dict[str, float], tau: float, fit_params: Sequence[str] | None = None,
                 dt_int: float = 0.1, fit_steps: int = 2, max_fit_origins: int = 1500,
                 divergence_factor: float = 5.0, name: str | None = None, family: str | None = None):
        super().__init__()
        self.rhs, self.params0, self.tau = rhs, dict(params), float(tau)
        self.fit_params = list(fit_params) if fit_params is not None else list(params)
        self.dt_int, self.fit_steps, self.max_fit_origins = float(dt_int), int(fit_steps), int(max_fit_origins)
        self.divergence_factor = divergence_factor
        self.name, self.family = name or self.name, family or self.family
        self.params_ = dict(params)

    def _history(self, y, origins, dt):
        m = int(round(self.tau / self.dt_int))
        back = int(np.ceil(self.tau / dt)) + 1
        tt = -np.arange(m, -1, -1) * self.dt_int                      # fine grid, relative to the origin
        coarse = -np.arange(back, -1, -1) * dt
        idx = np.asarray(origins)[:, None] - np.arange(back, -1, -1)[None]
        vals = y[np.maximum(idx, 0)]
        return np.stack([np.interp(tt, coarse, v) for v in vals])     # (B, m + 1)

    def _step(self, hist, p):
        h = self.dt_int
        f = lambda x, xt: self.rhs(x, xt, **p)  # noqa: E731
        x, a, b = hist[:, -1], hist[:, 0], hist[:, 1]
        mid = 0.5 * (a + b)
        k1 = f(x, a)
        k2 = f(x + 0.5 * h * k1, mid)
        k3 = f(x + 0.5 * h * k2, mid)
        k4 = f(x + h * k3, b)
        new = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return np.concatenate([hist[:, 1:], new[:, None]], axis=1)

    def integrate(self, hist, n_samples, dt, params=None):
        p = params or self.params_
        sub = int(round(dt / self.dt_int))
        out = np.empty((hist.shape[0], n_samples))
        with np.errstate(all="ignore"):
            for k in range(n_samples):
                for _ in range(sub):
                    hist = self._step(hist, p)
                out[:, k] = hist[:, -1]
        return out

    def _fit(self, data, train_origins, val_origins):
        from scipy.optimize import least_squares
        self.dt_ = float(data.dt)
        end = train_end(train_origins, self.horizons)
        y = data.y
        self._guard_fit(data.obs[:end], self.divergence_factor)
        back = int(np.ceil(self.tau / self.dt_)) + 1
        o = _sample_origins(np.arange(back, end - self.fit_steps), self.max_fit_origins)
        hist = self._history(y, o, self.dt_)
        target = np.stack([y[o + k] for k in range(1, self.fit_steps + 1)], axis=1)
        sd = y[:end].std()

        def resid(theta):
            p = dict(self.params0)
            p.update(dict(zip(self.fit_params, theta, strict=False)))
            r = ((self.integrate(hist, self.fit_steps, self.dt_, p) - target) / sd).ravel()
            return np.where(np.isfinite(r), r, 1e3)

        theta0 = np.array([self.params0[k] for k in self.fit_params], float)
        sol = least_squares(resid, theta0, x_scale=np.maximum(np.abs(theta0), 1e-3), max_nfev=200)
        self.params_ = dict(self.params0)
        self.params_.update({k: float(v) for k, v in zip(self.fit_params, sol.x, strict=False)})
        self.info = {"params": self.params_, "tau": self.tau}

    def _predict_mean(self, data, origins):
        H = int(self.horizons.max())
        path = self.integrate(self._history(data.y, origins, self.dt_), H, self.dt_)
        return self._guard(path[:, self.horizons - 1], 0)

    def physics_features(self, data, origins):
        hist = self._history(data.y, origins, self.dt_)
        return (self.rhs(hist[:, -1], hist[:, 0], **self.params_) * self.dt_)[:, None]

    def n_params(self):
        return len(self.fit_params)


class LinearSSMForecaster(Forecaster):
    """Linear Gaussian state-space model of a scalar observation y = mu + x_0 + noise.

    ``kind="ou"``: dx = -theta x dt + sigma dW. ``kind="oscillator"``: x'' + 2 zeta omega x' + omega^2 x =
    sigma xi(t) with the velocity hidden. Exact discretisation at the sampling step, maximum likelihood through the
    Kalman filter on the training segment (Nelder-Mead on log-parameters), Gaussian forecasts from the filtered
    state.
    """
    name, family = "linear_sde_kalman", "physical"
    probabilistic = True

    def __init__(self, kind: str = "ou", max_fit_length: int = 4000, name: str | None = None,
                 family: str | None = None):
        super().__init__()
        if kind not in ("ou", "oscillator"):
            raise ValueError("kind must be 'ou' or 'oscillator'")
        self.kind, self.max_fit_length = kind, int(max_fit_length)
        self.name, self.family = name or f"{kind}_kalman", family or self.family

    def _system(self, p, dt):
        if self.kind == "ou":
            th, sg = p["theta"], p["sigma"]
            phi = np.exp(-th * dt)
            return np.array([[phi]]), np.array([[sg ** 2 * (1 - phi ** 2) / (2 * th)]])
        return oscillator_transition(p["omega"], p["zeta"], p["sigma"], dt)

    def _unpack(self, z):
        if self.kind == "ou":
            return {"theta": np.exp(z[0]), "sigma": np.exp(z[1]), "obs_sigma": np.exp(z[2]), "mu": z[3]}
        return {"omega": np.exp(z[0]), "zeta": 1 / (1 + np.exp(-z[1])), "sigma": np.exp(z[2]),
                "obs_sigma": np.exp(z[3]), "mu": z[4]}

    def _filter(self, y, p, dt, keep=False):
        """Kalman filter with scalar observations, written with Python floats (a 1- or 2-state model is faster that
        way than with NumPy calls); returns the log-likelihood and, with ``keep``, the filtered means and
        covariances."""
        Phi, Q = self._system(p, dt)
        k = Phi.shape[0]
        R = float(p["obs_sigma"]) ** 2
        z = (np.asarray(y, float) - p["mu"]).tolist()
        p0 = max(float(np.var(y)), 1e-6) * 10
        ll = 0.0
        ms, Ps = (np.empty((len(z), k)), np.empty((len(z), k, k))) if keep else (None, None)
        if k == 1:
            a, q = float(Phi[0, 0]), float(Q[0, 0])
            m, P = 0.0, p0
            for i, v in enumerate(z):
                if i:
                    m, P = a * m, a * a * P + q
                S = P + R
                e = v - m
                K = P / S
                m, P = m + K * e, P - K * P
                ll -= 0.5 * (np.log(2 * np.pi * S) + e * e / S)
                if keep:
                    ms[i, 0], Ps[i, 0, 0] = m, P
            return ll, ms, Ps
        f11, f12, f21, f22 = (float(v) for v in Phi.ravel())
        q11, q12, q22 = float(Q[0, 0]), float(Q[0, 1]), float(Q[1, 1])
        m1 = m2 = 0.0
        P11, P12, P22 = p0, 0.0, p0
        log2pi = np.log(2 * np.pi)
        from math import log
        for i, v in enumerate(z):
            if i:
                m1, m2 = f11 * m1 + f12 * m2, f21 * m1 + f22 * m2
                A11, A12 = f11 * P11 + f12 * P12, f11 * P12 + f12 * P22
                A21, A22 = f21 * P11 + f22 * P12, f21 * P12 + f22 * P22
                P11 = A11 * f11 + A12 * f12 + q11
                P12 = A11 * f21 + A12 * f22 + q12
                P22 = A21 * f21 + A22 * f22 + q22
            S = P11 + R
            e = v - m1
            K1, K2 = P11 / S, P12 / S
            m1, m2 = m1 + K1 * e, m2 + K2 * e
            P11, P12, P22 = P11 - K1 * P11, P12 - K1 * P12, P22 - K2 * P12
            ll -= 0.5 * (log2pi + log(S) + e * e / S)
            if keep:
                ms[i] = (m1, m2)
                Ps[i] = ((P11, P12), (P12, P22))
        return ll, ms, Ps

    def _fit(self, data, train_origins, val_origins):
        from scipy.optimize import minimize
        self.dt_ = float(data.dt)
        y = data.y[: train_end(train_origins, self.horizons)][-self.max_fit_length:]
        sd = y.std()
        if self.kind == "ou":
            r1 = max(min(np.corrcoef(y[:-1], y[1:])[0, 1], 0.999), 0.01)
            th = -np.log(r1) / self.dt_
            z0 = [np.log(th), np.log(sd * np.sqrt(2 * th)), np.log(0.1 * sd), y.mean()]
        else:
            f = np.fft.rfftfreq(len(y), self.dt_)
            spec = np.abs(np.fft.rfft(y - y.mean())) ** 2
            om = 2 * np.pi * f[1:][np.argmax(spec[1:])]
            z0 = [np.log(om), np.log(0.1 / 0.9), np.log(sd * om * np.sqrt(0.4 * om)), np.log(0.1 * sd), y.mean()]
        nll = lambda z: -self._filter(y, self._unpack(z), self.dt_)[0]  # noqa: E731
        res = minimize(nll, np.asarray(z0, float), method="Nelder-Mead",
                       options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-4})
        self.params_ = {k: float(v) for k, v in self._unpack(res.x).items()}
        self.info = {"params": self.params_, "loglik": float(-res.fun)}

    def predict(self, data, origins):
        origins = np.asarray(origins, int)
        _, ms, Ps = self._filter(data.y[: origins.max() + 1], self.params_, self.dt_, keep=True)
        Phi, Q = self._system(self.params_, self.dt_)
        m, P = ms[origins], Ps[origins]
        H = int(self.horizons.max())
        mean, var = np.empty((len(origins), H)), np.empty((len(origins), H))
        for h in range(H):
            m = m @ Phi.T
            P = Phi @ P @ Phi.T + Q
            mean[:, h] = m[:, 0] + self.params_["mu"]
            var[:, h] = P[:, 0, 0] + self.params_["obs_sigma"] ** 2
        k = self.horizons - 1
        return Forecast(mean[:, k], np.sqrt(var[:, k]))

    def _predict_mean(self, data, origins):
        return self.predict(data, origins).mean

    def physics_features(self, data, origins):
        """Filtered state at the origin (position and, for the oscillator, velocity) and the oscillator energy."""
        origins = np.asarray(origins, int)
        _, ms, _ = self._filter(data.y[: origins.max() + 1], self.params_, self.dt_, keep=True)
        m = ms[origins]
        if self.kind == "ou":
            return m
        energy = 0.5 * m[:, 1] ** 2 + 0.5 * self.params_["omega"] ** 2 * m[:, 0] ** 2
        return np.column_stack([m, energy])

    def n_params(self):
        return 4 if self.kind == "ou" else 5


class SDEDriftForecaster(Forecaster):
    """dx = f(x) dt + sigma dW for a scalar series, f(x) = sum_k c_k x^k (k <= ``degree``) fitted to the increments,
    optionally plus a neural closure g(x) fitted to what the polynomial leaves (``closure=dict(hidden=..,
    epochs=..)``). Forecast: ``n_samples`` Euler-Maruyama paths with ``sub`` substeps per sample.

    Observation noise on y inflates the estimated diffusion and biases the drift towards zero; that is part of
    what the comparison measures (the stated assumption is that the observation noise is small).
    """
    name, family = "sde_drift", "physical"
    probabilistic = True

    def __init__(self, degree: int = 3, n_samples: int = 200, sub: int = 10, closure: dict | None = None,
                 seed: int = 0, name: str | None = None, family: str | None = None):
        super().__init__()
        self.degree, self.n_samples, self.sub, self.closure, self.seed = int(degree), int(n_samples), int(sub), \
            closure, int(seed)
        self.name = name or (f"sde_poly{degree}" + ("_nn_closure" if closure else ""))
        self.family = family or ("hybrid" if closure else self.family)

    def _basis(self, x):
        return np.stack([x ** k for k in range(self.degree + 1)], axis=-1)

    def drift(self, x):
        f = self._basis(x) @ self.coef_
        if self.net_ is not None:
            import torch
            with torch.no_grad():
                z = torch.as_tensor(((x - self.x_mu_) / self.x_sd_).reshape(-1, 1), dtype=torch.float32)
                f = f + self.net_(z).numpy().reshape(np.shape(x)) * self.g_sd_
        return f

    def _fit(self, data, train_origins, val_origins):
        from .learned import _mlp, train_torch_regressor
        self.dt_ = float(data.dt)
        end = train_end(train_origins, self.horizons)
        y = data.y[:end]
        x, dx = y[:-1], np.diff(y) / self.dt_
        B = self._basis(x)
        self.coef_ = np.linalg.lstsq(B, dx, rcond=None)[0]
        self.net_ = None
        resid = dx - B @ self.coef_
        if self.closure:
            self.x_mu_, self.x_sd_ = float(x.mean()), float(x.std())
            self.g_sd_ = float(resid.std())
            z = ((x - self.x_mu_) / self.x_sd_)[:, None]
            g = (resid / self.g_sd_)[:, None]
            if val_origins is not None:
                yv = data.y[int(np.min(val_origins)): train_end(val_origins, self.horizons)]
                zv = ((yv[:-1] - self.x_mu_) / self.x_sd_)[:, None]
                gv = ((np.diff(yv) / self.dt_ - self._basis(yv[:-1]) @ self.coef_) / self.g_sd_)[:, None]
            else:
                zv = gv = None
            c = {"hidden": 32, "layers": 2, "epochs": 200, "patience": 20, "lr": 1e-3, "weight_decay": 1e-4,
                 **self.closure}
            res = train_torch_regressor(lambda: _mlp(1, 1, c["hidden"], c["layers"], 0.0), z, g, zv, gv,
                                        epochs=c["epochs"], lr=c["lr"], weight_decay=c["weight_decay"],
                                        patience=c["patience"], batch_size=256, seed=self.seed)
            self.net_ = res["model"]
            resid = dx - self.drift(x)
        self.sigma_dw_ = float(np.sqrt(np.mean(resid ** 2) * self.dt_))
        self.info = {"coef": self.coef_.tolist(), "sigma": self.sigma_dw_}

    def predict(self, data, origins):
        origins = np.asarray(origins, int)
        rng = np.random.default_rng(self.seed)
        H = int(self.horizons.max())
        h = self.dt_ / self.sub
        x = np.repeat(data.y[origins][None], self.n_samples, axis=0)       # (m, n)
        lo, hi = data.y.min() - 10 * data.y.std(), data.y.max() + 10 * data.y.std()
        out = np.empty((self.n_samples, len(origins), H))
        for k in range(H):
            for _ in range(self.sub):
                x = x + self.drift(x) * h + self.sigma_dw_ * np.sqrt(h) * rng.standard_normal(x.shape)
                x = np.clip(x, lo, hi)
            out[:, :, k] = x
        s = out[:, :, self.horizons - 1]
        return Forecast(s.mean(axis=0), s.std(axis=0), s)

    def _predict_mean(self, data, origins):
        return self.predict(data, origins).mean

    def physics_features(self, data, origins):
        return (self.drift(data.y[np.asarray(origins)]) * self.dt_)[:, None]

    def n_params(self):
        n = len(self.coef_) + 1
        if self.net_ is not None:
            n += sum(p.numel() for p in self.net_.parameters())
        return int(n)


class EnKFForecaster(Forecaster):
    """Stochastic EnKF over the observations with a fitted :class:`ODEForecaster` as the model.

    The observation-noise variance is estimated as half the mean square one-sample prediction error of the fitted
    model (an assumption: model error over one sample is negligible next to the observation noise); the process
    noise is ``q_frac`` times that. Filtering starts ``spinup`` samples before the first origin, so a forecast
    uses only observations up to its origin.
    """
    name, family = "enkf", "physical"
    probabilistic = True

    def __init__(self, physical: ODEForecaster, n_ens: int = 50, inflation: float = 1.02, q_frac: float = 0.1,
                 spinup: int = 200, seed: int = 0, name: str | None = None, family: str | None = None):
        super().__init__()
        self.physical, self.n_ens, self.inflation, self.q_frac = physical, int(n_ens), float(inflation), q_frac
        self.spinup, self.seed = int(spinup), int(seed)
        self.name, self.family = name or self.name, family or self.family

    def _fit(self, data, train_origins, val_origins):
        self.physical.fit(data, train_origins, None, self.horizons)
        r2 = 0.5 * np.asarray(self.physical.one_step_rmse_) ** 2
        self.R_ = np.diag(np.maximum(r2, 1e-10))
        self.Q_ = self.q_frac * self.R_
        self.info = {"params": self.physical.params_, "obs_noise_std": np.sqrt(np.diag(self.R_)).tolist(),
                     "n_ens": self.n_ens}

    def predict(self, data, origins):
        from pinneapple_analysis.state_estimation.kalman import EnsembleKalmanFilter
        origins = np.asarray(origins, int)
        ph = self.physical
        d, dt = data.obs.shape[1], float(data.dt)
        step = lambda x: ph.integrate(x[None], 1, dt)[0, 0]  # noqa: E731
        enkf = EnsembleKalmanFilter(d, d, step, lambda x: x, Q=self.Q_, R=self.R_, n_ens=self.n_ens,
                                    inflation=self.inflation, seed=self.seed)
        k0 = max(0, int(origins.min()) - self.spinup)
        enkf.initialize(data.obs[k0], P0=self.R_)
        want = set(origins.tolist())
        snaps = {}
        if k0 in want:
            snaps[k0] = enkf.ensemble.copy()
        for k in range(k0 + 1, int(origins.max()) + 1):
            enkf.step(data.obs[k])
            if k in want:
                snaps[k] = enkf.ensemble.copy()
        H = int(self.horizons.max())
        ens = np.stack([snaps[o].T for o in origins], axis=1)               # (m, n, d)
        m, n = ens.shape[:2]
        path = ph.integrate(ens.reshape(m * n, d), H, dt).reshape(m, n, H, d)
        s = path[:, :, self.horizons - 1, data.target]
        bad = ~np.isfinite(s)
        self.info["diverged_fraction"] = float(bad.mean())
        s = np.where(bad, np.nanmean(data.obs[: origins.min(), data.target]), s)
        return Forecast(s.mean(axis=0), s.std(axis=0), s)

    def _predict_mean(self, data, origins):
        return self.predict(data, origins).mean

    def n_params(self):
        return self.physical.n_params() + 2 * len(np.diag(self.R_))


class SINDyForecaster(_DivergenceGuard, Forecaster):
    """Equations discovered by SINDy (STLSQ, polynomial library of degree 2) on the training trajectory of a fully
    observed state; forecasts integrate them (RK4, ``sub`` substeps per sample) from the observed state."""
    name, family = "sindy", "physical"

    def __init__(self, threshold: float = 0.05, l2: float = 1e-4, library: str = "poly", sub: int = 5,
                 divergence_factor: float = 5.0, name: str | None = None, family: str | None = None):
        super().__init__()
        self.threshold, self.l2, self.library, self.sub = float(threshold), float(l2), library, int(sub)
        self.divergence_factor = divergence_factor
        self.name, self.family = name or self.name, family or self.family

    def _fit(self, data, train_origins, val_origins):
        import torch

        from pinneapple_neural.architectures.rom.sindy import SINDy
        end = train_end(train_origins, self.horizons)
        seg = data.obs[:end]
        self._guard_fit(seg, self.divergence_factor)
        self.dt_ = float(data.dt)
        self.model_ = SINDy(seg.shape[1], library=self.library, poly_degree=2, threshold=self.threshold,
                            l2=self.l2, solver="stlsq", integrator="rk4")
        self.model_.fit(torch.as_tensor(seg[None], dtype=torch.float64).float(), dt=self.dt_)
        self.info = {"threshold": self.threshold, "n_active": self.n_params(),
                     "sparsity": float(self.model_.sparsity())}

    def _predict_mean(self, data, origins):
        import torch
        H = int(self.horizons.max())
        a0 = torch.as_tensor(data.obs[origins], dtype=torch.float32)
        with torch.no_grad():
            path = self.model_.rollout(a0, dt=self.dt_ / self.sub, steps=H * self.sub).numpy()
        pred = path[:, self.horizons * self.sub, data.target]
        return self._guard(pred.astype(float), data.target)

    def n_params(self):
        return int((self.model_.Xi.abs() > 1e-10).sum().item())


class PhysicsFeatures:
    """Input features for a data-driven model computed by a physical forecaster (fitted on the training origins
    when the data model is fitted): its ``physics_features(data, origins)`` -- the fitted vector field or drift at
    the origin state, or the filtered state and energy of a linear model."""

    def __init__(self, physical: Forecaster):
        self.physical = physical

    def fit(self, data: SeriesData, train_origins, horizons):
        self.physical.fit(data, train_origins, None, horizons)
        return self

    def __call__(self, data: SeriesData, origins):
        return self.physical.physics_features(data, origins)
