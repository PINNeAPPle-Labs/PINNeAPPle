"""Hybrid forecasters: a physical model plus something learned.

* :class:`ResidualCorrectionForecaster` -- physics forecast + statistical error correction: a gradient-boosting
  model (or MLP) learns y_{t+h} - physics_{t+h} from the lagged observations and the physical forecast itself.
* :class:`UDEForecaster` -- universal differential equation (Rackauckas et al. 2020, arXiv:2001.04385):
  dx/dt = f_known(x; theta) + g_NN(x), with theta estimated first by an :class:`ODEForecaster` on the training data
  and the neural closure g trained through the RK4 integrator on short windows of the training trajectory
  (multiple shooting). The SDE counterpart (drift closure) is
  :class:`~pinneapple_systems.time_series.forecast_lab.physical.SDEDriftForecaster` with ``closure=...``.
"""
from __future__ import annotations

import numpy as np

from .base import Forecaster, targets_at, train_end
from .learned import LagFeatures, _mlp, _Scaler, train_torch_regressor
from .physical import ODEForecaster, _DivergenceGuard

__all__ = ["ResidualCorrectionForecaster", "UDEForecaster"]


class ResidualCorrectionForecaster(Forecaster):
    """physical forecast + corrector(lags, physical forecast). ``corrector`` is ``"gbm"`` or ``"mlp"``."""
    name, family = "physics_plus_correction", "hybrid"

    def __init__(self, physical: Forecaster, corrector: str = "gbm", n_lags: int = 8, lag_step: int = 1,
                 max_iter: int = 200, learning_rate: float = 0.05, max_leaf_nodes: int = 15,
                 min_samples_leaf: int = 20, hidden: int = 64, layers: int = 2, epochs: int = 100,
                 patience: int | None = 10, weight_decay: float = 1e-4, seed: int = 0,
                 name: str | None = None):
        super().__init__()
        self.physical, self.corrector = physical, corrector
        self.lags = LagFeatures(n_lags, lag_step)
        self.gbm_kw = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                           min_samples_leaf=min_samples_leaf, early_stopping=False, random_state=seed)
        self.nn_kw = dict(hidden=hidden, layers=layers, epochs=epochs, patience=patience,
                          weight_decay=weight_decay, seed=seed)
        self.name = name or f"{physical.name}+{corrector}"

    def _features(self, data, origins, phys):
        rel = phys - data.y[origins][:, None]
        return np.concatenate([self.lags.flat(data, origins), rel], axis=1)

    def _fit(self, data, train_origins, val_origins):
        self.physical.fit(data, train_origins, None, self.horizons)
        P = self.physical._predict_mean(data, train_origins)
        X = self._features(data, train_origins, P)
        R = targets_at(data, train_origins, self.horizons) - P
        if self.corrector == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.models_ = [HistGradientBoostingRegressor(**self.gbm_kw).fit(X, R[:, j]) for j in range(R.shape[1])]
        else:
            self.xs_, self.rs_ = _Scaler().fit(X), _Scaler().fit(R)
            Xv = Rv = None
            if val_origins is not None:
                Pv = self.physical._predict_mean(data, val_origins)
                Xv = self.xs_(self._features(data, val_origins, Pv))
                Rv = self.rs_(targets_at(data, val_origins, self.horizons) - Pv)
            k = self.nn_kw
            res = train_torch_regressor(lambda: _mlp(X.shape[1], R.shape[1], k["hidden"], k["layers"], 0.0),
                                        self.xs_(X), self.rs_(R), Xv, Rv, epochs=k["epochs"],
                                        weight_decay=k["weight_decay"], patience=k["patience"], seed=k["seed"])
            self.net_ = res["model"]
        self.info = {"physical": self.physical.describe(), "corrector": self.corrector}

    def _predict_mean(self, data, origins):
        P = self.physical._predict_mean(data, origins)
        X = self._features(data, origins, P)
        if self.corrector == "gbm":
            return P + np.column_stack([m.predict(X) for m in self.models_])
        import torch
        with torch.no_grad():
            r = self.net_(torch.as_tensor(self.xs_(X), dtype=torch.float32)).numpy()
        return P + self.rs_.inverse(r)

    def n_params(self):
        n = self.physical.n_params()
        if self.corrector == "gbm":
            return int(n + sum(sum(p.get_n_leaf_nodes() * 2 - 1 for p in it) for m in self.models_
                               for it in m._predictors))
        return int(n + sum(p.numel() for p in self.net_.parameters()))


class UDEForecaster(_DivergenceGuard, Forecaster):
    """dx/dt = f_known(x; theta_hat) + s * g_NN((x - mu) / sd), trained through RK4 over ``window`` samples.

    ``physical`` is an (unfitted) :class:`ODEForecaster` with the known, possibly incomplete, vector field; its
    parameters are estimated first and kept fixed while the closure is trained (early stopping on windows from the
    validation block). ``s`` is the per-variable standard deviation of the known model's one-step derivative
    residual, so the closure starts at the scale of what is missing.
    """
    name, family = "ude", "hybrid"

    def __init__(self, physical: ODEForecaster, hidden: int = 64, layers: int = 2, window: int = 4,
                 epochs: int = 200, patience: int | None = 20, lr: float = 1e-3, weight_decay: float = 1e-5,
                 batch_size: int = 128, max_windows: int = 3000, seed: int = 0, divergence_factor: float = 5.0,
                 name: str | None = None):
        super().__init__()
        self.physical = physical
        self.hp = dict(hidden=hidden, layers=layers, window=window, epochs=epochs, patience=patience, lr=lr,
                       weight_decay=weight_decay, batch_size=batch_size)
        self.max_windows, self.seed, self.divergence_factor = int(max_windows), int(seed), divergence_factor
        self.name = name or f"ude({physical.name})"

    def _module(self):
        import torch
        import torch.nn as nn
        known = self.physical.torch_rhs()
        net = _mlp(self.d_, self.d_, self.hp["hidden"], self.hp["layers"], 0.0)
        mu, sd, sc = (torch.as_tensor(a, dtype=torch.float32) for a in (self.mu_, self.sd_, self.scale_))
        sub, h, W = self.physical.sub, self.dt_ / self.physical.sub, self.hp["window"]

        class UDE(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = net
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

            def rhs(self, x):
                return known(x) + sc * self.net((x - mu) / sd)

            def rollout(self, x0, n):
                x, out = x0 * sd + mu, []
                for _ in range(n):
                    for _ in range(sub):
                        k1 = self.rhs(x)
                        k2 = self.rhs(x + 0.5 * h * k1)
                        k3 = self.rhs(x + 0.5 * h * k2)
                        k4 = self.rhs(x + h * k3)
                        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    out.append((x - mu) / sd)
                return torch.stack(out, dim=1)

            def forward(self, x0):
                return self.rollout(x0, W).reshape(x0.shape[0], -1)

        return UDE

    def _windows(self, data, lo, hi):
        W = self.hp["window"]
        o = np.arange(lo, hi - W)
        if len(o) > self.max_windows:
            o = np.sort(np.random.default_rng(self.seed).choice(o, self.max_windows, replace=False))
        x0 = (data.obs[o] - self.mu_) / self.sd_
        Y = np.stack([(data.obs[o + k] - self.mu_) / self.sd_ for k in range(1, W + 1)], axis=1)
        return x0, Y.reshape(len(o), -1)

    def _fit(self, data, train_origins, val_origins):
        self.d_, self.dt_ = data.obs.shape[1], float(data.dt)
        self.physical.fit(data, train_origins, None, self.horizons)
        end = train_end(train_origins, self.horizons)
        seg = data.obs[:end]
        self._guard_fit(seg, self.divergence_factor)
        self.mu_, self.sd_ = seg.mean(axis=0), seg.std(axis=0) + 1e-8
        deriv = (seg[2:] - seg[:-2]) / (2 * self.dt_)
        self.scale_ = (deriv - self.physical.rhs_numpy(seg[1:-1])).std(axis=0) + 1e-8
        X, Y = self._windows(data, int(np.min(train_origins)), end)
        Xv = Yv = None
        if val_origins is not None:
            Xv, Yv = self._windows(data, int(np.min(val_origins)), train_end(val_origins, self.horizons))
        cls = self._module()
        hp = self.hp
        res = train_torch_regressor(cls, X, Y, Xv, Yv, epochs=hp["epochs"], lr=hp["lr"],
                                    weight_decay=hp["weight_decay"], batch_size=hp["batch_size"],
                                    patience=hp["patience"], seed=self.seed)
        self.model_ = res["model"]
        self.info = {"physical_params": self.physical.params_, "epochs_run": res["epochs_run"],
                     "best_epoch": res["best_epoch"], **hp}

    def _predict_mean(self, data, origins):
        import torch
        H = int(self.horizons.max())
        x0 = torch.as_tensor((data.obs[origins] - self.mu_) / self.sd_, dtype=torch.float32)
        with torch.no_grad():
            path = self.model_.rollout(x0, H).numpy() * self.sd_ + self.mu_
        return self._guard(path[:, self.horizons - 1, data.target].astype(float), data.target)

    def n_params(self):
        return int(self.physical.n_params() + sum(p.numel() for p in self.model_.net.parameters()))
