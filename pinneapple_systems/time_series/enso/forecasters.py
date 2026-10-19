"""ENSO forecasters with the :class:`~pinneapple_systems.time_series.forecast_lab.Forecaster` interface, so the
forecast_lab protocol (purged walk-forward tuning, untouched test period, DM tests) applies to them unchanged.

They read the monthly indices from ``data.obs`` (columns named in ``data.meta["columns"]``, see
:func:`.dataset.to_series`) and the maps from ``data.meta`` (``sst_maps``, ``d20_band``, ``taux``, ...); a
forecast from origin t uses rows <= t only, and everything estimated (EOFs, coefficients, couplings) comes from rows
before ``train_end`` of the training origins.

* :class:`ColumnSubset` -- run any forecaster on a subset of the columns (e.g. the forecast_lab ODE, UDE, SINDy
  and EnKF forecasters on the (T, h) pair).
* :class:`RechargeOscillatorForecaster` -- :mod:`.oscillators` recharge oscillator, deterministic or stochastic
  (ensemble), optionally seasonal and with multiplicative noise.
* :class:`LIMForecaster` -- linear inverse model (Penland & Sardeshmukh 1995, J. Climate 8:1999) on the leading
  SST EOFs plus thermocline indices, with its analytic Gaussian forecast distribution.
* :class:`MapCNNForecaster` -- convolutional network on the last months of SST anomaly maps plus thermocline
  indices (the design of Ham et al. 2019, Nature 573:568, scaled down), optionally pre-trained on a long stochastic
  run of the intermediate model fitted to the training period.
* :class:`IntermediateModelForecaster` -- the coupled model of :mod:`.intermediate`, fitted to the training
  period (:func:`fit_intermediate`) and initialised either by SST nudging or by a localised ensemble Kalman filter
  that assimilates SST maps and D20 bins (``init="enkf"``).
* :class:`EnsembleForecaster` -- equal-weight multi-model ensemble (mixture of the members' distributions).
"""
from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..forecast_lab.base import Forecast, Forecaster, SeriesData, targets_at, train_end
from ..forecast_lab.learned import _Scaler, train_torch_regressor
from . import intermediate as IM
from . import oscillators as O

__all__ = ["ColumnSubset", "RechargeOscillatorForecaster", "LIMForecaster", "MapCNNForecaster",
           "IntermediateModelForecaster", "EnsembleForecaster", "fit_intermediate", "box_index", "ro_rhs"]


def _col(data: SeriesData, name: str) -> int:
    return list(data.meta["columns"]).index(name)


def box_index(maps: np.ndarray, grid: IM.OceanGrid, box=(-5.0, 5.0, 190.0, 240.0)) -> np.ndarray:
    """cos-latitude weighted box mean of maps (..., ny, nx) on ``grid`` (default the Nino 3.4 box)."""
    iy = (grid.lat > box[0]) & (grid.lat < box[1])
    ix = (grid.lon > box[2]) & (grid.lon < box[3])
    w = np.cos(np.deg2rad(grid.lat[iy]))
    sub = maps[..., iy, :][..., ix]
    return np.einsum("...yx,y->...", sub, w) / (w.sum() * ix.sum())


def ro_rhs(x, R=-0.05, F1=0.02, F2=1.5, eps=0.03):
    """Constant-coefficient recharge oscillator vector field on (T, h) for the forecast_lab ODE forecasters."""
    T, h = x[..., 0], x[..., 1]
    import torch
    if isinstance(x, torch.Tensor):
        return torch.stack([R * T + F1 * h, -F2 * T - eps * h], dim=-1)
    return np.stack([R * T + F1 * h, -F2 * T - eps * h], axis=-1)


class ColumnSubset(Forecaster):
    """Wrap ``inner`` so that it sees only ``cols`` (names in ``data.meta["columns"]``; the target must be one)."""

    def __init__(self, inner: Forecaster, cols: Sequence[str], name: str | None = None, family: str | None = None):
        super().__init__()
        self.inner, self.cols = inner, list(cols)
        self.name = name or inner.name
        self.family = family or inner.family
        self.probabilistic = inner.probabilistic

    def _sub(self, data):
        names = list(data.meta["columns"])
        idx = [names.index(c) for c in self.cols]
        tgt = names[data.target]
        return SeriesData(data.obs[:, idx], self.cols.index(tgt), data.dt, data.name,
                          {**data.meta, "columns": self.cols})

    def fit(self, data, train_origins, val_origins=None, horizons=(1,)):
        self.inner.fit(self._sub(data), train_origins, val_origins, horizons)
        self.horizons, self.fit_seconds, self.sigma_ = self.inner.horizons, self.inner.fit_seconds, self.inner.sigma_
        self.info = self.inner.info
        return self

    def predict(self, data, origins):
        return self.inner.predict(self._sub(data), origins)

    def _fit(self, data, train_origins, val_origins):  # pragma: no cover -- fit() is overridden
        raise NotImplementedError

    def n_params(self):
        return self.inner.n_params()


class RechargeOscillatorForecaster(Forecaster):
    """Jin's recharge oscillator on (T, h) = (``target`` column, ``h_col``), fitted on the training segment.

    ``stochastic``: ensemble of ``n_members`` Euler-Maruyama paths (its own predictive distribution); otherwise the
    deterministic RK4 path with a Gaussian spread calibrated on the validation block.
    """
    name, family = "recharge_oscillator", "physical"

    def __init__(self, h_col: str = "wwv", seasonal: Sequence[str] = (), multiplicative: bool = False,
                 stochastic: bool = True, n_members: int = 200, sub: int = 4, seed: int = 0,
                 name: str | None = None, family: str | None = None):
        super().__init__()
        self.h_col, self.seasonal, self.multiplicative = h_col, tuple(seasonal), multiplicative
        self.stochastic, self.n_members, self.sub, self.seed = stochastic, int(n_members), int(sub), int(seed)
        self.probabilistic = bool(stochastic)
        self.name = name or self.name
        self.family = family or self.family

    def _fit(self, data, train_origins, val_origins):
        end = train_end(train_origins, self.horizons)
        hc = _col(data, self.h_col)
        moy = np.asarray(data.meta["month_of_year"])
        self.p_ = O.fit_recharge(data.y[:end], data.obs[:end, hc], moy[:end], seasonal=self.seasonal,
                                 multiplicative=self.multiplicative, sub=self.sub)
        self.hc_ = hc
        self.info = {"params": self.p_.to_dict()}

    def _paths(self, data, origins):
        moy = np.asarray(data.meta["month_of_year"])[origins]
        X0 = np.column_stack([data.y[origins], data.obs[origins, self.hc_]])
        H = int(self.horizons.max())
        n = self.n_members if self.stochastic else 1
        return O.simulate_recharge(self.p_, X0, H, sub=self.sub, n_members=n, stochastic=self.stochastic,
                                   seed=self.seed, month_offsets=moy)[:, :, self.horizons, 0]

    def predict(self, data, origins):
        origins = np.asarray(origins, int)
        s = self._paths(data, origins)
        if self.stochastic:
            return Forecast(s.mean(0), s.std(0), s)
        mean = s[0]
        std = None if self.sigma_ is None else np.broadcast_to(self.sigma_, mean.shape).copy()
        return Forecast(mean, std)

    def _predict_mean(self, data, origins):
        return self._paths(data, np.asarray(origins, int)).mean(0)

    def n_params(self):
        k = sum(3 if n in self.seasonal else 1 for n in O.PARAM_NAMES)
        return k + 2 + int(self.multiplicative)


class LIMForecaster(Forecaster):
    """Linear inverse model dx/dt = L x + noise on x = (leading ``n_eofs`` PCs of the SST anomaly maps, the
    ``extra_cols`` indices), each standardised; L = log(C(tau0) C(0)^-1) / tau0 from the training segment; the
    target is read out by least squares on x. Forecast mean a . exp(L h) x(t), variance
    a . (C(0) - G(h) C(0) G(h)^T) . a + readout residual variance (Gaussian; Penland & Sardeshmukh 1995)."""
    name, family = "lim", "data"
    probabilistic = True

    def __init__(self, n_eofs: int = 10, extra_cols: Sequence[str] = ("wwv",), tau0: int = 1,
                 name: str | None = None):
        super().__init__()
        self.n_eofs, self.extra_cols, self.tau0 = int(n_eofs), tuple(extra_cols), int(tau0)
        self.name = name or self.name

    def _state(self, data, rows):
        M = np.asarray(data.meta["sst_maps"])[rows].reshape(len(rows), -1)[:, self.mask_]
        pcs = ((M - self.mmu_) * self.w_) @ self.E_ / self.pc_sd_
        ex = [(data.obs[rows, _col(data, c)] - self.emu_[i]) / self.esd_[i] for i, c in enumerate(self.extra_cols)]
        return np.column_stack([pcs] + ex) if ex else pcs

    def _fit(self, data, train_origins, val_origins):
        from scipy.linalg import expm, logm
        end = train_end(train_origins, self.horizons)
        maps = np.asarray(data.meta["sst_maps"])[:end]
        grid = data.meta["grid"]
        M = maps.reshape(end, -1)
        self.mask_ = M.std(0) > 1e-6
        M = M[:, self.mask_]
        w = np.sqrt(np.cos(np.deg2rad(grid.lat)))[:, None] * np.ones(grid.nx)
        self.w_ = w.ravel()[self.mask_]
        self.mmu_ = M.mean(0)
        U, s, Vt = np.linalg.svd((M - self.mmu_) * self.w_, full_matrices=False)
        self.E_ = Vt[: self.n_eofs].T
        pcs = ((M - self.mmu_) * self.w_) @ self.E_
        self.pc_sd_ = pcs.std(0)
        self.evr_ = (s[: self.n_eofs] ** 2 / np.sum(s ** 2)).tolist()
        self.emu_ = [data.obs[:end, _col(data, c)].mean() for c in self.extra_cols]
        self.esd_ = [data.obs[:end, _col(data, c)].std() for c in self.extra_cols]
        X = self._state(data, np.arange(end))
        k = self.tau0
        C0 = X.T @ X / len(X)
        Ck = X[k:].T @ X[:-k] / (len(X) - k)
        cond = float(np.linalg.cond(C0))
        if cond > 1e8:
            raise ValueError(f"LIM state covariance is singular (condition {cond:.1e}): collinear state variables")
        G = Ck @ np.linalg.inv(C0)
        L = logm(G) / k
        self.L_, self.C0_ = np.real(L), C0
        self.imag_ = float(np.abs(np.imag(L)).max())
        A = np.column_stack([X, np.ones(end)])
        self.a_ = np.linalg.lstsq(A, data.y[:end], rcond=None)[0]
        self.r_var_ = float(np.var(data.y[:end] - A @ self.a_))
        self.G_ = {int(h): expm(self.L_ * h) for h in self.horizons}
        ev = np.linalg.eigvals(self.L_)
        self.info = {"n_eofs": self.n_eofs, "extra": list(self.extra_cols), "eof_variance": self.evr_,
                     "readout_r2": float(1 - self.r_var_ / data.y[:end].var()),
                     "max_real_eigenvalue": float(ev.real.max()), "logm_imag": self.imag_, "cond_C0": cond,
                     "least_damped_period_months": float(2 * np.pi / np.abs(ev[np.argmax(ev.real)].imag))
                     if abs(ev[np.argmax(ev.real)].imag) > 1e-9 else None}

    def predict(self, data, origins):
        origins = np.asarray(origins, int)
        X = self._state(data, origins)
        a, c = self.a_[:-1], self.a_[-1]
        mean = np.column_stack([X @ self.G_[int(h)].T @ a + c for h in self.horizons])
        var = np.array([a @ (self.C0_ - self.G_[int(h)] @ self.C0_ @ self.G_[int(h)].T) @ a + self.r_var_
                        for h in self.horizons])
        return Forecast(mean, np.broadcast_to(np.sqrt(np.maximum(var, 1e-12)), mean.shape).copy())

    def _predict_mean(self, data, origins):
        return self.predict(data, origins).mean

    def n_params(self):
        d = self.L_.shape[0]
        return int(d * d + d + 1)


# ---------------------------------------------------------------------------
# CNN on maps
# ---------------------------------------------------------------------------

def _cnn(L, ny, nx, n_scalar, n_out, channels, hidden, dropout):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            c1, c2 = channels
            self.conv = nn.Sequential(nn.Conv2d(L, c1, (3, 5), padding=(1, 2)), nn.Tanh(), nn.MaxPool2d(2),
                                      nn.Conv2d(c1, c2, (3, 5), padding=(1, 2)), nn.Tanh(), nn.MaxPool2d(2))
            with torch.no_grad():
                nf = self.conv(torch.zeros(1, L, ny, nx)).numel()
            self.head = nn.Sequential(nn.Linear(nf + n_scalar, hidden), nn.Tanh(), nn.Dropout(dropout),
                                      nn.Linear(hidden, n_out))
            self.L, self.ny, self.nx = L, ny, nx

        def forward(self, x):
            k = self.L * self.ny * self.nx
            m = x[:, :k].reshape(-1, self.L, self.ny, self.nx)
            z = self.conv(m).flatten(1)
            return self.head(torch.cat([z, x[:, k:]], dim=1))

    return Net


class MapCNNForecaster(Forecaster):
    """CNN on the last ``n_lags`` monthly SST anomaly maps (coarsened by ``pool``) plus scalars (``scalar_cols`` at
    the origin and the sine / cosine of the origin month); outputs the standardised change of the target at every
    horizon (persistence = zero output). ``n_members`` seeds are averaged.

    ``pretrain`` (dict, e.g. ``{"years": 400, "members": 8}``): first fit :class:`IntermediateModelForecaster`'s
    coupled model on the same training segment, run it with stochastic winds for that many model years, train on
    the synthetic maps and indices, then fine-tune on the observations (Ham et al. 2019 pre-trained on CMIP5 runs;
    here the long run comes from a physical model fitted to the training years only, so no test year leaks in).
    """
    name, family = "cnn", "data"

    def __init__(self, n_lags: int = 3, pool: int = 2, scalar_cols: Sequence[str] = ("wwv", "h_west", "h_east"),
                 channels: tuple[int, int] = (16, 32), hidden: int = 64, dropout: float = 0.1,
                 weight_decay: float = 1e-3, lr: float = 1e-3, epochs: int = 150, patience: int = 15,
                 batch_size: int = 32, n_members: int = 5, pretrain: dict | None = None, seed: int = 0,
                 name: str | None = None, family: str | None = None):
        super().__init__()
        self.n_lags, self.pool, self.scalar_cols = int(n_lags), int(pool), tuple(scalar_cols)
        self.hp = dict(channels=tuple(channels), hidden=hidden, dropout=dropout, weight_decay=weight_decay, lr=lr,
                       epochs=epochs, patience=patience, batch_size=batch_size)
        self.n_members, self.pretrain, self.seed = int(n_members), pretrain, int(seed)
        self.name = name or ("cnn_pretrained" if pretrain else self.name)
        self.family = family or self.family

    def _coarse(self, maps):
        p = self.pool
        n, ny, nx = maps.shape
        return maps[:, : ny // p * p, : nx // p * p].reshape(n, ny // p, p, nx // p, p).mean((2, 4))

    def _features(self, maps, scal, moy, rows):
        """maps (T, ny, nx) coarse, scal (T, k), moy (T,) -> X (n, L*ny*nx + k + 2)."""
        L = self.n_lags
        idx = np.asarray(rows)[:, None] - np.arange(L - 1, -1, -1)[None]
        m = (maps[np.maximum(idx, 0)] - self.mmu_) / self.msd_
        s = (scal[rows] - self.smu_) / self.ssd_
        ang = 2 * np.pi * moy[rows] / 12.0
        return np.concatenate([m.reshape(len(rows), -1), s, np.cos(ang)[:, None], np.sin(ang)[:, None]], axis=1)

    def _train(self, build, X, Y, Xv, Yv, seed, lr=None, epochs=None):
        hp = self.hp
        return train_torch_regressor(build, X, Y, Xv, Yv, epochs=epochs or hp["epochs"], lr=lr or hp["lr"],
                                     weight_decay=hp["weight_decay"], batch_size=hp["batch_size"],
                                     patience=hp["patience"], seed=seed)["model"]

    def _fit(self, data, train_origins, val_origins):
        end = train_end(train_origins, self.horizons)
        moy = np.asarray(data.meta["month_of_year"])
        maps = self._coarse(np.asarray(data.meta["sst_maps"], float))
        scal = np.column_stack([data.obs[:, _col(data, c)] for c in self.scalar_cols]) if self.scalar_cols \
            else np.zeros((len(data), 0))
        self.mmu_, self.msd_ = 0.0, float(maps[:end].std()) + 1e-8
        self.smu_, self.ssd_ = scal[:end].mean(0), scal[:end].std(0) + 1e-8
        H = self.horizons
        X = self._features(maps, scal, moy, train_origins)
        Yd = targets_at(data, train_origins, H) - data.y[train_origins][:, None]
        self.ys_ = _Scaler().fit(Yd)
        Xv = Yv = None
        if val_origins is not None:
            Xv = self._features(maps, scal, moy, val_origins)
            Yv = self.ys_(targets_at(data, val_origins, H) - data.y[val_origins][:, None])
        ny, nx = maps.shape[1:]
        Net = _cnn(self.n_lags, ny, nx, scal.shape[1] + 2, len(H), self.hp["channels"], self.hp["hidden"],
                   self.hp["dropout"])
        base = None
        self.info = {"n_lags": self.n_lags, "pool": self.pool, **{k: v for k, v in self.hp.items()}}
        if self.pretrain:
            base, pinfo = self._pretrain(data, train_origins, Net, maps.shape[1:])
            self.info["pretrain"] = pinfo
        self.members_ = []
        for m in range(self.n_members):
            if base is not None:
                pre = base[m % len(base)]

                def build(pre=pre):
                    return copy.deepcopy(pre)
                net = self._train(build, X, self.ys_(Yd), Xv, Yv, self.seed + 1000 * m, lr=0.3 * self.hp["lr"])
            else:
                net = self._train(Net, X, self.ys_(Yd), Xv, Yv, self.seed + 1000 * m)
            self.members_.append(net)

    def _pretrain(self, data, train_origins, Net, shape):
        cfg = {"years": 300, "members": 6, **self.pretrain}
        im = IntermediateModelForecaster(init="nudged")
        im.fit(data, train_origins, None, self.horizons)
        model = im.model_
        nm = int(cfg["members"])
        n_months = int(cfg["years"] * 12 / nm)
        state = model.zeros((nm,))
        rng = np.random.default_rng(self.seed)
        state["T"] = 0.3 * rng.standard_normal(state["T"].shape) * model.S
        spin = 24
        _, Tm, Hm = model.run(state, n_months + spin, seed=self.seed, stochastic=True)
        Tm, Hm = Tm[spin:], Hm[spin:]                                    # (t, members, ny, nx)
        grid = model.ocean.g
        y = box_index(Tm, grid)                                           # (t, members)
        band = np.abs(grid.lat) <= 5
        h_lon = Hm[:, :, band].mean(2) * im.h_scale_                      # D20-equivalent anomaly per lon
        west = grid.lon < 205
        sc = {"wwv": h_lon.mean(-1), "h_west": h_lon[..., west].mean(-1), "h_east": h_lon[..., ~west].mean(-1)}
        Xs, Ys = [], []
        H = self.horizons
        for k in range(nm):
            maps = self._coarse(Tm[:, k])
            scal = np.column_stack([sc[c][:, k] for c in self.scalar_cols]) if self.scalar_cols \
                else np.zeros((len(maps), 0))
            moy = np.arange(len(maps)) % 12
            rows = np.arange(self.n_lags - 1, len(maps) - H.max())
            Xs.append(self._features(maps, scal, moy, rows))
            Ys.append(self.ys_(y[rows[:, None] + H[None], k] - y[rows, k][:, None]))
        Xs, Ys = np.concatenate(Xs), np.concatenate(Ys)
        n_val = len(Xs) // 10
        nets = [self._train(Net, Xs[:-n_val], Ys[:-n_val], Xs[-n_val:], Ys[-n_val:], self.seed + 7 + m,
                            epochs=max(20, self.hp["epochs"] // 3)) for m in range(min(self.n_members, 3))]
        info = {"synthetic_months": int(n_months * nm), "synthetic_samples": int(len(Xs)),
                "synthetic_nino34_std": float(y.std()), "synthetic_wwv_std": float(sc["wwv"].std()),
                "coupled_params": im.info.get("coupled", {})}
        return nets, info

    def _predict_mean(self, data, origins):
        import torch
        moy = np.asarray(data.meta["month_of_year"])
        maps = self._coarse(np.asarray(data.meta["sst_maps"], float))
        scal = np.column_stack([data.obs[:, _col(data, c)] for c in self.scalar_cols]) if self.scalar_cols \
            else np.zeros((len(data), 0))
        X = torch.as_tensor(self._features(maps, scal, moy, np.asarray(origins)), dtype=torch.float32)
        with torch.no_grad():
            out = np.mean([m(X).numpy() for m in self.members_], axis=0)
        return data.y[origins][:, None] + self.ys_.inverse(out)

    def n_params(self):
        return int(sum(sum(p.numel() for p in m.parameters()) for m in self.members_))


# ---------------------------------------------------------------------------
# Intermediate coupled model
# ---------------------------------------------------------------------------

def _interp_bins(lon_c, vals, lon):
    """Linear interpolation of bin values (..., nb) at bin centres to longitudes ``lon`` (constant beyond)."""
    vals = np.asarray(vals, float)
    out = np.empty(vals.shape[:-1] + (len(lon),))
    flat = vals.reshape(-1, vals.shape[-1])
    o = out.reshape(-1, len(lon))
    for i in range(len(flat)):
        o[i] = np.interp(lon, lon_c, flat[i])
    return out


def fit_intermediate(meta: dict[str, Any], end: int, *, eps_a: float = 0.1, c: float = 2.7, r_days: float = 900.0,
                     coupling: float = 1.0, window: int = 4, ridge: float = 1e-3, L_T: float = 6.0,
                     dt_hours: float = 8.0, atm_every: int = 15, hindcast: bool = True, log=None
                     ) -> tuple[IM.CoupledModel, dict[str, Any]]:
    """Fit the coupled model to the rows [0, end) of the study data (``meta`` of :func:`.dataset.to_series`).

    1. The atmosphere's coupling mu: least squares of the ERA5 zonal stress anomalies (130E-80W, 10S-10N) on the Gill
       winds of the observed SST anomalies. The stochastic forcing ``noise_tau`` is the standard deviation of the
       residual stress.
    2. The ocean (c, r fixed) forced by the ERA5 stress over the same months: its equatorial h is regressed on the
       observed 2S-2N D20 anomalies to give the amplitude factor ``h_scale`` (model h -> D20), and its correlation
       with them is the ocean hindcast check.
    3. The SST equation per longitude (pooled over +-``window`` cells): monthly change of the equatorial (2S-2N)
       SST anomaly regressed on T, the observed D20 anomaly (in model units) and the model's own coupled stress.
    ``coupling`` scales mu afterwards (a tuning knob for the coupled stability).
    """
    grid = meta["grid"]
    maps = np.asarray(meta["sst_maps"], float)[:end]
    taux = np.asarray(meta["taux"], float)[:end]
    have = np.isfinite(taux).all(axis=(1, 2))
    model = IM.CoupledModel(IM.CoupledParams(mu=1.0, L_T=L_T),
                            IM.ShallowWaterOcean(grid, c=c, r_days=r_days, dtype="float32"),
                            IM.GillAtmosphere(eps=eps_a, dlon=grid.dlon, lon_offset=grid.lon[0]),
                            dt_hours=dt_hours, atm_every=atm_every)
    gu, _ = model.stress(maps)                                          # mu = 1: Gill u at u points
    reg = (np.abs(grid.lat) <= 10)[:, None] & ((grid.lon_u >= 130) & (grid.lon_u <= 280))[None, :]
    a, b = gu[have][:, reg].ravel(), taux[have][:, reg].ravel()
    mu = float(a @ b / (a @ a))
    r2 = float(1 - np.var(b - mu * a) / np.var(b))
    noise = float(np.std(b - mu * a))
    info: dict[str, Any] = {"mu": mu, "stress_r2": r2, "noise_tau": noise, "stress_months": int(have.sum())}
    # ocean hindcast with observed stress
    d20_eq = np.asarray(meta["d20_eq"], float)[:end]
    lon_c = np.asarray(meta["d20_lon"])
    eqr = np.abs(grid.lat) < 2
    h_scale = 1.0
    if hindcast and have.sum() > 24:
        rows = np.nonzero(have)[0]
        hfull, _ = IM.forced_ocean_run(model.ocean, taux[rows], steps_per_month=model.steps_per_month)
        hs = list(hfull)
        hm = np.stack(hs)[:, eqr].mean(1)                               # (t, nx) model equatorial h
        hm_bins = np.stack([hm[:, (grid.lon >= lo - 5) & (grid.lon < lo + 5)].mean(1) for lo in lon_c], 1)
        ob = d20_eq[rows]
        spin = min(24, len(rows) // 4)
        A, B = hm_bins[spin:].ravel(), ob[spin:].ravel()
        h_scale = float(A @ B / (A @ A))
        corr = [float(np.corrcoef(hm_bins[spin:, j], ob[spin:, j])[0, 1]) for j in range(len(lon_c))]
        wwv_m, wwv_o = hm_bins[spin:].mean(1), ob[spin:].mean(1)
        info["ocean_hindcast"] = {"h_scale": h_scale, "corr_by_bin": corr, "lon": lon_c.tolist(),
                                  "corr_zonal_mean": float(np.corrcoef(wwv_m, wwv_o)[0, 1]),
                                  "corr_tilt": float(np.corrcoef(hm_bins[spin:, -4:].mean(1) - hm_bins[spin:, :4].mean(1),
                                                                 ob[spin:, -4:].mean(1) - ob[spin:, :4].mean(1))[0, 1]),
                                  "months": int(len(rows) - spin)}
    # SST equation per longitude
    T_eq = maps[:, eqr].mean(1)                                         # (t, nx)
    h_eq = _interp_bins(lon_c, d20_eq, grid.lon) / h_scale              # model units
    tau_c = 0.5 * (gu[..., 1:] + gu[..., :-1])[:, eqr].mean(1) * mu     # coupled stress at centres
    dT = T_eq[1:] - T_eq[:-1]
    nx = grid.nx
    eps_T, gamma, kappa = np.zeros(nx), np.zeros(nx), np.zeros(nx)
    for i in range(nx):
        j = slice(max(0, i - window), min(nx, i + window + 1))
        A = np.column_stack([T_eq[:-1, j].ravel(), h_eq[:-1, j].ravel(), tau_c[:-1, j].ravel()])
        y = dT[:, j].ravel()
        sc = A.std(0) + 1e-12
        co = np.linalg.solve((A / sc).T @ (A / sc) + ridge * len(y) * np.eye(3), (A / sc).T @ y) / sc
        eps_T[i], gamma[i], kappa[i] = -co[0], co[1], co[2]
    info.update({"eps_T": eps_T.tolist(), "gamma": gamma.tolist(), "kappa": kappa.tolist(), "h_scale": h_scale,
                 "coupling": coupling, "c": c, "r_days": r_days, "eps_a": eps_a})
    params = IM.CoupledParams(eps_T=eps_T, gamma=gamma, kappa=kappa, mu=mu * coupling, L_T=L_T, noise_tau=noise,
                              noise_months=1.0)
    model.p = params
    model.gamma, model.kappa = gamma, kappa
    return model, info


class IntermediateModelForecaster(Forecaster):
    """The coupled intermediate model fitted on the training segment (:func:`fit_intermediate`).

    ``init="nudged"``: one continuous run with the SST nudged to the observed maps (5-day time scale) and the ocean
    driven by the model's own winds gives the state at every origin (only data up to the origin); forecasts are
    deterministic (``n_members=1``) or stochastic-wind ensembles. ``init="enkf"``: a localised stochastic EnKF of
    ``enkf_members`` members assimilates the monthly SST anomaly maps (|lat| <= 10, every second cell, error
    ``sst_err``) and the observed 5S-5N D20 bins (error ``d20_err``) month by month; forecasts start from the
    analysis members (``n_members`` of them) with stochastic winds.
    """
    name, family = "intermediate_model", "physical"

    def __init__(self, init: str = "nudged", n_members: int = 1, enkf_members: int = 24, coupling: float = 1.0,
                 eps_a: float = 0.1, c: float = 2.7, r_days: float = 900.0, nudge_days: float = 5.0,
                 sst_err: float = 0.3, d20_err: float = 8.0, inflation: float = 1.05, loc_lon: float = 25.0,
                 loc_lat: float = 8.0, seed: int = 0, name: str | None = None, family: str | None = None):
        super().__init__()
        self.init, self.n_members, self.enkf_members = init, int(n_members), int(enkf_members)
        self.kw = dict(coupling=coupling, eps_a=eps_a, c=c, r_days=r_days)
        self.nudge_days, self.sst_err, self.d20_err, self.inflation = nudge_days, sst_err, d20_err, inflation
        self.loc_lon, self.loc_lat, self.seed = loc_lon, loc_lat, int(seed)
        self.probabilistic = self.n_members > 1
        self.name = name or ("im_enkf" if init == "enkf" else "im_nudged")
        self.family = family or ("hybrid" if init == "enkf" else "physical")

    def _fit(self, data, train_origins, val_origins):
        end = train_end(train_origins, self.horizons)
        self.model_, info = fit_intermediate(data.meta, end, **self.kw)
        self.h_scale_ = info["h_scale"]
        self.info = {"coupled": info}
        self._states = {}

    # initial states ---------------------------------------------------------
    def _nudged_states(self, data, upto: int):
        m = self.model_
        maps = np.asarray(data.meta["sst_maps"], float)
        k0 = max(self._states) + 1 if self._states else 0
        st = self._states[k0 - 1] if k0 else m.zeros(())
        for k in range(k0, upto + 1):
            st, _, _ = m.run(st, 1, T_obs=maps[k:k + 1], nudge_days=self.nudge_days, record=False,
                             stochastic=False)
            self._states[k] = {kk: v.astype(np.float32) for kk, v in st.items()}

    def _enkf_setup(self, data):
        g = self.model_.ocean.g
        lat_c, lon_c = np.meshgrid(g.lat, g.lon, indexing="ij")
        coords = [np.meshgrid(g.lat, g.lon_u, indexing="ij"), np.meshgrid(g.lat_v, g.lon, indexing="ij"),
                  (lat_c, lon_c), (lat_c, lon_c)]
        self.sx_ = np.concatenate([c[0].ravel() for c in coords]), np.concatenate([c[1].ravel() for c in coords])
        self.sizes_ = [c[0].size for c in coords]
        sel = (np.abs(g.lat) <= 10)[:, None] & (np.arange(g.nx) % 2 == 0)[None, :]
        sel &= (np.arange(g.ny) % 2 == 0)[:, None]
        self.sst_idx_ = np.nonzero(sel.ravel())[0]
        bins = np.asarray(data.meta["d20_lon"])
        band = np.abs(g.lat) <= 5
        self.bin_w_ = []
        for lo in bins:
            w = band[:, None] & ((g.lon >= lo - 5) & (g.lon < lo + 5))[None, :]
            self.bin_w_.append(w.ravel() / w.sum())
        self.bin_w_ = np.stack(self.bin_w_)
        ox_lat = np.concatenate([lat_c.ravel()[self.sst_idx_], np.zeros(len(bins))])
        ox_lon = np.concatenate([lon_c.ravel()[self.sst_idx_], bins])
        from pinneapple_analysis.state_estimation import gaspari_cohn

        def loc(la1, lo1, la2, lo2):
            d = np.hypot((lo1[:, None] - lo2[None]) / self.loc_lon, (la1[:, None] - la2[None]) / self.loc_lat)
            return gaspari_cohn(d)                                       # zero beyond twice the scales
        self.loc_xy_full_ = loc(self.sx_[0], self.sx_[1], ox_lat, ox_lon)
        self.loc_yy_full_ = loc(ox_lat, ox_lon, ox_lat, ox_lon)

    def _flat(self, st):
        return np.concatenate([st[k].reshape(st[k].shape[0], -1) for k in ("u", "v", "h", "T")], axis=1)

    def _unflat(self, X):
        g = self.model_.ocean.g
        shapes = {"u": (g.ny, g.nx + 1), "v": (g.ny + 1, g.nx), "h": (g.ny, g.nx), "T": (g.ny, g.nx)}
        out, i = {}, 0
        for k, n in zip(("u", "v", "h", "T"), self.sizes_, strict=False):
            out[k] = X[:, i:i + n].reshape((len(X),) + shapes[k])
            i += n
        return out

    def _enkf_states(self, data, upto: int):
        m = self.model_
        if not hasattr(self, "loc_xy_full_"):
            self._enkf_setup(data)
        from pinneapple_analysis.state_estimation import enkf_analysis
        maps = np.asarray(data.meta["sst_maps"], float)
        d20 = np.asarray(data.meta["d20_band"], float)
        observed = np.asarray(data.meta.get("d20_observed", np.isfinite(d20)), bool)
        N = self.enkf_members
        k0 = max(self._states) + 1 if self._states else 0
        rng = np.random.default_rng(self.seed + k0)
        if k0:
            st = {kk: v.astype(float) for kk, v in self._states[k0 - 1].items()}
        else:
            st = m.zeros((N,))
            st["T"] = maps[0][None] + 0.3 * rng.standard_normal(st["T"].shape)
        for k in range(k0, upto + 1):
            if k > 0:
                st, _, _ = m.run(st, 1, record=False, stochastic=True, seed=self.seed + 7919 * k)
            X = self._flat(st)
            ob_d = np.nonzero(observed[k])[0]
            y = np.concatenate([maps[k].ravel()[self.sst_idx_], d20[k, ob_d] / self.h_scale_])
            hx = np.concatenate([st["T"].reshape(N, -1)[:, self.sst_idx_],
                                 st["h"].reshape(N, -1) @ self.bin_w_[ob_d].T], axis=1)
            r = np.concatenate([np.full(len(self.sst_idx_), self.sst_err ** 2),
                                np.full(len(ob_d), (self.d20_err / self.h_scale_) ** 2)])
            cols = np.concatenate([np.arange(len(self.sst_idx_)), len(self.sst_idx_) + ob_d])
            Xa = enkf_analysis(X, y, hx, r, loc_xy=self.loc_xy_full_[:, cols],
                               loc_yy=self.loc_yy_full_[np.ix_(cols, cols)], inflation=self.inflation, rng=rng)
            st = self._unflat(Xa)
            self._states[k] = {kk: v.astype(np.float32) for kk, v in st.items()}

    def _initial(self, data, origins):
        upto = int(np.max(origins))
        if not self._states or max(self._states) < upto:
            (self._enkf_states if self.init == "enkf" else self._nudged_states)(data, upto)
        if self.init == "enkf":
            n = self.n_members
            st = {k: np.stack([self._states[int(o)][k][:n] for o in origins], 1) for k in ("u", "v", "h", "T")}
            return st                                                    # (members, origins, ...)
        st = {k: np.stack([self._states[int(o)][k] for o in origins]) for k in ("u", "v", "h", "T")}
        if self.n_members > 1:
            st = {k: np.broadcast_to(v, (self.n_members,) + v.shape).copy() for k, v in st.items()}
        return st

    def forecast_fields(self, data, origins, n_months: int | None = None, chunk: int = 400):
        """Monthly mean T and h maps of the forecasts from ``origins``: arrays (months, [members,] origins, ny, nx)."""
        origins = np.asarray(origins, int)
        H = int(n_months or self.horizons.max())
        Ts, Hs = [], []
        for i in range(0, len(origins), chunk):
            o = origins[i:i + chunk]
            st = self._initial(data, o)
            stoch = self.n_members > 1
            _, T, Hh = self.model_.run(st, H, record=True, stochastic=stoch, seed=self.seed + int(o[0]))
            Ts.append(T)
            Hs.append(Hh)
        ax = 2 if (self.n_members > 1) else 1
        return np.concatenate(Ts, axis=ax), np.concatenate(Hs, axis=ax)

    def predict(self, data, origins):
        origins = np.asarray(origins, int)
        T, _ = self.forecast_fields(data, origins)
        idx = box_index(T, self.model_.ocean.g)                          # (H, [m,] n)
        sel = idx[self.horizons - 1]
        if self.n_members > 1:
            s = np.moveaxis(sel, 0, -1)                                  # (m, n, nh)
            bad = ~np.isfinite(s)
            self.info["diverged_fraction"] = float(bad.mean())
            s = np.where(bad, 0.0, s)
            return Forecast(s.mean(0), s.std(0), s)
        mean = sel.T
        bad = ~np.isfinite(mean) | (np.abs(mean) > 10)
        self.info["diverged_fraction"] = float(bad.mean())
        mean = np.where(bad, 0.0, mean)
        std = None if self.sigma_ is None else np.broadcast_to(self.sigma_, mean.shape).copy()
        return Forecast(mean, std)

    def _predict_mean(self, data, origins):
        return self.predict(data, origins).mean

    def n_params(self):
        g = self.model_.ocean.g
        return 3 * g.nx + 2


class EnsembleForecaster(Forecaster):
    """Equal-weight mixture of fitted-together members: mean = mean of the members' means; samples = ``n_draw``
    draws from each member's distribution (its samples, or its Gaussian)."""
    name, family = "ensemble", "hybrid"
    probabilistic = True

    def __init__(self, members: Sequence[Forecaster], n_draw: int = 100, seed: int = 0, name: str | None = None,
                 family: str | None = None):
        super().__init__()
        self.members, self.n_draw, self.seed = list(members), int(n_draw), int(seed)
        self.name = name or "ensemble(" + "+".join(m.name for m in self.members) + ")"
        self.family = family or self.family

    def _fit(self, data, train_origins, val_origins):
        for m in self.members:
            m.fit(data, train_origins, val_origins, self.horizons)
        self.info = {"members": [m.describe() for m in self.members]}

    def predict(self, data, origins):
        rng = np.random.default_rng(self.seed)
        draws = []
        for m in self.members:
            f = m.predict(data, origins)
            if f.samples is not None:
                k = rng.integers(0, f.samples.shape[0], self.n_draw)
                draws.append(f.samples[k])
            else:
                sd = f.std if f.std is not None else np.zeros_like(f.mean)
                draws.append(f.mean[None] + sd[None] * rng.standard_normal((self.n_draw,) + f.mean.shape))
        s = np.concatenate(draws)
        return Forecast(s.mean(0), s.std(0), s)

    def _predict_mean(self, data, origins):
        return self.predict(data, origins).mean

    def n_params(self):
        return int(sum(m.n_params() for m in self.members))
