"""Physics-informed networks for ENSO, trained with :class:`pinneapple_neural.trainer.inverse_pinn.InversePINN`
(network weights and unknown physical parameters optimised together; data misfit plus ODE/PDE residual at random
collocation points, adaptive weighting).

* :func:`identify_recharge` -- a PINN for the recharge oscillator on a stretch of observed (T, h): the network
  X(t) fits the data while dX/dt - f(X; theta) is penalised; theta = (R, F1, F2, eps) (optionally seasonal) is
  learned, and an optional neural closure g(h) of the thermocline feedback is added to dT/dt.
  :class:`PINNRechargeForecaster` forecasts by integrating the identified equations from the observed state.
* :class:`PINNExtrapolationForecaster` -- the "PINN as a forecaster" test: for every origin a PINN is fitted to
  the last ``window`` months and to the recharge-oscillator residual on [origin - window, origin + max(h)]; the
  forecast is the network beyond the data. One grouped network (independent weights per origin) trains all origins
  at once.
* :class:`EquatorialWavePINN` -- a PINN for the long-wave shallow-water equations projected on the equatorial
  Kelvin and first-meridional Rossby modes (the two-mode reduction of the linear model in :mod:`.intermediate`)::

      dq0/dt + c dq0/dx + r q0 = s X0                   q = g' h / c + u  (Matsuno variables, m/s)
      dq2/dt - c/3 dq2/dx + r q2 = s (2 X2 - X0) / 6     X_n = projections of tau_x / (rho H) on phi_n(y / L)
      h(x, 0, t) = (H / c) (q0 / 2 + q2)                 east wall: q2 = q0 / 2 (no net zonal mass flux)
                                                          west wall: q0 = a_W q2 (a_W = 2: closed, < 2: leaky)

  coupled to an equatorial SST equation dT/dt = -eps_T T + kappa tau_x + g_NN(x, h) whose thermocline-feedback
  closure g_NN is learned. Data: observed equatorial D20 and SST anomalies; forcing: ERA5 stress. Unknowns: c, r,
  a_W, the forcing scale s, eps_T, kappa and the closure.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..forecast_lab.base import Forecast, Forecaster, train_end
from . import oscillators as O

__all__ = ["FourierMLP", "GroupedMLP", "identify_recharge", "PINNRechargeForecaster", "PINNExtrapolationForecaster",
           "EquatorialWavePINN", "project_forcing"]


def _torch():
    import torch
    return torch


class FourierMLP(_torch().nn.Module):
    """MLP on [x, sin(2 pi k x_j), cos(2 pi k x_j)] (k = 1..n_freq for each input j, inputs scaled to [0, 1]),
    tanh hidden layers; ``out_scale`` multiplies the outputs."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, layers: int = 3, n_freq: int = 8,
                 out_scale: Sequence[float] | None = None, freq_max: float | None = None):
        torch = _torch()
        nn = torch.nn
        super().__init__()
        fm = freq_max or n_freq
        self.register_buffer("k", torch.linspace(1.0, fm, n_freq) if n_freq else torch.zeros(0))
        w = in_dim * (1 + 2 * n_freq)
        mods = []
        for _ in range(layers):
            mods += [nn.Linear(w, hidden), nn.Tanh()]
            w = hidden
        mods.append(nn.Linear(w, out_dim))
        self.net = nn.Sequential(*mods)
        self.register_buffer("out_scale", torch.ones(out_dim) if out_scale is None else
                             torch.as_tensor(np.asarray(out_scale, float), dtype=torch.float32))

    def features(self, x):
        torch = _torch()
        if len(self.k) == 0:
            return x
        a = 2 * np.pi * x[..., None] * self.k                               # (B, d, K)
        return torch.cat([x, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], dim=-1)

    def forward(self, x):
        return self.net(self.features(x)) * self.out_scale


class GroupedMLP(_torch().nn.Module):
    """G independent Fourier MLPs evaluated on the same inputs: (B, in) -> (B, G * out) (group-major)."""

    def __init__(self, groups: int, in_dim: int, out_dim: int, hidden: int = 32, layers: int = 2, n_freq: int = 4,
                 freq_max: float = 2.0, out_scale: Sequence[float] | None = None, seed: int = 0):
        torch = _torch()
        nn = torch.nn
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.G, self.out_dim = groups, out_dim
        self.register_buffer("k", torch.linspace(0.5, freq_max, n_freq))
        w = in_dim * (1 + 2 * n_freq)
        dims = [w] + [hidden] * layers + [out_dim]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for a, b in zip(dims[:-1], dims[1:], strict=False):
            self.W.append(nn.Parameter(torch.randn(groups, a, b, generator=g) * np.sqrt(1.0 / a)))
            self.b.append(nn.Parameter(torch.zeros(groups, 1, b)))
        self.register_buffer("out_scale", torch.ones(out_dim) if out_scale is None else
                             torch.as_tensor(np.asarray(out_scale, float), dtype=torch.float32))

    def forward(self, x):
        torch = _torch()
        a = 2 * np.pi * x[..., None] * self.k
        z = torch.cat([x, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], dim=-1)
        z = z[None].expand(self.G, -1, -1)                                  # (G, B, w)
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=False)):
            z = torch.bmm(z, W) + b
            if i < len(self.W) - 1:
                z = torch.tanh(z)
        z = z * self.out_scale
        return z.permute(1, 0, 2).reshape(x.shape[0], -1)                   # (B, G * out)


def _grad(y, x):
    torch = _torch()
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


# ---------------------------------------------------------------------------
# Recharge oscillator: identification
# ---------------------------------------------------------------------------

def identify_recharge(T: np.ndarray, h: np.ndarray, month: np.ndarray | None = None, *,
                      seasonal: Sequence[str] = (), closure: bool = False, epochs: int = 4000, n_col: int = 2048,
                      hidden: int = 64, layers: int = 3, n_freq: int = 32, lr: float = 2e-3, seed: int = 0,
                      init: O.ROParams | None = None, freq_period: float = 6.0) -> dict[str, Any]:
    """PINN identification of the recharge oscillator on monthly series (T, h) of length n.

    Time is scaled to s = t / n in [0, 1]; the network outputs (T, h) in data units; the residual of each equation
    is divided by that variable's standard deviation (per month). Learned: R, eps (free sign), F1, F2 (positive,
    log-space), annual harmonics of the coefficients in ``seasonal``, and with ``closure`` a 1-16-16-1 network
    g(h / sd_h) * sd_T added to dT/dt (its last layer starts at zero). ``init`` gives starting values (default:
    R = -0.05, F1 = 0.02 sd_T / sd_h, F2 = 1.5 sd_h / sd_T, eps = 0.03, which is not the answer).

    Returns the identified :class:`~.oscillators.ROParams` (noise from one-month integrated residuals), the data
    RMSE per variable, the residual RMS on 4096 fresh collocation points, the history, and the trained modules.
    """
    torch = _torch()
    nn = torch.nn
    from pinneapple_neural.trainer.inverse_pinn import InverseParam, InversePINN
    torch.manual_seed(seed)
    T, h = np.asarray(T, float), np.asarray(h, float)
    n = len(T)
    month = np.zeros(n) if month is None else np.asarray(month, float)
    sd = np.array([T.std(), h.std()])
    s_obs = torch.as_tensor((np.arange(n) / n)[:, None], dtype=torch.float32)
    y_obs = torch.as_tensor(np.column_stack([T, h]), dtype=torch.float32)
    p0 = init.mean() if init is not None else {"R": -0.05, "F1": 0.02 * sd[0] / sd[1] * 10, "F2": 1.5 * sd[1] / sd[0]
                                               / 10, "eps": 0.03}
    params = [InverseParam(p0["R"], name="R"), InverseParam(abs(p0["F1"]), log_space=True, name="F1"),
              InverseParam(abs(p0["F2"]), log_space=True, name="F2"), InverseParam(p0["eps"], name="eps")]
    for nm in seasonal:
        params += [InverseParam(0.0, name=f"{nm}_c"), InverseParam(0.0, name=f"{nm}_s")]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = FourierMLP(1, 2, hidden, layers, n_freq, out_scale=sd, freq_max=n / freq_period)
            self.g = None
            if closure:
                self.g = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
                nn.init.zeros_(self.g[-1].weight)
                nn.init.zeros_(self.g[-1].bias)

        def forward(self, s):
            return self.x(s)

    model = Model()
    m0 = float(month[0])
    sdt = torch.as_tensor(sd, dtype=torch.float32)

    def coef(p, name, mon):
        v = p[name]
        if name in seasonal:
            w = 2 * np.pi / 12.0
            v = v + p[f"{name}_c"] * torch.cos(w * mon) + p[f"{name}_s"] * torch.sin(w * mon)
        return v

    def residual(mdl, s, p):
        X = mdl(s)
        dX = torch.stack([_grad(X[:, 0:1], s)[:, 0], _grad(X[:, 1:2], s)[:, 0]], 1) / n     # per month
        mon = m0 + s[:, 0] * n
        Tn, hn = X[:, 0], X[:, 1]
        fT = coef(p, "R", mon) * Tn + coef(p, "F1", mon) * hn
        if mdl.g is not None:
            fT = fT + mdl.g(hn[:, None] / sdt[1])[:, 0] * sdt[0]
        fh = -coef(p, "F2", mon) * Tn - coef(p, "eps", mon) * hn
        return [(dX[:, 0] - fT) / sdt[0], (dX[:, 1] - fh) / sdt[1]]

    ip = InversePINN(model, params, residual, s_obs, y_obs, domain_bounds=[(0.0, 1.0)], n_col=n_col, lr=lr,
                     lr_params=5e-3, patience=epochs)
    ip.rng = np.random.default_rng(seed)
    hist = ip.train(n_epochs=epochs)
    got = ip.get_identified_params()
    coef_ = {k: (got[k], got.get(f"{k}_c", 0.0), got.get(f"{k}_s", 0.0)) for k in O.PARAM_NAMES}
    p = O.ROParams(coef_)
    with torch.no_grad():
        fit = model(s_obs).numpy()
    data_rmse = np.sqrt(np.mean((fit - np.column_stack([T, h])) ** 2, axis=0))
    s_new = torch.rand(4096, 1, generator=torch.Generator().manual_seed(seed + 1)).requires_grad_(True)
    r = residual(model, s_new, {pp.name: pp.value for pp in ip.inverse_params})
    res_rms = [float(torch.sqrt(torch.mean(x ** 2)).item()) for x in r]
    g_np = None
    if closure:
        def g_np(hv, _g=model.g, _sd=sd):
            with torch.no_grad():
                z = torch.as_tensor(np.asarray(hv, float).reshape(-1, 1) / _sd[1], dtype=torch.float32)
                return _g(z).numpy().reshape(np.shape(hv)) * _sd[0]
    # noise from the one-month integrated residuals of the identified model
    X = np.column_stack([T, h])
    one = _integrate(p, X[:-1], 1, month[:-1], g_np)[:, -1]
    res = X[1:] - one
    p.sigma_T, p.sigma_h = float(res[:, 0].std()), float(res[:, 1].std())
    p.info = {"identified_by": "pinn", "eigen": O._eig_json(p)}
    return {"params": p, "data_rmse": data_rmse.tolist(), "residual_rms_per_sd": res_rms,
            "history": {k: v for k, v in hist.items() if k != "params"}, "param_history": hist["params"],
            "model": model, "closure": g_np, "epochs_run": len(hist["total"])}


def _integrate(p: O.ROParams, X0, n_months, month0, closure=None, sub: int = 4, noise: bool = False,
               n_members: int = 1, seed: int = 0):
    """RK4 (+ optional Euler-Maruyama noise) of the recharge oscillator with an optional closure g(h) in dT/dt;
    returns (n_members, n, n_months + 1, 2) when ``noise`` else (n, n_months + 1, 2)."""
    if closure is None:
        out = O.simulate_recharge(p, X0, n_months, sub=sub, stochastic=noise, n_members=n_members if noise else 1,
                                  seed=seed, month_offsets=month0)
        return out if noise else out[0]
    rng = np.random.default_rng(seed)
    X = np.broadcast_to(np.atleast_2d(X0), (n_members if noise else 1,) + np.atleast_2d(X0).shape).copy()
    m = np.asarray(month0, float) + np.zeros(X.shape[1])
    dt = 1.0 / sub
    out = np.empty(X.shape[:2] + (n_months + 1, 2))
    out[:, :, 0] = X

    def f(Z, mm):
        d = O.ro_tendency(p, Z, mm)
        d[..., 0] = d[..., 0] + closure(Z[..., 1])
        return d

    with np.errstate(all="ignore"):
        for k in range(n_months):
            for _ in range(sub):
                k1 = f(X, m)
                k2 = f(X + 0.5 * dt * k1, m + 0.5 * dt)
                k3 = f(X + 0.5 * dt * k2, m + 0.5 * dt)
                k4 = f(X + dt * k3, m + dt)
                X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if noise:
                    X = X + np.sqrt(dt) * np.stack([p.sigma_T * rng.standard_normal(X.shape[:-1]),
                                                    p.sigma_h * rng.standard_normal(X.shape[:-1])], -1)
                m = m + dt
            out[:, :, k + 1] = X
    return out if noise else out[0]


class PINNRechargeForecaster(Forecaster):
    """Forecasts by integrating the recharge oscillator identified by :func:`identify_recharge` on the training
    segment (``closure``: with the learned thermocline-feedback network). ``stochastic``: ensemble with the noise
    of the one-month residuals; otherwise deterministic with a calibrated Gaussian spread."""
    name, family = "pinn_ro", "data+physics"

    def __init__(self, h_col: str = "wwv", seasonal: Sequence[str] = (), closure: bool = False,
                 stochastic: bool = False, n_members: int = 200, epochs: int = 4000, seed: int = 0,
                 name: str | None = None, family: str | None = None):
        super().__init__()
        self.h_col, self.seasonal, self.closure, self.stochastic = h_col, tuple(seasonal), closure, stochastic
        self.n_members, self.epochs, self.seed = int(n_members), int(epochs), int(seed)
        self.probabilistic = bool(stochastic)
        self.name = name or ("pinn_ro_closure" if closure else "pinn_ro")
        self.family = family or self.family

    def _fit(self, data, train_origins, val_origins):
        end = train_end(train_origins, self.horizons)
        names = list(data.meta["columns"])
        self.hc_ = names.index(self.h_col)
        moy = np.asarray(data.meta["month_of_year"])
        r = identify_recharge(data.y[:end], data.obs[:end, self.hc_], moy[:end], seasonal=self.seasonal,
                              closure=self.closure, epochs=self.epochs, seed=self.seed)
        self.p_, self.g_ = r["params"], r["closure"]
        self.n_net_ = int(sum(q.numel() for q in r["model"].parameters()))
        ls = O.fit_recharge(data.y[:end], data.obs[:end, self.hc_], moy[:end], seasonal=self.seasonal)
        self.info = {"params": self.p_.to_dict(), "least_squares_params": ls.to_dict(),
                     "data_rmse": r["data_rmse"], "residual_rms_per_sd": r["residual_rms_per_sd"],
                     "epochs_run": r["epochs_run"], "final_loss": r["history"]["total"][-1]}
        if self.closure:
            hv = np.linspace(np.percentile(data.obs[:end, self.hc_], 1), np.percentile(data.obs[:end, self.hc_], 99),
                             41)
            self.info["closure_curve"] = {"h": hv.tolist(), "g": self.g_(hv).tolist()}

    def _paths(self, data, origins):
        moy = np.asarray(data.meta["month_of_year"])[origins]
        X0 = np.column_stack([data.y[origins], data.obs[origins, self.hc_]])
        H = int(self.horizons.max())
        out = _integrate(self.p_, X0, H, moy, self.g_, noise=self.stochastic, n_members=self.n_members,
                         seed=self.seed)
        if not self.stochastic:
            out = out[None]
        s = out[:, :, self.horizons, 0]
        bad = ~np.isfinite(s) | (np.abs(s) > 10)
        self.info["diverged_fraction"] = float(bad.mean())
        return np.where(bad, 0.0, s)

    def predict(self, data, origins):
        origins = np.asarray(origins, int)
        s = self._paths(data, origins)
        if self.stochastic:
            return Forecast(s.mean(0), s.std(0), s)
        std = None if self.sigma_ is None else np.broadcast_to(self.sigma_, s[0].shape).copy()
        return Forecast(s[0], std)

    def _predict_mean(self, data, origins):
        return self._paths(data, np.asarray(origins, int)).mean(0)

    def n_params(self):
        return 4 + 2 * len(self.seasonal) + 2 + (self.n_net_ if self.closure else 0)


class PINNExtrapolationForecaster(Forecaster):
    """For each origin t: a PINN X(tau), tau = (s - t) / window in [-1, max(h) / window], fitted to the observed
    (T, h) of the last ``window`` months and to the recharge-oscillator residual (coefficients fitted by least
    squares on the training segment, held fixed) on the whole interval; the forecast at t + h is X(h / window).
    All origins of a call train together as one :class:`GroupedMLP` (chunks of ``chunk``)."""
    name, family = "pinn_extrapolation", "data+physics"

    def __init__(self, h_col: str = "wwv", window: int = 24, epochs: int = 3000, n_col: int = 512, hidden: int = 32,
                 chunk: int = 256, seed: int = 0, name: str | None = None):
        super().__init__()
        self.h_col, self.window, self.epochs, self.n_col = h_col, int(window), int(epochs), int(n_col)
        self.hidden, self.chunk, self.seed = int(hidden), int(chunk), int(seed)
        self.name = name or self.name

    def _fit(self, data, train_origins, val_origins):
        end = train_end(train_origins, self.horizons)
        self.hc_ = list(data.meta["columns"]).index(self.h_col)
        moy = np.asarray(data.meta["month_of_year"])
        self.p_ = O.fit_recharge(data.y[:end], data.obs[:end, self.hc_], moy[:end])
        self.sd_ = np.array([data.y[:end].std(), data.obs[:end, self.hc_].std()])
        self.info = {"params": self.p_.to_dict(), "window": self.window, "epochs": self.epochs}
        self._cache = {}

    def _run_chunk(self, data, origins):
        torch = _torch()
        from pinneapple_neural.trainer.inverse_pinn import InverseParam, InversePINN
        W, H = self.window, int(self.horizons.max())
        G = len(origins)
        sd = torch.as_tensor(self.sd_, dtype=torch.float32)
        model = GroupedMLP(G, 1, 2, hidden=self.hidden, layers=2, n_freq=4, freq_max=2.0, out_scale=self.sd_,
                           seed=self.seed)
        lag = np.arange(-W + 1, 1)
        tau_obs = torch.as_tensor((lag / W)[:, None], dtype=torch.float32)
        rows = np.asarray(origins)[None, :] + lag[:, None]                  # (W, G)
        Y = np.stack([data.y[rows], data.obs[rows, self.hc_]], -1)          # (W, G, 2)
        y_obs = torch.as_tensor(Y.reshape(W, -1), dtype=torch.float32)
        moy = torch.as_tensor(np.asarray(data.meta["month_of_year"])[origins], dtype=torch.float32)
        c = self.p_.coef
        w = 2 * np.pi / 12.0

        def k(name, mon):
            a, b, cc = c[name]
            return a + b * torch.cos(w * mon) + cc * torch.sin(w * mon)

        def residual(mdl, s, p):
            X = mdl(s).view(len(s), G, 2)
            mon = moy[None, :] + s * W                                       # (B, G) calendar month
            dXm = _per_group_derivative(mdl, s, G) / W                       # (B, G, 2) per month
            fT = k("R", mon) * X[..., 0] + k("F1", mon) * X[..., 1]
            fh = -k("F2", mon) * X[..., 0] - k("eps", mon) * X[..., 1]
            return [(dXm[..., 0] - fT) / sd[0], (dXm[..., 1] - fh) / sd[1]]

        dummy = InverseParam(0.0, name="unused")
        ip = InversePINN(model, [dummy], residual, tau_obs, y_obs, domain_bounds=[(-1.0, H / W)], n_col=self.n_col,
                         lr=3e-3, lr_params=0.0, patience=self.epochs)
        ip.rng = np.random.default_rng(self.seed)
        hist = ip.train(n_epochs=self.epochs)
        tau_f = torch.as_tensor((self.horizons / W)[:, None], dtype=torch.float32)
        with torch.no_grad():
            pred = model(tau_f).view(len(self.horizons), G, 2)[..., 0].T.numpy()
        s_new = (torch.rand(1024, 1) * (1 + H / W) - 1).requires_grad_(True)
        r = residual(model, s_new, {})
        future = (s_new.detach()[:, 0] > 0)
        rr = torch.sqrt(torch.mean(r[0][future] ** 2)).item(), torch.sqrt(torch.mean(r[0][~future] ** 2)).item()
        with torch.no_grad():
            fit = model(tau_obs).view(W, G, 2).numpy()
        data_rmse = np.sqrt(np.mean((fit - Y) ** 2, axis=(0, 1)))
        return pred, {"residual_rms_forecast_part": rr[0], "residual_rms_data_part": rr[1],
                      "data_rmse": data_rmse.tolist(), "final_loss": hist["total"][-1]}

    def _predict_mean(self, data, origins):
        origins = np.asarray(origins, int)
        out = np.empty((len(origins), len(self.horizons)))
        todo = [i for i, o in enumerate(origins) if int(o) not in self._cache]
        diag = []
        for i in range(0, len(todo), self.chunk):
            idx = todo[i:i + self.chunk]
            pred, d = self._run_chunk(data, origins[idx])
            diag.append(d)
            for j, ii in enumerate(idx):
                self._cache[int(origins[ii])] = pred[j]
        for i, o in enumerate(origins):
            out[i] = self._cache[int(o)]
        if diag:
            self.info.setdefault("chunks", []).extend(diag)
        return out

    def n_params(self):
        return 0


def _per_group_derivative(model, s, G):
    """d model / d s for each group and output, (B, G, out): since every group is a separate scalar function of
    the shared input s, the derivative of the sum over the batch dimension of output j w.r.t. s is per sample; we
    take one VJP per (group, output) column -- 2 G backward passes -- through a column-sum trick: outputs of
    different groups depend on disjoint weights but on the same s, so we evaluate the model on G copies of s."""
    torch = _torch()
    B = s.shape[0]
    sg = s.detach()[None].expand(G, B, 1).clone().requires_grad_(True)     # (G, B, 1): one copy per group
    a = 2 * np.pi * sg[..., None] * model.k
    z = torch.cat([sg, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], dim=-1)
    for i, (W, b) in enumerate(zip(model.W, model.b, strict=False)):
        z = torch.bmm(z, W) + b
        if i < len(model.W) - 1:
            z = torch.tanh(z)
    z = z * model.out_scale                                                 # (G, B, out)
    outs = []
    for j in range(z.shape[-1]):
        d = torch.autograd.grad(z[..., j].sum(), sg, create_graph=True)[0][..., 0]     # (G, B)
        outs.append(d.T)
    return torch.stack(outs, -1)                                            # (B, G, out)


# ---------------------------------------------------------------------------
# Equatorial wave (long-wave shallow-water) PINN
# ---------------------------------------------------------------------------

def project_forcing(taux: np.ndarray, lat: np.ndarray, c: float = 2.7, H: float = 150.0, rho: float = 1025.0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Projections X0, X2 of X = tau_x / (rho H) [m s-2] (..., nlat, nx) on phi_0 = exp(-y^2/2) and
    phi_2 = (4 y^2 - 2) exp(-y^2/2), y = lat / L with L = sqrt(c / beta) (norms sqrt(pi) and 8 sqrt(pi))."""
    from .intermediate import BETA, DEG, _phi
    y = np.asarray(lat) * DEG / np.sqrt(c / BETA)
    P = _phi(2, y)
    w = np.gradient(y)
    X = np.asarray(taux, float) / (rho * H)
    X0 = np.einsum("...yx,y,y->...x", X, P[0], w) / np.sqrt(np.pi)
    X2 = np.einsum("...yx,y,y->...x", X, P[2], w) / (8 * np.sqrt(np.pi))
    return X0, X2


class EquatorialWavePINN:
    """PINN for the two-mode long-wave equatorial ocean plus an equatorial SST equation (see the module docstring).

    ``lon_obs`` (nb,) and ``months`` define the observation grid of ``h_obs`` and ``T_obs`` (nt, nb) [m, K];
    ``X0``, ``X2`` (nt, nx) and ``tau_eq`` (nt, nx) are the projected forcing and the equatorial stress on
    ``lon_f``; the basin spans [lon_w, lon_e]. ``sst=False`` fits the ocean only (h data).
    """

    def __init__(self, lon_obs, h_obs, T_obs, lon_f, X0, X2, tau_eq, *, lon_w: float = 120.0, lon_e: float = 280.0,
                 H: float = 150.0, sst: bool = True, c0: float = 2.0, r0_days: float = 365.0, hidden: int = 96,
                 layers: int = 4, n_freq: int = 12, freq_max: float | None = None, learn: Sequence[str] = ("c", "r_T", "a_W"),
                 fixed: dict[str, float] | None = None, pde_weight: float = 1.0, seed: int = 0):
        from .intermediate import DEG, SECONDS_PER_MONTH
        self.pde_weight = float(pde_weight)
        self.learn = tuple(learn)
        self.fixed = {"s": 1.0, "h_scale": 1.0, **(fixed or {})}
        self.lon_obs, self.h_obs, self.T_obs = np.asarray(lon_obs, float), np.asarray(h_obs, float), \
            None if T_obs is None else np.asarray(T_obs, float)
        self.lon_f, self.X0, self.X2, self.tau_eq = np.asarray(lon_f, float), np.asarray(X0, float), \
            np.asarray(X2, float), np.asarray(tau_eq, float)
        self.lon_w, self.lon_e, self.H, self.sst = lon_w, lon_e, H, sst
        self.Lx = (lon_e - lon_w) * DEG
        self.nt = len(h_obs)
        self.Tt = self.nt * SECONDS_PER_MONTH
        self.c0, self.r0_days, self.seed = c0, r0_days, seed
        self.hp = dict(hidden=hidden, layers=layers, n_freq=n_freq, freq_max=freq_max)

    def _interp(self, F, xh, th):
        """Bilinear interpolation of a (nt, nx) forcing at scaled points (x in [0,1] of the basin, t in [0,1] of
        the window; months at cell centres)."""
        torch = _torch()
        lon = self.lon_w + xh * (self.lon_e - self.lon_w)
        fx = (lon - self.lon_f[0]) / (self.lon_f[1] - self.lon_f[0])
        ft = th * self.nt - 0.5
        fx = fx.clamp(0, len(self.lon_f) - 1.001)
        ft = ft.clamp(0, self.nt - 1.001)
        i0, j0 = ft.floor().long(), fx.floor().long()
        wt, wx = ft - i0, fx - j0
        Ft = torch.as_tensor(F, dtype=torch.float32)
        return ((1 - wt) * (1 - wx) * Ft[i0, j0] + (1 - wt) * wx * Ft[i0, j0 + 1] + wt * (1 - wx) * Ft[i0 + 1, j0]
                + wt * wx * Ft[i0 + 1, j0 + 1])

    def fit(self, epochs: int = 6000, n_col: int = 4096, lr: float = 2e-3) -> dict[str, Any]:
        """Train; returns the identified parameters, the residual RMS of each equation on 8192 fresh points (in
        units of the typical tendency), and the data RMSE and correlation."""
        torch = _torch()
        nn = torch.nn
        from pinneapple_neural.trainer.inverse_pinn import InverseParam, InversePINN
        torch.manual_seed(self.seed)
        hs = float(np.nanstd(self.h_obs))
        q_scale = hs * self.c0 / self.H                                    # q (m/s) of a typical h anomaly
        Ts = float(np.nanstd(self.T_obs)) if self.sst else 1.0
        H, Lx, Tt, nt, sst, pinn = self.H, self.Lx, self.Tt, self.nt, self.sst, self
        w_sqrt = float(np.sqrt(self.pde_weight))      # InversePINN balances the terms within [0.01, 100] x 100
        start = {"c": self.c0, "r_T": Tt / (self.r0_days * 86400), "a_W": 1.5, "s": 1.0, "h_scale": 1.0,
                 **self.fixed}
        params = [InverseParam(start[n], log_space=True, name=n) for n in ("c", "r_T", "a_W", "s", "h_scale")
                  if n in self.learn]
        if sst:
            params += [InverseParam(0.3, log_space=True, name="eps_T"), InverseParam(0.0, name="kappa")]

        class _Const:
            def __init__(self, v):
                self.value = torch.tensor(float(v))
        P = {p.name: p for p in params}
        for n in ("c", "r_T", "a_W", "s", "h_scale"):
            if n not in P:
                P[n] = _Const(start[n])
        consts = {n: P[n].value for n in P if isinstance(P[n], _Const)}
        out_scale = [q_scale, q_scale] + ([Ts] if sst else [])
        sc_obs = torch.as_tensor([hs] + ([Ts] if sst else []), dtype=torch.float32)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = FourierMLP(2, len(out_scale), pinn.hp["hidden"], pinn.hp["layers"], pinn.hp["n_freq"],
                                      out_scale=out_scale, freq_max=pinn.hp["freq_max"] or max(4.0, nt / 6.0))
                self.g = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
                nn.init.zeros_(self.g[-1].weight)
                nn.init.zeros_(self.g[-1].bias)
                self._p = [P]                                              # a list: not registered twice

            def fields(self, X):
                return self.net(X)

            def h_eq(self, f):
                p = self._p[0]
                return (H / p["c"].value) * (0.5 * f[:, 0] + f[:, 1]) * p["h_scale"].value

            def forward(self, X):                                          # observation equivalents / their std
                f = self.net(X)
                o = [self.h_eq(f)] + ([f[:, 2]] if sst else [])
                return torch.stack(o, 1) / sc_obs

        model = Model()

        def residual(mdl, X, p):
            p = {**consts, **p}
            c = p["c"]
            f = mdl.fields(X)
            q0, q2 = f[:, 0:1], f[:, 1:2]
            g0, g2 = _grad(q0, X), _grad(q2, X)
            xh, th = X[:, 0].detach(), X[:, 1].detach()
            X0 = pinn._interp(pinn.X0, xh, th)
            X2 = pinn._interp(pinn.X2, xh, th)
            r = p["r_T"] / Tt
            sc = q_scale / Tt                                              # typical dq/dt
            rk = (g0[:, 1] / Tt + c * g0[:, 0] / Lx + r * q0[:, 0] - p["s"] * X0) / sc
            rr = (g2[:, 1] / Tt - c / 3 * g2[:, 0] / Lx + r * q2[:, 0] - p["s"] * (2 * X2 - X0) / 6) / sc
            Xw = torch.stack([torch.zeros_like(th), th], 1)
            Xe = torch.stack([torch.ones_like(th), th], 1)
            fw, fe = mdl.fields(Xw), mdl.fields(Xe)
            out = [rk, rr, (fe[:, 1] - 0.5 * fe[:, 0]) / q_scale, (fw[:, 0] - p["a_W"] * fw[:, 1]) / q_scale]
            if sst:
                T = f[:, 2:3]
                dTdt = _grad(T, X)[:, 1] / nt                              # K per month
                h = mdl.h_eq(f)
                tau = pinn._interp(pinn.tau_eq, xh, th)
                g = mdl.g(torch.stack([xh, h / hs], 1))[:, 0] * Ts          # learned thermocline feedback, K/month
                out.append((dTdt + p["eps_T"] * T[:, 0] - 10.0 * p["kappa"] * tau - g) / Ts)
            return [o * w_sqrt for o in out]

        xo = (self.lon_obs - self.lon_w) / (self.lon_e - self.lon_w)
        to = (np.arange(self.nt) + 0.5) / self.nt
        Xg, Tg = np.meshgrid(xo, to)
        P_obs = torch.as_tensor(np.column_stack([Xg.ravel(), Tg.ravel()]), dtype=torch.float32)
        V = np.column_stack([self.h_obs.ravel()] + ([self.T_obs.ravel()] if sst else []))
        ok = np.isfinite(V).all(1)
        Y = torch.as_tensor(V[ok], dtype=torch.float32) / sc_obs
        P_learn = [p.name for p in params]
        ip = InversePINN(model, params, residual, P_obs[ok], Y, domain_bounds=[(0.0, 1.0), (0.0, 1.0)], n_col=n_col,
                         lr=lr, lr_params=5e-3, patience=epochs)
        ip.rng = np.random.default_rng(self.seed)
        hist = ip.train(n_epochs=epochs)
        got = ip.get_identified_params()
        self.model_, self.params_, self.hist_ = model, got, hist
        Pn = torch.rand(8192, 2, generator=torch.Generator().manual_seed(self.seed + 3)).requires_grad_(True)
        res = [r / w_sqrt for r in residual(model, Pn, {pp.name: pp.value for pp in ip.inverse_params})]
        names = ["kelvin", "rossby", "east_bc", "west_bc"] + (["sst"] if sst else [])
        got = {**{n: float(v) for n, v in consts.items()}, **got}
        with torch.no_grad():
            pred = (model(P_obs[ok]) * sc_obs).numpy()
        rmse = np.sqrt(np.mean((pred - V[ok]) ** 2, axis=0))
        corr = [float(np.corrcoef(pred[:, j], V[ok][:, j])[0, 1]) for j in range(V.shape[1])]
        out = {"learned": list(P_learn), "params": {"c_m_s": got["c"], "damping_days": Tt / got["r_T"] / 86400, "a_W": got["a_W"],
                          "forcing_scale": got["s"], "h_scale": got["h_scale"],
                          **({"eps_T_per_month": got["eps_T"], "kappa": 10.0 * got["kappa"]} if sst else {})},
               "residual_rms": {n: float(torch.sqrt(torch.mean(r ** 2)).item()) for n, r in zip(names, res,
                                                                                               strict=False)},
               "data_rmse": rmse.tolist(), "data_corr": corr, "epochs_run": len(hist["total"]),
               "final_loss": hist["total"][-1], "loss_history": hist["total"][:: max(1, len(hist["total"]) // 200)]}
        self.diag_ = out
        return out

    def closure_curve(self, lon: float, h: np.ndarray) -> np.ndarray:
        """The learned thermocline-feedback tendency g(x, h) [K/month] at longitude ``lon`` for D20 anomalies h."""
        torch = _torch()
        hs = float(np.nanstd(self.h_obs))
        Ts = float(np.nanstd(self.T_obs))
        xh = (lon - self.lon_w) / (self.lon_e - self.lon_w)
        Z = torch.as_tensor(np.column_stack([np.full(len(h), xh), np.asarray(h) / hs]), dtype=torch.float32)
        with torch.no_grad():
            return self.model_.g(Z)[:, 0].numpy() * Ts

    def predict_grid(self, lon: np.ndarray, n_t: int | None = None) -> dict[str, np.ndarray]:
        """h (D20-equivalent, m) and T (K) on (months, lon) for the fitted window."""
        torch = _torch()
        n_t = n_t or self.nt
        xo = (np.asarray(lon) - self.lon_w) / (self.lon_e - self.lon_w)
        to = (np.arange(n_t) + 0.5) / n_t
        Xg, Tg = np.meshgrid(xo, to)
        P = torch.as_tensor(np.column_stack([Xg.ravel(), Tg.ravel()]), dtype=torch.float32)
        hs = float(np.nanstd(self.h_obs))
        Ts = float(np.nanstd(self.T_obs)) if self.sst else 1.0
        with torch.no_grad():
            o = (self.model_(P) * torch.as_tensor([hs] + ([Ts] if self.sst else []))).numpy()
        out = {"h": o[:, 0].reshape(n_t, len(lon))}
        if self.sst:
            out["T"] = o[:, 1].reshape(n_t, len(lon))
        return out
