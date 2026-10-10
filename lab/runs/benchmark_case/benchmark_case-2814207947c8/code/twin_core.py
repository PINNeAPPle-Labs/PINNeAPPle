"""Core of the PINNeAPPle Physics Digital Twin (Natal, RN).

Physical system
---------------
A near-surface thermal-mass element of the facility (per unit area), whose temperature
T(t) is observed through Open-Meteo's ``soil_temperature_0_to_7cm`` (ERA5-Land). It is
heated by shortwave radiation S, exchanges heat with the ambient air T_amb through a
wind-dependent convective coefficient, and receives an internal/background source:

    C dT/dt = alpha*S - (k0 + k1*W) (T - T_amb) + Q_int

Divided by C and written per hour (the only identifiable form, since C cannot be
separated from alpha, k0, k1, Q_int with temperature data alone):

    dT/dt = a*S - (b0 + b1*W) (T - T_amb) + q + dq,    a=alpha*3600/C, b=k*3600/C, q=Q_int*3600/C

``dq`` is a what-if internal-heat increment (zero in all observed data).

Windows
-------
A window starts at hour n and covers tau in [0, H] hours (H=6). Its context holds
T(n), T_amb and W at n..n+H (linearly interpolated inside the window), S at n+1..n+H
(Open-Meteo radiation is the mean over the preceding hour, so S[n+k] applies to
(k-1, k]), relative humidity at n..n+H and the hour-of-day of n.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

H = 6
C_ASSUMED = 2.0e6 * 0.07   # J/(m2 K): rho*c of moist soil (~2.0e6 J/m3K) x 0.07 m layer (assumption)
DATA = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_weather(path: Optional[Path] = None) -> pd.DataFrame:
    d = pd.read_csv(path or DATA / "weather_dataset.csv", parse_dates=["time"])
    d["wind_ms"] = d["wind_speed_10m"] / 3.6
    return d


@dataclass
class Windows:
    T0: np.ndarray      # (N,)
    Ta: np.ndarray      # (N, H+1)
    W: np.ndarray       # (N, H+1)  m/s
    S: np.ndarray       # (N, H)    W/m2, interval means
    RH: np.ndarray      # (N, H+1)
    hour: np.ndarray    # (N,)
    Y: np.ndarray       # (N, H)    observed T at n+1..n+H
    start: np.ndarray   # (N,) index into the dataframe
    dq: np.ndarray      # (N,) what-if internal heat increment [K/h], 0 for data

    def __len__(self):
        return len(self.T0)

    def subset(self, m) -> "Windows":
        return Windows(**{k: v[m] for k, v in asdict(self).items()})

    def context(self) -> np.ndarray:
        """Flat feature vector used by every data-driven model (same information for all)."""
        return np.concatenate([
            self.T0[:, None], self.Ta, self.W, self.S, self.RH,
            np.sin(2 * np.pi * self.hour / 24)[:, None], np.cos(2 * np.pi * self.hour / 24)[:, None],
            self.dq[:, None],
        ], axis=1).astype(np.float32)


def make_windows(d: pd.DataFrame, idx: np.ndarray) -> Windows:
    T = d["soil_temperature_0_to_7cm"].to_numpy(np.float32)
    Ta = d["temperature_2m"].to_numpy(np.float32)
    W = d["wind_ms"].to_numpy(np.float32)
    S = d["shortwave_radiation"].to_numpy(np.float32)
    RH = d["relative_humidity_2m"].to_numpy(np.float32)
    idx = idx[idx + H < len(d)]
    k = np.arange(H + 1)
    J = idx[:, None] + k[None, :]
    return Windows(T0=T[idx], Ta=Ta[J], W=W[J], S=S[J[:, 1:]], RH=RH[J],
                   hour=d["time"].dt.hour.to_numpy()[idx].astype(np.float32),
                   Y=T[J[:, 1:]], start=idx, dq=np.zeros(len(idx), np.float32))


def split_indices(d: pd.DataFrame):
    """Chronological split with a 6-h guard so no window crosses a split boundary."""
    y = d["time"].dt.year.to_numpy()
    n = np.arange(len(d))
    ok = np.zeros(len(d), bool)
    ok[: len(d) - H] = y[: len(d) - H] == y[H:]  # window stays inside one calendar year
    tr = n[(y <= 2021) & ok]
    va = n[(y == 2022) & ok]
    te = n[(y >= 2023) & ok]
    return tr, va, te


# ---------------------------------------------------------------------------
# Physics model (grey-box ODE)
# ---------------------------------------------------------------------------

@dataclass
class ThermalParams:
    a: float    # K/h per W/m2
    b0: float   # 1/h
    b1: float   # 1/h per m/s
    q: float    # K/h

    def physical(self, C: float = C_ASSUMED) -> Dict[str, float]:
        """Dimensional parameters for an assumed heat capacity C [J/m2K]."""
        return {"C_J_m2K": C, "alpha": self.a * C / 3600, "k0_W_m2K": self.b0 * C / 3600,
                "k1_W_m2K_per_ms": self.b1 * C / 3600, "Q_int_W_m2": self.q * C / 3600}


def _forcing_at(tau, Ta, W, S):
    """Forcing inside a window at continuous tau (hours). Shapes: tau (N,M); Ta/W (N,H+1); S (N,H)."""
    tc = tau.clamp(0, H - 1e-6)
    k = tc.floor().long()
    w = tc - k
    ta = torch.gather(Ta, 1, k) * (1 - w) + torch.gather(Ta, 1, k + 1) * w
    ww = torch.gather(W, 1, k) * (1 - w) + torch.gather(W, 1, k + 1) * w
    s = torch.gather(S, 1, k)  # interval (k, k+1] uses S[k] (== radiation at hour n+k+1)
    return ta, ww, s


def rhs(T, tau, Ta, W, S, dq, p):
    ta, ww, s = _forcing_at(tau, Ta, W, S)
    a, b0, b1, q = p
    return a * s - (b0 + b1 * ww) * (T - ta) + q + dq


def simulate(T0, Ta, W, S, dq, p, substeps: int = 4) -> torch.Tensor:
    """RK4 integration of the thermal ODE over one window; returns T at tau=1..H, shape (N,H)."""
    N = T0.shape[0]
    T = T0.clone()
    out = []
    h = 1.0 / substeps
    dqc = dq[:, None]
    for k in range(H):
        for j in range(substeps):
            t = torch.full((N, 1), k + j * h, dtype=T.dtype)
            Tc = T[:, None]
            k1 = rhs(Tc, t, Ta, W, S, dqc, p)
            k2 = rhs(Tc + 0.5 * h * k1, t + 0.5 * h, Ta, W, S, dqc, p)
            k3 = rhs(Tc + 0.5 * h * k2, t + 0.5 * h, Ta, W, S, dqc, p)
            k4 = rhs(Tc + h * k3, t + h - 1e-7, Ta, W, S, dqc, p)
            T = T + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)[:, 0]
        out.append(T)
    return torch.stack(out, 1)


def to_t(w: Windows):
    f = lambda x: torch.as_tensor(x, dtype=torch.float32)
    return f(w.T0), f(w.Ta), f(w.W), f(w.S), f(w.dq), f(w.Y)


class PhysicsModel:
    """Grey-box model: parameters identified by multi-step (6-h) shooting least squares."""

    name = "Physics ODE (identified)"

    def __init__(self, params: Optional[ThermalParams] = None):
        self.params = params

    def fit(self, w: Windows, iters: int = 400, seed: int = 0) -> "PhysicsModel":
        torch.manual_seed(seed)
        T0, Ta, W, S, dq, Y = to_t(w)
        raw = torch.tensor([math.log(1e-3), math.log(0.15), math.log(0.02), 0.1], requires_grad=True)
        opt = torch.optim.LBFGS([raw], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

        def params():
            return (raw[0].exp(), raw[1].exp(), raw[2].exp(), raw[3])

        def closure():
            opt.zero_grad()
            loss = ((simulate(T0, Ta, W, S, dq, params()) - Y) ** 2).mean()
            loss.backward()
            return loss

        opt.step(closure)
        a, b0, b1, q = [float(x) for x in params()]
        self.params = ThermalParams(a, b0, b1, q)
        return self

    def ptuple(self):
        p = self.params
        return (torch.tensor(p.a), torch.tensor(p.b0), torch.tensor(p.b1), torch.tensor(p.q))

    @torch.no_grad()
    def predict(self, w: Windows, substeps: int = 4) -> np.ndarray:
        T0, Ta, W, S, dq, _ = to_t(w)
        return simulate(T0, Ta, W, S, dq, self.ptuple(), substeps).numpy()


# ---------------------------------------------------------------------------
# PINN (PINNeAPPle ModifiedMLP, continuous-time over the window)
# ---------------------------------------------------------------------------

class Standardizer:
    def __init__(self, X: np.ndarray):
        self.mu = X.mean(0)
        sd = X.std(0)
        self.sd = np.where(sd < 1e-3, 1.0, sd)  # constant columns (e.g. what-if heat, 0 in data) stay in raw units

    def __call__(self, X):
        return (X - self.mu) / self.sd


class ThermalPINN(nn.Module):
    """T(tau; c) = T0 + tau/H * g([c, tau/H]) -- the initial condition is exact by construction.

    The backbone is PINNeAPPle's ModifiedMLP. Physical parameters (a, b0, b1, q) are
    trainable (inverse problem) and positive where physics demands it.
    """

    def __init__(self, ctx_dim: int, hidden: int = 128, layers: int = 4, init: Optional[ThermalParams] = None):
        super().__init__()
        from pinneapple_neural.architectures.modified_mlp import ModifiedMLP
        self.net = ModifiedMLP(in_dim=ctx_dim + 1, out_dim=1, hidden_dim=hidden, n_layers=layers,
                               n_fourier=32, sigma=1.0)
        p0 = init or ThermalParams(5e-4, 0.5, 0.05, 0.0)  # neutral start, not the least-squares answer
        self.log_a = nn.Parameter(torch.tensor(math.log(p0.a)))
        self.log_b0 = nn.Parameter(torch.tensor(math.log(p0.b0)))
        self.log_b1 = nn.Parameter(torch.tensor(math.log(p0.b1)))
        self.q = nn.Parameter(torch.tensor(float(p0.q)))
        self.scale = 2.0  # K; typical 6-h change magnitude, keeps the net output O(1)

    def params(self):
        return (self.log_a.exp(), self.log_b0.exp(), self.log_b1.exp(), self.q)

    def forward(self, c_norm: torch.Tensor, T0: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """c_norm (N,D), T0 (N,), tau (N,M) -> T (N,M)."""
        N, M = tau.shape
        x = torch.cat([c_norm[:, None, :].expand(N, M, c_norm.shape[1]), (tau / H)[..., None]], -1)
        g = self.net(x.reshape(N * M, -1)).y.reshape(N, M)
        return T0[:, None] + (tau / H) * g * self.scale


def physics_residual(model: ThermalPINN, c_norm, T0, Ta, W, S, dq, tau, params=None):
    """r = dT/dtau - f(T, tau) at collocation times tau (N,M); returns (r, T)."""
    tau = tau.detach().requires_grad_(True)
    T = model(c_norm, T0, tau)
    dT = torch.autograd.grad(T.sum(), tau, create_graph=True)[0]
    f = rhs(T, tau, Ta, W, S, dq[:, None], params or model.params())
    return dT - f, T
