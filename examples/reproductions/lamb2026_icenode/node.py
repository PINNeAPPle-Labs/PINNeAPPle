"""Neural ODE for the depositional ice growth law: the two constraints of Lamb et al. (2026), Eqs. 3-8.

The network replaces only the unknown part of the growth law and never sees the mass ratio directly: the model is
integrated with a fixed-step Runge-Kutta 4, so the loss is the distance between the integrated and the measured
``m/m0`` time series (Eq. 5), exactly as in the paper.

Two constraints, the paper's Figure 1a:

``"strong"``
    ``dm/dt = 4 pi r (S_i - 1) G(r, T, S_i, alpha)`` with ``alpha = f_alpha(S_i, T)`` a network with a sigmoid
    output, ``0 <= alpha <= 1`` (Eq. 3). Everything else is the published capacitance model.
``"weak"``
    ``G / G_c = f_G(S_i, T, m)`` with a sigmoid output scaled to ``0 < G/G_c <= 2`` (Eq. 4).

``"none"`` and ``"nelson_baker"`` are the two parameterised baselines of Table 1 (``G = G_c`` and ``alpha`` from
Eq. 10), integrated with the same solver so the comparison isolates the kinetics.

Training follows the paper: AdamW at a base learning rate of 0.01, cosine decay, 500 epochs, MSE over all series
simultaneously with a mask for the 500 s window. ``torchdiffeq`` is not needed: the paper tests fixed-step RK4 as
well as an adaptive DOP853, and RK4 is both the cheaper and the differentiable-by-construction choice here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from pinneapple_physics.cloud import (
    alpha_nelson_baker,
)

# One thread per process: the state here is 286 experiments wide, so every op is small and intra-op parallelism
# costs more than it saves (the same note is in examples/reproductions/sbinn_daneker2022). Run seeds in parallel.
torch.set_num_threads(1)

__all__ = ["KineticsMLP", "IceGrowthODE", "KINETICS", "integrate", "train", "score", "sample_G",
           "TrainConfig", "TrainHistory"]

KINETICS = ("none", "nelson_baker", "strong", "weak")
#: Recognised by :class:`IceGrowthODE`: the four names above, or a callable ``G(r, T, p, m) -> G`` in s/m.

RHO_ICE = 910.0


def _radius(m):
    return (3.0 * m / (4.0 * np.pi * RHO_ICE)) ** (1.0 / 3.0)


# -------------------------------------------------------------------------------------------------------------
# thermodynamic scalar fields, as torch ops so the whole ODE is differentiable
# -------------------------------------------------------------------------------------------------------------
def _esat_ice(T):
    return torch.exp(9.550426 - 5723.265 / T + 3.53068 * torch.log(T) - 0.00728332 * T)


def _dv(T, p):
    return 2.11e-5 * (T / 273.15) ** 1.94 * (101325.0 / p)


def _ka(T):
    return 4.187e-3 * (5.69 + 0.017 * (T - 273.15))


def _G(r, T, p, alpha):
    """``G`` of Eq. 8 with ``Dv*`` of Eq. 9, in torch (alpha may be a tensor)."""
    Mw, R, Ls = 18.0e-3, 8.3144521, 2.837e6
    w = torch.sqrt(8.0 * R * T / (np.pi * Mw))
    dv = _dv(T, p)
    dv_star = dv / (r / (r + 1.04e-7) + 4.0 * dv / (r * alpha.clamp_min(1e-12) * w))
    vapour = R * T / (_esat_ice(T) * dv_star * Mw)
    thermal = (Ls / (_ka(T) * T)) * (Ls * Mw / (R * T) - 1.0)
    return 1.0 / (vapour + thermal)


def _Gc(r, T, p):
    return _G(r, T, p, torch.ones_like(r))


class KineticsMLP(nn.Module):
    """``3 x 50`` fully connected network, ReLU after the first two layers and a sigmoid output (paper, Methods).

    Inputs are the scaled ``(S_i - 1, T)`` for the strong constraint and ``(S_i - 1, T, log10 m)`` for the weak
    one. ``out_scale`` multiplies the sigmoid: 1 for ``alpha``, 2 for ``G/G_c``.
    """

    def __init__(self, n_in: int = 2, out_scale: float = 1.0, width: int = 50, seed: int | None = None):
        super().__init__()
        self.out_scale = float(out_scale)
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, 1))
        if seed is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                for layer in self.net:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_uniform_(layer.weight, a=5 ** 0.5)
                        nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.out_scale * torch.sigmoid(self.net(x))


@dataclass
class IceGrowthODE:
    """The right-hand side of Eq. 7 for a whole ensemble of experiments at once.

    ``Si``, ``T``, ``P`` and ``m0`` are per-experiment tensors; the state is the mass ratio ``y = m/m0``. With
    ``kinetics="strong"`` or ``"weak"`` a :class:`KineticsMLP` closes the unknown part.
    """

    Si: torch.Tensor
    T: torch.Tensor
    P: torch.Tensor
    m0: torch.Tensor
    kinetics: str = "none"
    net: KineticsMLP | None = None

    def __post_init__(self):
        if self.kinetics in ("strong", "weak") and self.net is None:
            raise ValueError(f"kinetics={self.kinetics!r} needs a network")
        self._alpha_nb = (torch.as_tensor(alpha_nelson_baker(self.Si.numpy(), self.T.numpy()), dtype=torch.float32)
                          if self.kinetics == "nelson_baker" else None)
        # scaled network inputs: Si - 1 in [0, 0.76] over the 286 series; T in [208, 238] K; log10 m in [-13, -9]
        self._x_env = torch.stack([(self.Si - 1.0 - 0.35) / 0.25, (self.T - 225.0) / 12.0], dim=1)

    def G(self, r, m=None):
        if callable(self.kinetics):                       # a closed form to test, e.g. the paper's Eq. 6
            return self.kinetics(r, self.T, self.P, m)
        if self.kinetics == "none":
            return _Gc(r, self.T, self.P)
        if self.kinetics == "nelson_baker":
            return _G(r, self.T, self.P, self._alpha_nb)
        if self.kinetics == "strong":
            alpha = self.net(self._x_env).squeeze(-1)
            return _G(r, self.T, self.P, alpha)
        # weak: the network corrects G relative to the continuum value, with a dependence on the crystal mass
        x = torch.cat([self._x_env, ((torch.log10(m.clamp_min(1e-18)) + 11.0) / 2.0).unsqueeze(-1)], dim=1)
        return self.net(x).squeeze(-1) * _Gc(r, self.T, self.P)

    def dmdt(self, y):
        """``d(m/m0)/dt`` in 1/s at mass ratio ``y``."""
        m = self.m0 * y
        r = _radius(m)
        return 4.0 * np.pi * r * (self.Si - 1.0) * self.G(r, m) / self.m0


def integrate(rhs: IceGrowthODE, t_max: float = 1000.0, dt: float = 1.0) -> torch.Tensor:
    """Fixed-step RK4 integration of ``y' = rhs.dmdt(y)`` from ``y(0) = 1``.

    Returns ``m/m0`` at ``t = 0, dt, 2 dt, ..., t_max``, shape ``(n_exp, n_steps + 1)``; column 0 is the initial
    condition and needs no evaluation. Use :func:`on_grid` to line the result up with the 1 Hz observations when
    ``dt > 1``.
    """
    n = rhs.Si.shape[0]
    y = torch.ones(n, dtype=torch.float32)
    out = [y]
    steps = int(round(t_max / dt))
    for _ in range(steps):
        k1 = rhs.dmdt(y)
        k2 = rhs.dmdt(y + 0.5 * dt * k1)
        k3 = rhs.dmdt(y + 0.5 * dt * k2)
        k4 = rhs.dmdt(y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out.append(y)
    return torch.stack(out, dim=1)


def _stride(dt: float) -> int:
    """How many 1 Hz observations one integration step of ``dt`` seconds spans (whole seconds only)."""
    s = int(round(dt))
    if dt < 1.0 or abs(s - dt) > 1e-9:
        raise ValueError("an integration step must be a whole number of seconds, one or more")
    return s


def on_grid(x, dt: float):
    """Take an array sampled at 1 Hz onto the integration grid of step ``dt`` (columns ``0, dt, 2 dt, ...``)."""
    return x[:, ::_stride(dt)]


# -------------------------------------------------------------------------------------------------------------
# training and scoring
# -------------------------------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Everything the paper's Methods section fixes, plus the knobs of a reduced-scale run."""

    epochs: int = 500
    lr: float = 0.01
    dt: float = 1.0
    t_train: float = 500.0
    t_max: float = 1000.0
    width: int = 50
    seed: int = 0
    log_every: int = 50


@dataclass
class TrainHistory:
    losses: list = field(default_factory=list)
    epochs: int = 0
    seconds: float = 0.0
    config: dict = field(default_factory=dict)


def _bundle(data, t_max: float, t_train: float):
    """Per-experiment tensors plus the observed and masked ``m/m0`` on the common 1 s grid."""
    n_t = int(round(t_max)) + 1
    obs = np.full((len(data), n_t), np.nan)
    k = min(n_t, data.mass_ratio.shape[1])
    obs[:, :k] = data.mass_ratio[:, :k]
    t = np.arange(n_t, dtype=float)
    mask = np.isfinite(obs) & (t[None, :] <= t_train)
    obs = np.where(np.isfinite(obs), obs, 0.0)
    to = lambda a, dtype=torch.float32: torch.as_tensor(np.asarray(a), dtype=dtype)  # noqa: E731
    return {"obs": to(obs), "W": to(mask), "Si": to(data.Si), "T": to(data.T), "P": to(data.P), "m0": to(data.m0)}


def train(data, kinetics: str, cfg: TrainConfig | None = None, verbose: bool = False):
    """Train one model (or integrate a baseline) and return ``(model, history, prediction)``."""
    import time

    cfg = cfg or TrainConfig()

    if kinetics not in KINETICS and not callable(kinetics):
        raise ValueError(f"kinetics must be one of {KINETICS} or a callable")
    torch.manual_seed(cfg.seed)
    b = _bundle(data, cfg.t_max, cfg.t_train)
    net = None
    if kinetics in ("strong", "weak"):
        net = KineticsMLP(n_in=2 if kinetics == "strong" else 3, out_scale=1.0 if kinetics == "strong" else 2.0,
                          width=cfg.width, seed=cfg.seed)
    rhs = IceGrowthODE(Si=b["Si"], T=b["T"], P=b["P"], m0=b["m0"], kinetics=kinetics, net=net)
    hist = TrainHistory(config={"kinetics": kinetics if isinstance(kinetics, str) else getattr(kinetics, "__name__", "callable"), **cfg.__dict__})
    t0 = time.time()
    if net is None:
        with torch.no_grad():
            pred = integrate(rhs, cfg.t_max, cfg.dt)
        hist.seconds = time.time() - t0
        hist.losses = [score(pred.numpy(), data, dt=cfg.dt, t_train=cfg.t_train, t_max=cfg.t_max)["sse_short"]]
        return rhs, hist, pred.detach().numpy()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    obs_train = on_grid(b["obs"], cfg.dt)[:, :int(round(cfg.t_train / cfg.dt)) + 1]
    w_train = on_grid(b["W"], cfg.dt)[:, :obs_train.shape[1]]
    for ep in range(cfg.epochs):
        opt.zero_grad()
        pred = integrate(rhs, cfg.t_train, cfg.dt)            # only the training window is needed while fitting
        n_t = pred.shape[1]
        loss = (((pred - obs_train[:, :n_t]) ** 2) * w_train[:, :n_t]).sum()
        loss.backward()
        opt.step()
        sched.step()
        hist.losses.append(loss.detach().item())
        if verbose and (ep % cfg.log_every == 0 or ep == cfg.epochs - 1):
            print(f"  [{kinetics} {ep:4d}] loss {loss.detach().item():.1f}  ({time.time() - t0:.0f} s)", flush=True)
    hist.epochs = cfg.epochs
    with torch.no_grad():
        pred = integrate(rhs, cfg.t_max, cfg.dt)
    hist.seconds = time.time() - t0
    return rhs, hist, pred.detach().numpy()


def sample_G(rhs: IceGrowthODE, data, n_times: int = 20) -> dict:
    """``G`` predicted by a trained model along each experiment's measured mass trajectory, for the symbolic step.

    The crystals grow over the length of each experiment, so the samples span the measured masses (log-spaced from
    ``m0`` to the observed final mass) at every experiment's own ``S_i`` and ``T`` -- the paper's Figure 3a/3b
    sampling, and the data its symbolic regression sees.
    """
    lengths = data.length()
    m_end = np.array([data.m0[i] * np.nanmax(data.mass_ratio[i, :lengths[i]]) for i in range(len(data))])
    frac = np.linspace(0.0, 1.0, n_times)
    masses = np.exp(np.log(data.m0)[:, None] + frac[None, :] *
                    (np.log(m_end) - np.log(data.m0))[:, None])              # (n_exp, n_times)
    m_t = torch.as_tensor(masses.reshape(-1), dtype=torch.float32)
    exp_t = torch.as_tensor(np.repeat(np.arange(len(data)), n_times))
    Si = torch.as_tensor(data.Si, dtype=torch.float32)[exp_t]
    T = torch.as_tensor(data.T, dtype=torch.float32)[exp_t]
    P = torch.as_tensor(data.P, dtype=torch.float32)[exp_t]
    r = _radius(m_t)
    sub = IceGrowthODE(Si=Si, T=T, P=P, m0=torch.ones_like(m_t), kinetics=rhs.kinetics, net=rhs.net)
    with torch.no_grad():
        G = sub.G(r, m_t).numpy()
        Gc = _Gc(r, T, P).numpy()
    return {"m": m_t.numpy().astype(float), "r": r.numpy().astype(float), "G": G.astype(float),
            "Gc": Gc.astype(float), "Si": Si.numpy().astype(float), "T": T.numpy().astype(float),
            "n_exp": len(data), "n_times": n_times}


def score(pred: np.ndarray, data, t_train: float = 500.0, t_max: float = 1000.0, dt: float = 1.0,
          resample_1hz: bool = True) -> dict:
    """MSE loss (summed and per point) and the "best experiment count" of Table 1, for one model.

    ``pred`` holds ``m/m0`` on the integration grid of :func:`integrate` (step ``dt``). The Table 1 losses are sums
    over the 1 Hz observations, so a model integrated at ``dt > 1`` is linearly resampled onto 1 Hz before scoring
    (``resample_1hz=False`` scores on its own grid instead, for a like-for-like comparison between runs).
    """
    if resample_1hz and dt != 1.0:
        n_1hz = min(int(round(pred.shape[1] * dt)), data.mass_ratio.shape[1])
        t_pred = np.arange(pred.shape[1], dtype=float) * dt
        t_1hz = np.arange(n_1hz, dtype=float)
        pred = np.stack([np.interp(t_1hz, t_pred, row) for row in pred])
        dt = 1.0
    obs_all = on_grid(np.where(np.isfinite(data.mass_ratio), data.mass_ratio, np.nan), dt)
    n_t = min(pred.shape[1], obs_all.shape[1])
    obs = obs_all[:, :n_t]
    t = np.arange(n_t, dtype=float) * dt
    finite = np.isfinite(obs)
    err = np.where(finite, (pred[:, :n_t] - np.nan_to_num(obs)) ** 2, 0.0)
    short = t <= t_train
    long = t <= t_max
    per_exp_short = err[:, short].sum(axis=1)
    per_exp_long = err[:, long].sum(axis=1)
    n_short = (finite & short[None, :]).sum()
    n_long = (finite & long[None, :]).sum()
    return {"sse_short": float(per_exp_short.sum()), "sse_long": float(per_exp_long.sum()),
            "mse_short": float(per_exp_short.sum() / max(n_short, 1)),
            "mse_long": float(per_exp_long.sum() / max(n_long, 1)),
            "n_points_short": int(n_short), "n_points_long": int(n_long),
            "per_experiment_short": per_exp_short, "per_experiment_long": per_exp_long}
