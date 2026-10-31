"""Potential-based surrogates: a network learns the potential, every property is a derivative of it.

:class:`PotentialSurrogate` holds a network ``a(T, rho)`` (specific Helmholtz energy, J/kg) and predicts pressure,
entropy, internal energy, heat capacities and speed of sound from its derivative tree
(:func:`~pinneapple_physics.potential.thermo.helmholtz_T_rho`, or :func:`~.thermo.helmholtz` in ``(T, v)`` for the
convex option), so the predictions satisfy the Maxwell relation and the response identities by construction.
``convex=True`` makes the network input-convex in the specific volume (a partially input-convex network, Amos, Xu &
Kolter, ICML 2017): ``(dp/dv)_T < 0`` everywhere, the mechanical stability of a single phase.

:class:`DirectSurrogate` is the usual alternative: one network with one output per property.

Both are trained by :func:`fit` on *mixed* data -- each sample carries one property (some give ``p``, others
``c_p``, others ``w``) -- with the same loss, optimiser and number of steps, so a comparison at equal budget is one
loop over the two classes.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .core import DTYPE
from .thermo import helmholtz, helmholtz_T_rho

__all__ = ["PROPERTIES", "MixedData", "Scaling", "MLP", "PICNN", "PotentialSurrogate", "DirectSurrogate", "fit",
           "n_parameters"]

#: properties the thermodynamic surrogates predict (specific, SI)
PROPERTIES = ("p", "s", "u", "cv", "cp", "w")


@dataclass
class MixedData:
    """Samples ``(T_k, rho_k)`` each carrying one property ``prop[k]`` with value ``y[k]``."""

    T: np.ndarray
    rho: np.ndarray
    prop: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        self.T, self.rho, self.y = (np.asarray(a, float) for a in (self.T, self.rho, self.y))
        self.prop = np.asarray(self.prop, dtype=object)

    def __len__(self) -> int:
        return len(self.y)

    def counts(self) -> dict[str, int]:
        return {p: int((self.prop == p).sum()) for p in np.unique(self.prop)}


@dataclass
class Scaling:
    """Centres and scales of the inputs and the per-property loss transform.

    The loss compares ``(g(pred) - g(y)) / sigma`` per property, with ``g = log`` for the heat capacities (they span a
    decade near the critical point) and the identity otherwise; ``sigma`` is the spread of ``g(y)`` in the training
    data."""

    T0: float
    sT: float
    r0: float
    sr: float
    v0: float = 0.0
    sv: float = 1.0
    a_scale: float = 1e4
    sigma: dict[str, float] = field(default_factory=dict)
    center: dict[str, float] = field(default_factory=dict)
    floor: dict[str, float] = field(default_factory=dict)
    log_props: tuple[str, ...] = ("cv", "cp")

    @classmethod
    def from_data(cls, data: MixedData) -> Scaling:
        T0, sT = float(data.T.mean()), float(data.T.std())
        r0, sr = float(data.rho.mean()), float(data.rho.std())
        v = 1.0 / data.rho
        s = cls(T0, sT, r0, sr, float(v.mean()), float(v.std()))
        if (data.prop == "p").any():
            s.a_scale = float(data.y[data.prop == "p"].mean() / data.rho[data.prop == "p"].mean())
        for p in np.unique(data.prop):
            if p in s.log_props:
                s.floor[p] = 0.1 * float(data.y[data.prop == p].min())
            g = s.g(p, data.y[data.prop == p])
            s.center[p], s.sigma[p] = float(np.mean(g)), float(np.std(g) or 1.0)
        return s

    def g(self, prop: str, y):
        """Loss transform: ``log`` for the heat capacities, continued linearly below ``floor`` (a tenth of the
        smallest training value) so that a negative prediction is a large error with a useful gradient."""
        if prop not in self.log_props:
            return y
        f = self.floor.get(prop, 0.0)
        if not torch.is_tensor(y):
            y = np.asarray(y, float)
            return np.where(y > f, np.log(np.maximum(y, f)), np.log(f) + (y - f) / f) if f > 0 else np.log(y)
        if f <= 0:
            return torch.log(y)
        return torch.where(y > f, torch.log(torch.clamp(y, min=f)), math.log(f) + (y - f) / f)


class MLP(nn.Module):
    """Fully connected network with a smooth activation (needed for second derivatives)."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 32, layers: int = 3, act: str = "tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh, "softplus": nn.Softplus, "silu": nn.SiLU}
        mods, d = [], n_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), acts[act]()]
            d = hidden
        mods.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*mods).to(DTYPE)

    def forward(self, x):
        return self.net(x)


class PICNN(nn.Module):
    """Partially input-convex network ``f(x, y)``, convex in ``y`` for every ``x``.

    ``z_{k+1} = softplus(W^z_k z_k + W^y_k y + U_k u_k + b_k)`` with ``W^z_k >= 0`` (softplus of free weights) and
    ``u_{k+1} = tanh(V_k u_k + c_k)`` the unconstrained path in ``x``; the output ``w . z_K + a(x) . y + c(x)`` with
    ``w >= 0``. Non-negative combinations and non-decreasing convex activations of convex functions are convex."""

    def __init__(self, n_x: int, n_y: int, hidden: int = 32, layers: int = 3):
        super().__init__()
        self.layers = layers
        self.u = nn.ModuleList([nn.Linear(n_x if k == 0 else hidden, hidden) for k in range(layers)])
        self.zy = nn.ModuleList([nn.Linear(n_y, hidden, bias=False) for _ in range(layers)])
        self.zu = nn.ModuleList([nn.Linear(n_x if k == 0 else hidden, hidden) for k in range(layers)])
        # softplus(-4) ~ 0.018: positive weights start small so the convex path does not grow layer by layer
        self.zz = nn.ParameterList([nn.Parameter(torch.randn(hidden, hidden) * 0.1 - 4.0) for _ in range(layers - 1)])
        self.w = nn.Parameter(torch.randn(hidden) * 0.1 - 4.0)
        self.lin = nn.Linear(hidden, n_y + 1)
        self.to(DTYPE)

    def forward(self, x, y):
        u, z = x, None
        sp = nn.functional.softplus
        for k in range(self.layers):
            pre = self.zy[k](y) + self.zu[k](u)
            if z is not None:
                pre = pre + z @ sp(self.zz[k - 1]).T
            z = sp(pre)
            u = torch.tanh(self.u[k](u))
        ac = self.lin(u)
        return z @ sp(self.w) + (ac[:, :-1] * y).sum(-1) + ac[:, -1]


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class PotentialSurrogate(nn.Module):
    """A network for the specific Helmholtz energy; properties are derivatives of it.

    ``a(T, rho) = c0 + c1 (T - T0) - c2 (T ln(T/T0) - (T - T0)) + a_scale net(x)``. The first three terms are a
    reference with constant heat capacity: ``c0`` and ``c1`` are the free reference of energy and entropy, ``c2`` a
    constant ``c_v``; they start at the training means of ``u``, ``s`` and ``c_v`` (the counterpart of the per-output
    centring of :class:`DirectSurrogate`) and are trained with the network. The network's last layer starts small
    (``init_scale``), so training starts from that reference. ``a_scale`` (J/kg) defaults to ``mean(p)/mean(rho)`` of
    the training data, the size of ``a`` variations that ``p = rho^2 da/drho`` needs.

    ``convex=False``: an :class:`MLP` of the standardised ``(T, rho)`` (tanh). ``convex=True``: a :class:`PICNN`
    convex in the specific volume ``v = 1/rho``, so ``(dp/dv)_T < 0`` everywhere (mechanical stability)."""

    def __init__(self, scaling: Scaling, hidden: int = 32, layers: int = 3, convex: bool = False,
                 a_scale: float | None = None, init_scale: float = 0.01):
        super().__init__()
        self.scaling, self.convex = scaling, convex
        s = scaling
        self.a_scale = float(a_scale if a_scale is not None else s.a_scale)
        s_mean, u_mean = s.center.get("s", 0.0), s.center.get("u", 0.0)
        cv_mean = math.exp(s.center["cv"]) if "cv" in s.center else 0.0
        self.c = nn.Parameter(torch.tensor([u_mean - s.T0 * s_mean, -s_mean, cv_mean], dtype=DTYPE))

        def reference(T):
            return self.c[0] + self.c[1] * (T - s.T0) - self.c[2] * (T * torch.log(T / s.T0) - (T - s.T0))

        if convex:
            self.net = PICNN(1, 1, hidden, layers)
            with torch.no_grad():
                self.net.lin.weight.mul_(init_scale)
                self.net.lin.bias.zero_()
                self.net.w.add_(math.log(init_scale))

            def fn(T, v):
                x = ((T - s.T0) / s.sT)[:, None]
                y = ((v - s.v0) / s.sv)[:, None]
                return reference(T) + self.a_scale * self.net(x, y)

            self.potential = helmholtz(fn, ("T", "v"), mass=True, name="a")
        else:
            self.net = MLP(2, 1, hidden, layers)
            with torch.no_grad():
                self.net.net[-1].weight.mul_(init_scale)
                self.net.net[-1].bias.zero_()

            def fn(T, rho):
                x = torch.stack([(T - s.T0) / s.sT, (rho - s.r0) / s.sr], dim=-1)
                return reference(T) + self.a_scale * self.net(x)[:, 0]

            self.potential = helmholtz_T_rho(fn)

    def properties(self, T: torch.Tensor, rho: torch.Tensor, create_graph: bool = False) -> dict[str, torch.Tensor]:
        """``p, s, u, cv, cp, w`` (specific, SI) at ``(T, rho)`` from the depth-2 tree."""
        if self.convex:
            z = torch.stack([T, 1.0 / rho], dim=1)
        else:
            z = torch.stack([T, rho], dim=1)
        z = z.detach().requires_grad_(True)
        _, q = self.potential.derivatives(z, 2, create_graph=create_graph)
        # speed of sound from w^2, kept finite where an untrained potential gives w^2 <= 0
        if self.convex:
            v = z[:, 1]
            w2 = v ** 2 * (-q["dpdV_T"] + T * q["dpdT_V"] ** 2 / q["cv"])
            out = {"p": q["p"], "s": q["S"], "u": q["U"], "cv": q["cv"], "cp": q["cp"]}
        else:
            w2 = q["dpdrho_T"] + T * q["dpdT_rho"] ** 2 / (rho ** 2 * q["cv"])
            out = {k: q[k] for k in ("p", "s", "u", "cv", "cp")}
        out["w"] = torch.sqrt(torch.clamp(w2, min=1e-6))
        return out

    def predict(self, T, rho) -> dict[str, np.ndarray]:
        T = torch.as_tensor(np.asarray(T, float).reshape(-1), dtype=DTYPE)
        r = torch.as_tensor(np.asarray(rho, float).reshape(-1), dtype=DTYPE)
        return {k: v.detach().numpy() for k, v in self.properties(T, r).items()}


class DirectSurrogate(nn.Module):
    """One network ``(T, rho) -> (p, s, u, cv, cp, w)``, each output scaled back from the training spread."""

    def __init__(self, scaling: Scaling, hidden: int = 32, layers: int = 3, props: Sequence[str] = PROPERTIES):
        super().__init__()
        self.scaling, self.props = scaling, tuple(props)
        self.net = MLP(2, len(self.props), hidden, layers)

    def properties(self, T: torch.Tensor, rho: torch.Tensor, create_graph: bool = False) -> dict[str, torch.Tensor]:
        s = self.scaling
        x = torch.stack([(T - s.T0) / s.sT, (rho - s.r0) / s.sr], dim=-1)
        out = self.net(x)
        res = {}
        for i, p in enumerate(self.props):
            g = s.center.get(p, 0.0) + s.sigma.get(p, 1.0) * out[:, i]
            res[p] = torch.exp(g) if p in s.log_props else g
        return res

    def predict(self, T, rho) -> dict[str, np.ndarray]:
        T = torch.as_tensor(np.asarray(T, float).reshape(-1), dtype=DTYPE)
        r = torch.as_tensor(np.asarray(rho, float).reshape(-1), dtype=DTYPE)
        with torch.no_grad():
            return {k: v.numpy() for k, v in self.properties(T, r).items()}


def _rho(e, robust: bool):
    """Squared error, or the soft-L1 ``2 (sqrt(1 + e^2) - 1)`` (quadratic for small errors, linear for outliers)."""
    return 2 * (torch.sqrt(1 + e ** 2) - 1) if robust else e ** 2


def _loss(model, data_t, scaling: Scaling, create_graph: bool, robust: bool = True):
    T, r, masks, ys = data_t
    q = model.properties(T, r, create_graph=create_graph)
    total, n = 0.0, 0
    for p, m in masks.items():
        pred = q[p][m]
        e = (scaling.g(p, pred) - scaling.g(p, ys[p])) / scaling.sigma[p]
        total = total + (_rho(e, robust)).sum()
        n += int(m.sum())
    return total / n


def fit(model: nn.Module, data: MixedData, *, epochs: int = 3000, lr: float = 3e-3, seed: int = 0,
        lr_final: float = 1e-5, robust: bool = True, clip: float | None = 1.0, log_every: int = 0) -> dict:
    """Full-batch Adam with a cosine learning-rate decay from ``lr`` to ``lr_final``, the same for every model;
    soft-L1 loss on the standardised errors (``robust``) and gradient-norm clipping (``clip``).
    Returns ``{"loss": [...], "seconds": wall-clock, "steps": epochs, "parameters": n}``. The network's initial
    weights are whatever it was built with: seed ``torch.manual_seed`` before building it."""
    torch.manual_seed(seed)
    T = torch.as_tensor(data.T, dtype=DTYPE)
    r = torch.as_tensor(data.rho, dtype=DTYPE)
    masks = {p: torch.as_tensor(data.prop == p) for p in np.unique(data.prop)}
    ys = {p: torch.as_tensor(data.y[data.prop == p], dtype=DTYPE) for p in masks}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_final)
    hist, t0 = [], time.perf_counter()
    for ep in range(epochs):
        opt.zero_grad()
        loss = _loss(model, (T, r, masks, ys), model.scaling, create_graph=True, robust=robust)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"non-finite loss at step {ep}")
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()
        hist.append(loss.item())
        if log_every and ep % log_every == 0:
            print(f"{type(model).__name__} step {ep}: loss {loss.item():.3e}")
    return {"loss": hist, "seconds": time.perf_counter() - t0, "steps": epochs, "parameters": n_parameters(model)}

