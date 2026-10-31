"""Cardiac activation mapping from sparse measurements with a physics-informed network (eikonal model).

The approach of Sahli Costabal, Yang, Perdikaris, Hurtado and Kuhl ("Physics-informed neural networks for cardiac
activation mapping", Front Phys 8:42, 2020), reimplemented from its general description (the paper itself was not
accessible from the session that wrote this, so no setting or number of it is used here):

* two networks on the tissue domain: the activation time ``T(x, y)`` and the conduction velocity ``V(x, y) > 0``;
* loss = mean squared misfit of ``T`` at the measurement sites + the eikonal residual ``(V |grad T| - 1)^2`` at
  random collocation points + a total-variation penalty on ``V`` (piecewise-smooth tissue, scar borders);
* ``V`` is bounded to ``[v_min, v_max]`` by a sigmoid output.

:func:`interpolate_baseline` is the comparison every activation-mapping method should beat: linear interpolation of
the measured times (Delaunay), which is what clinical mapping systems display between electrodes. The conduction
velocity of any map comes from :func:`pinneapple_physics.biology.excitable.conduction_velocity`.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

__all__ = ["EikonalPINNConfig", "EikonalPINN", "interpolate_baseline"]


@dataclass
class EikonalPINNConfig:
    hidden: tuple[int, ...] = (64, 64, 64)
    v_range: tuple[float, float] = (0.05, 2.0)
    w_data: float = 1.0
    w_eikonal: float = 1.0
    w_tv: float = 1e-3
    n_collocation: int = 2000
    n_iters: int = 20_000
    lr: float = 1e-3
    seed: int = 0
    time_budget: float = math.inf


class EikonalPINN:
    """Fit ``T`` and ``V`` on the rectangle ``bounds = ((x0, x1), (y0, y1))`` to measured activation times."""

    def __init__(self, bounds, config: EikonalPINNConfig | None = None):
        self.bounds = np.asarray(bounds, float)
        self.cfg = config or EikonalPINNConfig()
        self.history: list[dict] = []

    def _nets(self, torch):
        nn = torch.nn
        torch.manual_seed(self.cfg.seed)

        def mlp(out_act=None):
            dims = [2, *self.cfg.hidden, 1]
            layers = []
            for a, b in zip(dims[:-1], dims[1:], strict=True):
                layers += [nn.Linear(a, b), nn.Tanh()]
            layers = layers[:-1]
            if out_act is not None:
                layers.append(out_act)
            return nn.Sequential(*layers)

        self.T_net, self.V_net = mlp(), mlp(nn.Sigmoid())

    def _scale(self, X):
        lo, hi = self._lo, self._hi
        return 2 * (X - lo) / (hi - lo) - 1

    def T(self, X):
        return self._t0 + self._tspan * self.T_net(self._scale(X))

    def V(self, X):
        a, b = self.cfg.v_range
        return a + (b - a) * self.V_net(self._scale(X))

    def fit(self, points, times, verbose: bool = False) -> EikonalPINN:
        import torch

        cfg = self.cfg
        self._nets(torch)
        self._lo = torch.tensor(self.bounds[:, 0], dtype=torch.float32)
        self._hi = torch.tensor(self.bounds[:, 1], dtype=torch.float32)
        times = np.asarray(times, float)
        self._t0, self._tspan = float(times.min()), float(max(np.ptp(times), 1e-9))
        P = torch.tensor(np.asarray(points, float), dtype=torch.float32)
        Tm = torch.tensor(times[:, None], dtype=torch.float32)
        opt = torch.optim.Adam(list(self.T_net.parameters()) + list(self.V_net.parameters()), lr=cfg.lr)
        gen = torch.Generator().manual_seed(cfg.seed)
        start = time.time()
        for it in range(cfg.n_iters):
            opt.zero_grad()
            l_data = (((self.T(P) - Tm) / self._tspan) ** 2).mean()
            c = self._lo + (self._hi - self._lo) * torch.rand(cfg.n_collocation, 2, generator=gen)
            c.requires_grad_(True)
            T = self.T(c)
            gT = torch.autograd.grad(T.sum(), c, create_graph=True)[0]
            V = self.V(c)
            l_eik = ((V * gT.norm(dim=1, keepdim=True) - 1.0) ** 2).mean()
            gV = torch.autograd.grad(V.sum(), c, create_graph=True)[0]
            l_tv = gV.abs().sum(dim=1).mean()
            loss = cfg.w_data * l_data + cfg.w_eikonal * l_eik + cfg.w_tv * l_tv
            loss.backward()
            opt.step()
            if it % 1000 == 0 or it == cfg.n_iters - 1:
                rec = {"iter": it, "data": float(l_data.detach()), "eikonal": float(l_eik.detach()),
                       "tv": float(l_tv.detach()), "seconds": time.time() - start}
                self.history.append(rec)
                if verbose:
                    print(rec, flush=True)
            if time.time() - start > cfg.time_budget:
                break
        self.iters_run = it + 1
        return self

    def predict(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Activation time and conduction velocity at ``points`` (n, 2)."""
        import torch

        X = torch.tensor(np.asarray(points, float), dtype=torch.float32)
        with torch.no_grad():
            return self.T(X).numpy().ravel(), self.V(X).numpy().ravel()


def interpolate_baseline(points, times, query) -> np.ndarray:
    """Linear (Delaunay) interpolation of the measured times; nearest neighbour outside their convex hull."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    lin = LinearNDInterpolator(points, times)(query)
    near = NearestNDInterpolator(points, times)(query)
    return np.where(np.isfinite(lin), lin, near)
