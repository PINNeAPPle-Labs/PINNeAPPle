"""Biologically-informed neural networks (BINNs) for reaction-diffusion: learn the diffusivity, growth and delay
terms of ``u_t = T(t) (D(u) u_x)_x + G(u) u`` as networks, from sparse data.

Lagergren, Nardini, Baker, Simpson and Flores, PLoS Comput Biol 16(12):e1008462 (2020), arXiv:2005.13073,
reimplemented from the paper's Methods (no code copied). A :class:`ReactionDiffusionBINN` is a
:class:`~pinneapple_analysis.inverse_problems.missing_term.NeuralTermDiscovery` whose missing terms are functions of
the state (``D(u)``, ``G(u)``) and of time (``T(t)``) instead of the coordinates:

* ``u_MLP(x, t)``: 3 hidden layers of 128 sigmoid units, softplus output times the carrying capacity K;
* ``D_MLP(u / K)``, ``G_MLP(u / K)``, ``T_MLP(t)``: 3 x 32 sigmoid units; outputs softplus x D_max, linear x G_max and
  sigmoid;
* loss = GLS misfit (Eq 6, gamma = 0.2, the t = 0 rows weighted 10) on mini-batches of the training split + PDE
  residual at 10,000 uniformly random (x, t) per step + the constraints of Eq 16 (ranges, monotonicity of D, G, T)
  with the paper's weights alpha_i;
* Adam (default settings), 80 / 20 train / validation split, the network with the best validation GLS kept, early
  stopping after ``patience`` steps without improvement.

After training, :meth:`ReactionDiffusionBINN.forward_gls` solves the PDE with the learned terms
(:func:`pinneapple_physics.biology.cell_migration.solve_reaction_diffusion`) and scores it against the data, which is
how the paper selects among retrainings; :meth:`ReactionDiffusionBINN.mechanistic_guess` fits the closed forms of
Eq 12 to the learned curves (the paper's starting point for the BINN-guided model).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .missing_term import NeuralTermConfig, NeuralTermDiscovery

__all__ = ["BINNConfig", "ReactionDiffusionBINN"]


@dataclass
class BINNConfig(NeuralTermConfig):
    K: float = 1.7e3                 # carrying capacity, cells/mm^2 (output scale of u, input scale of D and G)
    D_range: tuple[float, float] = (0.0, 0.096)     # mm^2/day
    G_range: tuple[float, float] = (-0.48, 2.4)     # 1/day
    gamma: float = 0.2
    ic_weight: float = 10.0
    batch_size: int = 37
    n_collocation: int = 10_000
    n_iters: int = 50_000
    patience: int = 5_000
    val_fraction: float = 0.2
    delay: bool = True
    seed: int = 0
    time_budget: float = math.inf


def _mlp(torch, dims, out_act=None):
    nn = torch.nn
    layers = []
    for a, b in zip(dims[:-1], dims[1:], strict=True):
        layers += [nn.Linear(a, b), nn.Sigmoid()]
    layers = layers[:-1]
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class ReactionDiffusionBINN(NeuralTermDiscovery):
    """``fit(x, t, U)`` with ``U`` of shape (len(x), len(t)) in cells/mm^2, x in mm, t in days."""

    def __init__(self, config: BINNConfig | None = None):
        super().__init__(known_residual_fn=None, config=config or BINNConfig())
        self.terms: dict = {}
        self.u_net = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ networks
    def _build(self, torch):
        nn = torch.nn
        torch.manual_seed(self.config.seed)
        self.u_net = _mlp(torch, [2, 128, 128, 128, 1], nn.Softplus())
        self.terms = {"D": _mlp(torch, [1, 32, 32, 32, 1], nn.Softplus()),
                      "G": _mlp(torch, [1, 32, 32, 32, 1], None)}
        if self.config.delay:
            self.terms["T"] = _mlp(torch, [1, 32, 32, 32, 1], nn.Sigmoid())

    def _u(self, xt):
        return self.config.K * self.u_net(xt)

    def _D(self, u):
        return self.config.D_range[1] * self.terms["D"](u / self.config.K)

    def _G(self, u):
        return self.config.G_range[1] * self.terms["G"](u / self.config.K)

    def _T(self, t):
        return self.terms["T"](t) if "T" in self.terms else 1.0

    # ------------------------------------------------------------------ training
    def fit(self, x, t, U, verbose: bool = False) -> ReactionDiffusionBINN:
        import torch

        cfg = self.config
        self._build(torch)
        X, Tm = np.meshgrid(np.asarray(x, float), np.asarray(t, float), indexing="ij")
        pts = np.column_stack([X.ravel(), Tm.ravel()])
        vals = np.asarray(U, float).ravel()
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(vals))
        n_val = int(round(cfg.val_fraction * len(vals)))
        val_i, tr_i = perm[:n_val], perm[n_val:]
        self._data = (np.asarray(x, float), np.asarray(t, float), np.asarray(U, float))
        P = torch.tensor(pts, dtype=torch.float32)
        V = torch.tensor(vals, dtype=torch.float32)[:, None]
        W = torch.where(P[:, 1:2] == float(np.min(t)), cfg.ic_weight, 1.0)
        lo = torch.tensor([float(np.min(x)), float(np.min(t))])
        hi = torch.tensor([float(np.max(x)), float(np.max(t))])
        params = list(self.u_net.parameters()) + [p for n in self.terms.values() for p in n.parameters()]
        opt = torch.optim.Adam(params, lr=cfg.lr)
        K, Dlo, Dhi = cfg.K, cfg.D_range[0], cfg.D_range[1]
        Glo, Ghi = cfg.G_range
        a1, a2 = 1 / Dhi * 1e10, K / Dhi * 1e10
        a3, a4, a5 = 1 / Ghi * 1e10, K / Ghi * 1e10, 1e10

        def gls(idx):
            u = self._u(P[idx])
            return (W[idx] * ((u - V[idx]) / u.abs().clamp_min(1e-6) ** cfg.gamma) ** 2).mean()

        best, best_state, since = math.inf, None, 0
        start = time.time()
        for it in range(cfg.n_iters):
            opt.zero_grad()
            b = torch.as_tensor(rng.choice(tr_i, size=min(cfg.batch_size, len(tr_i)), replace=False))
            l_gls = gls(b)
            c = lo + (hi - lo) * torch.rand(cfg.n_collocation, 2)
            c.requires_grad_(True)
            u = self._u(c)
            du = torch.autograd.grad(u.sum(), c, create_graph=True)[0]
            ux, ut = du[:, :1], du[:, 1:]
            D = self._D(u)
            G = self._G(u)
            flux = D * ux
            flux_x = torch.autograd.grad(flux.sum(), c, create_graph=True)[0][:, :1]
            tcol = c[:, 1:2]
            Tt = self._T(tcol)
            l_pde = ((ut - (Tt * flux_x + G * u)) ** 2).mean()
            # Eq 16 constraints on the terms (evaluated at the collocation densities)
            dD = torch.autograd.grad(D.sum(), u, create_graph=True)[0]
            dG = torch.autograd.grad(G.sum(), u, create_graph=True)[0]
            l_c = (a1 * (torch.where((D < Dlo) | (D > Dhi), D, torch.zeros_like(D)) ** 2).mean()
                   + a2 * (torch.where(dD < 0, dD, torch.zeros_like(dD)) ** 2).mean()
                   + a3 * (torch.where((G < Glo) | (G > Ghi), G, torch.zeros_like(G)) ** 2).mean()
                   + a4 * (torch.where(dG > 0, dG, torch.zeros_like(dG)) ** 2).mean())
            if "T" in self.terms:
                dT = torch.autograd.grad(Tt.sum(), tcol, create_graph=True)[0]
                l_c = l_c + a5 * (torch.where(dT < 0, dT, torch.zeros_like(dT)) ** 2).mean()
            loss = l_gls + l_pde + l_c
            loss.backward()
            opt.step()
            with torch.no_grad():
                l_val = float(gls(torch.as_tensor(val_i))) if len(val_i) else float(l_gls)
            if l_val < best:
                best, since = l_val, 0
                best_state = ({k: v.clone() for k, v in self.u_net.state_dict().items()},
                              {n: {k: v.clone() for k, v in m.state_dict().items()} for n, m in self.terms.items()})
            else:
                since += 1
            if it % 1000 == 0:
                rec = {"iter": it, "gls": float(l_gls), "pde": float(l_pde), "constraints": float(l_c),
                       "val": l_val, "seconds": time.time() - start}
                self.history.append(rec)
                if verbose:
                    print(rec, flush=True)
            if since >= cfg.patience or time.time() - start > cfg.time_budget:
                break
        if best_state is not None:
            self.u_net.load_state_dict(best_state[0])
            for n, sd in best_state[1].items():
                self.terms[n].load_state_dict(sd)
        self.iters_run = it + 1
        self.best_val = best
        self._term_net = self.terms["D"]
        return self

    # ------------------------------------------------------------------ evaluation
    def predict_term(self, name: str, values) -> np.ndarray:
        """``D`` or ``G`` at densities (cells/mm^2), ``T`` at times (days)."""
        import torch

        v = torch.tensor(np.asarray(values, float).reshape(-1, 1), dtype=torch.float32)
        with torch.no_grad():
            out = {"D": self._D, "G": self._G, "T": self._T}[name](v)
        return np.broadcast_to(np.asarray(out, float).ravel() if not isinstance(out, float) else out,
                               (len(v),)).copy()

    def predict(self, x, t) -> np.ndarray:
        import torch

        X, Tm = np.meshgrid(np.asarray(x, float), np.asarray(t, float), indexing="ij")
        with torch.no_grad():
            u = self._u(torch.tensor(np.column_stack([X.ravel(), Tm.ravel()]), dtype=torch.float32))
        return u.numpy().reshape(X.shape)

    def forward_gls(self, **solver_kw) -> tuple[float, np.ndarray]:
        """GLS error of the PDE solved with the learned terms (the paper's model-selection score)."""
        from pinneapple_physics.biology.cell_migration import gls_error, solve_reaction_diffusion

        x, t, U = self._data
        Tfn = (lambda s: float(self.predict_term("T", [s])[0])) if "T" in self.terms else None
        pred = solve_reaction_diffusion(lambda u: self.predict_term("D", u), lambda u: self.predict_term("G", u),
                                        x, t, U[:, 0], Tfn, **solver_kw)
        return gls_error(pred, U, self.config.gamma), pred

    def mechanistic_guess(self) -> dict[str, float]:
        """Least-squares fit of Eq 12 to the learned curves: ``D0 + D (u/K)^m`` on the observed density range,
        ``r (1 - u/K)`` and the logistic ``1 / (1 + exp(-(beta1 t + beta0)))`` (units mm^2/day, 1/day)."""
        from scipy.optimize import curve_fit

        x, t, U = self._data
        u = np.linspace(float(U.min()), float(U.max()), 200)
        K = self.config.K
        Dv, Gv = self.predict_term("D", u), self.predict_term("G", u)
        (D0, D, m), _ = curve_fit(lambda u, D0, D, m: D0 + D * (u / K) ** m, u, Dv, p0=[0.01, 0.05, 1.0],
                                  bounds=([0, 0, 0], [0.096, 0.096, 4.0]), maxfev=20000)
        r = float(np.clip(np.linalg.lstsq((1 - u / K)[:, None], Gv, rcond=None)[0][0], 0.0, 2.4))
        out = {"D0": float(D0), "D": float(D), "m": float(m), "r": r}
        if "T" in self.terms:
            tt = np.linspace(float(t.min()), float(t.max()), 100)
            Tv = np.clip(self.predict_term("T", tt), 1e-4, 1 - 1e-4)
            b1, b0 = np.polyfit(tt, np.log(Tv / (1 - Tv)), 1)
            out.update(beta0=float(np.clip(b0, -10, 10)), beta1=float(np.clip(b1, -10, 10)))
        return out
