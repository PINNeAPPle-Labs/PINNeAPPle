"""Systems-biology informed neural networks (SBINN): parameters and hidden states of an ODE system from partial data.

Yazdani, Lu, Raissi and Karniadakis, PLoS Comput Biol 16(11):e1007575 (2020); the workflow as described by Daneker,
Zhang, Karniadakis and Lu, arXiv:2202.01723 (2022), section 4. Reimplemented here from that description (no code
copied) on any :class:`~pinneapple_physics.biology.ODESystem`:

* one network ``t -> x(t)`` for all states, with an input-scaling layer (``t / T``), a feature layer
  (``t~, sin(t~), ..., sin(K t~)`` with ``t~ = feature_scale * t``) and an output-scaling layer (each state times its
  magnitude);
* unknown parameters as trainable scalars mapped into their search range by ``lo + (hi - lo) (tanh(v) + 1) / 2``;
* loss ``L = L_data + L_ode + L_aux``: mean squared misfit to the observed states, mean squared ODE residual of every
  state at the collocation times, and the misfit of all states at two times (``aux_times``, e.g. the first and last
  measurement) when they are known. Every term is computed in units scaled by the state magnitudes and the time span,
  so the terms are of the same order without hand-tuned weights (``units="scaled"``), or in physical units with
  hand-set ``weights`` and per-state ``ode_weights`` as the paper's code does (``units="raw"``); the ODE term is the
  sum over states of each state's mean squared residual;
* training in two stages, as in the paper: data only (network fit, parameters frozen), then all terms with the
  parameters trainable. Adam throughout.

:class:`SBINNResult` keeps the inferred parameters, their history, the loss history and the network; ``predict``
evaluates the states. The inferred parameters should be re-checked by integrating the ODE with them
(:meth:`SBINNResult.forward_check`), which is what the trust report of an inference rests on.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SBINN", "SBINNConfig", "SBINNResult"]


@dataclass
class SBINNConfig:
    hidden: Sequence[int] = (128, 128, 128)
    activation: str = "swish"
    n_features: int = 5                 # sin(t~), ..., sin(5 t~) as in the paper's Fig 10
    feature_scale: float = 0.01         # t~ = 0.01 t (paper, Fig 10)
    lr: float = 1e-3
    epochs_data: int = 10_000           # stage 1 (paper: 10,000)
    epochs: int = 600_000               # stage 2 (paper: 600,000)
    weights: Mapping[str, float] = field(default_factory=lambda: {"data": 1.0, "ode": 1.0, "aux": 1.0})
    # "scaled": every term in units of the state magnitudes and the time span; "raw": physical units, with
    # ``weights`` and the per-state ``ode_weights`` set by hand as in the paper's code
    units: str = "scaled"
    ode_weights: Mapping[str, float] = field(default_factory=dict)
    record_every: int = 1000
    seed: int = 0
    dtype: str = "float64"
    time_budget: float = math.inf       # seconds; training stops (and says so) when exceeded


@dataclass
class SBINNResult:
    parameters: dict[str, float]
    history: list[dict[str, Any]]
    epochs_run: int
    stopped_by_budget: bool
    seconds: float
    config: SBINNConfig
    _model: Any = None

    def predict(self, t) -> dict[str, np.ndarray]:
        return self._model.predict(np.asarray(t, float))

    def forward_check(self, system, t, observed: Sequence[str] = (), data: Mapping[str, np.ndarray] | None = None,
                      **simulate_kw) -> dict[str, Any]:
        """Integrate ``system`` with the inferred parameters and compare with the data (relative L2 per observed
        state) and with the network's own states (how far the network is from an ODE solution)."""
        traj = system.simulate(np.asarray(t, float), self.parameters, **simulate_kw)
        net = self.predict(t)
        out = {"success": traj.success, "net_vs_ode": {}, "ode_vs_data": {}}
        for s in system.states:
            ref = traj[s]
            out["net_vs_ode"][s] = float(np.linalg.norm(net[s] - ref) / max(np.linalg.norm(ref), 1e-30))
        for s in observed:
            if data is not None and s in data:
                d = np.asarray(data[s], float)
                out["ode_vs_data"][s] = float(np.linalg.norm(traj[s] - d) / max(np.linalg.norm(d), 1e-30))
        out["trajectory"] = traj
        return out


class _Net:
    def __init__(self, n_states, cfg, T, scales, torch):
        nn = torch.nn
        act = {"swish": nn.SiLU, "tanh": nn.Tanh, "silu": nn.SiLU}[cfg.activation]
        dims = [1 + cfg.n_features, *cfg.hidden, n_states]
        layers = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(a, b), act()]
        self.body = nn.Sequential(*layers[:-1])
        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        self.cfg, self.T, self.torch = cfg, T, torch
        self.scales = scales

    def __call__(self, t):
        torch = self.torch
        ts = self.cfg.feature_scale * t
        feats = [t / self.T] + [torch.sin((k + 1) * ts) for k in range(self.cfg.n_features)]
        return self.body(torch.cat(feats, dim=1)) * self.scales


class SBINN:
    """Infer ``unknown`` parameters of ``system`` (with their ``search_range``) from observations of some states.

    ``data`` maps an observed state (or ``<state>`` of the system) to ``(t, values)``; ``t_collocation`` are the ODE
    residual times (default: the union of the data times); ``aux`` maps every state to its known values at
    ``aux_times`` (the paper's L_aux), or is None."""

    def __init__(self, system, unknown: Sequence[str], search_range: Mapping[str, tuple[float, float]],
                 data: Mapping[str, tuple[np.ndarray, np.ndarray]], *, t_collocation=None,
                 aux: Mapping[str, Sequence[float]] | None = None, aux_times: Sequence[float] = (),
                 config: SBINNConfig | None = None):
        import torch

        self.system, self.unknown = system, list(unknown)
        self.range = {k: tuple(map(float, search_range[k])) for k in self.unknown}
        self.cfg = config or SBINNConfig()
        self.torch = torch
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.dt = torch.float64 if self.cfg.dtype == "float64" else torch.float32
        self.data = {k: (np.asarray(t, float).ravel(), np.asarray(v, float).ravel()) for k, (t, v) in data.items()}
        t_all = np.concatenate([t for t, _ in self.data.values()])
        self.t0, self.t1 = float(t_all.min()), float(t_all.max())
        self.T = max(self.t1 - self.t0, 1e-12)
        tc = np.unique(t_all) if t_collocation is None else np.asarray(t_collocation, float).ravel()
        self.tc = torch.tensor(tc[:, None], dtype=self.dt)
        self.aux = {k: np.asarray(v, float) for k, v in (aux or {}).items()}
        self.aux_t = np.asarray(aux_times, float)
        # magnitude of each state: from the data and the aux values, else 1
        mags = []
        for s in system.states:
            vals = [np.abs(self.data[s][1])] if s in self.data else []
            if s in self.aux:
                vals.append(np.abs(self.aux[s]))
            m = max((float(v.max()) for v in vals if v.size), default=1.0)
            mags.append(m if m > 0 else 1.0)
        self.mag = np.asarray(mags)
        self.net = _Net(len(system.states), self.cfg, self.T, torch.tensor(self.mag, dtype=self.dt), torch)
        self.net.body.to(self.dt)
        self.raw = {k: torch.zeros((), dtype=self.dt, requires_grad=True) for k in self.unknown}
        self.rhs = system.torch_rhs(trainable=self.unknown)

    def _m(self, i: int) -> float:
        return 1.0 if self.cfg.units == "raw" else float(self.mag[i])

    def _tscale(self) -> float:
        return 1.0 if self.cfg.units == "raw" else self.T

    def theta(self) -> dict:
        th = self.torch.tanh
        return {k: lo + (hi - lo) * (th(self.raw[k]) + 1) / 2 for k, (lo, hi) in self.range.items()}

    def _losses(self, with_ode: bool):
        torch = self.torch
        out = {}
        sq = []
        for s, (t, v) in self.data.items():
            i = self.system.states.index(s)
            tt = torch.tensor(t[:, None], dtype=self.dt)
            pred = self.net(tt)[:, i]
            sq.append(((pred - torch.tensor(v, dtype=self.dt)) / self._m(i)) ** 2)
        out["data"] = torch.cat(sq).mean()
        if self.aux:
            ta = torch.tensor(self.aux_t[:, None], dtype=self.dt)
            pa = self.net(ta)
            terms = [((pa[:, i] - torch.tensor(self.aux[s], dtype=self.dt)) / self._m(i)) ** 2
                     for i, s in enumerate(self.system.states) if s in self.aux]
            out["aux"] = torch.cat(terms).mean()
        if with_ode:
            # one forward-mode pass gives dx/dt for every state (the input is the scalar t)
            import torch.autograd.forward_ad as fwAD

            t = self.tc
            with fwAD.dual_level():
                dual = fwAD.make_dual(t, torch.ones_like(t))
                Xd = self.net(dual)
                X, dX = fwAD.unpack_dual(Xd)
            x = {s: X[:, i:i + 1] for i, s in enumerate(self.system.states)}
            f = self.rhs(t, x, self.theta())
            ow = self.cfg.ode_weights
            res = [ow.get(s, 1.0) * ((dX[:, i:i + 1] - f[s]) * self._tscale() / self._m(i)) ** 2
                   for i, s in enumerate(self.system.states)]
            out["ode"] = torch.cat(res, dim=1).mean(dim=0).sum()
        return out

    def fit(self, verbose: bool = False) -> SBINNResult:
        torch = self.torch
        cfg = self.cfg
        w = {"data": 1.0, "ode": 1.0, "aux": 1.0, **cfg.weights}
        history: list[dict[str, Any]] = []
        start = time.time()
        stopped = False
        epochs_run = 0
        stages = [(cfg.epochs_data, False), (cfg.epochs, True)]
        for n_ep, with_ode in stages:
            params = list(self.net.body.parameters()) + (list(self.raw.values()) if with_ode else [])
            opt = torch.optim.Adam(params, lr=cfg.lr)
            for ep in range(n_ep):
                opt.zero_grad()
                L = self._losses(with_ode)
                loss = sum(w[k] * v for k, v in L.items())
                loss.backward()
                opt.step()
                epochs_run += 1
                if epochs_run % cfg.record_every == 0 or (with_ode and ep == n_ep - 1):
                    rec = {"epoch": epochs_run, "stage": 2 if with_ode else 1,
                           **{f"loss_{k}": float(v.detach()) for k, v in L.items()},
                           **{k: float(v.detach()) for k, v in self.theta().items()}, "seconds": time.time() - start}
                    history.append(rec)
                    if verbose:
                        print(rec, flush=True)
                if time.time() - start > cfg.time_budget:
                    stopped = True
                    break
            if stopped:
                break
        theta = {k: float(v.detach()) for k, v in self.theta().items()}
        return SBINNResult({**self.system.parameters, **theta}, history, epochs_run, stopped, time.time() - start,
                           cfg, self)

    def predict(self, t) -> dict[str, np.ndarray]:
        torch = self.torch
        with torch.no_grad():
            X = self.net(torch.tensor(np.asarray(t, float).reshape(-1, 1), dtype=self.dt)).numpy()
        return {s: X[:, i] for i, s in enumerate(self.system.states)}
