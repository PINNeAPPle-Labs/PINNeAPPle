"""Amortised inference: a mixture density network trained on simulated (parameters, observations) pairs that returns
the posterior of the parameters for any observation in one forward pass (Bishop 1994; the neural posterior
estimation of Papamakarios & Murray 2016).

    sim = lambda Z: surrogate(Z)                         # (n, d) unit-cube parameters -> (n, m) observations
    mdn = fit_mdn(sim, d=1, noise=lambda Y, rng: Y + 0.01 * rng.standard_normal(Y.shape), n_sims=20000)
    S = mdn.sample(y_obs, 4000)                          # posterior samples of the unit-cube parameters
    cal = posterior_coverage(mdn, Z_test, Y_test)        # are the credible intervals honest on known cases?

The prior is uniform on the unit cube (draws come from ``prior`` if given); ``noise(Y, rng)`` adds what separates
a measurement from the simulator (measurement noise, model discrepancy, surrogate uncertainty), so the posterior
is wide where the data cannot tell the parameters apart and multimodal where several designs give the same
measurement. Parameters are modelled in logit space (the posterior never leaves the cube). Training runs under a
:class:`~pinneapple_analysis.cost.equal_budget.ComputeBudget` with :func:`train_within_budget`.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["MDN", "fit_mdn", "posterior_coverage", "hdi"]

_EPS = 1e-4


def _logit(z):
    z = np.clip(z, _EPS, 1 - _EPS)
    return np.log(z) - np.log1p(-z)


def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def _net(d_in, d_out, hidden, depth):
    import torch.nn as nn
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


@dataclass
class MDN:
    """A trained mixture density network ``p(z | y)``: ``K`` diagonal Gaussians in logit space."""
    net: Any
    d: int
    K: int
    y_mean: np.ndarray
    y_std: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def _params(self, y):
        import torch
        Y = torch.as_tensor((np.atleast_2d(y) - self.y_mean) / self.y_std, dtype=torch.float32)
        with torch.no_grad():
            o = self.net(Y)
        return _split(o, self.d, self.K)

    def sample(self, y: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
        """``n`` posterior samples (n, d) in the unit cube for one observation ``y`` (m,)."""
        logw, mu, logs = (t.numpy()[0] for t in self._params(y))
        w = np.exp(logw - logw.max())
        w /= w.sum()
        rng = np.random.default_rng(seed)
        k = rng.choice(self.K, size=n, p=w)
        t = mu[k] + np.exp(logs[k]) * rng.standard_normal((n, self.d))
        return _sigmoid(t)

    def log_prob(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """log p(z | y) (unit-cube density) of each row of ``z`` (n, d) for one observation ``y``."""
        logw, mu, logs = (t.numpy()[0] for t in self._params(y))
        t = _logit(np.atleast_2d(z))
        comp = (-0.5 * ((t[:, None, :] - mu[None]) / np.exp(logs[None])) ** 2 - logs[None]
                - 0.5 * math.log(2 * math.pi)).sum(-1)
        lp = np.logaddexp.reduce(logw[None] + comp, axis=1)
        zc = np.clip(np.atleast_2d(z), _EPS, 1 - _EPS)
        return lp - np.log(zc * (1 - zc)).sum(1)                   # change of variables logit -> unit cube

    def summary(self, y: np.ndarray, level: float = 0.9, n: int = 4000, seed: int = 0) -> dict[str, Any]:
        S = self.sample(y, n, seed)
        a = (1 - level) / 2
        return {"mean": S.mean(0).tolist(), "median": np.median(S, 0).tolist(), "std": S.std(0).tolist(),
                "lo": np.quantile(S, a, 0).tolist(), "hi": np.quantile(S, 1 - a, 0).tolist(), "level": level,
                "hdi": [hdi(S[:, j], level) for j in range(self.d)]}


def _split(o, d, K):
    import torch
    logw = torch.log_softmax(o[:, :K], -1)
    mu = o[:, K:K + K * d].reshape(-1, K, d)
    logs = o[:, K + K * d:].reshape(-1, K, d).clamp(-7.0, 3.0)
    return logw, mu, logs


def fit_mdn(simulate: Callable[[np.ndarray], np.ndarray], d: int, *, noise: Callable | None = None,
            prior: Callable[[int, np.random.Generator], np.ndarray] | None = None, n_sims: int = 20000,
            K: int = 5, hidden: int = 64, depth: int = 3, budget_s: float = 60.0, lr: float = 2e-3,
            batch: int = 256, seed: int = 0, threads: int = 1) -> MDN:
    """Train on ``n_sims`` pairs ``z ~ prior`` (uniform on [0, 1]^d by default), ``y = noise(simulate(z))``;
    10 % of them are held out to keep the best checkpoint. ``threads``: torch threads while training (small nets
    train fastest on one)."""
    import torch
    old = torch.get_num_threads()
    torch.set_num_threads(max(1, int(threads)))
    try:
        return _fit_mdn(simulate, d, noise, prior, n_sims, K, hidden, depth, budget_s, lr, batch, seed)
    finally:
        torch.set_num_threads(old)


def _fit_mdn(simulate, d, noise, prior, n_sims, K, hidden, depth, budget_s, lr, batch, seed) -> MDN:
    import copy

    import torch

    from pinneapple_analysis.cost.equal_budget import (
        ComputeBudget,
        cosine_lr,
        set_lr,
        train_within_budget,
    )
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    Z = prior(n_sims, rng) if prior is not None else rng.random((n_sims, d))
    Y = np.asarray(simulate(Z), float).reshape(n_sims, -1)
    if noise is not None:
        Y = noise(Y, rng)
    ym, ys = Y.mean(0), Y.std(0) + 1e-12
    Yn = torch.as_tensor((Y - ym) / ys, dtype=torch.float32)
    T = torch.as_tensor(_logit(Z), dtype=torch.float32)
    n_val = max(1, n_sims // 10)
    perm = torch.as_tensor(rng.permutation(n_sims))
    iv, it = perm[:n_val], perm[n_val:]
    net = _net(Y.shape[1], K + 2 * K * d, hidden, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    def nll(idx):
        logw, mu, logs = _split(net(Yn[idx]), d, K)
        t = T[idx][:, None, :]
        comp = (-0.5 * ((t - mu) / logs.exp()) ** 2 - logs - 0.5 * math.log(2 * math.pi)).sum(-1)
        return -torch.logsumexp(logw + comp, -1).mean()

    best = {"loss": math.inf, "state": copy.deepcopy(net.state_dict())}

    def step(progress):
        set_lr(opt, cosine_lr(progress, lr))
        idx = it[torch.randint(0, len(it), (min(batch, len(it)),), generator=g)]
        loss = nll(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    def evaluate():
        net.eval()
        with torch.no_grad():
            v = float(nll(iv))
        net.train()
        if v < best["loss"]:
            best.update(loss=v, state=copy.deepcopy(net.state_dict()))
        return {"val_nll": v}

    run = train_within_budget(step, ComputeBudget(wall_s=budget_s), evaluate=evaluate,
                              eval_at=[i / 20 for i in range(1, 20)], count_first_step=False, log_every_s=1e9)
    net.load_state_dict(best["state"])
    net.eval()
    return MDN(net, d, K, ym, ys, {"n_sims": n_sims, "K": K, "hidden": hidden, "depth": depth,
                                   "steps": run.steps, "train_s": round(run.wall_s, 2), "best_val_nll": best["loss"]})


def hdi(samples: np.ndarray, level: float = 0.9, bins: int = 60) -> list[list[float]]:
    """Highest-density set of 1-D samples as a list of intervals (several when the posterior is multimodal), from a
    histogram: the smallest set of bins holding ``level`` of the mass. Bins are ranked by a [1, 2, 1]/4-smoothed
    count (so sampling noise does not punch one-bin holes into a mode) while the mass is counted from the raw
    histogram; intervals separated by a single bin are merged."""
    s = np.asarray(samples, float)
    h, e = np.histogram(s, bins=bins, range=(0.0, 1.0) if s.min() >= 0 and s.max() <= 1 else None)
    rank = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
    order = np.lexsort((-h, -rank))                              # by smoothed density, ties by raw count
    keep = np.zeros(len(h), bool)
    tot = 0
    for i in order:
        keep[i] = True
        tot += h[i]
        if tot >= level * h.sum():
            break
    for i in range(1, len(keep) - 1):                             # a one-bin hole between kept bins is noise
        if keep[i - 1] and keep[i + 1] and not keep[i]:
            keep[i] = True
    out, start = [], None
    for i, k in enumerate(np.r_[keep, False]):
        if k and start is None:
            start = i
        if not k and start is not None:
            out.append([float(e[start]), float(e[i])])
            start = None
    return out


def posterior_coverage(mdn: MDN, Z: np.ndarray, Y: np.ndarray, levels=(0.5, 0.8, 0.9, 0.95), n: int = 2000,
                       seed: int = 0) -> dict[str, Any]:
    """Calibration on cases with known parameters: for each nominal level, the fraction of true parameters inside the
    central credible interval (per parameter), and the rank of the truth among the posterior samples (uniform for a
    calibrated posterior; Talts et al. 2018). Also the posterior-mean error."""
    Z, Y = np.atleast_2d(Z), np.atleast_2d(Y)
    cover = {lv: np.zeros(Z.shape[1]) for lv in levels}
    ranks, err = [], []
    for i in range(len(Z)):
        S = mdn.sample(Y[i], n, seed + i)
        for lv in levels:
            a = (1 - lv) / 2
            lo, hi = np.quantile(S, a, 0), np.quantile(S, 1 - a, 0)
            cover[lv] += (Z[i] >= lo) & (Z[i] <= hi)
        ranks.append((S < Z[i]).mean(0))
        err.append(S.mean(0) - Z[i])
    R = np.array(ranks)
    return {"n": int(len(Z)), "coverage": {str(lv): (cover[lv] / len(Z)).tolist() for lv in levels},
            "rank_hist": [np.histogram(R[:, j], bins=10, range=(0, 1))[0].tolist() for j in range(Z.shape[1])],
            "mean_abs_error": np.abs(np.array(err)).mean(0).tolist()}
