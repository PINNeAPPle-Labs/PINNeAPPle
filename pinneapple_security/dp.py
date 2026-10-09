"""Differential privacy: noise mechanisms, private statistics, a privacy accountant and DP-SGD for PyTorch models.

Use case in Physics AI: a surrogate or a digital twin trained on a client's proprietary operating data (or on several
clients' data) must not let anyone recover individual records from the published model. Differential privacy bounds
how much one record can change any output: (ε, δ)-DP.

* :func:`laplace_mechanism` (pure ε-DP) and :func:`gaussian_mechanism` ((ε, δ)-DP, classic calibration valid for ε ≤ 1).
* :func:`dp_mean` and :func:`dp_histogram` with clipping to known bounds (the bounds must not be computed from the data).
* :class:`RDPAccountant`: Rényi-DP accounting of the sampled Gaussian mechanism at integer orders (Mironov, Talwar &
  Zhang 2019, "Rényi Differential Privacy of the Sampled Gaussian Mechanism", arXiv:1908.10530), converted to (ε, δ).
* :class:`DPSGD`: per-sample gradient clipping and Gaussian noise (Abadi et al. 2016, "Deep Learning with Differential
  Privacy", arXiv:1607.00133) for any ``torch.nn.Module`` and per-sample loss, using ``torch.func``.

References for the Laplace and Gaussian mechanisms: Dwork & Roth, "The Algorithmic Foundations of Differential
Privacy", 2014, §3.3 and Appendix A.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = ["laplace_mechanism", "gaussian_mechanism", "gaussian_sigma", "dp_mean", "dp_histogram", "RDPAccountant", "DPSGD"]


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def laplace_mechanism(value: Any, sensitivity: float, epsilon: float, rng: Any = None) -> Any:
    if epsilon <= 0 or sensitivity < 0:
        raise ValueError("epsilon must be > 0 and sensitivity >= 0")
    v = np.asarray(value, dtype=float)
    out = v + _rng(rng).laplace(0.0, sensitivity / epsilon, size=v.shape)
    return float(out) if out.ndim == 0 else out


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """σ = Δ₂ √(2 ln(1.25/δ)) / ε (Dwork & Roth Thm A.1; stated for ε ≤ 1)."""
    if not 0 < epsilon <= 1:
        raise ValueError("the classic Gaussian calibration holds for 0 < epsilon <= 1; use RDPAccountant beyond")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def gaussian_mechanism(value: Any, sensitivity: float, epsilon: float, delta: float, rng: Any = None) -> Any:
    v = np.asarray(value, dtype=float)
    out = v + _rng(rng).normal(0.0, gaussian_sigma(sensitivity, epsilon, delta), size=v.shape)
    return float(out) if out.ndim == 0 else out


def dp_mean(x: Sequence[float], lower: float, upper: float, epsilon: float, rng: Any = None) -> float:
    """ε-DP mean with values clipped to [lower, upper] and the record count treated as public."""
    x = np.clip(np.asarray(x, dtype=float), lower, upper)
    if len(x) == 0:
        raise ValueError("empty input")
    return float(np.clip(laplace_mechanism(x.mean(), (upper - lower) / len(x), epsilon, rng), lower, upper))


def dp_histogram(x: Sequence[float], bins: Sequence[float], epsilon: float, rng: Any = None) -> np.ndarray:
    """ε-DP counts per bin (one record changes one count by 1); negative noisy counts are set to 0."""
    counts, _ = np.histogram(np.asarray(x, dtype=float), bins=np.asarray(bins, dtype=float))
    return np.maximum(laplace_mechanism(counts.astype(float), 1.0, epsilon, rng), 0.0)


class RDPAccountant:
    """Privacy spent by ``steps`` applications of the sampled Gaussian mechanism (sampling rate ``q``, noise
    multiplier ``sigma`` = noise std / clipping norm)."""

    ORDERS = tuple(range(2, 129))

    def __init__(self):
        self.rdp = np.zeros(len(self.ORDERS))
        self.history = []

    @staticmethod
    def _rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
        if q == 0:
            return 0.0
        if q == 1.0:
            return alpha / (2 * sigma ** 2)
        # log A_alpha = log sum_k C(alpha,k) (1-q)^(alpha-k) q^k exp((k^2-k)/(2 sigma^2))  (integer alpha)
        terms = [math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
                 + (alpha - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2 * sigma ** 2)
                 for k in range(alpha + 1)]
        m = max(terms)
        return (m + math.log(sum(math.exp(t - m) for t in terms))) / (alpha - 1)

    def step(self, q: float, sigma: float, steps: int = 1) -> RDPAccountant:
        if not 0 <= q <= 1 or sigma <= 0 or steps < 0:
            raise ValueError("need 0 <= q <= 1, sigma > 0, steps >= 0")
        self.rdp += steps * np.array([self._rdp_sampled_gaussian(q, sigma, a) for a in self.ORDERS])
        self.history.append({"q": q, "sigma": sigma, "steps": steps})
        return self

    def epsilon(self, delta: float) -> dict[str, float]:
        """(ε, δ) from RDP: ε = min_α rdp(α) + log(1/δ)/(α-1) (Mironov 2017, Prop. 3)."""
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        eps = self.rdp + math.log(1 / delta) / (np.array(self.ORDERS) - 1)
        i = int(np.argmin(eps))
        return {"epsilon": float(eps[i]), "delta": delta, "order": float(self.ORDERS[i])}


class DPSGD:
    """DP-SGD wrapper: ``step(xb, yb)`` computes per-sample gradients of ``loss_fn(model_output, y)`` (one scalar per
    sample), clips each to ``max_grad_norm``, adds N(0, (σ·C)²) noise to the sum and lets ``optimizer`` apply the
    noisy mean. ``sample_rate`` = batch size / dataset size (Poisson sampling is assumed by the accountant; uniform
    shuffled batches are the common approximation)."""

    def __init__(self, model, optimizer, loss_fn: Callable, noise_multiplier: float, max_grad_norm: float,
                 sample_rate: float, seed: int | None = None):
        import torch

        self.model, self.opt, self.loss_fn = model, optimizer, loss_fn
        self.sigma, self.C, self.q = float(noise_multiplier), float(max_grad_norm), float(sample_rate)
        self.accountant = RDPAccountant()
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def _per_sample_grads(self, xb, yb):
        from torch.func import functional_call, grad, vmap

        params = {k: v.detach() for k, v in self.model.named_parameters() if v.requires_grad}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}

        def loss_one(p, x, y):
            out = functional_call(self.model, (p, buffers), (x.unsqueeze(0),))
            return self.loss_fn(out, y.unsqueeze(0)).sum()

        return vmap(grad(loss_one), in_dims=(None, 0, 0))(params, xb, yb)

    def step(self, xb, yb) -> dict[str, float]:
        import torch

        g = self._per_sample_grads(xb, yb)
        names = list(g)
        b = xb.shape[0]
        norms = torch.sqrt(sum(g[n].reshape(b, -1).pow(2).sum(1) for n in names))
        scale = (self.C / (norms + 1e-12)).clamp(max=1.0)
        params = dict(self.model.named_parameters())
        self.opt.zero_grad(set_to_none=True)
        for n in names:
            s = (g[n] * scale.view(-1, *[1] * (g[n].dim() - 1))).sum(0)
            noise = torch.normal(0.0, self.sigma * self.C, size=s.shape, generator=self.gen).to(s)
            params[n].grad = (s + noise) / b
        self.opt.step()
        self.accountant.step(self.q, self.sigma, 1)
        return {"clipped_fraction": float((norms > self.C).float().mean()), "mean_grad_norm": float(norms.mean())}

    def epsilon(self, delta: float) -> dict[str, float]:
        return self.accountant.epsilon(delta)
