"""Self-scaled quasi-Newton optimizers: BFGS, SSBFGS and SSBroyden (dense inverse Hessian).

Implements the self-scaled Broyden family exactly as written in

    Urban, Stefanou & Pons, "Unveiling the optimization process of physics informed
    neural networks: How accurate and competitive can PINNs be?", J. Comput. Phys. 523
    (2025) 113656, arXiv:2405.04230 -- Section 3.2, eqs. (7)-(23), and Appendix B.

with the parameter choices the paper takes from Al-Baali & Khalfan (SSBroyden, its
ref. [68]) and Al-Baali (SSBFGS, its ref. [69]). The same optimizers are distributed for
JAX/Optimistix by CrunchOptimizer/PINNs (MIT), whose benchmarks (Euler/HLLC, Helmholtz,
inviscid Burgers, Stokes, stiff PK-PD) come from Jnini, Kiyani, Shukla et al.,
"Curvature-Aware Optimization for High-Accuracy PINNs", arXiv:2604.05230. This module is
an independent PyTorch implementation of the published formulas; no code was copied.

Update (eq. 10), with s = Theta_{k+1} - Theta_k and y = grad_{k+1} - grad_k::

    H_{k+1} = (1/tau) * (H - (H y)(H y)^T / (y.H y) + phi * v v^T) + s s^T / (y.s)
    v = sqrt(y.H y) * (s / (y.s) - H y / (y.H y))

- BFGS: tau = 1, phi = 1.
- SSBFGS (eqs. 11-12): tau = min(1, y.s / (s.H^{-1} s)), phi = 1.
- SSBroyden (eqs. 13-23): tau and phi from the auxiliary quantities b, h, a, c, rho, theta, sigma.

``s.H^{-1}s`` is never formed by inverting H: since s = -alpha H g, it equals
``-alpha s.g`` (Appendix B, eq. B.1). The step length comes from a strong-Wolfe line
search. H is dense (n x n), so this is meant for the small networks where the paper shows
the gains (a few thousand parameters); memory grows as n^2.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lbfgs import _strong_wolfe

VARIANTS = ("bfgs", "ssbfgs", "ssbroyden")


def self_scaled_parameters(s, y, g, alpha, Hy, n, variant):
    """Return ``(tau, phi)`` for one update (eqs. 11-23). Tensors are 1-D, ``g`` is the old gradient."""
    ys = torch.dot(y, s)
    yHy = torch.dot(y, Hy)
    if variant == "bfgs":
        return 1.0, 1.0
    # b = s.H^{-1}s / y.s = -alpha s.g / y.s  (eq. 15 with B.1)
    b = float(-alpha * torch.dot(s, g) / ys)
    tau1 = min(1.0, 1.0 / b) if b > 0 else 1.0  # eq. 11 / B.2
    if variant == "ssbfgs":
        return tau1, 1.0
    h = float(yHy / ys)  # eq. 16
    a = max(h * b - 1.0, 0.0)  # eq. 17; >= 0 by Cauchy-Schwarz, clamp round-off
    c = math.sqrt(a / (a + 1.0))  # eq. 18
    rho_minus = min(1.0, h * (1.0 - c))  # eq. 19
    theta_minus = (rho_minus - 1.0) / a if a > 1e-14 else -math.inf  # eq. 20
    theta_plus = 1.0 / rho_minus if rho_minus > 0 else math.inf  # eq. 21
    theta = max(theta_minus, min(theta_plus, (1.0 - b) / b))  # eq. 22
    sigma = 1.0 + a * theta  # eq. 23
    exponent = -1.0 / (n - 1) if n > 1 else 0.0
    sigma_pow = sigma ** exponent if sigma > 0 else 1.0
    if theta > 0:  # eq. 13
        tau = tau1 * min(sigma_pow, 1.0 / theta)
    else:
        tau = min(tau1 * sigma_pow, sigma)
    phi = (1.0 - theta) / sigma if sigma != 0 else 1.0  # eq. 14
    if not (tau > 0 and math.isfinite(tau) and math.isfinite(phi)):
        return 1.0, 1.0  # fall back to plain BFGS for this step
    return tau, phi


class SelfScaledQuasiNewton(Optimizer):
    """Dense BFGS / SSBFGS / SSBroyden with strong-Wolfe line search (Urban et al., JCP 2025).

    Usage mirrors ``torch.optim.LBFGS``: pass a closure that zeroes grads, computes the
    loss, calls ``backward()`` and returns the loss. One ``step`` runs up to ``max_iter``
    quasi-Newton iterations.

    Parameters
    ----------
    variant : ``"bfgs"``, ``"ssbfgs"`` or ``"ssbroyden"``.
    max_iter : iterations per ``step`` call.
    tolerance_grad, tolerance_change : stopping criteria (as in ``torch.optim.LBFGS``).
    """

    def __init__(self, params: Iterable, variant: str = "ssbroyden", max_iter: int = 20,
                 tolerance_grad: float = 1e-12, tolerance_change: float = 1e-14,
                 max_eval: Optional[int] = None):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        defaults = dict(variant=variant, max_iter=max_iter, tolerance_grad=tolerance_grad,
                        tolerance_change=tolerance_change, max_eval=max_eval or max_iter * 25)
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("SelfScaledQuasiNewton supports a single parameter group")
        self._params = self.param_groups[0]["params"]
        self._numel = sum(p.numel() for p in self._params)

    # -- flat helpers ---------------------------------------------------
    def _gather_flat_grad(self):
        return torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in self._params
        ])

    def _add_grad(self, step_size, direction):
        offset = 0
        for p in self._params:
            n = p.numel()
            p.add_(direction[offset:offset + n].view_as(p), alpha=step_size)
            offset += n

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @property
    def inverse_hessian(self) -> Optional[torch.Tensor]:
        return self.state.get("H")

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        variant, max_iter = group["variant"], group["max_iter"]
        state = self.state
        state.setdefault("n_iter", 0)
        state.setdefault("taus", [])

        orig_loss = closure()
        loss = float(orig_loss)
        g = self._gather_flat_grad()
        if g.abs().max() <= group["tolerance_grad"]:
            return orig_loss

        H = state.get("H")
        if H is None:
            H = torch.eye(self._numel, dtype=g.dtype, device=g.device)

        evals = 1
        for _ in range(max_iter):
            state["n_iter"] += 1
            d = -H.mv(g)
            gtd = torch.dot(g, d)
            if gtd > -group["tolerance_change"]:
                H = torch.eye(self._numel, dtype=g.dtype, device=g.device)  # lost descent: restart
                d, gtd = -g, -torch.dot(g, g)
            x_init = self._clone_param()

            def obj_func(x, t, dd):
                return self._directional_evaluate(closure, x, t, dd)

            new_loss, new_g, t, ls_evals = _strong_wolfe(obj_func, x_init, 1.0, d, loss, g, gtd)
            evals += ls_evals
            self._add_grad(t, d)
            s = t * d
            y = new_g - g
            ys = torch.dot(y, s)
            if ys > 1e-16:  # curvature condition; otherwise skip the update
                Hy = H.mv(y)
                yHy = torch.dot(y, Hy)
                tau, phi = self_scaled_parameters(s, y, g, t, Hy, self._numel, variant)
                v = torch.sqrt(yHy) * (s / ys - Hy / yHy)
                H = (H - torch.outer(Hy, Hy) / yHy + phi * torch.outer(v, v)) / tau + torch.outer(s, s) / ys
                H = 0.5 * (H + H.T)
                state["taus"].append(tau)
            loss_change = abs(new_loss - loss)
            loss, g = new_loss, new_g
            if g.abs().max() <= group["tolerance_grad"] or loss_change < group["tolerance_change"]:
                break
            if evals >= group["max_eval"]:
                break

        state["H"] = H
        return torch.as_tensor(loss)
