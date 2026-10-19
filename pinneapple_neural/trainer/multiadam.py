"""MultiAdam: Adam with separate moment statistics per loss term.

Implements Algorithm 1 of

    Yao, Su, Hao, Liu, Su & Zhu, "MultiAdam: Parameter-wise Scale-invariant Optimizer for
    Multiscale Training of Physics-informed Neural Networks", ICML 2023, arXiv:2306.02816.

Each loss term ``L_i`` (PDE residual, boundary terms, data, ...) gets its own first and
second moments, so a term that is orders of magnitude larger than the others cannot
dominate the step::

    g_i = grad L_i
    m_i = beta1 m_i + (1 - beta1) g_i,       v_i = beta2 v_i + (1 - beta2) g_i**2
    theta -= lr * mean_i( m_i_hat / (sqrt(v_i_hat) + eps) )

Rescaling one term leaves its update unchanged (up to ``eps``). PINNacle
(``src/optimizer/multiadam.py``, MIT) ships the same algorithm. This module is an
independent PyTorch implementation of the paper; no code was copied.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch.optim import Optimizer


class MultiAdam(Optimizer):
    """Adam with one ``(m, v)`` pair per loss term (Yao et al., ICML 2023).

    Unlike ``torch.optim.Adam``, ``step`` takes the loss terms themselves and computes
    each term's gradient; do not call ``backward()`` first::

        opt = MultiAdam(net.parameters(), lr=1e-3)
        opt.step([loss_pde, loss_bc])

    Pass the terms in the same order (and the same number) on every call.

    Parameters
    ----------
    lr, eps : as in ``torch.optim.Adam``.
    betas : moment decay rates; (0.99, 0.99) is the PINNacle default for PINNs.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.99, 0.99),
        eps: float = 1e-8,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas must be in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        self._n_losses = None

    def step(self, losses: Sequence[torch.Tensor]):  # type: ignore[override]
        """Take one step from the loss terms; returns their sum (detached)."""
        losses = list(losses)
        if not losses:
            raise ValueError("step needs at least one loss term")
        if self._n_losses is None:
            self._n_losses = len(losses)
        elif len(losses) != self._n_losses:
            raise ValueError(f"expected {self._n_losses} loss terms, got {len(losses)}")

        params = [p for group in self.param_groups for p in group["params"] if p.requires_grad]
        grads = [
            torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            for loss in losses
        ]

        with torch.no_grad():
            offset = 0
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                live = [p for p in group["params"] if p.requires_grad]
                for j, p in enumerate(live):
                    g = torch.stack(
                        [
                            gi[offset + j] if gi[offset + j] is not None else torch.zeros_like(p)
                            for gi in grads
                        ]
                    )
                    state = self.state[p]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                    state["step"] += 1
                    t = state["step"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
                    p.sub_(group["lr"] * (m_hat / (v_hat.sqrt() + group["eps"])).mean(dim=0))
                offset += len(live)
        return sum(loss.detach() for loss in losses)
