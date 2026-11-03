"""Inverse lithography (pixel OPC): the mask that prints a target, by gradient descent through the imaging model.

The mask is ``m = sigmoid(theta)`` per pixel; the printed resist is the smooth step ``sigmoid(a (T - dose I))``
for a dark-tone target (``sigmoid(a (dose I - T))`` for clear tone); the loss is the mean squared difference from
the target, summed over process corners (focus and dose), plus a small term pulling the mask to binary values
(Poonawala & Milanfar, *IEEE Trans. Image Process.* 16 (2007) 774; Granik, *J. Micro/Nanolith.* 5 (2006) 043002).
``imager`` maps a mask tensor to its aerial image at one focus: the exact model (SOCS kernels, differentiable) or a
learned operator. The final mask is binarized and **always** re-imaged by the exact model by the caller.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .metrics import blur


def optimise_mask(target: np.ndarray, L: float, imager: Callable[[torch.Tensor, float], torch.Tensor], *,
                  corners: list[tuple[float, float]] = ((0.0, 1.0),), threshold: float = 0.3, diffusion: float = 0.0,
                  tone: str = "dark", steps: int = 200, lr: float = 0.3, steepness: float = 50.0,
                  binary_weight: float = 0.02, init: np.ndarray | None = None) -> dict:
    """``{"mask" (bool), "mask_continuous", "loss_history"}``; ``corners`` are ``(defocus nm, dose)``."""
    t = torch.as_tensor(np.asarray(target, float))
    m0 = np.asarray(init if init is not None else (target if tone == "clear" else ~np.asarray(target, bool)), float)
    # dark-tone target (resist lines) -> the mask is dark (0) on the lines, clear (1) elsewhere
    theta = torch.tensor(4.0 * (m0 - 0.5), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    hist = []
    for _ in range(int(steps)):
        opt.zero_grad()
        m = torch.sigmoid(theta)
        loss = 0.0
        for z, dose in corners:
            J = blur(imager(m, z), L, diffusion) * dose
            r = torch.sigmoid(steepness * ((threshold - J) if tone == "dark" else (J - threshold)))
            loss = loss + ((r - t) ** 2).mean()
        loss = loss / len(corners) + binary_weight * (m * (1 - m)).mean()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    mc = torch.sigmoid(theta).detach().numpy()
    return {"mask": mc >= 0.5, "mask_continuous": mc, "loss_history": hist}
