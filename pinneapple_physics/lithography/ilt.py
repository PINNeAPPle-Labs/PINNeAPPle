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
                  binary_weight: float = 0.02, init: np.ndarray | None = None, filter_nm: float = 0.0,
                  min_feature_nm: float = 0.0) -> dict:
    """``{"mask" (bool), "mask_continuous", "loss_history", "mrc_changed_px"}``; ``corners`` are ``(defocus nm, dose)``.

    ``filter_nm``: the pixel parameters are Gaussian-filtered before the sigmoid, so the mask cannot hold features
    much smaller than the filter (a smooth, manufacturable mask). ``min_feature_nm``: the binary mask is then opened
    and closed with a square of that size, the mask rule check (MRC) every mask shop applies; how many pixels it
    changed is returned, and the caller verifies the mask after it."""
    t = torch.as_tensor(np.asarray(target, float))
    m0 = np.asarray(init if init is not None else (target if tone == "clear" else ~np.asarray(target, bool)), float)
    # dark-tone target (resist lines) -> the mask is dark (0) on the lines, clear (1) elsewhere
    theta = torch.tensor(4.0 * (m0 - 0.5), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    hist = []
    for _ in range(int(steps)):
        opt.zero_grad()
        m = torch.sigmoid(blur(theta, L, filter_nm))
        loss = 0.0
        for z, dose in corners:
            J = blur(imager(m, z), L, diffusion) * dose
            r = torch.sigmoid(steepness * ((threshold - J) if tone == "dark" else (J - threshold)))
            loss = loss + ((r - t) ** 2).mean()
        loss = loss / len(corners) + binary_weight * (m * (1 - m)).mean()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    mc = torch.sigmoid(blur(theta, L, filter_nm)).detach().numpy()
    mask = mc >= 0.5
    changed = 0
    if min_feature_nm > 0:
        from scipy import ndimage
        k = max(1, int(round(min_feature_nm / (L / mask.shape[0]))))
        st = np.ones((k, k), bool)
        pad = 2 * k                                   # the cell is periodic: wrap it before the morphology
        big = np.pad(mask, pad, mode="wrap")
        big = ndimage.binary_closing(ndimage.binary_opening(big, st), st)
        cleaned = big[pad:-pad, pad:-pad]
        changed = int(np.sum(cleaned != mask))
        mask = cleaned
    return {"mask": mask, "mask_continuous": mc, "loss_history": hist, "mrc_changed_px": changed}
