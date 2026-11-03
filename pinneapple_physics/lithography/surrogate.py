"""A learned imaging operator: a Fourier neural operator from the mask to its aerial image, trained on exact images.

Training masks are random Manhattan clips (rectangles of the layout's sizes); labels are exact images (SOCS kernels
of :mod:`.imaging`, which equal Abbe's). The operator is checked before it is used:

* **held-out error**: relative L2 of the image on clips never trained on;
* **resolution**: the same physical cell sampled twice as finely (the FNO keeps its Fourier modes), against exact
  images at that resolution;
* **printed pattern**: the EPE of the pattern printed from the predicted image against the one printed from the
  exact image (what matters for OPC).

A mask optimised through the operator is re-imaged by the exact model by the caller: the operator never grades
itself (Li et al., *ICLR* 2021, FNO).

Measured with the defaults (64 x 64 clips of 1 024 nm, annular 0.6-0.9, NA 1.35, 192 training clips, 60 epochs,
171 s on four CPU cores): held-out relative L2 0.89 % (worst 1.7 %), 1.03 % at twice the sampling, printed-pattern
EPE 1.52 nm.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .imaging import aerial_image_socs
from .metrics import epe, printed


def random_clips(n: int, N: int, L: float, rng: np.random.Generator, *, min_nm: float = 48.0, max_nm: float = 320.0,
                 shapes: tuple[int, int] = (2, 7)) -> np.ndarray:
    """``(n, N, N)`` binary masks: dark rectangles (0) on a clear field (1)."""
    px = L / N
    out = np.ones((n, N, N))
    for k in range(n):
        for _ in range(int(rng.integers(*shapes))):
            w, h = (int(max(1, round(rng.uniform(min_nm, max_nm) / px))) for _ in range(2))
            i, j = int(rng.integers(0, N)), int(rng.integers(0, N))
            rows, cols = np.arange(i, i + w) % N, np.arange(j, j + h) % N
            out[k][np.ix_(rows, cols)] = 0.0
    return out


def _fno(modes: int, width: int, layers: int):
    from pinneapple_neural.architectures.neural_operators.fno import FNO2d
    return FNO2d(1, 1, width=width, modes1=modes, modes2=modes, layers=layers, use_grid=False)


def _out(model, x):
    y = model(x)
    return y.y if hasattr(y, "y") else y


def train_operator(socs: dict, L: float, N: int, *, n_train: int = 192, n_test: int = 32, epochs: int = 60,
                   modes: int = 16, width: int = 20, layers: int = 4, lr: float = 3e-3, seed: int = 0,
                   threshold: float = 0.3, socs_fine: dict | None = None) -> dict:
    """Train and check the operator: ``{"model", "held_out_rel_l2", "held_out_epe_nm", "fine_rel_l2", ...}``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    X = random_clips(n_train + n_test, N, L, rng)
    Y = np.stack([aerial_image_socs(x, socs).numpy() for x in X])
    label_s = (time.perf_counter() - t0) / len(X)
    xt = torch.tensor(X[:, None], dtype=torch.float32)
    yt = torch.tensor(Y[:, None], dtype=torch.float32)
    model = _fno(modes, width, layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr, te = slice(0, n_train), slice(n_train, None)
    t1 = time.perf_counter()
    for _ in range(epochs):
        perm = torch.randperm(n_train)
        for b in range(0, n_train, 32):
            idx = perm[b:b + 32]
            opt.zero_grad()
            loss = ((_out(model, xt[tr][idx]) - yt[tr][idx]) ** 2).mean()
            loss.backward()
            opt.step()
        sched.step()
    train_s = time.perf_counter() - t1
    model.eval()
    with torch.no_grad():
        P = _out(model, xt[te]).numpy()[:, 0]
    rel = np.linalg.norm((P - Y[te]).reshape(n_test, -1), axis=1) / np.linalg.norm(Y[te].reshape(n_test, -1), axis=1)
    e = [epe(printed(torch.tensor(p, dtype=torch.float64), L, threshold=threshold),
             printed(torch.tensor(y, dtype=torch.float64), L, threshold=threshold), L)
         for p, y in zip(P, Y[te], strict=True)]
    out = {"model": model, "held_out_rel_l2": float(rel.mean()), "held_out_rel_l2_max": float(rel.max()),
           "held_out_epe_nm": float(np.mean(e)), "n_train": n_train, "n_test": n_test,
           "label_seconds_each": label_s, "train_seconds": train_s}
    if socs_fine is not None:                      # the same cells at twice the sampling
        Xf = np.repeat(np.repeat(X[te], 2, axis=1), 2, axis=2)
        Yf = np.stack([aerial_image_socs(x, socs_fine).numpy() for x in Xf])
        with torch.no_grad():
            Pf = _out(model, torch.tensor(Xf[:, None], dtype=torch.float32)).numpy()[:, 0]
        relf = np.linalg.norm((Pf - Yf).reshape(n_test, -1), axis=1) / np.linalg.norm(Yf.reshape(n_test, -1), axis=1)
        out["fine_rel_l2"] = float(relf.mean())
    return out


def operator_imager(model):
    """A differentiable ``(mask tensor, defocus) -> image`` for :func:`.ilt.optimise_mask` (nominal focus only)."""
    def imager(m: torch.Tensor, z: float) -> torch.Tensor:
        if z != 0:
            raise ValueError("this operator was trained at best focus")
        return _out(model, m.to(torch.float32)[None, None])[0, 0].to(torch.float64)
    return imager
