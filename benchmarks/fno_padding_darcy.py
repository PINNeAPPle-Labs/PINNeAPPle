"""FNO2d with and without domain padding on Darcy flow (#1162).

Problem (Li et al. 2021, "Fourier Neural Operator for Parametric PDEs", sec. 5.2, at a reduced size):
``-div(a grad u) = 1`` on the unit square, ``u = 0`` on the boundary, ``a`` piecewise constant (12 / 3, a thresholded
Gaussian random field), solved by the library's finite volumes (``operator_benchmarks.darcy_2d``) on a 32 x 32 grid.
The domain is not periodic, while the spectral convolution is: the padded FNO appends 8 cells of zeros after the
lifting and crops them before the projection.

Both models get the same data, seed, width (32), modes (12), layers (4), optimiser and 200 epochs; two seeds.
Reported: the mean relative L2 error on 100 held-out samples, and the mean absolute error in the two-cell boundary
band and in the interior, relative to the mean |u|. Results go to ``benchmarks/_out/fno_padding_darcy.json``.
This is a reduced CPU budget, not a reproduction of the paper's numbers.

    python -m benchmarks.fno_padding_darcy          (from the repo root; about 10 min on 4 CPU cores)
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from pinneapple_neural.architectures.neural_operators.fno import FNO2d
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

N, N_TRAIN, N_TEST, EPOCHS, PAD = 32, 400, 100, 200, 8


def data(seed: int = 7):
    a = ob.threshold_coefficient(ob.periodic_grf(N_TRAIN + N_TEST, (N, N), alpha=2.0, tau=3.0, seed=seed))
    u = np.stack([ob.darcy_2d(c) for c in a])
    return a, u


def train_and_score(a, u, padding: int, seed: int) -> dict:
    x = torch.tensor((a - a.mean()) / a.std(), dtype=torch.float32)[:, None]
    scale = u.std()
    y = torch.tensor(u / scale, dtype=torch.float32)[:, None]
    torch.manual_seed(seed)
    model = FNO2d(1, 1, width=32, modes1=12, modes2=12, layers=4, padding=padding)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, EPOCHS // 4, 0.5)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(EPOCHS):
        perm = torch.randperm(N_TRAIN, generator=gen)
        for i in range(0, N_TRAIN, 20):
            b = perm[i:i + 20]
            opt.zero_grad()
            p = model(x[b]).y
            loss = ((p - y[b]).flatten(1).norm(dim=1) / y[b].flatten(1).norm(dim=1)).mean()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(x[N_TRAIN:]).y[:, 0].numpy() * scale
    true = u[N_TRAIN:]
    band = np.zeros((N, N), bool)
    band[:2] = band[-2:] = True
    band[:, :2] = band[:, -2:] = True
    err = np.abs(pred - true)
    rel = np.linalg.norm((pred - true).reshape(N_TEST, -1), axis=1) / np.linalg.norm(true.reshape(N_TEST, -1), axis=1)
    mean_u = float(np.mean(np.abs(true)))
    return {"padding": padding, "seed": seed, "rel_l2_test": float(rel.mean()),
            "mean_abs_err_boundary_band_rel": float(err[:, band].mean() / mean_u),
            "mean_abs_err_interior_rel": float(err[:, ~band].mean() / mean_u), "train_s": round(time.time() - t0, 1)}


def main() -> dict:
    torch.set_num_threads(4)
    a, u = data()
    out = {"grid": N, "train": N_TRAIN, "test": N_TEST, "epochs": EPOCHS, "runs": []}
    for seed in (0, 1):
        for padding in (0, PAD):
            r = train_and_score(a, u, padding, seed)
            out["runs"].append(r)
            print(json.dumps(r), flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "fno_padding_darcy.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    main()
