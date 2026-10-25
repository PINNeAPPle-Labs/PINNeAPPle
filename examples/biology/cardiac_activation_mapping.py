"""Aliev-Panfilov excitation in 2-D tissue with a scar, its activation map, and the map recovered from sparse sites.

    python examples/biology/cardiac_activation_mapping.py [--sites 64] [--iters 20000]

1. Reference: Aliev-Panfilov on a 50 x 50 sheet (dimensionless units, D = 1), dx = 0.25, a non-conducting scar
   (D = 0) in the middle, stimulus at the bottom-left corner; the activation map is the first crossing of u = 0.5.
   The grid is checked by repeating the run at dx = 0.5 (activation-time difference reported).
2. ``--sites`` random measurement sites in conducting tissue, activation times read from the reference.
3. Map reconstruction: the eikonal PINN (T and V networks) against linear interpolation of the same sites; RMSE of the
   activation time on all conducting grid points and the error of the conduction velocity away from the scar.
Writes results/cardiac_activation_mapping.json and results/activation_maps.png.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pinneapple_analysis.inverse_problems.activation_mapping import (EikonalPINN, EikonalPINNConfig,
                                                                     interpolate_baseline)
from pinneapple_physics.biology.excitable import Stimulus, aliev_panfilov_2d, conduction_velocity

HERE = os.path.dirname(os.path.abspath(__file__))
L = 50.0


def scar(X, Y):
    return (np.abs(X - 25) < 6) & (np.abs(Y - 25) < 12)


def simulate(nx):
    D = lambda X, Y: np.where(scar(X, Y), 0.0, 1.0)            # noqa: E731
    stim = [Stimulus(0.0, 1.0, lambda X, Y: (X < 3) & (Y < 3), 1.0)]
    sol = aliev_panfilov_2d(Lx=L, Ly=L, nx=nx, ny=nx, t_end=90.0, D=D, stimuli=stim, save_every=0.1)
    return sol, sol.activation_times()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch

    torch.set_num_threads(1)
    fine, T = simulate(201)
    coarse, Tc = simulate(101)
    X, Y = np.meshgrid(fine.x, fine.y)
    tissue = ~scar(X, Y) & np.isfinite(T)
    # grid check: coarse map on the coarse nodes vs the fine map at the same nodes
    Tf_on_c = T[::2, ::2]
    both = np.isfinite(Tc) & np.isfinite(Tf_on_c) & ~scar(*np.meshgrid(coarse.x, coarse.y))
    grid_diff = float(np.sqrt(np.mean((Tc[both] - Tf_on_c[both]) ** 2)))
    cv_true = conduction_velocity(T, fine.x[1] - fine.x[0], fine.y[1] - fine.y[0])
    far = tissue & (np.hypot(np.maximum(np.abs(X - 25) - 6, 0), np.maximum(np.abs(Y - 25) - 12, 0)) > 4) \
        & (np.hypot(X, Y) > 8)
    cv_ref = float(np.nanmedian(cv_true[far]))

    rng = np.random.default_rng(a.seed)
    idx = rng.choice(np.flatnonzero(tissue.ravel()), a.sites, replace=False)
    pts = np.column_stack([X.ravel()[idx], Y.ravel()[idx]])
    tms = T.ravel()[idx]
    query = np.column_stack([X[tissue], Y[tissue]])

    base = interpolate_baseline(pts, tms, query)
    pinn = EikonalPINN(((0, L), (0, L)), EikonalPINNConfig(n_iters=a.iters, seed=a.seed)).fit(pts, tms)
    Tp, Vp = pinn.predict(query)

    def rmse(p):
        return float(np.sqrt(np.mean((p - T[tissue]) ** 2)))

    def cv_err(Tmap):
        M = np.full(T.shape, np.nan)
        M[tissue] = Tmap
        cv = conduction_velocity(M, fine.x[1] - fine.x[0], fine.y[1] - fine.y[0])
        return float(np.nanmedian(np.abs(cv[far] - cv_true[far]) / cv_true[far]))

    Vgrid = np.full(T.shape, np.nan)
    Vgrid[tissue] = Vp
    out = {"reference": {"nx": 201, "dx": float(fine.x[1] - fine.x[0]), "dt": fine.dt,
                         "activation_range": [float(np.nanmin(T)), float(np.nanmax(T[tissue]))],
                         "grid_check_rms_dT_dx0.5_vs_0.25": grid_diff, "cv_median_far_from_scar": cv_ref},
           "sites": a.sites, "seed": a.seed,
           "linear_interpolation": {"rmse_T": rmse(base), "cv_median_rel_error": cv_err(base)},
           "eikonal_pinn": {"rmse_T": rmse(Tp), "cv_median_rel_error_from_T": cv_err(Tp),
                            "V_network_median_rel_error": float(np.nanmedian(np.abs(Vgrid[far] - cv_true[far])
                                                                              / cv_true[far])),
                            "iters": pinn.iters_run, "final": pinn.history[-1]}}
    print(json.dumps({k: v for k, v in out.items() if k != "eikonal_pinn"}, indent=1))
    print("PINN:", {k: v for k, v in out["eikonal_pinn"].items() if k != "final"})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "cardiac_activation_mapping.json"), "w") as f:
        json.dump(out, f, indent=1)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        maps = [T, np.full(T.shape, np.nan), np.full(T.shape, np.nan)]
        maps[1][tissue], maps[2][tissue] = base, Tp
        for k, (m, title) in enumerate(zip(maps, ["Aliev-Panfilov reference", "linear interpolation",
                                                  "eikonal PINN"], strict=True)):
            im = ax[k].contourf(fine.x, fine.y, m, 30)
            ax[k].plot(pts[:, 0], pts[:, 1], "k.", ms=3)
            ax[k].set_title(title)
            ax[k].set_aspect("equal")
            fig.colorbar(im, ax=ax[k], label="activation time")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "results", "activation_maps.png"), dpi=110)
    except ImportError:
        pass
    return out


if __name__ == "__main__":
    main()
