"""Generate the accretion-flow simulations used to train and test the black-hole weather forecaster (issue #399).

Each run starts from an equilibrium torus around a Schwarzschild black hole (Paczynski-Wiita potential) and
evolves the viscous, radiatively inefficient flow that forms, saving the primitives every ``--every`` M.
Runs checkpoint their state, so an interrupted run resumes where it stopped:

    python examples/black_hole_weather/simulate.py --out data/bh --name PL0SS0.1 --alpha 0.1 --viscosity SS
    python examples/black_hole_weather/simulate.py --out data/bh --name PL0SS0.3 --alpha 0.3 --viscosity SS

Output per run (folder ``<out>/<name>``): ``frames_XXXX.npy`` chunks of (n, 5, nr, ntheta) float32 primitives
(rho, v_r, v_theta, v_phi, p), ``times.npy``, ``mdot.npy``, ``grid.npz``, ``config.json`` and ``state.pt``.
``load_run`` reads a run back as one array. The logic lives in ``pinneapple_physics.blackhole.runs``.
"""
from __future__ import annotations

import argparse
import os

import torch

from pinneapple_physics.blackhole import RIAFConfig
from pinneapple_physics.blackhole.runs import load_run, simulate_run

__all__ = ["load_run", "main"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--viscosity", default="SS")
    ap.add_argument("--torus-a", type=float, default=0.0)
    ap.add_argument("--nr", type=int, default=128)
    ap.add_argument("--ntheta", type=int, default=64)
    ap.add_argument("--t-end", type=float, default=10000.0)
    ap.add_argument("--every", type=float, default=10.0)
    ap.add_argument("--chunk", type=int, default=50, help="frames per saved chunk / checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturbation-kind", default="cell", choices=["cell", "modes"],
                    help="'modes' gives the same initial perturbation on every grid (resolution study)")
    ap.add_argument("--backend", default="numba", choices=["numba", "torch", "warp"])
    ap.add_argument("--cfl", type=float, default=0.3)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)

    cfg = RIAFConfig(nr=a.nr, ntheta=a.ntheta, alpha=a.alpha, viscosity=a.viscosity, torus_a=a.torus_a, seed=a.seed,
                     perturbation_kind=a.perturbation_kind, backend=a.backend, cfl=a.cfl)
    simulate_run(cfg, os.path.join(a.out, a.name), t_end=a.t_end, every=a.every, chunk=a.chunk, name=a.name,
                 log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
