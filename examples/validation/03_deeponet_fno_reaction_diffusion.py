"""DeepONet and FNO steppers for reaction-diffusion: Allen-Cahn phase separation, ``u_t = D u_xx + u - u^3``
(domains u = +-1 joined by interfaces that move and merge), periodic, ``D = 0.01``.

1. Reference: ``numerical_solvers.reaction_diffusion.ReactionDiffusion1D`` (ETDRK4). Both operators learn the
   increment ``u(t + 0.2) - u(t)`` from 40 trajectories that start after the domains have formed.
2. ``pp.validate(step, {"rollout": ...})`` from a fresh state: pointwise error, interface (small-scale) energy and
   the invariants along 40 steps, divergence time.
3. Trust card and ``pinneapple check`` outputs.
4. Failure caught: the DeepONet stepper puts spurious energy in the highest resolved wavenumbers (its output is a
   combination of learned global basis functions); ``rollout.spectrum`` flags it after 2-4 time units, before its
   pointwise error reaches the 20 % tolerance. The FNO stepper, trained on the same data, passes.

Run: ``python examples/validation/03_deeponet_fno_reaction_diffusion.py [--full]``.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

import pinneapple as pp
from pinneapple_analysis.benchmark.builtin.fields import _operator_model
from pinneapple_analysis.validation.gallery import conclude
from pinneapple_neural.utils import fit_supervised
from pinneapple_simulation.numerical_solvers.reaction_diffusion import ReactionDiffusion1D

FAMILY, PHYSICS, CATCHES = "operator", "reaction-diffusion", "rollout.spectrum"
SOLVER = ReactionDiffusion1D("allen_cahn", diffusion=1e-2, n=64, length=2.0, dt=0.05)
EVERY = 4                                                               # frames 0.2 time units apart


def start(seed):                                                        # after the domains have formed (t = 3)
    return SOLVER.step(SOLVER.initial_state(seed, amplitude=0.3, modes=8), 60)


def stepper(method, cfg, X, dY, steps, lr):
    sc = float(dY.std())
    torch.manual_seed(0)
    model = _operator_model(method, cfg, (X.shape[1],))
    fit_supervised(model, X, dY / sc, steps=steps, lr=lr, batch=64)

    def step(u):
        with torch.no_grad():
            return u + sc * model(torch.tensor(u[None], dtype=torch.float32)).numpy()[0]
    return step


def run(smoke: bool = True, out_dir=None):
    n_traj, fno_steps, don_steps = (40, 1000, 3000) if smoke else (120, 4000, 10000)
    X, Y = [], []
    for s in range(n_traj):
        tr = SOLVER.trajectory(start(s), 40, EVERY)
        X.append(tr[:-1])
        Y.append(tr[1:])
    X, Y = np.concatenate(X), np.concatenate(Y)
    fno = stepper("fno", {"width": 32, "modes": 16}, X, Y - X, fno_steps, 2e-3)
    deeponet = stepper("deeponet", {"hidden": 128, "p": 64, "depth": 2}, X, Y - X, don_steps, 1e-3)
    ref = SOLVER.trajectory(start(123), 40, EVERY)
    problem = {"rollout": {"ndim": 1, "dt": 0.2, "steps": len(ref) - 1}, "name": "Allen-Cahn, D = 0.01 (periodic)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = pp.validate(fno, problem, data={"u0": ref[0], "reference": ref}, level="standard")
        bad = pp.validate(deeponet, problem, data={"u0": ref[0], "reference": ref}, level="standard")
    return conclude("03_deeponet_fno_reaction_diffusion", good, bad, expect=CATCHES, family=FAMILY,
                    physics=PHYSICS, out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full size instead of the CI smoke size")
    print(run(smoke=not ap.parse_args().full).good)
