"""A generative model (score-based diffusion) of porous-media fields, with a physics check of every sample.

1. Data: 400 permeability fields (thresholded Gaussian random fields, the FNO-paper Darcy set-up) and their
   pressure fields from ``operator_benchmarks.darcy_2d``. Two generators trained with denoising score matching
   (``pinns.physics_diffusion.PhysicsInformedDiffusion``):
   * physics in the loop: diffusion over the permeability field only; each generated field is solved by the
     reference solver for its pressure;
   * joint: diffusion over the stacked (permeability, pressure) fields, a common shortcut.
2. ``pp.validate(sampler, {"generative": {"physics": "darcy"}}, data={"train": ...})``: the samples are audited
   like a dataset: each one against the PDE in the solver's discretisation, diversity (mode collapse) and
   memorisation (generated samples that copy training samples).
3. Trust card and ``pinneapple check`` outputs (the same audit runs on a saved sample set with
   ``[[dataset]] physics = "darcy"``).
4. Failure caught: the joint generator's pressure fields do not solve the PDE for their own permeability
   (``data.physics``), although they look like pressure fields.

Run: ``python examples/validation/10_diffusion_darcy_generative.py [--full]``.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

import pinneapple as pp
from pinneapple_analysis.validation.gallery import conclude
from pinneapple_neural.architectures.pinns.physics_diffusion import PhysicsInformedDiffusion
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

FAMILY, PHYSICS, CATCHES = "generative", "porous-flow", "data.physics"
N = 8                                                                    # grid


def diffusion(x: np.ndarray, steps: int, seed: int = 0):
    """DSM training of a diffusion model on rows of ``x`` (standardised); returns ``sample(n) -> (n, d)``."""
    mu, sd = x.mean(0), x.std(0) + 1e-12
    torch.manual_seed(seed)
    model = PhysicsInformedDiffusion(x_dim=x.shape[1], hidden=(256, 256, 256))
    loss_fn = model.make_loss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.tensor((x - mu) / sd, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        batch = data[torch.randint(0, len(data), (128,), generator=g)]
        loss = loss_fn(batch)
        loss = loss["total"] if isinstance(loss, dict) else loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    def sample(n):
        torch.manual_seed(seed + 1)
        with torch.no_grad():
            return model.sample(n_samples=n, n_steps=50).numpy() * sd + mu
    return sample


def run(smoke: bool = True, out_dir=None):
    n_train, steps = (400, 1500) if smoke else (2000, 8000)
    field = ob.periodic_grf(n_train, (N, N), alpha=2.0, tau=3.0, seed=4)
    a = ob.threshold_coefficient(field)
    u = np.stack([ob.darcy_2d(x, 1.0) for x in a])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen_field = diffusion(field.reshape(n_train, -1), steps)
        gen_joint = diffusion(np.concatenate([a.reshape(n_train, -1), u.reshape(n_train, -1)], 1), steps)

    def physics_in_the_loop(n):
        aa = ob.threshold_coefficient(gen_field(n).reshape(n, N, N))
        return {"a": aa, "u": np.stack([ob.darcy_2d(x, 1.0) for x in aa])}

    def joint(n):
        z = gen_joint(n)
        return {"a": z[:, :N * N].reshape(n, N, N), "u": z[:, N * N:].reshape(n, N, N)}
    problem = {"generative": {"physics": "darcy", "n_samples": 32}, "name": "Darcy (a, u) sample generator"}
    data = {"train": {"a": a, "u": u}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = pp.validate(physics_in_the_loop, problem, data=data)
        bad = pp.validate(joint, problem, data=data)
    return conclude("10_diffusion_darcy_generative", good, bad, expect=CATCHES, family=FAMILY, physics=PHYSICS,
                    out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full size instead of the CI smoke size")
    print(run(smoke=not ap.parse_args().full).good)
