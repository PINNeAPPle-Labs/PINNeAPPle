"""A reduced-order model (POD + Operator Inference) of weather-like advection: a periodic tracer field carried by a
uniform wind with weak diffusion, ``u_t + c u_x = nu u_xx`` (``pinneapple_physics.advection_diffusion_1d``, exact
solution by Fourier modes).

1. Snapshots of 12 random tracer fields; POD basis; Operator Inference fits the linear one-step map of the POD
   coefficients (``pinneapple_neural.architectures.rom``).
2. ``pp.validate(step, {"rollout": ...}, data={"u0", "reference"})`` from a fresh initial field: mass (the tracer
   integral, conserved), small-scale energy along the rollout, divergence time.
3. Trust card and ``pinneapple check`` outputs.
4. Failure caught: the same ROM with the basis truncated by the usual "95 % of the snapshot energy" rule smears the
   small scales (``rollout.spectrum``) although it conserves mass exactly and its training error looks small.

Run: ``python examples/validation/06_opinf_weather_advection.py``.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

import pinneapple as pp
from pinneapple_analysis.validation.gallery import conclude
from pinneapple_neural.architectures.rom import POD, OperatorInference
from pinneapple_physics.advection_diffusion_1d import AdvectionDiffusion1D

FAMILY, PHYSICS, CATCHES = "rom", "advection", "rollout.spectrum"


def rom(train: np.ndarray, energy: float):
    X = torch.tensor(train, dtype=torch.float64)
    pod = POD(r=train.shape[-1], center=True, energy=energy).fit(X)
    a = pod.encode(X).reshape(train.shape[0], train.shape[1], -1)
    op = OperatorInference(a.shape[-1], use_quadratic=False, use_bias=False, center=False).fit(a)

    def step(u):
        a1 = op.rollout(pod.encode(torch.tensor(u[None], dtype=torch.float64)), 1)[:, -1]
        return pod.decode(a1)[0].numpy()
    return step, a.shape[-1]


def run(smoke: bool = True, out_dir=None):
    P = AdvectionDiffusion1D(nx=64, nt=201, t_end=4 * np.pi)            # two passes of the wind over the domain
    rng = np.random.default_rng(0)
    train = np.stack([P.exact(P.random_case(rng, nu=0.005, c=1.0, n_modes=6)) for _ in range(12)])
    ref = P.exact(P.random_case(np.random.default_rng(99), nu=0.005, c=1.0, n_modes=6))
    problem = {"rollout": {"ndim": 1, "dt": float(P.t[1]), "conserved": ["mass"], "steps": len(ref) - 1},
               "name": "periodic advection-diffusion (tracer in a uniform wind)"}
    good_step, r_good = rom(train, energy=1.0 - 1e-10)
    bad_step, r_bad = rom(train, energy=0.95)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = pp.validate(good_step, problem, data={"u0": ref[0], "reference": ref}, level="standard")
        bad = pp.validate(bad_step, problem, data={"u0": ref[0], "reference": ref}, level="standard")
    print(f"POD rank {r_good} (all the energy) vs {r_bad} (95 % of the energy)")
    return conclude("06_opinf_weather_advection", good, bad, expect=CATCHES, family=FAMILY, physics=PHYSICS,
                    out_dir=out_dir)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    print(run().good)
