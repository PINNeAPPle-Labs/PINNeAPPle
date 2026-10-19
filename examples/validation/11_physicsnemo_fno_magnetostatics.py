"""An external model: a PhysicsNeMo FNO for 2-D magnetostatics, validated with ``pp.validate``.

Magnetostatics in a cross-section (fields invariant along z): ``-div(nu grad A_z) = J_z``, ``A_z = 0`` on the box,
with ``nu = 1 / mu_r`` the reluctivity of a random iron / air layout (iron mu_r = 5 here) and ``J_z`` a coil. It is
the Darcy operator with ``a = nu`` and ``f = J_z``, so the reference solver is ``operator_benchmarks.darcy_2d`` and
``pp.validate`` runs the guaranteed residual bound, the flux balance (Ampere's law on the box) and the rest.

1. Train ``physicsnemo.models.fno.FNO`` (``nu -> A_z``).
2. ``pp.validate(model, {"pde": "darcy", "f": J}, data=...)``.
3. Trust card and ``pinneapple check`` outputs.
4. Failure caught: saturating-free iron with mu_r = 1000, far outside the training envelope
   (``applicability.envelope``); the residual bound grows with it.

Run: ``python examples/validation/11_physicsnemo_fno_magnetostatics.py [--full]`` (needs ``nvidia-physicsnemo``).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

import pinneapple as pp
from pinneapple_analysis.validation.gallery import conclude
from pinneapple_neural.utils import fit_supervised
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

FAMILY, PHYSICS, CATCHES = ("external", "operator"), "electromagnetics", "applicability.envelope"


def coil(n):
    """J_z = 1 in a square conductor left of centre, 0 elsewhere."""
    J = np.zeros((n, n))
    J[n // 4: n // 2, 3 * n // 8: 5 * n // 8] = 1.0
    return J


def layouts(m, n, seed, mu_iron):
    return ob.threshold_coefficient(ob.periodic_grf(m, (n, n), alpha=2.0, tau=3.0, seed=seed), 1.0, 1.0 / mu_iron)


def run(smoke: bool = True, out_dir=None):
    from physicsnemo.models.fno import FNO
    n, m, steps = (16, 80, 600) if smoke else (32, 400, 4000)
    J = coil(n)
    nu = layouts(m + 20, n, seed=2, mu_iron=5.0)
    A = np.stack([ob.darcy_2d(x, J) for x in nu])
    scale = float(np.abs(A).max())
    torch.manual_seed(0)
    net = FNO(1, 1, decoder_layer_size=32, dimension=2, latent_channels=16, num_fno_layers=3, num_fno_modes=6,
              padding=4)

    class Model(torch.nn.Module):                                      # reluctivity in, A_z (physical units) out
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x) * scale
    model = Model()
    fit_supervised(model, nu[:m, None], A[:m, None], steps=steps, lr=3e-3, batch=16, loss="rel_l2")
    problem = {"pde": "darcy", "f": J, "name": "2-D magnetostatics, -div(nu grad A_z) = J_z"}
    data = {"train": {"a": nu[:m], "u": A[:m]}, "test": {"a": nu[m:], "u": A[m:]}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = pp.validate(model, problem, data=data, level="standard")
        bad = pp.validate(model, problem, data={"train": data["train"]}, query=layouts(8, n, 9, mu_iron=1000.0),
                          level="standard")
    return conclude("11_physicsnemo_fno_magnetostatics", good, bad, expect=CATCHES, family=FAMILY, physics=PHYSICS,
                    out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full size instead of the CI smoke size")
    print(run(smoke=not ap.parse_args().full).good)
