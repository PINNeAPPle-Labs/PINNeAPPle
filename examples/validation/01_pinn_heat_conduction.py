"""PINN, steady heat conduction: -lap T = q on the unit square, T = 0 on the walls (manufactured source, exact T).

1. Train a PINN (``validation.pinn.train_pinn``).
2. ``pp.validate(net, {"pinn": problem})``: residual at fresh points, near-trivial solution, boundary condition,
   spectrum and the exact solution.
3. Trust card and ``pinneapple check`` outputs (``validation.gallery.conclude``).
4. Failure caught: the same PINN trained without its boundary loss satisfies the PDE but not the walls
   (``pinn.boundary``: the condition was traded off against the residual).

Run: ``python examples/validation/01_pinn_heat_conduction.py [--full]``.
"""
from __future__ import annotations

import argparse
import warnings

import pinneapple as pp
from pinneapple_analysis.validation.gallery import conclude
from pinneapple_analysis.validation.pinn import poisson_2d, train_pinn

FAMILY, PHYSICS, CATCHES = "pinn", "heat", "pinn.boundary"


def run(smoke: bool = True, out_dir=None):
    problem = poisson_2d()
    size = dict(n_collocation=600, adam_steps=600, lbfgs_steps=300) if smoke else \
        dict(n_collocation=2000, adam_steps=2000, lbfgs_steps=1500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net, info = train_pinn(problem, **size)
        no_bc, _ = train_pinn(problem, **size, weights={"boundary": 0.0})
    good = pp.validate(net, {"pinn": problem}, data={"train_points": info["train_points"]}, level="standard")
    bad = pp.validate(no_bc, {"pinn": problem}, level="standard")
    return conclude("01_pinn_heat_conduction", good, bad, expect=CATCHES, family=FAMILY, physics=PHYSICS,
                    out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full size instead of the CI smoke size")
    print(run(smoke=not ap.parse_args().full).good)
