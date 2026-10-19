"""FNO on porous flow (Darcy), validated as an exported ONNX file (an external model).

1. Train an FNO ``a -> u`` on ``operator_benchmarks.darcy_2d`` (permeability 12 / 3, the FNO-paper set-up) and export
   it to ONNX (``fno_export.export_onnx``: spectral convolutions in real arithmetic).
2. ``pp.validate("fno.onnx", "darcy_2d", data=..., query=...)``: training envelope, guaranteed residual bound, flux
   balance, reference error, spectrum, symmetries and the data audit.
3. Trust card and ``pinneapple check`` outputs.
4. Failure caught: queries with a permeability contrast of 100 / 1, outside the training envelope
   (``applicability.envelope``), where the residual bound also grows.

Run: ``python examples/validation/02_fno_darcy_onnx.py [--full]``.
"""
from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

import pinneapple as pp
from pinneapple_analysis.benchmark.builtin.fields import _operator_model
from pinneapple_analysis.validation.gallery import conclude
from pinneapple_neural.architectures.neural_operators.fno_export import export_onnx
from pinneapple_neural.utils import fit_supervised
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

FAMILY, PHYSICS, CATCHES = ("operator", "external"), "porous-flow", "applicability.envelope"


def darcy_set(n, n_grid, seed, high=12.0, low=3.0):
    a = ob.threshold_coefficient(ob.periodic_grf(n, (n_grid, n_grid), alpha=2.0, tau=3.0, seed=seed), high, low)
    return a, np.stack([ob.darcy_2d(x, 1.0) for x in a])


def run(smoke: bool = True, out_dir=None):
    n_grid, n_train, steps = (16, 80, 600) if smoke else (32, 400, 4000)
    a, u = darcy_set(n_train + 20, n_grid, seed=1)
    us = float(np.abs(u).max())
    torch.manual_seed(0)
    model = _operator_model("fno", {"width": 16, "modes": 6}, (n_grid, n_grid))
    fit_supervised(model, a[:n_train] / 12.0, u[:n_train] / us, steps=steps, lr=3e-3, batch=16, loss="rel_l2")

    class Physical(torch.nn.Module):                                   # physical units in and out
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x / 12.0) * us
    path = Path(tempfile.mkdtemp()) / "fno_darcy.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_onnx(Physical(model).eval(), torch.tensor(a[:2], dtype=torch.float32), path)
    data = {"train": {"a": a[:n_train], "u": u[:n_train]}, "test": {"a": a[n_train:], "u": u[n_train:]}}
    good = pp.validate(str(path), "darcy_2d", data=data, level="standard")
    shifted, _ = darcy_set(8, n_grid, seed=7, high=100.0, low=1.0)
    bad = pp.validate(str(path), "darcy_2d", data={"train": data["train"]}, query=shifted, level="standard")
    return conclude("02_fno_darcy_onnx", good, bad, expect=CATCHES, family=FAMILY, physics=PHYSICS, out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full size instead of the CI smoke size")
    print(run(smoke=not ap.parse_args().full).good)
