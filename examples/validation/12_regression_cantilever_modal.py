"""Classical ML for structural dynamics: the first natural frequency of a cantilever from a table of designs.

1. Data: the Euler-Bernoulli cantilever, ``f1 = (beta1 L)^2 / (2 pi L^2) sqrt(E I / (rho A))`` with ``beta1 L`` the
   first root of ``cos x cosh x = -1`` (``nonlinear_beam_fem.cantilever_frequency_roots``), rectangular section
   (``I / A = h^2 / 12``), 400 random designs. Two regressors: least squares on log features (a power law, the form
   dimensional analysis predicts) and an MLP on the raw standardised inputs (a common default).
2. ``pp.validate(predict, {"valid_range", "outputs", "relations"}, data={"test"}, query=...)``: validity range,
   reference error and two metamorphic relations from the physics: scaling E and rho together leaves the
   frequency unchanged (it depends on the wave speed sqrt(E / rho)), and E x 1.21 raises it by exactly 10 %.
3. Trust card and ``pinneapple check`` outputs; ``predict`` is the ``[[model]]`` target of
   ``examples/validation/pinneapple-check.toml`` (with the same relations).
4. Failure caught: the MLP breaks both scaling laws by about 6 % (``metamorphic``; its error on the test designs
   is also too large, 26 % at smoke size); the log-linear model recovers the exponents (-2, 1, 1/2, -1/2) to
   round-off, so both laws hold exactly.

Run: ``python examples/validation/12_regression_cantilever_modal.py``.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

import pinneapple as pp
from pinneapple_analysis.validation.gallery import conclude
from pinneapple_analysis.validation.pinn import mlp
from pinneapple_neural.utils import fit_supervised
from pinneapple_simulation.numerical_solvers.nonlinear_beam_fem import cantilever_frequency_roots

FAMILY, PHYSICS, CATCHES = "classical-ml", "structural-dynamics", "metamorphic"
BOX = {"L_m": (0.2, 1.0), "h_mm": (2.0, 20.0), "E_gpa": (50.0, 250.0), "rho": (2000.0, 9000.0)}
KEYS = list(BOX)
RELATIONS = [{"kind": "invariant", "name": "wave-speed", "inputs": {"E_gpa": "1.2 * E_gpa", "rho": "1.2 * rho"},
              "output": "f1_hz", "tol": 0.02},
             {"kind": "symmetric", "name": "stiffness-scaling", "inputs": {"E_gpa": "1.21 * E_gpa"},
              "outputs": {"f1_hz": "1.1 * f1_hz"}, "tol": 0.02}]


def first_frequency(L_m, h_mm, E_gpa, rho):
    beta = float(cantilever_frequency_roots(1)[0])
    return beta ** 2 / (2 * np.pi * np.asarray(L_m) ** 2) * np.sqrt(
        np.asarray(E_gpa) * 1e9 * (np.asarray(h_mm) * 1e-3) ** 2 / 12 / np.asarray(rho))


def designs(n, seed):
    rng = np.random.default_rng(seed)
    X = {k: rng.uniform(lo, hi, n) for k, (lo, hi) in BOX.items()}
    return X, first_frequency(**X)


def _matrix(q):
    return np.stack([np.asarray(q[k], float) for k in KEYS], 1)


def power_law(X, y):
    """Least squares on log features: ``log f1 = c0 + sum_i c_i log x_i``."""
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(y)), np.log(_matrix(X))]), np.log(y), rcond=None)[0]
    return lambda q: {"f1_hz": np.exp(np.column_stack([np.ones(len(_matrix(q))), np.log(_matrix(q))]) @ coef)}


def raw_mlp(X, y, steps=3000):
    A = _matrix(X)
    mu, sd, ym, ys = A.mean(0), A.std(0), y.mean(), y.std()
    torch.manual_seed(0)
    net = mlp(4, 1, 64, 3)
    fit_supervised(net, (A - mu) / sd, ((y - ym) / ys)[:, None], steps=steps, lr=2e-3, batch=64)

    def f(q):
        with torch.no_grad():
            return {"f1_hz": net(torch.tensor((_matrix(q) - mu) / sd, dtype=torch.float32)).numpy()[:, 0] * ys + ym}
    return f


_MODEL = None


def predict(X):
    """``pinneapple check`` entry point: ``X`` ``(n, 4)`` columns L_m, h_mm, E_gpa, rho -> ``(n, 1)`` f1_hz."""
    global _MODEL
    if _MODEL is None:
        _MODEL = power_law(*designs(400, 0))
    X = np.atleast_2d(np.asarray(X, float))
    return _MODEL({k: X[:, i] for i, k in enumerate(KEYS)})["f1_hz"][:, None]


def run(smoke: bool = True, out_dir=None):
    X, y = designs(400, 0)
    Xt, yt = designs(100, 1)
    spec = {"valid_range": BOX, "outputs": ["f1_hz"], "relations": RELATIONS,
            "name": "cantilever first natural frequency"}
    query = {k: v[:5] for k, v in Xt.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = pp.validate(power_law(X, y), spec, data={"test": {**Xt, "f1_hz": yt}}, query=query)
        bad = pp.validate(raw_mlp(X, y, 3000 if smoke else 10000), spec, data={"test": {**Xt, "f1_hz": yt}},
                          query=query)
    return conclude("12_regression_cantilever_modal", good, bad, expect=CATCHES, family=FAMILY, physics=PHYSICS,
                    out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full size instead of the CI smoke size")
    print(run(smoke=not ap.parse_args().full).good)
