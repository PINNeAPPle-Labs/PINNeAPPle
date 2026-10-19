"""Classical ML for heat convection: a Gaussian-process regressor for the hot spot of a convectively cooled plate.

1. Reference: ``numerical_solvers.heat_fv_reference.plate_2d`` (finite volumes, thin plate with convection on both
   faces and a heat source under a device; its power balance is tested). 150 designs (power, conductivity,
   heat-transfer coefficient, thickness) -> temperature rise of the hot spot. Two Gaussian processes
   (``workflows.hybrid_surrogate_physics.GaussianProcessSurrogate``, NumPy backend): on log features with a log
   target, and on the raw inputs.
2. ``pp.validate(predict, {"valid_range", "outputs", "relations"}, data={"test"}, query=...)``: validity range,
   error on 40 new designs, and three relations of the physics: the rise is linear in the power (the equation is
   linear in T - T_air), and it falls with conductivity and with the heat-transfer coefficient.
3. Trust card and ``pinneapple check`` outputs.
4. Failure caught: the raw-feature GP predicts a hotter plate for a more conductive material and breaks the
   linearity in power by 36 % (``metamorphic``); the log-feature GP keeps all three relations (0.2 % error).

Run: ``python examples/validation/08_gaussian_process_fin_convection.py``.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

import pinneapple as pp
from pinneapple_analysis.validation.gallery import conclude
from pinneapple_neural.workflows.hybrid_surrogate_physics import GaussianProcessSurrogate
from pinneapple_simulation.numerical_solvers.heat_fv_reference import plate_2d

FAMILY, PHYSICS, CATCHES = "classical-ml", "heat", "metamorphic"
BOX = {"power_w": (2.0, 20.0), "k_w_mk": (50.0, 400.0), "h_w_m2k": (10.0, 100.0), "t_mm": (1.0, 5.0)}
KEYS = list(BOX)
RELATIONS = [{"kind": "symmetric", "name": "linear-in-power", "inputs": {"power_w": "1.25 * power_w"},
              "outputs": {"rise_k": "1.25 * rise_k"}, "tol": 0.02},
             {"kind": "monotonic", "name": "falls-with-conductivity", "input": "k_w_mk", "output": "rise_k",
              "direction": "decreasing"},
             {"kind": "monotonic", "name": "falls-with-convection", "input": "h_w_m2k", "output": "rise_k",
              "direction": "decreasing"}]


def hot_spot_rise(power_w, k_w_mk, h_w_m2k, t_mm, grid=(40, 24)):
    """Temperature rise (K) of the hottest cell of a 100 x 60 mm plate, 10 mm device at its centre."""
    return np.array([plate_2d(h, lx=0.1, ly=0.06, t=t * 1e-3, k=k, power=p, t_air=25.0,
                              device=(0.05, 0.03, 0.01, 0.004), nx=grid[0], ny=grid[1])[2].max() - 25.0
                     for p, k, h, t in zip(*map(np.atleast_1d, (power_w, k_w_mk, h_w_m2k, t_mm)), strict=True)])


def designs(n, seed):
    rng = np.random.default_rng(seed)
    X = {k: rng.uniform(lo, hi, n) for k, (lo, hi) in BOX.items()}
    return X, hot_spot_rise(**X)


def gaussian_process(X, y, log: bool):
    def feats(q):
        A = np.stack([np.asarray(q[k], float) for k in KEYS], 1)
        return np.log(A) if log else A
    A = feats(X)
    mu, sd = A.mean(0), A.std(0)
    gp = GaussianProcessSurrogate(use_sklearn=False).fit((A - mu) / sd, np.log(y) if log else y)

    def predict(q):
        m = gp.predict((feats(q) - mu) / sd).mean[:, 0]
        return {"rise_k": np.exp(m) if log else m}
    return predict


def run(smoke: bool = True, out_dir=None):
    X, y = designs(150 if smoke else 600, 0)
    Xt, yt = designs(40, 1)
    spec = {"valid_range": BOX, "outputs": ["rise_k"], "relations": RELATIONS, "name": "plate hot spot (convection)"}
    query = {k: v[:5] for k, v in Xt.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = pp.validate(gaussian_process(X, y, log=True), spec, data={"test": {**Xt, "rise_k": yt}}, query=query)
        bad = pp.validate(gaussian_process(X, y, log=False), spec, data={"test": {**Xt, "rise_k": yt}}, query=query)
    return conclude("08_gaussian_process_fin_convection", good, bad, expect=CATCHES, family=FAMILY,
                    physics=PHYSICS, out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full size instead of the CI smoke size")
    print(run(smoke=not ap.parse_args().full).good)
