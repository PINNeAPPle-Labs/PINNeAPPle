"""POD + DMD of incompressible flow past a cylinder: the von Karman wake at Re 60 (lattice Boltzmann, D2Q9).

1. Reference: ``pinneapple_simulation.numerical_solvers.lbm`` via the lab's wake set-up (``pinneapple_lab``
   ``rom_study``). POD of the cross-stream velocity (8 modes), DMD of the POD coefficients
   (``pinneapple_neural.architectures.rom``), forecast beyond the training window.
2. ``pp.validate(step, {"rollout": ...})`` against the solver: pointwise error, small-scale energy and the
   fluctuation energy along the forecast, divergence time.
3. Trust card and ``pinneapple check`` outputs.
4. Failure caught: the same ROM fitted while the shedding was still growing (a common mistake: the transient is in
   the training window) has eigenvalues outside the unit circle and its forecast keeps amplifying the wake: the
   fluctuation energy leaves the reference (``rollout.invariants``).

Run: ``python examples/validation/05_pod_dmd_cylinder_wake.py [--full]`` (smoke: about 40 s of LBM on a CPU).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

import pinneapple as pp
from pinneapple_analysis.validation.gallery import conclude
from pinneapple_lab.experiments.rom import _wake
from pinneapple_neural.architectures.rom import POD, DynamicModeDecomposition

FAMILY, PHYSICS, CATCHES = "rom", "incompressible-flow", "rollout.invariants"


def pod_dmd(snapshots: np.ndarray, r: int = 8):
    X = torch.tensor(snapshots.reshape(len(snapshots), -1), dtype=torch.float64)
    pod = POD(r=r, center=True).fit(X)
    dmd = DynamicModeDecomposition(r=r, center=False).fit(pod.encode(X)[None])
    shape = snapshots.shape[1:]

    def step(u):
        a = pod.encode(torch.tensor(u.reshape(1, -1), dtype=torch.float64))
        return pod.decode(dmd.rollout(a, 1)[:, -1]).numpy().reshape(shape)
    return step, float(np.abs(np.asarray(dmd.eig()[0])).max())


def run(smoke: bool = True, out_dir=None):
    D, steps = (12, 24000) if smoke else (20, 60000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, uy, _, meta = _wake(60.0, steps, 50, D=D)
    n = len(uy)                                                          # frames 50 LBM steps apart
    sat, grow, test = slice(int(0.58 * n), int(0.75 * n)), slice(int(0.31 * n), int(0.48 * n)), int(0.75 * n)
    good_step, lam_good = pod_dmd(uy[sat])
    bad_step, lam_bad = pod_dmd(uy[grow])
    ref = uy[test:]
    problem = {"rollout": {"ndim": 2, "dt": 1.0, "periodic": False, "steps": len(ref) - 1},
               "name": "cylinder wake, Re 60 (LBM), cross-stream velocity"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = pp.validate(good_step, problem, data={"u0": ref[0], "reference": ref}, level="standard")
        bad = pp.validate(bad_step, problem, data={"u0": ref[0], "reference": ref}, level="standard")
    print(f"largest |DMD eigenvalue|: {lam_good:.4f} (saturated window), {lam_bad:.4f} (growing transient)")
    return conclude("05_pod_dmd_cylinder_wake", good, bad, expect=CATCHES, family=FAMILY, physics=PHYSICS,
                    out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full size instead of the CI smoke size")
    print(run(smoke=not ap.parse_args().full).good)
