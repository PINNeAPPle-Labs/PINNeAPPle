"""Cold plate with the library's own FNO: PINNeAPPle generates, trains and checks (#1163).

The problem of ``examples/nvidia/07_omniverse_physicsnemo_coldplate``, without PhysicsNeMo:
- the plate is 100 mm x 100 mm, with copper inserts (k = 400 W/m/K) in an aluminium base (k = 100 W/m/K);
- a uniform volumetric heat load, and the edges held at the coolant temperature;
- steady conduction ``-div(k grad T) = q``, ``T = T_c`` on the edges.

Steps:
- the library's finite volumes (``operator_benchmarks.darcy_2d``) make the data;
- ``FNO2d``, with 8 cells of domain padding because the edges are not periodic (#1162), learns layout ->
  temperature, under the same ``ComputeBudget`` as example 07 (2700 steps);
- ``pinneapple_physics.darcy_checks`` decides whether its fields can be used: the certified residual bound, the
  boundary condition, the energy balance, the maximum principle, the reference and the training envelope.

    python examples/fno_coldplate/run.py               # about 3 min on 4 CPU cores
    python examples/fno_coldplate/run.py --steps 8000  # a longer budget
    python examples/fno_coldplate/run.py --smoke       # seconds (CI)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import hashlib
import platform
import sys

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_neural.physicsnemo_bridge.field_operators import pinneapple_fno, train_grid_operator
from pinneapple_physics import darcy_checks as dc
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

HERE = os.path.dirname(os.path.abspath(__file__))
L = 0.10                      # plate side [m]
K0 = 100.0 / 3.0              # W/m/K per unit of the solver's coefficient: 3 -> aluminium (100), 12 -> copper (400)
Q = 2.0e6                     # volumetric heat load [W/m^3] (100 W in a 5 mm plate)
T_COOL = 300.0                # coolant / edge temperature [K]
DT = Q * L ** 2 / K0          # kelvin per unit of the dimensionless solution


def layouts(n_samples, n, seed):
    return ob.threshold_coefficient(ob.periodic_grf(n_samples, (n, n), alpha=2.0, tau=3.0, seed=seed))


def main(smoke: bool = False, base: str | None = None, steps: int | None = None, padding: int = 8,
         train_seed: int = 0, test_seed: int = 1, model_seed: int | None = None) -> dict:
    """``base``: folder for results/ (default: this folder; tests pass a temporary one)."""
    out = os.path.join(base or HERE, "results")
    os.makedirs(out, exist_ok=True)
    n, n_train, n_test = (16, 24, 8) if smoke else (32, 400, 50)
    steps = steps or (6 if smoke else 2700)
    if model_seed is not None:
        import torch
        torch.manual_seed(int(model_seed))
        np.random.seed(int(model_seed))
    if train_seed == test_seed:
        raise ValueError("train_seed and test_seed must differ for held-out evaluation")
    a_tr, a_te = layouts(n_train, n, train_seed), layouts(n_test, n, test_seed)                   # the layouts of example 07
    data_t0 = time.perf_counter()
    u_tr = np.stack([ob.darcy_2d(a) for a in a_tr])
    u_te = np.stack([ob.darcy_2d(a) for a in a_te])
    data_generation_s = time.perf_counter() - data_t0
    t0 = time.time()
    for a in a_te:                                                               # the solver's own cost per layout
        ob.darcy_2d(a)
    solver_s = (time.time() - t0) / n_test
    pred = pinneapple_fno(n, width=8 if smoke else 32, modes=4 if smoke else 12, layers=2 if smoke else 4,
                          padding=padding)
    train_t0 = time.perf_counter()
    res = train_grid_operator(pred, a_tr, u_tr, ComputeBudget(steps=steps), batch_size=4 if smoke else 8, lr=2e-3)
    training_s = time.perf_counter() - train_t0
    t0 = time.time()
    u_hat = pred.predict(a_te)
    infer_s = (time.time() - t0) / n_test
    env = dc.CoefficientEnvelope.fit(a_tr)
    report, checks = dc.darcy_trust_report(a_te, u_hat, 1.0, model="PINNeAPPle FNO2d", reference=u_te,
                                           envelope=env)
    # Transparent non-ML baselines: training-mean temperature field and
    # nearest-neighbour conductivity layout (both exclude held-out labels).
    baseline_t0 = time.perf_counter()
    flat_train = a_tr.reshape(n_train, -1).astype(np.float64)
    flat_test = a_te.reshape(n_test, -1).astype(np.float64)
    train_norm = np.sum(flat_train**2, axis=1)
    nearest = np.argmin(np.maximum(0.0, np.sum(flat_test**2, axis=1)[:, None]
                     + train_norm[None, :] - 2 * flat_test @ flat_train.T), axis=1)
    nn_pred = u_tr[nearest]
    mean_pred = np.broadcast_to(u_tr.mean(axis=0), u_te.shape)
    def _relative_l2(y):
        return float(np.mean(np.linalg.norm((y-u_te).reshape(n_test, -1), axis=1)
                    / np.linalg.norm(u_te.reshape(n_test, -1), axis=1)))
    baseline_s = time.perf_counter() - baseline_t0
    rel = float(np.mean(np.linalg.norm((u_hat - u_te).reshape(n_test, -1), axis=1)
                        / np.linalg.norm(u_te.reshape(n_test, -1), axis=1)))
    hotspot = float(np.max(np.abs(u_hat.max((1, 2)) - u_te.max((1, 2)))) * DT)
    # Shift both conductivity levels by x2. Because the model was trained on
    # coefficients {3, 12}, this changes the range to {6, 24}, outside the
    # pre-fitted training envelope. Do not silently approve extrapolation.
    a_ood = a_te[:4] * 2.0
    u_ood_ref = np.stack([ob.darcy_2d(a) for a in a_ood])
    u_ood_hat = pred.predict(a_ood)
    ood_report, ood_checks = dc.darcy_trust_report(
        a_ood, u_ood_hat, 1.0, model="PINNeAPPle FNO2d OOD",
        reference=u_ood_ref, envelope=env,
    )
    ood_rel = float(np.mean(
        np.linalg.norm((u_ood_hat - u_ood_ref).reshape(len(a_ood), -1), axis=1)
        / np.linalg.norm(u_ood_ref.reshape(len(a_ood), -1), axis=1)
    ))
    provenance = {"python": sys.version.split()[0], "platform": platform.platform(),
                  "numpy": np.__version__, "grid": n, "n_train": n_train,
                  "n_test": n_test, "source_forcing": 1.0}
    summary = {"model": "pinneapple_neural FNO2d", "padding": padding, "decision": report.decisao,
               "test_rel_l2": rel, "max_hotspot_error_K": hotspot, "fno_parameters": pred.n_parameters(),
               "training": res.to_dict() | {"history": None}, "train_steps": steps,
               "train_seed": train_seed, "test_seed": test_seed, "model_seed": model_seed,
               "infer_ms_per_layout": infer_s * 1e3, "solver_ms_per_layout": solver_s * 1e3,
               "data_generation_s": data_generation_s, "training_wall_s": training_s,
               "baseline_wall_s": baseline_s, "provenance": provenance,
               "baselines": {"training_mean_relative_l2": _relative_l2(mean_pred),
                             "nearest_layout_relative_l2": _relative_l2(nn_pred)},
               "same_data_as": "examples/nvidia/07_omniverse_physicsnemo_coldplate",
               "checks": [c.to_dict() for c in checks],
               "ood": {"shift": "multiply held-out conductivity by 2",
                       "relative_l2": ood_rel, "decision": ood_report.decisao,
                       "checks": [c.to_dict() for c in ood_checks]}}
    if not smoke:
        _figures(out, a_te, u_te, u_hat, checks, report.decisao)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: summary[k] for k in ("decision", "test_rel_l2", "max_hotspot_error_K", "train_steps")},
                     indent=1))
    return summary


def _figures(out, a_te, u_te, u_hat, checks, decision):
    from pinneapple_tools.nvidia.figures import checks_figure, field_panels
    n_test = len(a_te)
    e = np.linalg.norm((u_hat - u_te).reshape(n_test, -1), axis=1) / np.linalg.norm(u_te.reshape(n_test, -1), axis=1)
    i = int(np.argsort(e)[len(e) // 2])                                          # the median-error layout
    est = dc.estimated_error(u_hat[i], a_te[i])
    field_panels([a_te[i] * K0, T_COOL + DT * u_te[i], T_COOL + DT * u_hat[i], DT * np.abs(u_hat[i] - u_te[i]),
                  DT * np.abs(est)],
                 ["conductivity k", "T, finite volumes (reference)", "T, PINNeAPPle FNO2d", "|T_FNO - T_FV|",
                  "error estimated from the residual\n(no reference needed)"],
                 os.path.join(out, "fields.png"), cmaps=["cividis", "inferno", "inferno", "magma", "magma"],
                 units=["W/m/K", "K", "K", "K", "K"], shared=[(1, 2), (3, 4)], extent=(0, 100, 0, 100),
                 suptitle=f"Held-out layout {i} (median error): FNO relative L2 = {e[i]:.3%}")
    checks_figure(checks, os.path.join(out, "checks.png"), title="PINNeAPPle FNO2d, PINNeAPPle checks",
                  decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--padding", type=int, default=8)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--test-seed", type=int, default=1)
    ap.add_argument("--model-seed", type=int, default=None)
    a = ap.parse_args()
    main(a.smoke, steps=a.steps, padding=a.padding, train_seed=a.train_seed,
         test_seed=a.test_seed, model_seed=a.model_seed)
