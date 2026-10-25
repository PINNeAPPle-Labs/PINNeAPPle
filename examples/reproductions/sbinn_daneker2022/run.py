"""SBINN on the ultradian glucose-insulin model (targets frozen in TARGETS.md before any run).

    python examples/reproductions/sbinn_daneker2022/run.py --seed 0                    # the paper's 10k + 600k epochs
    python examples/reproductions/sbinn_daneker2022/run.py --seed 0 --epochs 20000     # a reduced-scale run

Data as in the paper (arXiv:2202.01723, p. 3): the model integrated from t = 0 to 1800 min with x(0) = [12, 4, 110,
0, 0, 0] and meals (300, 60), (650, 40), (1100, 50); 360 random measurements of G (plus the first and last), all
states known at t = 0 and t = 1799 (L_aux), Vp, Vi, Vg fixed, 17 parameters inferred in the Table 1 ranges. One
torch thread per run (multi-threaded CPU torch was 20x slower per step on this machine); run seeds in parallel.
Writes results/seed<k>.json with the inferred parameters, the history and the forward check.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FREE = ["E", "tp", "ti", "td", "k", "Rm", "a1", "C1", "C2", "C4", "C5", "Ub", "U0", "Um", "Rg", "alpha", "beta"]


def make_data(seed: int):
    from pinneapple_physics.biology.models import ultradian_glucose_insulin

    system = ultradian_glucose_insulin()
    t = np.arange(0, 1800, 1.0)
    traj = system.simulate(t, max_step=1.0)
    rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([[0], rng.choice(np.arange(1, len(t) - 1), 360, replace=False), [len(t) - 1]]))
    return system, t, traj, idx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs-data", type=int, default=10_000)
    ap.add_argument("--epochs", type=int, default=600_000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--budget", type=float, default=float("inf"), help="seconds")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--units", default="raw", choices=["raw", "scaled"])
    a = ap.parse_args(argv)

    import torch

    torch.set_num_threads(a.threads)
    from pinneapple_analysis.inverse_problems.sbinn import SBINN, SBINNConfig
    from pinneapple_physics.biology.models import ULTRADIAN_SEARCH_RANGE

    system, t, traj, idx = make_data(a.seed)
    # raw units with the loss weights of the authors' code (github.com/lu-group/sbinn, sbinn_pytorch.py: ODE weights
    # [1, 1, 1e-2, 1, 1, 1], data 1e-2; the aux term weighted like the data)
    weights = {"data": 1e-2, "ode": 1.0, "aux": 1e-2} if a.units == "raw" else {"data": 1.0, "ode": 1.0, "aux": 1.0}
    cfg = SBINNConfig(epochs_data=a.epochs_data, epochs=a.epochs, seed=a.seed, dtype="float32",
                      record_every=1000, time_budget=a.budget, units=a.units, weights=weights,
                      ode_weights={"G": 1e-2} if a.units == "raw" else {})
    sb = SBINN(system, FREE, ULTRADIAN_SEARCH_RANGE, {"G": (t[idx], traj["G"][idx])}, t_collocation=t,
               aux={s: [traj[s][0], traj[s][-1]] for s in system.states}, aux_times=[t[0], t[-1]], config=cfg)
    t0 = time.time()
    res = sb.fit()
    # forward check: integrate with the inferred parameters on [0, 1800] and forecast the meal at 2000 min
    fc = res.forward_check(system, t, observed=["G"], data={"G": traj["G"]}, max_step=1.0)
    from pinneapple_physics.biology.models import ultradian_glucose_insulin

    meals4 = ((300, 60), (650, 40), (1100, 50), (2000, 100))
    true4 = ultradian_glucose_insulin(meals=meals4)
    tf = np.arange(0, 3000, 1.0)
    ref = true4.simulate(tf, max_step=1.0)
    fit4 = true4.simulate(tf, {k: res.parameters[k] for k in FREE}, max_step=1.0)
    w = tf >= 1800
    forecast = float(np.linalg.norm(fit4["G"][w] - ref["G"][w]) / np.linalg.norm(ref["G"][w]))
    out = {"seed": a.seed, "units": a.units, "weights": weights, "epochs_data": a.epochs_data, "epochs": a.epochs, "epochs_run": res.epochs_run,
           "stopped_by_budget": res.stopped_by_budget, "seconds": time.time() - t0, "threads": a.threads,
           "torch": torch.__version__, "inferred": {k: res.parameters[k] for k in FREE},
           "true": {k: system.parameters[k] for k in FREE},
           "forward_check": {"ode_vs_data_G": fc["ode_vs_data"]["G"], "net_vs_ode": fc["net_vs_ode"]},
           "forecast_rel_l2_G_1800_3000": forecast, "history": res.history}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"seed{a.seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(path, {k: round(v, 4) for k, v in out["inferred"].items()}, "forecast", forecast)


if __name__ == "__main__":
    main()
