"""Pollutant plume forecast: a PhysicsNeMo FNO steps the field forward, PINNeAPPle decides how far to trust it.

A positive concentration field is carried by a steady wind and diffused on a periodic 1 km x 1 km domain:
c_t + w . grad c = kappa lap c. PINNeAPPle's spectral solver (exact in time) makes the training pairs
c(t) -> c(t + dt); ``physicsnemo.models.fno.FNO`` learns the one-step map and is rolled out autoregressively for 60
steps. PINNeAPPle's rollout validation (``validate_rollout``) gives the trust horizon against the reference and checks
the physics a forecaster must keep without any reference: total mass conserved, variance never growing (diffusion
only removes it), concentration non-negative. PhysicsNeMo only, no Omniverse.

    python examples/nvidia/03_plume_rollout_physicsnemo/run.py           # about 3 min on 4 CPU cores
    python examples/nvidia/03_plume_rollout_physicsnemo/run.py --smoke
"""
import argparse
import json
import os

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_analysis.validation.rollout import validate_rollout
from pinneapple_lab import experiment
from pinneapple_neural.physicsnemo_bridge.field_operators import (
    physicsnemo_fno,
    train_grid_operator,
)
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure, curves_figure
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
WIND = (1.0, 0.5)              # domain lengths per unit time (1 km side: 1 = 1 km per time unit)
KAPPA = 2e-3
DT = 0.05


def initial(n_samples, n, seed):
    g = ob.periodic_grf(n_samples, (n, n), alpha=2.5, tau=5.0, seed=seed)
    return np.exp(g / g.std())                                  # positive, log-normal plumes


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("03_plume_rollout_physicsnemo")
    architecture_diagram(
        "03 Plume forecast: PhysicsNeMo FNO rollout, PINNeAPPle trust horizon",
        [Block("ic", "Initial plumes\n(log-normal random fields)", "pinneapple", 0, 0),
         Block("spec", "Spectral solver, exact in time\nc_t + w.grad c = kappa lap c", "pinneapple", 1, 0),
         Block("fno", "physicsnemo.models.fno.FNO\none-step map c(t) -> c(t+dt)", "physicsnemo", 2, 0),
         Block("roll", "Autoregressive rollout\n60 steps", "physicsnemo", 3, 0),
         Block("val", "validate_rollout: trust horizon,\nmass, variance decay, c >= 0", "pinneapple", 3, 1),
         Block("dec", "How far the forecast\ncan be used", "pinneapple", 4, 1)],
        [("ic", "spec", "c0"), ("spec", "fno", "(c_n, c_n+1)"), ("fno", "roll", "weights"),
         ("roll", "val", "trajectory"), ("spec", "val", "reference"), ("val", "dec", "checks")],
        os.path.join(OUT, "architecture.png"), subtitle="PhysicsNeMo learns and steps; PINNeAPPle generates the "
                                                        "physics and says where the forecast stops being usable")
    n, n_traj, n_steps, n_test, horizon = (16, 8, 6, 2, 10) if smoke else (32, 96, 20, 8, 60)
    budget = ComputeBudget(steps=6) if smoke else ComputeBudget(wall_s=150.0)
    with experiment("examples/nvidia/03_plume_rollout_physicsnemo", title="Plume forecast: PhysicsNeMo FNO rollout "
                    "with PINNeAPPle rollout validation", domain="transport", methods=["FNO (PhysicsNeMo)", "spectral"],
                    out_dir=base, register=False, modules=["pinneapple_analysis.validation.rollout",
                                                           "pinneapple_neural.physicsnemo_bridge"],
                    params={"n": n, "trajectories": n_traj, "dt": DT, "kappa": KAPPA, "wind": WIND,
                            "rollout_steps": horizon}) as ex:
        with info.step("data (spectral solver)"):
            tr = ob.advection_diffusion_2d(initial(n_traj, n, 0), WIND, KAPPA, DT, n_steps)   # (steps+1, traj, n, n)
            a = tr[:-1].reshape(-1, n, n)
            u = tr[1:].reshape(-1, n, n)
            c0 = initial(n_test, n, 1)
            ref = ob.advection_diffusion_2d(c0, WIND, KAPPA, DT, horizon)                       # (h+1, test, n, n)
        pred = physicsnemo_fno(n, latent_channels=24 if not smoke else 8, num_fno_layers=4 if not smoke else 2,
                               num_fno_modes=10 if not smoke else 4, decoder_layer_size=48 if not smoke else 16,
                               padding=0)
        with info.step("train (PhysicsNeMo FNO)"):
            res = train_grid_operator(pred, a, u, budget, batch_size=16 if not smoke else 4, lr=2e-3)
        info.mark("PhysicsNeMo FNO training", ran=True, how=f"CPU, {res.run.steps} steps")
        with info.step("rollout"):
            roll = [c0]
            for _ in range(horizon):
                roll.append(pred.predict(roll[-1]))
            roll = np.stack(roll)                                                                 # (h+1, test, n, n)
        reports, horizons, first = [], [], []
        for j in range(n_test):
            rep = validate_rollout(roll[:, j], 2, ref[:, j], dt=DT, conserved=("mass",),
                                   tolerances={"drift": 1e-2, "error": 0.1})
            reports.append(rep)
            th = rep.evidence["divergence"]["trust_horizon"]
            horizons.append(horizon * DT if th is None else th)
            times = rep.evidence["divergence"]["times"]
            first.append("none" if th is None else min((v, k) for k, v in times.items() if v is not None)[1])
        err = np.linalg.norm((roll - ref).reshape(horizon + 1, n_test, -1), axis=2) / \
            np.linalg.norm(ref.reshape(horizon + 1, n_test, -1), axis=2)
        mass_p, mass_r = roll.mean((2, 3)), ref.mean((2, 3))
        var_p = roll.var(axis=(2, 3))
        var_growth = float(np.max(np.diff(var_p, axis=0) / var_p[:-1]))
        minc = float(roll.min())
        mass_drift = float(np.max(np.abs(mass_p - mass_p[0]) / mass_p[0]))
        one_step = float(err[1].mean())
        checks = [{"name": "one-step relative L2 <= 2 %", "value": one_step, "limit": 0.02, "passed": one_step <= 0.02},
                  {"name": "total mass conserved over the rollout (<= 1 %)", "value": mass_drift, "limit": 0.01,
                   "passed": mass_drift <= 0.01},
                  {"name": "variance never grows (diffusion only)", "value": max(var_growth, 0.0), "limit": 1e-3,
                   "passed": var_growth <= 1e-3, "detail": f"largest step-to-step relative growth {var_growth:.3g}"},
                  {"name": "concentration non-negative", "value": max(-minc, 0.0), "limit": 1e-3,
                   "passed": minc >= -1e-3, "detail": f"min c = {minc:.3g}"},
                  {"name": "every rollout state finite", "value": None, "limit": None,
                   "passed": all(r["rollout.finite"]["status"] == "pass" for r in reports)}]
        th_med = float(np.median(horizons))
        crossed = {k: first.count(k) for k in dict.fromkeys(first)}
        checks.append({"name": "trust horizon >= 1 time unit (20 steps)", "value": th_med, "limit": 1.0,
                       "passed": th_med >= 1.0, "detail": f"median over {n_test} plumes, {th_med / DT:.0f} steps; "
                       f"first criterion crossed: {crossed} (error > 10 %, high-frequency spectrum bias > 25 %, "
                       "invariant deviation > 5 %)"})
        decision = "APPROVED" if all(c["passed"] for c in checks) else (
            "REVIEW" if checks[0]["passed"] and checks[4]["passed"] else "REJECT")
        for c in checks:
            ex.check(c["name"], c["passed"], value=c["value"], detail=c.get("detail", ""))
        ex.log_metric("one_step_rel_l2", one_step, "", f"mean over {n_test} held-out plumes, first step")
        ex.log_metric("trust_horizon_steps", th_med / DT, "steps", "median first step where validate_rollout's error, "
                      "spectrum or invariant criterion is crossed")
        ex.log_metric("mass_drift", mass_drift, "", "max |M(t) - M(0)| / M(0) over the rollouts")
        t = np.arange(horizon + 1) * DT
        curves_figure([{"title": "Rollout error (held-out plumes)", "xlabel": "time", "ylabel": "relative L2",
                        "logy": True, "series": [(f"plume {j}", t[1:], err[1:, j]) for j in range(n_test)],
                        "hlines": [(0.1, "trust limit 10 %", "r")]},
                       {"title": "Total mass M(t)/M(0)", "xlabel": "time", "ylabel": "",
                        "series": [("FNO, plume 0", t, mass_p[:, 0] / mass_p[0, 0]),
                                   ("reference", t, mass_r[:, 0] / mass_r[0, 0], "k--")]},
                       {"title": "Variance (must decay)", "xlabel": "time", "ylabel": "var c",
                        "series": [("FNO, plume 0", t, var_p[:, 0]), ("reference", t, ref.var(axis=(2, 3))[:, 0], "k--")]},
                       {"title": "High-frequency spectrum bias", "xlabel": "time", "ylabel": "|bias|",
                        "series": [(f"plume {j}", t, reports[j].evidence["divergence"]["spectrum_bias"])
                                   for j in range(n_test)], "hlines": [(0.25, "limit 25 %", "r")]}],
                      os.path.join(OUT, "rollout_checks.png"), suptitle=f"Median trust horizon {th_med / DT:.0f} steps")
        checks_figure(checks, os.path.join(OUT, "checks.png"), title="PhysicsNeMo FNO rollout", decision=decision)
        _snapshots(roll[:, 0], ref[:, 0], t, os.path.join(OUT, "snapshots.png"))
        summary = {"decision": decision, "one_step_rel_l2": one_step, "trust_horizon_steps": th_med / DT,
                   "mass_drift": mass_drift, "first_crossing": crossed, "variance_growth": var_growth, "min_c": minc, "checks": checks,
                   "training": res.to_dict() | {"history": None}, "fno_parameters": pred.n_parameters()}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "one_step_rel_l2", "trust_horizon_steps", "mass_drift")},
                     indent=1))
    return summary


def _snapshots(p, r, t, path):
    from pinneapple_tools.nvidia.figures import field_panels
    k = [0, len(t) // 3, 2 * len(t) // 3, len(t) - 1]
    fields = [r[i] for i in k] + [p[i] for i in k]
    titles = [f"reference t={t[i]:.2f}" for i in k] + [f"FNO t={t[i]:.2f}" for i in k]
    field_panels(fields, titles, path, cmaps=["viridis"] * 8, shared=[range(8)], extent=(0, 1000, 0, 1000), ncols=4,
                 units=["c"] * 8, suptitle="Top: reference (spectral, exact in time). Bottom: PhysicsNeMo FNO rollout")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
