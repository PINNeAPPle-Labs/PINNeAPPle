"""Periodic unit cell pilot: EMerge Floquet campaign of a generic TCDA cell, an FNO over (frequency x scan), and the
equal-budget comparison of FNO + optimiser against Bayesian optimisation on the solver.

    python examples/use_cases/antennas/unit_cell_benchmark.py --quick        # 3 cells, budget 6 (minutes)
    python examples/use_cases/antennas/unit_cell_benchmark.py --cells 64 --budget 24

Every number comes from pinneapple_simulation.electromagnetics.periodic and the EMerge runs made here
(``pip install emerge``). See docs/guides/unit_cell_benchmark.md.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pinneapple_lab import experiment  # noqa: E402
from pinneapple_simulation.electromagnetics.antenna.runner import (  # noqa: E402
    read_table,
    write_table,
)
from pinneapple_simulation.electromagnetics.periodic import benchmark as bm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cells", type=int, default=64, help="DOE cells for the dataset and the learning curve")
    ap.add_argument("--budget", type=int, default=24, help="solver calls per strategy in the benchmark")
    ap.add_argument("--fno-seconds", type=float, default=60.0)
    ap.add_argument("--out")
    ap.add_argument("--reuse", action="store_true", help="reuse campaign.csv of an earlier run in --out")
    a = ap.parse_args()
    if a.quick:
        freqs, scans, n, budget, nv, fno_s = [3.0, 6.0], [0.0, 30.0], 3, 6, 2, 10.0
    else:
        freqs, scans, n, budget, nv, fno_s = list(np.arange(3.0, 6.01, 0.5)), [0.0, 15.0, 30.0, 45.0], a.cells, \
            a.budget, 4, a.fno_seconds
    out = a.out or os.path.join(HERE, "outputs", "unit_cell_quick" if a.quick else "unit_cell")
    os.makedirs(out, exist_ok=True)
    names = list(bm.CELL_RANGES)
    with experiment("unit_cell_benchmark", title="TCDA unit cell: FNO + optimiser vs Bayesian optimisation",
                    domain="electromagnetics", methods=["FEM (EMerge, Floquet)", "FNO", "Bayesian optimisation"],
                    params={"cells": n, "budget": budget, "freqs_GHz": freqs, "scans_deg": scans}, out_dir=out,
                    register=False) as ex:
        csv = os.path.join(out, "campaign.csv")
        if a.reuse and os.path.exists(csv):                # the solved campaign of an earlier run
            table, cost = read_table(csv), json.load(open(os.path.join(out, "campaign_cost.json")))
        else:
            table, sim = bm.campaign(n, freqs, scans, os.path.join(out, "runs"), seed=0, log=print)
            write_table(table, csv)
            cost = sim.summary()
            with open(os.path.join(out, "campaign_cost.json"), "w") as fh:
                json.dump(cost, fh)
        X, Y, tags = bm.to_arrays(table, names, freqs, scans)
        summary = {"freqs_GHz": freqs, "scans_deg": scans, "campaign": {"cells": n, "solved": len(tags), **cost}}
        # learning curve: the last quarter of the cells is never trained on
        n_test = max(1, len(X) // 4)
        curve = []
        for n_tr in sorted({max(2, int(f * (len(X) - n_test))) for f in (0.25, 0.5, 1.0)}):
            pred, info = bm.fit_fno(X[:n_tr], Y[:n_tr], wall_s=fno_s, seed=0)
            curve.append({"n_train": n_tr, **bm.heldout(pred, X[-n_test:], Y[-n_test:]), "fno": info})
            print("FNO", curve[-1])
        summary["learning_curve"] = curve
        solve = bm.solver_from(bm.cell_simulator(os.path.join(out, "bench_runs"), scans, log=print), names, freqs,
                               scans)
        eb = bm.equal_budget(solve, bm.CELL_RANGES, budget, n_verify=nv, seed=1, fno_wall_s=fno_s)
        summary["equal_budget"] = eb
        with open(os.path.join(out, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot([c["n_train"] for c in curve], [c["rmse_dB"] for c in curve], "o-", label="all (f, scan) points")
        ax[0].plot([c["n_train"] for c in curve], [c["worst_case_rmse_dB"] for c in curve], "s-",
                   label="worst case per cell")
        ax[0].set_xlabel("training cells")
        ax[0].set_ylabel(f"held-out RMSE [dB] ({n_test} cells)")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        for k, lab in (("fno_optimizer", "DOE + FNO + optimiser + verification"), ("bayesian", "Bayesian optimisation")):
            tr = eb[k]["trace"]
            ax[1].plot(np.arange(1, len(tr) + 1), tr, label=lab)
        ax[1].set_xlabel("solver calls")
        ax[1].set_ylabel("best worst-case active reflection [dB]")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "unit_cell_benchmark.png"), dpi=120)
        plt.close(fig)
        ex.log_metric("cell_wall_s_median", cost["wall_s_median_ok"] or 0.0, "s",
                      f"one EMerge process per cell: {len(freqs)} frequencies x {len(scans)} scans")
        ex.log_metric("cell_max_rss_mb", cost["max_rss_mb"] or 0.0, "MB", "peak RSS of a cell process")
        ex.log_metric("fno_heldout_rmse_dB", curve[-1]["rmse_dB"], "dB", f"{n_test} unseen cells")
        for k in ("fno_optimizer", "bayesian"):
            if eb[k]["best_dB"] is not None:
                ex.log_metric(f"{k}_best_dB", eb[k]["best_dB"], "dB", f"solver value, {eb[k]['calls']} calls")


if __name__ == "__main__":
    main()
