"""BINNs reproduction, part 2: the biologically-informed network itself, and the BINN-guided model it suggests.

    python examples/reproductions/binns_lagergren2020/binn.py --data <Data/cell_density_profiles.npy> [--iters 20000]

Per data set: train ``ReactionDiffusionBINN`` (paper architecture, losses and constraints; ``--retrains`` runs
instead of the paper's 20, ``--iters`` steps with the paper's 5,000-step early stopping), keep the run whose forward
PDE solution has the lowest GLS error, score the learned terms against the paper's qualitative description (T6),
then fit the BINN-guided delay-reaction-diffusion model (Eq 11-12) by L-BFGS-B starting from the closed forms fitted
to the learned curves, as the paper does (T1 row "BINN-guided model", T5). Fewer retrains or steps than the paper are
reported as reduced-scale. Writes results/binn.json (one entry per data set, resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def shape_checks(b, assay) -> dict:
    """T6: D non-zero at u = 0, increasing and concave up; G approximately linear (R^2 of a line); T increasing
    and sigmoidal (fit of a logistic)."""
    u = np.linspace(float(assay.U.min()), float(assay.U.max()), 100)
    D, G = b.predict_term("D", u), b.predict_term("G", u)
    d0 = float(b.predict_term("D", [0.0])[0])
    dD, ddD = np.gradient(D, u), np.gradient(np.gradient(D, u), u)
    A = np.column_stack([u, np.ones_like(u)])
    res = G - A @ np.linalg.lstsq(A, G, rcond=None)[0]
    r2 = 1 - float(np.sum(res ** 2) / max(np.sum((G - G.mean()) ** 2), 1e-30))
    out = {"D_at_0": d0, "D_increasing_fraction": float(np.mean(dD >= 0)),
           "D_concave_up_fraction": float(np.mean(ddD >= -1e-12)), "G_linear_R2": r2,
           "G_decreasing_fraction": float(np.mean(np.gradient(G, u) <= 0))}
    if "T" in b.terms:
        t = np.linspace(float(assay.t.min()), float(assay.t.max()), 50)
        T = b.predict_term("T", t)
        out["T_increasing_fraction"] = float(np.mean(np.diff(T) >= -1e-9))
        out["T_range"] = [float(T.min()), float(T.max())]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--retrains", type=int, default=1)
    ap.add_argument("--densities", default="10000,12000,14000,16000,18000,20000")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "binn.json"))
    a = ap.parse_args(argv)
    import torch

    torch.set_num_threads(1)
    from pinneapple_analysis.inverse_problems.binn import BINNConfig, ReactionDiffusionBINN
    from pinneapple_physics.biology.cell_migration import fit_mechanistic, load_scratch_assay

    results = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for dens in [int(d) for d in a.densities.split(",")]:
        if str(dens) in results:
            continue
        assay = load_scratch_assay(a.data, dens)
        runs = []
        for k in range(a.retrains):
            t0 = time.time()
            b = ReactionDiffusionBINN(BINNConfig(n_iters=a.iters, seed=k)).fit(assay.x, assay.t, assay.U)
            g, _ = b.forward_gls()
            runs.append({"seed": k, "forward_gls": g, "iters": b.iters_run, "best_val": b.best_val,
                         "seconds": time.time() - t0, "binn": b})
            print(dens, k, "forward GLS", round(g, 2), "iters", b.iters_run, flush=True)
        best = min(runs, key=lambda r: r["forward_gls"])
        b = best["binn"]
        guess = b.mechanistic_guess()
        mech = fit_mechanistic("delay_reaction_diffusion", assay, guess)
        p = mech["params"]
        results[str(dens)] = {
            "retrains": [{k: v for k, v in r.items() if k != "binn"} for r in runs],
            "best_seed": best["seed"], "forward_gls_learned_terms": best["forward_gls"],
            "shape": shape_checks(b, assay), "guess": guess,
            "binn_guided": {"gls": mech["gls"], "params": p,
                            "params_paper_units": {"D0": p["D0"] * 1e6 / 24, "D": p["D"] * 1e6 / 24, "m": p["m"],
                                                   "r": p["r"] / 24, "beta0": p["beta0"], "beta1": p["beta1"] / 24}},
            "reduced_scale": a.retrains < 20}
        print(dens, "BINN-guided GLS", round(mech["gls"], 2), {k: round(v, 4) for k, v in
                                                            results[str(dens)]["binn_guided"]["params_paper_units"]
                                                            .items()}, flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
