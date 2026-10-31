"""BINNs reproduction, part 1: the mechanistic models of Lagergren et al. fitted to the scratch-assay data.

    python examples/reproductions/binns_lagergren2020/classical.py --data <Data/cell_density_profiles.npy>

Targets T1-T4 of TARGETS.md (frozen before any run): GLS errors of the classical FKPP and Generalized Porous-FKPP
models (Table 2) and their parameters (Tab S1, S2). The BINN-guided model's row of Table 2 is fitted here too from
neutral multi-starts; the paper started it from its BINN term curves instead (bin.py runs that path).

No paper number is used as a starting point: every fit starts from a fixed grid of neutral points inside the paper's
bounds and keeps the best. The data file is downloaded from github.com/jlager/BINNs (master @ 41d4b8f3) and read
with a restricted unpickler; it is not committed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UM2_HR = 1e6 / 24          # mm^2/day -> um^2/hr
PER_HR = 1 / 24            # 1/day -> 1/hr

STARTS = {
    "fkpp": [{"D": d, "r": r} for d, r in itertools.product((0.005, 0.03, 0.08), (0.5, 1.2, 2.0))],
    "porous_fkpp": [{"D": d, "m": m, "r": r} for d, m, r in itertools.product((0.01, 0.05, 0.09), (0.5, 1.5, 3.0),
                                                                               (0.6, 1.4))],
    "delay_reaction_diffusion": [{"D0": d0, "D": d, "m": m, "r": r, "beta0": b0, "beta1": b1}
                                 for d0, d, m, r, b0, b1 in itertools.product((0.002, 0.02), (0.03, 0.09), (1.0, 3.0),
                                                                              (1.2, 2.0), (-3.0, 0.0), (2.0, 6.0))],
}


def to_paper_units(p: dict) -> dict:
    out = {}
    for k, v in p.items():
        out[k] = v * UM2_HR if k in ("D", "D0") else v * PER_HR if k in ("r", "beta1") else v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "classical.json"))
    ap.add_argument("--models", default="fkpp,porous_fkpp,delay_reaction_diffusion")
    a = ap.parse_args(argv)
    from pinneapple_physics.biology.cell_migration import (
        DENSITIES,
        fit_mechanistic,
        load_scratch_assay,
    )

    results = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            results = json.load(f)
    for model in a.models.split(","):
        results.setdefault(model, {})
        for dens in DENSITIES:
            if str(dens) in results[model]:
                continue
            assay = load_scratch_assay(a.data, dens)
            t0 = time.time()
            # finished starts are kept in "_partial" so a killed run resumes at the next start
            partial = results.setdefault("_partial", {}).setdefault(model, {}).setdefault(str(dens), [])
            for x0 in STARTS[model][len(partial):]:
                partial.append(fit_mechanistic(model, assay, x0))
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                with open(a.out, "w") as f:
                    json.dump(results, f, indent=1)
            fits = results["_partial"][model].pop(str(dens))
            best = min(fits, key=lambda r: r["gls"])
            results[model][str(dens)] = {"gls": best["gls"], "params": best["params"],
                                         "params_paper_units": to_paper_units(best["params"]),
                                         "n_starts": len(fits), "spread_gls": [min(f["gls"] for f in fits),
                                                                               max(f["gls"] for f in fits)],
                                         "seconds": time.time() - t0}
            print(model, dens, round(best["gls"], 2), {k: round(v, 4) for k, v in
                                                       to_paper_units(best["params"]).items()}, flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(results, f, indent=1)
    if not any(v for m in results.get("_partial", {}).values() for v in m.values()):
        results.pop("_partial", None)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
