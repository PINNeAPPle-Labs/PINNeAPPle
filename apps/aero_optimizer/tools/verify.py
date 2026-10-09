"""Check the optimizer's picks with OpenFOAM: full polars for designs along the Pareto front, compared with the surrogate.

    python apps/aero_optimizer/tools/verify.py MODEL.pt OUT_DIR [--designs 6] [--workers 4] [--round 1]

OUT_DIR/runs/ gets the CFD runs in the training-set format (set "verify"), so they can be added to the next training
(active learning: --extra OUT_DIR). OUT_DIR/verification_round<N>.json holds, per design, the surrogate prediction and
the OpenFOAM result for top speed, CO2 per 100 km, stall speed and the polar.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
from generate_dataset import one                                          # noqa: E402
from pinneapple_design.aero.aircraft import Aircraft, Polar, Requirements, evaluate  # noqa: E402
from pinneapple_design.aero.geometry import REFERENCE, properties        # noqa: E402
from pinneapple_design.aero.optimize import WING_MASS_PER_M2, Engine     # noqa: E402

CFD_ALPHAS = [-2, 0, 2, 4, 6, 8, 10, 12, 14]


def pick(front, designs, n):
    """n designs spread along the front (by top speed), always including both ends."""
    if len(front) <= n:
        return list(front)
    idx = np.unique(np.round(np.linspace(0, len(front) - 1, n)).astype(int))
    return [front[i] for i in idx]


def steady_values(runs, max_osc=0.1):
    """Same treatment as training: oscillating runs averaged over the kept snapshots, deep stall dropped."""
    out = []
    for r in runs:
        if r.get("status") == "unsteady":
            if r.get("Cl_osc", 9) > max_osc:
                continue
            H = np.array(r["history"])[-5:]
            r = {**r, "Cl": float(H[:, 1].mean()), "Cd": float(H[:, 2].mean()), "Cm": float(H[:, 3].mean())}
        out.append(r)
    return out


def cfd_eval(runs, shape, area, ac, req):
    runs = sorted(steady_values(runs), key=lambda r: r["alpha"])
    a = np.array([r["alpha"] for r in runs])
    cl, cd, cm = (np.array([r[k] for r in runs]) for k in ("Cl", "Cd", "Cm"))
    acx = dataclasses.replace(ac, wing_area=area, mass=ac.mass + WING_MASS_PER_M2 * (area - ac.wing_area))
    e = evaluate(properties(shape), Polar(a, cl, cd, cm), acx, req)
    e["polar"] = {"alpha": a.tolist(), "cl": cl.tolist(), "cd": cd.tolist(), "cm": cm.tolist(),
                  "status": [r["status"] for r in runs]}
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--designs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    eng = Engine.load(a.model)
    ac, req = Aircraft(), Requirements()
    res = eng.search(ac, req, population=96, generations=60, seed=a.seed)
    D = res["designs"]
    chosen = pick(res["pareto"], D, a.designs)
    print(f"{res['evaluations']} designs evaluated, {len(res['pareto'])} on the front, verifying {len(chosen)}", flush=True)
    jobs, meta = [], []
    for n, i in enumerate(chosen):
        d = D[i]
        did = f"r{a.round}_d{n}"
        meta.append((did, d))
        for al in CFD_ALPHAS:
            jobs.append({"id": f"{did}_a{al:+03d}", "set": "verify", "shape_id": did, "shape": d["shape"],
                         "alpha": float(al), "wing_area": d["wing_area"]})
    jobs = [j for j in jobs if not os.path.exists(os.path.join(a.out, "runs", j["id"] + ".json"))]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, jobs, [a.out] * len(jobs)):
            print(*r, flush=True)
    out = []
    for did, d in meta:
        runs = [json.load(open(os.path.join(a.out, "runs", f"{did}_a{al:+03d}.json"))) for al in CFD_ALPHAS]
        runs = [r for r in runs if "Cl" in r]
        cfd = cfd_eval(runs, np.array(d["shape"]), d["wing_area"], ac, req)
        # surrogate on the same angles as the CFD (same clmax resolution)
        co = eng.mlp_coefficients(np.array(d["shape"])[None], np.array(CFD_ALPHAS, float))
        acx = dataclasses.replace(ac, wing_area=d["wing_area"], mass=d["mass"])
        sur = evaluate(d["props"], Polar(np.array(CFD_ALPHAS, float), co["cl"][0], co["cd"][0], co["cm"][0]), acx, req)
        out.append({"id": did, "shape": d["shape"], "wing_area": d["wing_area"], "props": d["props"],
                    "surrogate": {k: sur[k] for k in ("vmax_kt", "co2_100km", "v_stall_kt", "cruise_ld", "clmax", "feasible", "violations")},
                    "surrogate_fine": {k: d[k] for k in ("vmax_kt", "co2_100km", "v_stall_kt", "cruise_ld", "clmax", "feasible", "trust")},
                    "surrogate_polar": {"alpha": CFD_ALPHAS, "cl": co["cl"][0].tolist(), "cd": co["cd"][0].tolist(), "cm": co["cm"][0].tolist()},
                    "openfoam": {k: cfd[k] for k in ("vmax_kt", "co2_100km", "v_stall_kt", "cruise_ld", "clmax", "feasible", "violations", "polar")}})
        s, c = out[-1]["surrogate"], out[-1]["openfoam"]
        print(f"{did}: vmax {s['vmax_kt']:.1f} vs CFD {c['vmax_kt']:.1f} kt | CO2 {s['co2_100km']:.2f} vs {c['co2_100km']:.2f} | "
              f"stall {s['v_stall_kt']:.1f} vs {c['v_stall_kt']:.1f} kt | feasible {s['feasible']} vs {c['feasible']}", flush=True)
    json.dump({"round": a.round, "aircraft": dataclasses.asdict(ac), "requirements": req.as_dict(), "designs": out,
               "evaluations": res["evaluations"], "pareto_size": len(res["pareto"])},
              open(os.path.join(a.out, f"verification_round{a.round}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
