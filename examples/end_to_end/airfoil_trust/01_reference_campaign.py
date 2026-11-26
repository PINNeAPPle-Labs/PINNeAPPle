"""Stage 1: the reference simulations. Mesh family with its grid-convergence study, then the design of experiments.

    python examples/end_to_end/airfoil_trust/01_reference_campaign.py [stage ...] [--workers N]

Stages (default: both, in this order; each is resumable -- a run already on disk is skipped):

* ``mesh``  the baseline section on L0, L1, L2 and L3, which is what the grid-convergence study needs. Four meshes,
            not three: with four the finest triplet gives the index and the others show whether the family is in the
            asymptotic range, instead of assuming it.
* ``doe1``  the symmetric family: 32 training designs, 12 for calibrating the uncertainty and 16 held out, drawn so
            that each split is space filling on its own and the three together cover the box. Splitting by whole
            designs -- not by points inside a simulation -- is the only split that makes the held-out error mean
            anything: fields from one design share a geometry and a mesh.

Everything goes to ``_runs/cfd`` (records and fields, not committed) and the plans to ``results/``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from pinneapple_autolab.domains import airfoil_figures as figures  # noqa: E402
from pinneapple_autolab.domains.airfoil import BASELINE, MESH_LEVELS, VARIABLES, AirfoilAero  # noqa: E402
from pinneapple_autolab.study.campaign import (  # noqa: E402
    load_records,
    plan_doe,
    run_campaign,
    space_from_variables,  # noqa: E402
)

RUNS = os.path.join(HERE, "_runs", "cfd")
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
ITERATIONS = 1000
N_TRAIN, N_CALIB, N_VALID = 32, 12, 16

args = [a for a in sys.argv[1:] if not a.startswith("--")]
workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
shard = int(sys.argv[sys.argv.index("--shard") + 1]) if "--shard" in sys.argv else None
stages = args or ["mesh", "doe1"]
os.makedirs(FIG, exist_ok=True)


def symmetric_plan() -> dict:
    """The three splits of the symmetric family, each space filling and jointly covering the box."""
    space = space_from_variables(VARIABLES[:3], "symmetric NACA four-digit family")
    path = os.path.join(RES, "doe_round1.json")
    if os.path.exists(path):
        return json.load(open(path))
    train = plan_doe(space, N_TRAIN, method="maximin", seed=11, id_prefix="tr")
    seen = train["unit"]
    import numpy as np
    calib = plan_doe(space, N_CALIB, method="maximin", seed=12, id_prefix="ca", existing=seen)
    valid = plan_doe(space, N_VALID, method="maximin", seed=13, id_prefix="va",
                     existing=np.vstack([seen, calib["unit"]]))
    for split in (train, calib, valid):                 # the symmetric family: camber is held at zero in round one
        for case in split["cases"]:
            case["camber"] = 0.0
    out = {"splits": {"train": train["cases"], "calibration": calib["cases"], "validation": valid["cases"]},
           "coverage": {"train": train["coverage"], "calibration": calib["coverage"],
                        "validation": valid["coverage"]},
           "random_reference": train["random_reference"],
           "variables": [v.name for v in VARIABLES[:3]],
           "conditions": AirfoilAero().conditions}
    json.dump(out, open(path, "w"), indent=1)
    return out


def spawn(script_args: list[str]) -> None:
    """One worker per shard, in its own process: a worker that dies does not take the campaign with it."""
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), *script_args,
                               "--shard", str(i), "--workers", str(workers)])
             for i in range(workers)]
    for p in procs:
        p.wait()


if shard is None:
    plan = symmetric_plan()
    figures.style()
    drawn = figures.doe_plan(os.path.join(FIG, "doe_plan.png"), plan)
    print(f"plan: {sum(len(v) for v in plan['splits'].values())} designs -> {os.path.relpath(drawn, ROOT)}",
          flush=True)
    if "mesh" in stages:
        for level in MESH_LEVELS:
            recs = run_campaign(AirfoilAero(mesh=level, iterations=ITERATIONS),
                                [{"id": f"mesh_{level}", **BASELINE}], RUNS, purpose="mesh_study")
            print(level, [(r.id, r.ok) for r in recs], flush=True)
    if "doe1" in stages:
        cases = [c for split in plan["splits"].values() for c in split]
        print(f"doe1: {len(cases)} designs on {workers} workers", flush=True)
        spawn(["doe1"])
        recs = load_records(RUNS, purposes=["doe1"])
        print(f"doe1: {sum(r.ok for r in recs)} of {len(recs)} done", flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ worker: one shard of the design of experiments
plan = symmetric_plan()
cases = [c for split in plan["splits"].values() for c in split]
run_campaign(AirfoilAero(mesh="L2", iterations=ITERATIONS, procs=1), cases[shard::workers], RUNS,
             purpose="doe1", retry_failed=True, log=lambda s: print(f"[w{shard}] {s}", flush=True))
