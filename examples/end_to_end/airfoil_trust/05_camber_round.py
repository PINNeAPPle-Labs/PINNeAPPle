"""Stage 5: open camber as a new parameter and buy the evidence for it.

    python examples/end_to_end/airfoil_trust/05_camber_round.py [--workers N] [--shard I]

The first model is a function of three variables. Camber is not one of them, and no amount of accuracy on the
symmetric family says anything about a cambered section -- the design is simply not in the space the model was
trained in. This stage runs the second campaign: 32 training designs, 12 for calibration and 24 held out, all with
camber strictly inside the new range, so every run adds information the first round could not have.

Results go to ``_runs/cfd`` (``purpose="doe2"``) and the plan to ``results/doe_round2.json``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pinneapple_autolab.domains.airfoil import VARIABLES, AirfoilAero  # noqa: E402
from pinneapple_autolab.study.campaign import (  # noqa: E402
    load_records,
    plan_doe,
    run_campaign,
    space_from_variables,
)
from pinneapple_design.design_optimizer.study import Variable  # noqa: E402

RUNS = os.path.join(HERE, "_runs", "cfd")
RES = os.path.join(HERE, "results")
ITERATIONS = 1000
N_TRAIN, N_CALIB, N_VALID = 32, 12, 24
CAMBER_LOW = 0.003                                     # strictly inside (0, 0.02]: no overlap with round one

args = [a for a in sys.argv[1:] if not a.startswith("--")]
workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
shard = int(sys.argv[sys.argv.index("--shard") + 1]) if "--shard" in sys.argv else None


def cambered_plan() -> dict:
    path = os.path.join(RES, "doe_round2.json")
    if os.path.exists(path):
        return json.load(open(path))
    variables = [*VARIABLES[:3], Variable("camber", CAMBER_LOW, 0.02, unit="1", description="maximum camber")]
    space = space_from_variables(variables, "cambered NACA four-digit family")
    train = plan_doe(space, N_TRAIN, method="maximin", seed=21, id_prefix="t2")
    calib = plan_doe(space, N_CALIB, method="maximin", seed=22, id_prefix="c2", existing=train["unit"])
    valid = plan_doe(space, N_VALID, method="maximin", seed=23, id_prefix="v2",
                     existing=np.vstack([train["unit"], calib["unit"]]))
    out = {"splits": {"train": train["cases"], "calibration": calib["cases"], "validation": valid["cases"]},
           "coverage": {"train": train["coverage"], "calibration": calib["coverage"],
                        "validation": valid["coverage"]},
           "variables": [v.name for v in variables],
           "camber_range": [CAMBER_LOW, 0.02]}
    json.dump(out, open(path, "w"), indent=1)
    return out


def spawn() -> None:
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "doe2",
                               "--shard", str(i), "--workers", str(workers)]) for i in range(workers)]
    for p in procs:
        p.wait()


if shard is None:
    cases = [c for split in cambered_plan()["splits"].values() for c in split]
    print(f"doe2: {len(cases)} designs on {workers} workers", flush=True)
    spawn()
    recs = load_records(RUNS, purposes=["doe2"])
    print(f"doe2: {sum(r.ok for r in recs)} of {len(recs)} done", flush=True)
    sys.exit(0)

plan = cambered_plan()
cases = [c for split in plan["splits"].values() for c in split]
run_campaign(AirfoilAero(mesh="L2", iterations=ITERATIONS, procs=1), cases[shard::workers], RUNS,
             purpose="doe2", retry_failed=True, log=lambda s: print(f"[w{shard}] {s}", flush=True))
