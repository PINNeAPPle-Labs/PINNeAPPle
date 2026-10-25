"""Stage 3 of the Ahmed-body study: the CFD runs the surrogate asked for, and what they bought.

    python examples/end_to_end/ahmed_body/03_more_cfd.py [run|evaluate]

``run``: the 8 designs where the GP of CD was most uncertain (``acquisition``), 8 random designs (the yardstick)
and the three Pareto picks at 40 m/s (``verification``), from ``results/next_runs.json`` (resumable).
``evaluate``: the CD and Cp errors on the untouched test split with the DOE training set alone, + the acquired runs,
+ the random runs; the Pareto picks predicted against simulated.
"""
import json
import os
import sys

from pinneapple_autolab.domains import ahmed as A
from pinneapple_autolab.domains import ahmed_study as S
from pinneapple_autolab.study import campaign as C
from pinneapple_autolab.study.dataset import StudyDataset

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, RES = os.path.join(HERE, "_runs", "cfd"), os.path.join(HERE, "results")
plan = json.load(open(os.path.join(RES, "next_runs.json")))
keys = [v.name for v in A.VARIABLES]


def cases(prefix, rows):
    return [{"id": f"{prefix}{i:02d}", **{k: float(r[k]) for k in keys}} for i, r in enumerate(rows)]


if (sys.argv[1] if len(sys.argv) > 1 else "run") == "run":
    sim = A.AhmedAero()
    C.run_campaign(sim, cases("al_", plan["acquisition"]), RUNS, purpose="acquisition",
                   reason="largest GP uncertainty of CD in the training region")
    C.run_campaign(sim, cases("rand_", plan["random"]), RUNS, purpose="acquisition", reason="random yardstick")
    C.run_campaign(sim, cases("pareto_", plan["pareto"]), RUNS, purpose="verification",
                   reason="Pareto pick of the surrogate (CD vs CL at 40 m/s)",
                   predicted={f"pareto_{i:02d}": p["predicted"] for i, p in enumerate(plan["pareto"])})
else:
    ds = StudyDataset.load(os.path.join(RES, "dataset", "ahmed_body"))
    print(json.dumps(S.evaluate_more_runs(ds, S.load_records(RUNS), RES), indent=1))
os._exit(0)
