"""Stage 4 of the Ahmed-body study: infer the slant angle from a drag measurement, with its posterior.

    python examples/end_to_end/ahmed_body/04_inverse.py

The amortised posterior p(slant | CD, clearance, speed) is a mixture density network trained on 40 000 draws of the
selected CD surrogate (noise: the surrogate's calibrated uncertainty + the CFD-vs-wind-tunnel discrepancy).
Synthetic checks first (calibration on surrogate draws; every held-out CFD case's slant from its CFD drag), then
the measurements of Ahmed et al. (1984) at 60 m/s, with a leave-one-out discrepancy (a measurement never
calibrates its own inversion), against multi-start least squares and Metropolis on the same likelihood.
"""
import json
import os

from pinneapple_autolab.domains import ahmed_study as S
from pinneapple_autolab.study.dataset import StudyDataset

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, RES = os.path.join(HERE, "_runs", "cfd"), os.path.join(HERE, "results")
if not os.path.isdir(RUNS):                       # a fresh checkout: the committed copy of the records
    RUNS = os.path.join(RES, "cfd_runs")
ds = StudyDataset.load(os.path.join(RES, "dataset", "ahmed_body"))
out = S.inverse(ds, S.load_records(RUNS, ok_only=True), RES)
print(json.dumps({k: out[k] for k in ("heldout_inside_hdi90", "sigma_d_all")}, indent=1))
for r in out["measurements"]:
    print(r["slant_deg"], r["CD_measured"], round(r["mdn_median"], 1), r["mdn_hdi90"], r["least_squares"])
os._exit(0)
