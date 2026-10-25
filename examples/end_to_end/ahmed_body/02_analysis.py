"""Stage 2 of the Ahmed-body study: from the CFD records to a chosen, calibrated, physics-checked surrogate.

    python examples/end_to_end/ahmed_body/02_analysis.py

1. grid convergence of the baseline (L0, L1, L2 [, L3]) and the validation sweep against Ahmed et al. (1984);
2. the dataset (DOE + validation runs; slant > 34 deg held out as the extrapolation test), saved as a DatasetSpec;
3. surrogate families at equal data and budget, cross-conformal 90 % intervals, physics checks, the selection rule;
4. Sobol indices, the Pareto front at 40 m/s with three picks, and the next CFD batch (8 most uncertain + 8 random),
   which ``03_more_cfd.py`` runs.

Library: :mod:`pinneapple_autolab.domains.ahmed_study` (stages) on :mod:`pinneapple_autolab.study`.
"""
import os

from pinneapple_autolab.domains import ahmed_study as S

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, RES = os.path.join(HERE, "_runs", "cfd"), os.path.join(HERE, "results")
if not os.path.isdir(RUNS):                       # a fresh checkout: the committed copy of the records
    RUNS = os.path.join(RES, "cfd_runs")

recs = S.load_records(RUNS)
print("fields re-sampled on the template:", S.refresh_fields(recs, RUNS))
S.grid_and_validation(recs, RES)
ds = S.study_dataset(recs)
ds.save(os.path.join(RES, "dataset", "ahmed_body"), name="ahmed_body_rans_l1",
        description="Ahmed body, OpenFOAM k-omega SST: force coefficients, drag breakdown and surface Cp over slant "
                    "0-40 deg, clearance 30-80 mm, speed 20-60 m/s")
print({k: len(v) for k, v in ds.split.items()})
boards = S.benchmark_all(ds, RES)
print("selection:", boards["selection"])
S.next_runs(ds, boards, RES)
os._exit(0)
