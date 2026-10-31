"""Stage 7 of the Ahmed-body study: cost accounting, the evidence report and the experiment manifest.

    python examples/end_to_end/ahmed_body/07_report.py

Reads every JSON the other stages wrote into ``results/`` (a stage that was not run reads as "not measured"),
writes ``results/report.html`` (self-contained: text, tables, figures), ``results/evidence.zip`` (every source
JSON hashed, verified on write), ``results/costs.json`` and ``results/summary.json`` (the measured numbers with
units and how they were measured), and records them in ``experiment.json`` through ``pinneapple_lab.experiment``.
"""
import os

import pinneapple_lab
from pinneapple_autolab.domains import ahmed as A
from pinneapple_autolab.domains import ahmed_study as S

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, RES = os.path.join(HERE, "_runs", "cfd"), os.path.join(HERE, "results")
if not os.path.isdir(RUNS):                       # a fresh checkout: the committed copy of the records
    RUNS = os.path.join(RES, "cfd_runs")

with pinneapple_lab.experiment(
        "ahmed_body_end_to_end", title="Ahmed body: an end-to-end CFD surrogate study", domain="fluids",
        description="OpenFOAM k-omega SST campaign on the Ahmed body (mesh study, validation against Ahmed et al. "
                    "1984, maximin DOE), surrogate families at equal budget with cross-conformal intervals and physics "
                    "checks, Sobol indices, a verified Pareto front, acquisition against random runs, an amortised "
                    "inverse on the wind-tunnel drag, a digital twin and an evidence report.",
        methods=["OpenFOAM simpleFoam k-omega SST", "GP", "MLP ensemble", "quadratic RSM", "POD-GP", "POD-MLP",
                 "POD-RBF", "DeepONet", "cross-conformal intervals", "NSGA-II", "Sobol (Saltelli)", "MDN"],
        modules=["pinneapple_autolab.study", "pinneapple_autolab.domains.ahmed", "pinneapple_autolab.domains.ahmed_study",
                 "pinneapple_analysis.inverse_problems.amortized"],
        references=[A.AHMED_1984_SOURCE, A.MEASUREMENTS["lienhart_becker_2003"]["source"]],
        out_dir=HERE, register=False) as ex:
    recs = S.load_records(RUNS)
    summary = S.study_report(recs, RES)
    for m in summary["metrics"]:
        ex.log_metric(m["name"], m["value"], m["unit"], m["how"])
    ex.check("evidence_package_intact", passed=summary["evidence_problems"] == [],
             detail="pinneapple_veriphysics.evidence_package.verify_package on evidence.zip")
    ex.check("every_stage_measured", passed=not summary["not_measured"],
             detail="stages without a result file: " + (", ".join(summary["not_measured"]) or "none"))
    for name in ("report.html", "evidence.zip", "summary.json", "costs.json"):
        ex.add_file(os.path.join(RES, name))
    for f in sorted(os.listdir(os.path.join(RES, "figures"))):
        ex.add_file(os.path.join(RES, "figures", f))
    print(summary["report"], len(summary["metrics"]), "metrics; not measured:", summary["not_measured"] or "nothing")
