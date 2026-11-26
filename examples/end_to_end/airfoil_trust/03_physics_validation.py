"""Stage 3: does the surrogate obey the physics it claims, not only the data?

    python examples/end_to_end/airfoil_trust/03_physics_validation.py

Accuracy against held-out simulations is one question; whether the prediction is a flow at all is another, and a
model can pass the first while failing the second. Three of the checks are the verification pipeline's own gates, run
through ``run_surrogate_gates`` on a bundle built over the lattice:

``gate_accuracy``
    relative L2 of the predicted pressure coefficient against the reference, on the same lattice.
``gate_boundary``
    the Dirichlet condition the case declares -- no-slip at the wall, which is the lower edge of the lattice in the
    wall-normal direction. The gate reads the condition from the bundle, not from this script.
``gate_residual``
    the continuity residual of the predicted velocity, differenced on the curvilinear lattice in physical space and
    normalised by the size of its own gradients.

Two more are the study's, because the gates do not cover them: ``freestream`` (the outermost samples should have
decayed to the freestream) and ``cp_integral`` (the lift implied by integrating the predicted surface pressure against
the lift the same network predicts -- a field and a scalar output that disagree are two predictions, not one).

Then an input-sensitivity measurement, so that "robustness" in the report is a number and not a promise: every input
of every held-out design is nudged by one percent of its range, and the movement of the prediction is compared with
the ensemble spread.

Results go to ``results/physics_v1.json``. The checks themselves live in ``pinneapple_autolab.domains.airfoil_study``
because stage 6 runs the same ones on the second campaign -- a second copy would be a second definition.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pinneapple_autolab.domains import airfoil_figures as figures  # noqa: E402
from pinneapple_autolab.domains.airfoil import STATIONS, VARIABLES  # noqa: E402
from pinneapple_autolab.domains.airfoil_study import (  # noqa: E402
    INPUTS,
    QOI,
    PodSurrogate,
    input_sensitivity,
    load_dataset,
    physics_checks,
    split_by_design,
)

RUNS = os.path.join(HERE, "_runs", "cfd")
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
figures.style()
RES = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")

plan = json.load(open(os.path.join(RES, "doe_round1.json")))
splits = {k: [c["id"] for c in v] for k, v in plan["splits"].items()}
data = load_dataset(RUNS, "doe1")
idx = split_by_design(data["ids"], splits)
model = PodSurrogate.load(os.path.join(MODELS, "v1"))

# The checks run on every design the study will later treat as evidence -- the sixteen held out and the twelve used
# for calibration -- because a reference that carries no physical checks cannot support a claim, and the domain in
# stage 4 reads exactly these records. The summary below is reported over the held-out split, which is the one the
# accuracy claim is about.
held = np.r_[idx["validation"], idx["calibration"]]
split_of = {int(i): "validation" for i in idx["validation"]} | {int(i): "calibration" for i in idx["calibration"]}
pred = model.predict(data["X"][held])
report = {"n_validation": int(len(idx["validation"])), "n_evidence": int(len(held)), "per_design": []}
for k, i in enumerate(held):
    params = dict(zip(INPUTS, data["X"][i], strict=True))
    params["id"] = data["ids"][i]
    reference = {n: data["fields"][n][i] for n in model.basis_meta["names"]}
    predicted = {n: pred["fields"][n][k] for n in model.basis_meta["names"]}
    checks = physics_checks(predicted, reference, params, speed=params["speed"], alpha_deg=params["alpha_deg"],
                            cl_predicted=float(pred["qoi"]["CL"][k]),
                            cl_reference=float(data["Y"][i, QOI.index("CL")]))
    checks["id"] = data["ids"][i]
    checks["split"] = split_of[int(i)]
    checks["params"] = {k: float(v) for k, v in params.items() if k != "id"}
    report["per_design"].append(checks)

keys = ("gate_accuracy", "gate_boundary", "gate_residual", "wall_condition", "continuity", "freestream",
        "cp_integral")
summary = {}
for key in keys:
    rows = [d for d in report["per_design"] if d["split"] == "validation"]
    values = [d[key]["surrogate"] if "surrogate" in d[key] else d[key]["gap"] for d in rows]
    summary[key] = {
        "surrogate_mean": float(np.mean(values)),
        "reference_mean": float(np.mean([d[key].get("reference", d[key].get("reference_gap", np.nan))
                                         for d in rows])),
        "worst": float(np.max(values)),
        "n": len(rows)}
report["summary"] = summary

bounds = {v.name: (v.lower, v.upper) for v in VARIABLES}
sensitivity_rows = input_sensitivity(model, data["X"][idx["validation"]], bounds)
report["sensitivity"] = sensitivity_rows
report["sensitivity_worst"] = max(r["ratio"] for r in sensitivity_rows.values())
json.dump(report, open(os.path.join(RES, "physics_v1.json"), "w"), indent=1)

gates_fig = figures.physics_gates(os.path.join(FIG, "physics_gates.png"), summary, (
    ("gate_accuracy", "field accuracy (Cp)"), ("gate_boundary", "no-slip wall edge"),
    ("gate_residual", "continuity residual"), ("wall_condition", "|U| at the wall"),
    ("freestream", "freestream decay")))
worst_row = max((d for d in report["per_design"] if d["split"] == "validation"),
                key=lambda d: d["cp_integral"]["gap"])
held_ids = [data["ids"][i] for i in held]
k_worst = held_ids.index(worst_row["id"])
ref_wall = figures.wall_pressure(data["fields"]["cp"][int(held[k_worst])])
prd_wall = figures.wall_pressure(pred["fields"]["cp"][k_worst])
trace_fig = figures.error_trace(
    os.path.join(FIG, "fields.png"), np.asarray(STATIONS, float),
    np.r_[ref_wall[0], ref_wall[1]], np.r_[prd_wall[0], prd_wall[1]], worst_row["id"])
sens_fig = figures.sensitivity(os.path.join(FIG, "sensitivity.png"), sensitivity_rows)

print(f"physics on {len(held)} evidence designs ({len(idx['validation'])} held out, "
      f"{len(idx['calibration'])} calibration); the summary is over the held-out split:")
for key in keys:
    s = summary[key]
    print(f"  {key:15s} surrogate {s['surrogate_mean']:.3e}   reference {s['reference_mean']:.3e}   "
          f"worst {s['worst']:.3e}")
worst = max((d for d in report["per_design"] if d["split"] == "validation"),
            key=lambda d: d["cp_integral"]["gap"])
print(f"  largest Cp-integral gap: {worst['cp_integral']['gap']:.3f} CL at {worst['id']}")
print("input sensitivity, the movement of CL under a 1 % nudge of each input, over the ensemble spread:")
for name, row in sensitivity_rows.items():
    print(f"  {name:10s} step {row['step']:.4f}  change {row['relative_change_CL']:.2e}  "
          f"spread {row['ensemble_spread_CL']:.2e}  ratio {row['ratio']:.2f}")
print("figures: " + ", ".join(os.path.relpath(p, ROOT) for p in (gates_fig, trace_fig, sens_fig)))
