"""Stage 2: what the reference simulations say, and the first surrogate.

    python examples/end_to_end/airfoil_trust/02_analysis.py

Reads ``_runs/cfd`` (nothing is re-simulated), writes ``results/``:

* ``mesh_study.json`` -- the grid-convergence study of CL and CD on the four meshes;
* ``dataset.json`` -- the splits, the cost, and the mesh quality of every run, so the numbers downstream can be
  traced back to a solved case;
* ``analysis_v1.json`` -- the POD basis, the training, and the accuracy on the held-out designs;
* ``models/v1`` -- the ensemble and its basis.

The held-out designs are never seen during training or during the choice of anything: the basis comes from the
training fields, the early choice of the ensemble size comes from the calibration split, and the validation split is
read once, at the end.
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
from pinneapple_autolab.domains.airfoil import MESH_LEVELS  # noqa: E402
from pinneapple_autolab.domains.airfoil_study import (  # noqa: E402
    INPUTS,
    QOI,
    PodSurrogate,
    accuracy_metrics,
    calibration_metrics,
    field_relative_l2,
    load_dataset,
    pod_basis,
    split_by_design,
)
from pinneapple_autolab.study.campaign import load_records  # noqa: E402
from pinneapple_autolab.study.mesh_study import grid_convergence  # noqa: E402

RUNS = os.path.join(HERE, "_runs", "cfd")
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
MODELS = os.path.join(HERE, "models")
os.makedirs(FIG, exist_ok=True)
figures.style()

def _coverage(pred, ref_y) -> dict[str, float]:
    """How often the held-out error falls inside one and two ensemble standard deviations.

    One sigma should hold about 68 % of the cases and two about 95 %. The two numbers are measured and reported
    side by side with the nominal values, because the distance between them is the whole question.
    """
    error = np.concatenate([np.abs(np.asarray(pred["qoi"][k], float) - np.asarray(ref_y[k], float)) for k in QOI])
    sigma = np.concatenate([np.asarray(pred["qoi_std"][k], float) for k in QOI])
    one = calibration_metrics(error, sigma, target=0.68)
    two = calibration_metrics(error, sigma, target=0.95)
    return {"coverage_1s": one["coverage"], "coverage_2s": two["coverage"], "target_1s": 0.68, "target_2s": 0.95,
            "sharpness_1s": one["sharpness"], "ece": one["ece"]}


plan = json.load(open(os.path.join(RES, "doe_round1.json")))
splits = {k: [c["id"] for c in v] for k, v in plan["splits"].items()}

# ------------------------------------------------------------------ grid convergence
mesh_recs = load_records(RUNS, purposes=["mesh_study"])
levels = []
for r in sorted(mesh_recs, key=lambda r: r.meta.get("mesh_quality", {}).get("cells") or 0):
    if not r.ok:
        continue
    q = r.meta.get("mesh_quality", {})
    levels.append({"name": r.id.replace("mesh_", ""), "cells": int(q.get("cells") or 0),
                   "values": {k: float(r.outputs[k]) for k in ("CL", "CD")}, "cost_s": r.cost_s,
                   "nonorth_avg": q.get("nonorth_avg"), "nonorth_max": q.get("nonorth_max"),
                   "skew_max": q.get("skew_max"), "checkMesh_ok": q.get("ok")})
# The representative size of each level is the one the family was built to refine by: the first cell height, which
# the case sets to a factor 1.45 apart between consecutive levels. Neither cells**(-1/3) nor cells**(-1/2) is the
# refinement ratio here -- the family is one cell thick in z, and its cell count does not follow the spacing -- and a
# ratio inferred from the cell count would understate the refinement the study actually made.
sizes = [float(MESH_LEVELS[level["name"]]["h_wall"]) for level in levels]
study = (grid_convergence(levels, ("CL", "CD"), sizes=sizes) if len(levels) >= 3
         else {"error": "fewer than three meshes"})
study["representative_size"] = "first cell height: every spacing is refined by the same factor 1.45 per level"
json.dump({"levels": levels, "study": study}, open(os.path.join(RES, "mesh_study.json"), "w"), indent=1)
print(f"mesh study: {len(levels)} meshes, CL p={study.get('CL', {}).get('p')}, "
      f"CD p={study.get('CD', {}).get('p')}, GCI_fine CL={study.get('CL', {}).get('gci_fine')}")
mesh_fig = figures.mesh_convergence(os.path.join(FIG, "mesh_convergence.png"), {"levels": levels, "study": study})

# ------------------------------------------------------------------ dataset
data = load_dataset(RUNS, "doe1")
idx = split_by_design(data["ids"], splits)
print(f"dataset: {len(data['ids'])} designs ({len(data['failed'])} failed), "
      f"{data['X'].shape[1]} inputs, fields {list(data['fields'])}")

# ------------------------------------------------------------------ POD on the training fields only
train, calib, valid = idx["train"], idx["calibration"], idx["validation"]
train_fields = {k: v[train] for k, v in data["fields"].items()}
basis, basis_info = pod_basis(train_fields, n_modes=8)
pod_fig = figures.pod_spectrum(os.path.join(FIG, "pod_spectrum.png"), basis_info)
model = PodSurrogate({"center": basis[0], "modes": basis[1:], **basis_info})
model.fit(data["X"][train], data["Y"][train], train_fields)

# the basis's own reconstruction error, on designs it never saw: this is a property of the basis, not the model
held_fields = {k: v[valid] for k, v in data["fields"].items()}
coeffs_held = np.hstack([held_fields[n] for n in basis_info["names"]]) - basis[0]
projected = coeffs_held @ basis[1:].T
recon = projected @ basis[1:] + basis[0]
recon_fields, at = {}, 0
for name in basis_info["names"]:
    size = basis_info["sizes"][name]
    recon_fields[name] = recon[:, at:at + size]
    at += size
basis_error = field_relative_l2(recon_fields, held_fields)

# ------------------------------------------------------------------ accuracy
report = {"dataset": {"n": len(data["ids"]), "failed": data["failed"], "cost_s": data["cost_s"],
                      "splits": {k: len(v) for k, v in idx.items()}},
          "basis": {**basis_info, "held_out_reconstruction_rel_l2": basis_error},
          "mesh_study": study}

for split_name, ii in (("calibration", calib), ("validation", valid)):
    pred = model.predict(data["X"][ii])
    ref_y = {k: data["Y"][ii, j] for j, k in enumerate(QOI)}
    ref_f = {k: v[ii] for k, v in data["fields"].items()}
    report[split_name] = {
        "qoi": accuracy_metrics(pred["qoi"], ref_y),
        "field_rel_l2_mean": float(np.mean([field_relative_l2({n: pred["fields"][n][i]}, {n: ref_f[n][i]})
                                            for n in pred["fields"] for i in range(len(ii))])),
        "field_rel_l2_worst": float(max(field_relative_l2({n: pred["fields"][n][i]}, {n: ref_f[n][i]})
                                        for n in pred["fields"] for i in range(len(ii)))),
        "ensemble_spread_qoi": {k: float(np.mean(pred["qoi_std"][k])) for k in QOI},
        # the spread is a disagreement bar until its coverage is measured, on the split that was held out for it
        "calibration": _coverage(pred, ref_y),
    }

model.save(os.path.join(MODELS, "v1"))
json.dump({"inputs": INPUTS, "qoi": QOI, **report}, open(os.path.join(RES, "analysis_v1.json"), "w"), indent=1)

held_pred = model.predict(data["X"][valid])
parity_fig = figures.accuracy(
    os.path.join(FIG, "held_out_parity.png"),
    {q: data["Y"][valid, j] for j, q in enumerate(QOI)}, held_pred["qoi"], held_pred["qoi_std"],
    [data["ids"][i] for i in valid], {q: report["validation"]["qoi"][f"{q}_rmse"] for q in ("CL", "CD")})

for split_name in ("calibration", "validation"):
    q = report[split_name]["qoi"]
    c = report[split_name]["calibration"]
    print(f"{split_name}: CL rmse {q['CL_rmse']:.4f} rel {q['CL_rel_l2']:.2%} | "
          f"CD rmse {q['CD_rmse']:.5f} rel {q['CD_rel_l2']:.2%} | "
          f"field rel L2 mean {report[split_name]['field_rel_l2_mean']:.2%} "
          f"worst {report[split_name]['field_rel_l2_worst']:.2%} | "
          f"spread covers {c['coverage_1s']:.0%} (1 sigma, nominal 68 %) and {c['coverage_2s']:.0%} "
          f"(2 sigma, nominal 95 %), ECE {c['ece']:.3f}")
print(f"basis: {basis_info['n_modes']} modes, {basis_info['energy_captured']:.2%} energy, "
      f"held-out reconstruction {basis_error:.2%}")
print(f"model written to {os.path.join(MODELS, 'v1')}")
print("figures: " + ", ".join(os.path.relpath(p, ROOT) for p in (mesh_fig, pod_fig, parity_fig)))
