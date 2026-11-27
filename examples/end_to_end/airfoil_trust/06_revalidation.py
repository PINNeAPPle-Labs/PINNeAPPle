"""Stage 6: the expanded model, and evidence arriving after the model is frozen.

    python examples/end_to_end/airfoil_trust/06_revalidation.py

Two things happen here that the earlier stages could not show.

The first is that opening a parameter is not free: the model that did not have camber in its schema cannot answer a
query that carries it, and its evidence does not transfer. The second model is a different model -- a different
model id, a wider space, and a domain built from scratch.

The second is that a model version can outlive the evidence it was born with. ``ValidatedDomain.extend`` returns a new
snapshot with more evidence against the *same* model: the question "is this geometry supported" can go from no to yes
without retraining anything, and the record says which references changed the answer.
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
from pinneapple_autolab.domains.airfoil_study import (  # noqa: E402
    INPUTS,
    QOI,
    PodSurrogate,
    accuracy_metrics,
    load_dataset,
    physics_checks,
    pod_basis,
    split_by_design,
)
from pinneapple_veriphysics.validated_domain import DomainPolicy, GeometryEvidence, ValidatedDomain  # noqa: E402

RUNS = os.path.join(HERE, "_runs", "cfd")
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
MODELS = os.path.join(HERE, "models")
MODEL_ID = "naca-family-v2"
BOUNDS = {"alpha_deg": (-4.0, 8.0), "speed": (10.0, 30.0), "thickness": (0.08, 0.16), "camber": (0.0, 0.02)}
NAMES = tuple(BOUNDS)
#: the same policy rules as stage 4: a 5 % bar on lift, the wall condition and the continuity ratio, and a radius and
#: an uncertainty tolerance read from the campaign itself rather than picked.
REFERENCE_TOL = 0.05
PHYSICS_TOLS = (1.5, 2.0)              # ratios to the reference's own departure, as in stage 4
LIFT_FLOOR = 0.10                      # as in stage 4: a relative error stops being one near zero lift
MIN_NEIGHBORS = 2

plan2 = json.load(open(os.path.join(RES, "doe_round2.json")))
splits2 = {k: [c["id"] for c in v] for k, v in plan2["splits"].items()}
plan1 = json.load(open(os.path.join(RES, "doe_round1.json")))
splits1 = {k: [c["id"] for c in v] for k, v in plan1["splits"].items()}

old = load_dataset(RUNS, "doe1")
new = load_dataset(RUNS, "doe2")
old_idx = split_by_design(old["ids"], splits1)
new_idx = split_by_design(new["ids"], splits2)

# ---------------------------------------------------------------- the expanded model
train_idx = old_idx["train"]
X_train = np.vstack([old["X"][train_idx], new["X"][new_idx["train"]]])
Y_train = np.vstack([old["Y"][train_idx], new["Y"][new_idx["train"]]])
F_train = {k: np.vstack([v[train_idx], new["fields"][k][new_idx["train"]]])
           for k, v in old["fields"].items()}

basis, basis_info = pod_basis(F_train, n_modes=10)
model = PodSurrogate({"center": basis[0], "modes": basis[1:], **basis_info}, n_members=5, seed=30)
model.fit(X_train, Y_train, F_train)
model.save(os.path.join(MODELS, "v2"))

def evidence_for(ids, X, Y, prediction):
    """One record per simulated design the second model was not trained on, measured exactly as in stage 4."""
    rows = []
    for k in range(len(X)):
        params = dict(zip(INPUTS, X[k], strict=True))
        predicted = {n: prediction["fields"][n][k] for n in model.basis_meta["names"]}
        reference = {n: prediction["reference_fields"][n][k] for n in model.basis_meta["names"]}
        cl_predicted = float(prediction["qoi"]["CL"][k])
        cl_reference = float(Y[k, QOI.index("CL")])
        checks = physics_checks(predicted, reference, params, speed=params["speed"],
                                alpha_deg=params["alpha_deg"], cl_predicted=cl_predicted,
                                cl_reference=cl_reference)
        rows.append(GeometryEvidence(
            point=tuple(float(v) for v in X[k]), model_id=MODEL_ID, reference_id=ids[k], independent=True,
            reference_error=abs(cl_predicted - cl_reference) / max(abs(cl_reference), LIFT_FLOOR),
            physics_errors=(checks["wall_condition"]["surrogate"] / max(checks["wall_condition"]["reference"], 1e-12),
                            checks["continuity"]["surrogate"] / max(checks["continuity"]["reference"], 1e-12)),
            uncertainty=float(prediction["qoi_std"]["CL"][k]) / max(abs(cl_predicted), LIFT_FLOOR)))
    return rows


held = np.r_[new_idx["calibration"], new_idx["validation"]]
X_held = np.vstack([new["X"][new_idx["calibration"]], new["X"][new_idx["validation"]]])
Y_held = np.vstack([new["Y"][new_idx["calibration"]], new["Y"][new_idx["validation"]]])
pred = model.predict(X_held)
pred["reference_fields"] = {k: np.vstack([new["fields"][k][new_idx["calibration"]],
                                          new["fields"][k][new_idx["validation"]]])
                            for k in model.basis_meta["names"]}
report = {"model_id": MODEL_ID, "n_train": int(len(X_train)), "n_held_out": int(len(held)),
          "basis": basis_info,
          "held_out": accuracy_metrics(pred["qoi"], {k: Y_held[:, j] for j, k in enumerate(QOI)})}

# ---------------------------------------------------------------- the domain of the new model
# The evidence carries the same three numbers as stage 4, measured the same way by the same functions: the error
# against the design's own simulation, the two physical checks, and the ensemble spread. Nothing is carried over --
# the second model is asked about the first campaign's designs again, and it answers with its own predictions, so
# what carries over is the reference simulations and never the evidence.
old_held = np.r_[old_idx["calibration"], old_idx["validation"]]
X_old = old["X"][old_held]
Y_old = old["Y"][old_held]
pred_old = model.predict(X_old)
pred_old["reference_fields"] = {k: old["fields"][k][old_held] for k in model.basis_meta["names"]}
ev_new = evidence_for([new["ids"][int(i)] for i in held], X_held, Y_held, pred)
ev_old = evidence_for([old["ids"][int(i)] for i in old_held], X_old, Y_old, pred_old)
evidence = ev_old + ev_new
n_calibration = len(new_idx["calibration"])

# the policy, from the campaign and not from the answer: the spacing of the training designs, and the spread the
# members show on the calibration designs
train_points = np.vstack([old["X"][old_idx["train"]], new["X"][new_idx["train"]]])
lo = np.array([BOUNDS[n][0] for n in NAMES])
span = np.array([BOUNDS[n][1] - BOUNDS[n][0] for n in NAMES])
norm = (train_points - lo) / span
gaps = np.sort(np.linalg.norm(norm[:, None, :] - norm[None, :, :], axis=-1) + np.eye(len(norm)) * 9.0, axis=1)[:, 0]
calibration_spreads = [float(pred["qoi_std"]["CL"][k]) / max(abs(float(pred["qoi"]["CL"][k])), LIFT_FLOOR)
                       for k in range(len(new_idx["calibration"]))]
POLICY = DomainPolicy(reference_tol=REFERENCE_TOL, physics_tols=PHYSICS_TOLS,
                      uncertainty_tol=float(max(0.01, np.quantile(calibration_spreads, 0.90))),
                      radius=float(np.median(gaps)), min_neighbors=MIN_NEIGHBORS)
print(f"v2 policy: reference {REFERENCE_TOL:.0%}, uncertainty {POLICY.uncertainty_tol:.1%}, "
      f"radius {POLICY.radius:.3f}, at least {MIN_NEIGHBORS} references")

# ---------------------------------------------------------------- evidence arriving after the fact
# The campaign ran in two waves, so the domain is asked the same questions at two moments: after the calibration
# designs had been simulated, and after the held-out batch joined them. Both answers are about the same weights.
first_half = ev_old + ev_new[:n_calibration]
rest = ev_new[n_calibration:]
early = ValidatedDomain(BOUNDS, MODEL_ID, POLICY, training=[tuple(X_train[i]) for i in range(len(X_train))],
                        evidence=first_half)
later = early.extend(rest)

# Two samples, because they answer two different questions and neither is the flattering one.
# The first is uniform over the design box: "how much of the box does the evidence cover", before and after. In four
# dimensions with this many references the answer is a thin shell, and the point of reporting it is that the number is
# small.
rng = np.random.default_rng(11)
candidates = [{"alpha_deg": float(rng.uniform(-4, 8)), "speed": float(rng.uniform(10, 30)),
               "thickness": float(rng.uniform(0.08, 0.16)), "camber": float(rng.uniform(0.003, 0.02))}
              for _ in range(400)]

# The second is built rather than sampled, so that it cannot be tuned: for each late reference, the geometry halfway
# between it and the nearest reference the domain already had. Those are exactly the geometries that had one
# neighbour and now have two, so they are the geometries a new reference can unlock -- and if the domain answered
# them from density alone, all of them would flip. Where it does not, the reason is recorded.
early_points = np.array([e.point for e in first_half])
norm_early = (early_points - lo) / span
near = []
for e in rest:
    d = np.linalg.norm(norm_early - (np.asarray(e.point) - lo) / span, axis=1)
    partner = early_points[int(np.argmin(d))]
    near.append(dict(zip(NAMES, (float(0.5 * (a + b)) for a, b in zip(e.point, partner, strict=True)), strict=True)))


def _flips(sample):
    out = []
    for q in sample:
        a, b = early.assess(q), later.assess(q)
        if a["status"] != b["status"]:
            out.append({"query": q, "before": a, "after": b})
    out.sort(key=lambda d: (d["after"]["status"] != "EMPIRICALLY_SUPPORTED",
                            len(d["after"].get("reference_ids", []))))
    return out


flipped = _flips(candidates)
flipped_near = _flips(near)
sample = flipped_near[:5]
after_reasons: dict[str, int] = {}
for q in near:
    reason = later.assess(q)["status"] if later.assess(q)["status"] == "EMPIRICALLY_SUPPORTED" else later.assess(q)["reason"]
    after_reasons[reason] = after_reasons.get(reason, 0) + 1

out = {**report,
       "domain_before_extend": {"evidence": len(first_half),
                                "supported": sum(early.assess(q)["status"] == "EMPIRICALLY_SUPPORTED"
                                                 for q in candidates)},
       "domain_after_extend": {"evidence": len(evidence),
                               "supported": sum(later.assess(q)["status"] == "EMPIRICALLY_SUPPORTED"
                                                for q in candidates)},
       "status_changes": len(flipped), "examples": sample,
       "midpoints_between_new_and_nearest_old": {"candidates": len(near), "status_changes": len(flipped_near),
                                     "to_supported": sum(f["after"]["status"] == "EMPIRICALLY_SUPPORTED"
                                                         for f in flipped_near),
                                     "reasons_after": after_reasons},
       "policy": {"reference_tol": POLICY.reference_tol, "physics_tols": list(POLICY.physics_tols),
                  "uncertainty_tol": POLICY.uncertainty_tol, "radius": POLICY.radius,
                  "min_neighbors": POLICY.min_neighbors},
       "bounds": BOUNDS}
json.dump(out, open(os.path.join(RES, "revalidation_v2.json"), "w"), indent=1, default=float)
figures.style()
extend_fig = figures.extend(os.path.join(FIG, "extend.png"), out)

q = report["held_out"]
print(f"v2: trained on {len(X_train)} designs (both rounds), held out {len(held)}")
print(f"  CL rmse {q['CL_rmse']:.4f} rel {q['CL_rel_l2']:.2%} | CD rmse {q['CD_rmse']:.5f} "
      f"rel {q['CD_rel_l2']:.2%}")
print(f"  basis {basis_info['n_modes']} modes, {basis_info['energy_captured']:.2%} energy")
print(f"  evidence: {len(first_half)} references before the late batch, {len(evidence)} after "
      f"({len(evidence) - len(first_half)} arrived)")
print(f"  uniform 400 candidates, supported before extend: {out['domain_before_extend']['supported']}, "
      f"after: {out['domain_after_extend']['supported']} ({out['status_changes']} answers changed)")
m = out["midpoints_between_new_and_nearest_old"]
print(f"  the {m['candidates']} midpoints between a late reference and its nearest earlier one: "
      f"{m['status_changes']} answers changed, {m['to_supported']} of them to EMPIRICALLY_SUPPORTED")
if sample:
    s = sample[0]
    print(f"  example: camber {s['query']['camber']:.4f} -> {s['before']['status']} "
          f"({s['before']['reason']}) then {s['after']['status']} ({s['after']['reason']})")
print(f"  figure: {os.path.relpath(extend_fig, ROOT)}")
