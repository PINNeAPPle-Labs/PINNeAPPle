"""Stage 4: where the model's predictions are backed by evidence, and where they are not.

    python examples/end_to_end/airfoil_trust/04_validated_domain.py

A small held-out error says the model does well *on average*. It says nothing about whether a particular geometry has
anything behind it. This stage builds the evidence domain of the trained model: one record per held-out design, each
carrying the error measured against its own independent simulation, the two physical checks and the ensemble spread;
and it answers three queries that look alike to a marginal-range check and are not alike at all.

Results go to ``results/domain_v1.json``.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pinneapple_autolab.domains.airfoil_study import (  # noqa: E402
    QOI,
    PodSurrogate,
    load_dataset,
    split_by_design,
)
from pinneapple_veriphysics.validated_domain import DomainPolicy, GeometryEvidence, ValidatedDomain  # noqa: E402

RUNS = os.path.join(HERE, "_runs", "cfd")
RES = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")
MODEL_ID = "naca-family-v1"

#: The quality bar the study declares before looking at the answer: 5 % on lift, the wall condition within 2 % of
#: the freestream speed, and continuity no worse than twice the reference's own residual.
REFERENCE_TOL = 0.05
#: both physical checks read as ratios to what the reference itself does, not as absolutes: a wall-function solution
#: has no sample at the wall, so "the predicted wall speed is 2 % of the freestream" says nothing until the same
#: number for the solver is beside it. 1.5 x the reference's own departure, and twice its continuity residual.
PHYSICS_TOLS = (1.5, 2.0)
#: The neighbourhood radius is the geometry of the evidence, not a quality threshold, so it is read from the design
#: of experiments rather than picked: the median distance between a training design and its nearest neighbour in the
#: normalised box (set below, once the campaign is on disk). A radius smaller than the campaign's own spacing asks a
#: question no campaign of this size can answer; a larger one would call two designs neighbours that share nothing.
#: ``MIN_NEIGHBORS`` is the second half of the same statement: one reference is an anecdote.
MIN_NEIGHBORS = 2
#: The uncertainty tolerance is not a quality bar, and pretending otherwise would be the mistake this whole example is
#: about: it is how much the members of *this* ensemble disagree on designs it was calibrated on, at the 90th
#: percentile. A model whose members disagree by 20 % everywhere cannot be held to 5 % -- the honest response is a
#: tolerance that says so out loud, printed in the report next to everything it let through.
UNCERTAINTY_QUANTILE = 0.90
#: A relative error on lift stops being a relative error near zero incidence, where a symmetric section carries no
#: lift: an absolute error of 0.004 is 4 % of a design at four degrees and 400 % of a design at zero. Both the error
#: and the spread are therefore read against the larger of the value and this floor -- a fifth of the lift the family
#: reaches -- and the report says which denominator was used.
LIFT_FLOOR = 0.10

#: v1 is a function of three variables: camber is not in this schema, and a query that carries it is not a query
#: this model can answer at all -- which is the whole point of the third query below.
NAMES = ("alpha_deg", "speed", "thickness")
BOUNDS = {"alpha_deg": (-4.0, 8.0), "speed": (10.0, 30.0), "thickness": (0.08, 0.16)}


def uncertainty_of(pred, k: int) -> float:
    """The ensemble's relative spread on the quantity the declared use turns on.

    Lift, not the average over every output: a moment coefficient sits near zero by construction, so its *relative*
    spread is large whatever the model does, and a mean that includes it measures the offset of a coordinate system.
    """
    return float(pred["qoi_std"]["CL"][k]) / max(abs(float(pred["qoi"]["CL"][k])), LIFT_FLOOR)


def build_evidence(model, data, idx, physics: dict) -> list[GeometryEvidence]:
    """One record per held-out design: what its own simulation says about the model there."""
    by_id = {d["id"]: d for d in physics["per_design"]} if physics else {}
    held = list(idx["validation"]) + list(idx["calibration"])
    pred = model.predict(data["X"][held])
    evidence = []
    for k, i in enumerate(held):
        rid = data["ids"][i]
        point = tuple(float(v) for v in data["X"][i][:len(NAMES)])
        ref = {q: float(data["Y"][i, j]) for j, q in enumerate(QOI)}
        prd = {q: float(pred["qoi"][q][k]) for q in QOI}
        # the error the declared use turns on: lift, whose tolerance the validation profile states. Drag is compared
        # in the dossier, where a quantity with its own tolerance can be given its own line.
        rel = abs(prd["CL"] - ref["CL"]) / max(abs(ref["CL"]), LIFT_FLOOR)
        checks = by_id.get(rid, {})
        wall = checks.get("wall_condition", {}).get("surrogate", np.nan)
        wall_ref = checks.get("wall_condition", {}).get("reference", np.nan)
        cont = checks.get("continuity", {}).get("surrogate", np.nan)
        cont_ref = checks.get("continuity", {}).get("reference", np.nan)
        wall_ratio = wall / wall_ref if wall_ref and np.isfinite(wall_ref) else np.nan
        cont_ratio = cont / cont_ref if cont_ref and np.isfinite(cont_ref) else np.nan
        evidence.append(GeometryEvidence(
            point=point, model_id=MODEL_ID, reference_id=rid, independent=True,
            reference_error=float(rel),
            physics_errors=(float(wall_ratio) if np.isfinite(wall_ratio) else 9.99,
                            float(cont_ratio) if np.isfinite(cont_ratio) else 9.99),
            uncertainty=uncertainty_of(pred, k)))
    return evidence


plan = json.load(open(os.path.join(RES, "doe_round1.json")))
splits = {k: [c["id"] for c in v] for k, v in plan["splits"].items()}
data = load_dataset(RUNS, "doe1")
idx = split_by_design(data["ids"], splits)
model = PodSurrogate.load(os.path.join(MODELS, "v1"))
physics = json.load(open(os.path.join(RES, "physics_v1.json"))) if os.path.exists(
    os.path.join(RES, "physics_v1.json")) else {}

train_points = np.array([[float(v) for v in data["X"][i][:len(NAMES)]] for i in idx["train"]])
lo = np.array([BOUNDS[n][0] for n in NAMES])
span = np.array([BOUNDS[n][1] - BOUNDS[n][0] for n in NAMES])
norm = (train_points - lo) / span
gaps = np.sort(np.linalg.norm(norm[:, None, :] - norm[None, :, :], axis=-1) + np.eye(len(norm)) * 9.0, axis=1)[:, 0]
RADIUS = float(np.median(gaps))
calib_pred = model.predict(data["X"][idx["calibration"]])
calibration_spreads = [uncertainty_of(calib_pred, k) for k in range(len(idx["calibration"]))]
uncertainty_tol = float(max(0.01, np.quantile(calibration_spreads, UNCERTAINTY_QUANTILE)))
POLICY = DomainPolicy(reference_tol=REFERENCE_TOL, physics_tols=PHYSICS_TOLS, uncertainty_tol=uncertainty_tol,
                      radius=RADIUS, min_neighbors=MIN_NEIGHBORS)
print(f"policy: reference error {REFERENCE_TOL:.0%}, wall ratio {PHYSICS_TOLS[0]:.1f}, continuity ratio "
      f"{PHYSICS_TOLS[1]:.1f}, uncertainty {uncertainty_tol:.1%} (the {UNCERTAINTY_QUANTILE:.0%} quantile of the "
      f"calibration spread), radius {RADIUS:.3f} (the median distance between two training designs), at least "
      f"{MIN_NEIGHBORS} references")

evidence = build_evidence(model, data, idx, physics)
domain = ValidatedDomain(BOUNDS, MODEL_ID, POLICY,
                         training=[data["X"][i][:len(NAMES)] for i in idx["train"]], evidence=evidence)

# --- Four queries that a marginal-range check cannot tell apart from each other.
# ---
# --- The first is a design nobody simulated that is surrounded by designs that were: the midpoint of the two closest
# --- evidence points. It is the case the domain exists to answer yes to, and it is defined from the evidence rather
# --- than picked to look good.
points = np.array([e.point for e in evidence])
pair_gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1) + np.eye(len(points)) * 9.0
a, b = np.unravel_index(np.argmin(pair_gaps), pair_gaps.shape)
between = 0.5 * (points[a] + points[b])
queries = {
    "locally_supported": dict(zip(NAMES, (float(v) for v in between), strict=True)),
    # the exact centre of the design box: every marginal range is as satisfied as it can be, which is the whole of
    # what a per-variable range check can say, and the answer turns on whether independent references are near it.
    "inside_but_uncovered": {"alpha_deg": 0.5 * (BOUNDS["alpha_deg"][0] + BOUNDS["alpha_deg"][1]),
                             "speed": 0.5 * (BOUNDS["speed"][0] + BOUNDS["speed"][1]),
                             "thickness": 0.5 * (BOUNDS["thickness"][0] + BOUNDS["thickness"][1])},
    "outside_training_bounds": {"alpha_deg": 4.0, "speed": 20.0, "thickness": 0.18},
    "new_parameter_dimension": {"alpha_deg": 4.0, "speed": 20.0, "thickness": 0.12, "camber": 0.015},
}
results = {name: domain.assess(q) for name, q in queries.items()}

# --- what to simulate next: the candidates with no evidence behind them
candidates = []
rng = np.random.default_rng(7)
for _ in range(200):
    candidates.append({"alpha_deg": float(rng.uniform(-4, 8)), "speed": float(rng.uniform(10, 30)),
                       "thickness": float(rng.uniform(0.08, 0.16))})
todo = domain.prioritize(candidates)

# --- how much the answer depends on the neighbourhood radius. The radius is a policy choice, not a measurement, so
# --- the study reports the curve instead of defending one point of it: a claim that only survives at one radius is
# --- a claim about the radius, not about the model.
radii = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
share_by_radius = {}
for r in radii:
    wider = ValidatedDomain(BOUNDS, MODEL_ID,
                            DomainPolicy(POLICY.reference_tol, POLICY.physics_tols, POLICY.uncertainty_tol,
                                         r, POLICY.min_neighbors),
                            training=[data["X"][i][:len(NAMES)] for i in idx["train"]], evidence=evidence)
    share_by_radius[r] = float(np.mean([wider.assess(q)["status"] == "EMPIRICALLY_SUPPORTED"
                                        for q in candidates]))

out = {"model_id": MODEL_ID, "policy": {"reference_tol": POLICY.reference_tol,
                                        "physics_tols": list(POLICY.physics_tols),
                                        "uncertainty_tol": POLICY.uncertainty_tol,
                                        "radius": POLICY.radius, "min_neighbors": POLICY.min_neighbors},
       "bounds": BOUNDS, "n_training": int(len(idx["train"])), "n_evidence": len(evidence),
       "evidence": [e.__dict__ for e in evidence],
       "queries": {k: {"query": queries[k], "result": results[k]} for k in queries},
       "evidence_gap": {"candidates": len(candidates), "unsupported": len(todo),
                        "examples": [{"query": q, "reason": r["reason"]} for q, r in todo[:10]]},
       "supported_share_of_candidates": float(1 - len(todo) / len(candidates)),
       "radius_sensitivity": {"radii": list(radii), "supported_share": share_by_radius}}
json.dump(out, open(os.path.join(RES, "domain_v1.json"), "w"), indent=1, default=float)

print(f"domain {MODEL_ID}: {len(evidence)} evidence records, {len(idx['train'])} training designs, "
      f"radius {POLICY.radius}")
for name in queries:
    r = results[name]
    print(f"  {name:22s} -> {r['status']:22s} ({r['reason']})")
print(f"  candidates without support: {len(todo)} of {len(candidates)} "
      f"({out['supported_share_of_candidates']:.0%} supported)")
print("  supported share against the neighbourhood radius: "
      + ", ".join(f"{r:.2f}: {v:.0%}" for r, v in share_by_radius.items()))
