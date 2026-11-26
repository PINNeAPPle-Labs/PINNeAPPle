"""Stage 7: the engineering evidence report, and the figures the demo shows.

    python examples/end_to_end/airfoil_trust/07_report.py

Everything here is a rendering of numbers the earlier stages measured -- this script runs no check of its own and
invents no value. The dossier is built with the validation-case machinery, so every verdict carries the check that
produced it, the reference it was compared against, and the revision it was sealed in.

Writes ``results/dossier.md``, ``results/dossier.txt``, ``results/dossier.json``, ``results/figures/*.png`` and,
through ``pinneapple_lab.experiment``, ``experiment.json``.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from dataclasses import replace  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pinneapple_lab  # noqa: E402
from pinneapple_autolab.domains.airfoil import STATIONS  # noqa: E402
from pinneapple_autolab.domains.airfoil_study import (  # noqa: E402
    QOI,
    PodSurrogate,
    load_dataset,
    split_by_design,
)
from pinneapple_veriphysics.dossier import ValidationDossier  # noqa: E402
from pinneapple_veriphysics.ledger import EvidenceLedger, sha256_file, sha256_json, utc_now  # noqa: E402
from pinneapple_veriphysics.usecase import ValidationProfile  # noqa: E402
from pinneapple_veriphysics.validated_domain import DomainPolicy, GeometryEvidence, ValidatedDomain  # noqa: E402
from pinneapple_veriphysics.validation_case import (  # noqa: E402
    ApplicabilityFacts,
    CheckVerdict,
    ModelVersion,
    ProblemRef,
    ReferenceComparison,
    UncertaintyEvidence,
    ValidationCase,
    checks_from_ledger,
)

RUNS = os.path.join(HERE, "_runs", "cfd")
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
MODELS = os.path.join(HERE, "models")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREY, GREEN = "#58a6ff", "#ffa657", "#8b949e", "#3fb950"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.edgecolor": "#444", "axes.labelcolor": "#ddd", "text.color": "#ddd",
                     "xtick.color": "#bbb", "ytick.color": "#bbb", "figure.facecolor": "#0d1117",
                     "axes.facecolor": "#0d1117", "savefig.facecolor": "#0d1117"})

analysis = json.load(open(os.path.join(RES, "analysis_v1.json")))
physics = json.load(open(os.path.join(RES, "physics_v1.json")))
domain_report = json.load(open(os.path.join(RES, "domain_v1.json")))
mesh = json.load(open(os.path.join(RES, "mesh_study.json")))
plan = json.load(open(os.path.join(RES, "doe_round1.json")))
splits = {k: [c["id"] for c in v] for k, v in plan["splits"].items()}
data = load_dataset(RUNS, "doe1")
idx = split_by_design(data["ids"], splits)
model = PodSurrogate.load(os.path.join(MODELS, "v1"))
valid, calib = idx["validation"], idx["calibration"]
pred = model.predict(data["X"][valid])

# --------------------------------------------------------------------------- the spread, and what it is worth
# The ensemble spread is a disagreement bar. Whether it is a confidence interval is a measurement, not a claim:
# count how often the held-out error of the calibration designs actually falls inside it.
pred_cal = model.predict(data["X"][calib])
z_scores = []
for k, q in enumerate(QOI):
    sd = np.asarray(pred_cal["qoi_std"][q], float)
    sd = np.where(sd > 0, sd, 1e-12)
    z_scores.append(np.abs(np.asarray(pred_cal["qoi"][q], float) - data["Y"][calib, k]) / sd)
z_scores = np.concatenate(z_scores)
coverage_1s = float(np.mean(z_scores <= 1.0))
coverage_2s = float(np.mean(z_scores <= 2.0))
# a one-sigma bar should cover about 68 % of the time; wider than that is not "calibrated" either, it is a bar that
# buys coverage by being uninformative, and the report says which of the two happened
calibrated = bool(0.60 <= coverage_1s <= 0.76)
coverage_note = ("wider than nominal: the bar is conservative, not sharp" if coverage_1s > 0.76
                 else "narrower than nominal: the bar under-covers" if coverage_1s < 0.60 else "within nominal")


# --------------------------------------------------------------------------- figures
def fig_mesh_convergence() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, q, colour in ((axes[0], "CL", BLUE), (axes[1], "CD", ORANGE)):
        cells = [m["cells"] for m in mesh["levels"]]
        vals = [m["values"][q] for m in mesh["levels"]]
        ax.semilogx(cells, vals, "o-", color=colour, lw=1.6, ms=5, label="OpenFOAM")
        ext = mesh["study"][q].get("extrapolated")
        if ext is not None:
            ax.axhline(ext, color=GREY, ls="--", lw=1.0, label=f"Richardson {ext:.4f}")
            gci = mesh["study"][q].get("gci_fine")
            if gci:
                ax.fill_between([min(cells) * 0.7, max(cells) * 1.3], ext * (1 - gci), ext * (1 + gci),
                                color=GREY, alpha=0.15, label=f"GCI {gci:.2%}")
        ax.set_xticks(cells)
        ax.set_xticklabels([f"{c / 1000:g}k" for c in cells], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("cells")
        ax.set_ylabel(q)
        ax.legend(fontsize=7, framealpha=0.2)
        p = mesh["study"][q].get("p")
        ax.set_title(f"{q}: observed order {p:.2f}" if p else f"{q}: {mesh['study'][q].get('status')}",
                     color="#ddd", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG, "mesh_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_accuracy() -> str:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    for ax, q in zip(axes[:2], ("CL", "CD"), strict=False):
        ref = data["Y"][valid, QOI.index(q)]
        pr = pred["qoi"][q]
        lo, hi = float(min(ref.min(), pr.min())), float(max(ref.max(), pr.max()))
        pad = 0.08 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GREY, lw=0.8, ls="--")
        ax.errorbar(ref, pr, yerr=pred["qoi_std"][q], fmt="o", ms=4, color=BLUE, ecolor=GREY, elinewidth=0.7,
                    capsize=0, label="held-out designs")
        ax.set_xlabel(f"{q} reference")
        ax.set_ylabel(f"{q} surrogate")
        ax.set_title(f"{q}: RMSE {analysis['validation']['qoi'][f'{q}_rmse']:.4f}", color="#ddd", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.2)
    err = np.array([physics["per_design"][k]["cp_integral"]["gap"] for k in range(len(valid))])
    axes[2].bar(np.arange(len(valid)), err, color=ORANGE)
    axes[2].set_xlabel("held-out design")
    axes[2].set_ylabel("|CL from Cp - CL predicted|")
    axes[2].set_title("pressure integral vs predicted lift", color="#ddd", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG, "accuracy.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_fields() -> str:
    worst = int(np.argmax([physics["per_design"][k]["cp_integral"]["gap"] for k in range(len(valid))]))
    i = valid[worst]
    n_st = len(STATIONS)
    ref_cp = np.asarray(data["fields"]["cp"][i]).ravel()
    prd_cp = np.asarray(pred["fields"]["cp"][worst]).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for side, label in ((0, "upper"), (1, "lower")):
        sl = slice(side * n_st, (side + 1) * n_st)
        style = "-" if side == 0 else "--"
        axes[0].plot(STATIONS, -ref_cp[sl], style, color=BLUE, lw=1.4, label=f"reference {label}")
        axes[0].plot(STATIONS, -prd_cp[sl], style, color=ORANGE, lw=1.1, label=f"surrogate {label}")
        axes[1].plot(STATIONS, prd_cp[sl] - ref_cp[sl], style, color=ORANGE, lw=1.1)
    axes[0].set_xlabel("x / c")
    axes[0].set_ylabel("-Cp")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, framealpha=0.2)
    axes[0].set_title(f"surface pressure, {data['ids'][i]}", color="#ddd", fontsize=9)
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].set_xlabel("x / c")
    axes[1].set_ylabel("Cp error")
    axes[1].set_title("where the model is wrong", color="#ddd", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG, "fields.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_uncertainty() -> str:
    """What the ensemble spread is worth: the held-out error in units of the spread, and the coverage it buys."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].hist(np.clip(z_scores, 0, 6), bins=np.linspace(0, 6, 25), color=BLUE, alpha=0.85)
    for x, lab in ((1.0, f"one sigma ({coverage_1s:.0%})"), (2.0, f"two ({coverage_2s:.0%})")):
        axes[0].axvline(x, color=ORANGE if x == 1.0 else GREEN, ls="--", lw=1.2, label=lab)
    axes[0].set_xlabel("|error| / ensemble spread")
    axes[0].set_ylabel("held-out cases")
    axes[0].set_title("is the spread the right size?", color="#ddd", fontsize=9)
    axes[0].legend(fontsize=7, framealpha=0.2)
    grid = np.linspace(0.05, 3.0, 60)
    cover = [float(np.mean(z_scores <= g)) for g in grid]
    axes[1].plot(grid, cover, color=BLUE, lw=1.6, label="measured")
    axes[1].plot(grid, [1 - np.exp(-0.5 * g ** 2) for g in grid], color=GREY, ls="--", lw=1.0,
                 label="a calibrated Gaussian")
    axes[1].set_xlabel("bar width, in ensemble standard deviations")
    axes[1].set_ylabel("fraction of errors inside")
    axes[1].set_title("calibration curve", color="#ddd", fontsize=9)
    axes[1].legend(fontsize=7, framealpha=0.2)
    fig.tight_layout()
    path = os.path.join(FIG, "uncertainty.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def rebuild_domain() -> ValidatedDomain:
    """The domain stage 4 built, rebuilt from its own record so the figure and the case read the same evidence."""
    p = domain_report["policy"]
    policy = DomainPolicy(reference_tol=p["reference_tol"], physics_tols=tuple(p["physics_tols"]),
                          uncertainty_tol=p["uncertainty_tol"], radius=p["radius"],
                          min_neighbors=p["min_neighbors"])
    evidence = [GeometryEvidence(point=tuple(e["point"]), model_id=e["model_id"], reference_id=e["reference_id"],
                                 independent=e["independent"], reference_error=e["reference_error"],
                                 physics_errors=tuple(e["physics_errors"]), uncertainty=e["uncertainty"])
                for e in domain_report["evidence"]]
    return ValidatedDomain(domain_report["bounds"], domain_report["model_id"], policy,
                           training=[tuple(float(v) for v in data["X"][i][:3]) for i in idx["train"]],
                           evidence=evidence)


def fig_domain() -> str:
    """The evidence map at three speeds: the supported region is a shape in the design space, not a range."""
    dom = rebuild_domain()
    alphas = np.linspace(-4, 8, 121)      # fine enough to draw the boundary, not to flatter it
    thick = np.linspace(0.08, 0.16, 81)
    speeds = (10.0, 20.0, 30.0)
    fig, axes = plt.subplots(1, len(speeds), figsize=(8.4, 2.9), sharey=True)
    for ax, speed in zip(axes, speeds, strict=True):
        grid = np.zeros((len(thick), len(alphas)))
        for a, al in enumerate(alphas):
            for t, th in enumerate(thick):
                r = dom.assess({"alpha_deg": float(al), "speed": float(speed), "thickness": float(th)})
                grid[t, a] = 1.0 if r["status"] == "EMPIRICALLY_SUPPORTED" else 0.0
        ax.contourf(alphas, thick * 100, grid, levels=[-0.5, 0.5, 1.5], colors=["#21262d", "#0b3d24"])
        if grid.max() > 0:
            ax.contour(alphas, thick * 100, grid, levels=[0.5], colors=[GREEN], linewidths=1.0)
        near = [e for e in domain_report["evidence"] if abs(e["point"][1] - speed) <= 5.0]
        ax.plot([e["point"][0] for e in near], [e["point"][2] * 100 for e in near], "o", ms=3.5, color=BLUE,
                label="independent reference" if speed == speeds[0] else None)
        ax.set_xlabel("angle of attack [deg]")
        ax.set_title(f"{speed:.0f} m/s", color="#ddd", fontsize=9)
        if speed == speeds[0]:
            ax.set_ylabel("thickness [% chord]")
            ax.plot([], [], "s", color="#0b3d24", ms=8, label="supported")
            ax.legend(fontsize=7, framealpha=0.2, loc="lower right")
    fig.suptitle("where the evidence supports a prediction, at three freestream speeds", color="#ddd", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG, "domain_map.png")
    fig.savefig(path)
    plt.close(fig)
    return path


figures = [fig_mesh_convergence(), fig_accuracy(), fig_fields(), fig_uncertainty(), fig_domain()]

# --------------------------------------------------------------------------- the case
acc = analysis["validation"]["qoi"]
phy = physics["summary"]
tolerance = 0.05                                    # the study's declared tolerance, set before the answer was looked at
LIFT_FLOOR = 0.10                                   # as in the domain stage: no relative error at zero lift
# The mesh family converges on lift only in the sense that matters: successive refinements move CL by about a
# percent, in both directions, so no observed order and no grid-convergence index can be claimed -- the family
# oscillates at the level of its own discretisation error. What the study can state is the change between the two
# finest meshes, and that is what the report uses. Drag is the opposite case: it converges monotonically, but the
# index on the finest mesh is large, which is a statement about the reference, not about the surrogate.
cl_change = float(mesh["study"]["CL"]["error_vs_finest"][-2])
cl_oscillatory = mesh["study"]["CL"].get("p") is None
cd_gci = mesh["study"].get("CD", {}).get("gci_fine")
cd_gci_value = float(cd_gci) if cd_gci is not None else float("nan")
# The condition the dossier assesses applicability at is the one stage 4 defined and reported: the centre of the
# design box, where every marginal range is satisfied and the only open question is whether evidence sits near it.
query = {k: float(v) for k, v in domain_report["queries"]["inside_but_uncovered"]["query"].items()
         if k in domain_report["bounds"]}
verdict = rebuild_domain().assess(query)
# the field the scalar is computed from has its own accuracy, and its own floor: the POD basis reconstructs the
# held-out fields to a few percent on its own, so the model is asked to stay within twice that.
recon_floor = float(analysis["basis"]["held_out_reconstruction_rel_l2"])
field_rel = float(analysis["validation"]["field_rel_l2_mean"])
field_bar = 2.0 * recon_floor
def _ratio(key):
    """The surrogate's number over the reference's own, when the check has a yardstick on both sides."""
    ref = float(phy.get(key, {}).get("reference_mean") or float("nan"))
    return float(phy.get(key, {}).get("surrogate_mean", float("nan"))) / ref if np.isfinite(ref) and ref else float("nan")


wall_ratio = _ratio("wall_condition")
residual_ratio = _ratio("gate_residual")
freestream_ratio = _ratio("freestream")
integral_ratio = _ratio("cp_integral")

checks = [
    CheckVerdict("mesh_convergence_CL", "verification", cl_change <= tolerance, value=cl_change,
                 threshold=tolerance, critical=True,
                 message=(f"change in CL between the two finest of {len(mesh['levels'])} meshes"
                          + ("; the family oscillates at this level, so no grid-convergence index is claimed"
                             if cl_oscillatory else "")),
                 evidence="results/mesh_study.json"),
    CheckVerdict("gate_continuity_residual", "verification", residual_ratio <= 2.0,
                 value=residual_ratio, threshold=2.0, critical=True,
                 message=(f"the prediction's continuity residual is {phy['gate_residual']['surrogate_mean']:.3f} "
                          f"against the reference's own {phy['gate_residual']['reference_mean']:.3f} -- both "
                          "normalised by the size of their own gradients"),
                 evidence="results/physics_v1.json"),
    CheckVerdict("boundary_condition_no_slip", "verification", wall_ratio <= 1.5,
                 value=wall_ratio, threshold=1.5, critical=True,
                 message=(f"the declared Dirichlet condition at the wall: the prediction departs from it by "
                          f"{phy['wall_condition']['surrogate_mean']:.4f} of the freestream, the reference itself by "
                          f"{phy['wall_condition']['reference_mean']:.4f} -- a ratio of {wall_ratio:.2f}, because a "
                          "wall-function solution has no sample at the wall"),
                 evidence="results/physics_v1.json"),
    CheckVerdict("freestream_decay", "physics", freestream_ratio <= 1.5,
                 value=freestream_ratio, threshold=1.5,
                 message=(f"the outermost ring departs from the freestream by {phy['freestream']['surrogate_mean']:.3f} "
                          f"of U_inf, against the reference's {phy['freestream']['reference_mean']:.3f} -- the ring is "
                          "at a quarter chord from the surface, so neither reaches zero"),
                 evidence="results/physics_v1.json"),
    CheckVerdict("pressure_integral_consistency", "physics", integral_ratio <= 1.5,
                 value=integral_ratio, threshold=1.5,
                 message=(f"lift from integrating the predicted wall pressure differs from the predicted CL by "
                          f"{phy['cp_integral']['surrogate_mean']:.3f}; the same integral on the reference differs "
                          f"from the reference's CL by {phy['cp_integral']['reference_mean']:.3f}"),
                 evidence="results/physics_v1.json"),
    CheckVerdict("accuracy_CL", "validation", acc["CL_rel_l2"] <= tolerance, value=acc["CL_rel_l2"],
                 threshold=tolerance, critical=True,
                 message=f"relative L2 on {len(valid)} designs held out by design",
                 evidence="results/analysis_v1.json"),
    CheckVerdict("field_accuracy", "validation", field_rel <= field_bar, value=field_rel, threshold=field_bar,
                 critical=True,
                 message=(f"relative L2 over u, v and cp on the lattice, mean over the held-out designs: "
                          f"{field_rel:.1%}. The {analysis['basis']['n_modes']}-mode basis alone reconstructs the "
                          f"same fields to {recon_floor:.1%}, so the bar is twice the floor the model's own "
                          "representation imposes -- a scalar can be right while the field it integrates is not"),
                 evidence="results/analysis_v1.json"),
    CheckVerdict("applicability_domain_measured", "applicability", verdict["status"] == "EMPIRICALLY_SUPPORTED",
                 message=f"evidence domain queried at {query}: {verdict['status']} ({verdict['reason']})",
                 evidence="results/domain_v1.json"),
    CheckVerdict("uncertainty_calibration_coverage", "uncertainty", calibrated, value=coverage_1s, threshold=0.68,
                 message=f"the raw ensemble spread covers {coverage_1s:.0%} of held-out errors at one sigma "
                         f"({coverage_note})",
                 evidence="results/analysis_v1.json"),
    CheckVerdict("robustness_input_sensitivity", "robustness",
                 float(physics["sensitivity_worst"]) <= 1.0, value=float(physics["sensitivity_worst"]), threshold=1.0,
                 message="movement of CL under a 1 % nudge of each input, over the ensemble spread, worst input: "
                         + ", ".join(f"{k} {v['ratio']:.2f}" for k, v in physics["sensitivity"].items()),
                 evidence="results/physics_v1.json"),
    CheckVerdict("reproducibility_rerun", "reproducibility", None,
                 message="not run in this round: no independent re-execution of the campaign", evidence=""),
]

# --------------------------------------------------------------------------- the ledger
# A check reads "passed" because a sealed record shows that it ran, not because this script says so. Each check that
# actually ran is recorded against the SHA-256 of the file its number came from, and the verdicts are then derived
# from the ledger -- so a check whose evidence file has changed under it, or whose record is edited afterwards, comes
# back as "not run" instead of as a pass. The one check that did not run has no record to derive anything from, and is
# added back explicitly: dropping it is the failure the ledger exists to prevent.
LEDGER = EvidenceLedger()
LEDGER_PATH = os.path.join(RES, "ledger.json")
_started = utc_now()
for c in checks:
    if c.passed is None:
        continue
    source = os.path.join(ROOT, c.evidence) if c.evidence else ""
    LEDGER.record(c.check_id, passed=c.passed, name=c.check_id, category=c.category,
                  inputs_sha256=(sha256_file(source) if source and os.path.exists(source)
                                 else sha256_json({"check": c.check_id, "value": c.value})),
                  outputs={"value": c.value, "threshold": c.threshold, "detail": c.message},
                  started_at=_started, finished_at=utc_now(), tool="pinneapple_veriphysics",
                  tool_version=getattr(pinneapple_lab, "__version__", ""))
_records = {c.check_id: c.evidence for c in checks}
_not_run = tuple(c for c in checks if c.passed is None)
checks = tuple(replace(v, evidence=_records.get(v.check_id, v.evidence))
               for v in checks_from_ledger(LEDGER)) + _not_run
json.dump(LEDGER.to_dict(), open(LEDGER_PATH, "w"), indent=1)

model_version = ModelVersion(
    name="naca-family-surrogate", version="v1",
    configuration=tuple(sorted({"architecture": "MLP ensemble on POD coefficients",
                                "members": "5", "pod_modes": str(analysis["basis"]["n_modes"]),
                                "inputs": ",".join(analysis["inputs"]),
                                "energy_captured": f"{analysis['basis']['energy_captured']:.6f}"}.items())))
problem = ProblemRef(
    description="steady incompressible flow over a NACA four-digit section, unit chord, sea-level air",
    pde="steady incompressible RANS, k-omega SST with wall functions",
    geometry="NACA four-digit, thickness 8-16 % of chord, zero camber in this schema",
    conditions=tuple(f"{k}={v}" for k, v in plan["conditions"].items() if isinstance(v, (int, float))))

case = ValidationCase.open(
    profile=ValidationProfile(name="airfoil preliminary screening", objective="industrial",
                              primary_metric="CL", target_error_pct=100 * tolerance,
                              reference_evidence=("numerical reference",),
                              critical_quantities=("CL",),
                              decision="screen candidate sections of a family before committing a design cycle",
                              consequence="medium"),
    model=model_version, problem=problem, actor="study",
    title="NACA four-digit surrogate: an evidence-based validation study",
    requested_use="rank candidate sections of a family for a first design pass")
case = case.with_checks(checks)
for k, i in enumerate(valid):
    rel = 100.0 * abs(float(pred["qoi"]["CL"][k]) - float(data["Y"][i, QOI.index("CL")])) / max(
        abs(float(data["Y"][i, QOI.index("CL")])), LIFT_FLOOR)
    case = case.with_reference(ReferenceComparison(
        evidence_id=f"cfd-{data['ids'][i]}", kind="numerical reference", metric="CL",
        value_pct=rel, tolerance_pct=100 * tolerance, independent=True, covers=("CL",),
        description=f"OpenFOAM simpleFoam run of design {data['ids'][i]}, never seen in training",
        source="examples/end_to_end/airfoil_trust/_runs/cfd"), actor="study")
# Drag is compared against the uncertainty the *reference* carries on drag, not against the lift tolerance: a 17 %
# grid-convergence index at the production mesh is the smallest difference this reference can resolve, so a tighter
# bar would compare the model with the reference's own noise. The comparison is recorded either way -- inconclusive
# is a result, and it is the one drag gets here.
case = case.with_reference(ReferenceComparison(
    evidence_id="cfd-held-out-CD", kind="numerical reference", metric="CD",
    value_pct=100.0 * acc["CD_rel_l2"],
    tolerance_pct=100.0 * cd_gci_value if np.isfinite(cd_gci_value) else 100 * tolerance,
    independent=True, covers=("CD",),
    description=(f"relative L2 over {len(valid)} designs held out by design, against the grid-convergence index of "
                 f"the reference on drag ({100 * cd_gci_value:.0f} % at the finest mesh) rather than against the "
                 "lift tolerance"),
    source="results/analysis_v1.json"), actor="study")
case = case.with_uncertainty(UncertaintyEvidence(
    method="ensemble spread over five members on the POD coefficients",
    calibrated=calibrated, spread_pct=100.0 * float(np.mean(pred_cal["qoi_std"]["CL"])),
    coverage=coverage_1s, target_coverage=0.68,
    limitation=("a disagreement bar detects instability; it is a confidence interval only once its coverage is "
                f"measured, and here one sigma covers {coverage_1s:.0%} of the held-out errors "
                f"(two sigma, {coverage_2s:.0%})"),
    evidence="results/analysis_v1.json"), actor="study")
outside = () if verdict["status"] == "EMPIRICALLY_SUPPORTED" else tuple(verdict.get("outside_names") or ())
case = case.with_applicability(ApplicabilityFacts(
    checked=True,
    requested_condition=", ".join(f"{k}={v:g}" for k, v in query.items()),
    outside_variables=outside,
    verified_ranges=tuple((k, float(lo), float(hi)) for k, (lo, hi) in domain_report["bounds"].items()),
    notes=(f"the evidence domain answers {verdict['status']} at the requested condition ({verdict['reason']})",
           f"{domain_report['evidence_gap']['unsupported']} of {domain_report['evidence_gap']['candidates']} sampled "
           "candidates have no reference within the neighbourhood radius",
           "camber is not a parameter of this model's schema: a query carrying it is outside the model, not merely "
           "outside the evidence")), actor="study")
case = case.with_limitations((
    "the surrogate is a function of three variables; camber is outside its schema and its evidence does not transfer",
    "the evidence covers the sampled design box only, and only where an independent reference sits within the "
    "neighbourhood radius",
    "steady RANS with wall functions: separated or transitional regimes are outside what the reference itself "
    "resolves, so no amount of surrogate accuracy extends the claim there",
    "the campaign is numerical throughout: no experimental data enters this case",
    f"drag is not a validated quantity here: the reference's own grid-convergence index on CD is "
    f"{100 * cd_gci_value:.0f} % at the finest mesh, larger than any difference the surrogate could be judged on",
))

dossier = ValidationDossier.from_case(case)
open(os.path.join(RES, "dossier.md"), "w").write(dossier.render_markdown())
open(os.path.join(RES, "dossier.txt"), "w").write(dossier.render_text())
json.dump(dossier.to_dict(), open(os.path.join(RES, "dossier.json"), "w"), indent=1, default=str)

# --------------------------------------------------------------------------- the manifest
with pinneapple_lab.experiment(
        "airfoil_trust_end_to_end",
        title="Trust beyond accuracy: an evidence-based validation study of a NACA surrogate",
        domain="fluids",
        description=("OpenFOAM campaign on a NACA four-digit family -- a mesh family with a grid-convergence study "
                     "and a design of experiments split by design -- a POD+MLP ensemble surrogate, accuracy and "
                     "physical-consistency validation, an evidence domain that answers where a prediction is "
                     "supported, camber opened as a new parameter dimension with its own campaign, and the dossier "
                     "that records all of it."),
        methods=["OpenFOAM simpleFoam k-omega SST", "gmsh hex meshing", "Richardson extrapolation / GCI",
                 "maximin Latin hypercube", "POD", "MLP ensemble", "evidence domain screening"],
        modules=["pinneapple_autolab.domains.airfoil", "pinneapple_autolab.domains.airfoil_study",
                 "pinneapple_simulation.numerical_solvers.airfoil2d", "pinneapple_veriphysics.validated_domain",
                 "pinneapple_veriphysics.validation_case", "pinneapple_veriphysics.dossier"],
        params={"train": len(idx["train"]), "calibration": len(calib), "validation": len(valid),
                "iterations": 1000, "mesh": "L2"},
        references=["Celik, I.B., Ghia, U., Roache, P.J. (2008). Procedure for estimation and reporting of "
                    "uncertainty due to discretization in CFD applications. J. Fluids Eng. 130, 078001."],
        out_dir=HERE, register=False) as ex:
    for name, value, unit, how in (
            ("CL_rel_l2_held_out", acc["CL_rel_l2"], "1", "relative L2 over designs held out by design"),
            ("CL_rmse_held_out", acc["CL_rmse"], "1", "RMSE over designs held out by design"),
            ("CD_rel_l2_held_out", acc["CD_rel_l2"], "1", "relative L2 over designs held out by design"),
            ("CD_rmse_held_out", acc["CD_rmse"], "1", "RMSE over designs held out by design"),
            ("field_rel_l2_mean", analysis["validation"]["field_rel_l2_mean"], "1",
             "relative L2 of u, v and cp on the lattice, mean over held-out designs"),
            ("basis_energy_captured", analysis["basis"]["energy_captured"], "1",
             f"{analysis['basis']['n_modes']} POD modes on the training fields"),
            ("mesh_change_CL", cl_change, "1",
             "change in CL between the two finest meshes (the family oscillates: no GCI is claimed)"),
            ("gci_CD", cd_gci_value, "1", "grid-convergence index on CD at the finest triplet"),
            ("observed_order_CL", mesh["study"].get("CL", {}).get("p") or float("nan"), "1",
             "observed order of convergence on CL"),
            ("observed_order_CD", mesh["study"].get("CD", {}).get("p") or float("nan"), "1",
             "observed order of convergence on CD"),
            ("wall_speed_ratio", phy["wall_condition"]["surrogate_mean"], "1",
             "mean |U|/U_inf on the wall samples of the surrogate"),
            ("continuity_residual", phy["continuity"]["surrogate_mean"], "1",
             "mean |du/dx + dv/dy| times the chord, on the lattice of each held-out design"),
            ("gate_residual_normalised", phy["gate_residual"]["surrogate_mean"], "1",
             "surrogate_gates continuity residual, normalised by the size of its own gradients"),
            ("gate_boundary_deviation", phy["gate_boundary"]["surrogate_mean"], "1",
             "surrogate_gates deviation from the declared no-slip value on the wall edge"),
            ("gate_accuracy_cp", phy["gate_accuracy"]["surrogate_mean"], "1",
             "surrogate_gates relative L2 of the predicted Cp against the reference"),
            ("input_sensitivity_worst", float(physics["sensitivity_worst"]), "1",
             "movement of CL under a 1 % nudge of an input, over the ensemble spread, worst input"),
            ("uncertainty_coverage_1sigma", coverage_1s, "1",
             "fraction of held-out errors inside one ensemble standard deviation")):
        ex.log_metric(name, value, unit, how)
    categories = {"verification": "verification", "validation": "validation", "physics": "physics",
                  "applicability": "generalization", "uncertainty": "coverage", "robustness": "other",
                  "reproducibility": "other"}
    for v in checks:
        if not v.ran:                       # a check that did not run is not a pass and has no value to log
            continue
        ex.check(v.check_id, passed=v.passed, value=v.value, max=v.threshold, detail=v.message,
                 kind=categories[v.category])
    missing = [v.check_id for v in checks if not v.ran]
    ex.check("declared_checks_all_ran", passed=not missing,
             detail="checks declared but not run: " + (", ".join(missing) or "none"))
    ex.check("dossier_status", passed=dossier.assessment.status in ("VALIDATED", "REVIEW_REQUIRED"),
             detail=f"{case.case_id}: {dossier.assessment.status} ({dossier.assessment.reason})")
    for f in figures:
        ex.add_file(f)
    for name in ("dossier.md", "dossier.txt", "dossier.json", "analysis_v1.json", "physics_v1.json",
                 "domain_v1.json", "mesh_study.json", "doe_round1.json"):
        p = os.path.join(RES, name)
        if os.path.exists(p):
            ex.add_file(p)
    # the capture records the script that ran; this study is seven stages plus an orchestrator, and a manifest that
    # lists only one of them describes a folder it does not know
    ex.manifest.scripts[:] = sorted(f for f in os.listdir(HERE) if f.endswith(".py"))

print(f"dossier {case.case_id}: {dossier.assessment.status} ({dossier.assessment.status_source})")
print(f"  {dossier.assessment.reason}")
print(f"  uncertainty: one sigma covers {coverage_1s:.0%} of held-out errors, two sigma {coverage_2s:.0%} "
      f"-> calibrated={calibrated}")
print(f"  domain at {query}: {verdict['status']} ({verdict['reason']})")
print(f"  robustness: worst input-sensitivity ratio {physics['sensitivity_worst']:.2f} "
      f"(movement over ensemble spread)")
print(f"  figures: {', '.join(os.path.basename(f) for f in figures)}")
for c in checks:
    print(f"  {LEDGER.claim(c.check_id) if c.passed is not None else f'{c.check_id}: not run (no record)'}")
