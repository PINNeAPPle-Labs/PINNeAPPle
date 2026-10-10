"""Applicability map: a structured evidence chain for one Veriphysics run.

Turns validation from pass/fail into an *engineering applicability map*:

* **what physics was directly verified** (tier ``VERIFIED`` -- a check
  actually executed in this run and produced a number),
* **what is only inferred from the model/design** (tier ``INFERRED`` --
  plausible given what ran, but no check measured it directly),
* **where the evidence is too sparse to support extrapolation** (tier
  ``UNSUPPORTED`` -- nothing was checked there; do not use the surrogate
  there without new evidence).

It also answers a fixed eight-item review checklist (surrogate, error,
convergence, generalization, variables, weights, uncertainty
quantification, conservation/standard physics).

Like ``decision.py`` and ``evidence_report.py`` this module never runs a
check and never invents a number: it only *classifies and renders* facts
the pipeline already produced. A check that did not run is ``NOT_RUN``,
never a neutral pass. It deliberately imports nothing from
``pinneapple_*`` so it is cheap to test and reproducible from a saved
``decision_record.json``.

(Extracted from Veriphysics' orchestrator; Apache-2.0 like the rest of PINNeAPPle. Execution queue, API and billing stay in Veriphysics.)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Status of one checklist item.
PASS, FAIL, NOT_RUN, INFO = "PASS", "FAIL", "NOT_RUN", "INFO"
# Evidence tier of one claim.
VERIFIED, INFERRED, UNSUPPORTED = "VERIFIED", "INFERRED", "UNSUPPORTED"

# Heuristic, documented here rather than hidden: training is called
# "plateaued" when the loss over the last 10% of epochs moved by less
# than this relative amount. This is a reviewer aid, not a guarantee.
PLATEAU_REL_TOL = 0.05
# Mirrors robustness.SPREAD_REL_TOL (not imported: this module stays dependency-free).
SPREAD_TOL = 0.10
# Calibration tolerances (heuristic reviewer aids): expected calibration error and
# |empirical - target| interval coverage.
ECE_TOL = 0.10
COVERAGE_TOL = 0.05


@dataclass
class ChecklistItem:
    key: str
    question: str
    status: str
    tier: str
    summary: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class VariableEnvelope:
    name: str
    kind: str  # "coordinate" | "parameter" | "field"
    verified_range: Optional[Tuple[float, float]]  # None for a single point / unknown
    verified_point: Optional[float]
    extrapolation_supported: bool
    note: str


@dataclass
class ApplicabilityMap:
    checklist: List[ChecklistItem]
    envelope: List[VariableEnvelope]
    verified: List[str]
    inferred: List[str]
    unsupported: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = ["Applicability map (evidence chain)", "", "Review checklist:"]
        for item in self.checklist:
            lines.append(f"  [{item.status}] {item.question} ({item.tier.lower()})")
            lines.append(f"      {item.summary}")
        for title, rows in (
            ("Directly verified", self.verified),
            ("Inferred from the model (not directly checked)", self.inferred),
            ("Unsupported -- do not extrapolate here", self.unsupported),
        ):
            lines += ["", f"{title}:"]
            lines += [f"  - {r}" for r in rows] or ["  - (none)"]
        lines += ["", "Variable envelope:"]
        for v in self.envelope:
            where = (
                f"verified at {v.verified_point:g}" if v.verified_point is not None
                else f"verified on [{v.verified_range[0]:g}, {v.verified_range[1]:g}]" if v.verified_range
                else "no verified range recorded"
            )
            lines.append(f"  - {v.name} ({v.kind}): {where}; extrapolation "
                         f"{'supported' if v.extrapolation_supported else 'NOT supported'}. {v.note}")
        return "\n".join(lines)


def _tail_plateau(history: Sequence[float]) -> Optional[float]:
    """Relative change between the mean loss of the last 10% of epochs and
    the mean of the preceding 10%, or None. Window means, not endpoints:
    collocation points are resampled every step, so a single epoch's loss
    is noisy."""
    n = len(history)
    if n < 20:
        return None
    k = max(n // 10, 1)
    last = sum(history[-k:]) / k
    prev = sum(history[-2 * k:-k]) / k
    return abs(last - prev) / (abs(prev) or 1e-30)


def _by_prefix(checks: Sequence[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    return [c for c in checks if str(c.get("name", "")).startswith(prefix)]


def _check_line(c: Dict[str, Any]) -> str:
    return f"{c['name']}: {'passed' if c.get('passed') else 'FAILED'} -- {c.get('detail', '')}"


def build_applicability_map(
    *,
    architecture: str,
    coords: Sequence[str],
    fields: Sequence[str],
    domain_bounds: Dict[str, Tuple[float, float]],
    physical_parameters: Optional[Dict[str, float]],
    guardrail_checks: Sequence[Dict[str, Any]],
    confidence_components: Sequence[Dict[str, Any]],
    final_loss: Optional[float] = None,
    loss_history: Optional[Sequence[float]] = None,
    epochs: Optional[int] = None,
    n_collocation: Optional[int] = None,
    n_parameters: Optional[int] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    convergence: Optional[Dict[str, Any]] = None,
    heldout_error: Optional[Dict[str, Any]] = None,
    extrapolation: Optional[Dict[str, Any]] = None,
    ensemble: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    parameter_shift: Optional[Dict[str, Any]] = None,
    weight_sensitivity: Optional[Dict[str, Any]] = None,
    condition_coverage: Optional[Dict[str, Any]] = None,
) -> ApplicabilityMap:
    """Classify the facts of one run. ``heldout_error`` (optional) is
    ``{"metric": str, "value": float, "threshold": float, "description": str}``
    from an out-of-sample test the caller ran. ``extrapolation``,
    ``ensemble`` and ``weight_sensitivity`` are the dicts returned by
    ``robustness.extrapolation_study`` / ``ensemble_study`` (minus the
    ``members`` models) / ``weight_sensitivity_study``. Without any of
    them the corresponding item is ``NOT_RUN`` -- the guardrail residual
    is evaluated on fresh points but *inside the training domain*, which
    is not generalization."""
    comps = {c["name"]: c for c in confidence_components}
    checklist: List[ChecklistItem] = []
    verified: List[str] = []
    inferred: List[str] = []
    unsupported: List[str] = []

    # 0. Well-posedness: were the boundary/initial conditions ever enforced? -----
    if condition_coverage is not None:
        bad = condition_coverage["unenforced"]
        n = condition_coverage["n_samples"]
        if bad:
            checklist.append(ChecklistItem(
                "conditions_enforced", "Were boundary/initial conditions actually enforced in training?", FAIL, VERIFIED,
                f"Condition(s) {bad} select 0 of {n} sampled points, so training silently skipped them: the "
                "problem was trained under-constrained and any 'solution' is not unique. Fix the selector "
                "(tolerance band or explicit boundary points) before trusting any result.",
                [f"{k}: {v} points" for k, v in condition_coverage["selected_points"].items()]))
            unsupported.append(f"Everything: condition(s) {bad} were never enforced (0 sampled points).")
        else:
            checklist.append(ChecklistItem(
                "conditions_enforced", "Were boundary/initial conditions actually enforced in training?", PASS, VERIFIED,
                "Every auto-sampled condition selects points under the training sampler (tag conditions need real geometry and are not counted).",
                [f"{k}: {v} points" if v is not None else f"{k}: tag (explicit geometry)"
                 for k, v in condition_coverage["selected_points"].items()]))

    # 1. Surrogate -----------------------------------------------------
    bits = [f"architecture={architecture}", f"inputs={list(coords)}", f"outputs={list(fields)}"]
    if n_parameters is not None:
        bits.append(f"{n_parameters} trainable parameters")
    if epochs is not None and n_collocation is not None:
        bits.append(f"trained {epochs} epochs on {n_collocation} collocation points")
    checklist.append(ChecklistItem(
        "surrogate", "What surrogate is this?", INFO, VERIFIED, "; ".join(bits)))

    # 2. Error ---------------------------------------------------------
    resid = next((c for c in guardrail_checks if c.get("name") == "pde_residual"), None)
    ref = next((c for c in guardrail_checks if c.get("name") == "reference_data_match"), None)
    err_ev: List[str] = []
    err_status, err_tier = NOT_RUN, UNSUPPORTED
    if resid is not None:
        err_ev.append(_check_line(resid))
        err_status = PASS if resid.get("passed") else FAIL
        err_tier = VERIFIED
        (verified if resid.get("passed") else unsupported).append(
            f"Governing-equation residual re-computed on fresh points inside the trained domain: {resid.get('detail', '')}")
    if ref is not None:
        err_ev.append(_check_line(ref))
        if not ref.get("passed"):
            err_status = FAIL
        verified.append(f"Agreement with reference data: {ref.get('detail', '')}")
    else:
        inferred.append("Error against ground truth is NOT measured (no reference data supplied); "
                        "only the equation residual is.")
    if final_loss is not None:
        err_ev.append(f"final training loss = {final_loss:.4g} (training-side number, not independent evidence)")
    checklist.append(ChecklistItem(
        "error", "What is the error?", err_status, err_tier,
        "Residual and/or reference mismatch below." if err_ev else "No error check ran.", err_ev))

    # 3. Convergence ---------------------------------------------------
    conv_ev: List[str] = []
    conv_status, conv_tier = NOT_RUN, UNSUPPORTED
    plateau = _tail_plateau(loss_history) if loss_history else None
    if plateau is not None:
        plateaued = plateau < PLATEAU_REL_TOL
        conv_ev.append(f"training loss (window means) moved {plateau:.2%} between the last two 10% epoch windows "
                       f"({'plateaued' if plateaued else 'STILL DECREASING'}; tolerance {PLATEAU_REL_TOL:.0%}, heuristic)")
        conv_status, conv_tier = (PASS if plateaued else FAIL), VERIFIED
    if convergence:
        asym = convergence.get("is_asymptotic")
        conv_ev.append(f"numerical convergence study: observed_order={convergence.get('observed_order')}, "
                       f"GCI_fine={convergence.get('gci_fine')}, is_asymptotic={asym}")
        if asym is False:
            conv_status = FAIL
        elif conv_status == NOT_RUN:
            conv_status = PASS
        conv_tier = VERIFIED
        verified.append("Residual-evaluation refinement study (Richardson/GCI) on the trained model.")
    else:
        inferred.append("Numerical (collocation-density) convergence not demonstrated for this run.")
    checklist.append(ChecklistItem(
        "convergence", "Did it converge?", conv_status, conv_tier,
        "Optimizer and numerical convergence evidence below." if conv_ev else "No convergence evidence.", conv_ev))

    # 4. Generalization -----------------------------------------------
    margin_m = None
    _param_limits: Dict[str, float] = {}
    if heldout_error:
        ok = heldout_error["value"] <= heldout_error["threshold"]
        checklist.append(ChecklistItem(
            "generalization", "Can it generalize?", PASS if ok else FAIL, VERIFIED,
            f"{heldout_error.get('metric', 'held-out error')}={heldout_error['value']:.4g} "
            f"(threshold {heldout_error['threshold']:.4g})",
            [heldout_error.get("description", "")]))
        verified.append(f"Held-out test: {heldout_error.get('description', heldout_error.get('metric', ''))}")
    elif extrapolation or parameter_shift:
        ev: List[str] = []
        parts: List[str] = []
        any_supported = False
        if extrapolation:
            margin_m = extrapolation["supported_margin"]
            for sh in extrapolation["shells"]:
                sp = f", ensemble spread={sh['rel_spread']:.1%}" if sh.get("rel_spread") is not None else ""
                ev.append(f"coordinates +{sh['margin']:.0%} outside the trained box: residual={sh['residual']:.3g} "
                          f"(threshold {extrapolation['residual_threshold']:.3g}){sp} -> {'ok' if sh['ok'] else 'NOT ok'}")
            any_supported = any_supported or margin_m > 0
            parts.append(f"coordinates up to +{margin_m:.0%} beyond the trained range" if margin_m > 0
                         else "coordinates: fails already in the first shell outside the trained range")
            if margin_m > 0:
                verified.append(f"Equation residual stays acceptable up to +{margin_m:.0%} outside the trained range.")
                unsupported.append(f"Coordinates beyond +{margin_m:.0%} outside the trained range.")
            else:
                unsupported.append("Any coordinate outside the trained range (fails in the first shell).")
        param_limits: Dict[str, float] = {}
        if parameter_shift:
            for pr in parameter_shift["parameters"]:
                param_limits[pr["parameter"]] = pr["supported_shift"]
                for r in pr["shifts"]:
                    why = ("" if r["reference_trusted"] else
                           f" (reference model itself failed the residual check: {r['reference_residual']:.3g})")
                    ev.append(f"{pr['parameter']} {r['shift']:+.0%} ({r['value']:.4g}): surrogate differs from a model "
                              f"retrained there by {r['rel_prediction_error']:.1%} of field scale -> "
                              f"{'ok' if r['ok'] else 'NOT ok'}{why}")
                any_supported = any_supported or pr["supported_shift"] > 0
                parts.append(f"{pr['parameter']} within +{pr['supported_shift']:.0%} of {pr['base_value']:g}"
                             if pr["supported_shift"] > 0 else
                             f"{pr['parameter']}: not usable even at +{parameter_shift['parameters'][0]['shifts'][0]['shift']:.0%}")
                if pr["supported_shift"] > 0:
                    verified.append(f"Surrogate matches a retrained model for {pr['parameter']} up to "
                                    f"+{pr['supported_shift']:.0%} of its base value.")
                unsupported.append(f"{pr['parameter']} beyond +{pr['supported_shift']:.0%} of {pr['base_value']:g} "
                                   "(upward shifts only were tested).")
        checklist.append(ChecklistItem(
            "generalization", "Can it generalize?", PASS if any_supported else FAIL, VERIFIED,
            "Measured limits: " + "; ".join(parts) + ". Geometry changes were NOT tested. A small residual outside "
            "the domain shows the PDE is satisfied there, not that boundary-driven truth is matched; parameter "
            "tests compare against a retrained model, trusted only if its own residual passes.", ev))
        unsupported.append("Geometry changes (no geometry-variation test ran).")
        if not parameter_shift:
            unsupported.append("Other parameters or regimes (no parameter-shift test ran).")
        _param_limits = param_limits
    else:
        checklist.append(ChecklistItem(
            "generalization", "Can it generalize?", NOT_RUN, UNSUPPORTED,
            "No out-of-sample test (held-out geometry/parameters/regime or extrapolation shells) was run. "
            "The residual check samples fresh points but only inside the training domain, so it does not "
            "establish generalization.",
        ))
        unsupported.append("Any parameter, geometry or regime not explicitly tested (no held-out test ran).")

    # 5. Variables -----------------------------------------------------
    envelope: List[VariableEnvelope] = []
    for c in coords:
        b = domain_bounds.get(c)
        envelope.append(VariableEnvelope(
            c, "coordinate", (float(b[0]), float(b[1])) if b else None, None, bool(margin_m),
            (f"Residual sampled uniformly over this range; measured to hold up to +{margin_m:.0%} beyond it."
             if margin_m else "Residual sampled uniformly over this range; outside it nothing was checked.")
            if b else "Training range not recorded."))
        if b:
            verified.append(f"Equation residual sampled over {c} in [{b[0]:g}, {b[1]:g}].")
            if not extrapolation:
                unsupported.append(f"{c} outside [{b[0]:g}, {b[1]:g}].")
    for name, val in (physical_parameters or {}).items():
        envelope.append(VariableEnvelope(
            name, "parameter", None, float(val), bool(_param_limits.get(name)),
            (f"Surrogate measured usable up to +{_param_limits[name]:.0%} above this value (upward shifts only)."
             if _param_limits.get(name) else "Single operating point; shifting it was not shown to be safe.")))
        if not parameter_shift:
            unsupported.append(f"{name} other than {val:g} (single operating point).")
    for name, lim in _param_limits.items():
        if name not in (physical_parameters or {}):
            envelope.append(VariableEnvelope(
                name, "parameter", None, None, bool(lim),
                f"Surrogate measured usable up to +{lim:.0%} above its base value (upward shifts only)."
                if lim else "Shifting it was not shown to be safe."))
    for f in fields:
        envelope.append(VariableEnvelope(f, "field", None, None, False, "Predicted output; no range claim made."))
    checklist.append(ChecklistItem(
        "variables", "Which variables, and over what range?", INFO, VERIFIED,
        f"{len(coords)} coordinate(s), {len(physical_parameters or {})} fixed parameter(s), {len(fields)} field(s); "
        "see the variable envelope."))

    # 6. Weights -------------------------------------------------------
    if weight_sensitivity:
        mx, tol = weight_sensitivity["max_rel_change"], weight_sensitivity["tolerance"]
        ev = [f"w_pde x{v['w_pde_factor']:g}: prediction moved {v['rel_prediction_change']:.1%} of field scale, "
              f"residual={v['residual']:.3g}" for v in weight_sensitivity["variants"]]
        ok = mx <= tol
        checklist.append(ChecklistItem(
            "weights", "Which loss weights, and does the result depend on them?", PASS if ok else FAIL, VERIFIED,
            f"Retrained with the PDE weight scaled; max prediction change {mx:.1%} (tolerance {tol:.0%}, heuristic; "
            "includes seed noise). " + ("Result is insensitive to PDE weighting." if ok
                                         else "Result DEPENDS on loss weighting -- treat as a modelling choice."),
            ev))
        (verified if ok else unsupported).append(
            f"Sensitivity to PDE loss weighting: max change {mx:.1%} of field scale across weight x"
            + "/x".join(f"{v['w_pde_factor']:g}" for v in weight_sensitivity["variants"]) + ".")
    elif loss_weights:
        w = ", ".join(f"{k}={v:g}" for k, v in loss_weights.items())
        checklist.append(ChecklistItem(
            "weights", "Which loss weights?", INFO, VERIFIED,
            f"Explicit loss weights: {w}. Sensitivity was not swept; the verification residual is "
            "re-computed independently of these weights."))
    else:
        checklist.append(ChecklistItem(
            "weights", "Which loss weights?", INFO, INFERRED,
            "Solver defaults (not recorded per term). Sensitivity not swept; the verification residual is "
            "re-computed independently of the training weights, so ranking cannot be gamed by them."))
        inferred.append("Result sensitivity to PDE/BC/IC loss weighting (defaults used, no sweep).")

    # 7. UQ ------------------------------------------------------------
    uq = comps.get("uq_calibration")
    if uq is not None:
        if calibration:
            ece, cov, tgt = calibration["ece"], calibration["coverage"], calibration["target_coverage"]
            ok = ece <= ECE_TOL and abs(cov - tgt) <= COVERAGE_TOL
            kind = ("well calibrated" if ok else
                    "OVER-conservative (intervals wider than needed)" if cov > tgt else "OVERCONFIDENT (intervals too narrow)")
            checklist.append(ChecklistItem(
                "uq", "Uncertainty quantification?", PASS if ok else FAIL, VERIFIED,
                f"Calibrated against reference data: ECE={ece:.3f} (tolerance {ECE_TOL}), empirical coverage "
                f"{cov:.2f} vs target {tgt:.2f} (tolerance +/-{COVERAGE_TOL}) -> {kind}. Heuristic tolerances.",
                [uq["source_summary"]]))
            (verified if ok else unsupported).append(f"UQ calibration: {kind}; {uq['source_summary']}")
        else:
            checklist.append(ChecklistItem(
                "uq", "Uncertainty quantification?", INFO, VERIFIED,
                f"Calibration score {uq['score']:.2f} -- {uq['source_summary']}", [uq["source_summary"]]))
            verified.append(f"UQ calibration: {uq['source_summary']}")
    elif ensemble:
        sp = ensemble["interior_rel_spread"]
        ok = sp <= SPREAD_TOL
        checklist.append(ChecklistItem(
            "uq", "Uncertainty quantification?", PASS if ok else FAIL, VERIFIED,
            f"Epistemic spread from {ensemble['n_members']} independently trained members: {sp:.1%} of field "
            f"scale inside the domain (tolerance {SPREAD_TOL:.0%}, heuristic). UNCALIBRATED: no reference data "
            "was supplied, so these are disagreement bars, not calibrated confidence intervals.",
            [f"+{m:.0%} outside: spread {v:.1%}" for m, v in ensemble["shell_rel_spread"].items()]))
        (verified if ok else unsupported).append(
            f"Ensemble disagreement inside the domain = {sp:.1%} of field scale (uncalibrated).")
        inferred.append("Calibrated coverage of the uncertainty bars (needs reference data).")
    else:
        checklist.append(ChecklistItem(
            "uq", "Uncertainty quantification?", NOT_RUN, UNSUPPORTED,
            "No calibrated uncertainty for this run: predictions are point estimates without error bars."))
        unsupported.append("Prediction confidence intervals (UQ not run).")

    # 8. Physical laws -------------------------------------------------
    cons = _by_prefix(guardrail_checks, "conservation")
    dim = next((c for c in guardrail_checks if c.get("name") == "dimensional_analysis"), None)
    if dim:
        # Units consistency is real evidence, but it is not a conservation law.
        (verified if dim.get("passed") else unsupported).append(_check_line(dim))
    if cons:
        failed = any(not c.get("passed") for c in cons)
        checklist.append(ChecklistItem(
            "physics_laws", "Does it obey standard physics (e.g. mass conservation)?",
            FAIL if failed else PASS, VERIFIED, "Explicit conservation check(s) ran.",
            [_check_line(c) for c in cons] + ([_check_line(dim)] if dim else [])))
        for c in cons:
            (unsupported if not c.get("passed") else verified).append(_check_line(c))
    else:
        checklist.append(ChecklistItem(
            "physics_laws", "Does it obey standard physics (e.g. mass conservation)?", NOT_RUN, INFERRED,
            "No explicit conservation check ran. A passing PDE residual implies the governing equations are "
            "satisfied, but global conservation (mass/energy balance) was not measured."
            + (" Dimensional consistency did run." if dim else ""),
            [_check_line(dim)] if dim else []))
        inferred.append("Global conservation (mass/energy) follows from the PDE residual but was not measured directly.")

    return ApplicabilityMap(checklist, envelope, verified, inferred, unsupported)
