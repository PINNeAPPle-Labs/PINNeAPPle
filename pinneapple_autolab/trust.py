"""Can this prediction be used for a decision? Triage of surrogate candidates and the choice of the next simulation.

Every candidate design the optimiser proposes gets a :class:`CandidateAssessment`:

* its predicted objectives and constraints with their std (from the chosen surrogate),
* the probability that it meets every constraint, ``P(feasible) = prod Phi(margin_i / std_i)``,
* its expected improvement over the best *simulated* feasible design, weighted by ``P(feasible)`` (constrained EI),
* whether it is inside the validated domain (:mod:`.domain`), and the error the field surrogate expects in the
  region that decides the constraint (when there is one),
* a :class:`~pinneapple_analysis.trust.TrustReport` built from those checks, and one of four labels:

  - ``"favourable"``: inside the domain, constraints met with >= ``p_ok`` probability, uncertainty within the
    acceptance tolerance -> worth a confirming simulation;
  - ``"uncertain"``: inside the domain, but the prediction cannot settle the decision (P(feasible) between
    ``p_drop`` and ``p_ok``, an objective std beyond tolerance, or a large expected field error in the critical
    region) -> a simulation there teaches the model where the decision changes;
  - ``"out_of_domain"``: the model is extrapolating; its numbers there are not evidence -> simulate before
    believing it;
  - ``"unlikely"``: P(feasible) below ``p_drop``; not pursued.

  After a reference run the label becomes ``"verified"``, ``"rejected by reference"`` or ``"reference failed"``.

:func:`choose_next` picks the next simulation by what the lab still needs to learn, in this order (a written rule,
not a learned one): (1) the candidate with the best predicted objective if it is out of domain (the optimum the
model believes in has no evidence); (2) the uncertain candidate with the highest constrained EI, if its EI is at
least that of the best favourable one (resolving it can change the decision); (3) the favourable candidate with the
highest constrained EI (exploit). Each choice comes with its reason and the alternatives it beat.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinneapple_analysis.trust.trust_report import (
    CONTRADICTS,
    NOT_RUN,
    SUPPORTS,
    Check,
    TrustReport,
)
from pinneapple_design.design_optimizer.study import Constraint, Objective

from .domain import DomainVerdict

__all__ = ["CandidateAssessment", "assess_candidates", "choose_next", "NextExperiment", "LABELS",
           "verification_checks"]

LABELS = ("favourable", "uncertain", "out_of_domain", "unlikely", "verified", "rejected by reference",
          "reference failed")


def _phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(z, float) / math.sqrt(2)))


def _pdf(z):
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / math.sqrt(2 * math.pi)


@dataclass
class CandidateAssessment:
    index: int
    params: dict[str, float]
    predicted: dict[str, float]
    std: dict[str, float]
    p_feasible: float
    cei: float
    domain: DomainVerdict
    label: str
    reasons: list[str]
    report: TrustReport
    critical_field_std: float | None = None
    reference: dict[str, float] = field(default_factory=dict)
    rel_error: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "params": self.params, "predicted": self.predicted, "std": self.std,
                "p_feasible": self.p_feasible, "cei": self.cei, "label": self.label, "reasons": self.reasons,
                "domain": {"inside": self.domain.inside, "ratio": self.domain.ratio, "reasons": self.domain.reasons},
                "critical_field_std": self.critical_field_std, "trust_decision": self.report.decisao,
                "trust_score": self.report.score,
                "checks": [{"name": c.name, "status": c.status, "critical": c.critical, "value": c.value,
                            "limit": c.limit, "reason": c.reason} for c in self.report.checks],
                "reference": self.reference, "rel_error": self.rel_error}


def _p_feasible(pred: Mapping[str, tuple[np.ndarray, np.ndarray]], cons: Sequence[Constraint]) -> np.ndarray:
    n = len(next(iter(pred.values()))[0])
    p = np.ones(n)
    for c in cons:
        m, s = pred[c.name]
        s = np.maximum(s, 1e-12)
        if c.lower is not None:
            p = p * _phi((m - c.lower) / s)
        if c.upper is not None:
            p = p * _phi((c.upper - m) / s)
    return p


def _ei(m: np.ndarray, s: np.ndarray, best: float, sense: str) -> np.ndarray:
    s = np.maximum(s, 1e-12)
    imp = (best - m) if sense == "min" else (m - best)
    z = imp / s
    return imp * _phi(z) + s * _pdf(z)


def assess_candidates(X: np.ndarray, names: Sequence[str], pred: Mapping[str, tuple[np.ndarray, np.ndarray]],
                      objectives: Sequence[Objective], constraints: Sequence[Constraint],
                      domain_verdicts: Sequence[DomainVerdict], best_value: float | None,
                      acceptance: Mapping[str, float], model_errors: Mapping[str, float] | None = None,
                      critical_field_std: np.ndarray | None = None, field_tol: float | None = None,
                      p_ok: float = 0.9, p_drop: float = 0.05) -> list[CandidateAssessment]:
    """Assess candidates ``X`` (physical units, columns ``names``). ``pred``: output -> (mean, std) arrays;
    ``best_value``: the best simulated feasible value of the first objective (None: no feasible simulation yet,
    then the EI is replaced by the predicted improvement over the candidates' worst); ``model_errors``: the
    cross-validated relative error of the chosen model per output; ``critical_field_std``: per candidate, the RMS
    predicted std of the field in the region that decides the constraint, judged against ``field_tol``."""
    obj = objectives[0]
    m0, s0 = pred[obj.name]
    pf = _p_feasible(pred, constraints)
    ref = best_value if best_value is not None else (np.max(m0) if obj.sense == "min" else np.min(m0))
    cei = _ei(m0, s0, float(ref), obj.sense) * pf
    out = []
    for i, x in enumerate(np.atleast_2d(X)):
        dv = domain_verdicts[i]
        p = float(pf[i])
        checks: list[Check] = []
        reasons: list[str] = []
        checks.append(Check("inside validated domain", SUPPORTS if dv.inside else CONTRADICTS, critical=True,
                            value=dv.ratio, reason="; ".join(dv.reasons) or "within the simulated designs"))
        checks.append(Check("constraints met (probability)", SUPPORTS if p >= p_ok else CONTRADICTS, value=p,
                            limit=p_ok, reason=f"P(feasible) = {p:.2f}"))
        unc_bad = []
        for o in [*objectives, *constraints]:
            tol = acceptance.get(o.name)
            if tol is None:
                continue
            m, s = float(pred[o.name][0][i]), float(pred[o.name][1][i])
            rel = 1.96 * s / max(abs(m), 1e-12)
            if rel > tol:
                unc_bad.append(f"{o.name}: 95 % interval +-{100 * rel:.1f} % > {100 * tol:.1f} %")
            checks.append(Check(f"{o.name} uncertainty within acceptance", SUPPORTS if rel <= tol else CONTRADICTS,
                                value=rel, limit=tol, reason=f"+-{100 * rel:.1f} % (95 %)"))
        for k, e in (model_errors or {}).items():
            tol = acceptance.get(k)
            if tol is not None:
                checks.append(Check(f"{k} cross-validated error within acceptance",
                                    SUPPORTS if e <= tol else CONTRADICTS, value=e, limit=tol,
                                    reason=f"held-out mean relative error {100 * e:.1f} %"))
        cfs = None
        if critical_field_std is not None and field_tol is not None:
            cfs = float(critical_field_std[i])
            checks.append(Check("field error in the critical region", SUPPORTS if cfs <= field_tol else CONTRADICTS,
                                value=cfs, limit=field_tol, reason=f"predicted Cp std {cfs:.3f} (limit {field_tol:g})"))
        checks.append(Check("confirmed by the reference solver", NOT_RUN, reason="not simulated yet"))
        rep = TrustReport(checks, min_coverage=0.5, min_score=0.8, model="surrogate")
        if not dv.inside:
            label = "out_of_domain"
            reasons = ["the model is extrapolating: " + "; ".join(dv.reasons)]
        elif p < p_drop:
            label = "unlikely"
            reasons = [f"P(feasible) = {p:.2f} < {p_drop:g}"]
        elif p < p_ok or unc_bad or (cfs is not None and cfs > field_tol):
            label = "uncertain"
            if p < p_ok:
                reasons.append(f"P(feasible) = {p:.2f}: the prediction cannot tell whether the constraints hold")
            reasons += unc_bad
            if cfs is not None and cfs > field_tol:
                reasons.append(f"expected Cp error in the critical region {cfs:.3f} > {field_tol:g}")
        else:
            label = "favourable"
            reasons = [f"inside the validated domain (spacing ratio {dv.ratio:.2f}), P(feasible) = {p:.2f}, "
                       "uncertainty within the acceptance tolerances"]
        out.append(CandidateAssessment(i, {k: float(v) for k, v in zip(names, x, strict=True)},
                                       {k: float(v[0][i]) for k, v in pred.items()},
                                       {k: float(v[1][i]) for k, v in pred.items()}, p, float(cei[i]), dv, label,
                                       reasons, rep, cfs))
    return out


@dataclass
class NextExperiment:
    candidate: CandidateAssessment | None
    rule: str
    reason: str
    alternatives: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "reason": self.reason, "alternatives": self.alternatives,
                "candidate": None if self.candidate is None else self.candidate.to_dict()}


def choose_next(cands: Sequence[CandidateAssessment], objective: Objective,
                exclude: Sequence[int] = ()) -> NextExperiment:
    """The next simulation, by the written rule in the module docstring."""
    pool = [c for c in cands if c.index not in set(exclude) and c.label != "unlikely"]
    if not pool:
        return NextExperiment(None, "none", "no candidate left that could meet the constraints", [])
    sgn = 1 if objective.sense == "min" else -1
    key = objective.name

    def short(c):
        return {"index": c.index, "label": c.label, key: c.predicted[key], "std": c.std[key],
                "p_feasible": round(c.p_feasible, 3), "cei": c.cei}
    plausible = [c for c in pool if c.p_feasible >= 0.5]
    best_pred = min(plausible or pool, key=lambda c: sgn * c.predicted[key])
    fav = sorted([c for c in pool if c.label == "favourable"], key=lambda c: -c.cei)
    unc = sorted([c for c in pool if c.label == "uncertain"], key=lambda c: -c.cei)
    alts = [short(c) for c in (fav[:2] + unc[:2])]
    if best_pred.label == "out_of_domain":
        return NextExperiment(best_pred, "best prediction is out of domain",
                              f"the model's best design ({key} {best_pred.predicted[key]:.4g} "
                              f"+- {best_pred.std[key]:.2g}) lies outside the validated domain "
                              f"({'; '.join(best_pred.domain.reasons)}): its value is not evidence until simulated",
                              alts)
    if unc and (not fav or unc[0].cei >= fav[0].cei):
        c = unc[0]
        return NextExperiment(c, "resolve uncertainty where the decision changes",
                              f"highest constrained expected improvement ({c.cei:.4g}) is an uncertain design: "
                              + "; ".join(c.reasons), alts)
    if fav:
        c = fav[0]
        return NextExperiment(c, "confirm the most promising favourable design",
                              f"favourable design with the highest constrained expected improvement ({c.cei:.4g}); "
                              + "; ".join(c.reasons), alts)
    c = max(pool, key=lambda c: c.cei)
    return NextExperiment(c, "explore", f"no favourable or uncertain design; highest constrained EI ({c.cei:.4g})",
                          alts)


def verification_checks(c: CandidateAssessment, reference: Mapping[str, float], ok: bool,
                        acceptance: Mapping[str, float], constraints: Sequence[Constraint],
                        message: str = "") -> CandidateAssessment:
    """Update an assessment with the reference result: measured relative errors, the confirmation check and the
    feasibility under the reference values."""
    checks = [ch for ch in c.report.checks if ch.name != "confirmed by the reference solver"]
    if not ok:
        checks.append(Check("confirmed by the reference solver", CONTRADICTS, critical=True,
                            reason=f"reference failed: {message}"))
        c.label = "reference failed"
        c.reasons = [f"reference failed: {message}"]
    else:
        c.reference = dict(reference)
        c.rel_error = {k: abs(c.predicted[k] - v) / max(abs(v), 1e-12) for k, v in reference.items() if k in c.predicted}
        bad = [k for k, e in c.rel_error.items() if k in acceptance and e > acceptance[k]]
        checks.append(Check("confirmed by the reference solver", CONTRADICTS if bad else SUPPORTS,
                            reason=("prediction off by " + ", ".join(f"{k} {100 * c.rel_error[k]:.1f} %" for k in bad))
                            if bad else "every prediction within its acceptance tolerance"))
        viol = [cn.describe() for cn in constraints if cn.name in reference and cn.violation(reference[cn.name]) > 0]
        checks.append(Check("constraints met (reference)", CONTRADICTS if viol else SUPPORTS, critical=True,
                            reason=("violates " + "; ".join(viol)) if viol else "all constraints met"))
        c.label = "rejected by reference" if viol else "verified"
        c.reasons = (["the reference shows it " + "; ".join(f"violates {v}" for v in viol)] if viol else
                     ["the reference confirms the constraints"]) + ([f"prediction error beyond tolerance on {bad}"]
                                                                    if bad else [])
    c.report = TrustReport(checks, min_coverage=0.5, min_score=0.8, model="surrogate + reference")
    return c
