"""The vocabulary of a validation report, in one place: statuses, credibility factors and levels, and the stable check
IDs used by ``pp.validate`` (``pinneapple_analysis.validation.validate``), the trust card and ``pinneapple check``.

Other modules map their own results onto these IDs; new checks add an entry here (the report refuses unknown IDs).

Statuses
    ``pass``, ``warn``, ``fail``, ``not-applicable``. A check that cannot apply (no reference, no PDE stated, a
    tabular model asked for a spectrum) is reported as ``not-applicable`` with the reason, never silently skipped.

Credibility factors
    Modelled on NASA-STD-7009A (credibility assessment) and ASME V&V 40 (credibility factors): the evidence for a
    model is grouped by what it supports. The levels are this library's own reading of those standards, not a
    certification under them.

Credibility levels (per factor)
    0 no evidence (every check not applicable), 1 evidence contradicts the model (a critical check failed), 2 passes
    with warnings (a non-critical check failed or warned) or only qualitative evidence, 3 passes quantitative checks,
    4 passes against an independent reference or a guaranteed bound.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PASS", "WARN", "FAIL", "NA", "STATUSES", "FACTORS", "LEVELS", "CheckSpec", "CHECKS", "spec_of",
           "DECISIONS"]

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "not-applicable"
STATUSES = (PASS, WARN, FAIL, NA)

#: Decision of the whole report -> the trust-card decision it corresponds to.
DECISIONS = {PASS: "APPROVED", WARN: "REVIEW", FAIL: "REJECT"}

FACTORS: dict[str, str] = {
    "code_verification": "the method and the reference solvers are verified (catalog status, manufactured solutions)",
    "solution_verification": "the predictions satisfy the discrete equations: residual, boundary conditions, balances",
    "validation": "the predictions agree with a reference: held-out data, the model card, physical relations",
    "input_pedigree": "the training data solve their PDE, are diverse and do not leak into the test set",
    "uncertainty": "a stated error bound or spread, with measured coverage",
    "robustness": "the predictions are stable under perturbations and injected faults",
    "applicability": "the queries are inside the range the model was trained and validated on",
}

LEVELS: dict[int, str] = {
    0: "no evidence",
    1: "evidence contradicts the model (a critical check failed)",
    2: "passes with warnings or qualitative evidence only",
    3: "passes quantitative checks",
    4: "passes against an independent reference or a guaranteed bound",
}


@dataclass(frozen=True)
class CheckSpec:
    id: str
    factor: str
    title: str
    rule: str = ""              # the `pinneapple check` rule its outcome is reported under ("" when none)
    strength: int = 3           # credibility level the check supports when it passes
    critical: bool = True       # a failure rejects the model (else it asks for review)
    better: str = "lower"       # direction of the check's value: "lower" (an error) or "higher" (a coverage, a
    #                             spread ratio, a crossing time); training monitors compare in this direction


_SPECS = [
    CheckSpec("method.status", "code_verification", "method validated in the catalog", "PC901", 3, False),
    CheckSpec("residual.energy_bound", "solution_verification",
              "guaranteed bound on the error of the discrete solution, from the residual", "PC302", 4),
    CheckSpec("residual.pde", "solution_verification", "PDE residual of the predictions", "PC302", 3),
    CheckSpec("boundary.values", "solution_verification", "boundary conditions satisfied", "PC201", 3),
    CheckSpec("conservation.flux", "solution_verification", "net boundary flux balances the source", "PC101", 3,
              False),
    CheckSpec("physics.model_checks", "solution_verification", "physics checks shipped with the model", "PC302", 3),
    CheckSpec("reference.rel_l2", "validation", "relative L2 error against the reference", "PC701", 4),
    CheckSpec("reference.spectral", "validation", "small-scale energy against the reference", "PC704", 3, False),
    CheckSpec("reference.card_metrics", "validation", "card metrics reproduced on the stored reference", "PC701", 4),
    CheckSpec("reference.acceptance", "validation", "held-out errors within the acceptance limits", "PC701", 4),
    CheckSpec("metamorphic", "validation", "relations the physics implies", "PC303", 3),
    CheckSpec("data.physics", "input_pedigree", "training data solve their PDE", "PC451", 4),
    CheckSpec("data.diversity", "input_pedigree", "training data are diverse, without duplicates", "PC452", 3),
    CheckSpec("data.leakage", "input_pedigree", "no test sample is a copy of a training sample", "PC453", 3),
    CheckSpec("data.finite", "input_pedigree", "data finite (missing values documented)", "PC403", 3, False),
    CheckSpec("uncertainty.residual_bound", "uncertainty",
              "calibrated error bound from the residual covers its target", "", 3, False, better="higher"),
    CheckSpec("uncertainty.spread", "uncertainty", "ensemble or predictive spread reported", "", 2, False),
    CheckSpec("robustness.input_noise", "robustness", "predictions stable under small input perturbations", "", 3,
              False),
    CheckSpec("robustness.fault_injection", "robustness", "checks detect injected faults (check power)", "", 3,
              False),
    CheckSpec("rollout", "validation", "long rollouts keep invariants and spectra", "", 3, False),
    CheckSpec("rollout.finite", "solution_verification", "the rollout stays finite", "", 3),
    CheckSpec("rollout.invariant", "solution_verification", "conserved quantities stay constant along the rollout",
              "PC105", 3),
    CheckSpec("rollout.invariants", "validation", "invariants follow the reference trajectory", "", 3, False, better="higher"),
    CheckSpec("rollout.spectrum", "validation", "small-scale energy kept along the rollout", "PC704", 3, False, better="higher"),
    CheckSpec("rollout.horizon", "uncertainty", "trust horizon (first divergence from the reference)", "", 2, False, better="higher"),
    CheckSpec("rollout.statistics", "validation", "long-run statistics match the reference attractor", "", 3),
    CheckSpec("pinn.residual", "solution_verification", "PDE residual at points not used in training", "PC302", 3),
    CheckSpec("pinn.trivial", "validation", "the solution is not near-trivial (spread, gradient energy)", "", 3, better="higher"),
    CheckSpec("pinn.causality", "validation", "the solution propagates in time from the initial condition", "", 3),
    CheckSpec("pinn.boundary", "solution_verification", "boundary conditions at fresh points", "PC201", 3),
    CheckSpec("pinn.initial", "solution_verification", "initial condition at fresh points", "PC201", 3),
    CheckSpec("pinn.spectrum", "validation", "frequency content against the exact solution or the forcing", "PC704", 3,
              False),
    CheckSpec("pinn.exact", "solution_verification", "error against an exact solution", "", 4),
    CheckSpec("applicability.range", "applicability", "queries inside the validity range", "PC601", 3),
    CheckSpec("applicability.envelope", "applicability", "query inputs inside the training envelope", "PC601", 3),
]
CHECKS: dict[str, CheckSpec] = {s.id: s for s in _SPECS}


def spec_of(check_id: str) -> CheckSpec:
    """The spec of ``check_id``; sub-checks inherit from their family (``metamorphic.symmetry.reflect-x`` ->
    ``metamorphic``, ``data.physics.solve`` -> ``data.physics``)."""
    parts = check_id.split(".")
    for i in range(len(parts), 0, -1):
        key = ".".join(parts[:i])
        if key in CHECKS:
            return CHECKS[key]
    raise KeyError(f"unknown check id {check_id!r}: add it to pinneapple_analysis.validation.schema.CHECKS")
