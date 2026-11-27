"""Case-specific numerical admission, not a blanket solver certification."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .asme_vv20 import ValidationResult


@dataclass(frozen=True)
class SolverAdmission:
    approved: bool
    reasons: tuple[str, ...]


def assess_solver_result(
    comparison: ValidationResult,
    *,
    max_absolute_error: float,
    max_numerical_uncertainty: float,
    max_validation_uncertainty: float,
    convergence_confirmed: bool,
) -> SolverAdmission:
    """Apply independent engineering tolerances to one quantity of interest.

    All tolerances are in the quantity's units. Agreement within uncertainty
    alone is insufficient: large uncertainty must not excuse poor accuracy.
    Convergence must come from a real case-specific study supplied by caller.
    """
    limits = (max_absolute_error, max_numerical_uncertainty,
              max_validation_uncertainty)
    if any(not isfinite(v) or v < 0 for v in limits):
        raise ValueError("All tolerances must be finite and nonnegative")
    reasons = []
    if convergence_confirmed is not True:
        reasons.append("numerical convergence not confirmed")
    if not all(isfinite(v) for v in (comparison.simulation, comparison.data,
                                     comparison.e, comparison.u_val)):
        reasons.append("nonfinite comparison")
    else:
        if abs(comparison.e) > max_absolute_error:
            reasons.append("absolute error exceeds tolerance")
        if comparison.u_val > max_validation_uncertainty:
            reasons.append("validation uncertainty exceeds tolerance")
        if not comparison.achieved:
            reasons.append("comparison outside validation uncertainty")
    if comparison.budget.u_numerical > max_numerical_uncertainty:
        reasons.append("numerical uncertainty exceeds tolerance")
    return SolverAdmission(not reasons, tuple(reasons))
