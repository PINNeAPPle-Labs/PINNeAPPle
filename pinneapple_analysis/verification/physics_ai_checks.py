"""Independent, assumption-scoped Physics AI checks (stdlib only)."""
from __future__ import annotations
from dataclasses import asdict
import math
from typing import Sequence

from .pipeline_assurance import Check, Status, numeric_close


def _finite_vector(values: Sequence[float]) -> bool:
    return bool(values) and all(math.isfinite(float(x)) for x in values)


def field_parity(name: str, stage: str, reference: Sequence[float],
                 candidate: Sequence[float], *, atol: float = 1e-8,
                 rtol: float = 1e-5) -> Check:
    """Compare raw/converted, PyTorch/ONNX, or solver/surrogate fields."""
    if len(reference) != len(candidate) or not _finite_vector(reference) or not _finite_vector(candidate):
        return Check(name, stage, Status.INCONCLUSIVE, reason="Nonfinite, empty or mismatched fields")
    if atol < 0 or rtol < 0:
        raise ValueError("Tolerances must be nonnegative")
    errors = [abs(float(a) - float(b)) for a, b in zip(reference, candidate)]
    violations = [i for i, (a, b) in enumerate(zip(reference, candidate))
                  if abs(a - b) > atol + rtol * abs(a)]
    return Check(name, stage, Status.FAIL if violations else Status.PASS,
                 evidence={"count": len(reference), "max_absolute_error": max(errors),
                           "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
                           "violations": len(violations), "first_violation_index": violations[0] if violations else None},
                 assumptions=("Same physical coordinates, ordering, units and components",))


def synthetic_recovery(name: str, true_parameters: Sequence[float],
                       estimated_parameters: Sequence[float], *, atol: float = 1e-6,
                       rtol: float = 1e-3) -> Check:
    """Synthetic recovery checks estimation, NOT inverse identifiability."""
    check = field_parity(name, "inverse", true_parameters, estimated_parameters, atol=atol, rtol=rtol)
    return Check(check.name, check.stage, check.status, evidence=check.evidence,
                 reason="Synthetic recovery does not prove parameter uniqueness.",
                 assumptions=("Known synthetic ground truth", "Consistent parameter order and units"))


def inverse_identifiability(name: str, jacobian: Sequence[Sequence[float]],
                            *, relative_threshold: float = 1e-8) -> Check:
    """Local rank check using optional NumPy; never claims global identifiability."""
    if relative_threshold <= 0:
        raise ValueError("relative_threshold must be positive")
    try:
        import numpy as np
    except ImportError:
        return Check(name, "inverse", Status.NOT_RUN, reason="NumPy unavailable")
    try:
        j = np.asarray(jacobian, dtype=float)
        if j.ndim != 2 or not j.size or not np.isfinite(j).all():
            return Check(name, "inverse", Status.INCONCLUSIVE, reason="Invalid Jacobian")
        singular = np.linalg.svd(j, compute_uv=False)
        rank = int(np.sum(singular > relative_threshold * singular[0])) if singular[0] > 0 else 0
        parameters = j.shape[1]
        return Check(name, "inverse", Status.PASS if rank == parameters else Status.FAIL,
                     evidence={"local_rank": rank, "parameter_count": parameters,
                               "singular_values": singular.tolist(),
                               "relative_threshold": relative_threshold},
                     assumptions=("Differentiable local forward map", "Jacobian columns are comparably scaled"),
                     reason="Full local rank does not establish global uniqueness.")
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        return Check(name, "inverse", Status.INCONCLUSIVE, reason=str(exc))


def holdout_leakage(name: str, train_groups: Sequence[str],
                    evaluation_groups: Sequence[str]) -> Check:
    """Check group-level leakage (geometry, simulation, patient, trajectory, etc.)."""
    if not train_groups or not evaluation_groups:
        return Check(name, "dataset", Status.INCONCLUSIVE, reason="Both splits must be nonempty")
    overlap = sorted(set(train_groups) & set(evaluation_groups))
    return Check(name, "dataset", Status.FAIL if overlap else Status.PASS,
                 evidence={"overlapping_groups": overlap, "overlap_count": len(overlap)},
                 assumptions=("Group identifiers capture all correlated samples",))


def evaluate_physical_limits(name: str, stage: str, values: Sequence[float],
                             *, lower: float | None = None,
                             upper: float | None = None,
                             assumptions: tuple[str, ...]) -> Check:
    if not assumptions or (lower is None and upper is None):
        return Check(name, stage, Status.INCONCLUSIVE, reason="Applicability and a bound are required")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("Lower bound exceeds upper bound")
    if not _finite_vector(values):
        return Check(name, stage, Status.INCONCLUSIVE, reason="Nonfinite or empty field")
    count = sum((lower is not None and v < lower) or (upper is not None and v > upper) for v in values)
    return Check(name, stage, Status.FAIL if count else Status.PASS,
                 evidence={"violations": count, "samples": len(values),
                           "observed_min": min(values), "observed_max": max(values)},
                 assumptions=assumptions)


def temporal_composition(name: str, direct: Sequence[float],
                         composed: Sequence[float], *, atol: float = 1e-6,
                         rtol: float = 1e-4) -> Check:
    result = field_parity(name, "training", direct, composed, atol=atol, rtol=rtol)
    return Check(result.name, result.stage, result.status, evidence=result.evidence,
                 reason=result.reason,
                 assumptions=("Autonomous deterministic evolution", "Identical state representation",
                              "Consistent integration times"))
