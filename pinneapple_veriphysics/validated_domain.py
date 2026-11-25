"""Conservative, model-version-bound joint parameter evidence.

Local support is an empirical screening criterion, not a mathematical error bound.
All distances use the declared training scales; no marginal range implies validation.
"""
from dataclasses import dataclass
from math import isfinite, dist
from typing import Mapping, Sequence


@dataclass(frozen=True)
class GeometryEvidence:
    point: tuple[float, ...]
    model_id: str
    reference_id: str
    independent: bool
    reference_error: float
    physics_errors: tuple[float, ...]
    uncertainty: float


@dataclass(frozen=True)
class DomainPolicy:
    reference_tol: float
    physics_tols: tuple[float, ...]
    uncertainty_tol: float
    radius: float
    min_neighbors: int = 2

    def __post_init__(self):
        if not self.physics_tols or any(not isfinite(x) or x < 0 for x in
                (self.reference_tol, *self.physics_tols, self.uncertainty_tol, self.radius)):
            raise ValueError("Explicit finite nonnegative tolerances required")
        if self.min_neighbors < 1 or self.radius <= 0:
            raise ValueError("Positive radius and neighbor count required")


class ValidatedDomain:
    """Evidence for a specific ordered schema and immutable model identifier.

    Reference errors and uncertainty must be computed by the caller using
    independent geometry-level holdouts and calibrated uncertainty respectively.
    """

    def __init__(self, bounds: Mapping[str, tuple[float, float]], model_id: str,
                 policy: DomainPolicy, training: Sequence[Sequence[float]],
                 evidence: Sequence[GeometryEvidence] = ()):
        self.names = tuple(bounds)
        self.bounds = tuple(tuple(bounds[n]) for n in self.names)
        self.model_id = model_id
        self.policy = policy
        if not model_id or not self.names or any(not isfinite(a) or not isfinite(b)
                or a >= b for a, b in self.bounds):
            raise ValueError("Model ID and finite nondegenerate bounds required")
        self.training = tuple(self._normalize(p) for p in training)
        self.evidence = tuple(evidence)
        for e in self.evidence:
            self._normalize(e.point)
            if e.model_id != model_id:
                raise ValueError("Evidence belongs to a different model")
            if len(e.physics_errors) != len(policy.physics_tols):
                raise ValueError("Physics check schema mismatch")

    def _normalize(self, point):
        if len(point) != len(self.names) or not all(isfinite(x) for x in point):
            raise ValueError("Invalid point or parameter schema")
        return tuple((x-a)/(b-a) for x, (a,b) in zip(point, self.bounds))

    def _passes(self, e):
        values = (e.reference_error, *e.physics_errors, e.uncertainty)
        limits = (self.policy.reference_tol, *self.policy.physics_tols,
                  self.policy.uncertainty_tol)
        return bool(e.independent and e.reference_id and
                    all(isfinite(v) and 0 <= v <= t for v,t in zip(values, limits)))

    def assess(self, query: Mapping[str, float]):
        if set(query) != set(self.names):
            return {"status": "UNSUPPORTED", "reason": "parameter_schema_changed"}
        q = self._normalize(tuple(query[n] for n in self.names))
        if any(x < 0 or x > 1 for x in q):
            return {"status": "UNSUPPORTED", "reason": "outside_training_bounds"}
        nearby = [e for e in self.evidence
                  if dist(q, self._normalize(e.point)) <= self.policy.radius]
        if any(not self._passes(e) for e in nearby):
            return {"status": "UNSUPPORTED", "reason": "local_failed_or_missing_evidence"}
        if not self.training or min(dist(q,p) for p in self.training) > self.policy.radius:
            return {"status": "UNSUPPORTED", "reason": "training_coverage_gap"}
        if len({tuple(e.point) for e in nearby}) < self.policy.min_neighbors:
            return {"status": "UNSUPPORTED", "reason": "insufficient_local_reference"}
        # Require bracketing in every dimension; this is not a convex hull test.
        points = [self._normalize(e.point) for e in nearby]
        if any(not min(p[i] for p in points) <= x <= max(p[i] for p in points)
               for i,x in enumerate(q)):
            return {"status": "UNSUPPORTED", "reason": "local_extrapolation"}
        return {"status": "EMPIRICALLY_SUPPORTED", "reason": "local_checks_passed",
                "reference_ids": sorted({e.reference_id for e in nearby}),
                "neighbors": len(points)}

    def extend(self, evidence: Sequence[GeometryEvidence]):
        """Return a new snapshot; failed observations are retained."""
        training = [tuple(a+x*(b-a) for x,(a,b) in zip(p,self.bounds))
                    for p in self.training]
        return ValidatedDomain(dict(zip(self.names,self.bounds)), self.model_id,
                               self.policy, training, self.evidence + tuple(evidence))

    def prioritize(self, candidates: Sequence[Mapping[str, float]]):
        """Return unsupported candidate geometries for additional solver runs."""
        return [(dict(q), self.assess(q)) for q in candidates
                if self.assess(q)["status"] != "EMPIRICALLY_SUPPORTED"]
