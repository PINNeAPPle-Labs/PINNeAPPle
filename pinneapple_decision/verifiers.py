"""Verifiers: turn an ExecutionResult into a Verification.

* :class:`ThresholdVerifier` -- simple, dependency-free: each configured metric
  must be <= its threshold. A metric the result does not report is a check that
  did not run (absent from ``checks``), never a pass. No check ran -> not passed.
* :class:`VeriPhysicsVerifier` -- adapter for a VeriPhysics ``DecisionRecord``
  (``trust_score`` 0-100, ``trust_coverage`` 0-1, ``trustworthy``) or a
  PINNeAPPle ``PhysicsConfidenceScore`` (``overall_score`` 0-1, ``coverage``),
  found in ``result.artifacts["decision_record"]`` /
  ``result.artifacts["physics_confidence"]``. Duck-typed: it reads attributes and
  does not import ``veriphysics``. When neither artifact is present it delegates
  to its ``fallback`` verifier and says so in ``source``.
"""
from __future__ import annotations

import importlib.util
from typing import Dict, List, Optional, Protocol, runtime_checkable

from .schema import ExecutionResult, Verification

# Metric name -> tag emitted when the check fails (fed to the next decision).
FAIL_TAGS = {"residual": "residual_high", "rel_l2": "error_high", "bc_error": "bc_violation"}


@runtime_checkable
class Verifier(Protocol):
    def verify(self, result: ExecutionResult, problem: Optional[dict] = None) -> Verification:
        ...


class ThresholdVerifier:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = dict(thresholds or {"residual": 1e-3, "rel_l2": 5e-2})

    def verify(self, result: ExecutionResult, problem: Optional[dict] = None) -> Verification:
        checks: Dict[str, bool] = {}
        tags: List[str] = list(result.diagnostics)
        for metric, limit in self.thresholds.items():
            if metric in result.metrics:
                ok = float(result.metrics[metric]) <= limit
                checks[metric] = ok
                if not ok:
                    tags.append(FAIL_TAGS.get(metric, f"{metric}_above_threshold"))
        passed = result.status == "ok" and bool(checks) and all(checks.values())
        notes = "" if checks else "no configured metric was reported; nothing was verified"
        return Verification(passed=passed, checks=checks, tags=sorted(set(tags)), source="threshold", notes=notes)


class VeriPhysicsVerifier:
    def __init__(self, min_trust: float = 70.0, min_coverage: float = 0.5,
                 fallback: Optional[Verifier] = None):
        self.min_trust = min_trust
        self.min_coverage = min_coverage
        self.fallback = fallback or ThresholdVerifier()

    @staticmethod
    def veriphysics_installed() -> bool:
        return importlib.util.find_spec("veriphysics") is not None

    def verify(self, result: ExecutionResult, problem: Optional[dict] = None) -> Verification:
        rec = result.artifacts.get("decision_record")
        pcs = result.artifacts.get("physics_confidence")
        if rec is not None:
            score, coverage = getattr(rec, "trust_score", None), float(getattr(rec, "trust_coverage", 0.0))
            trustworthy = getattr(rec, "trustworthy", None)
            source = "veriphysics.DecisionRecord"
        elif pcs is not None:
            raw = getattr(pcs, "overall_score", None)
            score, coverage, trustworthy = (None if raw is None else 100.0 * raw), float(pcs.coverage), None
            source = "pinneapple.PhysicsConfidenceScore"
        else:
            v = self.fallback.verify(result, problem)
            v.source = f"fallback:{v.source} (no VeriPhysics artifact in result)"
            return v

        checks: Dict[str, bool] = {"coverage": coverage >= self.min_coverage}
        if score is not None:
            checks["trust_score"] = score >= self.min_trust
        if trustworthy is not None:
            checks["trustworthy"] = bool(trustworthy)
        tags = list(result.diagnostics)
        if not checks["coverage"]:
            tags.append("low_verification_coverage")
        if score is not None and not checks["trust_score"]:
            tags.append("low_trust")
        passed = result.status == "ok" and score is not None and all(checks.values())
        return Verification(passed=passed, checks=checks, tags=sorted(set(tags)), source=source,
                            notes=f"trust_score={score}, coverage={coverage}")


__all__ = ["ThresholdVerifier", "VeriPhysicsVerifier", "Verifier"]
