"""VeriPhysics adapter for trusted experiment worker callbacks.

The host supplies a reviewed model, symbolic PDE system, sampling bounds and
reference observations. No LLM-supplied import paths or executable expressions.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def veriphysics_model_handler(
    *, model: Callable, system: Any, bounds: dict,
    reference: tuple | None = None, validity: dict | None = None,
    policy: Any = None,
) -> Callable:
    """Return a stage handler backed by the actual VeriPhysics model grader."""
    def verify(params: dict, ctx: Any) -> dict:
        from pinneapple_veriphysics.model_grade import grade_model
        from pinneapple_veriphysics.policy import load_policy

        resolved_policy = policy if policy is not None else load_policy(None)
        grade = grade_model(
            model, system, bounds, reference=reference,
            validity=validity, policy=resolved_policy,
        )
        decision = grade.get("decision")
        return {
            "approved": decision == "APPROVED",
            "decision": decision,
            "checks": grade.get("checks", []),
            "reasons": grade.get("reasons", []),
            "policy": grade.get("policy"),
            "verifier": "pinneapple_veriphysics.model_grade.grade_model",
        }
    return verify
