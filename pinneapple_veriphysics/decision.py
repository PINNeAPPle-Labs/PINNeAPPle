"""``DecisionRecord`` -- the human-facing output of one Veriphysics run:
"here's what I recommend, here's the evidence, here's what I'd try next
if you disagree." Built ONLY from real objects the orchestrator already
produced (a ``ProvenanceRecord``, a
``pinneapple_analysis.verification.physics_confidence_score.PhysicsConfidenceScore``,
a ``pinneapple_analysis.verification.tool_recommendation.ToolRecommendation``,
an evidence-graph summary) -- this module never computes a new number or
verdict of its own, it only renders ones that already exist, matching
the anti-fabrication discipline of every ``pinneapple_analysis.verification``
module it draws from.

(Extracted from Veriphysics' orchestrator; Apache-2.0 like the rest of PINNeAPPle. Execution queue, API and billing stay in Veriphysics.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# The fixed, documented set of checks
# ``pinneapple_analysis.verification.physics_confidence_score
# .compute_physics_confidence`` knows how to aggregate (that module's own
# ``N_POSSIBLE_COMPONENTS = 4`` and its "four checks this module knows how
# to aggregate" docstring section). Duplicated here as a plain string tuple
# -- rather than importing the scoring module -- because this module's own
# contract is "renders real numbers that already exist, never computes a
# new one" (see class docstring below); this list is not a computation, it
# is only the fixed vocabulary needed to render "not run" for a component
# genuinely absent from ``confidence_components`` without fabricating a
# score for it (the same "absent is absent" rule
# ``PhysicsConfidenceScore``/``GuardrailReport`` already follow).
ALL_CONFIDENCE_COMPONENT_NAMES = (
    "physics_guardrail",
    "numerical_convergence",
    "uq_calibration",
    "benchmark_agreement",
)


@dataclass
class DecisionRecord:
    run_id: str
    problem_description: str

    recommended_strategy: str
    reasoning: str

    trust_score: Optional[float]  # 0-100, or None if no confidence checks ran
    trust_coverage: float  # 0-1, fraction of possible confidence checks that ran
    trustworthy: Optional[bool]  # from provenance.guardrail_trustworthy, never re-derived here

    alternatives_considered: List[Dict[str, str]] = field(default_factory=list)
    external_tools_recommended: List[str] = field(default_factory=list)
    external_tools_reasoning: str = ""

    next_best_action: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    # ``PhysicsConfidenceScore.components`` (see
    # ``pinneapple_analysis.verification.physics_confidence_score``),
    # carried through UNCHANGED as plain dicts
    # (``{"name": ..., "score": ..., "source_summary": ...}``) -- one per
    # real check that actually ran. This is the per-check breakdown a
    # reviewing engineer needs to see WHERE ``trust_score``/``trust_coverage``
    # came from, rather than trusting one aggregate number (see this
    # module's own docstring, and README.md's "Anti-fabrication design").
    # A check that did not run is simply absent from this list -- never a
    # fabricated placeholder entry with a neutral score; see
    # ``ALL_CONFIDENCE_COMPONENT_NAMES`` above and ``render()`` below for
    # how "not run" is displayed without inventing a number for it.
    confidence_components: List[Dict[str, Any]] = field(default_factory=list)

    # ``applicability.ApplicabilityMap.to_dict()`` -- the evidence chain
    # (verified / inferred / unsupported), the eight-item review checklist
    # and the per-variable envelope. None for records built before this
    # field existed or by callers that did not supply one.
    applicability: Optional[Dict[str, Any]] = None

    def render(self) -> str:
        """A plain-text report in the format the platform's design spec
        asked for: recommended strategy, trust score, evidence, and a
        concrete next action -- never hiding an untrustworthy result."""
        lines = [
            f"Decision Record -- run {self.run_id}",
            f"Problem: {self.problem_description}",
            "",
            f"Recommended strategy: {self.recommended_strategy}",
            f"Reasoning: {self.reasoning}",
            "",
        ]
        if self.trust_score is None:
            lines.append("Trust score: not computed (no confidence checks were available for this run)")
        else:
            lines.append(
                f"Trust score: {self.trust_score:.0f}/100 "
                f"(coverage={self.trust_coverage:.2f}, i.e. based on a subset of possible checks "
                f"if less than 1.0)"
            )
        if self.trustworthy is not None:
            lines.append(f"Guardrail verdict: {'TRUSTWORTHY' if self.trustworthy else 'NOT TRUSTWORTHY'}")

        lines.append("")
        lines.append("Per-check confidence breakdown:")
        by_name = {c["name"]: c for c in self.confidence_components}
        for name in ALL_CONFIDENCE_COMPONENT_NAMES:
            comp = by_name.get(name)
            if comp is None:
                lines.append(f"  - {name}: not run")
            else:
                lines.append(f"  - {name}: {comp['score']:.2f} -- {comp['source_summary']}")

        if self.alternatives_considered:
            lines.append("")
            lines.append("Alternatives considered:")
            for alt in self.alternatives_considered:
                lines.append(f"  - {alt.get('option', '?')}: {alt.get('why_not', '')}")

        if self.external_tools_recommended:
            lines.append("")
            lines.append(f"External tools also worth considering: {', '.join(self.external_tools_recommended)}")
            if self.external_tools_reasoning:
                lines.append(f"  {self.external_tools_reasoning}")

        if self.next_best_action:
            lines.append("")
            lines.append(f"Next best action: {self.next_best_action}")

        if self.applicability:
            lines += ["", _render_applicability(self.applicability)]

        return "\n".join(lines)


def _render_applicability(data: Dict[str, Any]) -> str:
    """Render ``DecisionRecord.applicability`` (the dict form) as text by
    rebuilding the dataclass view -- kept here so a record loaded from
    JSON renders identically to a live one."""
    from .applicability import ApplicabilityMap, ChecklistItem, VariableEnvelope

    amap = ApplicabilityMap(
        checklist=[ChecklistItem(**i) for i in data["checklist"]],
        envelope=[
            VariableEnvelope(**{**v, "verified_range": tuple(v["verified_range"]) if v.get("verified_range") else None})
            for v in data["envelope"]
        ],
        verified=data["verified"], inferred=data["inferred"], unsupported=data["unsupported"],
    )
    return amap.render()
