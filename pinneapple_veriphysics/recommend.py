"""The "formulation + method intelligence" half of ``pipeline.analyze()``,
factored out so it can run on its own -- fast (no training, one
``draft_problem`` LLM call), for the "Solution Recommendation" screen:
internal solver/architecture choice and external tools/players, shown
side by side with real, documented pros/cons for every candidate
(never fabricated accuracy/cost numbers -- see ``tool_recommendation``/
``architecture_recommendation``'s own honesty-scope docstrings), BEFORE
committing to a real training run. ``pipeline.analyze()`` itself calls
this function too, so the two entry points can never silently disagree
about what formulation/method intelligence step 1-3 produced.

(Extracted from Veriphysics' orchestrator; Apache-2.0 like the rest of PINNeAPPle. Execution queue, API and billing stay in Veriphysics.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pinneapple_analysis.verification.dimensional_analysis import (
    compute_dimensionless_numbers,
    classify_flow_regime,
)
from pinneapple_analysis.verification.solver_orchestration import select_solver_family
from pinneapple_analysis.verification.tool_recommendation import (
    recommend_tools,
    get_player_for_tool,
    TOOL_CATALOG,
)
from pinneapple_analysis.verification.architecture_recommendation import (
    recommend_architecture,
    ARCHITECTURE_CATALOG,
)

from .execution_log import LogCollector, emit
from .example_cases import get_example_case

__all__ = ["RecommendationResult", "formulate_and_recommend"]


@dataclass
class CandidateDetail:
    """One real, documented option -- an architecture or an external
    tool -- with its real strengths/weaknesses, never a fabricated
    score. ``is_recommended`` distinguishes the primary pick from an
    alternative; every field beyond that comes straight from the real
    catalog entry (``ARCHITECTURE_CATALOG`` / ``TOOL_CATALOG``)."""
    key: str  # registry_key (architecture) or catalog key (tool)
    name: str
    category: str
    is_recommended: bool
    when_to_use: str = ""  # architectures only
    license: str = ""  # tools only: "open-source" | "commercial"
    typical_cost_tier: str = ""  # tools only: "free" | "$" | "$$" | "$$$"
    player: Optional[str] = None  # tools only: the real company behind it, or None (community/consortium)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)  # architectures: "weaknesses"; tools: "caveats"
    source: str = ""


@dataclass
class RecommendationResult:
    rejected: bool
    rejection_reason: str = ""

    dimensionless_numbers: Dict[str, Any] = None
    flow_regime: Optional[str] = None

    drafted_preset: Optional[str] = None
    drafted_preset_kwargs: Optional[Dict[str, Any]] = None
    draft_reasoning: str = ""
    spec: Any = None

    # Internal method intelligence
    recommended_solver_family: Optional[str] = None
    solver_reasoning: str = ""
    solver_fallback_families: List[str] = None

    recommended_architectures: List[str] = None  # real registry_keys, e.g. "vanilla_pinn"
    architecture_reasoning: str = ""
    architecture_alternatives: List[Dict[str, Any]] = None
    architecture_details: List[CandidateDetail] = None  # recommended + alternatives, full real detail

    # External method intelligence
    recommended_tools: List[str] = None
    tool_reasoning: str = ""
    tool_alternatives: List[Dict[str, Any]] = None
    tool_details: List[CandidateDetail] = None  # recommended + alternatives, full real detail

    # A real, synthesized end-to-end pipeline string combining the
    # internal architecture pick with any recommended external data-
    # generation tool -- e.g. "openfoam (generate training data) ->
    # fno3d (train surrogate)" or, when no external tool matched,
    # just the architecture trained directly. Never invents a tool/
    # architecture not already in recommended_tools/recommended_architectures.
    recommended_pipeline: str = ""


def _architecture_details(arch_rec, recommended_architectures: List[str]) -> List[CandidateDetail]:
    details: List[CandidateDetail] = []
    seen = set()
    for key in arch_rec.recommended:
        candidate = ARCHITECTURE_CATALOG.get(key)
        if candidate is None or candidate.registry_key in seen:
            continue
        seen.add(candidate.registry_key)
        details.append(CandidateDetail(
            key=candidate.registry_key, name=candidate.name, category=candidate.category,
            is_recommended=candidate.registry_key in recommended_architectures,
            when_to_use=candidate.when_to_use, strengths=list(candidate.strengths),
            weaknesses=list(candidate.weaknesses), source=candidate.source,
        ))
    for alt in arch_rec.alternatives:
        registry_key = alt["registry_key"]
        if registry_key in seen:
            continue
        seen.add(registry_key)
        candidate = ARCHITECTURE_CATALOG.get(alt["architecture"])
        if candidate is None:
            continue
        details.append(CandidateDetail(
            key=registry_key, name=candidate.name, category=candidate.category, is_recommended=False,
            when_to_use=candidate.when_to_use, strengths=list(candidate.strengths),
            weaknesses=list(candidate.weaknesses), source=candidate.source,
        ))
    return details


def _tool_details(tool_rec) -> List[CandidateDetail]:
    details: List[CandidateDetail] = []
    seen = set()
    for key in tool_rec.recommended:
        tool = TOOL_CATALOG.get(key)
        if tool is None or key in seen:
            continue
        seen.add(key)
        details.append(CandidateDetail(
            key=key, name=tool.name, category=tool.category, is_recommended=True,
            license=tool.license, typical_cost_tier=tool.typical_cost_tier,
            player=get_player_for_tool(key), strengths=list(tool.strengths),
            weaknesses=list(tool.caveats), source=tool.source,
        ))
    for alt in tool_rec.alternatives:
        key = alt["tool"]
        if key in seen:
            continue
        seen.add(key)
        tool = TOOL_CATALOG.get(key)
        if tool is None:
            continue
        details.append(CandidateDetail(
            key=key, name=tool.name, category=tool.category, is_recommended=False,
            license=tool.license, typical_cost_tier=tool.typical_cost_tier,
            player=get_player_for_tool(key), strengths=list(tool.strengths),
            weaknesses=list(tool.caveats), source=tool.source,
        ))
    return details


def _synthesize_pipeline(recommended_architectures: List[str], recommended_tools: List[str]) -> str:
    if not recommended_architectures:
        return ""
    arch = recommended_architectures[0]
    # A real (not fabricated) heuristic: a CFD/FEM data-generation tool
    # in the recommended external list implies "generate training data
    # with it, then train the recommended surrogate" is a coherent real
    # pipeline -- ONLY stated when a real matching tool was actually
    # recommended, never invented.
    data_gen_categories = {"CFD", "FEM", "multiphysics"}
    data_tools = [
        TOOL_CATALOG[k].name for k in recommended_tools
        if k in TOOL_CATALOG and TOOL_CATALOG[k].category in data_gen_categories
    ]
    if data_tools:
        return f"{data_tools[0]} (generate training data) -> {arch} (train surrogate)"
    return f"{arch} (trained directly on the drafted preset, no external data generation needed)"


def formulate_and_recommend(
    description: str,
    *,
    physical_parameters: Optional[Dict[str, float]] = None,
    flow_geometry: str = "internal_pipe",
    provider: str = "ollama",
    model_name: Optional[str] = None,
    has_lots_of_data: Optional[bool] = None,
    needs_parameter_generalization: bool = False,
    geometry_varies: bool = False,
    is_inverse_problem: bool = False,
    example_case: Optional[str] = None,
    log: Optional[LogCollector] = None,
) -> RecommendationResult:
    """Formulate the problem (dimensional analysis + LLM preset draft) and
    compute internal + external method-intelligence recommendations --
    no training, no PhysicsGuardrail check. Fast enough for an
    interactive "recommend before you commit" UI screen.

    example_case
        Optional key into
        ``pinneapple_veriphysics.example_cases.EXAMPLE_CASES`` (one of
        the platform's fixed, pre-defined demo cases). When given, the
        ``description``/``physical_parameters``/``flow_geometry``
        arguments above are still recorded, but the preset itself is
        looked up directly from that catalog instead of calling
        ``pinneapple_llm.draft_problem`` -- see that module's docstring
        for exactly why (measured LLM kwargs unreliability for 2 of the
        3 catalog entries, not a hypothetical concern). Raises
        ``ValueError`` for an unknown key (mechanically checked, never a
        silent fallback to a different case).

    ``log``, if given, receives a real, step-by-step trace of what this
    function actually does and decides (see
    ``pinneapple_veriphysics.execution_log``).
    """
    import pinneapple_llm as pl
    import pinneapple_physics as pp

    emit(log, f"Computing dimensionless numbers from {len(physical_parameters or {})} supplied physical parameter(s)...")
    physical_parameters = physical_parameters or {}
    dn = compute_dimensionless_numbers(**physical_parameters)
    regime = classify_flow_regime(dn.reynolds, geometry=flow_geometry)
    dn_dict = {k: v for k, v in dn.__dict__.items() if k != "notes"}
    dn_dict["notes"] = dn.notes
    emit(log, f"Flow regime classified as: {regime}")

    if example_case is not None:
        case = get_example_case(example_case)  # raises ValueError for an unknown key
        emit(log, f"example_case={example_case!r} given -- using its fixed, known-good preset "
                  f"{case.preset!r} (kwargs={case.preset_kwargs}), skipping LLM preset drafting entirely.")
        draft = pl.DraftResult(
            preset=case.preset, kwargs=dict(case.preset_kwargs),
            reasoning=f"pre-defined example case {example_case!r} -- LLM preset drafting bypassed for reliability.",
            raw_response="",
        )
    else:
        emit(log, f"Drafting a registered preset from the problem description via LLM ({provider}/{model_name or 'default'})...")
        try:
            draft = pl.draft_problem(description, provider=provider, model=model_name)
        except ValueError as e:
            emit(log, f"draft_problem rejected a hallucinated response: {e}")
            return RecommendationResult(
                rejected=True, rejection_reason=f"draft_problem rejected a hallucinated response: {e}",
                dimensionless_numbers=dn_dict, flow_regime=regime,
            )

        if draft.preset is None:
            emit(log, f"No registered preset judged a fit: {draft.reasoning}")
            return RecommendationResult(
                rejected=True, rejection_reason=f"no registered preset judged a fit: {draft.reasoning}",
                dimensionless_numbers=dn_dict, flow_regime=regime, draft_reasoning=draft.reasoning,
            )
    emit(log, f"Drafted preset: {draft.preset!r} (kwargs={draft.kwargs}) -- {draft.reasoning}")

    spec = pp.get_preset(draft.preset, **draft.kwargs)

    emit(log, "Selecting internal solver family (checking live availability of PINN/FDM/FEM/OpenFOAM/FEniCS)...")
    solver_rec = select_solver_family(spec, dn)
    emit(log, f"Internal solver family selected: {solver_rec.recommended_family!r} -- {solver_rec.reasoning}")

    emit(log, "Matching external tools/players against this problem's documented PDE scope...")
    tool_rec = recommend_tools(spec, dn)
    emit(log, f"External tools recommended: {tool_rec.recommended or '(none matched)'}")

    emit(log, "Running the architecture-recommendation decision tree (data availability, generalization, inverse-problem axes)...")
    arch_rec = recommend_architecture(
        n_high_fidelity_simulations=0,
        has_lots_of_data=has_lots_of_data,
        has_analytical_solver=(solver_rec.recommended_family != "pinn"),
        needs_parameter_generalization=needs_parameter_generalization,
        geometry_varies=geometry_varies,
        is_inverse_problem=is_inverse_problem,
    )
    recommended_architectures = [
        ARCHITECTURE_CATALOG[key].registry_key for key in arch_rec.recommended if key in ARCHITECTURE_CATALOG
    ]
    emit(log, f"Architectures recommended: {recommended_architectures} -- {arch_rec.reasoning}")

    pipeline = _synthesize_pipeline(recommended_architectures, list(tool_rec.recommended))
    emit(log, f"Synthesized end-to-end strategy: {pipeline}")

    return RecommendationResult(
        rejected=False,
        dimensionless_numbers=dn_dict, flow_regime=regime,
        drafted_preset=draft.preset, drafted_preset_kwargs=draft.kwargs, draft_reasoning=draft.reasoning,
        spec=spec,
        recommended_solver_family=solver_rec.recommended_family, solver_reasoning=solver_rec.reasoning,
        solver_fallback_families=list(solver_rec.fallback_families),
        recommended_architectures=recommended_architectures, architecture_reasoning=arch_rec.reasoning,
        architecture_alternatives=list(arch_rec.alternatives),
        architecture_details=_architecture_details(arch_rec, recommended_architectures),
        recommended_tools=list(tool_rec.recommended), tool_reasoning=tool_rec.reasoning,
        tool_alternatives=list(tool_rec.alternatives),
        tool_details=_tool_details(tool_rec),
        recommended_pipeline=pipeline,
    )
