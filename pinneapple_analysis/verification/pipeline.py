"""Problem description -> dimensional analysis -> LLM-drafted preset -> PINN training -> guardrail -> selection.

``analyze(description, ...)`` runs one natural-language physics problem through the library calls that already
exist, in order:

1. :func:`~pinneapple_analysis.verification.dimensional_analysis.compute_dimensionless_numbers` and
   :func:`~pinneapple_analysis.verification.dimensional_analysis.classify_flow_regime` on the caller's physical
   parameters (context for the record; they do not choose the preset);
2. :func:`pinneapple_llm.draft_problem` picks a registered preset and its kwargs (an unknown preset or kwarg name
   is rejected by that function, not trusted);
3. :func:`pinneapple_physics.get_preset` + :func:`pinneapple_physics.solve_pde` train one PINN per candidate
   architecture;
4. :class:`pinneapple_llm.PhysicsGuardrail` re-evaluates each trained model on fresh points;
5. :func:`rank_key` picks the winner: a trustworthy candidate beats any untrustworthy one, then the smaller
   re-computed PDE residual wins, and only then the training loss;
6. a :class:`~pinneapple_analysis.verification.provenance.ProvenanceRecord` records the run.

What this does not check: whether the drafted preset is the right one for the description. The guardrail
re-checks the physics of the preset that was chosen, not the choice (a heat-conduction request drafted as Burgers
can come back trustworthy). Read ``provenance.drafted_preset`` and ``provenance.draft_reasoning``.

Moved here from the PINNeAPPle-apps ``physics_verification_engine`` product (``core/pipeline.py``), where the
other verification modules of this package also came from; the app is now ``apps/verified_physics``.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from .dimensional_analysis import classify_flow_regime, compute_dimensionless_numbers
from .provenance import ProvenanceRecord

__all__ = ["CandidateResult", "AnalysisResult", "rank_key", "analyze", "guardrail_checks_as_dicts"]


@dataclass
class CandidateResult:
    architecture: str
    final_loss: float
    guardrail_trustworthy: bool
    guardrail_checks: list[dict[str, Any]]
    residual_value: float | None   # the guardrail's pde_residual value, when it was computable
    model: Any = None                 # the trained nn.Module (kept out of serialized reports)


@dataclass
class AnalysisResult:
    provenance: ProvenanceRecord
    best_candidate: CandidateResult | None
    all_candidates: list[CandidateResult]
    rejected: bool = False
    rejection_reason: str = ""


def rank_key(candidate: CandidateResult) -> tuple:
    """Sort key for physics-aware model selection: (untrustworthy?, re-computed residual, training loss)."""
    trust_rank = 0 if candidate.guardrail_trustworthy else 1
    residual_rank = candidate.residual_value if candidate.residual_value is not None else float("inf")
    return (trust_rank, residual_rank, candidate.final_loss)


def guardrail_checks_as_dicts(report: Any) -> list[dict[str, Any]]:
    """A ``GuardrailReport``'s checks as plain dicts (name, passed, detail, value, threshold)."""
    return [{"name": c.name, "passed": bool(c.passed), "detail": c.detail,
             "value": None if c.value is None else float(c.value),
             "threshold": None if c.threshold is None else float(c.threshold)} for c in report.checks]


def analyze(
    description: str,
    *,
    physical_parameters: dict[str, float] | None = None,
    flow_geometry: str = "internal_pipe",
    provider: str = "ollama",
    model_name: str | None = None,
    candidate_architectures: list[str] | None = None,
    epochs: int = 1500,
    n_collocation: int = 1024,
    residual_threshold: float = 1e-2,
    reference_benchmark: str | None = None,
    hidden_dim: int = 64,
    n_layers: int = 4,
    api_key: str | None = None,
) -> AnalysisResult:
    """Run the pipeline in the module docstring for one description (blocking: training happens inside).

    ``physical_parameters`` (velocity, length, density, dynamic_viscosity, kinematic_viscosity, specific_heat,
    thermal_conductivity, ...) go to ``compute_dimensionless_numbers``. With several ``candidate_architectures``,
    every one is trained and :func:`rank_key` picks the best; an architecture that cannot be built or trained for
    the preset's input/output shape is skipped. ``provider``/``model_name``/``api_key`` go to
    ``pinneapple_llm.draft_problem`` (``"ollama"`` is a local server; hosted providers read their key from the
    environment when ``api_key`` is None).
    """
    import pinneapple_llm as pl
    import pinneapple_physics as pp

    t0 = time.time()
    provenance = ProvenanceRecord(run_id=str(uuid.uuid4()), problem_description=description)

    dn = compute_dimensionless_numbers(**(physical_parameters or {}))
    provenance.dimensionless_numbers = {k: v for k, v in dn.__dict__.items() if k != "notes"}
    provenance.dimensionless_numbers["notes"] = dn.notes
    provenance.flow_regime = classify_flow_regime(dn.reynolds, geometry=flow_geometry)

    try:
        draft = pl.draft_problem(description, provider=provider, model=model_name, api_key=api_key)
    except ValueError as e:
        return AnalysisResult(provenance, None, [], rejected=True,
                              rejection_reason=f"draft_problem rejected the response: {e}")
    provenance.draft_reasoning = draft.reasoning
    if draft.preset is None:
        return AnalysisResult(provenance, None, [], rejected=True,
                              rejection_reason=f"no registered preset judged a fit: {draft.reasoning}")
    provenance.drafted_preset = draft.preset
    provenance.drafted_preset_kwargs = dict(draft.kwargs)

    spec = pp.get_preset(draft.preset, **draft.kwargs)
    from pinneapple_physics.pde_environment.presets.tag_geometry import tag_solve_kwargs

    try:
        geometry = tag_solve_kwargs(draft.preset, spec)     # boundary points for presets with tagged regions
    except KeyError as e:
        return AnalysisResult(provenance, None, [], rejected=True, rejection_reason=str(e))

    import pinneapple_neural.architectures  # noqa: F401  registers the zoo
    from pinneapple_neural.architectures.registry import ModelRegistry

    architectures = list(candidate_architectures or ["modified_mlp"])
    candidates: list[CandidateResult] = []
    for arch in architectures:
        try:
            net = ModelRegistry.build(arch, in_dim=len(spec.coords), out_dim=len(spec.fields),
                                      hidden_dim=hidden_dim, n_layers=n_layers)
            result = pp.solve_pde(spec, net, epochs=epochs, n_collocation=n_collocation, **geometry)
        except Exception:  # noqa: BLE001 -- this architecture does not fit this preset; try the next one
            continue
        report = pl.PhysicsGuardrail(spec, residual_threshold=residual_threshold).check(
            net, **({} if reference_benchmark is None else {"reference_benchmark": reference_benchmark}))
        checks = guardrail_checks_as_dicts(report)
        candidates.append(CandidateResult(
            architecture=arch, final_loss=float(result["history"]["loss"][-1]),
            guardrail_trustworthy=bool(report.trustworthy), guardrail_checks=checks,
            residual_value=next((c["value"] for c in checks if c["name"] == "pde_residual"), None), model=net))

    if not candidates:
        return AnalysisResult(provenance, None, [], rejected=True,
                              rejection_reason="no candidate architecture could be built and trained for this preset")

    best = min(candidates, key=rank_key)
    provenance.architecture = best.architecture
    provenance.architecture_config = {"hidden_dim": hidden_dim, "n_layers": n_layers}
    provenance.training_config = {"epochs": epochs, "n_collocation": n_collocation,
                                  "residual_threshold": residual_threshold}
    provenance.candidate_architectures_tried = [c.architecture for c in candidates]
    provenance.final_training_loss = best.final_loss
    provenance.guardrail_trustworthy = best.guardrail_trustworthy
    provenance.guardrail_checks = best.guardrail_checks
    n_pass = sum(bool(c["passed"]) for c in best.guardrail_checks)
    provenance.guardrail_summary = (
        f"{'TRUSTWORTHY' if best.guardrail_trustworthy else 'NOT TRUSTWORTHY'} "
        f"({len(best.guardrail_checks)} check(s) run, {n_pass} passed) -- picked from {len(candidates)} "
        f"candidate(s) in {time.time() - t0:.1f}s")
    return AnalysisResult(provenance, best, candidates)
