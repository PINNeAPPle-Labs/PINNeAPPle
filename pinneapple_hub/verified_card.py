"""Model cards whose ``validation_metrics`` come only from a guardrail run against the model, refused otherwise.

:func:`certify_model` builds (or loads) a model for a registered preset, trains it when no weights are given,
runs :class:`pinneapple_llm.PhysicsGuardrail` on it and builds a :class:`~pinneapple_hub.model_card.ModelCard`
with :func:`card_from_guardrail`: every numeric check value goes into ``validation_metrics`` under the check's
name, every pass/fail-only check as ``<name>_passed`` (1.0 / 0.0), and ``reference_source`` says what was run on
how many fresh points. A card that fails ``ModelCard.validate()`` (for example because nothing computable ran)
raises :class:`CardRefused` instead of being returned, the same rule ``push_to_hub`` applies.

Re-auditing an existing card can only re-run the schema check: the weights and the exact preset kwargs behind a
card's numbers do not travel in the card JSON, so its physics claims cannot be re-executed from it alone.

Moved here from the PINNeAPPle-apps ``physics_verified_model_cards`` product (GovernedCard); the HTTP service is
``apps/governed_card``.
"""
from __future__ import annotations

import io
from typing import Any

from .model_card import ModelCard

__all__ = ["CardRefused", "validation_metrics_from_report", "card_from_guardrail", "certify_model"]


class CardRefused(ValueError):
    """The card built from a guardrail run did not pass ``ModelCard.validate()``; ``.problems`` lists why."""

    def __init__(self, problems: list[str]):
        super().__init__("model card refused: " + "; ".join(problems))
        self.problems = problems


def validation_metrics_from_report(report: Any) -> dict[str, float]:
    """``{check name: value}`` for numeric checks, ``{<name>_passed: 1.0 | 0.0}`` for pass/fail-only checks."""
    out: dict[str, float] = {}
    for c in report.checks:
        if c.value is not None:
            out[c.name] = float(c.value)
        else:
            out[f"{c.name}_passed"] = 1.0 if c.passed else 0.0
    return out


def card_from_guardrail(spec: Any, report: Any, *, name: str, preset: str, preset_kwargs: dict | None = None,
                        architecture: str, architecture_config: dict, training_config: dict,
                        n_check_points: int, reference_benchmark: str | None = None, description: str = "",
                        citation: str = "", license: str = "apache-2.0", tags: list[str] | None = None) -> ModelCard:
    """A ``ModelCard`` for a model of preset ``preset`` from its guardrail ``report`` (not validated here)."""
    summaries = [f"{c.name} (threshold={c.threshold})" if c.threshold is not None else c.name for c in report.checks]
    ref = (f"pinneapple_llm.PhysicsGuardrail.check() against pinneapple_physics preset {preset!r} "
           f"(pde_kind={spec.pde.kind!r}) on {n_check_points} fresh collocation points; checks run: "
           f"{', '.join(summaries) or '(none)'}")
    if reference_benchmark:
        ref += f"; reference data: pinneapple_pdb benchmark {reference_benchmark!r}"
    tags = [t for t in (tags or []) if t]
    if "physics-verified" not in tags:
        tags.append("physics-verified")
    return ModelCard(
        name=name, description=description, architecture=architecture, architecture_config=dict(architecture_config),
        equations=[spec.pde.kind], domain_bounds={k: list(v) for k, v in spec.domain_bounds.items()},
        boundary_conditions={c.name: c.kind for c in spec.conditions if c.kind in ("dirichlet", "neumann", "robin")},
        initial_condition={c.name: c.kind for c in spec.conditions if c.kind == "initial"},
        training_config=dict(training_config),
        data_lineage={"preset": preset, "preset_kwargs": dict(preset_kwargs or {}), "pde_kind": spec.pde.kind,
                      "pde_params": dict(spec.pde.params or {})},
        validation_metrics=validation_metrics_from_report(report), reference_source=ref, citation=citation,
        license=license, tags=tags)


def certify_model(preset: str, name: str, *, preset_kwargs: dict | None = None, architecture: str = "modified_mlp",
                  architecture_config: dict | None = None, state_dict: bytes | dict | None = None,
                  train_epochs: int = 300, n_collocation: int = 1024, residual_threshold: float = 1e-2,
                  reference_benchmark: str | None = None, seed: int | None = None, **card_fields: Any) -> dict:
    """Train (or load ``state_dict``: a dict or ``torch.save`` bytes) a model of ``preset``, check it, build its card.

    Returns ``{"card": ModelCard, "trustworthy", "checks", "skipped", "trained_live", "final_training_loss"}``;
    raises :class:`CardRefused` when the card fails ``ModelCard.validate()``, ``KeyError`` for an unknown preset
    or architecture and ``TypeError`` / ``ValueError`` for kwargs or weights that do not fit.
    """
    import torch

    import pinneapple_llm as pl
    import pinneapple_neural.architectures  # noqa: F401  registers the zoo
    import pinneapple_physics as pp
    from pinneapple_neural.architectures.registry import ModelRegistry
    from pinneapple_physics.pde_environment.presets.tag_geometry import tag_solve_kwargs

    spec = pp.get_preset(preset, **(preset_kwargs or {}))
    config = {"in_dim": len(spec.coords), "out_dim": len(spec.fields), "hidden_dim": 32, "n_layers": 3}
    config.update(architecture_config or {})
    if seed is not None:
        torch.manual_seed(seed)
    model = ModelRegistry.build(architecture, **config)
    final_loss = None
    if state_dict is None:
        result = pp.solve_pde(spec, model, epochs=train_epochs, n_collocation=n_collocation,
                              **tag_solve_kwargs(preset, spec))
        final_loss = float(result["history"]["loss"][-1])
        training = {"trained_live": True, "epochs": train_epochs, "n_collocation": n_collocation,
                    "final_training_loss": final_loss}
    else:
        sd = torch.load(io.BytesIO(state_dict), map_location="cpu") if isinstance(state_dict, bytes) else state_dict
        try:
            model.load_state_dict(sd)
        except Exception as e:  # noqa: BLE001 -- a shape/key mismatch, reported as such
            raise ValueError(f"weights do not fit {architecture!r} with {config}: {e}") from e
        training = {"trained_live": False, "note": "weights supplied by the caller"}
    guard = pl.PhysicsGuardrail(spec, residual_threshold=residual_threshold)
    report = guard.check(model, **({"reference_benchmark": reference_benchmark} if reference_benchmark else {}))
    card = card_from_guardrail(spec, report, name=name, preset=preset, preset_kwargs=preset_kwargs,
                               architecture=architecture, architecture_config=config, training_config=training,
                               n_check_points=guard.n_check_points, reference_benchmark=reference_benchmark,
                               **card_fields)
    problems = card.validate()
    if problems:
        raise CardRefused(problems)
    return {"card": card, "trustworthy": bool(report.trustworthy),
            "checks": [{"name": c.name, "passed": bool(c.passed), "detail": c.detail, "value": c.value,
                        "threshold": c.threshold} for c in report.checks],
            "skipped": list(report.skipped), "trained_live": state_dict is None, "final_training_loss": final_loss}
