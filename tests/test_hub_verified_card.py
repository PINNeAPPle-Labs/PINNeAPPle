"""pinneapple_hub.verified_card: a card's validation_metrics are the numbers of a guardrail run (reproduced by an
independent, identically seeded direct run), weights can be loaded instead of trained, and a card with nothing
computable is refused."""
from __future__ import annotations

import io

import pytest

torch = pytest.importorskip("torch")

import pinneapple_llm as pl  # noqa: E402
import pinneapple_neural.architectures  # noqa: E402,F401
import pinneapple_physics as pp  # noqa: E402
from pinneapple_hub.verified_card import (  # noqa: E402
    CardRefused,
    card_from_guardrail,
    certify_model,
    validation_metrics_from_report,
)
from pinneapple_neural.architectures.registry import ModelRegistry  # noqa: E402
from pinneapple_physics.pde_environment.presets.tag_geometry import tag_solve_kwargs  # noqa: E402

SEED, EPOCHS, NCOL = 1234, 60, 256


def _direct_metrics():
    spec = pp.get_preset("laplace_2d")
    torch.manual_seed(SEED)
    model = ModelRegistry.build("modified_mlp", in_dim=len(spec.coords), out_dim=len(spec.fields), hidden_dim=32,
                                n_layers=3)
    pp.solve_pde(spec, model, epochs=EPOCHS, n_collocation=NCOL, **tag_solve_kwargs("laplace_2d", spec))
    return validation_metrics_from_report(pl.PhysicsGuardrail(spec, residual_threshold=1e-2).check(model)), model


def test_card_metrics_are_the_guardrail_numbers():
    out = certify_model("laplace_2d", "laplace-card", train_epochs=EPOCHS, n_collocation=NCOL, seed=SEED)
    card = out["card"]
    direct, _ = _direct_metrics()
    assert card.validation_metrics == direct and "pde_residual" in direct
    assert card.validate() == [] and "physics-verified" in card.tags
    assert "laplace_2d" in card.reference_source and "4096 fresh collocation points" in card.reference_source
    assert out["trained_live"] and out["final_training_loss"] is not None
    assert out["trustworthy"] == all(c["passed"] for c in out["checks"])


def test_supplied_weights_are_checked_not_trained():
    _, model = _direct_metrics()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    out = certify_model("laplace_2d", "uploaded", state_dict=buf.getvalue())
    assert not out["trained_live"] and out["card"].training_config["trained_live"] is False
    with pytest.raises(ValueError):
        certify_model("laplace_2d", "wrong-shape", state_dict=buf.getvalue(), architecture_config={"hidden_dim": 8})
    with pytest.raises(KeyError):
        certify_model("not_a_preset", "x")


class _Report:
    checks: list = []


def test_a_card_with_nothing_computed_is_refused():
    spec = pp.get_preset("laplace_2d")
    card = card_from_guardrail(spec, _Report(), name="empty", preset="laplace_2d", architecture="modified_mlp",
                               architecture_config={}, training_config={}, n_check_points=0)
    problems = card.validate()
    assert any("validation_metrics is empty" in p for p in problems)
    assert str(CardRefused(problems)).startswith("model card refused")
