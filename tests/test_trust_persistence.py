"""TrustReport persisted in ModelCard and ModelStore; a REJECT blocks publication unless overridden (#157, D2)."""
import pytest
import torch.nn as nn

from pinneapple_analysis.trust.trust_report import TrustReport, checagem_booleana, checagem_erro_relativo
from pinneapple_hub.model_card import ModelCard
from pinneapple_registry.model_store import ModelStore


def _report(ok: bool) -> TrustReport:
    return TrustReport([checagem_erro_relativo("accuracy", 0.01 if ok else 0.5, 0.02, critica=True),
                        checagem_booleana("mass_conservation", True)], model="m")


def _card() -> ModelCard:
    return ModelCard(name="m", architecture="mlp", validation_metrics={"rel_l2": 0.01}, reference_source="analytic")


def test_report_round_trips_through_its_dict():
    r = _report(False)
    back = TrustReport.from_dict(r.to_dict())
    assert back.decisao == r.decisao == "REJECT" and back.motivos == r.motivos


def test_model_card_holds_the_report_and_blocks_a_reject(tmp_path):
    card = _card()
    card.attach_trust_report(_report(True))
    assert card.validate() == [] and card.trust_decision == "APPROVED" and "## Trust" in card.to_markdown()

    card.attach_trust_report(_report(False))
    assert any("REJECT" in p for p in card.validate())
    with pytest.raises(ValueError):
        card.override_trust("  ")
    card.override_trust("known bias, accepted for internal demo", by="reviewer")
    assert card.validate() == []
    path = tmp_path / "card.json"
    card.save(str(path))
    loaded = ModelCard.load(str(path))
    assert loaded.trust_decision == "REJECT" and loaded.trust_override["reason"].startswith("known bias")
    assert "Published despite REJECT" in loaded.to_markdown()
    loaded.attach_trust_report(_report(False))           # a new report clears the old override
    assert loaded.validate()


def test_model_store_blocks_publishing_a_reject_without_a_reason(tmp_path):
    store = ModelStore(str(tmp_path))
    v = store.save("p", nn.Linear(1, 1), trust_report=_report(False))
    assert store.trust_report("p", v)["decisao"] == "REJECT"
    store.promote("p", v, "archived")                   # not a publication
    with pytest.raises(PermissionError, match="REJECT"):
        store.promote("p", v, "production")
    with pytest.raises(PermissionError):
        store.save("p", nn.Linear(1, 1), stage="staging", trust_report=_report(False))
    store.promote("p", v, "production", override_reason="validated by hand against wind tunnel data")
    meta = store.metadata("p", v)
    assert meta["stage"] == "production" and meta["trust_override"]["reason"].startswith("validated")

    store.set_trust_report("p", v, _report(True))
    assert "trust_override" not in store.metadata("p", v)
    store.promote("p", v, "staging")
