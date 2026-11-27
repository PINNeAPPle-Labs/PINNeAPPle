"""A model card generated from an experiment record and validated in one command (#40).

The hub refuses a card without validation metrics; before this, producing one meant typing the numbers in. These
tests hold that the generated card carries what the record actually measured — metrics, cost, validity domain — and
that a record which cannot support a card is refused for the reason that describes it, not with a stub.
"""
from __future__ import annotations

import json

import pytest

import pinneapple as pp
from pinneapple_check.cli import main as pinneapple_main
from pinneapple_hub.experiment_card import card_from_record, generate_and_validate

SMALL_PINN = {"epochs": 10, "n_collocation": 64, "hidden_dim": 8, "n_layers": 2}


def record(tmp_path, *, reference="analytic", name="run"):
    directory = tmp_path / name
    experiment = pp.Experiment("burgers_1d", method="pinn", options=dict(SMALL_PINN), seed=3,
                               reference=reference, n_eval=128)
    experiment.save(str(directory))
    experiment.run().save(str(directory))
    return directory


def test_the_card_carries_the_recorded_metrics_cost_and_validity(tmp_path):
    generated = generate_and_validate(record(tmp_path))
    assert generated.ok, generated.problems
    card = generated.card
    assert set(card.validation_metrics) >= {"relative_l2.u", "rmse.u"}
    assert all(isinstance(v, float) for v in card.validation_metrics.values())
    assert card.cost["wall_time_s"] > 0 and card.cost["n_eval_points"] == 128
    assert card.cost["epochs"] == 10, "the training options are part of the cost"
    assert card.validity_range["x (bounds)"] == [-1.0, 1.0]
    assert "analytic" in card.validity_range["compared against"]
    assert card.reference_source and "analytic" in card.reference_source


def test_the_provenance_is_complete_enough_for_the_hub_to_check(tmp_path):
    generated = generate_and_validate(record(tmp_path))
    prov = generated.card.provenance
    for key in ("solver", "seed", "data_hash", "commit", "library_version"):
        assert prov.get(key) not in (None, ""), key
    assert prov["seed"] == 3


def test_the_limitations_say_what_the_record_does_not_cover(tmp_path):
    card = card_from_record(record(tmp_path))
    text = " ".join(card.known_limitations)
    assert "outside them the error is not measured" in text
    assert "one run" in text


def test_a_record_without_a_reference_is_refused_before_a_card_exists(tmp_path):
    directory = tmp_path / "sem-referencia"
    experiment = pp.Experiment("burgers_1d", method="pinn", options=dict(SMALL_PINN), seed=1)
    experiment.save(str(directory))
    experiment.run().save(str(directory))
    with pytest.raises(ValueError, match="no metrics"):
        card_from_record(directory)


def test_a_folder_that_is_not_a_record_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="experiment.json"):
        card_from_record(tmp_path)


def test_the_card_is_validated_with_cost_and_validity_required(tmp_path):
    """A hand-built card that omits them is refused by the same validation the generator runs."""
    card = card_from_record(record(tmp_path))
    card.cost = {}
    card.validity_range = {}
    problems = card.validate(require_cost=True, require_validity=True)
    assert any("cost is empty" in p for p in problems)
    assert any("validity_range is empty" in p for p in problems)


def test_a_cost_without_a_measured_time_is_refused(tmp_path):
    card = card_from_record(record(tmp_path))
    card.cost.pop("wall_time_s")
    assert any("wall_time_s" in p for p in card.validate(require_cost=True))


def test_the_markdown_shows_the_cost_and_the_reasons_when_refused(tmp_path):
    generated = generate_and_validate(record(tmp_path))
    assert "## Cost" in generated.to_markdown() and "wall_time_s" in generated.to_markdown()
    generated.problems = ["cost is empty"]
    assert "**Refused:**" in generated.to_markdown()


def test_the_one_command_writes_the_card_and_exits_zero(tmp_path, capsys):
    directory = record(tmp_path)
    out, md = tmp_path / "card.json", tmp_path / "card.md"
    code = pinneapple_main(["model-card", str(directory), "-o", str(out), "--markdown", str(md)])
    assert code == 0, capsys.readouterr().err
    payload = json.loads(out.read_text())
    assert payload["validation_metrics"] and payload["cost"]["wall_time_s"] > 0
    assert "## Cost" in md.read_text()


def test_the_command_exits_two_when_there_is_no_record(tmp_path, capsys):
    assert pinneapple_main(["model-card", str(tmp_path / "nada")]) == 2
    assert "experiment.json" in capsys.readouterr().err


def test_the_command_exits_one_and_names_the_refusal(tmp_path, capsys):
    directory = tmp_path / "sem-referencia"
    experiment = pp.Experiment("burgers_1d", method="pinn", options=dict(SMALL_PINN), seed=1)
    experiment.save(str(directory))
    experiment.run().save(str(directory))
    assert pinneapple_main(["model-card", str(directory)]) == 1
    assert "no metrics" in capsys.readouterr().err
