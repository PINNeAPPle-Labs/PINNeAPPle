"""The published model family (``pp.zoo.anchors``): every anchor points at a shipped, approved model and repeats
that model's card numbers -- no metric of its own, no anchor that drifts from what ships."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from pinneapple_hub import zoo  # noqa: E402
from pinneapple_hub.zoo import _artifacts as art  # noqa: E402
from pinneapple_hub.zoo import anchors  # noqa: E402


def test_every_anchor_resolves_to_shipped_models():
    ids = anchors.list()
    assert ids, "the family has no anchor"
    shipped = zoo.list()
    for anchor_id in ids:
        a = anchors.info(anchor_id)
        assert a.name and a.domain and a.problem and a.replaces
        assert a.models, f"{anchor_id} has no model"
        for name in a.models:
            assert name in shipped, f"{anchor_id} points at {name!r}, which the zoo does not ship"


def test_the_family_is_publishable():
    assert anchors.problems() == []


def test_metrics_are_the_card_metrics():
    """The family must not carry a number the card does not: that is the whole point of publishing it."""
    manifest = art.manifest()["models"]
    for anchor_id in anchors.list():
        a = anchors.info(anchor_id)
        for member in a.members:
            card = zoo.info(member["model"])
            assert member["trust"] == card.trust_decision
            for metric, value in member["metrics"].items():
                assert value == pytest.approx(card.validation_metrics[metric])
            # the guard metric is one of the card's own metrics, not a new one invented for the page
            assert a.guard_metric in member["metrics"]
            assert set(member["metrics"]) <= set(manifest[member["model"]].get("guard_metrics", []))


def test_catalogue_lists_every_anchor_with_its_number():
    page = anchors.catalogue()
    assert anchors.FAMILY["name"] in page
    assert anchors.FAMILY["promise"] in page
    for anchor_id in anchors.list():
        a = anchors.info(anchor_id)
        assert a.name in page
        for member in a.members:
            assert member["model"] in page
            for metric in member["metrics"]:
                assert metric.split("/")[-1] in page


def test_card_says_where_the_model_stops():
    for anchor_id in anchors.list():
        page = anchors.card(anchor_id)
        a = anchors.info(anchor_id)
        assert a.name in page and a.replaces in page
        for member in a.members:
            assert member["trust"] in page
            if member["limitations"]:
                assert member["limitations"][0] in page


def test_public_payload_is_json_ready():
    payload = anchors.public()
    assert set(payload) == {"family", "anchors", "problems"}
    assert payload["problems"] == []
    round_trip = json.loads(anchors.dumps())
    assert round_trip == json.loads(json.dumps(payload))


def test_unknown_anchor_is_an_error_not_a_warning():
    with pytest.raises(KeyError):
        anchors.info("anchor-does-not-exist")
    with pytest.raises(KeyError):
        anchors.card("anchor-does-not-exist")
