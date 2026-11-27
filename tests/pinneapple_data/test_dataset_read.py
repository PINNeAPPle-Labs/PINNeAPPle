"""Choosing which catalogued archive a stage should use, and reading it into a bundle.

The catalogue says which archives exist and which have a reader; `readable` alone does not say that the reader
returns a bundle (the meshgraphnets one returns trajectories). The chooser is the part a pipeline actually calls, so
these tests pin the three things that make a chooser trustworthy: the rule is deterministic, a refusal says which
catalogue entry and why, and a non-commercial licence is dropped with its text quoted rather than guessed at.
"""
from __future__ import annotations

import pytest

from pinneapple_data.datasets import read as R


def test_the_choice_is_deterministic_and_says_which_rule_picked_it():
    first, second = R.choose("data"), R.choose("data")
    assert first["id"] == second["id"] and first["candidates"] == second["candidates"]
    assert first["why"] and "priority" in first["why"]
    assert first["id"] in first["candidates"]


def test_a_stage_whose_only_bundle_reader_is_non_commercial_is_refused_by_default():
    """The problem stage has FlowBench (a bundle reader) and nothing else readable — and FlowBench is CC-BY-NC."""
    refused = R.choose("problem")
    assert refused["id"] is None
    assert "flowbench" in {d["id"]: d for d in refused["dropped"]}
    permissive = R.choose("problem", commercial_ok=True)
    assert permissive["id"] == "flowbench"
    dropped = {d["id"]: d["why"] for d in permissive["dropped"]}
    assert "nasa_tmr" in dropped and dropped["nasa_tmr"], "the archive with no reader must say why it is out"


def test_a_reader_that_stops_at_trajectories_is_not_offered_as_a_bundle():
    result = R.choose("physics_ai")
    assert "deepmind_mgn" in {d["id"] for d in result["dropped"]}
    assert "deepmind_mgn" not in result["candidates"]
    with pytest.raises(ValueError, match="bundle"):
        R.read("deepmind_mgn", dataset="cylinder_flow")


def test_a_non_commercial_licence_is_dropped_and_quoted(monkeypatch):
    from pinneapple_data.datasets import catalog

    flowbench = catalog.CATALOG["flowbench"]
    assert R._non_commercial(flowbench.license), "FlowBench is CC-BY-NC; the check stopped seeing it"
    criticised = R.choose("physics_ai", commercial_ok=False)
    assert "flowbench" in {d["id"]: d for d in criticised["dropped"]}
    permissive = R.choose("physics_ai", commercial_ok=True)
    assert "flowbench" in permissive["candidates"], "commercial_ok did not admit the NC archive"


def test_prefer_wins_only_when_it_survived_the_filters():
    assert R.choose("data", prefer="pdebench")["id"] == "pdebench"
    refused = R.choose("data", prefer="airfrans")      # catalogued, no reader
    assert refused["id"] is None and "airfrans" in refused["why"]


def test_an_unknown_id_lists_what_can_be_read():
    with pytest.raises(KeyError, match="readable"):
        R.read("not_a_dataset")


def test_reading_needs_the_arguments_the_adapter_insists_on():
    with pytest.raises(TypeError, match="needs"):
        R.read("pdebench")                             # no source


def test_the_dispatch_passes_the_arguments_and_returns_what_the_reader_gave(monkeypatch):
    """The read path without a multi-gigabyte archive: what is under test is the dispatch, not the adapter."""
    sentinel = object()
    seen: dict = {}

    def fake_reader(**kw):
        seen.update(kw)
        return sentinel

    monkeypatch.setitem(R._table(), "pdebench", R.Choice("pdebench", fake_reader, needs=("source",)))
    out = R.read("pdebench", gates=False, source="https://example.invalid/Sod6.hdf5", index=3)
    assert out["bundle"] is sentinel and out["dataset"] == "pdebench"
    assert out["license"], "the licence has to travel with what was read"
    assert seen == {"source": "https://example.invalid/Sod6.hdf5", "index": 3}


def test_the_stages_that_can_be_read_from_a_catalogue_are_named():
    stages = R.readable_stages()
    assert "inverse" in stages and stages["inverse"] == ["spe10"]
    assert "deepmind_mgn" not in {d for ids in stages.values() for d in ids}
