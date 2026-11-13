"""The dataset catalogue: priorities, what is readable, and what is not claimed about licences."""
from __future__ import annotations

import json

import pytest

from pinneapple_data.datasets import catalog


def test_every_entry_has_the_fields_a_decision_needs():
    for entry in catalog.CATALOG.values():
        assert entry.id and entry.title and entry.domain and entry.formats and entry.homepage
        assert entry.license
        assert entry.priority in ("P0", "P1", "P2")
        assert entry.access in ("open", "registration", "restricted")


def test_the_p0_set_is_the_one_the_roadmap_names():
    assert {e.id for e in catalog.entries(priority="P0")} == {"pdebench", "the_well", "cfdbench", "spe10", "nasa_tmr"}


def test_entries_are_sorted_by_priority_then_id():
    priorities = [e.priority for e in catalog.entries()]
    assert priorities == sorted(priorities)
    for priority in ("P0", "P1", "P2"):
        ids = [e.id for e in catalog.entries(priority=priority)]
        assert ids == sorted(ids)


def test_only_the_adapters_that_exist_are_readable():
    assert {e.id for e in catalog.implemented()} == {"pdebench", "the_well"}
    assert "cfdbench" in {e.id for e in catalog.missing("P0")}


def test_redistribution_is_only_claimed_where_the_licence_was_checked():
    assert catalog.redistributable("pdebench") is True
    assert catalog.redistributable("the_well") is False        # per-scenario data terms, not verified here
    assert catalog.redistributable("jhtdb") is False


def test_get_reports_what_it_knows():
    with pytest.raises(KeyError, match="not catalogued"):
        catalog.get("no_such_dataset")


def test_to_json_is_the_api_payload():
    payload = catalog.to_json()
    assert json.loads(json.dumps(payload)) == payload
    pdebench = next(e for e in payload if e["id"] == "pdebench")
    assert pdebench["readable"] is True and pdebench["license_verified"] is True
    assert isinstance(pdebench["formats"], list)
