"""An artifact written by another version: read it, migrate it, or say what to do (#225).

A result outlives the release that produced it. The rule these tests hold: the same schema reads, an older one reads
through the migration registered for it, and anything else is refused **with the way out** — which version wrote the
file and what to do next. A silent read of a file from a different major version would put a number from another
solver under this version's name.
"""
from __future__ import annotations

import json
import warnings
import zipfile

import numpy as np
import pytest

from pinneapple_core.artifact import (
    MIGRATIONS,
    SCHEMA,
    Artifact,
    UnsupportedSchema,
    compatibility,
    load,
    save,
    verify,
)


def _artifact(tmp_path, name="case"):
    art = Artifact(problem={"name": "p"}, units={"u": "m"}, metrics=[{"name": "rel_l2", "value": 0.01, "unit": ""}],
                   arrays={"u": np.linspace(0, 1, 5)})
    art.provenance = {"source": "test", "created_at": "2026-11-06T00:00:00+00:00",
                      "environment": {"python": "3"}, "git": {"commit": "0" * 40, "dirty": False}}
    return save(art, tmp_path / name)


def _rewrite(path, **changes):
    """Rewrite the manifest of an existing .ppx: the fixture is a real container, only the version block changes."""
    with zipfile.ZipFile(path) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(members.pop("manifest.json"))
    for key, value in changes.items():
        if key == "pinneapple":
            manifest.setdefault("environment", {}).setdefault("packages", {})["pinneapple"] = value
        else:
            manifest[key] = value
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_an_artifact_of_this_version_reads_with_nothing_to_say(tmp_path):
    art = load(_artifact(tmp_path))
    assert art.compatibility is not None and art.compatibility.level == "ok"


def test_one_written_by_an_older_release_reads_with_a_warning(tmp_path):
    path = _rewrite(_artifact(tmp_path), pinneapple="0.7.3")
    with pytest.warns(UserWarning, match="written by pinneapple 0.7.3"):
        art = load(path)
    assert art.compatibility.level == "warn" and art.compatibility.written_by == "0.7.3"
    assert art.metrics[0]["value"] == pytest.approx(0.01)


def test_a_different_major_version_is_refused_with_the_way_out(tmp_path):
    path = _rewrite(_artifact(tmp_path), pinneapple="1.0.0")
    with pytest.raises(UnsupportedSchema, match="major version changed"):
        load(path)


def test_an_older_schema_with_no_migration_names_the_version_and_what_to_do(tmp_path):
    path = _rewrite(_artifact(tmp_path), schema="pinneapple.ppx/0", pinneapple="0.6.0")
    with pytest.raises(UnsupportedSchema, match="no migration from it is registered"):
        load(path)


def test_an_older_schema_reads_through_the_registered_migration(tmp_path, monkeypatch):
    seen = {}

    def v0_to_v1(manifest, members):
        seen["schema"] = manifest["schema"]
        seen["members"] = sorted(members)
        manifest["schema"] = SCHEMA
        manifest["migrated_by_test"] = True
        return manifest, members

    monkeypatch.setitem(MIGRATIONS, "pinneapple.ppx/0", v0_to_v1)
    path = _rewrite(_artifact(tmp_path), schema="pinneapple.ppx/0", pinneapple="0.6.0")
    art = load(path)
    assert seen["schema"] == "pinneapple.ppx/0" and "u.npy" in " ".join(seen["members"])
    assert art.compatibility.level == "migrated" and "pinneapple.ppx/0" in art.compatibility.detail
    assert art.metrics[0]["value"] == pytest.approx(0.01)


def test_a_migration_that_does_not_move_forward_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(MIGRATIONS, "pinneapple.ppx/0", lambda m, d: (m, d))
    path = _rewrite(_artifact(tmp_path), schema="pinneapple.ppx/0")
    with pytest.raises(UnsupportedSchema, match="did not move"):
        load(path)


def test_a_newer_schema_is_refused_and_says_to_upgrade(tmp_path):
    path = _rewrite(_artifact(tmp_path), schema="pinneapple.ppx/2", pinneapple="0.9.0")
    with pytest.raises(UnsupportedSchema, match="newer pinneapple"):
        load(path)


def test_a_file_that_is_not_a_ppx_says_which_schema_it_declares(tmp_path):
    path = _rewrite(_artifact(tmp_path), schema="somebody-else/1")
    with pytest.raises(UnsupportedSchema, match="declares schema"):
        load(path)


def test_compatibility_is_a_value_the_caller_can_read(tmp_path):
    path = _rewrite(_artifact(tmp_path), pinneapple="0.7.3")
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    note = compatibility(manifest)
    assert note.ok and note.level == "warn" and "0.7.3" in note.detail


def test_verify_reports_an_unreadable_schema_instead_of_raising(tmp_path):
    path = _rewrite(_artifact(tmp_path), schema="pinneapple.ppx/9")
    report = verify(path)
    assert not report.passed
    assert report.failed[0]["name"] == "container" and "newer pinneapple" in report.failed[0]["detail"]


def test_the_warning_does_not_fire_for_the_current_version(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load(_artifact(tmp_path))
