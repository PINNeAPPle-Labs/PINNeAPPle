"""Test shards: the wide suite blocks where it has been measured green (#90).

The wide suite ran only at night with one failure ceiling for everything, so a green part could not veto a pull
request and a regression in it was tolerated as long as another part improved. These tests hold the declaration that
replaces that: every test file belongs to exactly one shard, a blocking shard carries the measurement that put it
there, a non-blocking one carries a reason and a ceiling, and the page describes what the declaration says.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tier_shards as ts  # noqa: E402


def test_every_test_file_belongs_to_one_shard():
    problems = ts.check()
    assert problems == [], "\n".join(problems)


def test_a_test_file_outside_every_shard_is_reported():
    declaration = copy.deepcopy(ts.load())
    orphan = ts.all_test_files()[0]
    declaration["shards"] = [
        {**shard, "paths": [p for p in shard["paths"] if not any(rel == orphan for rel in ts.files_of(shard))]}
        for shard in declaration["shards"]
    ]
    assert any(orphan in p and "belongs to no shard" in p for p in ts.check(declaration)), ts.check(declaration)


def test_a_shard_cannot_block_without_a_measurement():
    declaration = copy.deepcopy(ts.load())
    victim = next(s for s in declaration["shards"] if s.get("blocking"))
    victim["measured"] = ""
    assert any(victim["id"] in p and "recorded measurement" in p for p in ts.check(declaration))


def test_a_blocking_shard_cannot_allow_a_failure():
    declaration = copy.deepcopy(ts.load())
    victim = next(s for s in declaration["shards"] if s.get("blocking"))
    victim["ceiling"] = 3
    assert any(victim["id"] in p and "ceiling 0" in p for p in ts.check(declaration))


def test_a_shard_that_does_not_block_has_to_say_why():
    declaration = copy.deepcopy(ts.load())
    quiet = next(s for s in declaration["shards"] if not s.get("blocking"))
    quiet["reason"] = "  "
    assert any(quiet["id"] in p and "no reason" in p for p in ts.check(declaration))


def test_a_path_that_matches_nothing_is_reported():
    declaration = copy.deepcopy(ts.load())
    declaration["shards"].append({"id": "ghost", "paths": ["tests/nowhere"], "blocking": False,
                                  "reason": "x", "ceiling": 0})
    assert any("tests/nowhere" in p for p in ts.check(declaration))


def test_every_shard_owns_at_least_one_file():
    for shard in ts.load()["shards"]:
        assert ts.files_of(shard), shard["id"]


def test_the_page_matches_the_declaration():
    declaration = ts.load()
    page = ts.docs(declaration)
    for shard in declaration["shards"]:
        assert f"`{shard['id']}`" in page
    on_disk = (ROOT / "docs" / "dev" / "test_shards.md").read_text(encoding="utf-8")
    assert on_disk == page, "regenerate with: python scripts/tier_shards.py --docs"


def test_promoting_refuses_a_shard_that_fails(tmp_path):
    """A promotion is a measurement: a shard that fails is not made blocking, whatever the declaration says."""
    declaration = copy.deepcopy(ts.load())
    declaration["shards"].append({"id": "red", "paths": ["tests/tier_shards.json"], "blocking": False,
                                  "reason": "not a real shard, used here as one that cannot pass", "ceiling": 1})
    # a directory that exists but holds no pytest file of its own: pytest exits non-zero, so the promotion refuses
    assert ts.promote(declaration, "red") != 0
    assert not next(s for s in declaration["shards"] if s["id"] == "red").get("blocking")


def test_running_the_blocking_shards_is_what_the_job_does():
    """A shard that cannot run fails the tier instead of passing silently, which is the whole point of blocking."""
    # a small file: pointing this at this file would run this test again, which runs the shard again, ...
    green = {"schema": ts.SCHEMA,
             "shards": [{"id": "dev-env", "paths": ["tests/test_dev_environment.py"], "blocking": True, "ceiling": 0,
                         "measured": "2026-11-21: this file, by hand"}]}
    assert ts.run_blocking(green) == 0
    red = {"schema": ts.SCHEMA,
           "shards": [{"id": "nowhere", "paths": ["tests/does_not_exist"], "blocking": True, "ceiling": 0,
                       "measured": "2026-11-21: cannot run"}]}
    assert ts.run_blocking(red) == 1
