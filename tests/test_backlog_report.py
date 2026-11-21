"""The repository's own backlog page (#1497 aftermath): generated from a snapshot, and checked without the network.

The page is a list, so the thing worth testing is that the list *adds up*: every issue appears once, the counts in the
header match the rows, a closed issue is a checked box, and a page that drifted from its snapshot is refused. The
snapshot is the source of truth here, so the tests never touch the network.
"""
from __future__ import annotations

import json

import pytest

from scripts import backlog_report as bl


def snapshot(issues):
    return {"schema": bl.SCHEMA, "repo": bl.REPO, "fetched": "2026-11-21", "issues": [
        {"number": n, "title": t, "state": s, "url": f"https://example/{n}", "scope": sc, "area": a,
         "priority": p, "effort": "effort: M", "closed_at": ""}
        for n, t, s, sc, a, p in issues], "unscoped_seen": 3}


FIXTURE = snapshot([
    (1, "first", "closed", "escopo: 2026", "area: core-api", "priority: P0"),
    (2, "second", "open", "escopo: 2026", "area: core-api", "priority: P1"),
    (3, "third", "open", "escopo: 2026", "area: solvers", ""),
    (4, "fourth", "open", "escopo: futuro", "area: core-api", "priority: P2"),
    (5, "fifth", "open", "escopo: 2026", "", ""),
])


def test_the_counts_are_the_rows_it_lists():
    c = bl.counts(FIXTURE)
    assert c["escopo: 2026"] == {"total": 4, "open": 3, "closed": 1}
    assert c["escopo: futuro"] == {"total": 1, "open": 1, "closed": 0}
    page = bl.render(FIXTURE)
    assert "| This year (`escopo: 2026`) | 4 | 3 | 1 |" in page
    assert "| Next cycle (`escopo: futuro`) | 1 | 1 | 0 |" in page


def test_a_closed_issue_is_a_checked_box_and_an_open_one_is_not():
    page = bl.render(FIXTURE)
    assert "- [x] [#1](https://example/1) first (P0 M)" in page
    assert "- [ ] [#2](https://example/2) second (P1 M)" in page


def test_every_issue_appears_exactly_once():
    page = bl.render(FIXTURE)
    for n in (1, 2, 3, 4, 5):
        assert page.count(f"[#{n}](") == 1, n


def test_an_issue_without_an_area_gets_its_own_group_instead_of_disappearing():
    page = bl.render(FIXTURE)
    assert "### (no area label) — 1 issues (1 open)" in page
    assert "[#5](https://example/5) fifth" in page


def test_the_page_is_deterministic():
    assert bl.render(FIXTURE) == bl.render(json.loads(json.dumps(FIXTURE)))


def _write(root, snap, page=None):
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "backlog.json").write_text(json.dumps(snap), encoding="utf-8")
    (root / "docs" / "backlog.md").write_text(bl.render(snap) if page is None else page, encoding="utf-8")


def test_check_passes_on_a_page_that_matches_its_snapshot(tmp_path):
    _write(tmp_path, FIXTURE)
    assert bl.check(tmp_path) == []


def test_check_catches_a_page_that_drifted(tmp_path):
    _write(tmp_path, FIXTURE, page=bl.render(FIXTURE).replace("[#2]", "[#20]"))
    problems = bl.check(tmp_path)
    assert any("is not what the snapshot renders" in p for p in problems)


def test_check_catches_a_snapshot_with_a_duplicate_or_an_unscoped_issue(tmp_path):
    duplicated = snapshot([(1, "a", "open", "escopo: 2026", "area: core-api", ""),
                           (1, "a again", "open", "escopo: 2026", "area: core-api", "")])
    _write(tmp_path, duplicated)
    assert any("more than once" in p for p in bl.check(tmp_path))
    unscoped = json.loads(json.dumps(FIXTURE))
    unscoped["issues"][0]["scope"] = "escopo: outro"
    _write(tmp_path, unscoped)
    assert any("no scope label" in p for p in bl.check(tmp_path))


def test_a_missing_snapshot_says_what_to_run(tmp_path):
    problems = bl.check(tmp_path)
    assert problems and "--fetch" in problems[0]


def test_the_committed_page_is_the_one_the_snapshot_renders():
    """The repository's own pair, checked here as well as in CI so a local edit is caught by the test suite too."""
    assert bl.check(bl.ROOT) == []
