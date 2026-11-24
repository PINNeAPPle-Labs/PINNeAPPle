"""The lint-debt gate: it records what is there and refuses what is new (#51).

The comparison is the part worth testing without running ruff over the whole tree: the gate has to be per (file, rule),
so that removing debt in one module cannot buy new debt in another.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import lint_debt  # noqa: E402


def test_a_pair_above_the_baseline_is_new_debt():
    worse, better = lint_debt.compare({"a.py": {"UP006": 2}}, {"a.py": {"UP006": 3}})
    assert worse == [("a.py", "UP006", 2, 3)]
    assert better == []


def test_a_pair_below_the_baseline_is_the_ratchet():
    worse, better = lint_debt.compare({"a.py": {"I001": 5}}, {"a.py": {"I001": 2}})
    assert worse == []
    assert better == [("a.py", "I001", 5, 2)]


def test_removing_debt_elsewhere_does_not_pay_for_new_debt_here():
    """The total is unchanged, and the gate still fails: that is the point of counting per pair."""
    baseline = {"a.py": {"UP006": 4}, "b.py": {"UP006": 4}}
    current = {"a.py": {"UP006": 9}, "b.py": {"UP006": 0}}
    worse, better = lint_debt.compare(baseline, current)
    assert worse == [("a.py", "UP006", 4, 9)]
    assert better == [("b.py", "UP006", 4, 0)]


def test_a_whole_new_file_is_new_debt_and_a_removed_one_is_a_ratchet():
    worse, better = lint_debt.compare({"a.py": {"F401": 1}}, {"b.py": {"F401": 1}})
    assert worse == [("b.py", "F401", 0, 1)]
    assert better == [("a.py", "F401", 1, 0)]


def test_fixing_a_rule_that_was_not_there_invents_nothing():
    worse, better = lint_debt.compare({"a.py": {"F401": 1}}, {"a.py": {"F401": 1}})
    assert (worse, better) == ([], [])


def test_a_baseline_of_another_schema_is_refused(tmp_path):
    path = tmp_path / "lint_debt.json"
    path.write_text(json.dumps({"schema": "something-else/1", "files": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="schema|something-else"):
        lint_debt.load(path)


def test_a_missing_baseline_says_how_to_make_one(tmp_path):
    with pytest.raises(SystemExit, match="--record"):
        lint_debt.load(tmp_path / "not-there.json")


def test_the_committed_baseline_adds_up():
    """The file is committed; a hand edit that leaves the totals disagreeing is caught here."""
    payload = json.loads((ROOT / "docs" / "dev" / "lint_debt.json").read_text(encoding="utf-8"))
    assert payload["schema"] == lint_debt.SCHEMA
    counted = sum(sum(rules.values()) for rules in payload["files"].values())
    assert payload["total"] == counted
    assert payload["by_rule"] == lint_debt.by_rule(payload["files"])
