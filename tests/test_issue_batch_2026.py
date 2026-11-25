"""Offline unit tests for the mirrored day batch parser (not executed)."""
from pathlib import Path

import pytest

from scripts.issue_batch_2026 import expand_ids, parse_day, classify, REPOS


def test_first_day_has_exactly_91_unique_issues():
    path = Path(__file__).resolve().parents[1] / "docs" / "EXECUTION_PLAN_2026.md"
    result = parse_day(path, "2026-11-23")
    assert len(result[REPOS[0]]) == 69
    assert len(result[REPOS[1]]) == 22
    assert 31 in result[REPOS[0]]
    assert 105 in result[REPOS[1]]


def test_expand_ranges_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        expand_ids("1-3,2")


def test_labels_take_precedence_over_guessing():
    issue = {"title": "test", "body": "blocked by #42",
             "labels": [{"name": "priority: P1"}, {"name": "effort: M"}]}
    result = classify(issue)
    assert result["priority"] == "P1"
    assert result["complexity"] == "M"
    assert result["dependencies_mentioned"] == ["42"]


def test_unknown_labels_remain_unclassified():
    result = classify({"title": "Unknown", "labels": []})
    assert result["priority"] == "unclassified"
    assert result["complexity"] == "unclassified"
