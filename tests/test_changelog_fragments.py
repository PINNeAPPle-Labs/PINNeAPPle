"""Changelog fragments (scripts/changelog.py): names are checked, and a release folds the Unreleased entries and the
fragments under one dated heading without losing or reordering any entry."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("changelog_tool", ROOT / "scripts" / "changelog.py")
changelog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(changelog)

BASE = """# Changelog

## [Unreleased]

### Added
- old added entry

### Fixed
- old fix

## [0.6.0] - 2026-01-01

### Added
- shipped
"""


def test_every_fragment_in_the_repo_is_well_named():
    assert changelog.check() == []


@pytest.mark.parametrize("name", ["x.md", "x.feature.md", ".added.md", "x.added.txt"])
def test_bad_names_are_reported(tmp_path, name):
    (tmp_path / name).write_text("- entry\n")
    if name.endswith(".md"):
        assert changelog.check(tmp_path)
    else:
        assert changelog.check(tmp_path) == []      # only .md files are fragments


def test_empty_fragment_is_reported(tmp_path):
    (tmp_path / "a.added.md").write_text("\n")
    assert "empty" in changelog.check(tmp_path)[0]


def test_render_merges_fragments_under_their_headings_in_release_order(tmp_path):
    (tmp_path / "b.added.md").write_text("- new added\n  - detail\n")
    (tmp_path / "a.security.md").write_text("- a CVE fix\n")
    out = changelog.render(BASE, tmp_path)
    assert out.index("### Security") < out.index("### Added") < out.index("### Fixed")
    added = out.split("### Added")[1].split("###")[0]
    assert "- old added entry" in added and "- new added\n  - detail" in added
    assert "shipped" not in out


def test_release_moves_everything_and_deletes_fragments(tmp_path, monkeypatch):
    log = tmp_path / "CHANGELOG.md"
    log.write_text(BASE)
    frags = tmp_path / "changelog.d"
    frags.mkdir()
    (frags / "x.fixed.md").write_text("- new fix\n")
    (frags / "README.md").write_text("not a fragment")
    monkeypatch.setattr(changelog, "CHANGELOG", log)
    changelog.release("0.7.0", "2026-11-01", frags)
    text = log.read_text()
    assert text.index("## [Unreleased]") < text.index("## [0.7.0] - 2026-11-01") < text.index("## [0.6.0]")
    unreleased = text.split("## [Unreleased]")[1].split("## [0.7.0]")[0]
    assert unreleased.strip() == ""
    released = text.split("## [0.7.0] - 2026-11-01")[1].split("## [0.6.0]")[0]
    for entry in ("- old added entry", "- old fix", "- new fix"):
        assert entry in released
    assert [p.name for p in frags.iterdir()] == ["README.md"]
