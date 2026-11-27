"""The release preflight: each exit criterion it can check, on a tree built for the test.

The point of the tool is that it reports **every** blocker at once (the version gate in
``scripts/check_release_version.py`` stops at the first), so the tests check each rule on its own and one tree that
breaks all of them together.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_preflight", ROOT / "scripts" / "release_preflight.py")
preflight_tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = preflight_tool          # dataclasses look the module up while the class is created
spec.loader.exec_module(preflight_tool)


def _tree(tmp_path: Path, version: str = "0.9.0", unreleased: str = "", heading: str | None = None,
          fragments: int = 0) -> Path:
    """A minimal repository: the four files the preflight reads, and as many fragments as asked for."""
    (tmp_path / "pinneapple").mkdir(parents=True)
    (tmp_path / "changelog.d").mkdir()
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text(f'version: "{version}"\n', encoding="utf-8")
    (tmp_path / "pinneapple" / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    body = "## [Unreleased]\n" + unreleased + "\n"
    if heading:
        body += f"## [{heading}] - 2026-11-01\n\n- shipped\n"
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n" + body, encoding="utf-8")
    for i in range(fragments):
        (tmp_path / "changelog.d" / f"{i:03d}-coisa.added.md").write_text("- algo\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    def build(**kwargs) -> Path:
        root = _tree(tmp_path, **kwargs)
        monkeypatch.setattr(preflight_tool, "ROOT", root)
        return root
    return build


def test_a_clean_tree_has_nothing_blocking(tree):
    tree(version="0.9.0", heading="0.9.0")
    report = preflight_tool.preflight("0.9.0")
    assert report.ok, [b.what for b in report.blockers]
    assert any("já carrega '0.9.0'" in note for note in report.notes)


def test_the_version_files_have_to_agree_with_the_release(tree):
    tree(version="0.8.0", heading="0.8.0")
    report = preflight_tool.preflight("0.9.0")
    assert any(b.kind == "version" for b in report.blockers)


def test_an_unreleased_section_is_a_blocker_and_an_empty_one_is_not(tree):
    tree(version="0.9.0", heading="0.9.0", unreleased="### Added\n- uma entrada por lançar")
    assert any("por lançar" in b.what for b in preflight_tool.preflight("0.9.0").blockers)


def test_fragments_left_in_changelog_d_are_a_blocker(tree):
    tree(version="0.9.0", heading="0.9.0", fragments=3)
    blockers = preflight_tool.preflight("0.9.0").blockers
    assert any("3 fragmento(s)" in b.what for b in blockers)


def test_a_missing_dated_heading_is_a_blocker(tree):
    tree(version="0.9.0", heading=None)
    assert any("cabeçalho datado" in b.what for b in preflight_tool.preflight("0.9.0").blockers)


def test_every_blocker_is_reported_at_once(tree):
    """The reason this tool exists: one run shows all of them, not the first one."""
    tree(version="0.8.0", heading=None, unreleased="- algo", fragments=2)
    kinds = {b.kind for b in preflight_tool.preflight("0.9.0").blockers}
    assert {"version", "changelog"} <= kinds


def test_without_github_the_run_says_so_instead_of_passing_silently(tree, monkeypatch):
    tree(version="0.9.0", heading="0.9.0")
    monkeypatch.setattr(preflight_tool, "_gh", lambda *a, **k: None)
    report = preflight_tool.preflight("0.9.0", with_github=True)
    assert any("GitHub não respondeu" in note for note in report.notes)
    assert report.ok, "a milestone that could not be read is a note, not a blocker invented from nothing"


def test_an_open_milestone_blocks_with_the_items_named():
    """The exit criterion is explicit: every item closed, or moved with the reason written in the item."""
    data = {"title": "v0.9.0", "state": "open", "open": 2, "closed": 14,
            "open_items": ["#422 Release v0.9.0", "#418 Synthetic datasets"]}
    original = preflight_tool.milestone
    preflight_tool.milestone = lambda version: data
    try:
        report = preflight_tool.preflight("0.9.0", with_github=True)
    finally:
        preflight_tool.milestone = original
    blocker = next(b for b in report.blockers if b.kind == "milestone")
    assert "#422" in blocker.what and "mov" in blocker.fix


def test_the_render_is_a_table_with_a_row_per_blocker(tree):
    tree(version="0.9.0", heading=None)
    text = preflight_tool.render([preflight_tool.preflight("0.9.0")])
    assert text.startswith("# Preflight de release") and "| bloqueio |" in text


def test_the_real_repository_is_readable_by_the_tool():
    """The tool has to work on this repository as it is: three versions, three files, one CHANGELOG."""
    found = preflight_tool.versions()
    assert set(found) == {"pyproject.toml", "CITATION.cff", "pinneapple/__init__.py"}
    assert all(v != "?" for v in found.values())
    assert preflight_tool.released_heading(found["pyproject.toml"]) is not None
