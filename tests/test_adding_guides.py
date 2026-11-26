"""The "how to add" guides: the page, the template and the test are one promise (#42).

A guide that names a template that has moved, or a test that no longer exists, is worse than no guide: it teaches a
contributor that the documentation is decorative. So this checks the promise mechanically -- every path the two
pages name exists, the preset template runs end to end (registers, solves, is described by its card), and the page
keeps the section the issue asks for: an outside contributor can add a preset without asking anything.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "dev" / "adding_a_preset.md"
INDEX = ROOT / "docs" / "dev" / "adding_to_pinneapple.md"
TEMPLATE = ROOT / "templates" / "36_add_a_preset.py"

#: The six extension points of #42, each of which the index has to name.
EXTENSION_POINTS = ("preset", "solver", "model", "benchmark", "adapter", "app")

PATH = re.compile(r"`((?:templates|tests|pinneapple_[a-z_]+|apps|scripts|docs)/[A-Za-z0-9_./-]+)`")


def _paths(page: Path) -> list[str]:
    return sorted(set(PATH.findall(page.read_text(encoding="utf-8"))))


@pytest.mark.parametrize("page", [GUIDE, INDEX])
def test_every_path_the_guide_names_exists(page):
    missing = [p for p in _paths(page) if not (ROOT / p).exists()]
    assert not missing, f"{page.name} names paths that do not exist: {missing}"


def test_the_index_covers_the_six_extension_points():
    text = INDEX.read_text(encoding="utf-8")
    for point in EXTENSION_POINTS:
        assert point in text.lower(), f"the index does not cover adding {point}"
    assert "adding_a_preset.md" in text


def test_the_preset_guide_has_the_steps_a_contributor_needs():
    text = GUIDE.read_text(encoding="utf-8")
    for heading in ("What a preset is", "The four steps", "The test that has to come with it", "Checklist"):
        assert heading in text, f"the guide has no {heading!r} section"
    # the four things the issue asks a guide to carry: the template, the change, the minimum test, the command
    assert "36_add_a_preset.py" in text and "register" in text
    assert "pytest" in text and "changelog" in text


def test_the_preset_template_runs_and_checks_itself(tmp_path):
    """The template is executable documentation: it registers, solves, and asserts before it prints."""
    run = subprocess.run([sys.executable, str(TEMPLATE)], capture_output=True, text=True, cwd=str(ROOT),
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"},
                         timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "registered" in run.stdout and "reference:" in run.stdout and "card:" in run.stdout


def test_the_template_is_not_a_registered_preset_of_the_library():
    """It is a template: importing it must not put its problem into everybody's registry."""
    import pinneapple as pp
    assert "plate_steady_conduction_template" not in pp.list_presets()
