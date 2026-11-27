"""The page chrome of the twin3d viewer and of the apps takes its colours from the house theme (#65).

``theme.json`` being one file is half of the issue; the other half is that nobody draws a colour of its own next to
it. The 2-D figures and the two viewers' *scales* are covered by ``tests/test_house_theme_shared.py``. This file
covers the frame around them -- the background, panels, text, borders and the two status colours -- because that is
where a second palette hides: a colour that is copied once, looks right, and is then edited in one place only.

The checks are static: they read the page and the script that dresses it. A browser is not in the loop, so what is
asserted is the seam -- the page declares no colour, every variable it uses is written by the viewer, and every
value written comes from the theme file -- not the pixels that come out.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pinneapple_tools.visualization import theme

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pinneapple_twin3d/viewer/index.html"
VIEWER = ROOT / "pinneapple_twin3d/viewer/viewer.js"
APP_CSS = ROOT / "apps/_shared/static/shared/app.css"
APP_THEME_CSS = ROOT / theme.GENERATED["apps"]

HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def _page_vars(path: Path) -> set[str]:
    """The custom properties a stylesheet reads through ``var()``."""
    return set(re.findall(r"var\((--[a-z0-9-]+)", path.read_text(encoding="utf-8")))


def _page_vars_written() -> dict[str, str]:
    """The variables ``pageVars`` sets and the expression each is set from."""
    text = VIEWER.read_text(encoding="utf-8")
    start = text.index("function pageVars(")
    body = text[start:text.index("\n}", start)]
    return dict(re.findall(r'"(--[a-z0-9-]+)":\s*([^,\n]+)', body))


# ---------------------------------------------------------------------------
# the twin3d page
# ---------------------------------------------------------------------------

def test_the_page_declares_no_colour_of_its_own():
    """A hex value in the page is a colour edited in one place only; the theme is the place."""
    assert HEX.findall(PAGE.read_text(encoding="utf-8")) == []


def test_every_variable_the_page_uses_is_written_by_the_viewer():
    """An unwritten ``var()`` is a colour that silently stays at the browser default."""
    assert _page_vars(PAGE) - set(_page_vars_written()) == set()


@pytest.mark.parametrize("name", ["--bg", "--panel", "--text", "--muted", "--line", "--accent", "--alarm", "--ok",
                                  "--viewport"])
def test_the_variable_is_set_from_the_theme_file(name):
    """``s.`` is the surface of the theme, bound in the same function; a literal would read as a quoted colour."""
    source = _page_vars_written()[name]
    assert re.search(r"\b(?:s|THEME)\.", source), f"{name} is set from {source!r}, which is not the theme file"


def test_the_status_colours_are_the_brand_and_the_palette():
    """Alarm is the house ember and ok is the palette's green: the page does not name a red and a green of its own."""
    written = _page_vars_written()
    assert "brand.ember" in written["--alarm"]
    assert "categorical[2]" in written["--ok"]
    assert theme.palette()[2] == "#009E73"


def test_the_roles_the_page_asks_for_exist_in_the_theme_file():
    """A role renamed in ``theme.json`` has to be renamed here, or the viewer writes ``undefined``."""
    assert {"blue", "ember", "ember_lifted"} <= set(theme.brand())
    assert len(theme.palette()) >= 5
    for dark in (False, True):
        assert {"plot", "paper", "ink", "muted", "line"} <= set(theme.surface(dark))


# ---------------------------------------------------------------------------
# the apps' shared stylesheet
# ---------------------------------------------------------------------------

def test_the_apps_sheet_does_not_declare_the_theme_surfaces_again():
    """It used to: ``--line``, ``--muted`` and ``--panel`` were copies, and ``--line`` had already drifted."""
    text = APP_CSS.read_text(encoding="utf-8")
    for name in ("--line", "--muted", "--panel", "--plot", "--paper"):
        assert not re.search(rf"^\s*{name}:", text, re.MULTILINE), f"app.css declares {name} again"


def test_the_apps_sheet_aliases_the_generated_theme():
    text = APP_CSS.read_text(encoding="utf-8")
    assert "--bg: var(--plot);" in text
    assert "--panel: var(--paper);" in text
    assert "--brand: var(--house-blue);" in text
    generated = APP_THEME_CSS.read_text(encoding="utf-8")
    for name in ("--plot", "--paper", "--ink", "--muted", "--line"):
        assert f"{name}:" in generated, f"theme.css does not define {name}"


def test_the_apps_sheet_reads_the_theme_before_it_uses_it():
    """``@import`` has to stay the first statement, or the aliases resolve to nothing."""
    assert APP_CSS.read_text(encoding="utf-8").startswith('@import url("theme.css");')


def test_the_only_colours_left_in_the_apps_sheet_are_status_and_detail():
    """Status (--ok/--warn/--bad), paper and shadows are not the palette; a series colour here would be."""
    text = APP_CSS.read_text(encoding="utf-8")
    for colour in theme.palette():
        assert colour.lower() not in text.lower(), f"app.css hard-codes the palette colour {colour}"
