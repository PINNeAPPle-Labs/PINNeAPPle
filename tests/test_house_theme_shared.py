"""One theme file, read by the 2-D figures, the studio/apps viewer and the twin3d viewer (#65).

The issue's acceptance is a single theme file consumed by all three. A file that three renderers are *supposed* to
read is not the same as three renderers that read it, so these tests check the seam on both sides: what Python
resolves from ``theme.json``, and what the generated copies handed to the browsers actually contain. The colour-blind
check is here too: the palette is simulated under protanopia, deuteranopia and tritanopia, and the floor is a distance
in linear RGB, not a promise in a comment.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pinneapple_tools.visualization import style, theme
from scripts import build_theme

ROOT = Path(__file__).resolve().parents[1]
STUDIO_JS = ROOT / theme.GENERATED["studio"]
TWIN3D_JS = ROOT / theme.GENERATED["twin3d"]
APPS_CSS = ROOT / theme.GENERATED["apps"]

#: the worst pair of the palette under simulated vision is 0.127 (tritanopia); 0.10 is the floor a map step apart
#: sits at, so the test fails on a palette that collapses two colours rather than on a re-tune of one.
MIN_SEPARATION = 0.10


# ---------------------------------------------------------------------------
# the file itself
# ---------------------------------------------------------------------------

def test_theme_file_is_the_one_the_module_reads():
    on_disk = json.loads((ROOT / "pinneapple_tools/visualization/theme.json").read_text(encoding="utf-8"))
    assert theme.theme() == on_disk
    assert theme.THEME_PATH.name == "theme.json"


@pytest.mark.parametrize("key", ["brand", "surface", "categorical", "colormaps", "field_cmaps", "font", "figure"])
def test_theme_file_carries_every_section(key):
    assert key in theme.theme()


def test_brand_is_the_house_palette():
    """#262626 ink, #0081be blue, #a23a00 ember: the colour the site and the mark already use."""
    brand = theme.brand()
    assert brand["ink"] == "#262626"
    assert brand["blue"] == "#0081be"
    assert brand["ember"] == "#a23a00"


def test_the_figure_palette_is_okabe_ito_in_order():
    assert theme.palette()[0] == "#0072B2"
    assert len(theme.palette()) == 8
    assert theme.palette()[3] == "#CC79A7"


def test_surface_gives_a_plot_and_a_paper_for_each_mode():
    for dark in (False, True):
        s = theme.surface(dark)
        assert set(s) >= {"plot", "paper", "ink", "muted", "line"}
        assert s["plot"] != s["paper"] or not dark      # the dark plot and paper are different greys
    assert theme.surface(True)["plot"] == "#0b1220"
    assert theme.viewport(dark=True) == theme.surface(True)["plot"]


# ---------------------------------------------------------------------------
# consumer 1: the matplotlib/PyVista side
# ---------------------------------------------------------------------------

def test_style_constants_come_from_the_file():
    assert style.PALETTE == theme.palette()
    assert style.FIELD_CMAPS == theme.field_cmaps()
    assert list(style._FONT_STACK) == list(theme.font()["stack"])


@pytest.mark.parametrize("name", ["cfd_sequential", "cfd_diverging", "cfd_cyclic", "jet", "coolwarm", "magma", "err"])
def test_every_map_in_the_style_module_is_the_theme_map(name):
    """No figure draws a map with stops of its own: they are the file's, sampled identically."""
    ours = style.CMAPS[name]
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert ours(value)[:3] == pytest.approx(theme.colormap(name)(value)[:3])


def test_pyvista_draws_on_the_theme_surface():
    """PyVista's default background is the theme's dark plot colour, not a literal of its own."""
    from pinneapple_tools.visualization import pyvista_bridge

    assert pyvista_bridge.BACKGROUND == theme.viewport(dark=True)


def test_orbit_draws_on_the_theme_surface():
    from pinneapple_tools.visualization import orbit

    assert orbit.turntable.__defaults__ is not None
    assert theme.viewport(dark=True) in orbit.turntable.__defaults__ or \
        theme.viewport(dark=True) in orbit.turntable.__kwdefaults__.values()


# ---------------------------------------------------------------------------
# consumers 2 and 3: the two browser viewers and the apps' stylesheet
# ---------------------------------------------------------------------------

def test_the_generated_copies_are_in_step_with_the_file():
    """Changing a colour in theme.json without rebuilding is a consumer drawing yesterday's palette."""
    assert build_theme.drift() == []


def _js_theme(path: Path) -> dict:
    """The THEME object out of a generated module, as data (the payload is JSON inside the module)."""
    text = path.read_text(encoding="utf-8")
    start = text.index("export const THEME = ")
    body = text[start + len("export const THEME = "):]
    return json.loads(body[:body.index("\n\nexport default")].rstrip().rstrip(";"))


@pytest.mark.parametrize("path", [STUDIO_JS, TWIN3D_JS])
def test_the_viewers_carry_the_house_palette(path):
    data = _js_theme(path)
    assert data["brand"] == theme.brand()
    assert data["categorical"] == list(theme.palette())
    assert data["surface"] == theme.theme()["surface"]
    assert data["defaultCmap"] == "cfd_sequential"


@pytest.mark.parametrize("path", [STUDIO_JS, TWIN3D_JS])
@pytest.mark.parametrize("name", ["cfd_sequential", "cfd_diverging", "cfd_cyclic", "coolwarm", "magma", "err"])
def test_a_colour_is_the_same_colour_in_python_and_in_the_browser(path, name):
    """The stop tables agree to 1/255, which is what the byte form in the JS file can lose and no more."""
    data = _js_theme(path)
    browser = data["colormaps"][name]["stops"]
    desk = theme.colormap_stops(name)
    assert [pos for pos, _ in browser] == pytest.approx([pos for pos, _ in desk])
    for (_, rgb_bytes), (_, rgb_float) in zip(browser, desk, strict=True):
        assert [c / 255 for c in rgb_bytes] == pytest.approx(list(rgb_float), abs=1 / 255)


def test_the_apps_stylesheet_carries_the_brand():
    css = APPS_CSS.read_text(encoding="utf-8")
    assert "--house-ink: #262626;" in css
    assert "--house-blue: #0081be;" in css
    assert "--house-ember: #a23a00;" in css
    for i, colour in enumerate(theme.palette(), start=1):
        assert f"--series-{i}: {colour};" in css


def test_the_apps_stylesheet_is_imported_where_the_apps_look_for_it():
    app_css = (ROOT / "apps/_shared/static/shared/app.css").read_text(encoding="utf-8")
    assert app_css.startswith('@import url("theme.css");')
    assert "--brand: var(--house-blue);" in app_css


def test_the_studio_core_has_no_palette_of_its_own():
    """The scales are read from theme.js; a literal stop table here would be the second palette the issue forbids."""
    core = (ROOT / "pinneapple_tools/visualization/studio/web/studio-core.js").read_text(encoding="utf-8")
    assert 'from "./theme.js"' in core
    literal = re.findall(r"export const (?:JET|COOLWARM|BWR|RDBU|MAGMA|ERR) = \[\[", core)
    assert literal == []


# ---------------------------------------------------------------------------
# colour-blind safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["normal", "protanopia", "deuteranopia", "tritanopia"])
def test_the_palette_stays_separable_for_a_colour_blind_reader(kind):
    assert theme.separation([theme.hex_to_rgb(c) for c in theme.palette()], kind) >= MIN_SEPARATION


def test_the_sequential_map_still_ranks_values_under_simulated_vision():
    """Losing the palette is not the only harm: a map that folds back reads as two values sharing a colour."""
    cmap = theme.colormap("cfd_sequential")
    rgb = [cmap(float(v))[:3] for v in [i / 16 for i in range(17)]]
    for kind in ("protanopia", "deuteranopia", "tritanopia"):
        seen = theme.simulate(rgb, kind)
        lightness = seen @ [0.2126, 0.7152, 0.0722]
        assert lightness[-1] > lightness[0], kind
        # monotone apart from one rounding wobble
        assert sum(d < -1e-3 for d in __import__("numpy").diff(lightness)) <= 1, kind


def test_an_unknown_quantity_is_refused_rather_than_defaulted():
    with pytest.raises(KeyError, match="no colormap registered"):
        theme.field_cmap_name("entropy_production")


def test_an_unknown_map_name_says_what_exists():
    with pytest.raises(KeyError, match="cfd_sequential"):
        theme.colormap_stops("chartreuse")
