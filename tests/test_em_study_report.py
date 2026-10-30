"""The self-planning compact-range study and the datasheet it renders (``reflector.study_report`` and
``pinneapple_tools.reporting.datasheet``).

No solver runs here: what is checked is the part that has no physics in it and therefore cannot be caught by
looking at a field — the sentence parser, the rule that puts a step in the plan, and the layout. The parser is
the one that broke silently once, so it is tested on the three ways a design is written rather than on the one
way it was written first.
"""
from __future__ import annotations

import os

import pytest

from pinneapple_simulation.electromagnetics.reflector.compact_range import CompactRange
from pinneapple_simulation.electromagnetics.reflector.study_report import Problem, plan, from_text
from pinneapple_tools.reporting.datasheet import Datasheet, Figure, KPI, Step, Table, find_chrome

FT = 0.3048


# --------------------------------------------------------------------------------------------------
# the problem: units in, metres out
# --------------------------------------------------------------------------------------------------
def test_defaults_are_the_library_example():
    """``Problem()`` must describe the same range ``CompactRange()`` does, or the two disagree out of the box."""
    p, cr = Problem(), CompactRange()
    assert p.freq_hz == cr.freq_hz
    assert p.focal == pytest.approx(cr.reflector.focal)
    assert p.section == pytest.approx(2 * cr.reflector.half_x)
    assert p.offset == pytest.approx(cr.reflector.centre_y)
    assert p.roll == pytest.approx(cr.reflector.roll)
    assert p.feed_gain_dbi == cr.feed_gain_dbi
    assert p.qz_diameter == pytest.approx(cr.qz_diameter)
    assert p.qz_z == pytest.approx(cr.qz_z)


@pytest.mark.parametrize("text", [
    "3 GHz, focal length 1.5 m, 1.2 m square reflector, 0.15 m rolled edge, 0.6 m quiet zone at 3 m, feed 12 dBi",
    "3 GHz, 1.5 m focal length, reflector 1.2 m, rolled edge 0.15 m, quiet zone 0.6 m at 3 m, 12 dBi",
])
def test_from_text_reads_the_clause_that_names_the_thing(text):
    """Size before the noun and size after it must both land on the right field, in metres."""
    p = Problem.from_text(text)
    assert p.freq_hz == pytest.approx(3e9)
    assert p.focal == pytest.approx(1.5)
    assert p.section == pytest.approx(1.2)
    assert p.roll == pytest.approx(0.15)
    assert p.qz_diameter == pytest.approx(0.6)
    assert p.qz_z == pytest.approx(3.0)
    assert p.feed_gain_dbi == pytest.approx(12.0)


def test_from_text_accepts_imperial_and_keeps_a_decimal_point_intact():
    """``ft`` is read as feet, ``1.5`` is one number and not a clause break, and the fit is not guessed."""
    p = Problem.from_text("2 GHz, focal length 6 ft, 4 ft square reflector, 1 ft rolled edge, "
                          "3 ft quiet zone at 12 ft, feed 12.9 dBi")
    assert p.focal == pytest.approx(6 * FT)
    assert p.section == pytest.approx(4 * FT)
    assert p.roll == pytest.approx(1 * FT)
    assert p.qz_diameter == pytest.approx(3 * FT)
    assert p.qz_z == pytest.approx(12 * FT)
    assert p.feed_gain_dbi == pytest.approx(12.9)


def test_from_text_reads_a_band_when_one_is_given():
    p = Problem.from_text("3 GHz, focal length 1.5 m, 1.2 m square reflector, 0.6 m quiet zone at 3 m, "
                          "12 dBi, band 2 to 4 GHz")
    assert p.band_hz == pytest.approx((2e9, 4e9))


def test_from_text_names_what_it_could_not_find_instead_of_guessing():
    with pytest.raises(ValueError) as err:
        Problem.from_text("2 GHz, 4 ft square reflector")
    message = str(err.value)
    for missing in ("focal", "qz_diameter", "qz_z"):
        assert missing in message
    assert "section" not in message            # that one was given


def test_the_module_level_helper_is_the_same_reader():
    assert from_text("3 GHz, focal length 1.5 m, 1.2 m square reflector, "
                     "0.6 m quiet zone at 3 m").section == pytest.approx(1.2)


# --------------------------------------------------------------------------------------------------
# the plan: every step carries the number that put it there
# --------------------------------------------------------------------------------------------------
def test_plan_always_starts_with_the_model_and_its_validation():
    keys = [s.key for s in plan(Problem())]
    assert keys[:2] == ["model", "validation"]


def test_plan_carries_the_number_that_justifies_the_step():
    """A step whose reason does not name its cause is a step nobody can audit."""
    p = Problem()
    f = p.facts()
    assert f["fresnel_number"] < 2.0                       # the default range is a near-field one
    steps = {s.key: s for s in plan(p)}
    assert f"{f['fresnel_number']:.2f}" in steps["qz"].reason
    assert f"{f['fresnel_number']:.2f}" in steps["sweep"].reason
    assert all(s.reason.strip() for s in steps.values())


def test_plan_sweeps_only_when_the_geometry_asks_for_it():
    """Far field *and* a quiet zone well inside the section: no sweep, and the plan says why it is absent."""
    p = Problem(qz_z=1.5, qz_diameter=0.3)
    f = p.facts()
    assert f["fresnel_number"] >= 2.0 and f["qz_to_section"] <= 0.6
    keys = [s.key for s in plan(p)]
    assert "sweep_skipped" in keys
    assert "sweep" not in keys and "optimize" not in keys


def test_plan_adds_the_reference_calibration_and_band_steps_when_they_are_supplied():
    p = Problem(reference={("V", 3.0, "horizontal"): (1.0, 5.0, None)},
                reference_geometry={"x_ft": 1.0, "area_m2": 0.5},
                band_hz=(2e9, 4e9))
    keys = [s.key for s in plan(p)]
    assert {"reference", "calibration", "bandwidth"} <= set(keys)
    assert "bandwidth_skipped" not in keys


def test_plan_is_monochromatic_when_no_band_is_given():
    assert "bandwidth_skipped" in [s.key for s in plan(Problem())]


# --------------------------------------------------------------------------------------------------
# the datasheet: layout only, and the glyphs it promises
# --------------------------------------------------------------------------------------------------
def _sheet() -> Datasheet:
    d = Datasheet("Study under test", subtitle="layout only", eyebrow="EM / reflector")
    d.kpis([KPI("Wavelength", "5.90", "in", "$\\lambda$ = 149.9 mm")])
    d.verdict_box("The zone sits at $\\pm$0.5 dB and 30&deg; from the focus.", "warn", "Near field")
    s = d.section("One", lede="Nothing here is computed; every string was passed in.")
    s.row(Table([["focal length", "1.50 m"]], header=["quantity", "value"], caption="A table."),
          Figure("", "no figure"))
    s.note("A note inside a section.", "info", "Titled note")
    d.plan(done=[Step("Model", "the section is twelve wavelengths across", "done", "3 s")],
           queued=[Step("Band", "the range must cover 2-4 GHz", "queued", "")])
    d.confirm(["Which tolerance, over which diameter?"])
    d.limitations(["Physical optics carries no fringe edge current."])
    return d


def test_datasheet_writes_the_html_it_was_given(tmp_path):
    out = _sheet().build(os.path.join(tmp_path, "sheet.html"))
    html = open(out["html"], encoding="utf-8").read()
    for text in ("Study under test", "the section is twelve wavelengths across", "A table.",
                 "Which tolerance, over which diameter?", "Physical optics carries no fringe edge current."):
        assert text in html


def test_datasheet_translates_the_glyphs_it_advertises(tmp_path):
    """``$\\lambda$``, ``$\\pm$`` and ``&deg;`` must reach the page as glyphs, never as literal tokens."""
    out = _sheet().build(os.path.join(tmp_path, "sheet.html"))
    html = open(out["html"], encoding="utf-8").read()
    for glyph in ("λ", "±", "°"):
        assert glyph in html
    for token in ("$\\lambda$", "$\\pm$", "&deg;", "&amp;deg;"):
        assert token not in html


@pytest.mark.skipif(find_chrome() is None, reason="no headless Chromium/Chrome on this machine")
def test_datasheet_prints_a_pdf_when_a_browser_is_available(tmp_path):
    out = _sheet().build(os.path.join(tmp_path, "sheet.html"), pdf=os.path.join(tmp_path, "sheet.pdf"))
    assert out["pdf"] and os.path.exists(out["pdf"])
    with open(out["pdf"], "rb") as fh:
        assert fh.read(5) == b"%PDF-"
