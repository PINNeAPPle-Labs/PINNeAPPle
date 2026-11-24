"""The regime audit on a public benchmark: which bands the NASA TMR suite never reaches (#171).

The first half of the issue runs on our own data, where the answer could be argued with. This half runs on published
freestream conditions, and what it reports is that a reference set used to validate turbulence models has no case in
Mach 0.3-0.8 — the band most external aerodynamics lives in — and no laminar or transitional flat plate. Those are
the numbers the case pages state; the expectations below were read off the pages, not off the audit.
"""
from __future__ import annotations

import pytest

from pinneapple_data.benchmark_suites import NASA_TMR_CASES, audit_nasa_tmr, nasa_tmr_cases, nasa_tmr_splits


def test_the_suite_keeps_every_published_number_and_no_guessed_one():
    cases = {c.case_id: c for c in nasa_tmr_cases()}
    assert cases["2DZP"].reynolds == pytest.approx(5e6) and cases["2DZP"].mach == pytest.approx(0.2)
    assert cases["2DN00"].reynolds == pytest.approx(6e6) and cases["2DN00"].reynolds_basis == "chord"
    assert cases["2DBFS"].reynolds == pytest.approx(3.6e4)
    assert cases["ACC-SBLI"].mach == pytest.approx(7.11) and cases["ACC-SBLI"].reynolds is None
    assert cases["ATB"].reynolds is None                   # the page states no Reynolds number
    assert all(c.reference.startswith("https://tmbwg.github.io/turbmodels/") for c in NASA_TMR_CASES)


def test_a_reynolds_number_without_its_reference_length_is_refused():
    from pinneapple_data.benchmark_suites import PublicCase

    with pytest.raises(ValueError, match="length it is based on"):
        PublicCase("x", "a case", "https://example.org", reynolds=1e5)


def test_reynolds_numbers_based_on_different_lengths_never_share_a_column():
    cols = nasa_tmr_splits()["suite"]
    x_based = [v for v in cols["reynolds"] if v == v]                       # NaN-aware: what was published
    assert x_based == [5e6, 15e6, 15e6]                                     # the two flat plates, per unit length of x
    assert [v for v in cols["reynolds_chord"] if v == v] == [6e6]           # NACA 0012, per chord
    assert [v for v in cols["reynolds_step_h"] if v == v] == [3.6e4]        # backward-facing step, per step height


def test_the_audit_names_the_mach_band_the_suite_never_reaches():
    card = audit_nasa_tmr()
    mach = card["regime.mach"]
    assert mach.status == "fail"
    assert mach.evidence["counts"] == {"incompressible": 3, "subsonic": 0, "transonic": 1, "supersonic": 1,
                                       "hypersonic": 2}
    assert mach.evidence["missing"] == ["subsonic"]
    assert mach.evidence["sample_here"] == [{"column": "mach", "low": 0.3, "high": 0.8, "label": "subsonic"}]


def test_the_audit_names_the_reynolds_bands_the_suite_never_reaches():
    card = audit_nasa_tmr()
    re = card["regime.reynolds_flat_plate"]
    assert re.status == "fail"
    assert re.evidence["missing"] == ["laminar", "transitional"]
    assert re.evidence["counts"]["turbulent"] == 3                  # Re 5e6 and 15e6 per unit length of x
    # the empty bands come back as the intervals a new case would have to fall in
    assert [r["label"] for r in re.evidence["sample_here"]] == ["laminar", "transitional"]


def test_the_two_hypersonic_entries_are_reported_as_covered_not_thin():
    """Two samples of a band is the minimum the audit calls covered, and the M = 5 plate is why the band is not
    reported missing at all: the suite does reach hypersonic conditions."""
    mach = audit_nasa_tmr()["regime.mach"]
    assert "hypersonic" not in mach.evidence["missing"] and "hypersonic" not in mach.evidence["thin"]
    assert mach.evidence["share"]["hypersonic"] == pytest.approx(2 / 7)
