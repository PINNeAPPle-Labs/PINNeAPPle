"""The CalculiX run files a contact analysis is judged by (``.frd`` CONTACT block, ``.dat`` reaction totals).

``tests/fixtures/calculix_contact/`` holds a real run of the sliding-block case, written by CalculiX 2.23: two blocks
in contact with Coulomb friction, a prescribed pressure and a tangential load at half the friction limit, four
increments. ``golden.json`` in that folder is **what the engineering API's own parser read from these files**
(``app/prepomax/results.py``, commit a63f5c5, before the library took over) — every number below is compared against
it, so the assertions are not the library agreeing with itself: they are the library agreeing with the parser that was
in production, on the same bytes.

The file also carries the case the fixed-column reading exists for: ``ccx`` writes a value and its neighbour with no
separator between them when the first one is wide, so several ``-1`` rows read ``... 99-6.74101E-05 4.97508E-02-1.37``
and a whitespace split would silently find five numbers where there are six.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from pinneapple_data import cae

HERE = Path(__file__).resolve().parents[1] / "fixtures" / "calculix_contact"
FRD = (HERE / "fixture.frd").read_bytes()
DAT = (HERE / "fixture.dat").read_bytes()
STA = (HERE / "fixture.sta").read_bytes()
LOG = (HERE / "fixture.log").read_bytes()
GOLDEN = json.loads((HERE / "golden.json").read_text(encoding="utf-8"))
CLOSE = dict(rel=0, abs=1e-12)


def _as_int(d: dict[str, object]) -> dict[int, object]:
    return {int(k): v for k, v in d.items()}


# ------------------------------------------------------------------------------------- the file itself is the hard case
def test_the_fixture_has_values_that_touch_without_a_separator():
    """The pattern a whitespace split loses: a negative value beginning right after the exponent of the previous one."""
    touching = [ln for ln in FRD.decode("latin-1").splitlines()
                if ln.startswith(" -1") and re.search(r"[0-9]E[+-][0-9]{2}-[0-9]", ln)]
    assert len(touching) > 50, "the fixture must exercise the fixed-column case"
    row = next(ln for ln in touching if " 2." in ln)                  # a node of the contact patch
    assert re.search(r"[0-9]E[+-][0-9]{2}-[0-9]", row[13:])           # a value butted against its neighbour


def test_the_contact_block_is_read_with_its_six_components():
    blocks = [b for b in cae.read_blocks(FRD) if b.name == "CONTACT"]
    assert len(blocks) == 4                                          # one per increment
    assert blocks[0].components == ["COPEN", "CSLIP1", "CSLIP2", "CPRESS", "CSHEAR1", "CSHEAR2"]
    assert [b.time for b in blocks] == [0.25, 0.5, 0.875, 1.0]
    assert all(len(b.values) == 54 for b in blocks)                  # one row per node of the model
    assert cae.CONTACT_FIELDS["CSHEAR1"] == "shear1"


def test_the_contact_block_is_also_a_field_of_the_mesh():
    mesh = cae.read_any({"fixture.frd": FRD})
    assert mesh.point_data["CONTACT"].shape == (54, 6)
    assert mesh.source["components"]["CONTACT"] == ["COPEN", "CSLIP1", "CSLIP2", "CPRESS", "CSHEAR1", "CSHEAR2"]
    assert mesh.point_data["DISP"].shape == (54, 3)                  # four components announced, three written
    assert "CONTACT" in mesh.point_data and mesh.units["CONTACT"].startswith("model units")


# --------------------------------------------------------------------------------------------- the library x the app
def test_contact_per_node_is_exactly_what_the_app_parser_read():
    got = cae.contact(FRD)
    want = _as_int(GOLDEN["frd"]["contact"])
    assert set(got) == set(want)
    for node, row in want.items():
        assert set(got[node]) == set(row)
        for key, value in row.items():
            assert got[node][key] == pytest.approx(value, **CLOSE), (node, key)


def test_the_contact_state_at_a_step_time_is_read_apart():
    want = _as_int(GOLDEN["frd"]["contact_at_0_5"])
    got = cae.contact(FRD, until=0.5)
    assert set(got) == set(want)
    for node, row in want.items():
        for key, value in row.items():
            assert got[node][key] == pytest.approx(value, **CLOSE), (node, key)
    assert cae.contact(FRD) != got                                   # the pull moved the interface


def test_coordinates_displacement_and_stress_agree_with_the_app_parser():
    for got, want in ((cae.coordinates(FRD), _as_int(GOLDEN["frd"]["coordinates"])),
                      (cae.displacement(FRD), _as_int(GOLDEN["frd"]["displacement"]))):
        assert set(got) == set(want)
        for node, row in want.items():
            assert list(got[node]) == pytest.approx(list(row), **CLOSE), node
    got_s, want_s = cae.stress(FRD), _as_int(GOLDEN["frd"]["stress"])
    assert set(got_s) == set(want_s)
    for node, row in want_s.items():
        for key, value in row.items():
            assert got_s[node][key] == pytest.approx(value, **CLOSE), (node, key)


def test_the_reaction_totals_and_their_history_are_what_the_app_parser_read():
    want = GOLDEN["dat"]["totals"]
    got = cae.read_totals(DAT)
    assert set(got) == set(want)
    for name, ref in want.items():
        assert list(got[name].values) == pytest.approx(ref["values"], **CLOSE)
        assert got[name].time == pytest.approx(ref["time"], **CLOSE)
        assert [t for t, _ in got[name].history] == pytest.approx([h[0] for h in ref["history"]], **CLOSE)
        for (t, values), (_, ref_values) in zip(got[name].history, ref["history"], strict=True):
            assert list(values) == pytest.approx(ref_values, **CLOSE), t


def test_the_printed_displacements_of_a_set_agree_with_the_app_parser():
    want = GOLDEN["dat"]["displacements"]
    got = cae.read_displacements(DAT)
    assert set(got) == set(want)
    for name, nodes in want.items():
        assert set(got[name]) == {int(n) for n in nodes}
        for node, row in nodes.items():
            assert list(got[name][int(node)]) == pytest.approx(row, **CLOSE), (name, node)


def test_status_and_verdict_agree_with_the_app_parser():
    status = cae.read_status(STA)
    assert status.increments == GOLDEN["sta"]["increments"]
    assert status.attempts == GOLDEN["sta"]["attempts"]
    assert status.last_time == pytest.approx(GOLDEN["sta"]["last_time"], **CLOSE)
    assert status.converged is GOLDEN["sta"]["converged"]
    verdict = cae.read_log(LOG, status, produced=True)
    assert (verdict.status, verdict.detail) == (GOLDEN["verdict"]["status"], GOLDEN["verdict"]["detail"])


# --------------------------------------------------------------------------------------------- the physics of the run
def test_the_coulomb_ratio_is_the_friction_the_run_was_built_with():
    """The tangential load is half the friction limit, so the joint sticks and ``|tau|/p`` must sit below ``mu``."""
    ratio = cae.coulomb_ratio(cae.contact(FRD))
    assert ratio is not None and 0.0 < ratio < 0.3
    assert ratio == pytest.approx(0.2775, abs=1e-3)
    # the normal reaction is the pressure times the area, and the tangential one the load applied
    totals = cae.read_totals(DAT)["BFIX"]
    assert abs(totals.values[1]) == pytest.approx(5.0 * 16.0, rel=1e-3)
    assert abs(totals.values[0]) == pytest.approx(0.5 * 0.3 * 80.0, rel=1e-3)


def test_a_node_outside_the_interface_contributes_nothing_to_the_ratio():
    contact = cae.contact(FRD)
    loaded = [n for n, r in contact.items() if r["pressure"] > 0]
    assert len(loaded) == 9                                          # the 3x3 patch of a 2x2 mesh
    assert cae.coulomb_ratio(contact, loaded) == cae.coulomb_ratio(contact)
    assert cae.coulomb_ratio(contact, [n for n, r in contact.items() if r["pressure"] == 0]) is None


def test_a_file_without_a_contact_block_reads_as_empty_not_as_zero():
    """The deck asked for no ``*CONTACT FILE``: nothing was written, which is not the same as a joint carrying nothing."""
    text = FRD.decode("latin-1")
    head = text.split("  100CL")[0]                                   # mesh only, no result blocks
    assert cae.contact(head.encode("latin-1")) == {}
    assert cae.read_blocks(head.encode("latin-1")) == []


def test_the_values_are_read_by_column_when_two_of_them_touch():
    """A row written by hand with no space between the third and fourth value reads the same as with spaces."""
    spaced = (b"  100CL  101 2.50000E-01          2                     0    1           1\n"
              b" -4  CONTACT     6    1\n"
              b" -5  COPEN       1    1    0    0\n"
              b" -5  CSLIP1      1    1    0    0\n"
              b" -5  CSLIP2      1    1    0    0\n"
              b" -5  CPRESS      1    1    0    0\n"
              b" -5  CSHEAR1     1    1    0    0\n"
              b" -5  CSHEAR2     1    1    0    0\n"
              b" -1         1 1.00000E+00 2.00000E+00 3.00000E+00 4.00000E+00 5.00000E+00 6.00000E+00\n"
              b" -1         2-1.00000E+00-2.00000E+00-3.00000E+00-4.00000E+00-5.00000E+00-6.00000E+00\n"
              b" -3\n")
    rows = cae.contact(spaced)
    assert rows[1] == {"opening": 1.0, "slip1": 2.0, "slip2": 3.0, "pressure": 4.0, "shear1": 5.0, "shear2": 6.0}
    assert rows[2] == {"opening": -1.0, "slip1": -2.0, "slip2": -3.0, "pressure": -4.0, "shear1": -5.0,
                       "shear2": -6.0}


def test_a_short_row_reads_as_nan_and_never_shifts_the_columns():
    """``DISP`` announces four components and writes three: the fourth is ``nan``, not the next node's first value."""
    block = cae.read_blocks(FRD)
    disp = next(b for b in block if b.name == "DISP")
    assert disp.components == ["D1", "D2", "D3", "ALL"]
    assert all(len(row) == 4 and np.isnan(row[3]) for row in disp.values.values())
