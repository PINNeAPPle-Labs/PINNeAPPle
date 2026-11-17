"""The committed tables must be exactly what the committed records say.

A table that was generated once and then left behind while a record changed is the failure mode this
file exists to prevent. It regenerates every table and figure into a temporary directory and compares
with what is in the tree, so `pytest` fails on a stale table.

Run::

    python -m pytest research/arxiv_paper/scripts/test_tables_in_sync.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_tables  # noqa: E402


@pytest.fixture()
def regenerated(tmp_path):
    tables, figures = tmp_path / "tables", tmp_path / "figures"
    tables.mkdir()
    figures.mkdir()
    with mock.patch.object(make_tables, "TABLES", tables), \
         mock.patch.object(make_tables, "FIGURES", figures):
        make_tables.main()
    return tables


#: The figures this script owns; ``residual_*.pdf`` belong to experiments/verification/analyze.py.
OWNED_FIGURES = ("a1_rel_l2.pdf", "a2_rel_l2.pdf", "b_darcy.pdf", "c_nu.pdf")

#: The tables this script owns. ``verification_*.tex`` belong to
#: experiments/verification/analyze.py, which needs about three minutes and owns its own comparison.
OWNED = ("a1.tex", "a1_paired.tex", "a2.tex", "b.tex", "c.tex", "c_diag.tex", "d.tex",
         "environment.tex", "deviations.tex")


def test_every_committed_table_matches_a_fresh_one(regenerated):
    fresh = sorted(p.name for p in regenerated.glob("*.tex"))
    assert fresh == sorted(OWNED), "the generator and OWNED disagree"
    missing = [n for n in OWNED if not (make_tables.TABLES / n).is_file()]
    assert not missing, f"table(s) never generated: {missing} — run scripts/make_tables.py"
    stale = [n for n in fresh
             if (make_tables.TABLES / n).read_text() != (regenerated / n).read_text()]
    assert not stale, f"stale table(s): {stale} — run scripts/make_tables.py"


def test_the_verification_figures_exist():
    for name in ("residual_calibration.pdf", "residual_floor.pdf"):
        path = make_tables.FIGURES / name
        assert path.is_file(), name
        assert path.stat().st_size > 5000, name


def test_the_verification_tables_exist_and_are_not_hand_written():
    owned = ("verification_matrix.tex", "verification_rates.tex", "verification_floors.tex")
    for name in owned:
        path = make_tables.TABLES / name
        assert path.is_file(), name
        body = path.read_text()
        assert "rule" in body.lower() or "problem" in body.lower(), name
        # a generated table never carries the marker a hand edit would leave behind
        assert "TODO" not in body and "FIXME" not in body, name


def test_the_committed_figures_match_fresh_ones(regenerated):
    """The PDFs are byte-deterministic, so a stale figure is detectable and not just a size change."""
    fresh = sorted(p.name for p in (regenerated.parent / "figures").glob("*.pdf"))
    assert fresh == sorted(OWNED_FIGURES), "the generator and OWNED_FIGURES disagree"
    missing = [n for n in OWNED_FIGURES if not (make_tables.FIGURES / n).is_file()]
    assert not missing, f"figure(s) never generated: {missing} — run scripts/make_tables.py"
    stale = [n for n in fresh
             if (make_tables.FIGURES / n).read_bytes()
             != (regenerated.parent / "figures" / n).read_bytes()]
    assert not stale, f"stale figure(s): {stale} — run scripts/make_tables.py"


def test_the_records_the_tables_come_from_exist():
    for name in ("A1.json", "A2.json", "B.json", "C.json", "C_diagnostics.json", "D.json",
                 "environment.json"):
        assert (make_tables.RESULTS / name).is_file(), name


def test_no_table_is_empty(regenerated):
    for path in regenerated.glob("*.tex"):
        body = path.read_text()
        assert "\\toprule" in body and "\\bottomrule" in body, path.name
        assert len(body) > 80, path.name


# --- the submission metadata, which is a hand-written artifact and can go stale in the same way -----

PAPER = HERE.parent / "paper.tex"
METADATA = HERE.parent / "arxiv" / "METADATA.md"

#: Figures the metadata quotes; each must appear in both, so a corrected abstract cannot leave the
#: submission form behind. (The abstract shipped to arXiv is the one a reader cites, so it is the
#: hand-written text most worth guarding.)
QUOTED_IN_BOTH = ("166", "1156", "0.94", "0.73", "28 of 36", "12.5", "3.4")


def _abstract(path: Path) -> str:
    body = path.read_text()
    return body[body.index("\\begin{abstract}"):body.index("\\end{abstract}")]


def _metadata_abstract() -> str:
    body = METADATA.read_text()
    start = body.index("## Abstract (plain text, for the arXiv field)")
    start = body.index("```", start) + 3
    return body[start:body.index("```", start)]


def test_the_submission_abstract_quotes_the_manuscript():
    paper, meta = _abstract(PAPER), _metadata_abstract()
    for number in QUOTED_IN_BOTH:
        assert number in paper, f"{number} left the manuscript's abstract"
        assert number in meta, f"{number} is in the manuscript but not in the arXiv metadata"


def test_the_submission_abstract_carries_no_retired_figure():
    """These were in an earlier abstract and are wrong against the records; they must not come back."""
    meta = _metadata_abstract()
    for retired in ("factor of 10", "2.4 on Helmholtz", "0.92 of the term scale",
                    "six significant digits"):
        assert retired not in meta, retired
