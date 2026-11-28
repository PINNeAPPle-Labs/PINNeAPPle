"""The report reads a campaign and writes one PDF: no number invented, and a missing one says so."""
from __future__ import annotations

import json

from benchmarks import lpm_report


def _card(out, family: str, *, held: float | None, baseline: float | None = 0.9, tolerance: float = 0.05,
          refined: list[int] | None = None) -> None:
    directory = out / family / "model"
    directory.mkdir(parents=True, exist_ok=True)
    card = {"schema": "pinneapple.lpm.family_card/1", "family": {"name": family, "title": f"{family} title"},
            "dataset": {"n_designs": 40, "n_geometries": 10, "refined_geometries": refined or []},
            "split": {"policy": "random by geometry", "n_train_designs": 30, "n_holdout_designs": 10}}
    if held is not None:
        card["held_out"] = {"relative_l2": held, "constant_field_baseline_relative_l2": baseline,
                            "relative_l2_p90": held * 1.2, "n_designs": 10, "n_points": 960,
                            "tolerance": tolerance, "within_tolerance": held <= tolerance}
    (directory / "card.json").write_text(json.dumps(card), encoding="utf-8")


def test_every_family_gets_a_row_with_its_own_verdict(tmp_path):
    _card(tmp_path, "pp_wing3d", held=0.08, tolerance=0.10)
    _card(tmp_path, "pp_nozzle", held=0.30)
    document = lpm_report.collect(tmp_path)
    rows = {row["family"]: row for row in document["rows"]}
    assert rows["pp_wing3d"]["verdict"] == "inside tolerance"
    assert rows["pp_nozzle"]["verdict"] == "over tolerance"
    assert rows["pp_nozzle"]["beats_baseline"] is True
    assert document["commit"]                            # a number without a commit cannot be placed


def test_a_family_without_a_measured_held_out_says_so_rather_than_zero(tmp_path):
    _card(tmp_path, "pp_novo", held=None)
    row = lpm_report.collect(tmp_path)["rows"][0]
    assert row["held_out"] is None and row["verdict"] == "not fitted yet"


def test_the_snapshot_is_what_the_refinement_is_compared_against(tmp_path):
    stamp = tmp_path / "report" / "snapshots" / "20260101T000000Z"
    stamp.mkdir(parents=True)
    (tmp_path / "report" / "snapshots" / "latest").write_text("20260101T000000Z", encoding="utf-8")
    (stamp / "pp_wing3d.card.json").write_text(json.dumps({"held_out": {"relative_l2": 0.361}}), encoding="utf-8")
    _card(tmp_path, "pp_wing3d", held=0.25, refined=[10, 11])
    (tmp_path / "pp_wing3d" / "model" / "refinement.json").write_text(
        json.dumps({"round": 1, "proposed": 2, "added": 2, "held_out_relative_l2": 0.25}), encoding="utf-8")
    document = lpm_report.collect(tmp_path)
    assert document["before"]["pp_wing3d"] == 0.361
    assert document["rows"][0]["refined"] == [10, 11]


def test_the_pdf_is_written_and_carries_the_comparison_it_was_given(tmp_path):
    _card(tmp_path, "pp_wing3d", held=0.36)
    (tmp_path / "report").mkdir(exist_ok=True)
    (tmp_path / "report" / "comparison-wing3d-shift.json").write_text(json.dumps({
        "corpus": "WING — an outside corpus, 1 case(s), cc-by-nc-4.0, validation-only only",
        "rows": [{"model": "lpm", "relative_l2": 5.631, "p90": 5.6, "worst": 5.6, "baseline": 0.952,
                  "gates": {"accuracy": "FAILED", "baseline": "FAILED"}}],
        "not_compared": [{"model": "DoMINO", "reason": "its inputs are a grid, not a point cloud"}]}), encoding="utf-8")
    document = lpm_report.collect(tmp_path)
    assert document["comparisons"][0]["rows"][0]["relative_l2"] == 5.631
    pdf = lpm_report.build_pdf(document, tmp_path / "out" / "relatorio.pdf")
    assert pdf.is_file() and pdf.stat().st_size > 5000
    assert pdf.read_bytes()[:5] == b"%PDF-"


def test_an_empty_campaign_is_refused_by_the_command_line_rather_than_producing_a_blank_page(tmp_path):
    assert lpm_report.main(["--out", str(tmp_path), "--pdf", str(tmp_path / "x.pdf")]) == 2
    assert not (tmp_path / "x.pdf").exists()
