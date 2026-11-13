import csv
from pathlib import Path

from pinneapple_analysis.verification.pipeline_assurance import Status
from pinneapple_analysis.verification.solver_field_adapters import (
    CSVFieldSpec, read_csv_field, compare_solver_csv,
)


def _write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "temperature"])
        writer.writerows(rows)


def test_solver_field_parity(tmp_path):
    a, b = tmp_path / "solver.csv", tmp_path / "surrogate.csv"
    _write(a, [(0, 300), (1, 320)])
    _write(b, [(0, 300), (1, 321)])
    ref = CSVFieldSpec(str(a), ("x",), "temperature", "K", "m")
    candidate = CSVFieldSpec(str(b), ("x",), "temperature", "K", "m")
    assert read_csv_field(ref).values == (300, 320)
    assert compare_solver_csv(ref, candidate).status == Status.FAIL
    _write(b, [(0, 300), (1, 320)])
    assert compare_solver_csv(ref, candidate).status == Status.PASS


def test_solver_field_rejects_unaligned_and_bad_units(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, [(0, 300), (1, 320)])
    _write(b, [(1, 320), (0, 300)])
    ref = CSVFieldSpec(str(a), ("x",), "temperature", "K", "m")
    cand = CSVFieldSpec(str(b), ("x",), "temperature", "K", "m")
    assert compare_solver_csv(ref, cand).status == Status.INCONCLUSIVE
    cand_bad = CSVFieldSpec(str(b), ("x",), "temperature", "C", "m")
    assert compare_solver_csv(ref, cand_bad).status == Status.FAIL


def test_duplicate_coordinates_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"
    _write(path, [(0, 300), (0, 301)])
    spec = CSVFieldSpec(str(path), ("x",), "temperature", "K", "m")
    assert compare_solver_csv(spec, spec).status == Status.INCONCLUSIVE
