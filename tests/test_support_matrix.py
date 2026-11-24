"""The supported versions matrix (#93): the declaration, the verdict, the evidence, and the page.

The page is generated from ``pinneapple_stability/support_matrix.json``, so these tests are what keeps it honest:
a cell cannot be printed as executed without a run that matches it, and the declared range cannot drift from
``pyproject.toml``.
"""
import json
import os
import re
import subprocess
import sys

import pytest

from pinneapple_stability import support

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "docs", "support_matrix.md")


# ── the declaration and the project metadata ─────────────────────────────────────────────────────────────
def _pyproject():
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)["project"]


def test_declared_range_matches_the_classifiers_and_requires_python():
    project = _pyproject()
    classifiers = [c for c in project["classifiers"] if c.startswith("Programming Language :: Python :: 3.")]
    declared = sorted(support.version_line(c.rsplit(" ", 1)[-1]) for c in classifiers)
    data = support.matrix()
    assert declared == sorted([data["python"]["min"], "3.11", "3.12", data["python"]["max"]]), declared
    assert project["requires-python"] == f">={data['python']['min']}", project["requires-python"]
    assert "<" not in project["requires-python"], "an upper bound needs a cell for every version it admits"


def test_torch_floor_matches_the_dependency():
    deps = _pyproject()["dependencies"]
    floor = support.matrix()["torch"]["floor"]
    assert any(d.replace(" ", "") == f"torch>={floor}" for d in deps), (floor, [d for d in deps if "torch" in d])


def test_every_cell_is_inside_the_declared_axes_and_unique():
    data = support.matrix()
    seen = set()
    for cell in data["cells"]:
        assert cell["os"] in data["os"], cell
        lo, hi = support.version_tuple(data["python"]["min"]), support.version_tuple(data["python"]["max"])
        assert lo <= support.version_tuple(cell["python"]) <= hi, cell
        assert cell["torch"] in data["torch"]["lines"], cell
        key = (cell["os"], support.version_line(cell["python"]), cell["torch"])
        assert key not in seen, f"duplicate cell {key}"
        seen.add(key)


def test_the_declared_range_is_covered_on_linux_and_both_torch_lines_on_one_os():
    data = support.matrix()
    python_versions = {support.version_line(c["python"]) for c in data["cells"] if c["os"] == "linux"}
    lo, hi = support.version_tuple(data["python"]["min"]), support.version_tuple(data["python"]["max"])
    assert {str(v) for v in range(lo[1], hi[1] + 1)} <= {v.split(".")[1] for v in python_versions}, python_versions
    by_os = {}
    for cell in data["cells"]:
        by_os.setdefault(cell["os"], set()).add(cell["torch"])
    assert any(lines == set(data["torch"]["lines"]) for lines in by_os.values()), by_os


def test_the_smoke_list_has_a_step_for_every_id():
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_matrix", os.path.join(ROOT, "scripts", "check_matrix.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert [s["id"] for s in support.matrix()["smoke"]] == list(mod.STEPS), "the page lists a step nobody runs"


# ── the verdict ──────────────────────────────────────────────────────────────────────────────────────────
def _env(python, torch, osname):
    return {"python": python, "torch": torch, "os": osname, "arch": "arm64"}


def test_an_executed_cell_is_reported_as_executed():
    done = [c for c in support.cells() if c.status == "executed"]
    if not done:
        pytest.skip("no cell has a recorded run yet")
    for cell in done:
        verdict = support.classify(_env(cell.python, cell.torch_version, cell.os))
        assert verdict.status == "executed" and verdict.in_matrix and verdict.executed, (cell, verdict.reasons)
        assert verdict.cell is not None and verdict.cell.label() == cell.label()


def test_a_cell_without_a_run_is_declared_and_never_executed():
    wanted = [c for c in support.cells() if c.status == "declared"]
    if not wanted:
        pytest.skip("every cell is executed: nothing left to check for a declared-only cell")
    cell = wanted[0]
    verdict = support.classify(_env(cell.python, cell.torch_version, cell.os))
    assert verdict.status == "declared" and verdict.in_matrix and not verdict.executed
    assert any("nobody has recorded" in r for r in verdict.reasons), verdict.reasons


def test_outside_the_axes_is_outside():
    data = support.matrix()
    older = support.classify(_env("3.9.6", data["torch"]["lines"]["current"], "macos"))
    newer = support.classify(_env("3.99", data["torch"]["lines"]["current"], "macos"))
    old_torch = support.classify(_env("3.11", "1.13.0", "macos"))
    for verdict in (older, newer, old_torch):
        assert verdict.status == "outside" and not verdict.in_matrix, verdict.reasons
    assert "older than the oldest declared" in older.reasons[0]
    assert "newer than the newest declared" in newer.reasons[0]
    assert "older than the declared floor" in old_torch.reasons[0]


def test_an_interpreter_without_torch_is_incomplete_not_green():
    verdict = support.classify(_env("3.11.16", None, "macos"))
    assert verdict.status == "incomplete" and not verdict.in_matrix
    assert any("torch is not installed" in r for r in verdict.reasons), verdict.reasons


def test_a_torch_between_the_two_lines_is_inside_the_range_but_in_no_cell():
    data = support.matrix()
    between = "2.8.0"
    assert support.version_tuple(between) >= support.version_tuple(data["torch"]["floor"])
    verdict = support.classify(_env("3.11", between, "macos"))
    assert verdict.status == "declared" and verdict.cell is None, verdict.reasons


# ── recording evidence ──────────────────────────────────────────────────────────────────────────────────
def test_recording_replaces_the_run_of_the_same_environment(tmp_path, monkeypatch):
    """On a scratch copy of the matrix: one environment recorded twice leaves one run, the newest."""
    target = tmp_path / "matrix.json"
    target.write_text(json.dumps(support.matrix()), encoding="utf-8")
    monkeypatch.setattr(support, "MATRIX_PATH", target)
    try:
        support.reload()
        before = len(support.runs())
        results = [{"id": "imports", "ok": True, "seconds": 0.5}, {"id": "autograd", "ok": True, "seconds": 1.5}]
        data = support.matrix()
        env = _env(data["python"]["min"], data["torch"]["lines"]["current"], "linux")
        first = support.record(env, results, at="2026-01-01", note="first")
        assert len(support.runs()) == before + 1, "a new environment adds exactly one run"
        again = support.record(env, results, at="2026-02-02", note="second")
        assert len(support.runs()) == before + 1, "the same environment again replaces its run"
        mine = [r for r in support.runs() if r["os"] == "linux" and r["python"].startswith("3.10")]
        assert len(mine) == 1, mine
        assert mine[0]["at"] == "2026-02-02" and mine[0]["seconds"] == 2.0
        assert mine[0]["smoke"] == ["imports", "autograd"] and mine[0]["failed"] == []
        assert first["at"] == "2026-01-01" and again["note"] == "second"
        written = json.loads(target.read_text(encoding="utf-8"))
        assert any(r["os"] == "linux" and r["at"] == "2026-02-02" for r in written["runs"])
    finally:
        monkeypatch.undo()
        support.reload()


def test_the_recorded_runs_carry_the_steps_they_ran():
    data = support.matrix()
    ids = {s["id"] for s in data["smoke"]}
    for run in data["runs"]:
        assert set(run["smoke"]) <= ids, run
        assert run["failed"] == [], run


# ── the page ───────────────────────────────────────────────────────────────────────────────────────────
def test_the_page_matches_the_declaration():
    with open(PAGE, encoding="utf-8") as f:
        assert f.read() == support.render_markdown(), "run: python scripts/check_matrix.py --markdown"


def test_the_page_prints_executed_exactly_where_a_run_exists():
    page = support.render_markdown()
    rows = [line for line in page.splitlines() if re.match(r"\| (linux|macos|windows) \|", line)]
    assert len(rows) == len(support.cells())
    for row, cell in zip(rows, support.cells(), strict=True):
        cols = [c.strip() for c in row.strip("|").split("|")]
        assert cols[0] == cell.os and cols[1] == cell.python
        if cell.status == "executed":
            assert cols[3].endswith("executed") and cell.run is not None, row
        else:
            assert cols[3].endswith("declared") and cell.run is None, row


def test_the_page_says_how_many_cells_are_declared_only():
    page = support.render_markdown()
    done = sum(1 for c in support.cells() if c.status == "executed")
    total = len(support.cells())
    assert f"**{done} of {total} cells executed.**" in page
    if done < total:
        assert "declared only" in page


# ── the command line ──────────────────────────────────────────────────────────────────────────────────
def _run(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_matrix.py"), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_the_script_reports_an_unsupported_python_with_exit_code_2():
    out = _run("--no-smoke", "--python", "3.9.6")
    assert out.returncode == 2, out.stdout + out.stderr
    assert "OUTSIDE" in out.stdout


def test_the_script_and_the_page_agree():
    out = _run("--check-docs")
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_smoke_passes_in_the_environment_that_runs_the_suite():
    pytest.importorskip("torch")
    sys.path.insert(0, ROOT)
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_matrix_smoke", os.path.join(ROOT, "scripts", "check_matrix.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results = mod.smoke(["imports", "granted-none"])
    assert [r["id"] for r in results] == ["imports"]
    full = mod.smoke()
    failed = [r for r in full if not r["ok"]]
    assert not failed, failed
