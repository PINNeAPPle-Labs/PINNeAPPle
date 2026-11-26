"""A run result is evidence only if it can be checked: hash, environment, parameters, and the command that redoes it
(#38, roadmap B12).

The record format lives in ``pinneapple_core.run_record``; these tests go at it the way a verification tool would:
build one, edit it, and see whether the check notices. They also check the two integrations that make "every published
number points to a verifiable record" true in the tree instead of in a document -- the benchmark's per-job record and
the examples runner's per-script record.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pinneapple_core import run_record

REPO = Path(__file__).resolve().parent.parent


def _run(tmp_path: Path, *, name: str = "run.json", metrics: dict | None = None) -> run_record.RunRecord:
    out = tmp_path / "out.txt"
    out.write_text("result: 1.234\n", encoding="utf-8")
    return run_record.record(
        "case", "demo/one", root=tmp_path, outputs=[out],
        command=[sys.executable, "demo.py", "--flag"], parameters={"n": 1024, "seed": 0},
        metrics=metrics if metrics is not None else {"drag": 0.0314, "n": 1024},
    )


def test_a_record_carries_what_ran_on_what_and_how_to_redo_it(tmp_path):
    rec = _run(tmp_path)
    assert rec.kind == "case" and rec.id == "demo/one"
    assert rec.command[1] == "demo.py", "the command is data, so a tool can run it"
    assert rec.parameters == {"n": 1024, "seed": 0}
    assert rec.environment["python"] and "cpu_count" in rec.environment
    assert rec.outputs and rec.outputs[0]["path"] == "out.txt"
    assert len(rec.outputs[0]["sha256"]) == 64
    assert rec.metrics["drag"] == 0.0314


def test_the_digest_does_not_depend_on_key_order(tmp_path):
    rec = _run(tmp_path)
    same = json.loads(json.dumps(rec.to_dict(), sort_keys=True))
    assert run_record.RunRecord.from_dict(same).hash == rec.hash
    shuffled = dict(reversed(list(rec.to_dict().items())))
    assert run_record.RunRecord.from_dict(shuffled).compute_hash() == rec.hash


def test_a_clean_record_verifies(tmp_path):
    rec = _run(tmp_path)
    assert run_record.verify(rec, tmp_path) == []


def test_editing_a_published_number_is_detected(tmp_path):
    """The number is the thing a report prints: editing it has to break the evidence, not slip through."""
    rec = _run(tmp_path)
    rec.metrics["drag"] = 0.0200
    problems = run_record.verify(rec, tmp_path)
    assert any("digest mismatch" in p for p in problems), problems


def test_a_changed_output_file_is_detected(tmp_path):
    rec = _run(tmp_path)
    (tmp_path / "out.txt").write_text("result: 9.999\n", encoding="utf-8")
    problems = run_record.verify(rec, tmp_path)
    assert any("changed output" in p for p in problems), problems


def test_a_missing_output_file_is_detected(tmp_path):
    rec = _run(tmp_path)
    (tmp_path / "out.txt").unlink()
    problems = run_record.verify(rec, tmp_path)
    assert any("missing output" in p for p in problems), problems


def test_a_record_without_a_command_is_not_evidence(tmp_path):
    rec = _run(tmp_path)
    rec.command = []
    rec.sealed()
    assert any("no command" in p for p in run_record.verify(rec, tmp_path))


def test_an_unsealed_record_is_refused(tmp_path):
    rec = _run(tmp_path)
    rec.hash = ""
    assert any("no digest" in p for p in run_record.verify(rec, tmp_path))


def test_records_are_queryable_by_id_kind_and_metric(tmp_path):
    rec = _run(tmp_path)
    rec.write(tmp_path / "records" / "demo.record.json")
    other = run_record.record("example", "demo/two", root=tmp_path, command=["python", "x.py"],
                             parameters={}, metrics={"elapsed_s": 1.0})
    other.write(tmp_path / "records" / "two.record.json")
    assert [r.id for r in run_record.find(tmp_path, id="demo/one")] == ["demo/one"]
    assert [r.id for r in run_record.find(tmp_path, kind="example")] == ["demo/two"]
    assert [r.id for r in run_record.find(tmp_path, metric="drag")] == ["demo/one"]
    rows = run_record.index(tmp_path)
    assert {r["id"] for r in rows} == {"demo/one", "demo/two"}
    assert all(r["digest_ok"] for r in rows)


def test_verify_tree_reports_only_the_records_with_problems(tmp_path):
    good = _run(tmp_path)
    good.write(tmp_path / "records" / "good.record.json")
    bad_path = _run(tmp_path).write(tmp_path / "records" / "bad.record.json")
    edited = json.loads(bad_path.read_text())           # someone edits the file after the fact, digest and all
    edited["metrics"]["drag"] = -1.0
    bad_path.write_text(json.dumps(edited))
    findings = run_record.verify_tree(tmp_path)
    assert list(findings) == [str(tmp_path / "records" / "bad.record.json")]


def test_the_cli_verifies_a_tree(tmp_path):
    _run(tmp_path).write(tmp_path / "records" / "one.record.json")
    out = subprocess.run([sys.executable, "-m", "pinneapple_core.run_record", "verify", str(tmp_path)],
                         capture_output=True, text=True, env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin",
                                                              "HOME": str(Path.home())})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "1 record(s), 0 with problems" in out.stdout


def test_the_benchmark_writes_one_record_per_job(tmp_path):
    """The benchmark's published numbers point at these: one record per job, next to its result file."""
    from pinneapple_analysis.benchmark.runner import write_job_record
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "final__burgers__fno__L1.000__s0.json").write_text(
        json.dumps({"ok": True, "result": {"rel_l2": 0.02}}), encoding="utf-8")
    spec = {"id": "final__burgers__fno__L1.000__s0", "problem": "burgers", "method": "fno", "stage": "final",
            "seed": 0, "cfg": {"lr": 1e-3}}
    path = write_job_record(spec, str(tmp_path))
    rec = run_record.load(path)
    assert rec.kind == "benchmark-run" and rec.id == spec["id"]
    assert rec.parameters["problem"] == "burgers" and rec.parameters["cfg"] == {"lr": 1e-3}
    assert rec.outputs[0]["path"] == "runs/final__burgers__fno__L1.000__s0.json"
    assert rec.root == "..", "outputs live one folder up from records/, and the record says so"
    assert run_record.verify(rec, tmp_path / "records") == []
    assert "--out" in rec.command


def test_the_workspace_lists_the_records_behind_a_row(tmp_path):
    from pinneapple_analysis.benchmark.config import BenchmarkConfig
    from pinneapple_analysis.benchmark.runner import Workspace, write_job_record
    ws = Workspace(tmp_path, BenchmarkConfig(problems=["burgers"], methods=["fno"]))
    (tmp_path / "runs").mkdir(exist_ok=True)
    spec = {"id": "final__burgers__fno__L1.000__s0", "problem": "burgers", "method": "fno", "stage": "final",
            "seed": 0, "cfg": {}}
    (tmp_path / "runs" / f"{spec['id']}.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    write_job_record(spec, str(tmp_path))
    assert ws.records_for("burgers", "fno") == [f"records/{spec['id']}.record.json"]
    assert ws.records_for("burgers", "pinn") == []


def test_the_examples_runner_writes_a_record_per_script(tmp_path):
    """A published elapsed time and exit code have to point at the file and the command that produced them."""
    sys.path.insert(0, str(REPO))
    import run_all_examples as rae
    run_dir = tmp_path / "sanity" / "sanity_01"
    run_dir.mkdir(parents=True)
    (run_dir / "stdout.txt").write_text("ok\n", encoding="utf-8")
    (run_dir / "stderr.txt").write_text("\n", encoding="utf-8")
    script = rae.Script(path="examples/sanity/sanity_01.py", category="sanity", name="sanity_01", args=[])
    result = rae.RunResult(name="sanity_01", path=script.path, category="sanity", status="passed", elapsed_s=0.25)
    where = rae._write_record(result, script, [sys.executable, script.path], run_dir, tmp_path, "cpu", 0)
    assert where == "sanity/sanity_01/sanity_01.record.json"
    rec = run_record.load(tmp_path / where)
    assert rec.kind == "example" and rec.id == "sanity/sanity_01"
    assert rec.metrics["elapsed_s"] == pytest.approx(0.25)
    assert "python" in rec.command[0]
    assert rec.root == "../..", "the outputs are relative to the output folder, two levels up"
    assert run_record.verify(rec, run_dir) == []
