"""The queue: one job at a time, on record, and it stops when three in a row fail."""
from __future__ import annotations

import json
import os

import pytest

from benchmarks import lpm_queue


def _queue(tmp_path, jobs, report=None):
    path = tmp_path / "queue" / "queue.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema": lpm_queue.QUEUE_SCHEMA, "jobs": jobs}
    if report:
        document["report"] = report
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _quiet(*_: object) -> None:
    return None


def test_a_job_without_a_command_is_a_typo_not_a_job(tmp_path):
    path = _queue(tmp_path, [{"id": "sem-comando"}])
    with pytest.raises(ValueError, match="has no"):
        lpm_queue.load(path)
    with pytest.raises(FileNotFoundError, match="not there"):
        lpm_queue.load(tmp_path / "nao-existe.json")


def test_the_jobs_run_in_order_and_each_one_is_on_record(tmp_path):
    out = tmp_path / "campaign"
    out.mkdir()
    trace = out / "trace.txt"
    jobs = [{"id": "a", "command": ["/bin/sh", "-c", f"echo a >> {trace}"]},
            {"id": "b", "command": ["/bin/sh", "-c", f"echo b >> {trace}"], "depends_on": ["a"]}]
    path = _queue(tmp_path, jobs)
    assert lpm_queue.run_queue(out, path, report=False, echo=_quiet) == 0
    assert trace.read_text(encoding="utf-8").split() == ["a", "b"]
    document = json.loads(path.read_text(encoding="utf-8"))
    assert [job["status"] for job in document["jobs"]] == ["done", "done"]
    assert all(job["started_at"] and job["finished_at"] for job in document["jobs"])


def test_a_failure_marks_the_job_and_leaves_the_next_one_blocked(tmp_path):
    out = tmp_path / "campaign"
    out.mkdir()
    jobs = [{"id": "ruim", "command": ["/bin/sh", "-c", "echo 'sem solver' >&2; exit 3"]},
            {"id": "depois", "command": ["/usr/bin/true"], "depends_on": ["ruim"]}]
    path = _queue(tmp_path, jobs)
    assert lpm_queue.run_queue(out, path, report=False, echo=_quiet) == 1
    document = json.loads(path.read_text(encoding="utf-8"))
    failed, blocked = document["jobs"]
    assert failed["status"] == "failed" and "exit 3" in failed["detail"]
    assert "sem solver" in failed["detail"]                      # the last line of the log travels with the failure
    assert blocked["status"] == "blocked" and "ruim" in blocked["detail"]


def test_a_job_whose_output_is_already_there_is_skipped(tmp_path):
    out = tmp_path / "campaign"
    out.mkdir()
    (out / "pronto.json").write_text("{}", encoding="utf-8")
    path = _queue(tmp_path, [{"id": "feito", "command": ["/usr/bin/true"], "skip_if": str(out / "pronto.json")}])
    lpm_queue.run_queue(out, path, report=False, echo=_quiet)
    assert json.loads(path.read_text(encoding="utf-8"))["jobs"][0]["status"] == "skipped"


def test_a_live_campaign_lock_is_what_the_queue_waits_for(tmp_path):
    out = tmp_path / "campaign"
    out.mkdir()
    assert lpm_queue.campaign_alive(out) is False                  # no lock at all
    (out / ".campaign.lock").write_text("999999 2026-11-28T20:00\n", encoding="utf-8")
    assert lpm_queue.campaign_alive(out) is False                  # a pid that is gone is a stale lock
    (out / ".campaign.lock").write_text(f"{os.getpid()} 2026-11-28T20:00\n", encoding="utf-8")
    assert lpm_queue.campaign_alive(out) is True


def test_the_report_is_rebuilt_after_each_job(tmp_path):
    out = tmp_path / "campaign"
    out.mkdir()
    rebuilt = out / "rebuilt.txt"
    # ``{out}`` is substituted by the runner, which is how the real queue passes the campaign directory
    report = {"cwd": str(tmp_path), "command": ["/bin/sh", "-c", f"echo {{out}} >> {rebuilt}"]}
    path = _queue(tmp_path, [{"id": "um", "command": ["/usr/bin/true"]},
                             {"id": "dois", "command": ["/usr/bin/true"]}], report=report)
    lpm_queue.run_queue(out, path, echo=_quiet)
    lines = rebuilt.read_text(encoding="utf-8").split()
    assert len(lines) == 2 and all(line == str(out) for line in lines)   # once per job, {out} substituted
