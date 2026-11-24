"""Structured run events and ``pinneapple runs why`` (#1481).

What is measured here is the record a failed run leaves behind — that it names the step, the attempt, the command,
the exit code and the last lines of output, and that it does so without re-running anything. The peak-memory field is
checked twice: present for a command the local executor sampled, and **absent, not zero**, for an executor that cannot
measure it.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pinneapple_core.workflows import Step, Workflow, WorkflowRunner
from pinneapple_core.workflows.events import (
    EVENT_KINDS,
    EventStream,
    OtlpExporter,
    dag_hash,
    read_events,
    tail_of,
    to_otlp,
)
from pinneapple_core.workflows.runs_cli import main as runs_main


def _runner(tmp_path, **kwargs) -> WorkflowRunner:
    return WorkflowRunner(tmp_path / "runner", **kwargs)


def _failed_run(runner: WorkflowRunner, *, retries: int = 0) -> dict:
    wf = Workflow("demo", [
        Step("fine", "command", {"argv": ["sh", "-c", "echo first"]}),
        Step("boom", "command", {"argv": ["sh", "-c", "echo 'step says no' >&2; exit 7"]}, needs=["fine"],
             retries=retries),
    ])
    return runner.run(wf)


# ------------------------------------------------------------------------------------------------ the stream
def test_every_event_of_a_failed_run_is_written_and_readable(tmp_path):
    runner = _runner(tmp_path)
    run = _failed_run(runner)
    events = read_events(tmp_path / "runner" / "runs" / run["id"])
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run.started" and kinds[-1] == "run.finished"
    assert {"step.started", "step.finished", "step.log"} <= set(kinds)
    assert set(kinds) <= set(EVENT_KINDS)

    started = events[0]
    assert started["run_id"] == run["id"] and started["dag"] == dag_hash(runner.workflow(run["id"]).to_dict())
    assert started["host"] and started["executor"] == ["local"]

    failed = next(e for e in events if e["event"] == "step.finished" and e["step_id"] == "boom")
    assert failed["status"] == "failed" and failed["exit_code"] == 7
    assert failed["seconds"] >= 0 and "peak_rss_bytes" in failed, "the local executor measures the step's memory"

    tail = next(e for e in events if e["event"] == "step.log" and e["step_id"] == "boom")
    assert any("step says no" in line for line in tail["tail"]), "the tail of the output is kept on failure"
    assert tail["reason"] == "failed"


def test_a_retry_is_recorded_with_the_reason_it_was_retried(tmp_path):
    runner = _runner(tmp_path)
    run = _failed_run(runner, retries=2)
    events = read_events(tmp_path / "runner" / "runs" / run["id"])
    retries = [e for e in events if e["event"] == "step.retry" and e["step_id"] == "boom"]
    assert [e["attempt"] for e in retries] == [1, 2]
    assert "exited with code 7" in retries[0]["why"]
    finished = [e for e in events if e["event"] == "step.finished" and e["step_id"] == "boom"][-1]
    assert "3 attempt" in finished["error"]


def test_peak_memory_is_absent_not_zero_when_the_executor_cannot_measure_it(tmp_path):
    runner = _runner(tmp_path)
    wf = Workflow("dry", [Step("sbatch", "command", {"argv": ["echo", "hi"]}, executor="slurm",
                               executor_options={"dry_run": True})])
    run = runner.run(wf)
    events = read_events(tmp_path / "runner" / "runs" / run["id"])
    finished = next(e for e in events if e["event"] == "step.finished" and e["step_id"] == "sbatch")
    assert finished["status"] == "done"
    assert "peak_rss_bytes" not in finished, "slurm wrote a script and measured nothing: the field is absent"
    assert finished["seconds"] >= 0, "wall time is always there"


def test_cancelling_by_request_is_an_event(tmp_path):
    runner = _runner(tmp_path)
    wf = Workflow("slow", [Step("wait", "command", {"argv": ["sh", "-c", "sleep 5"]}),
                           Step("after", "command", {"argv": ["echo", "x"]}, needs=["wait"])])
    run = runner.run(wf, should_stop=lambda: True)
    events = read_events(tmp_path / "runner" / "runs" / run["id"])
    assert any(e["event"] == "step.cancelled" for e in events)
    assert run["status"] == "cancelled"


def test_the_event_stream_never_costs_a_run_and_survives_a_truncated_line(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventStream(path) as stream:
        stream.emit("run.started", run_id="r1")
        assert stream.emit("step.started", run_id="r1", step_id="a")["seq"] == 2
    with open(path, "a") as fh:
        fh.write('{"event": "step.finished", "run_id"')       # a crash mid-write
    events = read_events(path)
    assert [e["event"] for e in events] == ["run.started", "step.started"]
    with pytest.raises(ValueError):
        EventStream(None).emit("nothing.like.this")
    assert read_events(tmp_path / "absent") == []


def test_tail_is_bounded():
    text = "\n".join(f"line {i}" for i in range(1000))
    tail = tail_of(text)
    assert len(tail) == 20 and tail[-1] == "line 999"
    assert sum(len(line) for line in tail) <= 4000
    assert tail_of("x" * 10000) == []


def test_the_event_stream_adds_no_measurable_time_to_a_step(tmp_path):
    """A successful run's events must not show up in the run's wall time (#1481, acceptance)."""
    def measure(events: bool) -> float:
        runner = WorkflowRunner(tmp_path / f"r{events}", events=events)
        wf = Workflow("six", [Step(f"s{i}", "command", {"argv": ["sh", "-c", "sleep 0.05"]}) for i in range(6)])
        t0 = time.perf_counter()
        runner.run(wf)
        return time.perf_counter() - t0

    off, on = measure(False), measure(True)
    assert on < off + 0.5, f"events cost {on - off:.3f} s over six 50 ms steps"


# ------------------------------------------------------------------------------------------------ the command line
def test_runs_show_and_runs_why_explain_a_failed_run_from_its_record(tmp_path, capsys):
    runner = _runner(tmp_path)
    run = _failed_run(runner)
    root = str(tmp_path / "runner")

    assert runs_main(["--root", root, "show", run["id"]]) == 0
    out = capsys.readouterr().out
    assert run["id"] in out and "failed" in out and "local" in out

    assert runs_main(["--root", root, "why", run["id"]]) == 0
    why = capsys.readouterr().out
    assert "step boom failed" in why
    assert "exit code   7" in why, "the exit code comes from the record"
    assert "command     " in why and "exit 7" in why, "the command that ran is named"
    assert "step says no" in why, "the last output of the step is shown"
    assert "kind        command" in why

    assert runs_main(["--root", root, "show", "run-does-not-exist"]) == 2
    assert runs_main(["--root", root, "list"]) == 0
    assert run["id"] in capsys.readouterr().out


def test_runs_why_falls_back_to_the_step_log_when_the_run_had_events_off(tmp_path, capsys):
    runner = WorkflowRunner(tmp_path / "quiet", events=False)
    run = _failed_run(runner)
    assert runs_main(["--root", str(tmp_path / "quiet"), "why", run["id"]]) == 0
    out = capsys.readouterr().out
    assert "step boom failed" in out and "step says no" in out
    assert "no events recorded" in out
    assert runs_main(["--root", str(tmp_path / "quiet"), "show", run["id"]]) == 0
    assert "no events recorded" in capsys.readouterr().out


def test_runs_why_says_there_is_nothing_to_explain_for_a_run_that_finished(tmp_path, capsys):
    runner = _runner(tmp_path)
    run = runner.run(Workflow("ok", [Step("a", "command", {"argv": ["echo", "done"]})]))
    assert runs_main(["--root", str(tmp_path / "runner"), "why", run["id"]]) == 0
    assert "nothing to explain" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ OTLP
class _Collector(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):                                       # noqa: N802 - the name http.server wants
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Collector.received.append({"path": self.path, "body": json.loads(body)})
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):                            # keep the test output clean
        pass


def test_otlp_export_posts_spans_to_a_collector(tmp_path):
    _Collector.received = []
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    runner = _runner(tmp_path)
    run = _failed_run(runner)
    events = read_events(tmp_path / "runner" / "runs" / run["id"])
    result = OtlpExporter(f"http://127.0.0.1:{server.server_port}").export(events)
    thread.join(5)
    server.server_close()
    assert result["ok"] and result["status"] == 200
    assert _Collector.received and _Collector.received[0]["path"] == "/v1/traces"
    spans = _Collector.received[0]["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    names = [s["name"] for s in spans]
    assert any(name.startswith("run ") for name in names) and "step boom" in names
    trace = spans[0]["traceId"]
    assert len(trace) == 32 and all(s["traceId"] == trace for s in spans)
    failed = next(s for s in spans if s["name"] == "step boom")
    assert failed["status"]["code"] == 2


def test_otlp_export_reports_an_unreachable_collector_instead_of_pretending():
    result = OtlpExporter("http://127.0.0.1:9", timeout=0.5).export([])
    assert not result["ok"] and result["error"]


def test_to_otlp_marks_a_successful_run_as_ok():
    events = [{"event": "run.started", "ts": 1.0, "run_id": "r", "host": "h"},
              {"event": "run.finished", "ts": 2.0, "run_id": "r", "status": "done"},
              {"event": "step.started", "ts": 1.1, "run_id": "r", "step_id": "a", "kind": "command"},
              {"event": "step.finished", "ts": 1.9, "run_id": "r", "step_id": "a", "status": "done", "seconds": 0.8}]
    payload = to_otlp(events)
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert [s["status"]["code"] for s in spans] == [1, 1]
    step = spans[1]
    assert step["parentSpanId"] == spans[0]["spanId"]
    attrs = {a["key"]: a["value"] for a in step["attributes"]}
    assert attrs["kind"]["stringValue"] == "command"
