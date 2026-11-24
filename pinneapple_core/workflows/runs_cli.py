"""``pinneapple runs ...``: read a finished run's record and explain it (#1481).

    pinneapple runs list                       # the runs under this root, newest first
    pinneapple runs show <id>                  # what happened: status, wall time, per-step table
    pinneapple runs why <id>                   # a failed run, explained: step, attempt, command, exit code, tail
    pinneapple runs show <id> --otlp URL        # post the run's spans to an OTLP/HTTP collector

The root is ``--root DIR``, or the nearest folder upwards that holds a ``runner.sqlite``, or the run's own directory when
``<id>`` is a path. Everything printed comes from the run's record: the database and ``events.jsonl``. Nothing here
re-runs a step, which is the acceptance line — "a failed run is explainable from its run record alone".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .engine import WorkflowRunner
from .events import OtlpExporter, read_events

__all__ = ["main", "find_root"]

#: A command line as the executors log it: ``$ argv``, with the runner's ``HH:MM:SS `` prefix in front.
_COMMAND_LINE = re.compile(r"^(?:\d{2}:\d{2}:\d{2} )?\$ ")


def find_root(explicit: str | None = None, start: str | Path | None = None) -> Path | None:
    """The runner root: ``--root``, else the nearest ancestor of ``start`` with a ``runner.sqlite``."""
    if explicit:
        return Path(explicit).expanduser()
    here = Path(start or Path.cwd()).expanduser().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "runner.sqlite").is_file():
            return candidate
        if (candidate / "runs").is_dir() and any((candidate / "runs").glob("*/events.jsonl")):
            return candidate
    return None


def _resolve(args: argparse.Namespace) -> tuple[Path, str] | None:
    target = Path(args.run_id).expanduser()
    if target.is_dir():                                 # a path to the run's own folder
        run_id = target.name
        root = find_root(args.root, start=target.parent.parent if target.parent.name == "runs" else target.parent)
        return (root, run_id) if root is not None else None
    root = find_root(args.root)
    return (root, args.run_id) if root is not None else None


def _events(root: Path, run_id: str) -> list[dict[str, Any]]:
    return read_events(root / "runs" / run_id)


def _human_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 120:
        return f"{seconds:.2f} s"
    return f"{seconds // 60:.0f} min {seconds % 60:.0f} s"


def _mb(value: Any) -> str:
    if not isinstance(value, int | float):
        return "-"
    return f"{value / 1e6:.1f} MB"


def _show(root: Path, run_id: str, as_json: bool, otlp: str | None) -> int:
    runner = WorkflowRunner(root, events=False)
    try:
        run = runner.get_run(run_id)
    except KeyError:
        print(f"error: no run {run_id!r} under {root}", file=sys.stderr)
        return 2
    events = _events(root, run_id)
    started = next((e for e in events if e["event"] == "run.started"), {})
    finished = next((e for e in events if e["event"] == "run.finished"), {})
    steps = runner.steps(run_id)
    step_events: dict[str, dict[str, Any]] = {}
    for e in events:
        if e["event"] == "step.finished" and e.get("step_id"):
            step_events.setdefault(str(e["step_id"]), e)       # the last one wins (a resumed run)
    if as_json:
        print(json.dumps({"run": run, "started": started, "finished": finished, "steps": steps,
                          "step_events": step_events}, indent=2, default=str))
    else:
        print(f"run        {run_id}")
        print(f"workflow   {run.get('name') or '-'}")
        print(f"status     {run['status']}")
        print(f"started    {_ts(run.get('started'))}   finished {_ts(run.get('finished'))}   "
              f"wall {_human_seconds(_wall(run))}")
        print("steps      " + ", ".join(f"{k} {v}" for k, v in sorted(run["counts"].items()) if v))
        if started.get("dag"):
            print(f"dag        {str(started['dag'])[:16]}   host {started.get('host', '-')}   "
                  f"executor {', '.join(started.get('executor') or []) or '-'}")
        if started.get("plugins"):
            names = ", ".join(f"{p['name']} {p['version']}" for p in started["plugins"])
            print(f"plugins    {names}")
        print()
        rows = [("STEP", "STATUS", "ATTEMPTS", "TIME", "EXIT", "PEAK MEMORY")]
        for s in steps:
            ev = step_events.get(s["step_id"], {})
            rows.append((s["step_id"], s["status"], str(s.get("attempts") or 0),
                         _human_seconds(ev.get("seconds", s.get("seconds"))),
                         str(ev.get("exit_code", "-")), _mb(ev.get("peak_rss_bytes"))))
        for row in _table(rows):
            print(row)
        if not events:
            print("\n(no events recorded for this run: it ran with events off, or before events existed)")
    if otlp:
        result = OtlpExporter(otlp).export(events)
        if result["ok"]:
            print(f"\nOTLP: {result['spans']} span(s) accepted by {result['url']} (HTTP {result['status']})")
        else:
            print(f"\nOTLP: not exported — {result['error']}", file=sys.stderr)
            return 1
    return 0


def _why(root: Path, run_id: str) -> int:
    runner = WorkflowRunner(root, events=False)
    try:
        run = runner.get_run(run_id)
    except KeyError:
        print(f"error: no run {run_id!r} under {root}", file=sys.stderr)
        return 2
    if run["status"] == "done":
        print(f"run {run_id} finished: there is nothing to explain. `pinneapple runs show {run_id}` has the summary.")
        return 0
    events = _events(root, run_id)
    started = next((e for e in events if e["event"] == "run.started"), {})
    steps = runner.steps(run_id)
    failed = [s for s in steps if s["status"] == "failed"]
    cancelled = [s for s in steps if s["status"] == "cancelled"]
    print(f"run {run_id}: {run['status']} ({len(failed)} failed step(s), {len(cancelled)} cancelled)")
    if started.get("dag"):
        print(f"dag {str(started['dag'])[:16]} on {started.get('host', '-')}, "
              f"executor {', '.join(started.get('executor') or []) or '-'}")
    for step in failed:
        sid = step["step_id"]
        fin = next((e for e in reversed(events) if e["event"] == "step.finished" and e.get("step_id") == sid
                    and e.get("status") == "failed"), {})
        retries = [e for e in events if e["event"] == "step.retry" and e.get("step_id") == sid]
        print(f"\nstep {sid} failed")
        print(f"  kind        {_kind(events, sid)}")
        print(f"  attempts    {step.get('attempts') or 0}"
              + (f" (retried {len(retries)}x)" if retries else ""))
        print(f"  executor    {fin.get('executor', step.get('executor') or '-')}")
        print(f"  exit code   {fin.get('exit_code', '-')}"
              + ("  (timed out)" if fin.get("timed_out") else ""))
        print(f"  time        {_human_seconds(fin.get('seconds', step.get('seconds')))}"
              + (f"   peak memory {_mb(fin.get('peak_rss_bytes'))}" if fin.get("peak_rss_bytes") else
                 "   peak memory not measured by this executor"))
        error = step.get("error") or fin.get("error") or "-"
        print(f"  error       {error}")
        for line in _commands(root, run_id, sid):
            print(f"  command     {line}")
        tail = next((e for e in reversed(events) if e["event"] == "step.log" and e.get("step_id") == sid), None)
        lines = tail.get("tail") if tail else []
        if not lines:                                    # events off, or an older run: the step log is still there
            from .events import tail_of

            lines = tail_of(runner.log(run_id, sid))
        if lines:
            print("  last output")
            for line in lines:
                print(f"    {line}")
        for retry in retries:
            print(f"  attempt {retry.get('attempt')} failed: {retry.get('why')}")
    for step in cancelled:
        why = next((e.get("why") for e in reversed(events)
                    if e["event"] == "step.cancelled" and e.get("step_id") == step["step_id"]), step.get("error"))
        print(f"\nstep {step['step_id']} never ran: {why or 'cancelled'}")
    if not events:
        print("\n(no events recorded for this run: it ran with events off, or before events existed; the step table "
              "above and the per-step log.txt are all there is)")
    return 0


def _kind(events: Sequence[dict[str, Any]], step_id: str) -> str:
    return str(next((e.get("kind") for e in events if e["event"] == "step.started" and e.get("step_id") == step_id),
                    "-"))


def _commands(root: Path, run_id: str, step_id: str) -> list[str]:
    """The command lines the executor logged (``$ argv``), taken from the step's own log."""
    path = root / "runs" / run_id / step_id / "log.txt"
    if not path.is_file():
        return []
    lines = (line.strip() for line in path.read_text(errors="replace").splitlines())
    return [line for line in lines if _COMMAND_LINE.match(line)][:5]


def _ts(value: Any) -> str:
    import datetime as _dt

    if not value:
        return "-"
    return _dt.datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")


def _wall(run: dict[str, Any]) -> float | None:
    if run.get("started") and run.get("finished"):
        return float(run["finished"]) - float(run["started"])
    return None


def _table(rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    return ["  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]


def _list(root: Path) -> int:
    runner = WorkflowRunner(root, events=False)
    rows = [("RUN", "STATUS", "STARTED", "STEPS")]
    for run in runner.runs(limit=50):
        count = runner.db.conn.execute("SELECT COUNT(*) FROM steps WHERE run_id=?", (run["id"],)).fetchone()[0]
        rows.append((run["id"], run["status"], _ts(run["created"]), str(count)))
    for line in _table(rows):
        print(line)
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pinneapple runs",
                                 description="What happened in a workflow run, read from its record.")
    ap.add_argument("--root", help="the runner root (default: the nearest folder upwards with a runner.sqlite)")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the runs under this root, newest first")
    p_show = sub.add_parser("show", help="the summary of one run")
    p_show.add_argument("run_id", help="the run id, or the path of the run's folder")
    p_show.add_argument("--json", action="store_true")
    p_show.add_argument("--otlp", metavar="ENDPOINT", help="post the run's spans to an OTLP/HTTP collector")
    p_why = sub.add_parser("why", help="a failed run, explained: step, attempt, command, exit code, tail")
    p_why.add_argument("run_id")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "list":
        root = find_root(args.root)
        if root is None:
            print("error: no runner root here (--root DIR, or run this from inside one)", file=sys.stderr)
            return 2
        return _list(root)
    resolved = _resolve(args)
    if resolved is None:
        print("error: no runner root here (--root DIR, or run this from inside one)", file=sys.stderr)
        return 2
    root, run_id = resolved
    if args.command == "show":
        return _show(root, run_id, args.json, args.otlp)
    return _why(root, run_id)


if __name__ == "__main__":                              # pragma: no cover
    sys.exit(main())
