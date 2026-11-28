"""Run a list of campaign jobs one at a time: the next starts when the one before it ends, and each one is on record.

The campaign stops when it is killed and picks up where it stopped, so the thing that is missing overnight is not
resilience but *sequence*: the wing's refinement has to finish before the comparison against the outside corpus means
anything, the airfoil refit has to wait until no other fit is using the machine, and the report has to be rebuilt after
each of them. That is a queue, and this is it — a JSON list of jobs, a runner that holds the lock while one runs, and
`queue.json` itself as the record of what happened.

    python -m benchmarks.lpm_queue --out ~/lpm-campaign                    # run the queue that is there
    python -m benchmarks.lpm_queue --out ~/lpm-campaign --detach queue.log  # outlive the shell
    python -m benchmarks.lpm_queue --out ~/lpm-campaign --status            # what is queued, what ran

A job is ``{"id", "title", "cwd", "command", "status"}`` and optionally ``"log"``, ``"depends_on"`` and ``"skip_if"``.
``command`` is an argument vector, never a shell string: a job that needs a shell says so by asking for ``sh -c``, so
the queue cannot quietly turn a path with a space into two arguments.

Two waits are built in, because both were learnt the hard way:

* it waits for the campaign's own lock (``<out>/.campaign.lock``) to be free before every job, so a queue job never
  competes with a campaign for the same run directory or for the machine;
* it stops after three consecutive failures. A queue that keeps starting a job that cannot run is not a queue, it is a
  loop with a log file, and the third failure is the point where a person has to look.

The report hook is a job like any other and is simply run after each one, so the PDF in ``~/Downloads`` is never more
than one job out of date.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks import _daemon

__all__ = ["QUEUE_SCHEMA", "load", "save", "campaign_alive", "run_job", "run_queue", "main"]

QUEUE_SCHEMA = "pinneapple.lpm.queue/1"
JOB_STATUSES = ("pending", "blocked", "running", "done", "skipped", "failed")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load(path: Path) -> dict[str, Any]:
    """The queue file, with the schema checked: a job without an id or a command is a typo, not a job."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not there; a queue is a file with the jobs in it")
    document = json.loads(path.read_text(encoding="utf-8"))
    document.setdefault("schema", QUEUE_SCHEMA)
    for index, job in enumerate(document.get("jobs") or []):
        missing = [key for key in ("id", "command") if not job.get(key)]
        if missing:
            raise ValueError(f"{path}: job {index} has no {missing}")
    return document


def save(path: Path, document: dict[str, Any]) -> None:
    """Write the queue back, atomically: a half-written queue is a queue that loses a job on the next read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=1), encoding="utf-8")
    temporary.replace(path)


def campaign_alive(out: Path) -> bool:
    """Whether a campaign holds this directory's lock — the same test the guard script uses."""
    lock = out / ".campaign.lock"
    if not lock.is_file():
        return False
    try:
        pid = int(lock.read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_job(job: dict[str, Any], out: Path, *, echo=None) -> tuple[str, str]:
    """Run one job; returns ``(status, detail)``. The log is the job's own, so a failure names its own file."""
    skip_if = job.get("skip_if")
    if skip_if and Path(skip_if).expanduser().exists():
        return "skipped", f"{skip_if} is already there"
    command = [str(part) for part in job["command"]]
    cwd = Path(job.get("cwd") or out).expanduser()
    log = Path(job.get("log") or (out / "queue" / f"{job['id']}.log")).expanduser()
    log.parent.mkdir(parents=True, exist_ok=True)
    if echo:
        echo(f"{job['id']}: {' '.join(command)}")
    environment = dict(os.environ, PINNEAPPLE_QUIET_EXPERIMENTAL="1", MPLBACKEND="Agg")
    with open(log, "ab") as handle:
        handle.write(f"\n=== {_now()} {' '.join(command)}\n".encode())
        handle.flush()
        finished = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT,
                                  env=environment, check=False)
    if finished.returncode == 0:
        return "done", f"exit 0, log {log}"
    tail = ""
    try:
        tail = " | ".join(log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:])
    except OSError:
        pass
    return "failed", f"exit {finished.returncode}, log {log}{f' — {tail[-200:]}' if tail else ''}"


def run_queue(out: Path, queue_path: Path, *, idle_s: float = 60.0, report: bool = True, echo=print) -> int:
    """Work through the queue; ``0`` when everything that could run, ran."""
    document = load(queue_path)
    jobs = document["jobs"]
    failures = 0
    for job in jobs:
        if job.get("status") in ("done", "skipped"):
            continue
        waiting = [dep for dep in job.get("depends_on") or []
                   if next((j.get("status") for j in jobs if j["id"] == dep), None) not in ("done", "skipped")]
        if waiting:
            job["status"] = "blocked"
            job["detail"] = f"waiting for {', '.join(waiting)}"
            save(queue_path, document)
            echo(f"{job['id']}: blocked on {', '.join(waiting)}; not run")
            continue
        while campaign_alive(out):
            echo(f"{job['id']}: a campaign holds {out}/.campaign.lock; waiting {idle_s:.0f}s")
            time.sleep(idle_s)
        job |= {"status": "running", "started_at": _now(), "finished_at": None, "detail": ""}
        save(queue_path, document)
        status, detail = run_job(job, out, echo=echo)
        job |= {"status": status, "finished_at": _now(), "detail": detail}
        save(queue_path, document)
        echo(f"{job['id']}: {status} — {detail}")
        failures = failures + 1 if status == "failed" else 0
        if report and document.get("report"):
            _run_report(document["report"], out, echo=echo)
        if failures >= 3:
            echo("three failures in a row: stopping, this needs a person")
            break
    document["updated_at"] = _now()
    save(queue_path, document)
    return 0 if all(job.get("status") in ("done", "skipped", "blocked") for job in jobs) else 1


def _run_report(report: dict[str, Any], out: Path, *, echo) -> None:
    """Rebuild the PDF after a job. A report that fails is logged and never stops the queue."""
    command = [str(part).replace("{out}", str(out)) for part in report["command"]]
    cwd = Path(report.get("cwd") or out).expanduser()
    log = Path(report.get("log") or (out / "queue" / "report.log")).expanduser()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as handle:
        finished = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, PINNEAPPLE_QUIET_EXPERIMENTAL="1", MPLBACKEND="Agg"),
                                  check=False)
    echo(f"report: {'rebuilt' if finished.returncode == 0 else f'failed (exit {finished.returncode})'}, log {log}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the campaign's queued jobs one at a time")
    parser.add_argument("--out", type=Path, default=Path.home() / "lpm-campaign", help="the campaign directory")
    parser.add_argument("--queue", type=Path, default=None, help="the queue file (default: <out>/queue/queue.json)")
    parser.add_argument("--idle", type=float, default=60.0, metavar="SECONDS",
                        help="how often to look again while a campaign holds the lock")
    parser.add_argument("--status", action="store_true", help="print the queue and exit")
    parser.add_argument("--no-report", action="store_true", help="do not rebuild the report after each job")
    parser.add_argument("--detach", type=Path, metavar="LOG", help="run in its own session, so it outlives the shell")
    args = parser.parse_args(argv)

    out = args.out.expanduser().resolve()
    queue_path = (args.queue or out / "queue" / "queue.json").expanduser()
    if args.status:
        document = load(queue_path)
        for job in document["jobs"]:
            print(f"{job.get('status', 'pending'):9s} {job['id']:24s} {job.get('title', '')}"
                  f"{('  — ' + job['detail']) if job.get('detail') else ''}")
        return 0
    if args.detach:
        _daemon.detach("benchmarks.lpm_queue", args.detach.expanduser(), marker="LPM_QUEUE_DETACHED")
        return 0
    return run_queue(out, queue_path, idle_s=args.idle, report=not args.no_report)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
