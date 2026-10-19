"""A persistent job queue for app servers: SQLite state, thread or process workers, progress, logs, cancel, retry.

    queue = JobQueue("/data/jobs.sqlite", app="campaign", max_concurrent=2)
    queue.register("plan", plan_campaign)               # fn(params: dict, ctx: JobContext) -> dict (JSON-able)
    job_id = queue.submit("plan", {"n": 40}, owner="key:ab12cd34")
    queue.wait(job_id)["status"]                        # "done"

``fn`` reports through ``ctx.progress(0.3, "sampling")`` and ``ctx.log("...")``; ``ctx.progress`` raises
:class:`JobCancelled` once a cancel was requested, so cancellation is cooperative in thread mode. In process mode
(``executor="process"``, ``fn`` must be importable at module level) the job runs in its own interpreter and cancel
or the time limit kills it.

The state lives in SQLite (WAL), so several server workers (``uvicorn --workers 2``) share one queue: a job is
claimed atomically, ``max_concurrent`` holds across all of them, and each running job has a heartbeat. A job whose
worker died (restart, crash) is found by its stale heartbeat and put back in the queue (``resume_interrupted``),
at most ``max_retries + 1`` extra times, then marked failed.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import signal
import sqlite3
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .db import LocalDB, connect, transaction

_log = logging.getLogger("pinneapple.jobs")

STATUSES = ("queued", "running", "done", "failed", "cancelled")
FINAL = ("done", "failed", "cancelled")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, app TEXT, kind TEXT NOT NULL, params TEXT, status TEXT NOT NULL,
    progress REAL DEFAULT 0, message TEXT DEFAULT '', result TEXT, error TEXT,
    attempts INTEGER DEFAULT 0, max_retries INTEGER DEFAULT 0, owner TEXT DEFAULT '',
    created REAL, started REAL, finished REAL, heartbeat REAL, cancel_requested INTEGER DEFAULT 0,
    worker TEXT, max_seconds REAL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created);
CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner, created);
CREATE TABLE IF NOT EXISTS job_logs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, ts REAL, attempt INTEGER, line TEXT
);
CREATE INDEX IF NOT EXISTS job_logs_job ON job_logs(job_id, seq);
"""


class JobCancelled(Exception):
    """Raised inside a job by ``ctx.progress`` / ``ctx.check_cancel`` after a cancel request."""


class JobTimeout(JobCancelled):
    """Raised inside a thread job that ran past its time limit."""


class JobContext:
    """What a running job sees: its id, parameters, a work directory, progress and log reporting."""

    def __init__(self, db_path: str | Path, job_id: str, params: dict[str, Any], workdir: Path, attempt: int,
                 deadline: float | None = None):
        self.db_path = str(db_path)
        self.job_id = job_id
        self.params = params
        self.workdir = workdir
        self.attempt = attempt
        self.deadline = deadline
        self._conn: sqlite3.Connection | None = None
        self._last_write = 0.0

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = connect(self.db_path)
        return self._conn

    @property
    def cancelled(self) -> bool:
        row = self.conn.execute("SELECT cancel_requested FROM jobs WHERE id=?", (self.job_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def check_cancel(self) -> None:
        if self.deadline is not None and time.time() > self.deadline:
            raise JobTimeout("time limit reached")
        if self.cancelled:
            raise JobCancelled("cancelled on request")

    def progress(self, fraction: float, message: str | None = None) -> None:
        """Report progress in [0, 1] (written at most every 0.2 s) and stop here if a cancel was requested."""
        now = time.time()
        if now - self._last_write >= 0.2 or fraction >= 1.0 or message:
            frac = min(max(float(fraction), 0.0), 1.0)
            if message is None:
                self.conn.execute("UPDATE jobs SET progress=?, heartbeat=? WHERE id=?", (frac, now, self.job_id))
            else:
                self.conn.execute("UPDATE jobs SET progress=?, message=?, heartbeat=? WHERE id=?",
                                  (frac, str(message)[:500], now, self.job_id))
            self._last_write = now
        self.check_cancel()

    def log(self, line: str) -> None:
        for part in str(line).splitlines() or [""]:
            self.conn.execute("INSERT INTO job_logs(job_id, ts, attempt, line) VALUES (?,?,?,?)",
                              (self.job_id, time.time(), self.attempt, part[:4000]))

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


JobFn = Callable[[dict[str, Any], JobContext], Any]


class JobQueue:
    """Persistent queue with at most ``max_concurrent`` running jobs (across every process sharing ``db_path``)."""

    def __init__(self, db_path: str | Path, *, app: str = "", max_concurrent: int = 2, executor: str = "thread",
                 workdir: str | Path | None = None, poll_interval: float = 0.2, heartbeat_interval: float = 2.0,
                 stale_after: float = 30.0, resume_interrupted: bool = True, max_seconds: float | None = None,
                 limits: dict[str, float] | None = None, log_tracebacks: bool = True):
        """``limits`` (process executor): ``memory_mb`` (address space), ``cpu_seconds``, ``max_files`` per job
        process, applied with ``setrlimit`` before the job runs; a job and every process it starts are killed
        together. ``log_tracebacks=False`` keeps tracebacks in the server log and out of the job log the caller
        can read."""
        if executor not in ("thread", "process"):
            raise ValueError("executor must be 'thread' or 'process'")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.app = app
        self.max_concurrent = max(1, int(max_concurrent))
        self.executor = executor
        self.workdir = Path(workdir) if workdir else self.db_path.parent / "jobs"
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.stale_after = stale_after
        self.resume_interrupted = resume_interrupted
        self.max_seconds = max_seconds
        self.limits = {k: float(v) for k, v in (limits or {}).items() if v}
        self.log_tracebacks = log_tracebacks
        self.worker_id = f"{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self._kinds: dict[str, tuple[JobFn, str]] = {}
        self._active: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._db = LocalDB(self.db_path, _SCHEMA)

    # -- setup ---------------------------------------------------------------------------------------------------
    def register(self, kind: str, fn: JobFn, *, executor: str | None = None) -> None:
        """Register the function that runs jobs of ``kind`` (per-kind ``executor`` overrides the queue's)."""
        ex = executor or self.executor
        if ex == "process" and (fn.__name__ == "<lambda>" or "<locals>" in fn.__qualname__):
            raise ValueError(f"job kind {kind!r}: a process job needs a module-level function")
        self._kinds[kind] = (fn, ex)
        self.start()

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name=f"jobqueue-{self.app}", daemon=True)
            self._thread.start()

    def stop(self, wait: float = 5.0) -> None:
        """Stop dispatching (running jobs keep going in their threads/processes)."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(wait)

    # -- client API ----------------------------------------------------------------------------------------------
    def submit(self, kind: str, params: dict[str, Any] | None = None, *, owner: str = "", max_retries: int = 0,
               max_seconds: float | None = None) -> str:
        if kind not in self._kinds:
            raise KeyError(f"unknown job kind {kind!r}; registered: {sorted(self._kinds)}")
        job_id = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        limit = max_seconds or self.max_seconds
        if self.max_seconds and limit:
            limit = min(limit, self.max_seconds)
        self._db.conn.execute(
            "INSERT INTO jobs(id, app, kind, params, status, owner, created, max_retries, max_seconds) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (job_id, self.app, kind, json.dumps(params or {}, default=_json_default), "queued", owner, time.time(),
             int(max_retries), limit))
        self._wake.set()
        return job_id

    def get(self, job_id: str) -> dict[str, Any]:
        row = self._db.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown job {job_id!r}")
        return _row(row)

    def list(self, *, owner: str | None = None, status: str | None = None, kind: str | None = None,
             limit: int = 50) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM jobs WHERE 1=1", []
        for col, val in (("owner", owner), ("status", status), ("kind", kind)):
            if val is not None:
                sql += f" AND {col}=?"
                args.append(val)
        sql += " ORDER BY created DESC LIMIT ?"
        args.append(int(limit))
        return [_row(r, with_result=False) for r in self._db.conn.execute(sql, args)]

    def logs(self, job_id: str, since: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        """Log lines after sequence number ``since`` (pass the last ``seq`` back to follow a log)."""
        rows = self._db.conn.execute("SELECT seq, ts, attempt, line FROM job_logs WHERE job_id=? AND seq>? "
                                  "ORDER BY seq LIMIT ?", (job_id, int(since), int(limit)))
        return [dict(r) for r in rows]

    def log(self, job_id: str, line: str, attempt: int = 0) -> None:
        self._db.conn.execute("INSERT INTO job_logs(job_id, ts, attempt, line) VALUES (?,?,?,?)",
                           (job_id, time.time(), attempt, str(line)[:4000]))

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued job now, or ask a running one to stop. A finished job is left as it is."""
        self.get(job_id)
        with transaction(self._db.conn):
            self._db.conn.execute("UPDATE jobs SET status='cancelled', finished=?, message='cancelled before start' "
                               "WHERE id=? AND status='queued'", (time.time(), job_id))
            self._db.conn.execute("UPDATE jobs SET cancel_requested=1 WHERE id=? AND status='running'", (job_id,))
        return self.get(job_id)

    def retry(self, job_id: str) -> dict[str, Any]:
        """Put a failed or cancelled job back in the queue (same id, its log keeps the earlier attempts)."""
        job = self.get(job_id)
        if job["status"] not in ("failed", "cancelled"):
            raise ValueError(f"job {job_id} is {job['status']}; only failed or cancelled jobs can be retried")
        self._db.conn.execute("UPDATE jobs SET status='queued', cancel_requested=0, error=NULL, result=NULL, "
                           "progress=0, message='retry requested', finished=NULL WHERE id=?", (job_id,))
        self._wake.set()
        return self.get(job_id)

    def wait(self, job_id: str, timeout: float = 60.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while True:
            job = self.get(job_id)
            if job["status"] in FINAL or time.time() >= deadline:
                return job
            time.sleep(0.05)

    def stats(self) -> dict[str, Any]:
        counts = {s: 0 for s in STATUSES}
        for r in self._db.conn.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status"):
            counts[r["status"]] = r["n"]
        return {"counts": counts, "max_concurrent": self.max_concurrent, "kinds": sorted(self._kinds)}

    # -- dispatcher ----------------------------------------------------------------------------------------------
    def _loop(self) -> None:
        last_maintenance = 0.0
        while not self._stop.is_set():
            try:
                now = time.time()
                if now - last_maintenance >= self.heartbeat_interval:
                    self._heartbeat()
                    self._recover_stale()
                    last_maintenance = now
                while self._claim_and_start():
                    pass
            except sqlite3.Error:  # pragma: no cover - a locked or busy database: try again next tick
                pass
            self._wake.wait(self.poll_interval)
            self._wake.clear()

    def _heartbeat(self) -> None:
        with self._lock:
            ids = [j for j, t in self._active.items() if t.is_alive()]
        if ids:
            self._db.conn.execute(f"UPDATE jobs SET heartbeat=? WHERE id IN ({','.join('?' * len(ids))})",
                               (time.time(), *ids))

    def _recover_stale(self) -> None:
        with self._lock:
            mine = set(self._active)
        cutoff = time.time() - self.stale_after
        with transaction(self._db.conn):
            rows = self._db.conn.execute("SELECT id, attempts, max_retries FROM jobs WHERE status='running' "
                                      "AND COALESCE(heartbeat, started, 0) < ?", (cutoff,)).fetchall()
            for r in rows:
                if r["id"] in mine:
                    continue
                if self.resume_interrupted and r["attempts"] <= r["max_retries"] + 1:
                    self._db.conn.execute("UPDATE jobs SET status='queued', worker=NULL, message=? WHERE id=?",
                                       ("resumed after its worker stopped (restart or crash)", r["id"]))
                else:
                    self._db.conn.execute("UPDATE jobs SET status='failed', finished=?, error=? WHERE id=?",
                                       (time.time(), "interrupted: its worker stopped (restart or crash)", r["id"]))
                self.log(r["id"], "worker lost (stale heartbeat)", r["attempts"])

    def _claim_and_start(self) -> bool:
        if not self._kinds:
            return False
        kinds = list(self._kinds)
        with transaction(self._db.conn):
            running = self._db.conn.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0]
            if running >= self.max_concurrent:
                return False
            row = self._db.conn.execute(
                f"SELECT * FROM jobs WHERE status='queued' AND kind IN ({','.join('?' * len(kinds))}) "
                "ORDER BY created LIMIT 1", kinds).fetchone()
            if row is None:
                return False
            now = time.time()
            self._db.conn.execute("UPDATE jobs SET status='running', worker=?, started=?, heartbeat=?, "
                               "attempts=attempts+1, cancel_requested=0, progress=0 WHERE id=?",
                               (self.worker_id, now, now, row["id"]))
        job = self.get(row["id"])
        t = threading.Thread(target=self._run, args=(job,), name=f"job-{job['id']}", daemon=True)
        with self._lock:
            self._active[job["id"]] = t
        t.start()
        return True

    def _run(self, job: dict[str, Any]) -> None:
        job_id, attempt = job["id"], job["attempts"]
        fn, ex = self._kinds[job["kind"]]
        workdir = self.workdir / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + job["max_seconds"] if job["max_seconds"] else None
        self.log(job_id, f"attempt {attempt} started ({ex})", attempt)
        try:
            if ex == "process":
                outcome = self._run_process(fn, job, workdir, deadline)
            else:
                outcome = self._run_thread(fn, job, workdir, deadline)
            self._finish(job, *outcome)
        except Exception as exc:  # noqa: BLE001 - a broken queue must not kill the dispatcher
            self._finish(job, "error", f"{type(exc).__name__}: {exc}", traceback.format_exc())
        finally:
            with self._lock:
                self._active.pop(job_id, None)
            self._wake.set()

    def _run_thread(self, fn: JobFn, job: dict[str, Any], workdir: Path, deadline: float | None):
        ctx = JobContext(self.db_path, job["id"], job["params"], workdir, job["attempts"], deadline)
        try:
            return ("ok", fn(job["params"], ctx), "")
        except JobTimeout:
            return ("timeout", f"time limit of {job['max_seconds']:g} s reached", "")
        except JobCancelled:
            return ("cancelled", "cancelled on request", "")
        except Exception as exc:  # noqa: BLE001 - reported as the job's error
            return ("error", f"{type(exc).__name__}: {exc}", traceback.format_exc())
        finally:
            ctx.close()

    def _run_process(self, fn: JobFn, job: dict[str, Any], workdir: Path, deadline: float | None):
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        ref = f"{fn.__module__}:{fn.__qualname__}"
        proc = ctx.Process(target=_process_entry, daemon=True,
                           args=(child, ref, str(self.db_path), job["id"], job["params"], str(workdir),
                                 job["attempts"], self.limits))
        proc.start()
        child.close()
        outcome = None
        while proc.is_alive() or parent.poll():
            if parent.poll(0.2):
                try:
                    outcome = parent.recv()
                except EOFError:
                    break
                proc.join(5)
                break
            if deadline is not None and time.time() > deadline:
                _kill_group(proc)
                return ("timeout", f"killed after the time limit of {job['max_seconds']:g} s", "")
            row = self._db.conn.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job["id"],)).fetchone()
            if row and row["cancel_requested"]:
                _kill_group(proc)
                return ("cancelled", "cancelled on request (process killed)", "")
        if outcome is None:
            proc.join(5)
            tail = _tail(workdir / "output.log")
            return ("error", f"job process exited with code {proc.exitcode}", tail)
        return outcome

    def _finish(self, job: dict[str, Any], kind: str, payload: Any, detail: str) -> None:
        job_id, attempt = job["id"], job["attempts"]
        now = time.time()
        if detail and self.log_tracebacks:
            for line in detail.rstrip().splitlines()[-40:]:
                self.log(job_id, line, attempt)
        elif detail:
            _log.error("job %s (%s) attempt %d failed:\n%s", job_id, self.app, attempt, detail.rstrip())
            self.log(job_id, "details are in the server log", attempt)
        if kind == "ok":
            try:
                result = json.dumps(payload, default=_json_default)
            except (TypeError, ValueError) as exc:
                kind, payload = "error", f"result is not JSON-serialisable: {exc}"
            else:
                # the final log line first: a caller that sees the final status also sees the whole log
                self.log(job_id, f"attempt {attempt} done", attempt)
                self._db.conn.execute("UPDATE jobs SET status='done', result=?, progress=1, finished=?, error=NULL "
                                   "WHERE id=?", (result, now, job_id))
                return
        if kind in ("cancelled",):
            self.log(job_id, payload, attempt)
            self._db.conn.execute("UPDATE jobs SET status='cancelled', finished=?, error=?, message=? WHERE id=?",
                               (now, None, payload, job_id))
            return
        current = self.get(job_id)
        if attempt <= current["max_retries"] and not current["cancel_requested"]:
            self.log(job_id, f"attempt {attempt} failed ({payload}); retrying", attempt)
            self._db.conn.execute("UPDATE jobs SET status='queued', error=?, message=? WHERE id=?",
                               (payload, f"retrying after: {payload}"[:500], job_id))
            self._wake.set()
        else:
            self.log(job_id, f"attempt {attempt} failed: {payload}", attempt)
            self._db.conn.execute("UPDATE jobs SET status='failed', finished=?, error=? WHERE id=?",
                               (now, payload, job_id))


def _kill_group(proc) -> None:
    """Kill a job process and everything it started (it leads its own process group)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        proc.kill()
    proc.join(5)


def _apply_limits(limits: dict[str, float] | None) -> None:
    if hasattr(os, "setsid"):
        try:
            os.setsid()                                       # own process group: a kill reaches its children too
        except OSError:
            pass
    if not limits:
        return
    import resource
    for key, res, scale in (("memory_mb", resource.RLIMIT_AS, 2**20), ("cpu_seconds", resource.RLIMIT_CPU, 1),
                            ("max_files", resource.RLIMIT_NOFILE, 1)):
        if limits.get(key):
            v = int(limits[key] * scale)
            soft, hard = resource.getrlimit(res)
            if hard != resource.RLIM_INFINITY:
                v = min(v, hard)
            resource.setrlimit(res, (v, hard if hard != resource.RLIM_INFINITY else v))


def _process_entry(pipe, ref: str, db_path: str, job_id: str, params: dict[str, Any], workdir: str,
                   attempt: int, limits: dict[str, float] | None = None) -> None:
    import importlib
    _apply_limits(limits)
    out = open(Path(workdir) / "output.log", "a", buffering=1)
    os.dup2(out.fileno(), 1)
    os.dup2(out.fileno(), 2)
    ctx = JobContext(db_path, job_id, params, Path(workdir), attempt)
    try:
        module, _, qualname = ref.partition(":")
        fn: Any = importlib.import_module(module)
        for part in qualname.split("."):
            fn = getattr(fn, part)
        result = fn(params, ctx)
        json.dumps(result, default=_json_default)
        pipe.send(("ok", result, ""))
    except JobCancelled:
        pipe.send(("cancelled", "cancelled on request", ""))
    except BaseException as exc:  # noqa: BLE001 - every failure goes back to the parent
        pipe.send(("error", f"{type(exc).__name__}: {exc}", traceback.format_exc()))
    finally:
        ctx.close()
        out.flush()


def _row(row: sqlite3.Row, with_result: bool = True) -> dict[str, Any]:
    d = dict(row)
    d["params"] = json.loads(d["params"] or "{}")
    d["cancel_requested"] = bool(d["cancel_requested"])
    if with_result:
        d["result"] = json.loads(d["result"]) if d["result"] else None
    else:
        d.pop("result", None)
    return d


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except FileNotFoundError:
        return ""


def _json_default(o: Any):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"{type(o).__name__} is not JSON-serialisable")
