"""Run workflows: traceable runs, per-step logs and states, retries, resume, skip-done by content hash.

    runner = WorkflowRunner("/data/runner")
    summary = runner.run(workflow)                    # a new run id
    runner.run(workflow, run_id=summary["run_id"])    # resume: done steps stay done, the rest runs again
    runner.steps(run_id), runner.log(run_id, "c01.solve")

Layout: ``<root>/runner.sqlite`` (runs, steps, the content-hash cache) and ``<root>/runs/<run id>/<step id>/``
(the step's files and ``log.txt``).

**Skip-done.** A step's hash covers its kind, its parameters, the content (SHA-256) of its ``inputs`` files and the
hashes of the steps it needs. A step whose hash already finished in any run, and whose directory still exists, is
not run again: it is marked ``skipped`` and its outputs and directory are reused.

**States.** pending, running, done, failed, skipped (done earlier, reused), cancelled (by request, or because a step
it needs failed). ``fail_fast`` stops starting new steps after a failure; otherwise independent branches go on.
A run interrupted by a crash or restart has steps left ``running``: running it again with the same id resumes it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from ..service.db import LocalDB, transaction
from .adapters import StepCancelled, StepContext, StepFailed, get_adapter
from .dag import Step, Workflow
from .executors import get_executor

STATES = ("pending", "running", "done", "failed", "skipped", "cancelled")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, name TEXT, workflow TEXT, status TEXT, created REAL, started REAL, finished REAL,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT, step_id TEXT, status TEXT, hash TEXT, attempts INTEGER DEFAULT 0, started REAL, finished REAL,
    seconds REAL, outputs TEXT, error TEXT, executor TEXT, reused_from TEXT, grp TEXT,
    PRIMARY KEY (run_id, step_id)
);
CREATE TABLE IF NOT EXISTS cache (
    hash TEXT PRIMARY KEY, outputs TEXT, workdir TEXT, run_id TEXT, step_id TEXT, finished REAL, seconds REAL
);
"""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def step_hash(step: Step, upstream_hashes: dict[str, str], root: Path | None = None) -> str:
    """Content hash of a step: kind, parameters, executor options, input file contents, upstream hashes."""
    files = {}
    for f in step.inputs:
        p = Path(f) if root is None or Path(f).is_absolute() else root / f
        files[f] = sha256_file(p) if p.is_file() else "missing"
    doc = {"kind": step.kind, "params": step.params, "executor_options": step.executor_options, "files": files,
           "upstream": {k: upstream_hashes[k] for k in sorted(step.needs)}}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


class WorkflowRunner:
    def __init__(self, root: str | Path, *, cache: bool = True, input_root: str | Path | None = None):
        self.root = Path(root)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self.db = LocalDB(self.root / "runner.sqlite", _SCHEMA)
        self.cache = cache
        self.input_root = Path(input_root) if input_root else None
        self._log_lock = threading.Lock()

    # -- queries --------------------------------------------------------------------------------------------------
    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.conn.execute("SELECT id, name, status, created, started, finished FROM runs "
                                    "ORDER BY created DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.db.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id!r}")
        d = dict(row)
        d["workflow"] = json.loads(d["workflow"])
        d["metadata"] = json.loads(d["metadata"] or "{}")
        d["counts"] = {s: 0 for s in STATES}
        for r in self.db.conn.execute("SELECT status, COUNT(*) n FROM steps WHERE run_id=? GROUP BY status",
                                      (run_id,)):
            d["counts"][r["status"]] = r["n"]
        return d

    def steps(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.conn.execute("SELECT * FROM steps WHERE run_id=? ORDER BY rowid", (run_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["outputs"] = json.loads(d["outputs"]) if d["outputs"] else None
            d["group"] = d.pop("grp")
            out.append(d)
        return out

    def step_dir(self, run_id: str, step_id: str) -> Path:
        return self.root / "runs" / run_id / step_id

    def log(self, run_id: str, step_id: str) -> str:
        p = self.step_dir(run_id, step_id) / "log.txt"
        return p.read_text(errors="replace") if p.is_file() else ""

    def workflow(self, run_id: str) -> Workflow:
        return Workflow.from_dict(self.get_run(run_id)["workflow"])

    # -- running --------------------------------------------------------------------------------------------------
    def create(self, workflow: Workflow, run_id: str | None = None, metadata: dict | None = None) -> str:
        run_id = run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        conn = self.db.conn
        with transaction(conn):
            conn.execute("INSERT INTO runs(id, name, workflow, status, created, metadata) VALUES (?,?,?,?,?,?)",
                         (run_id, workflow.name, json.dumps(workflow.to_dict()), "pending", time.time(),
                          json.dumps(metadata or {})))
            for s in workflow.steps:
                conn.execute("INSERT INTO steps(run_id, step_id, status, executor, grp) VALUES (?,?,?,?,?)",
                             (run_id, s.id, "pending", s.executor, s.group))
        return run_id

    def run(self, workflow: Workflow | None = None, run_id: str | None = None, *,
            should_stop: Callable[[], bool] = lambda: False,
            progress: Callable[[dict[str, int]], None] | None = None, metadata: dict | None = None
            ) -> dict[str, Any]:
        """Run (or resume, when ``run_id`` exists) a workflow; returns the run summary."""
        exists = run_id is not None and self.db.conn.execute("SELECT 1 FROM runs WHERE id=?",
                                                             (run_id,)).fetchone()
        if exists:
            workflow = self.workflow(run_id)
            self._reset_for_resume(run_id)
        else:
            if workflow is None:
                raise ValueError("a new run needs a workflow")
            run_id = self.create(workflow, run_id, metadata)
        self._set_run(run_id, status="running", started=time.time())
        status = self._execute(workflow, run_id, should_stop, progress)
        self._set_run(run_id, status=status, finished=time.time())
        self._write_manifest(run_id)
        return self.get_run(run_id)

    def _write_manifest(self, run_id: str) -> None:
        """``runs/<id>/experiment.json``: the run in the experiment-manifest format the Lab site reads."""
        try:
            from pinneapple_lab.manifest import from_workflow_run, save
            save(from_workflow_run(self, run_id), str(self.root / "runs" / run_id))
        except Exception:  # noqa: BLE001 - a manifest must never fail the run it describes
            pass

    def _reset_for_resume(self, run_id: str) -> None:
        conn = self.db.conn
        for r in conn.execute("SELECT step_id, status FROM steps WHERE run_id=? AND status NOT IN ('done','skipped')",
                              (run_id,)).fetchall():
            if r["status"] == "running":
                self._append_log(run_id, r["step_id"], "interrupted (the runner stopped); resuming")
            conn.execute("UPDATE steps SET status='pending', error=NULL WHERE run_id=? AND step_id=?",
                         (run_id, r["step_id"]))

    def set_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        self._set_run(run_id, metadata=json.dumps(metadata, default=str))

    def _set_run(self, run_id: str, **cols) -> None:
        sets = ", ".join(f"{k}=?" for k in cols)
        self.db.conn.execute(f"UPDATE runs SET {sets} WHERE id=?", (*cols.values(), run_id))

    def _set_step(self, run_id: str, step_id: str, **cols) -> None:
        if "outputs" in cols and cols["outputs"] is not None:
            cols["outputs"] = json.dumps(cols["outputs"], default=_jsonable)
        sets = ", ".join(f"{k}=?" for k in cols)
        self.db.conn.execute(f"UPDATE steps SET {sets} WHERE run_id=? AND step_id=?", (*cols.values(), run_id,
                                                                                     step_id))

    def _append_log(self, run_id: str, step_id: str, line: str) -> None:
        d = self.step_dir(run_id, step_id)
        d.mkdir(parents=True, exist_ok=True)
        with self._log_lock, open(d / "log.txt", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {line}\n")

    def _execute(self, wf: Workflow, run_id: str, should_stop, progress) -> str:
        state = {r["step_id"]: r for r in self.steps(run_id)}
        hashes: dict[str, str] = {k: v["hash"] for k, v in state.items() if v["status"] in ("done", "skipped")}
        status = {k: v["status"] for k, v in state.items()}
        failed_any = False
        order = wf.order()
        pool = ThreadPoolExecutor(max_workers=max(1, wf.max_parallel))
        running: dict[Any, str] = {}

        def report():
            if progress is not None:
                counts = {s: 0 for s in STATES}
                for v in status.values():
                    counts[v] += 1
                progress(counts)

        try:
            while True:
                stop = should_stop()
                for sid in order:
                    if status[sid] != "pending":
                        continue
                    step = wf.step(sid)
                    dep = [status[d] for d in step.needs]
                    if any(s in ("failed", "cancelled") for s in dep):
                        status[sid] = "cancelled"
                        self._set_step(run_id, sid, status="cancelled", error="a step it needs did not finish")
                        continue
                    if stop or (wf.fail_fast and failed_any):
                        status[sid] = "cancelled"
                        self._set_step(run_id, sid, status="cancelled",
                                       error="cancelled" if stop else "stopped after a failure (fail_fast)")
                        continue
                    if len(running) >= max(1, wf.max_parallel):
                        continue
                    if all(s in ("done", "skipped") for s in dep):
                        h = step_hash(step, hashes, self.input_root)
                        if self.cache and self._reuse(run_id, step, h):
                            status[sid], hashes[sid] = "skipped", h
                            continue
                        status[sid] = "running"
                        upstream = {d: {"outputs": self._outputs(run_id, d), "workdir": str(self._workdir(run_id, d))}
                                    for d in step.needs}
                        fut = pool.submit(self._run_step, run_id, step, h, upstream, should_stop)
                        running[fut] = sid
                report()
                if not running:
                    if all(s != "pending" for s in status.values()):
                        break
                    if not any(status[s] == "pending" and all(status[d] in ("done", "skipped")
                                                              for d in wf.step(s).needs) for s in order):
                        break                     # nothing runnable left (should not happen in a DAG)
                    continue
                done, _ = wait(list(running), timeout=0.2, return_when=FIRST_COMPLETED)
                for fut in done:
                    sid = running.pop(fut)
                    ok, h = fut.result()
                    status[sid] = ok
                    if ok == "done":
                        hashes[sid] = h
                    elif ok == "failed":
                        failed_any = True
        finally:
            pool.shutdown(wait=True)
        report()
        values = set(status.values())
        if "failed" in values:
            return "failed"
        if "cancelled" in values or "pending" in values:
            return "cancelled"
        return "done"

    def _outputs(self, run_id: str, step_id: str) -> dict[str, Any]:
        row = self.db.conn.execute("SELECT outputs FROM steps WHERE run_id=? AND step_id=?",
                                   (run_id, step_id)).fetchone()
        return json.loads(row["outputs"]) if row and row["outputs"] else {}

    def step_workdir(self, run_id: str, step_id: str) -> Path:
        """Where a step's files are: its own folder, or the folder of the run it was reused from (skip-done)."""
        return self._workdir(run_id, step_id)

    def _workdir(self, run_id: str, step_id: str) -> Path:
        row = self.db.conn.execute("SELECT reused_from FROM steps WHERE run_id=? AND step_id=?",
                                   (run_id, step_id)).fetchone()
        if row and row["reused_from"]:
            return Path(row["reused_from"])
        return self.step_dir(run_id, step_id)

    def _reuse(self, run_id: str, step: Step, h: str) -> bool:
        row = self.db.conn.execute("SELECT * FROM cache WHERE hash=?", (h,)).fetchone()
        if row is None or not Path(row["workdir"]).is_dir():
            return False
        self._set_step(run_id, step.id, status="skipped", hash=h, outputs=json.loads(row["outputs"]),
                       reused_from=row["workdir"], finished=time.time(), seconds=0.0,
                       error=None)
        self._append_log(run_id, step.id, f"skipped: same content hash as {row['run_id']}/{row['step_id']} "
                                          f"({h[:12]})")
        return True

    def _run_step(self, run_id: str, step: Step, h: str, upstream: dict, should_stop) -> tuple[str, str]:
        workdir = self.step_dir(run_id, step.id)
        attempts_before = self.db.conn.execute("SELECT attempts FROM steps WHERE run_id=? AND step_id=?",
                                               (run_id, step.id)).fetchone()["attempts"] or 0
        log = lambda line: self._append_log(run_id, step.id, line)  # noqa: E731
        error = ""
        for k in range(step.retries + 1):
            attempt = attempts_before + k + 1
            if workdir.is_dir():                              # a clean directory per attempt, keep the log
                for p in workdir.iterdir():
                    if p.name != "log.txt":
                        shutil.rmtree(p) if p.is_dir() else p.unlink()
            t0 = time.time()
            self._set_step(run_id, step.id, status="running", hash=h, attempts=attempt, started=t0, error=None)
            log(f"attempt {attempt}: {step.kind} on {step.executor}")
            try:
                executor = get_executor(step.executor, **step.executor_options)
                adapter = get_adapter(step.kind)
                ok, why = adapter.available(step) if executor.name == "local" else executor.available()
                if not ok:
                    raise StepFailed(why)
                ctx = StepContext(run_id, step, workdir, executor, upstream, log, should_stop, attempt)
                outputs = adapter.run(ctx)
                json.dumps(outputs, default=_jsonable)
            except StepCancelled:
                self._set_step(run_id, step.id, status="cancelled", finished=time.time(),
                               seconds=time.time() - t0, error="cancelled")
                log("cancelled")
                return "cancelled", h
            except Exception as exc:  # noqa: BLE001 - every failure is the step's error
                error = f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, StepFailed):
                    for line in traceback.format_exc().rstrip().splitlines()[-15:]:
                        log(line)
                log(f"attempt {attempt} failed: {error}")
                if should_stop():
                    break
                if k < step.retries:
                    time.sleep(min(0.2 * 2 ** k, 5.0))
                continue
            now = time.time()
            self._set_step(run_id, step.id, status="done", outputs=outputs, finished=now, seconds=now - t0)
            if self.cache:
                self.db.conn.execute("INSERT OR REPLACE INTO cache(hash, outputs, workdir, run_id, step_id, finished,"
                                     " seconds) VALUES (?,?,?,?,?,?,?)",
                                     (h, json.dumps(outputs, default=_jsonable), str(workdir), run_id, step.id, now,
                                      now - t0))
            log(f"done in {now - t0:.2f} s")
            return "done", h
        self._set_step(run_id, step.id, status="failed", finished=time.time(), error=error)
        return "failed", h

    def compute_seconds(self, run_id: str) -> float:
        """Seconds the steps of this run actually ran (reused steps count 0)."""
        row = self.db.conn.execute("SELECT COALESCE(SUM(seconds), 0) FROM steps WHERE run_id=? AND reused_from IS "
                                   "NULL", (run_id,)).fetchone()
        return float(row[0])

    def storage_bytes(self, run_id: str) -> int:
        d = self.root / "runs" / run_id
        return sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.is_dir() else 0


def _jsonable(o: Any):
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
    raise TypeError(f"{type(o).__name__} is not JSON-serialisable")
