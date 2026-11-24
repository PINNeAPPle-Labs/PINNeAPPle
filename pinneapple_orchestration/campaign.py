"""A simulation campaign: thousands of runs through one queue, with content addressing, retries and accounting.

Every campaign in this library used to be its own loop — ``apps/aero_optimizer/tools/generate_dataset.py`` ran 436
OpenFOAM jobs in a local process pool, the OpenFOAM campaign module ran a plan file, the factory ran its designs —
and each of them rewrote skipping, retrying and bookkeeping (issue #295). This is that execution layer, once:

* **Content-addressed runs.** A run's key is the SHA-256 of its geometry (the bytes of the file, or of the whole
  directory tree), its solver settings and its conditions. A campaign that is repeated, resumed or run on another
  machine skips every key already done — and a *hit is proven*, not assumed: the stored record is re-read and its
  own key and the hash of its outputs are checked before the run is skipped (``RunCache``). A corrupt record is a
  miss, so the work is redone instead of trusted. Within one campaign the same address is executed once: 2000 cases
  of 25 distinct contents run 25 jobs and answer 1975 rows from them (``source`` says which is which).
* **A queue with retries and timeouts.** A failed run is never dropped: it is recorded with an excerpt of its log
  and a classification (:func:`classify_failure`) — ``mesh``, ``diverged``, ``timeout``, ``setup``, ``crashed`` —
  built on :func:`pinneapple_data.simulation_metadata.openfoam.parse_log`. Only the kinds that can pass on a second
  attempt are retried (``retry_on``, default ``timeout``, ``crashed`` and ``unknown``); a mesh that is wrong or a
  solution that diverges is deterministic and is not retried. Each finished run reaches the cache *before* the queue
  can move on (the backend calls ``on_outcome``), so a process killed mid-campaign loses only what was executing.
* **Backends.** :class:`LocalBackend` (the default), :class:`SlurmBackend` and :class:`CloudBatchBackend` (a cloud
  batch service behind the same ``submit``/``poll`` contract, with the AWS Batch CLI as the provided adapter). All
  three take concurrency, cores per run and cleanup from the same configuration.
* **Accounting with lineage.** ``campaign_report.json`` records wall time, CPU-hours (wall time x cores per run),
  the cache and failure counts with their kinds, and the machine the campaign ran on; ``lineage.json`` writes the
  chain (geometry -> run -> dataset) with :class:`pinneapple_data.lineage.Lineage`.

    import pinneapple as pp
    result = pp.campaign.run(cases, solver, backend="local", workers=8, cache_dir="cache")
    result.rows            # one dict per case: outputs, status, failure kind, log excerpt
    result.report.cpu_hours, result.report.failure_rate
    result.save("runs")    # results.json + campaign_report.json + lineage.json

A solver is anything the campaign can call per case: an object with ``run(params, workdir) -> mapping`` (the
laboratory's ``Simulator``, unchanged), a plain callable ``(params, workdir) -> mapping``, or a command line (then
the job writes ``outputs.json`` in its workdir). A callable solver runs in a worker thread of this process, so
NumPy and solver subprocesses overlap but pure-Python work does not; command jobs are real processes. That limit is
the reason the command form exists, and it is stated here rather than discovered in a benchmark.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pinneapple_core.jsonable import to_jsonable
from pinneapple_data.lineage import Artifact, Lineage, sha256

__all__ = [
    "Case", "CaseGeometry", "RunCache", "CacheRecord", "Job", "Outcome", "CommandResult", "CommandRunner",
    "subprocess_runner", "Backend", "LocalBackend", "SlurmBackend", "CloudBatchBackend", "CampaignReport",
    "CampaignResult", "Campaign", "run", "classify_failure", "case_key", "PERMANENT_FAILURES", "TRANSIENT_FAILURES",
]

#: Failure kinds that can pass on a second attempt (the machine was busy, the wall clock ran out).
TRANSIENT_FAILURES = ("timeout", "crashed", "unknown")
#: Failure kinds that are deterministic: retrying repeats the same answer and wastes the machine.
PERMANENT_FAILURES = ("mesh", "diverged", "setup")

SCHEMA = "pinneapple.campaign/1"
#: Bumped when the meaning of a key changes, so an old cache cannot answer a new question.
KEY_VERSION = "1"
LOG_TAIL_LINES = 40


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canon(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, numbers as they are. Two equal inputs give equal text."""
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_path(path: str | Path) -> str:
    """SHA-256 of a geometry: the bytes of a file, or the relative path and bytes of every file in a directory.

    The tree form is what makes a case directory a geometry: renaming a file changes the digest, so a mesh that was
    swapped cannot answer a run addressed to the old one.
    """
    p = Path(path)
    if p.is_file():
        return sha256(p.read_bytes())
    if not p.is_dir():
        raise FileNotFoundError(f"geometry not found: {path}")
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        h.update(str(f.relative_to(p)).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- cases and keys
@dataclass
class CaseGeometry:
    """What a run is made of geometrically: a path (hashed on first use) or a digest already known."""
    path: str | None = None
    digest: str | None = None

    def resolved(self) -> str | None:
        if self.digest:
            return self.digest
        if self.path:
            self.digest = _digest_path(self.path)
            return self.digest
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "digest": self.resolved()}


@dataclass
class Case:
    """One run: an identifier, a geometry, the parameters, the solver settings and the conditions.

    ``params`` is what goes into the solver, ``settings`` is how it is configured (discretisation, tolerances, mesh
    level) and ``conditions`` is the operating point (inlet velocity, temperature, angle). All three are part of the
    key: a run is the hash of exactly these, so nothing that changes the answer can be left out of it.
    """
    id: str
    params: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    conditions: Mapping[str, Any] = field(default_factory=dict)
    geometry: str | CaseGeometry | None = None
    command: Sequence[str] | None = None            # for the backends that submit a command line
    note: str = ""

    def geometry_digest(self) -> str | None:
        if self.geometry is None:
            return None
        if isinstance(self.geometry, CaseGeometry):
            return self.geometry.resolved()
        text = str(self.geometry)
        if len(text) == 64 and all(c in "0123456789abcdef" for c in text):
            return text                              # already a digest: nothing to hash
        return _digest_path(text)


def case_key(case: Case, *, solver: str = "", version: str = KEY_VERSION) -> str:
    """The content address of one run: geometry + solver settings + conditions (+ the solver's identity)."""
    payload = {"v": version, "solver": solver, "geometry": case.geometry_digest(),
               "params": dict(case.params), "settings": dict(case.settings),
               "conditions": dict(case.conditions), "command": list(case.command) if case.command else None}
    return sha256(_canon(payload).encode())


# ----------------------------------------------------------------------------------------------- cache
@dataclass
class CacheRecord:
    """One finished run as the cache keeps it: proven by its own key and the hash of its outputs."""
    key: str
    case_id: str
    solver: str
    status: str
    outputs: dict[str, Any]
    outputs_sha256: str
    wall_s: float
    cores: int
    created: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CacheRecord":
        return cls(**{k: d[k] for k in ("key", "case_id", "solver", "status", "outputs", "outputs_sha256", "wall_s",
                                        "cores", "created")})


class RunCache:
    """Content-addressed store of finished runs: one JSON per key, validated before it is believed.

    A record answers a run only when its key matches the key being asked for, its status is ``done`` and the hash it
    carries still matches its own outputs. Anything else (a truncated file, a hand-edited record, a record from
    another solver) is a miss and is counted as one: the campaign redoes the work rather than trusting it.
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.corrupt: list[str] = []

    def _path(self, key: str) -> Path:
        return self.folder / key[:2] / f"{key}.json"

    def lookup(self, key: str) -> CacheRecord | None:
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            raw = json.loads(path.read_text())
            rec = CacheRecord.from_dict(raw)
            valid = (rec.key == key and rec.status == "done"
                     and rec.outputs_sha256 == sha256(_canon(rec.outputs).encode()))
        except (ValueError, KeyError, TypeError, OSError):
            valid = False
        if not valid:
            self.corrupt.append(str(path))
            self.misses += 1
            return None
        self.hits += 1
        return rec

    def store(self, outcome: "Outcome", *, solver: str = "") -> CacheRecord:
        outs = to_jsonable(outcome.outputs or {})
        rec = CacheRecord(key=outcome.key, case_id=outcome.case_id, solver=solver, status="done", outputs=outs,
                          outputs_sha256=sha256(_canon(outs).encode()), wall_s=outcome.wall_s,
                          cores=outcome.cores, created=_now())
        path = self._path(rec.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(rec.to_dict(), indent=2))
        tmp.replace(path)                                  # atomic: an interrupted write is not a hit
        return rec

    def to_dict(self) -> dict[str, Any]:
        return {"folder": str(self.folder), "hits": self.hits, "misses": self.misses, "corrupt": list(self.corrupt)}


# ----------------------------------------------------------------------------------------------- failures
_MESH = ("blockmesh", "snappyhexmesh", "mesh", "negative volume", "zero or negative", "cells in mesh",
         "cannot find patch", "patchfield", "polyMesh")
_SETUP = ("foam fatal io error", "cannot find file", "keyword", "not found in dictionary", "no matching",
          "undefined", "entry", "dimensions")


def classify_failure(log_text: str = "", *, timed_out: bool = False, error: str = "") -> str:
    """What kind of failure a run hit, from its log (and whether the wall clock cut it off).

    ``mesh``       the case never reached the solver: blockMesh/snappyHexMesh/checkMesh failed or a patch is missing;
    ``setup``      OpenFOAM refused the case (a missing keyword, a missing file, a dictionary entry);
    ``diverged``   it ran and the residuals grew by three orders or more, or a floating point exception was raised;
    ``timeout``    the wall clock ran out while it was still solving (the kind a longer wall time can fix);
    ``crashed``    a non-zero exit with no recognisable signature;
    ``unknown``    nothing to go on (no log and no error), which is retried once instead of being forgotten.

    Parsed with :func:`pinneapple_data.simulation_metadata.openfoam.parse_log`, so the same log that a reviewer
    reads is the one the queue classifies: ``fatal``, ``floating_point_exception``, the residual history and the
    end-of-run marker come from there rather than from yet another regular expression.
    """
    if timed_out:
        return "timeout"
    text = log_text or ""
    parsed: dict[str, Any] = {}
    if text.strip():
        try:
            from pinneapple_data.simulation_metadata.openfoam import parse_log
            parsed = parse_log(text)
        except Exception:                                 # a log the parser cannot read is still evidence
            parsed = {}
    fatal = (parsed.get("fatal") or "").lower()
    where = fatal or text.lower()
    if fatal or "foam fatal error" in where or "foam fatal io error" in where:
        # a case OpenFOAM refused to read is a setup problem even when the file it could not read was the mesh:
        # deciding by the keyword "polyMesh" alone would call an incomplete case a bad mesh, and retrying a
        # deterministic refusal is exactly the waste this classification exists to avoid
        if "foam fatal io error" in where or any(k in where for k in _SETUP):
            return "setup"
        if any(k.lower() in where for k in _MESH):
            return "mesh"
        return "setup"
    if parsed.get("floating_point_exception"):
        return "diverged"
    residuals = parsed.get("residuals") or {}
    for series in residuals.values():
        vals = [v for v in series if v == v and v > 0]     # finite, positive (a residual of 0 is a solved field)
        if len(vals) >= 2 and vals[-1] / max(vals[0], 1e-300) > 1e3:
            return "diverged"
    if text.strip():
        return "crashed"
    return "unknown"


# ----------------------------------------------------------------------------------------------- jobs and outcomes
@dataclass
class Job:
    """One attempt at one run."""
    case: Case
    key: str
    workdir: Path
    attempt: int = 1
    timeout_s: float | None = None
    command: Sequence[str] | None = None
    cores: int = 1

    @property
    def id(self) -> str:
        return self.case.id


@dataclass
class Outcome:
    """What came of one attempt: the outputs, or the failure with its class and an excerpt of the log."""
    case_id: str
    key: str
    attempt: int = 1
    ok: bool = False
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    failure: str = ""
    log_excerpt: str = ""
    wall_s: float = 0.0
    cores: int = 1
    worker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "key": self.key, "attempt": self.attempt, "ok": self.ok,
                "outputs": to_jsonable(self.outputs or {}), "error": self.error, "failure": self.failure,
                "log_excerpt": self.log_excerpt, "wall_s": self.wall_s, "cores": self.cores, "worker": self.worker}


def _excerpt(text: str, lines: int = LOG_TAIL_LINES, chars: int = 4000) -> str:
    if not text:
        return ""
    tail = "\n".join(text.strip().splitlines()[-lines:])
    return tail[-chars:]


def _read_log(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


# ----------------------------------------------------------------------------------------------- backends
@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    wall_s: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@runtime_checkable
class CommandRunner(Protocol):
    """How a backend runs a command: ``argv`` in, a :class:`CommandResult` out. Swapped in tests, and in production
    it is ``subprocess`` for the cluster CLI or ``ssh`` for a remote host."""

    def __call__(self, argv: Sequence[str], *, cwd: str | None = None,
                 timeout: float | None = None) -> CommandResult: ...


def subprocess_runner(argv: Sequence[str], *, cwd: str | None = None,
                      timeout: float | None = None) -> CommandResult:
    """The default :class:`CommandRunner`: one local subprocess, its output captured, its timeout enforced."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(list(argv), cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return CommandResult(list(argv), p.returncode, p.stdout, p.stderr, time.perf_counter() - t0)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else ""
        return CommandResult(list(argv), -9, out, err, time.perf_counter() - t0, timed_out=True)
    except OSError as e:
        return CommandResult(list(argv), 127, "", str(e), time.perf_counter() - t0)


@runtime_checkable
class Backend(Protocol):
    """The only thing a campaign asks of a backend: run these pending jobs, yield outcomes as they finish.

    ``on_outcome`` is called once per finished run, from whichever thread finished it, *before* the outcome is
    yielded: that is what makes a campaign durable. The queue cannot run ahead of the record, so a process killed
    mid-campaign loses at most the runs still executing, never the ones already answered.
    """
    name: str

    def run(self, jobs: Sequence[Job], solver: Any = None,
            on_outcome: Callable[[Outcome], None] | None = None) -> Iterator[Outcome]: ...


def _call_solver(solver: Any, job: Job) -> dict[str, Any]:
    """Call a callable or a ``Simulator``-shaped object and normalise its answer to a mapping of outputs."""
    job.workdir.mkdir(parents=True, exist_ok=True)
    if hasattr(solver, "run") and callable(getattr(solver, "run")):
        out = solver.run(dict(job.case.params), str(job.workdir))
    else:
        out = solver(dict(job.case.params), str(job.workdir))
    if out is None:
        return {}
    if hasattr(out, "ok") and hasattr(out, "outputs"):          # a laboratory SimOutcome
        if not getattr(out, "ok"):
            raise RuntimeError(getattr(out, "error", "") or "the simulator reported a failure")
        return dict(getattr(out, "outputs") or {})
    if isinstance(out, Mapping):
        return {k: v for k, v in out.items() if not str(k).startswith("__")}
    raise TypeError(f"a solver must return a mapping of outputs (or a SimOutcome), got {type(out).__name__}")


class LocalBackend:
    """The default backend: ``workers`` threads in this process, one workdir per attempt, cleaned after success.

    A callable solver runs in a worker thread, so allocations and solver subprocesses overlap while pure-Python work
    does not (state the limit, do not hide it). A job with a ``command`` is a real subprocess and therefore a real
    second process. A callable that outlives its timeout is recorded as ``timeout`` and abandoned — Python cannot
    kill a thread, so the log says the worker was left running rather than pretending it stopped.
    """
    name = "local"

    def __init__(self, workers: int = 4, *, cores_per_run: int = 1, runner: CommandRunner = subprocess_runner,
                 cleanup: bool = True):
        self.workers = max(1, int(workers))
        self.cores_per_run = max(1, int(cores_per_run))
        self.runner = runner
        self.cleanup = cleanup

    def _run_one(self, job: Job, solver: Any) -> Outcome:
        t0 = time.perf_counter()
        log = job.workdir / "job.log"
        job.workdir.mkdir(parents=True, exist_ok=True)
        if job.command:                                        # a real process: the timeout really kills it
            try:
                with open(log, "w") as fh:
                    r = subprocess.run(list(job.command), cwd=job.workdir, stdout=fh, stderr=subprocess.STDOUT,
                                       timeout=job.timeout_s)
            except subprocess.TimeoutExpired:
                return Outcome(job.case.id, job.key, job.attempt, False, {}, "timeout", "timeout",
                               _excerpt(_read_log(log)), time.perf_counter() - t0, job.cores, self.name)
            except OSError as exc:
                return Outcome(job.case.id, job.key, job.attempt, False, {}, f"{type(exc).__name__}: {exc}",
                               "crashed", "", time.perf_counter() - t0, job.cores, self.name)
            text = _read_log(log)
            wall = time.perf_counter() - t0
            if r.returncode != 0:
                err = f"command exited {r.returncode}"
                return Outcome(job.case.id, job.key, job.attempt, False, {}, err, classify_failure(text, error=err),
                               _excerpt(text), wall, job.cores, self.name)
            outs = job.workdir / "outputs.json"
            try:
                data = json.loads(outs.read_text()) if outs.is_file() else {}
            except ValueError as exc:
                return Outcome(job.case.id, job.key, job.attempt, False, {}, f"outputs.json is not JSON: {exc}",
                               "setup", _excerpt(text), wall, job.cores, self.name)
            return Outcome(job.case.id, job.key, job.attempt, True, dict(data), "", "", _excerpt(text), wall,
                           job.cores, self.name)

        # a callable solver: a worker thread, because Python cannot kill one (the limit is stated in the docstring)
        result: dict[str, Any] = {}
        error: list[str] = []
        done = threading.Event()

        def body():
            try:
                result.update(_call_solver(solver, job))
            except BaseException as exc:                        # every failure is an outcome, never a lost run
                error.append(f"{type(exc).__name__}: {exc}")
            finally:
                done.set()

        th = threading.Thread(target=body, name=f"campaign-{job.case.id}", daemon=True)
        th.start()
        finished = done.wait(job.timeout_s)
        wall = time.perf_counter() - t0
        text = _read_log(log)
        if not finished:
            return Outcome(job.case.id, job.key, job.attempt, False, {},
                           "timeout (the worker thread was left running: Python cannot kill it)", "timeout",
                           _excerpt(text), wall, job.cores, self.name)
        if error:
            kind = classify_failure(text, error=error[0])
            return Outcome(job.case.id, job.key, job.attempt, False, {}, error[0], kind, _excerpt(text), wall,
                           job.cores, self.name)
        return Outcome(job.case.id, job.key, job.attempt, True, dict(result), "", "", _excerpt(text), wall,
                       job.cores, self.name)

    def _finish(self, job: Job, solver: Any, on_outcome: Callable[[Outcome], None] | None) -> Outcome:
        out = self._run_one(job, solver)
        if on_outcome is not None:
            try:
                on_outcome(out)                     # durable here, before the queue can outrun the record
            except Exception:                       # never turn a bookkeeping failure into a lost run
                traceback.print_exc()
        if out.ok and self.cleanup:
            shutil.rmtree(job.workdir, ignore_errors=True)
        return out

    def run(self, jobs: Sequence[Job], solver: Any = None,
            on_outcome: Callable[[Outcome], None] | None = None) -> Iterator[Outcome]:
        if not jobs:
            return iter(())
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for out in pool.map(lambda j: self._finish(j, solver, on_outcome), jobs):
                yield out


class SlurmBackend:
    """SLURM: one ``sbatch`` per job, one ``squeue``/``sacct`` poll for all of them.

    The command line of the job is its own ``command`` (or the solver must be a command); the job is expected to
    write ``outputs.json`` in its work directory, which is the same contract the local backend uses for command
    jobs, so a campaign moves between them without being rewritten. Concurrency, cores per run and the wall-clock
    limit come from the same configuration through ``--cpus-per-task`` and ``--time``.
    """
    name = "slurm"
    workers = 0                     # the scheduler decides how many run at once; recorded as 0, not invented

    def __init__(self, *, runner: CommandRunner = subprocess_runner, partition: str | None = None, header: Sequence[str] = (),
                 poll_s: float = 5.0, nodes: int = 1, cores_per_run: int = 1):
        self.runner = runner
        self.partition = partition
        self.header = list(header)
        self.poll_s = poll_s
        self.nodes = nodes
        self.cores_per_run = cores_per_run

    def script(self, job: Job) -> str:
        if not job.command:
            raise ValueError(f"case {job.case.id!r} has no command: the slurm backend submits a command line")
        lines = ["#!/bin/bash", f"#SBATCH --job-name=pp-{job.case.id}",
                 f"#SBATCH --cpus-per-task={job.cores}", f"#SBATCH --nodes={self.nodes}",
                 "#SBATCH --output=log.slurm", "#SBATCH --error=log.slurm"]
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if job.timeout_s:
            lines.append(f"#SBATCH --time={max(1, int(job.timeout_s // 60) or 1)}")      # slurm counts minutes
        lines += self.header
        lines.append(" ".join(str(c) for c in job.command))
        return "\n".join(lines) + "\n"

    def run(self, jobs: Sequence[Job], solver: Any = None,
            on_outcome: Callable[[Outcome], None] | None = None) -> Iterator[Outcome]:
        pending: dict[str, Job] = {}
        for job in jobs:
            job.workdir.mkdir(parents=True, exist_ok=True)
            (job.workdir / "submit.sbatch").write_text(self.script(job))
            r = self.runner(["sbatch", "--parsable", str(job.workdir / "submit.sbatch")], cwd=str(job.workdir))
            jid = (r.stdout or "").strip().split(";")[0].strip()
            if not r.ok or not jid:
                out = Outcome(job.case.id, job.key, job.attempt, False, {},
                              f"sbatch failed: {r.stderr.strip()[:200]}", "crashed",
                              _excerpt(r.stderr), r.wall_s, job.cores, "slurm")
                if on_outcome is not None:
                    on_outcome(out)
                yield out
                continue
            pending[jid] = job
        t0 = time.perf_counter()
        while pending:
            r = self.runner(["squeue", "-h", "-j", ",".join(pending), "-o", "%i|%T"])
            states: dict[str, str] = {}
            if r.ok:
                for line in (r.stdout or "").splitlines():
                    if "|" in line:
                        jid, state = line.split("|", 1)
                        states[jid.strip()] = state.strip().upper()
            for jid in list(pending):
                job = pending[jid]
                state = states.get(jid)
                if state in (None, "COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"):
                    if state is None and _still_polling(job, t0, self.poll_s):
                        continue                      # squeue does not list it yet, or the CLI answered nothing
                    del pending[jid]
                    out = self._outcome(job, state or "UNKNOWN")
                    if on_outcome is not None:
                        on_outcome(out)
                    yield out
                    continue
            if pending:
                time.sleep(self.poll_s)

    def _outcome(self, job: Job, state: str) -> Outcome:
        text = _read_log(job.workdir / "log.slurm") or _read_log(job.workdir / "job.log")
        outs: dict[str, Any] = {}
        out_file = job.workdir / "outputs.json"
        if out_file.is_file():
            try:
                outs = json.loads(out_file.read_text())
            except ValueError:
                outs = {}
        if state == "COMPLETED":
            return Outcome(job.case.id, job.key, job.attempt, True, outs, "", "", _excerpt(text), 0.0, job.cores,
                           "slurm")
        return Outcome(job.case.id, job.key, job.attempt, False, {}, state, _scheduler_failure(state, text), _excerpt(text),
                       0.0, job.cores, "slurm")


def _scheduler_failure(state: str, text: str) -> str:
    """A scheduler state, classified: the state is evidence even when the job left no log to read."""
    if state == "TIMEOUT":
        return "timeout"
    return classify_failure(text, error=state) if text.strip() else "crashed"


def _still_polling(job: Job, t0: float, poll_s: float) -> bool:
    """A job that ``squeue`` no longer lists is finished — unless it was just submitted and has not appeared yet."""
    if job.timeout_s:
        return time.perf_counter() - t0 < job.timeout_s
    return time.perf_counter() - t0 < max(30.0, 3 * poll_s)


class CloudBatchBackend:
    """One cloud batch service behind the same contract: ``submit`` a job, ``poll`` until it is done.

    The service is injected as two callables (``submit(job) -> id`` and ``poll(ids) -> {id: state}``) because the
    useful part of this backend is the contract, not the vendor: the campaign passes jobs, gets states and reads the
    ``outputs.json`` the job wrote to its work directory. :func:`aws_batch_backend` is the adapter for AWS Batch
    through the ``aws`` CLI. It was **not** exercised against a real account: the submit/poll calls are tested, the
    vendor's answer is not, and a campaign that needs S3 between the job and the work directory must provide it.
    """
    name = "cloud"

    def __init__(self, submit: Callable[[Job], str], poll: Callable[[Sequence[str]], Mapping[str, str]], *,
                 cores_per_run: int = 1, poll_s: float = 10.0):
        self._submit = submit
        self._poll = poll
        self.cores_per_run = cores_per_run
        self.poll_s = poll_s

    def run(self, jobs: Sequence[Job], solver: Any = None,
            on_outcome: Callable[[Outcome], None] | None = None) -> Iterator[Outcome]:
        pending: dict[str, Job] = {}
        for job in jobs:
            job.workdir.mkdir(parents=True, exist_ok=True)
            try:
                pending[str(self._submit(job))] = job
            except Exception as exc:
                out = Outcome(job.case.id, job.key, job.attempt, False, {}, f"submit failed: {exc}", "crashed",
                              "", 0.0, job.cores, self.name)
                if on_outcome is not None:
                    on_outcome(out)
                yield out
        while pending:
            states = {str(k): str(v).upper() for k, v in (self._poll(list(pending)) or {}).items()}
            for jid in list(pending):
                state = states.get(jid)
                if state is None or state in ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"):
                    continue
                job = pending.pop(jid)
                text = _read_log(job.workdir / "job.log")
                outs: dict[str, Any] = {}
                out_file = job.workdir / "outputs.json"
                if out_file.is_file():
                    try:
                        outs = json.loads(out_file.read_text())
                    except ValueError:
                        outs = {}
                out = (Outcome(job.case.id, job.key, job.attempt, True, outs, "", "", _excerpt(text), 0.0,
                               job.cores, self.name) if state == "SUCCEEDED" else
                       Outcome(job.case.id, job.key, job.attempt, False, {}, state,
                               _scheduler_failure(state, text), _excerpt(text), 0.0, job.cores, self.name))
                if on_outcome is not None:
                    on_outcome(out)
                yield out
            if pending:
                time.sleep(self.poll_s)


def aws_batch_backend(*, job_queue: str, job_definition: str, runner: CommandRunner = subprocess_runner,
                      region: str | None = None, cores_per_run: int = 1, poll_s: float = 10.0) -> CloudBatchBackend:
    """The AWS Batch adapter: ``aws batch submit-job`` and ``aws batch describe-jobs`` through the CLI."""
    base = ["aws", "batch"] + (["--region", region] if region else [])
    common = ["--output", "json"]

    def submit(job: Job) -> str:
        cmd = list(job.command or [])
        overrides = json.dumps({"command": cmd} if cmd else {})
        r = runner(base + ["submit-job", "--job-name", f"pp-{job.case.id}"[:120],
                           "--job-queue", job_queue, "--job-definition", job_definition,
                           "--container-overrides", overrides] + common, cwd=str(job.workdir))
        if not r.ok:
            raise RuntimeError(r.stderr.strip()[:200] or "submit-job failed")
        return json.loads(r.stdout)["jobId"]

    def poll(ids: Sequence[str]) -> Mapping[str, str]:
        r = runner(base + ["describe-jobs", "--jobs", *ids] + common)
        if not r.ok:
            return {}
        return {j["jobId"]: j.get("status", "") for j in json.loads(r.stdout).get("jobs", [])}

    return CloudBatchBackend(submit, poll, cores_per_run=cores_per_run, poll_s=poll_s)


BACKENDS = {"local": LocalBackend, "slurm": SlurmBackend}


# ----------------------------------------------------------------------------------------------- report
@dataclass
class CampaignReport:
    """What the campaign cost and what it ran on: the accounting the dataset is written with."""
    name: str
    schema: str = SCHEMA
    backend: str = "local"
    workers: int = 1
    cores_per_run: int = 1
    n_cases: int = 0
    n_executed: int = 0
    n_cached: int = 0
    n_shared: int = 0                 # cases answered by the run of an identical case in this same campaign
    n_failed: int = 0
    n_retried: int = 0
    wall_s: float = 0.0
    cpu_hours: float = 0.0
    failures: dict[str, int] = field(default_factory=dict)
    machine: dict[str, Any] = field(default_factory=dict)
    started: str = ""
    ended: str = ""

    @property
    def failure_rate(self) -> float:
        return self.n_failed / self.n_cases if self.n_cases else 0.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["failure_rate"] = round(self.failure_rate, 4)
        return out


@dataclass
class CampaignResult:
    """The dataset (one row per case) plus the report and the lineage it was produced with."""
    name: str
    rows: list[dict[str, Any]]
    report: CampaignReport
    lineage: Lineage
    cache: RunCache | None = None

    @property
    def ok(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r.get("ok")]

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if not r.get("ok")]

    def save(self, folder: str | Path) -> dict[str, str]:
        """Write ``results.json``, ``campaign_report.json`` and ``lineage.json`` next to each other."""
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        results = out / "results.json"
        results.write_text(json.dumps({"schema": SCHEMA, "name": self.name, "rows": self.rows}, indent=2))
        (out / "campaign_report.json").write_text(json.dumps(self.report.to_dict(), indent=2))
        self.lineage.add(Artifact(id="dataset:campaign", kind="dataset", name=self.name, file=str(results),
                                  hash=sha256(results.read_bytes()), software="pinneapple.campaign",
                                  timestamp=self.report.ended or _now(),
                                  inputs=[f"run:{r['case_id']}" for r in self.rows if r.get("ok")]))
        (out / "lineage.json").write_text(json.dumps(self.lineage.to_json(), indent=2))
        return {"results": str(results), "report": str(out / "campaign_report.json"),
                "lineage": str(out / "lineage.json")}


# ----------------------------------------------------------------------------------------------- the campaign
class Campaign:
    """A list of cases and the queue that runs them, once, on the backend it is given.

    The order of work is: address every case by its content, answer what the cache proves, run one job per remaining
    address with retries, keep every outcome (successful or not) in ``results.json``, and account for the whole
    thing. Each finished run is stored as it finishes, so a campaign interrupted at case 900 of 2000 resumes with
    the 900 already proven.
    """

    def __init__(self, cases: Sequence[Case], solver: Any, *, name: str = "campaign", backend: Any = "local",
                 workers: int = 4, cores_per_run: int = 1, cache_dir: str | Path | None = None,
                 workroot: str | Path | None = None, max_attempts: int = 2, timeout_s: float | None = None,
                 retry_on: Sequence[str] = TRANSIENT_FAILURES, cleanup: bool = True,
                 runner: CommandRunner = subprocess_runner):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        ids = [c.id for c in cases]
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        if dupes:
            raise ValueError(f"case ids must be unique; repeated: {sorted(dupes)[:5]}")
        self.cases, self.solver, self.name = list(cases), solver, name
        self.cores_per_run = max(1, int(cores_per_run))
        self.max_attempts = int(max_attempts)
        self.timeout_s = timeout_s
        self.retry_on = tuple(retry_on)
        self.workroot = Path(workroot or Path(os.environ.get("TMPDIR", "/tmp")) / f"pp-campaign-{name}")
        self.workroot.mkdir(parents=True, exist_ok=True)
        self.cache = RunCache(cache_dir) if cache_dir else None
        solver_id = getattr(solver, "name", None) or getattr(solver, "__name__", None) or (
            solver.__class__.__name__ if solver is not None else "")
        self.solver_id = str(solver_id)
        self.backend = self._make_backend(backend, workers, runner, cleanup)
        self.outcomes: list[Outcome] = []

    def _make_backend(self, backend: Any, workers: int, runner: CommandRunner, cleanup: bool) -> Any:
        if isinstance(backend, str):
            key = backend.strip().lower()
            if key == "local":
                return LocalBackend(workers, cores_per_run=self.cores_per_run, runner=runner, cleanup=cleanup)
            if key == "slurm":
                return SlurmBackend(runner=runner, cores_per_run=self.cores_per_run)
            raise ValueError(f"unknown backend {backend!r}: use 'local', 'slurm', a CloudBatchBackend, or "
                             f"aws_batch_backend(...)")
        if not hasattr(backend, "run"):
            raise TypeError("a backend must implement run(jobs, solver)")
        return backend

    # ------------------------------------------------------------------ plan
    def plan(self) -> tuple[list[tuple[Case, str, CacheRecord | None]], list[str]]:
        """``[(case, key, cache record or None)]`` and the keys that are answered by a proven record."""
        out = []
        for case in self.cases:
            key = case_key(case, solver=self.solver_id)
            out.append((case, key, self.cache.lookup(key) if self.cache else None))
        return out, [k for _, k, rec in out if rec is not None]

    # ------------------------------------------------------------------ run
    def run(self) -> CampaignResult:
        started, t0 = _now(), time.perf_counter()
        planned, _ = self.plan()
        groups: dict[str, list[Case]] = {}
        records: dict[str, CacheRecord | None] = {}
        for case, key, rec in planned:
            groups.setdefault(key, []).append(case)
            records[key] = rec if key not in records else records[key]
        done: dict[str, Outcome] = {}

        for key, members in groups.items():                       # proven hits never re-execute
            rec = records[key]
            if rec is None:
                continue
            for case in members:
                done[case.id] = Outcome(case.id, key, 0, True, dict(rec.outputs), "", "", "[cached]",
                                        rec.wall_s, rec.cores, "cache")

        # one job per distinct content: a campaign of thousands of cases with repeated content runs the distinct
        # ones and answers the rest from the answer, which is the whole point of addressing a run by its content
        pending = [Job(members[0], key, self.workroot / f"{members[0].id}-a1", 1, self.timeout_s,
                       members[0].command, self.cores_per_run)
                   for key, members in groups.items() if records[key] is None]
        attempt = 1
        total_attempts = 0
        while pending and attempt <= self.max_attempts:
            for job in pending:
                job.attempt = attempt
                job.workdir = self.workroot / f"{job.case.id}-a{attempt}"

            def record(outcome: Outcome) -> None:
                self.outcomes.append(outcome)
                if outcome.ok and self.cache:
                    self.cache.store(outcome, solver=self.solver_id)

            for outcome in self.backend.run(pending, self.solver, on_outcome=record):
                total_attempts += 1
                for case in groups[outcome.key]:                  # the representative and every identical case
                    source = "executed" if case.id == outcome.case_id else "shared"
                    done[case.id] = Outcome(case.id, outcome.key, outcome.attempt, outcome.ok,
                                            dict(outcome.outputs), outcome.error, outcome.failure,
                                            outcome.log_excerpt, outcome.wall_s, outcome.cores, source)
            retry = [job for job in pending if (o := done.get(job.case.id)) and not o.ok
                     and o.failure in self.retry_on]
            if not retry or attempt >= self.max_attempts:
                break
            pending, attempt = retry, attempt + 1

        wall = time.perf_counter() - t0
        rows = self._rows(done)
        report = self._report(rows, wall, started, total_attempts)
        return CampaignResult(self.name, rows, report, self._lineage(rows), self.cache)

    def _rows(self, done: Mapping[str, Outcome]) -> list[dict[str, Any]]:
        rows = []
        for case in self.cases:
            out = done.get(case.id)
            if out is None:                                       # never attempted: still a row, never a hole
                rows.append({"case_id": case.id, "key": case_key(case, solver=self.solver_id), "ok": False,
                             "status": "not run", "failure": "unknown", "error": "the campaign stopped before this "
                             "case", "outputs": {}, "wall_s": 0.0, "attempt": 0, "source": "not run",
                             "log_excerpt": ""})
                continue
            rows.append({"case_id": case.id, "key": out.key, "ok": out.ok,
                         "status": "done" if out.ok else (out.failure or "failed"),
                         "failure": "" if out.ok else out.failure, "error": out.error,
                         "outputs": to_jsonable(out.outputs), "wall_s": round(out.wall_s, 6),
                         "attempt": out.attempt, "source": out.worker or "executed",
                         "cached": out.worker in ("cache", "shared"),
                         "log_excerpt": out.log_excerpt,
                         "params": to_jsonable(dict(case.params)),
                         "settings": to_jsonable(dict(case.settings)),
                         "conditions": to_jsonable(dict(case.conditions))})
        return rows

    def _report(self, rows: Sequence[Mapping[str, Any]], wall: float, started: str, attempts: int) -> CampaignReport:
        failures = Counter(r["failure"] for r in rows if not r["ok"])
        executed = sum(1 for r in rows if r.get("source") == "executed")
        return CampaignReport(
            name=self.name, backend=getattr(self.backend, "name", "custom"),
            workers=getattr(self.backend, "workers", 0), cores_per_run=self.cores_per_run,
            n_cases=len(rows), n_executed=executed,
            n_cached=sum(1 for r in rows if r.get("cached")),
            n_shared=sum(1 for r in rows if r.get("source") == "shared"),
            n_failed=sum(1 for r in rows if not r["ok"]),
            n_retried=max(0, attempts - executed), wall_s=round(wall, 3),
            cpu_hours=round(sum(float(r["wall_s"]) for r in rows if r.get("source") == "executed")
                            * self.cores_per_run / 3600.0, 6),
            failures=dict(failures), machine=_machine(), started=started, ended=_now())

    def _lineage(self, rows: Sequence[Mapping[str, Any]]) -> Lineage:
        lin = Lineage(project=self.name)
        seen: dict[str, str] = {}
        for case in self.cases:
            digest = case.geometry_digest()
            if digest and digest not in seen:
                aid = f"geometry:{digest[:12]}"
                seen[digest] = aid
                path = case.geometry.path if isinstance(case.geometry, CaseGeometry) else case.geometry
                lin.add(Artifact(id=aid, kind="geometry", name=str(path or digest[:12]), file=str(path or ""),
                                 hash=digest, origin=str(path or "")))
        for row in rows:
            case = next(c for c in self.cases if c.id == row["case_id"])
            digest = case.geometry_digest()
            lin.add(Artifact(id=f"run:{case.id}", kind="simulation", name=case.id, software=self.solver_id,
                             parameters={**dict(case.params), **dict(case.conditions),
                                         **dict(case.settings)},
                             hash=row.get("key"), origin=getattr(self.backend, "name", "custom"),
                             inputs=[seen[digest]] if digest and digest in seen else [],
                             notes=("" if row["ok"] else f"{row['failure']}: {row['error']}")))
        return lin


def _machine() -> dict[str, Any]:
    """The machine the campaign ran on, recorded next to the cost so a number is never read out of context."""
    mem = None
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        mem = None
    return {"system": platform.system(), "machine": platform.machine(), "cpu_count": os.cpu_count() or 1,
            "total_memory_bytes": mem, "python": platform.python_version(), "hostname": platform.node()}


def run(cases: Sequence[Case], solver: Any = None, *, backend: Any = "local", workers: int = 4,
        cores_per_run: int = 1, cache_dir: str | Path | None = None, workroot: str | Path | None = None,
        name: str = "campaign", max_attempts: int = 2, timeout_s: float | None = None,
        retry_on: Sequence[str] = TRANSIENT_FAILURES) -> CampaignResult:
    """Run a campaign: the short form of ``Campaign(...).run()`` (``pp.campaign.run``).

    ``backend`` is ``"local"``, ``"slurm"``, or a backend instance (``CloudBatchBackend``, ``aws_batch_backend``).
    ``cache_dir`` turns on content-addressed caching, which is what makes a repeated or resumed campaign cheap.
    """
    return Campaign(cases, solver, name=name, backend=backend, workers=workers, cores_per_run=cores_per_run,
                    cache_dir=cache_dir, workroot=workroot, max_attempts=max_attempts,
                    timeout_s=timeout_s, retry_on=retry_on).run()
