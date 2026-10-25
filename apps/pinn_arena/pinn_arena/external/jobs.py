"""
Minimal in-memory job manager.

Inference on the tiny demo datasets finishes in well under a second for
most models, but fine-tuning (and a full-catalog benchmark run) can take
longer, so every long-running action goes through here instead of blocking
the request indefinitely: the caller submits a job, waits briefly for it to
finish, and gets either the final result or a job id to poll.

In-memory only: a job store is lost on restart and is not shared across worker processes (the app runs one
uvicorn worker). Finished jobs are forgotten after ``_KEEP_S`` seconds.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="job")
_LOCK = Lock()


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"          # running | done | failed
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    _future: Future | None = field(default=None, repr=False)


_JOBS: dict[str, Job] = {}
_KEEP_S = 3600.0


def _prune() -> None:
    now = time.time()
    with _LOCK:
        for jid in [j.id for j in _JOBS.values() if j.finished_at and now - j.finished_at > _KEEP_S]:
            del _JOBS[jid]


def submit(kind: str, fn: Callable[[], Any]) -> Job:
    _prune()
    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, kind=kind)

    def _run():
        try:
            result = fn()
            job.result = result
            job.status = "done"
        except Exception as exc:  # noqa: BLE001
            job.error = f"{exc.__class__.__name__}: {exc}"
            job.status = "failed"
        finally:
            job.finished_at = time.time()

    future = _EXECUTOR.submit(_run)
    job._future = future
    with _LOCK:
        _JOBS[job_id] = job
    return job


def await_job(job: Job, timeout: float) -> Job:
    """Block up to `timeout` seconds for the job to finish; returns the job
    either way (still "running" if it didn't finish in time)."""
    try:
        job._future.result(timeout=timeout)
    except FutureTimeoutError:
        pass
    except Exception:
        pass  # the job's own error is already recorded on `job`
    return job


def get(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(job_id)


def to_dict(job: Job) -> dict:
    return {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
    }
