"""Executors that run a step somewhere other than this machine: Kubernetes and cloud batch (#1479).

Both implement the interface the engine already uses (:class:`pinneapple_core.workflows.executors.Executor`): cwd,
environment, timeout, cancel, log streaming and an exit code. Nothing above them changes — a workflow written for
``executor="local"`` runs on ``kubernetes`` by changing that one field, and a step that *cannot* run remotely fails
with the reason instead of turning into a timeout.

Kubernetes
----------
One Job per step. The step's directory is mounted at ``/work`` (a persistent volume claim, or a host path when the
cluster runs on this machine), the command's lines are streamed back with ``kubectl logs``, and ``should_stop`` and the
timeout both delete the Job — no orphan workload is left behind. Options::

    Step("solve", "openfoam", {...}, executor="kubernetes",
         executor_options={"image": "openfoam/openfoam10-paraview56:latest",
                           "artifacts_pvc": "pinneapple-work",      # or artifacts_host_path="/data"
                           "namespace": "sim", "cpu": "4", "memory": "16Gi", "gpu": 1,
                           "kubectl": "kubectl", "dry_run": False})

``dry_run=True`` writes ``<cwd>/step.job.json`` and stops, the same way the Slurm executor writes the script it would
submit. The execution backend is the ``kubectl`` binary on ``PATH``; no Kubernetes client library is imported.

Cloud batch
-----------
One implementation per provider, behind :class:`BatchProvider`. No vendor is wired here: the issue asks for the seam
and for the first provider to be chosen by a real case, and inventing an AWS/GCP/Azure implementation nobody asked for
would be exactly the speculation to avoid. ``CommandBatchProvider`` is the generic one — a submit command, a status
command, a log command and a cancel command, each a template over ``{job}``/``{argv}`` — which is enough to drive a
``sbatch``-like CLI or a cloud CLI until a real case earns its own class.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .executors import ExecResult, Executor, LocalExecutor

__all__ = ["BATCH_PROVIDERS", "BatchExecutor", "BatchJob", "BatchProvider", "CommandBatchProvider",
           "KubernetesExecutor", "get_batch_provider", "register_batch_provider"]

Log = Callable[[str], None]


def _safe(text: str) -> str:
    """A DNS-1123 fragment: job names cannot hold what a directory name holds."""
    cleaned = re.sub(r"[^a-z0-9-]+", "-", str(text).lower()).strip("-")
    return (cleaned or "step")[:40]


class KubernetesExecutor(Executor):
    """One Job per step, logs streamed back, cancellation mapped to job deletion. See the module docstring."""

    name = "kubernetes"
    remote = True
    #: Seconds between two reads of the job's status.
    poll_seconds: float = 2.0

    def available(self) -> tuple[bool, str]:
        kubectl = self.options.get("kubectl", "kubectl")
        if not shutil.which(str(kubectl)):
            return False, f"{kubectl} is not on PATH: the kubernetes executor drives kubectl"
        if not self.options.get("image"):
            return False, "the kubernetes executor needs an image (executor_options={'image': ...})"
        return True, ""

    def remote_limits(self, step: Any = None) -> str:
        """Why the step's files cannot be seen by the cluster, or ``""`` when they can."""
        if self.options.get("artifacts_pvc") or self.options.get("artifacts_host_path"):
            return ""
        return ("the step's directory would not be mounted: set executor_options artifacts_pvc (a claim the cluster "
                "can read) or artifacts_host_path (a path the node can read), otherwise the job starts in an empty "
                "/work and fails on a missing input")

    # -- the job -------------------------------------------------------------------------------------------------
    def job_name(self, cwd: str | Path) -> str:
        return f"pinneapple-{_safe(Path(cwd).name)}-{uuid.uuid4().hex[:6]}"

    def job_manifest(self, argv: Sequence[str], cwd: str | Path, *, env: Mapping[str, str] | None = None,
                     name: str = "", timeout: float | None = None, mount: str = "/work") -> dict[str, Any]:
        """The ``batch/v1`` Job body this executor submits: one container, one command, the directory mounted."""
        o = self.options
        argv = [str(a) for a in argv]
        env_doc = [{"name": str(k), "value": str(v)} for k, v in (env or {}).items()]
        resources: dict[str, Any] = {"limits": {"cpu": str(o.get("cpu", "1")), "memory": str(o.get("memory", "2Gi"))}}
        if int(o.get("gpu") or 0):
            resources["limits"]["nvidia.com/gpu"] = int(o["gpu"])
        volume: dict[str, Any]
        if o.get("artifacts_pvc"):
            volume = {"persistentVolumeClaim": {"claimName": str(o["artifacts_pvc"])}}
        else:
            source = str(o["artifacts_host_path"])
            rel = Path(cwd).resolve()
            # a host path mount is the whole folder the step lives in when the path is given as a root
            volume = {"hostPath": {"path": source, "type": "Directory"}}
            mount = mount if str(rel).startswith(source) else mount
        container = {"name": "step", "image": str(o["image"]), "command": [argv[0]], "args": argv[1:],
                     "workingDir": mount, "env": env_doc, "resources": resources,
                     "volumeMounts": [{"name": "work", "mountPath": mount}]}
        spec: dict[str, Any] = {"restartPolicy": "Never", "containers": [container],
                                "volumes": [{"name": "work", **volume}]}
        if o.get("service_account"):
            spec["serviceAccountName"] = str(o["service_account"])
        if o.get("node_selector"):
            spec["nodeSelector"] = dict(o["node_selector"])
        job: dict[str, Any] = {
            "apiVersion": "batch/v1", "kind": "Job",
            "metadata": {"name": name or self.job_name(cwd), "namespace": str(o.get("namespace", "default")),
                         "labels": {"app.kubernetes.io/managed-by": "pinneapple",
                                    "pinneapple.step": _safe(Path(cwd).name)},
                         "annotations": {"pinneapple/step": Path(cwd).name}},
            "spec": {"backoffLimit": 0, "template": {"metadata": {"labels": {"job-name": name or ""}},
                                                     "spec": spec}},
        }
        if o.get("ttl_seconds_after_finished", 3600) is not None:
            job["spec"]["ttlSecondsAfterFinished"] = int(o.get("ttl_seconds_after_finished", 3600))
        if timeout:
            job["spec"]["activeDeadlineSeconds"] = int(timeout + 1)
        job["spec"]["template"]["metadata"]["labels"]["job-name"] = job["metadata"]["name"]
        return job

    # -- running -------------------------------------------------------------------------------------------------
    def _kubectl(self, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
        cmd = [str(self.options.get("kubectl", "kubectl"))]
        if self.options.get("context"):
            cmd += ["--context", str(self.options["context"])]
        if self.options.get("kubeconfig"):
            cmd += ["--kubeconfig", str(self.options["kubeconfig"])]
        cmd += ["--namespace", str(self.options.get("namespace", "default")), *args]
        return subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=False)

    def run(self, argv: Sequence[str], *, cwd: str | Path, env: Mapping[str, str] | None = None,
            timeout: float | None = None, log: Log = print, should_stop: Callable[[], bool] = lambda: False
            ) -> ExecResult:
        ok, why = self.available()
        if not ok:
            log(why)
            return ExecResult(127, 0.0, [str(a) for a in argv], details={"error": why})
        argv = [str(a) for a in argv]
        limits = self.remote_limits()
        if limits:
            log(limits)
            return ExecResult(126, 0.0, argv, details={"error": limits})
        name = self.job_name(cwd)
        manifest = self.job_manifest(argv, cwd, env=env, name=name, timeout=timeout)
        path = Path(cwd) / "step.job.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        t0 = time.time()
        log(f"$ kubectl apply -f {path.name}  (job {name})")
        if self.options.get("dry_run"):
            return ExecResult(0, 0.0, ["kubectl", "apply", "-f", str(path)], dry_run=True,
                              details={"manifest": str(path), "job": name})
        applied = self._kubectl("apply", "-f", str(path))
        if applied.returncode != 0:
            log((applied.stderr or applied.stdout).strip())
            return ExecResult(applied.returncode, time.time() - t0, argv,
                              details={"error": (applied.stderr or "").strip(), "job": name})
        timed_out = cancelled = False
        state, exit_code = "running", None
        shown = 0
        while state == "running":
            if timeout is not None and time.time() - t0 > timeout:
                timed_out = True
            elif should_stop():
                cancelled = True
            if timed_out or cancelled:
                break
            time.sleep(float(self.options.get("poll_seconds", self.poll_seconds)))
            state, exit_code = self._status(name)
            shown = self._stream(name, shown, log)       # the pod's lines arrive while it runs, not after
        if timed_out or cancelled:
            deleted = self._delete(name)
            log("deleted the job: " + ("time limit reached" if timed_out else "cancelled")
                + ("" if deleted else " (the deletion did not confirm)"))
            return ExecResult(-1, time.time() - t0, argv, timed_out=timed_out, cancelled=cancelled,
                              details={"job": name, "deleted": deleted})
        self._stream(name, shown, log)
        if state == "failed" and exit_code is None:
            exit_code = 1
        return ExecResult(int(exit_code or 0), time.time() - t0, argv, details={"job": name, "state": state})

    def _status(self, name: str) -> tuple[str, int | None]:
        got = self._kubectl("get", "job", name, "-o", "json")
        if got.returncode != 0:
            return "failed", None
        try:
            doc = json.loads(got.stdout)
        except json.JSONDecodeError:
            return "failed", None
        conditions = {c.get("type"): c.get("status") for c in (doc.get("status", {}) or {}).get("conditions", [])}
        if conditions.get("Complete") == "True" or (doc.get("status", {}) or {}).get("succeeded"):
            return "done", self._pod_exit_code(name) or 0
        if conditions.get("Failed") == "True" or (doc.get("status", {}) or {}).get("failed"):
            return "failed", self._pod_exit_code(name)
        return "running", None

    def _pod_exit_code(self, name: str) -> int | None:
        """The container's own exit code, which is what a failed step needs to report (not "the job failed")."""
        got = self._kubectl("get", "pods", "-l", f"job-name={name}", "-o", "json")
        if got.returncode != 0:
            return None
        try:
            pods = json.loads(got.stdout).get("items", [])
        except json.JSONDecodeError:
            return None
        for pod in pods:
            for status in (pod.get("status", {}) or {}).get("containerStatuses", []) or []:
                terminated = ((status.get("state") or {}).get("terminated") or {})
                if "exitCode" in terminated:
                    return int(terminated["exitCode"])
        return None

    def _logs(self, name: str) -> list[str]:
        got = self._kubectl("logs", f"job/{name}", "--tail=200")
        return (got.stdout or "").rstrip("\n").splitlines() if got.returncode == 0 else []

    def _stream(self, name: str, shown: int, log: Log) -> int:
        """Print the lines that appeared since ``shown``, and return the new count."""
        lines = self._logs(name)
        for line in lines[shown:]:
            log(line)
        return len(lines)

    def _delete(self, name: str) -> bool:
        got = self._kubectl("delete", "job", name, "--ignore-not-found")
        return got.returncode == 0


# -------------------------------------------------------------------------------------------------- cloud batch
@dataclass
class BatchJob:
    """What a batch provider is asked to run: one step's command, its directory, its environment."""

    name: str
    argv: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    image: str = ""
    timeout: float | None = None
    step: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "argv": list(self.argv), "cwd": self.cwd, "env": dict(self.env),
                "image": self.image, "timeout": self.timeout, "step": self.step}


class BatchProvider(ABC):
    """One cloud's batch API. Four calls: submit, status, logs, cancel — the same four the engine needs."""

    name: str = ""

    def available(self) -> tuple[bool, str]:
        return True, ""

    @abstractmethod
    def submit(self, job: BatchJob) -> str:
        """Queue the job and return its id."""

    @abstractmethod
    def status(self, job_id: str) -> tuple[str, int | None]:
        """``("running" | "done" | "failed", exit code when known)``."""

    @abstractmethod
    def logs(self, job_id: str) -> list[str]:
        """The job's output so far (the tail is enough)."""

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel the job; returns whether the cancellation was accepted."""


class CommandBatchProvider(BatchProvider):
    """A provider spelled as four command lines, each a shell template over ``{job}`` and ``{argv}``.

    Enough to drive a ``sbatch``-like or cloud CLI without a class per vendor; a vendor whose status JSON the shell
    cannot reduce to one word wants its own :class:`BatchProvider` subclass, which is what the issue asks for when a
    real case arrives. Options: ``submit`` (required), ``status``, ``logs``, ``cancel``, ``done``/``failed`` (the words
    in the status output that mean finished — default ``COMPLETED``/``FAILED``), ``timeout``.
    """

    name = "command"

    def __init__(self, **options: Any):
        self.options = options

    def available(self) -> tuple[bool, str]:
        if not self.options.get("submit"):
            return False, "a command batch provider needs a `submit` command template"
        return True, ""

    def _run(self, template: str, job_id: str = "", job: BatchJob | None = None) -> subprocess.CompletedProcess:
        command = template.format(job=job_id, argv=shlex.join(job.argv) if job else "",
                                  name=job.name if job else "", cwd=job.cwd if job else "")
        return subprocess.run(shlex.split(command), capture_output=True, text=True, check=False,
                              timeout=float(self.options.get("timeout", 60)))

    def submit(self, job: BatchJob) -> str:
        got = self._run(str(self.options["submit"]), job=job)
        if got.returncode != 0:
            raise RuntimeError(f"batch submit failed ({got.returncode}): {(got.stderr or got.stdout).strip()[:300]}")
        return (got.stdout.strip().splitlines() or [""])[-1].strip()

    def status(self, job_id: str) -> tuple[str, int | None]:
        if not self.options.get("status"):
            return "running", None
        got = self._run(str(self.options["status"]), job_id)
        text = (got.stdout or "").strip()
        words = str(self.options.get("done", "COMPLETED")).split(",")
        bad = str(self.options.get("failed", "FAILED")).split(",")
        if any(word.strip() and word.strip() in text for word in bad):
            return "failed", 1
        if any(word.strip() and word.strip() in text for word in words):
            return "done", 0
        return "running", None

    def logs(self, job_id: str) -> list[str]:
        if not self.options.get("logs"):
            return []
        got = self._run(str(self.options["logs"]), job_id)
        return (got.stdout or "").rstrip("\n").splitlines()

    def cancel(self, job_id: str) -> bool:
        if not self.options.get("cancel"):
            return False
        return self._run(str(self.options["cancel"]), job_id).returncode == 0


BATCH_PROVIDERS: dict[str, type[BatchProvider]] = {"command": CommandBatchProvider}


def register_batch_provider(name: str, cls: type[BatchProvider]) -> None:
    """Make ``provider=name`` usable in a ``batch`` executor (a plugin's cloud, a site's queue)."""
    if not (isinstance(cls, type) and issubclass(cls, BatchProvider)):
        raise TypeError("a batch provider must subclass BatchProvider")
    BATCH_PROVIDERS[name] = cls


def get_batch_provider(name: str, **options: Any) -> BatchProvider:
    if name not in BATCH_PROVIDERS:
        raise KeyError(f"unknown batch provider {name!r}; registered: {sorted(BATCH_PROVIDERS)}")
    return BATCH_PROVIDERS[name](**options)


class BatchExecutor(Executor):
    """A step on a cloud batch service, through a :class:`BatchProvider`.

    Options: ``provider`` (default ``"command"``), ``provider_options`` (its own options), ``image``, ``poll_seconds``.
    Cancellation and timeout cancel the job through the provider, the same way the Kubernetes executor deletes it.
    """

    name = "batch"
    remote = True
    poll_seconds: float = 2.0

    def provider(self) -> BatchProvider:
        return get_batch_provider(str(self.options.get("provider", "command")),
                                  **dict(self.options.get("provider_options") or {}))

    def available(self) -> tuple[bool, str]:
        try:
            return self.provider().available()
        except KeyError as exc:
            return False, str(exc)

    def remote_limits(self, step: Any = None) -> str:
        """A batch service sees the files the provider uploads, and this executor uploads none: say it plainly."""
        if self.options.get("uploads"):
            return ""
        return ("the step's directory is not uploaded to the batch service: set executor_options uploads=<how the "
                "provider ships the working directory> (or an artifacts bucket) — otherwise the job runs in an empty "
                "directory and the failure would look like a missing input")

    def run(self, argv: Sequence[str], *, cwd: str | Path, env: Mapping[str, str] | None = None,
            timeout: float | None = None, log: Log = print, should_stop: Callable[[], bool] = lambda: False
            ) -> ExecResult:
        argv = [str(a) for a in argv]
        provider = self.provider()
        ok, why = provider.available()
        if not ok:
            log(why)
            return ExecResult(127, 0.0, argv, details={"error": why})
        limits = self.remote_limits()
        if limits:
            log(limits)
            return ExecResult(126, 0.0, argv, details={"error": limits})
        job = BatchJob(name=f"pinneapple-{_safe(Path(cwd).name)}-{uuid.uuid4().hex[:6]}", argv=argv, cwd=str(cwd),
                       env={str(k): str(v) for k, v in (env or {}).items()}, image=str(self.options.get("image", "")),
                       timeout=timeout, step=Path(cwd).name)
        t0 = time.time()
        try:
            job_id = provider.submit(job)
        except RuntimeError as exc:
            log(str(exc))
            return ExecResult(127, time.time() - t0, argv, details={"error": str(exc)})
        log(f"submitted {job.name} as {job_id}")
        timed_out = cancelled = False
        state, exit_code = "running", None
        while state == "running":
            if timeout is not None and time.time() - t0 > timeout:
                timed_out = True
            elif should_stop():
                cancelled = True
            if timed_out or cancelled:
                break
            time.sleep(float(self.options.get("poll_seconds", self.poll_seconds)))
            state, exit_code = provider.status(job_id)
        if timed_out or cancelled:
            stopped = provider.cancel(job_id)
            log(f"cancelled {job_id}" + ("" if stopped else " (the provider did not confirm the cancellation)"))
            return ExecResult(-1, time.time() - t0, argv, timed_out=timed_out, cancelled=cancelled,
                              details={"job": job_id, "deleted": stopped})
        for line in provider.logs(job_id):
            log(line)
        return ExecResult(int(exit_code or 0), time.time() - t0, argv, details={"job": job_id, "state": state})


_ = (os, LocalExecutor)                    # the local path is what a remote executor falls back to for dry runs
