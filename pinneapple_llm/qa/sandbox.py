"""Stage D: controlled execution of generated artifacts.

Read this before relying on it
------------------------------
The isolation implemented here is **subprocess isolation with in-process
policy enforcement**, not an operating-system sandbox. Concretely, an
artifact runs as a separate process (:mod:`subprocess`) in its own
temporary directory, with:

* a scrubbed environment (no inherited secrets, ``HOME`` pointing at the
  temporary directory);
* POSIX resource limits (CPU time, address space, file size, open files,
  no core dumps) applied with :mod:`resource` before the interpreter
  starts;
* a wall-clock timeout after which the process is killed;
* a Python audit hook installed *inside* the child process that refuses
  network connections, process spawning, and file access outside the
  artifact directory and the interpreter's own library paths;
* every produced file recorded with its SHA-256.

What this does **not** do: it does not stop C extensions or ``ctypes``
from issuing syscalls directly, it does not create a container, a user
namespace, a seccomp filter or a chroot, and it does not stop a process
that escapes the audit hook. A determined artifact can read memory it
should not. The audit hook is a *policy* layer that catches the ordinary
failure modes (a generated script phoning home, writing to ``~/.ssh``,
shelling out) and reports them; it is not a security boundary between
hostile code and the host.

Because of that, :class:`SandboxExecutor` refuses to run anything unless
constructed with ``allow_execution=True``, and the pipeline's default is
to leave execution ``BLOCKED``. See :attr:`SandboxResult.limitations`,
which carries this same statement into every report.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from .report import ArtifactRecord, Status

VIOLATION_MARKER = "__PINNEAPPLE_SANDBOX_VIOLATION__:"

ISOLATION_LIMITATIONS = (
    "subprocess isolation only: no container, namespace, seccomp or chroot",
    "the audit hook is enforced in-process and can be bypassed by native code",
    "resource limits are POSIX rlimits; they are advisory on some platforms",
    "no CPU/memory accounting beyond what rlimits and wall-clock timeout provide",
)


class IsolationLevel(str, Enum):
    BLOCKED = "blocked"      # execution not permitted by configuration
    SUBPROCESS = "subprocess"  # separate process, rlimits, audit hook


@dataclass(frozen=True)
class SandboxLimits:
    """Resource ceilings. ``None`` disables the corresponding limit."""

    timeout_s: float = 30.0
    cpu_s: Optional[int] = 30
    memory_mb: Optional[int] = 2048
    file_size_mb: Optional[int] = 16
    open_files: Optional[int] = 256
    max_processes: Optional[int] = None
    write_bytes: int = 64 * 1024 * 1024  # total bytes of produced files


@dataclass
class SandboxResult:
    status: Status
    isolation: IsolationLevel
    workdir: str = ""
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    violations: List[str] = field(default_factory=list)
    produced: List[ArtifactRecord] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is Status.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value, "isolation": self.isolation.value,
            "exit_code": self.exit_code, "duration_s": round(self.duration_s, 4),
            "timed_out": self.timed_out, "violations": list(self.violations),
            "produced": [p.to_dict() for p in self.produced],
            "stdout_tail": self.stdout[-4000:], "stderr_tail": self.stderr[-4000:],
            "limitations": list(self.limitations), "reason": self.reason,
        }


_RUNNER = r'''
import json, os, runpy, sys

WORKDIR = os.path.dirname(os.path.abspath(__file__))
VIOLATIONS = os.path.join(WORKDIR, "_violations.jsonl")
ALLOW_WRITE = os.environ.get("PP_SANDBOX_ALLOW_WRITE", "1") == "1"
ALLOW_NET = os.environ.get("PP_SANDBOX_ALLOW_NET", "0") == "1"

_allow = [WORKDIR]
for p in list(sys.path) + [sys.base_prefix, sys.prefix, os.path.dirname(os.__file__)]:
    if p and os.path.isdir(p):
        _allow.append(os.path.abspath(p))
        _allow.append(os.path.realpath(p))
_allow.append(os.path.realpath(WORKDIR))
_allow = tuple(dict.fromkeys(_allow))


def _record(kind, detail):
    with open(VIOLATIONS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": kind, "detail": str(detail)}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _inside(path):
    try:
        p = os.path.abspath(path)
        rp = os.path.realpath(path)
    except Exception:
        return False
    return any(
        p == root or p.startswith(root + os.sep) or rp == root or rp.startswith(root + os.sep)
        for root in _allow
    )


def _hook(event, args):
    if event == "open":
        path, mode, flags = args
        if path is None or isinstance(path, int):
            return
        try:
            p = os.fspath(path)
        except TypeError:
            return
        if not isinstance(p, str):
            return
        writing = bool(mode and any(c in str(mode) for c in "wax+")) or bool(
            flags and (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        )
        if not _inside(p):
            _record("file_write" if writing else "file_read", p)
            raise PermissionError("pinneapple sandbox: file access outside the artifact directory: %s" % p)
        return
    if event in ("socket.connect", "socket.bind", "socket.sendto", "socket.getaddrinfo"):
        if not ALLOW_NET:
            _record("network", event)
            raise PermissionError("pinneapple sandbox: network access is disabled (%s)" % event)
        return
    if event in ("subprocess.Popen", "os.system", "os.exec", "os.fork", "os.posix_spawn", "os.spawn"):
        _record("process", event)
        raise PermissionError("pinneapple sandbox: process creation is disabled (%s)" % event)
    # NOTE: ``ctypes.dlopen`` is deliberately NOT blocked. numpy and torch load
    # native libraries at import time (including ``dlopen(None)``), so blocking
    # it would break every scientific import. Native code is the documented
    # escape hatch of this isolation level -- see ISOLATION_LIMITATIONS.


sys.addaudithook(_hook)
sys.path.insert(0, WORKDIR)
runpy.run_path(os.path.join(WORKDIR, "main.py"), run_name="__main__")
'''


class SandboxExecutor:
    """Run a generated Python artifact in a separate, limited process."""

    def __init__(self, *, limits: Optional[SandboxLimits] = None,
                 allow_execution: bool = False,
                 allow_network: bool = False,
                 base_dir: Optional[str] = None,
                 keep_workdir: bool = False,
                 python: Optional[str] = None):
        self.limits = limits or SandboxLimits()
        self.allow_execution = allow_execution
        self.allow_network = allow_network
        self.base_dir = base_dir
        self.keep_workdir = keep_workdir
        self.python = python or sys.executable

    # -- public API -------------------------------------------------------
    def run_python(self, source: str, *, files: Optional[Dict[str, bytes]] = None,
                   argv: Iterable[str] = (), output_dir: Optional[str] = None) -> SandboxResult:
        """Execute ``source`` as ``main.py`` and report what happened.

        ``files`` are written next to it before the run (inputs the artifact
        reads). ``output_dir`` receives a copy of every produced file, so the
        caller can inspect them after the temporary tree is removed.
        """
        if not self.allow_execution:
            return SandboxResult(
                status=Status.BLOCKED,
                isolation=IsolationLevel.BLOCKED,
                reason="execution is disabled: construct SandboxExecutor(allow_execution=True) "
                       "or QualityPipeline(run_artifacts=True) to permit it",
                limitations=list(ISOLATION_LIMITATIONS),
            )

        workdir = tempfile.mkdtemp(prefix="pp-qa-", dir=self.base_dir)
        started = time.monotonic()
        try:
            with open(os.path.join(workdir, "main.py"), "w", encoding="utf-8") as fh:
                fh.write(source)
            with open(os.path.join(workdir, "_runner.py"), "w", encoding="utf-8") as fh:
                fh.write(_RUNNER)
            for name, blob in (files or {}).items():
                target = os.path.join(workdir, name)
                os.makedirs(os.path.dirname(target) or workdir, exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(blob)

            env = {
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "HOME": workdir,
                "TMPDIR": workdir,
                "LC_ALL": "C",
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PP_SANDBOX_ALLOW_WRITE": "1",
                "PP_SANDBOX_ALLOW_NET": "1" if self.allow_network else "0",
            }
            cmd = [self.python, "-I", "-B", os.path.join(workdir, "_runner.py"), *argv]
            try:
                proc = subprocess.run(
                    cmd, cwd=workdir, env=env, capture_output=True, text=True,
                    timeout=self.limits.timeout_s,
                    preexec_fn=self._rlimits(workdir) if os.name == "posix" else None,
                )
                exit_code, stdout, stderr, timed_out = proc.returncode, proc.stdout, proc.stderr, False
            except subprocess.TimeoutExpired as exc:
                exit_code = None
                stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
                stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
                timed_out = True

            duration = time.monotonic() - started
            violations = self._read_violations(workdir)
            produced = self._collect(workdir, files or {}, output_dir)
            limitations = list(ISOLATION_LIMITATIONS) + self._read_ignored_limits(workdir)

            if timed_out:
                status = Status.FAIL
                reason = f"timed out after {self.limits.timeout_s}s and was killed"
            elif violations:
                status = Status.FAIL
                reason = f"policy violation(s): {len(violations)}"
            elif exit_code != 0:
                status = Status.FAIL
                reason = f"exited with code {exit_code}"
            else:
                status = Status.PASS
                reason = "exited 0 within limits"

            return SandboxResult(
                status=status, isolation=IsolationLevel.SUBPROCESS, workdir=workdir,
                exit_code=exit_code, stdout=stdout, stderr=stderr, duration_s=duration,
                timed_out=timed_out, violations=violations, produced=produced,
                limitations=limitations, reason=reason,
            )
        finally:
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)

    # -- internals --------------------------------------------------------
    def _rlimits(self, workdir: str):
        limits = self.limits

        def apply():  # runs in the child, after fork, before exec
            import json as _json
            import resource

            ignored = []

            def setr(which, value, label):
                if value is None:
                    return
                try:
                    resource.setrlimit(which, (value, value))
                except (ValueError, OSError) as exc:
                    # a limit the platform refused is reported, not hidden: a
                    # sandbox that silently dropped its own ceiling is worse
                    # than one that never claimed it
                    ignored.append(f"{label}: {type(exc).__name__}: {exc}")

            setr(resource.RLIMIT_CPU, limits.cpu_s, "cpu")
            setr(resource.RLIMIT_FSIZE, (limits.file_size_mb or 0) * 1024 * 1024, "fsize")
            setr(resource.RLIMIT_NOFILE, limits.open_files, "nofile")
            setr(resource.RLIMIT_CORE, 0, "core")
            if limits.max_processes is not None:
                setr(resource.RLIMIT_NPROC, limits.max_processes, "nproc")
            if limits.memory_mb is not None:
                try:
                    resource.setrlimit(resource.RLIMIT_AS,
                                       (limits.memory_mb * 1024 * 1024,) * 2)
                except (ValueError, OSError) as exc:
                    ignored.append(f"address_space: {type(exc).__name__}: {exc}")
            if ignored:
                try:
                    with open(os.path.join(workdir, "_limits.json"), "w", encoding="utf-8") as fh:
                        _json.dump(ignored, fh)
                except OSError:
                    pass

        return apply

    @staticmethod
    def _read_ignored_limits(workdir: str) -> List[str]:
        path = os.path.join(workdir, "_limits.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path, encoding="utf-8") as fh:
                ignored = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return ["resource limits: the child could not report which limits it applied"]
        return [f"resource limit not applied: {entry}" for entry in ignored]

    @staticmethod
    def _read_violations(workdir: str) -> List[str]:
        path = os.path.join(workdir, "_violations.jsonl")
        out: List[str] = []
        if not os.path.exists(path):
            return out
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    out.append(line)
                    continue
                out.append(f"{rec.get('kind')}: {rec.get('detail')}")
        return out

    def _collect(self, workdir: str, inputs: Dict[str, bytes],
                 output_dir: Optional[str]) -> List[ArtifactRecord]:
        produced: List[ArtifactRecord] = []
        total = 0
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        for root, dirs, names in os.walk(workdir):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in sorted(names):
                if name in ("main.py", "_runner.py", "_violations.jsonl", "_limits.json"):
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, workdir)
                if rel in inputs:
                    continue
                size = os.path.getsize(full)
                total += size
                if total > self.limits.write_bytes:
                    produced.append(ArtifactRecord(name=rel, kind="rejected",
                                                   metadata={"reason": "write budget exceeded"}))
                    break
                record = ArtifactRecord.from_file(rel, full, kind="sandbox_output")
                if output_dir:
                    dest = os.path.join(output_dir, rel)
                    os.makedirs(os.path.dirname(dest) or output_dir, exist_ok=True)
                    shutil.copy2(full, dest)
                    record.path = dest
                produced.append(record)
        return produced


def hash_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()
