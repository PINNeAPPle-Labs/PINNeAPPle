"""Executors: where a step's command runs. A plugin interface, so other tools (a GUI agent, a cluster) add theirs.

=============  ==================================================================================================
``local``      a subprocess on this machine: timeout, environment, working directory, output to the step log
``docker``     ``docker run --rm`` with the step directory mounted at /work (no network, memory/CPU limits)
``slurm``      an sbatch script per step; ``dry_run=True`` (default) writes the script and does not submit it
``kubernetes`` one Job per step through ``kubectl``, logs streamed back, cancellation deletes the Job (#1479)
``batch``      one job per step on a cloud batch service, through a ``BatchProvider`` (#1479)
=============  ==================================================================================================

Write your own::

    class MyExecutor(Executor):
        name = "mine"
        def run(self, argv, *, cwd, env=None, timeout=None, log=print, should_stop=lambda: False) -> ExecResult:
            ...
    register_executor("mine", MyExecutor)       # steps with executor="mine" now use it

An executor only runs commands (argv lists). It does not install solvers, schedule a queue or replace Slurm: it
hands the command to what is already there.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Log = Callable[[str], None]


@dataclass
class ExecResult:
    """What one command did. ``metrics`` holds what the executor could *measure* — ``peak_rss_bytes`` for the local
    executor; a field the executor cannot measure is absent, never zero (#1481)."""

    returncode: int
    seconds: float
    command: list[str] = field(default_factory=list)
    dry_run: bool = False
    timed_out: bool = False
    cancelled: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not (self.timed_out or self.cancelled)


class Executor(ABC):
    """Runs one command for one step. Subclasses set ``name`` and implement :meth:`run`.

    Two declarations a caller reads without trying: ``remote`` (the command does not run on this machine, so a step
    that needs a local file or a screen cannot be handed to it — #1479) and ``measures_memory`` (whether
    ``ExecResult.metrics`` gets a ``peak_rss_bytes`` at all).
    """

    name: str = ""
    #: True when the command runs somewhere other than this process tree (a cluster, a container with its own memory).
    remote: bool = False
    #: True when this executor can measure the step's own peak memory (not the wrapper's).
    measures_memory: bool = False

    def __init__(self, **options: Any):
        self.options = options

    def available(self) -> tuple[bool, str]:
        """(usable here, reason if not)."""
        return True, ""

    def remote_limits(self, step: Any = None) -> str:
        """Why ``step`` cannot run here, or ``""``. The default is "it can"; a remote executor overrides it."""
        return ""

    @abstractmethod
    def run(self, argv: Sequence[str], *, cwd: str | Path, env: Mapping[str, str] | None = None,
            timeout: float | None = None, log: Log = print, should_stop: Callable[[], bool] = lambda: False
            ) -> ExecResult:
        ...

    def describe(self) -> dict[str, Any]:
        ok, why = self.available()
        return {"name": self.name, "available": ok, "reason": why, "options": self.options}


class LocalExecutor(Executor):
    """Subprocess on this machine. Output lines go to ``log``; timeout and ``should_stop`` kill the process.

    ``measure=True`` (the default) samples the process tree every iteration and reports ``peak_rss_bytes`` in
    ``ExecResult.metrics``; without ``psutil`` installed the field is absent, and the log says so once.
    """

    name = "local"
    measures_memory = True

    def run(self, argv, *, cwd, env=None, timeout=None, log=print, should_stop=lambda: False):
        argv = [str(a) for a in argv]
        full_env = {**os.environ, **{k: str(v) for k, v in (env or {}).items()}}
        t0 = time.time()
        log("$ " + shlex.join(argv))
        try:
            proc = subprocess.Popen(argv, cwd=str(cwd), env=full_env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
        except OSError as exc:
            log(f"cannot start: {exc}")
            return ExecResult(127, time.time() - t0, argv, details={"error": str(exc)})
        reader = threading.Thread(target=_pump, args=(proc, log), daemon=True)
        reader.start()
        peak = _PeakRSS(proc.pid) if self.options.get("measure", True) else None
        if peak is not None:
            peak.sample()                              # a command shorter than the poll interval still gets a reading
        timed_out = cancelled = False
        while proc.poll() is None:
            if timeout is not None and time.time() - t0 > timeout:
                timed_out = True
            elif should_stop():
                cancelled = True
            if timed_out or cancelled:
                _kill(proc)
                break
            if peak is not None:
                peak.sample()
            time.sleep(0.05)
        proc.wait()
        reader.join(5)
        if timed_out:
            log(f"killed after the time limit of {timeout:g} s")
        if cancelled:
            log("killed: cancelled")
        metrics: dict[str, Any] = {}
        if peak is not None:
            measured = peak.result()
            if measured:
                metrics["peak_rss_bytes"] = measured
            elif peak.unavailable:
                log(f"peak memory not measured: {peak.unavailable}")
        return ExecResult(proc.returncode, time.time() - t0, argv, timed_out=timed_out, cancelled=cancelled,
                          metrics=metrics)


class DockerExecutor(Executor):
    """``docker run --rm`` in ``image`` with the step directory at /work. Options: ``image`` (required),
    ``network`` (default "none"), ``memory`` ("2g"), ``cpus`` (1), ``user`` (this uid:gid), ``mounts``
    ({host: container}, read-only), ``extra`` (more docker run arguments), ``docker`` (binary)."""

    name = "docker"

    def available(self):
        if not shutil.which(self.options.get("docker", "docker")):
            return False, "docker is not installed"
        return True, ""

    def command_line(self, argv: Sequence[str], cwd: str | Path, env: Mapping[str, str] | None = None) -> list[str]:
        o = self.options
        if not o.get("image"):
            raise ValueError("DockerExecutor needs an image")
        cmd = [o.get("docker", "docker"), "run", "--rm", "--network", o.get("network", "none"),
               "-v", f"{Path(cwd).resolve()}:/work", "-w", "/work"]
        if o.get("memory", "2g"):
            cmd += ["--memory", str(o.get("memory", "2g"))]
        if o.get("cpus", 1):
            cmd += ["--cpus", str(o.get("cpus", 1))]
        user = o.get("user", f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else None)
        if user:
            cmd += ["--user", str(user)]
        for host, inside in (o.get("mounts") or {}).items():
            cmd += ["-v", f"{Path(host).resolve()}:{inside}:ro"]
        for k, v in (env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [str(x) for x in o.get("extra", [])]
        return cmd + [str(o["image"])] + [str(a) for a in argv]

    def run(self, argv, *, cwd, env=None, timeout=None, log=print, should_stop=lambda: False):
        ok, why = self.available()
        if not ok:
            log(why)
            return ExecResult(127, 0.0, list(argv), details={"error": why})
        result = LocalExecutor().run(self.command_line(argv, cwd, env), cwd=cwd, timeout=timeout, log=log,
                                     should_stop=should_stop)
        # what was sampled is the `docker` CLI's memory, not the container's: reporting it as the step's would be a
        # measurement of the wrong process, so it is dropped and the reason recorded instead
        if result.metrics.pop("peak_rss_bytes", None) is not None:
            result.details["memory_note"] = ("peak memory not reported: the docker CLI's own RSS is not the "
                                             "container's; set a --memory limit and read it from the engine")
        return result


class SlurmExecutor(Executor):
    """One sbatch script per step. Options: ``partition``, ``time`` ("01:00:00"), ``nodes`` (1), ``ntasks`` (1),
    ``cpus_per_task`` (1), ``mem`` ("4G"), ``account``, ``modules`` (``module load`` lines), ``setup`` (shell
    lines before the command), ``dry_run`` (default True: write ``<cwd>/step.sbatch`` and stop; False: submit with
    ``sbatch --wait`` and return its exit code)."""

    name = "slurm"

    def available(self):
        if self.options.get("dry_run", True):
            return True, ""
        if not shutil.which("sbatch"):
            return False, "sbatch not found (use dry_run to generate scripts)"
        return True, ""

    def script(self, argv: Sequence[str], cwd: str | Path, job_name: str = "pinneapple-step",
               env: Mapping[str, str] | None = None, timeout: float | None = None) -> str:
        o = self.options
        lines = ["#!/bin/bash", f"#SBATCH --job-name={_safe(job_name)}", f"#SBATCH --chdir={Path(cwd).resolve()}",
                 "#SBATCH --output=slurm-%j.out", f"#SBATCH --nodes={int(o.get('nodes', 1))}",
                 f"#SBATCH --ntasks={int(o.get('ntasks', 1))}",
                 f"#SBATCH --cpus-per-task={int(o.get('cpus_per_task', 1))}", f"#SBATCH --mem={o.get('mem', '4G')}",
                 f"#SBATCH --time={o.get('time') or _hms(timeout) or '01:00:00'}"]
        if o.get("partition"):
            lines.append(f"#SBATCH --partition={o['partition']}")
        if o.get("account"):
            lines.append(f"#SBATCH --account={o['account']}")
        lines += ["set -euo pipefail"]
        lines += [f"module load {m}" for m in o.get("modules", [])]
        lines += [f"export {k}={shlex.quote(str(v))}" for k, v in (env or {}).items()]
        lines += list(o.get("setup", []))
        launcher = ["srun"] if int(o.get("ntasks", 1)) > 1 else []
        lines.append(shlex.join(launcher + [str(a) for a in argv]))
        return "\n".join(lines) + "\n"

    def run(self, argv, *, cwd, env=None, timeout=None, log=print, should_stop=lambda: False):
        text = self.script(argv, cwd, self.options.get("job_name", Path(cwd).name), env, timeout)
        path = Path(cwd) / "step.sbatch"
        path.write_text(text)
        if self.options.get("dry_run", True):
            log(f"dry run: wrote {path.name} (not submitted)")
            return ExecResult(0, 0.0, ["sbatch", str(path)], dry_run=True, details={"script": str(path)})
        return LocalExecutor().run(["sbatch", "--wait", str(path)], cwd=cwd, timeout=timeout, log=log,
                                   should_stop=should_stop)


EXECUTORS: dict[str, type[Executor]] = {"local": LocalExecutor, "docker": DockerExecutor, "slurm": SlurmExecutor}

# The remote executors live in their own module (they import Executor from here, so they come after the classes above);
# registering them here keeps one map of what a step may name.
from .executors_remote import BatchExecutor, KubernetesExecutor  # noqa: E402

EXECUTORS["kubernetes"] = KubernetesExecutor
EXECUTORS["batch"] = BatchExecutor


def register_executor(name: str, cls: type[Executor]) -> None:
    """Make ``executor=name`` usable in steps (e.g. a GUI-driven executor from another app)."""
    if not (isinstance(cls, type) and issubclass(cls, Executor)):
        raise TypeError("an executor must subclass pinneapple_core.workflows.Executor")
    EXECUTORS[name] = cls


def get_executor(name: str, **options: Any) -> Executor:
    if name not in EXECUTORS:
        raise KeyError(f"unknown executor {name!r}; registered: {sorted(EXECUTORS)}")
    return EXECUTORS[name](**options)


class _PeakRSS:
    """Peak resident memory of a process and its children, sampled while the step runs.

    Sampling (rather than ``resource.getrusage(RUSAGE_CHILDREN)``) because ``ru_maxrss`` is the maximum over *every*
    child this process ever had: it would report the largest step of the session for every step after it.
    """

    def __init__(self, pid: int):
        self.pid = pid
        self.peak = 0
        self.unavailable = ""
        self._process = None
        try:
            import psutil

            self._process = psutil.Process(pid)
        except ImportError:
            self.unavailable = "psutil is not installed (pip install psutil)"
        except Exception as exc:                        # noqa: BLE001 - a measurement must not break the step
            self.unavailable = f"{type(exc).__name__}: {exc}"

    def sample(self) -> None:
        if self._process is None:
            return
        try:
            total = int(self._process.memory_info().rss)
            for child in self._process.children(recursive=True):
                try:
                    total += int(child.memory_info().rss)
                except Exception:                       # noqa: BLE001 - a child may exit mid-sample
                    continue
            self.peak = max(self.peak, total)
        except Exception:                               # noqa: BLE001 - the process may already be gone
            self._process = None

    def result(self) -> int:
        return self.peak


def _pump(proc, log: Log) -> None:
    for line in proc.stdout:
        log(line.rstrip("\n"))


def _kill(proc) -> None:
    try:
        os.killpg(proc.pid, 9)
    except (OSError, AttributeError):
        proc.kill()


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:64]


def _hms(seconds: float | None) -> str | None:
    if not seconds:
        return None
    s = int(seconds + 59) // 60 * 60
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:00"
