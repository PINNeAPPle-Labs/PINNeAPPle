"""What every adapter implements, and the context a step runs in.

An adapter wraps one application (FreeCAD, OpenFOAM, ParaView, a web UI) and offers named actions. Each action
receives its rendered ``params`` and a :class:`StepContext`, does the work, preferably through the application's own
API or command line, and returns an :class:`ActionResult`: outputs for later steps, the files it produced, the
images of what happened and the checks it ran on its own artifacts.

Executors of other runners (the Simulation Workflow Runner) can call the same adapters: ``get_adapter(name)``,
``adapter.available()``, ``adapter.run(action, params, ctx)``.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..display import VirtualDisplay
from ..record import Check, Evidence
from ..toolchain import Toolchain

__all__ = ["Adapter", "ActionResult", "StepContext", "ActionFailed", "register", "get_adapter", "list_adapters",
           "known_actions"]


class ActionFailed(RuntimeError):
    """The action ran and failed; the message says why, the log says where."""


@dataclass
class ActionResult:
    outputs: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass
class StepContext:
    run_dir: Path
    step_dir: Path
    toolchain: Toolchain
    display: VirtualDisplay | None = None
    workflow_dir: Path | None = None
    criteria: list[Any] = field(default_factory=list)
    previous: dict[str, Any] = field(default_factory=dict)       # step id -> StepRecord of the steps before
    timeout_s: float = 1800.0
    commands: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    on_progress: Callable[[str], None] | None = None

    def rel(self, p: str | Path) -> str:
        p = Path(p).resolve()
        try:
            return p.relative_to(self.run_dir.resolve()).as_posix()
        except ValueError:
            return str(p)

    def progress(self, msg: str) -> None:
        if self.on_progress:
            self.on_progress(msg)

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = self.display.env() if self.display else dict(os.environ)
        env.update(extra or {})
        return env

    def run(self, cmd: list[str] | str, cwd: Path, log_name: str, env: dict[str, str] | None = None,
            timeout_s: float | None = None, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command, writing stdout+stderr to ``<step_dir>/<log_name>`` and recording both."""
        shown = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
        self.commands.append(f"(cd {self.rel(cwd)} && {shown})")
        log = self.step_dir / log_name
        t0 = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            try:
                r = subprocess.run(cmd, cwd=cwd, env=env or self.env(), stdout=f, stderr=subprocess.STDOUT,
                                   shell=isinstance(cmd, str), timeout=timeout_s or self.timeout_s,
                                   executable="/bin/bash" if isinstance(cmd, str) else None)
            except subprocess.TimeoutExpired:
                self.logs.append(self.rel(log))
                raise ActionFailed(f"{shown} did not finish within {timeout_s or self.timeout_s:g} s "
                                   f"(log {self.rel(log)})") from None
        self.logs.append(self.rel(log))
        self.progress(f"{log_name} written in {time.time() - t0:.1f} s (exit {r.returncode})")
        if check and r.returncode != 0:
            tail = "".join(log.read_text(errors="replace").splitlines(keepends=True)[-8:]).strip()
            raise ActionFailed(f"{shown} exited with {r.returncode} (log {self.rel(log)}): {tail[-600:]}")
        return r


class Adapter:
    name = ""
    mode = "api"                             # api | gui
    requires: tuple[str, ...] = ()           # Toolchain.missing() names
    actions: tuple[str, ...] = ()
    needs_display: tuple[str, ...] = ()      # actions that need a (virtual) screen

    def available(self, toolchain: Toolchain) -> tuple[bool, str]:
        missing = toolchain.missing(list(self.requires))
        return (not missing, "missing: " + ", ".join(missing) if missing else "ok")

    def run(self, action: str, params: dict[str, Any], ctx: StepContext) -> ActionResult:
        fn = getattr(self, "do_" + action, None)
        if action not in self.actions or fn is None:
            raise ActionFailed(f"adapter {self.name} has no action {action!r}; it has {list(self.actions)}")
        return fn(params, ctx)


_REGISTRY: dict[str, Adapter] = {}


def register(adapter: Adapter) -> Adapter:
    _REGISTRY[adapter.name] = adapter
    return adapter


def get_adapter(name: str) -> Adapter:
    if name not in _REGISTRY:
        raise KeyError(f"no adapter {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_adapters() -> dict[str, Adapter]:
    return dict(_REGISTRY)


def known_actions() -> set[str]:
    return {f"{a.name}.{act}" for a in _REGISTRY.values() for act in a.actions}
