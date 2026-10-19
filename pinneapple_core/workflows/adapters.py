"""Adapters: what a step of a given ``kind`` does. Each prepares its directory, runs commands through the step's
executor and returns its outputs (a JSON-able dict).

=============  ==================================================================================================
``command``    ``argv`` (list, ``{param}`` placeholders filled); outputs: ``outputs.json`` if the command wrote one
``python``     ``function: "module:fn"`` called in-process as ``fn(params, ctx)`` (local executor only), or
               ``script: path`` run with the executor as ``python script params.json``; outputs as for ``command``
``openfoam``   copies a case (``case`` directory, or ``case_from`` an upstream step), applies ``edits``
               (``openfoam.campaign.edit_file``), runs ``commands`` (default blockMesh + the solver) and reads the
               solver log health (``openfoam.campaign.parse_solver_log``)
``calculix``   copies ``inp`` (a file, or ``inp_from`` an upstream step's ``<job>.inp``), runs ``ccx -i <job>`` and
               reads the .frd (``calculix.frd.read_frd``): maximum displacement, node count
=============  ==================================================================================================

A new kind::

    class GuiStep(Adapter):
        kind = "gui"
        def run(self, ctx: StepContext) -> dict: ...
    register_adapter(GuiStep())
"""
from __future__ import annotations

import glob
import importlib
import json
import os
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dag import Step
from .executors import ExecResult, Executor


class StepFailed(RuntimeError):
    """The step ran and failed (non-zero exit, solver error, missing output)."""


@dataclass
class StepContext:
    run_id: str
    step: Step
    workdir: Path
    executor: Executor
    upstream: dict[str, dict[str, Any]] = field(default_factory=dict)   # step id -> {"outputs", "workdir"}
    log: Callable[[str], None] = print
    should_stop: Callable[[], bool] = lambda: False
    attempt: int = 1

    @property
    def params(self) -> dict[str, Any]:
        return self.step.params

    def execute(self, argv: list[str], env: dict[str, str] | None = None) -> ExecResult:
        """Run ``argv`` with the step's executor in its directory; StepFailed unless it succeeded."""
        res = self.executor.run(argv, cwd=self.workdir, env=env, timeout=self.step.timeout, log=self.log,
                                should_stop=self.should_stop)
        if res.cancelled:
            raise StepCancelled("cancelled")
        if res.timed_out:
            raise StepFailed(f"time limit of {self.step.timeout:g} s reached")
        if res.returncode != 0:
            raise StepFailed(f"{argv[0]} exited with code {res.returncode}")
        return res

    def upstream_dir(self, step_id: str) -> Path:
        """Directory of an upstream step; inside a per-case template, ``"deck"`` means ``"<case>.deck"``."""
        if step_id not in self.upstream and f"{self.step.group}.{step_id}" in self.upstream:
            step_id = f"{self.step.group}.{step_id}"
        if step_id not in self.upstream:
            raise StepFailed(f"{step_id!r} is not an upstream step of {self.step.id!r}")
        return Path(self.upstream[step_id]["workdir"])


class StepCancelled(RuntimeError):
    pass


class Adapter(ABC):
    kind: str = ""

    def available(self, step: Step | None = None) -> tuple[bool, str]:
        return True, ""

    @abstractmethod
    def run(self, ctx: StepContext) -> dict[str, Any]:
        ...


def _outputs_json(ctx: StepContext) -> dict[str, Any]:
    p = ctx.workdir / "outputs.json"
    if p.is_file():
        try:
            out = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise StepFailed(f"outputs.json is not valid JSON: {exc}") from None
        if not isinstance(out, dict):
            raise StepFailed("outputs.json must hold a JSON object")
        return out
    return {}


class CommandAdapter(Adapter):
    kind = "command"

    def run(self, ctx):
        argv = ctx.params.get("argv")
        if not argv or not isinstance(argv, list):
            raise StepFailed("a command step needs params.argv (a list)")
        res = ctx.execute([str(a) for a in argv], env=ctx.params.get("env"))
        return {**_outputs_json(ctx), "returncode": res.returncode, "dry_run": res.dry_run}


class PythonAdapter(Adapter):
    kind = "python"

    def run(self, ctx):
        if "function" in ctx.params:
            if ctx.executor.name != "local":
                raise StepFailed("a python function step runs in-process (executor 'local'); use 'script' for "
                                 f"the {ctx.executor.name!r} executor")
            fn = resolve_function(ctx.params["function"])
            out = fn(dict(ctx.params.get("kwargs", {})), ctx)
            if out is None:
                return {}
            if not isinstance(out, dict):
                raise StepFailed("a python step function must return a dict")
            return out
        if "script" in ctx.params:
            (ctx.workdir / "params.json").write_text(json.dumps(ctx.params.get("kwargs", {})))
            python = ctx.params.get("python") or (sys.executable if ctx.executor.name == "local" else "python3")
            res = ctx.execute([python, str(ctx.params["script"]), "params.json"])
            return {**_outputs_json(ctx), "dry_run": res.dry_run}
        raise StepFailed("a python step needs params.function ('module:fn') or params.script")


def resolve_function(ref: str) -> Callable:
    module, _, qual = str(ref).partition(":")
    if not module or not qual:
        raise StepFailed(f"function reference {ref!r} must be 'module:function'")
    obj: Any = importlib.import_module(module)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


_FOAM_BASHRCS = ("/usr/share/openfoam/etc/bashrc", "/usr/lib/openfoam/openfoam*/etc/bashrc",
                 "/opt/openfoam*/etc/bashrc", "/opt/OpenFOAM-*/etc/bashrc")


def _foam_bashrc(p: dict) -> str | None:
    """The OpenFOAM bashrc to source before the commands: the step's ``bashrc``, else ``FOAM_BASHRC``, else (when the
    environment is not loaded already) the first installation found in the usual places. None: nothing to source."""
    for b in (p.get("bashrc"), os.environ.get("FOAM_BASHRC")):
        if b and Path(b).is_file():
            return str(b)
    if os.environ.get("WM_PROJECT_DIR"):
        return None
    for pattern in _FOAM_BASHRCS:
        found = sorted(glob.glob(pattern))
        if found:
            return found[-1]
    return None


class OpenFOAMAdapter(Adapter):
    """params: ``case`` (directory) or ``case_from`` (upstream step id), ``edits`` [{file, key, value}],
    ``solver`` (default icoFoam), ``commands`` (default ["blockMesh", solver]), ``bashrc`` (sourced first)."""

    kind = "openfoam"

    def available(self, step=None):
        p = step.params if step else {}
        if _foam_bashrc(p):
            return True, ""
        solver = p.get("solver", "icoFoam")
        if not (shutil.which(solver) and shutil.which("blockMesh")):
            return False, f"OpenFOAM not found ({solver} not on PATH and no bashrc)"
        if not os.environ.get("WM_PROJECT_DIR"):
            # distribution packages put the binaries on PATH, but they only run with the environment loaded
            return False, (f"{solver} is on PATH but the OpenFOAM environment is not loaded and no bashrc was found: "
                           "source it, or set FOAM_BASHRC or the step's bashrc")
        return True, ""

    def run(self, ctx):
        from pinneapple_simulation.external_solvers.openfoam.campaign import (
            edit_file,
            parse_solver_log,
        )
        p = ctx.params
        src = ctx.upstream_dir(p["case_from"]) / p.get("case_subdir", "case") if p.get("case_from") \
            else Path(p["case"])
        if not src.is_dir():
            raise StepFailed(f"OpenFOAM case {src} not found")
        case = ctx.workdir / "case"
        shutil.rmtree(case, ignore_errors=True)
        shutil.copytree(src, case)
        for e in p.get("edits", []):
            edit_file(case / e["file"], e["key"], str(e["value"]))
        solver = p.get("solver", "icoFoam")
        bashrc = _foam_bashrc(p)
        cmds = p.get("commands") or ["blockMesh", solver]
        line = "cd case && " + " && ".join(f"{c} > log.{c.split()[0]} 2>&1" for c in cmds)
        if bashrc:
            line = f"source {bashrc} >/dev/null 2>&1 && " + line
        try:
            res = ctx.execute(["bash", "-c", line])
        finally:
            for c in cmds:
                log = case / f"log.{c.split()[0]}"
                if log.is_file():
                    for ln in log.read_text(errors="replace").splitlines()[-5:]:
                        ctx.log(f"[{c.split()[0]}] {ln}")
        if res.dry_run:
            return {"solver": solver, "dry_run": True}
        log = case / f"log.{solver}"
        health = parse_solver_log(log.read_text(errors="replace")) if log.is_file() else {}
        times = sorted((d.name for d in case.iterdir() if d.is_dir() and _is_number(d.name)), key=float)
        return {"solver": solver, "latest_time": float(times[-1]) if times else None, "health": health}


class CalculiXAdapter(Adapter):
    """params: ``inp`` (file) or ``inp_from`` (upstream step id holding ``<job>.inp``), ``job`` (default "job"),
    ``threads`` (1)."""

    kind = "calculix"

    def available(self, step=None):
        if shutil.which("ccx"):
            return True, ""
        return False, "CalculiX (ccx) not on PATH"

    def run(self, ctx):
        import numpy as np

        from pinneapple_simulation.external_solvers.calculix.frd import read_frd
        p = ctx.params
        job = p.get("job", "job")
        src = ctx.upstream_dir(p["inp_from"]) / f"{job}.inp" if p.get("inp_from") else Path(p["inp"])
        if not src.is_file():
            raise StepFailed(f"CalculiX input {src} not found")
        shutil.copy(src, ctx.workdir / f"{job}.inp")
        res = ctx.execute(["ccx", "-i", job], env={"OMP_NUM_THREADS": str(p.get("threads", 1))})
        if res.dry_run:
            return {"dry_run": True}
        frd = ctx.workdir / f"{job}.frd"
        if not frd.is_file():
            raise StepFailed("ccx wrote no .frd file")
        res = read_frd(str(frd))
        disp = [s["DISP"] for s in res.steps if "DISP" in s]
        out: dict[str, Any] = {"nodes": int(len(res.node_ids)), "frd": frd.name}
        if disp:
            out["max_displacement"] = float(np.linalg.norm(np.asarray(disp[-1])[:, :3], axis=1).max())
        return out


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


ADAPTERS: dict[str, Adapter] = {a.kind: a for a in (CommandAdapter(), PythonAdapter(), OpenFOAMAdapter(),
                                                    CalculiXAdapter())}


def register_adapter(adapter: Adapter) -> None:
    """Make steps of ``adapter.kind`` runnable (e.g. a GUI-driven step from another app)."""
    if not isinstance(adapter, Adapter) or not adapter.kind:
        raise TypeError("an adapter must be an Adapter instance with a kind")
    ADAPTERS[adapter.kind] = adapter


def get_adapter(kind: str) -> Adapter:
    if kind not in ADAPTERS:
        raise KeyError(f"unknown step kind {kind!r}; registered: {sorted(ADAPTERS)}")
    return ADAPTERS[kind]
