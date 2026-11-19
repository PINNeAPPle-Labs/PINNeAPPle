"""Solver-neutral execution contract for native OpenFOAM, CalculiX and FEniCSx cases.

This is an additive interface. Native cases preserve solver-specific physics; the
adapter never claims to translate arbitrary PDEs or meshes between solvers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
import os
import re
import shutil
import subprocess
import time


class SimulationError(RuntimeError):
    """A solver failed or its required output was not produced."""


class UnsupportedSimulation(SimulationError):
    """The requested execution mode is unsupported by this backend."""


@dataclass(frozen=True)
class SimulationSpec:
    backend: str
    workspace: Path
    case: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)
    physics: str = "native"
    reference_frame: str = "global"


@dataclass(frozen=True)
class SimulationResult:
    backend: str
    status: str
    workspace: Path
    artifacts: Mapping[str, str]
    diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]


class SolverAdapter(Protocol):
    def capabilities(self) -> Mapping[str, Any]: ...
    def validate(self, spec: SimulationSpec) -> None: ...
    def execute(self, spec: SimulationSpec) -> SimulationResult: ...


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.+-]*", value):
        raise ValueError("Unsafe solver/job name")
    return value


def _run(argv: list[str], cwd: Path, timeout: float, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SimulationError(f"Could not execute {argv[0]}: {exc}") from exc
    if p.returncode:
        raise SimulationError(f"{argv[0]} failed (exit {p.returncode}): {(p.stdout + p.stderr)[-3000:]}")
    return p


class OpenFOAMAdapter:
    """Run an existing, complete native case; no hidden case generation."""
    def capabilities(self) -> Mapping[str, Any]:
        return {"native_case": True, "physics": "solver-native", "parallel": False,
                "meshing": False, "field_conversion": False}

    def validate(self, spec: SimulationSpec) -> None:
        if spec.backend != "openfoam":
            raise UnsupportedSimulation("Expected openfoam backend")
        if not (spec.workspace / "system" / "controlDict").is_file():
            raise SimulationError("OpenFOAM system/controlDict missing")
        _safe_name(str(spec.options.get("solver", "simpleFoam")))
        if int(spec.options.get("n_procs", 1)) != 1:
            raise UnsupportedSimulation("This native-case MVP supports serial runs only")
        if spec.options.get("foam_bashrc"):
            raise UnsupportedSimulation("Source OpenFOAM environment before invoking the adapter")
        if shutil.which(str(spec.options.get("solver", "simpleFoam"))) is None:
            raise SimulationError("OpenFOAM executable not found on PATH")

    def execute(self, spec: SimulationSpec) -> SimulationResult:
        self.validate(spec)
        solver = str(spec.options.get("solver", "simpleFoam"))
        timeout = float(spec.options.get("timeout", 3600))
        start = time.monotonic()
        p = _run([solver, "-case", str(spec.workspace.resolve())], spec.workspace, timeout)
        log = spec.workspace / "pinneapple-solver.log"
        log.write_text(p.stdout + p.stderr, encoding="utf-8")
        return SimulationResult("openfoam", "completed", spec.workspace, {"log": str(log)},
                                {"returncode": p.returncode, "elapsed_s": time.monotonic() - start},
                                {"solver": solver, "reference_frame": spec.reference_frame,
                                 "field_extraction": "not_performed"})


class CalculiXAdapter:
    """Run an existing .inp job via the repository's established ccx runner."""
    def capabilities(self) -> Mapping[str, Any]:
        return {"native_case": True, "physics": "solver-native", "parallel": "threads",
                "field_conversion": False}

    def validate(self, spec: SimulationSpec) -> None:
        if spec.backend != "calculix":
            raise UnsupportedSimulation("Expected calculix backend")
        job = _safe_name(spec.case)
        if not (spec.workspace / (job + ".inp")).is_file():
            raise SimulationError(f"Missing CalculiX input deck: {job}.inp")
        if int(spec.options.get("threads", 1)) < 1:
            raise ValueError("threads must be positive")

    def execute(self, spec: SimulationSpec) -> SimulationResult:
        self.validate(spec)
        from .calculix.runner import run_ccx
        job = spec.case
        start = time.monotonic()
        try:
            result = run_ccx(str(spec.workspace.resolve()), job,
                             threads=int(spec.options.get("threads", 1)),
                             timeout=float(spec.options.get("timeout", 3600)),
                             backend=spec.options.get("execution_backend"))
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            raise SimulationError(f"CalculiX execution failed: {exc}") from exc
        if result.returncode:
            raise SimulationError(f"CalculiX exit {result.returncode}: {result.stdout[-3000:]}")
        outputs = {key: path for key, path in {"frd": result.frd, "dat": result.dat}.items()
                   if Path(path).is_file()}
        if not outputs:
            raise SimulationError("CalculiX returned success but no FRD/DAT artifacts")
        return SimulationResult("calculix", "completed", spec.workspace, outputs,
                                {"returncode": result.returncode, "elapsed_s": time.monotonic() - start},
                                {"job": job, "reference_frame": spec.reference_frame,
                                 "field_extraction": "not_performed"})


class FEniCSxAdapter:
    """Execute a user-supplied dolfinx variational solve callable in-process.

    Only use trusted Python callables: this is NOT a remote-code sandbox.
    The callable receives the SimulationSpec and returns a non-None solution.
    """
    def __init__(self, solve: Callable[[SimulationSpec], Any] | None = None):
        self.solve = solve

    def capabilities(self) -> Mapping[str, Any]:
        return {"variational_form": True, "physics": "user-defined",
                "execution": "trusted-in-process", "field_conversion": False}

    def validate(self, spec: SimulationSpec) -> None:
        if spec.backend != "fenicsx":
            raise UnsupportedSimulation("Expected fenicsx backend")
        if not callable(self.solve):
            raise UnsupportedSimulation("Provide a trusted dolfinx solve callable")

    def execute(self, spec: SimulationSpec) -> SimulationResult:
        self.validate(spec)
        start = time.monotonic()
        try:
            solution = self.solve(spec)
        except Exception as exc:
            raise SimulationError(f"FEniCSx solve failed: {exc}") from exc
        if solution is None:
            raise SimulationError("FEniCSx callable returned no solution")
        return SimulationResult("fenicsx", "completed", spec.workspace, {},
                                {"elapsed_s": time.monotonic() - start},
                                {"solution_type": type(solution).__name__,
                                 "reference_frame": spec.reference_frame,
                                 "field_extraction": "caller_owned"})


def get_adapter(backend: str, *, fenicsx_solve: Callable[[SimulationSpec], Any] | None = None) -> SolverAdapter:
    adapters: dict[str, SolverAdapter] = {
        "openfoam": OpenFOAMAdapter(),
        "calculix": CalculiXAdapter(),
        "fenicsx": FEniCSxAdapter(fenicsx_solve),
    }
    try:
        return adapters[backend]
    except KeyError as exc:
        raise UnsupportedSimulation(f"Unknown backend {backend!r}; available: {', '.join(adapters)}") from exc


def simulate(spec: SimulationSpec, *, fenicsx_solve: Callable[[SimulationSpec], Any] | None = None) -> SimulationResult:
    return get_adapter(spec.backend, fenicsx_solve=fenicsx_solve).execute(spec)
