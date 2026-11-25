"""One interface every external solver is reached through.

The problem: OpenFOAM, CalculiX, Fluent, FEniCSx and MATLAB each arrived with their own case layout, their own health
check, their own log parsing and their own idea of what "finished" means, so wiring a fourth solver meant reading the
third one's code. The gap was never coverage, it was the contract (PINNeAPPle#1476).

Every backend answers the same six questions::

    check_environment()  what is missing here -- executable, version, licence, environment, Python module; never raises
    prepare(case)        materialise the input from a neutral :class:`SolverCase`; returns the run directory
    run()                start it, return a :class:`RunHandle`
    status()             queued / running / converged / diverged / failed / cancelled, with the evidence
    results()            fields and scalars in one schema, provenance naming the backend and its version
    cancel()             idempotent

:func:`classify_run` is *the* one definition of "finished": a divergence marker in the log outranks the exit code (a
diverged run usually exits non-zero and would otherwise read as a plain failure), then the exit code, then the
artefact the solver was supposed to write, then the solver's own end marker. A backend that diverges therefore never
looks like a backend that crashed.

The contract is a layer over the implementations, not a rewrite: :mod:`.backends` maps each existing bridge onto it,
and ``pinneapple_core.workflows.adapters`` asks it for the environment verdict, so the workflow surface and this one
cannot disagree about whether a solver is installed::

    from pinneapple_simulation.external_solvers import SolverCase, check_all, get_solver

    for name, report in check_all().items():
        print(report.explain())                    # every solver: what it needs, what this machine lacks, the fix

    solver = get_solver("calculix", workdir="run")
    solver.check_environment().raise_if_missing()  # a readable diagnosis instead of a crash mid-run
    solver.prepare(SolverCase.cantilever(P=1000.0))
    solver.run()
    solver.status()                                # -> converged / diverged / failed, with the log lines
    sample = solver.results().to_upd()             # the neutral scientific schema (UPD), provenance included

A new backend subclasses :class:`ExternalSolver`, supplies ``_requirements``, ``prepare``, ``_start``, ``_results`` and
declares its log markers, then registers itself with :func:`register_solver`.

What is *not* here, deliberately: **COMSOL**. This interface does not pretend to cover it. There is no COMSOL
implementation in this repository and there cannot be an honest one here -- its Python coupling (``mph``/LiveLink)
needs the commercial client installed and licensed, and its model files are binary and proprietary, so nothing about it
could be exercised or verified by this repository's tests. A backend nobody can run is worse than no backend: it moves
the failure from :meth:`ExternalSolver.check_environment` to the middle of a run, which is the problem this contract
exists to fix. COMSOL is therefore out of scope until a licensed client can be exercised (see
``docs/guides/external_solver_contract.md``), and adding it later is one class plus :func:`register_solver` --
``get_solver("comsol")`` says so today with a ``KeyError`` that names the backends which do exist.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    # the neutral description
    "SolverCase", "conduction_profile",
    "LINEAR_ELASTICITY", "INCOMPRESSIBLE_FLOW", "HEAT_TRANSFER", "OPAQUE",
    # what the environment says
    "Requirement", "EnvironmentReport", "EXECUTABLE", "VERSION", "LICENCE", "ENVIRONMENT", "MODULE",
    # what a run says
    "SolverState", "SolverStatus", "SolverResults", "RunHandle", "classify_run",
    # the interface, its errors and the registry
    "ExternalSolver", "SolverError", "SolverUnavailable", "CaseUnsupported", "RunUnusable",
    "SOLVERS", "register_solver", "solver_names", "get_solver", "check_all",
]

# ---------------------------------------------------------------------------------------------- the neutral description
#: A structural case: a clamped solid under a load, solved for displacement and stress.
LINEAR_ELASTICITY = "linear_elasticity"
#: A flow case: an incompressible case directory or deck the backend stages as it is.
INCOMPRESSIBLE_FLOW = "incompressible_flow"
#: Steady or transient heat conduction.
HEAT_TRANSFER = "heat_transfer"
#: The backend stages ``source`` as it is; the neutral description carries no physics of its own (a case directory, a
#: deck, a journal, a script).
OPAQUE = "opaque"

_PHYSICS = (LINEAR_ELASTICITY, INCOMPRESSIBLE_FLOW, HEAT_TRANSFER, OPAQUE)


@dataclass
class SolverCase:
    """A case in neutral terms: the physics, the geometry, the material, the mesh, the boundary conditions and the
    loads -- nothing about a backend's file layout. ``prepare`` turns it into whatever the backend runs.

    ``parameters`` carries what only some backends read (solver name, commands, edits, end time); ``source`` points at
    an existing case directory or deck when the neutral description is a pointer (physics :data:`OPAQUE`).
    """

    physics: str = LINEAR_ELASTICITY
    geometry: dict[str, float] = field(default_factory=dict)
    material: dict[str, float] = field(default_factory=dict)
    mesh: dict[str, int] = field(default_factory=dict)
    boundary_conditions: list[dict[str, Any]] = field(default_factory=list)
    loads: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    source: str | None = None
    dimension: int = 3

    def __post_init__(self):
        if self.physics not in _PHYSICS:
            raise ValueError(f"unknown physics {self.physics!r}; known: {', '.join(_PHYSICS)}")

    @classmethod
    def cantilever(cls, *, L: float = 1.0, b: float = 0.1, h: float = 0.1, E: float = 210e9, nu: float = 0.3,
                   P: float = 1000.0, nx: int = 20, ny: int = 2, nz: int = 2) -> SolverCase:
        """A box cantilever clamped at ``x = 0`` and pulled down (``-z``) on the ``x = L`` face: the case a structural
        FEM backend and CalculiX both take, so the two results are numbers of the same physical quantity."""
        return cls(physics=LINEAR_ELASTICITY, dimension=3, geometry={"L": L, "b": b, "h": h},
                   material={"E": E, "nu": nu}, mesh={"nx": nx, "ny": ny, "nz": nz},
                   boundary_conditions=[{"type": "clamp", "side": "x-", "dofs": [0, 1, 2]}],
                   loads=[{"type": "force", "side": "x+", "value": [0.0, 0.0, -P]}],
                   outputs=("U", "S"))

    @classmethod
    def plane_cantilever(cls, *, L: float = 1.0, h: float = 0.1, b: float = 0.1, E: float = 210e9, nu: float = 0.3,
                         q: float = 100.0, nx: int = 40, ny: int = 8) -> SolverCase:
        """The same cantilever as a 2-D plane-stress rectangle ``L x h`` of thickness ``b`` under a uniform downward
        load ``q`` (N/m along the beam, so the tip deflection is ``q L^4 / (8 E I)`` with ``I = b h^3 / 12``).

        The uniform load is the one a plane finite-element form carries without a traction boundary term: it becomes
        the body force ``f = q / h`` (the load spread over the depth of the beam).
        """
        return cls(physics=LINEAR_ELASTICITY, dimension=2, geometry={"L": L, "h": h, "b": b},
                   material={"E": E, "nu": nu}, mesh={"nx": nx, "ny": ny},
                   boundary_conditions=[{"type": "clamp", "side": "x-", "dofs": [0, 1]}],
                   loads=[{"type": "uniform_load", "direction": [0.0, -1.0], "value": q}],
                   outputs=("U",))

    @classmethod
    def conduction(cls, *, L: float = 1.0, H: float = 1.0, k: float = 1.0, source: float = 0.0,
                   T_left: float = 0.0, T_right: float = 1.0, nx: int = 16, ny: int = 16) -> SolverCase:
        """Steady conduction through a rectangle: ``-div(k grad T) = source`` with the vertical sides held at
        ``T_left`` and ``T_right``.

        The one-dimensional answer is a polynomial, so a backend's field can be measured against a closed form with
        :func:`conduction_profile` rather than against another solver -- with ``source = 0`` a linear finite-element
        space reproduces it exactly, and with a source the error is the discretisation's own, which is what a mesh
        study should show falling.
        """
        return cls(physics=HEAT_TRANSFER, dimension=2, geometry={"L": L, "H": H}, material={"k": k},
                   mesh={"nx": nx, "ny": ny},
                   boundary_conditions=[{"type": "dirichlet", "side": "x-", "value": T_left},
                                        {"type": "dirichlet", "side": "x+", "value": T_right}],
                   parameters={"source": source}, outputs=("T",))

    @classmethod
    def case_directory(cls, path: str | Path, *, physics: str = OPAQUE, **parameters: Any) -> SolverCase:
        """A case that already exists on disk (a case directory, a deck, a script): the backend stages it its own way."""
        return cls(physics=physics, source=str(path), parameters=dict(parameters))

    def to_json(self) -> dict[str, Any]:
        return {"physics": self.physics, "dimension": self.dimension, "geometry": self.geometry,
                "material": self.material, "mesh": self.mesh, "boundary_conditions": self.boundary_conditions,
                "loads": self.loads, "parameters": self.parameters, "outputs": list(self.outputs),
                "source": self.source}


def conduction_profile(case: SolverCase, x: Any) -> Any:
    """The closed-form temperature of a :meth:`SolverCase.conduction` case at ``x``.

    ``T(x) = T0 + (T1 - T0) x / L + source x (L - x) / (2 k)``: the solution of ``-k T'' = source`` with ``T(0) = T0``
    and ``T(L) = T1``, which is what a backend's field is measured against. Raises :class:`CaseUnsupported` when the
    case is not a conduction case, so a call that cannot have a closed form says so instead of returning numbers.
    """
    if case.physics != HEAT_TRANSFER or "L" not in case.geometry:
        raise CaseUnsupported(f"a closed-form profile is defined for a conduction case, not {case.physics!r}")
    import numpy as np

    length = float(case.geometry["L"])
    left = float(next((bc.get("value", 0.0) for bc in case.boundary_conditions if bc.get("side") == "x-"), 0.0))
    right = float(next((bc.get("value", 1.0) for bc in case.boundary_conditions if bc.get("side") == "x+"), 1.0))
    k = float(case.material.get("k", 1.0))
    source = float(case.parameters.get("source", 0.0))
    x = np.asarray(x, dtype=float)
    return left + (right - left) * x / length + source * x * (length - x) / (2.0 * k)


# ---------------------------------------------------------------------------------------------- what the environment says
#: One of the five things a solver can lack.
EXECUTABLE = "executable"
VERSION = "version"
LICENCE = "licence"
ENVIRONMENT = "environment"
MODULE = "module"


@dataclass(frozen=True)
class Requirement:
    """One thing a solver needs before it can run, and what this machine found.

    ``ok`` is the answer; ``detail`` is the evidence (the path found, or what is missing and why); ``fix`` is what the
    reader does about it. ``verified=False`` marks a check that cannot be made without starting the solver (a licence
    the launcher alone can validate): it never counts as missing, and ``explain`` prints it with a ``?`` so nobody
    reads it as a pass.
    """

    kind: str
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    verified: bool = True

    def line(self) -> str:
        mark = "ok" if self.ok and self.verified else ("?" if self.ok else "missing")
        out = f"[{mark}] {self.name}"
        if self.detail:
            out += f": {self.detail}"
        if not self.ok and self.fix:
            out += f" (fix: {self.fix})"
        return out


@dataclass(frozen=True)
class EnvironmentReport:
    """What :meth:`ExternalSolver.check_environment` returns: never an exception, always an answer with its evidence."""

    solver: str
    requirements: tuple[Requirement, ...] = ()

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.requirements)

    @property
    def missing(self) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if not r.ok)

    @property
    def version(self) -> str | None:
        """The version the probe actually read -- a placeholder ("not read ...") is unverified and gives None, so it
        never ends up in provenance as if it were a version."""
        return next((r.detail for r in self.requirements if r.kind == VERSION and r.ok and r.verified and r.detail),
                    None)

    @property
    def executable(self) -> str | None:
        return next((r.detail for r in self.requirements if r.kind == EXECUTABLE and r.ok and r.detail), None)

    def why(self) -> str:
        """One line naming every missing item with its fix; empty when nothing is missing."""
        return "; ".join(f"{r.name}: {r.detail or 'missing'}" + (f" (fix: {r.fix})" if r.fix else "")
                         for r in self.missing)

    def explain(self) -> str:
        """The whole verdict, one requirement per line -- the diagnosis a caller shows instead of a traceback."""
        head = f"{self.solver}: ready" if self.ok else f"{self.solver}: {len(self.missing)} of " \
            f"{len(self.requirements)} checks missing"
        return "\n".join([head] + [r.line() for r in self.requirements])

    def raise_if_missing(self) -> EnvironmentReport:
        """Raise :class:`SolverUnavailable` with :meth:`explain` as the message; return ``self`` when ready."""
        if not self.ok:
            raise SolverUnavailable(self.solver, self)
        return self

    def to_json(self) -> dict[str, Any]:
        return {"solver": self.solver, "ok": self.ok,
                "requirements": [{"kind": r.kind, "name": r.name, "ok": r.ok, "detail": r.detail, "fix": r.fix,
                                  "verified": r.verified} for r in self.requirements]}


# ---------------------------------------------------------------------------------------------- what a run says
class SolverState(str, Enum):
    """The terminal states are :attr:`CONVERGED`, :attr:`DIVERGED`, :attr:`FAILED` and :attr:`CANCELLED`; the other two
    are what a live run is."""

    QUEUED = "queued"
    RUNNING = "running"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (SolverState.CONVERGED, SolverState.DIVERGED, SolverState.FAILED, SolverState.CANCELLED)

    @property
    def ok(self) -> bool:
        return self is SolverState.CONVERGED

    def __str__(self) -> str:                 # ``print(status.state)`` reads "converged", not "SolverState.CONVERGED"
        return self.value


@dataclass(frozen=True)
class SolverStatus:
    """Where a run stands, with the evidence that says so: ``evidence`` holds the log lines (a divergence marker, the
    solver's end marker, the fatal error) and ``reason`` the one sentence a person needs."""

    state: SolverState
    backend: str = ""
    version: str | None = None
    returncode: int | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    reason: str = ""
    elapsed_s: float | None = None

    @property
    def terminal(self) -> bool:
        return self.state.terminal

    @property
    def ok(self) -> bool:
        return self.state.ok

    def explain(self) -> str:
        bits = [f"{self.backend or 'solver'}: {self.state}"]
        if self.version:
            bits.append(f"version {self.version}")
        if self.returncode is not None:
            bits.append(f"exit {self.returncode}")
        if self.elapsed_s is not None:
            bits.append(f"{self.elapsed_s:.1f} s")
        if self.reason:
            bits.append(self.reason)
        out = ", ".join(bits)
        if self.progress:
            out += "\n" + ", ".join(f"{k}={v}" for k, v in self.progress.items())
        for line in self.evidence:
            out += f"\n  | {line}"
        return out

    def to_json(self) -> dict[str, Any]:
        return {"state": str(self.state), "backend": self.backend, "version": self.version,
                "returncode": self.returncode, "progress": self.progress, "evidence": list(self.evidence),
                "reason": self.reason, "elapsed_s": self.elapsed_s}


@dataclass
class SolverResults:
    """Fields and scalars of one converged run, in one schema whoever produced them.

    ``fields`` maps a name to an array whose first axis is the points of the domain; ``scalars`` holds the numbers a
    person reads off (tip displacement, last time, cell count); ``provenance`` names the backend and its version, which
    is what makes two backends' answers comparable rather than merely similar. :meth:`to_upd` hands it to the rest of
    the library as a ``PhysicalSample``.
    """

    backend: str
    version: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    scalars: dict[str, float | int] = field(default_factory=dict)
    coords: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_upd(self):
        """The neutral scientific schema of the library (``pinneapple_data.physical_sample.PhysicalSample``)."""
        import numpy as np
        import torch

        from pinneapple_data.physical_sample import PhysicalSample

        state = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in self.fields.items()}
        return PhysicalSample(state=state,
                              domain={"type": "grid", "coords": {k: np.asarray(v) for k, v in self.coords.items()}},
                              provenance={"version": "0.1", **self.provenance, "source": self.backend,
                                          "backend": self.backend, "backend_version": self.version},
                              schema={"units": dict(self.units), "scalars": dict(self.scalars)})

    def summary(self) -> dict[str, Any]:
        """A JSON-able digest for a step's outputs: the scalars, the field names and shapes, the provenance."""
        return {"backend": self.backend, "version": self.version, "scalars": dict(self.scalars),
                "fields": {k: list(getattr(v, "shape", ())) for k, v in self.fields.items()},
                "provenance": dict(self.provenance)}

    def to_json(self) -> dict[str, Any]:
        return self.summary()


def _lines_matching(text: str, patterns: Sequence[str], limit: int = 3) -> list[str]:
    """The first ``limit`` log lines matching any of ``patterns`` (regex, compiled multiline), for the evidence.

    A pattern may span several lines (``^\\s*Job finished\\s*$`` matches across the blank lines before it), so the
    line reported is the first non-empty one inside the match, not the first line after its start.
    """
    out: list[str] = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.MULTILINE):
            chunk = text[m.start():m.end()][:2000]
            line = next((ln.strip()[:300] for ln in chunk.splitlines() if ln.strip()), "")
            if line and line not in out:
                out.append(line)
            if len(out) >= limit:
                return out
    return out


def classify_run(*, backend: str, log: str, returncode: int | None, converged: Sequence[str] = (),
                 divergence: Sequence[str] = (), artefacts: Mapping[str, bool] | None = None,
                 elapsed_s: float | None = None, progress: dict[str, Any] | None = None,
                 version: str | None = None) -> SolverStatus:
    """The one definition of "finished", in the order the answer is decided:

    1. a divergence marker in the log -> :attr:`SolverState.DIVERGED`, with those lines as the evidence (checked before
       the exit code: a diverged run usually exits non-zero and must not read as a plain failure);
    2. a non-zero exit code -> :attr:`SolverState.FAILED`, with the last lines of the log;
    3. an artefact the solver had to write is missing -> :attr:`SolverState.FAILED`, naming it;
    4. the solver's own end marker in the log -> :attr:`SolverState.CONVERGED`, with the marker as the evidence;
    5. a backend that declares no end marker at all (a MATLAB script, which has none of its own) ->
       :attr:`SolverState.CONVERGED` on exit 0 with its artefact written, and what it wrote as the evidence;
    6. otherwise -> :attr:`SolverState.FAILED`: the run ended and nothing says it finished, which is a failure worth
       reading rather than a success assumed.
    """
    diverged = _lines_matching(log, divergence)
    if diverged:
        return SolverStatus(SolverState.DIVERGED, backend, version, returncode, dict(progress or {}),
                            tuple(diverged), "the solver reported a divergence", elapsed_s)
    if returncode not in (0, None):
        tail = [ln.strip()[:300] for ln in log.strip().splitlines()[-3:] if ln.strip()]
        return SolverStatus(SolverState.FAILED, backend, version, returncode, dict(progress or {}), tuple(tail),
                            f"the solver exited with code {returncode}", elapsed_s)
    missing = [name for name, exists in (artefacts or {}).items() if not exists]
    if missing:
        return SolverStatus(SolverState.FAILED, backend, version, returncode, dict(progress or {}),
                            (), f"the solver wrote no {', '.join(missing)}", elapsed_s)
    ended = _lines_matching(log, converged, limit=1)
    if ended:
        return SolverStatus(SolverState.CONVERGED, backend, version, returncode, dict(progress or {}),
                            tuple(ended), "the solver reached its own end marker", elapsed_s)
    if not converged:
        wrote = ", ".join(artefacts or {}) or "nothing to read back"
        return SolverStatus(SolverState.CONVERGED, backend, version, returncode, dict(progress or {}),
                            (f"exit 0, wrote {wrote}",), "the run ended cleanly and wrote its result", elapsed_s)
    tail = [ln.strip()[:300] for ln in log.strip().splitlines()[-3:] if ln.strip()]
    return SolverStatus(SolverState.FAILED, backend, version, returncode, dict(progress or {}), tuple(tail),
                        "the run ended without the solver's end marker", elapsed_s)


@dataclass
class RunHandle:
    """What :meth:`ExternalSolver.run` returns: the run's identity, its directory, its log and (when it is a process)
    the process itself, so ``status``/``cancel`` do not need the solver to be asked twice about the same run."""

    id: str
    backend: str
    workdir: Path
    argv: tuple[str, ...] = ()
    log_path: Path | None = None
    process: Any | None = None                 # subprocess.Popen when the backend spawns one
    started: float = field(default_factory=time.time)
    status: SolverStatus | None = None         # the terminal verdict, set once and then reused

    @classmethod
    def new(cls, backend: str, workdir: str | Path, *, argv: Sequence[str] = (),
            log_path: str | Path | None = None, process: Any | None = None,
            status: SolverStatus | None = None) -> RunHandle:
        return cls(id=uuid.uuid4().hex[:12], backend=backend, workdir=Path(workdir), argv=tuple(argv),
                   log_path=Path(log_path) if log_path else None, process=process, status=status)

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def poll(self) -> int | None:
        """The exit code, or None while the process runs (a finished in-process run keeps its recorded status)."""
        if self.process is not None:
            return self.process.poll()
        return self.status.returncode if self.status is not None else None

    def read_log(self, *, tail: int | None = None) -> str:
        if self.log_path is None or not self.log_path.is_file():
            return ""
        text = self.log_path.read_text(errors="replace")
        if tail and len(text.splitlines()) > tail:
            return "\n".join(text.splitlines()[-tail:])
        return text

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the process ends (or ``timeout``); returns the exit code, or None if it is still running."""
        if self.process is None:
            return self.poll()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        return self.process.returncode

    def cancel(self) -> bool:
        """Stop the process if it is still running; True when this call is what stopped it."""
        if not self.running:
            return False
        proc = self.process
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except OSError:
            return False
        return True

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "backend": self.backend, "workdir": str(self.workdir), "argv": list(self.argv),
                "log": str(self.log_path) if self.log_path else None, "elapsed_s": self.elapsed_s,
                "status": self.status.to_json() if self.status else None}


# ---------------------------------------------------------------------------------------------- the errors
class SolverError(RuntimeError):
    """Base of everything this module raises deliberately; ``diagnosis`` is the readable text for the caller."""

    def __init__(self, diagnosis: str):
        super().__init__(diagnosis)
        self.diagnosis = diagnosis


class SolverUnavailable(SolverError):
    """The environment cannot run the solver: carries the :class:`EnvironmentReport` that says what is missing."""

    def __init__(self, solver: str, report: EnvironmentReport):
        super().__init__(report.explain())
        self.solver = solver
        self.report = report


class CaseUnsupported(SolverError):
    """The backend cannot express this neutral case (a physics, a dimension or a geometry it does not implement)."""


class RunUnusable(SolverError):
    """``results()`` was asked for a run that did not converge; carries the :class:`SolverStatus` that says why."""

    def __init__(self, status: SolverStatus):
        super().__init__(status.explain())
        self.status = status


# ---------------------------------------------------------------------------------------------- the interface
class ExternalSolver(ABC):
    """One external solver behind one contract. Subclasses fill in the fives hooks and set :attr:`name`.

    ``check_environment``, ``status`` and ``cancel`` are implemented here once, so every backend answers the same way;
    a backend only decides *what it needs* (``_requirements``), *how the case becomes input* (``prepare``), *how it
    starts* (``_start``), *what it reads back* (``_results``), and *which log lines mean converged or diverged*.
    """

    #: What provenance calls this backend (also the registry key unless :attr:`kind` says otherwise).
    name: str = ""
    kind: str = ""

    #: Log markers of the one definition of "finished" (:func:`classify_run`), as regexes.
    converged_markers: Sequence[str] = ()
    divergence_markers: Sequence[str] = ()

    def __init__(self, workdir: str | Path | None = None, *, log_name: str = "solver.log"):
        self.workdir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix=f"pinneapple_{self.key}_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.workdir / log_name
        self.case: SolverCase | None = None
        self.handle: RunHandle | None = None
        self._environment: EnvironmentReport | None = None
        self._log_file: Any = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, workdir={str(self.workdir)!r})"

    @property
    def key(self) -> str:
        """The registry key: :attr:`kind` when set, else :attr:`name`."""
        return self.kind or self.name

    # -- environment ---------------------------------------------------------------------------
    def check_environment(self) -> EnvironmentReport:
        """What this machine lacks for the solver to run (executable, version, licence, environment, module).

        Never raises: a probe that blows up -- a missing binary, a timeout, a launcher that cannot even be asked --
        becomes a missing requirement whose ``detail`` is that error, which is exactly the diagnosis a caller needs
        instead of an exception thrown halfway through a run.
        """
        if self._environment is not None:
            return self._environment
        try:
            requirements = tuple(self._requirements())
        except Exception as exc:                      # noqa: BLE001 - the contract says this never raises
            requirements = (Requirement(ENVIRONMENT, self.name, False,
                                        f"the environment check itself failed: {type(exc).__name__}: {exc}",
                                        "report it with the solver's version, the check should not fail"),)
        self._environment = EnvironmentReport(self.name, requirements)
        return self._environment

    @abstractmethod
    def _requirements(self) -> Sequence[Requirement]:
        """Every requirement, each with its evidence. Probes that can fail go through :meth:`_probe`."""

    def _probe(self, kind: str, name: str, fn: Callable[[], Any], *, verified: bool = True,
               fix: str = "") -> Requirement:
        """Run one check, turning anything it raises into a missing requirement (``detail`` = the error).

        ``fn`` either returns ``(ok, detail, fix)`` or a ready :class:`Requirement` (for a check that knows whether
        what it found was actually verified -- a version that could not be read, for instance).
        """
        try:
            out = fn()
        except Exception as exc:                      # noqa: BLE001 - a probe never takes the report down with it
            return Requirement(kind, name, False, f"{type(exc).__name__}: {exc}", fix or "check the installation",
                               verified)
        if isinstance(out, Requirement):
            return out
        ok, detail, why = out
        return Requirement(kind, name, ok, detail, why or fix, verified)

    def version(self) -> str | None:
        """The version the environment probe found, for provenance. This asks the machine (once)."""
        return self.check_environment().version

    def _known_version(self) -> str | None:
        """The version already learned from the environment, without launching a probe: what ``status()`` may use, so
        that asking a run where it stands never starts a subprocess of its own."""
        return self._environment.version if self._environment is not None else None

    # -- case ----------------------------------------------------------------------------------
    @abstractmethod
    def prepare(self, case: SolverCase) -> Path:
        """Materialise ``case`` in :attr:`workdir` the way this backend runs it; returns the run directory."""

    def _require_case_supported(self, case: SolverCase, *, physics: Sequence[str], dimension: int | None = None) -> None:
        if case.physics not in physics:
            raise CaseUnsupported(f"{self.name} runs {', '.join(physics)}, not {case.physics!r}")
        if dimension is not None and case.dimension != dimension:
            raise CaseUnsupported(f"{self.name} takes a {dimension}-D case, this one is {case.dimension}-D")

    # -- run -----------------------------------------------------------------------------------
    def run(self, *, timeout: float | None = None) -> RunHandle:
        """Start the prepared case; returns the handle ``status``/``cancel``/``results`` speak about."""
        if self.case is None:
            raise SolverError(f"{self.name}: prepare() a case before run()")
        self.handle = self._start(timeout)
        return self.handle

    @abstractmethod
    def _start(self, timeout: float | None) -> RunHandle:
        """Launch the solver and return its handle (a backend that solves in-process returns a finished one)."""

    def _spawn(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None,
               cwd: str | Path | None = None) -> subprocess.Popen:
        """Start ``argv`` with its output going to :attr:`log_path`, in its own session so ``cancel`` can end the whole
        process group (a launcher such as ``bash`` or ``mpirun`` leaves children behind otherwise)."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w")     # noqa: SIM115 - closed when the run ends, see _close_log
        full_env = dict(os.environ)
        full_env.update(env or {})
        return subprocess.Popen([str(a) for a in argv], cwd=str(cwd or self.workdir), stdout=self._log_file,
                                stderr=subprocess.STDOUT, env=full_env, start_new_session=True)

    def _close_log(self) -> None:
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.close()

    # -- state ---------------------------------------------------------------------------------
    def status(self) -> SolverStatus:
        """Where the run stands. ``queued`` before ``run()``, ``running`` while the process lives, then the verdict of
        :func:`classify_run` -- cached on the handle, so asking twice gives the same answer."""
        h = self.handle
        if h is None:
            return SolverStatus(SolverState.QUEUED, self.name, self._known_version(),
                                reason="nothing has been started")
        if h.status is not None:
            return h.status
        returncode = h.poll()
        if returncode is None and h.process is not None:
            return SolverStatus(SolverState.RUNNING, self.name, self._known_version(),
                                progress=self._progress(h.read_log()))
        h.status = self._classify(h, returncode)
        self._close_log()
        return h.status

    def wait(self, timeout: float | None = None) -> SolverStatus:
        """Block until the run ends (or ``timeout``) and return its status."""
        if self.handle is not None and self.handle.process is not None:
            self.handle.wait(timeout=timeout)
        return self.status()

    def _progress(self, log: str) -> dict[str, Any]:
        """What a live log says about progress; backends with a time or an iteration count override it."""
        return {}

    @abstractmethod
    def _classify(self, handle: RunHandle, returncode: int | None) -> SolverStatus:
        """The backend's verdict: call :func:`classify_run` with its markers and the artefacts it expects."""

    # -- results -------------------------------------------------------------------------------
    def results(self) -> SolverResults:
        """Fields and scalars of the run, in the one schema; raises :class:`RunUnusable` unless it converged."""
        status = self.status()
        if not status.terminal:
            raise RunUnusable(self.wait())
        if not status.ok:
            raise RunUnusable(status)
        return self._results()

    @abstractmethod
    def _results(self) -> SolverResults:
        """Read the converged run back."""

    # -- cancel --------------------------------------------------------------------------------
    def cancel(self) -> SolverStatus:
        """Stop the run; idempotent -- a second call (or a call after the run ended) reports the same state."""
        h = self.handle
        if h is None:
            return SolverStatus(SolverState.CANCELLED, self.name, self._known_version(),
                                reason="nothing was running")
        if h.status is not None and h.status.terminal:
            return h.status
        if h.cancel():
            h.status = SolverStatus(SolverState.CANCELLED, self.name, self._known_version(),
                                    reason="cancelled by the caller",
                                    evidence=(f"the process was stopped after {h.elapsed_s:.1f} s",),
                                    elapsed_s=h.elapsed_s)
            self._close_log()
            return h.status
        return self.status()                           # it ended on its own first: report the real verdict


# ---------------------------------------------------------------------------------------------- the registry
#: registry key -> factory. The factories live in :mod:`.backends`; import that module to fill this in.
SOLVERS: dict[str, Callable[..., ExternalSolver]] = {}


def register_solver(factory: Callable[..., ExternalSolver], *, key: str | None = None,
                    replace: bool = False) -> Callable[..., ExternalSolver]:
    """Make ``factory(**options)`` reachable as :func:`get_solver` under ``key`` (its class's ``kind`` by default)."""
    if not (isinstance(factory, type) and issubclass(factory, ExternalSolver)):
        raise TypeError("a solver factory must be an ExternalSolver subclass")
    name = key or factory.kind or factory.name
    if not name:
        raise TypeError("a solver needs a name or a kind to be registered under")
    if name in SOLVERS and not replace:
        raise ValueError(f"solver {name!r} is already registered")
    SOLVERS[name] = factory
    return factory


def solver_names() -> list[str]:
    """Every registered backend, by key."""
    return sorted(SOLVERS)


def get_solver(name: str, **options: Any) -> ExternalSolver:
    """An instance of the backend ``name`` (``openfoam``, ``calculix``, ``fenicsx``, ``ansys-fluent``, ``matlab``,
    ``solid-fem``); it does not check the environment -- :meth:`ExternalSolver.check_environment` is that question, and
    it answers instead of raising."""
    if name not in SOLVERS:
        raise KeyError(f"unknown solver {name!r}; registered: {solver_names()}")
    return SOLVERS[name](**options)


def check_all(**options: Any) -> dict[str, EnvironmentReport]:
    """``check_environment()`` for every registered backend: what this machine can run, and what it cannot."""
    return {name: get_solver(name, **options).check_environment() for name in solver_names()}
