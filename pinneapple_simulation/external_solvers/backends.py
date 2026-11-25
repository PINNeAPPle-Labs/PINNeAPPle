"""The solver implementations this repository already had, behind the one contract of :mod:`.contract`.

Nothing is reimplemented here. Each backend calls the module it wraps -- ``openfoam.campaign``/``openfoam.field_reader``,
``calculix.inp``/``frd``/``runner``, ``ansys.runner``/``journal_builder``, ``numerical_solvers.fenics_bridge``,
``matlab.runner`` -- and adds only the four things the contract asks of it: what it needs (:meth:`_requirements`),
how a neutral :class:`~.contract.SolverCase` becomes its input (``prepare``), how it starts (``_start``), and which
log lines mean converged or diverged (``converged_markers`` / ``divergence_markers`` fed to
:func:`~.contract.classify_run`).

Two of the six are the pair the cross-backend test runs: :class:`CalculiXBackend` (the ``ccx`` binary or the bundled
Docker image) and :class:`SolidFEMBackend` (the in-process hexahedral elasticity solver of
``numerical_solvers.solid_fem``, which needs no executable and no licence). They take the same
:meth:`~.contract.SolverCase.cantilever`, and their displacement fields arrive in the same schema with provenance
naming each backend and its version -- which is what makes two solvers comparable rather than merely both installed.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .contract import (
    ENVIRONMENT,
    EXECUTABLE,
    HEAT_TRANSFER,
    INCOMPRESSIBLE_FLOW,
    LICENCE,
    LINEAR_ELASTICITY,
    MODULE,
    OPAQUE,
    VERSION,
    CaseUnsupported,
    ExternalSolver,
    Requirement,
    RunHandle,
    SolverCase,
    SolverError,
    SolverResults,
    SolverState,
    SolverStatus,
    classify_run,
    register_solver,
)

__all__ = ["BackendMissing", "OpenFOAMBackend", "CalculiXBackend", "FEniCSxBackend", "AnsysFluentBackend",
           "MatlabBackend", "SolidFEMBackend", "BACKENDS", "BACKEND_CLASSES"]


class BackendMissing(SolverError):
    """A backend module is missing although its own check said it was there (an extra is required)."""


def _module_requirement(name: str, *, fix: str, kind: str = MODULE) -> Requirement:
    """``import name`` as a requirement: present with its version, or missing with the reason."""
    try:
        module = import_module(name)
    except Exception as exc:                          # noqa: BLE001 - a probe answer, not a failure
        return Requirement(kind, name, False, f"{type(exc).__name__}: {exc}", fix)
    return Requirement(kind, name, True, str(getattr(module, "__version__", "installed")), "")


def _first_line(text: str, default: str = "") -> str:
    return next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), default)


# ---------------------------------------------------------------------------------------------- OpenFOAM
class OpenFOAMBackend(ExternalSolver):
    """OpenFOAM: a case directory, the mesh command, the solver, the log.

    ``check_environment`` asks the environment itself rather than trusting ``PATH``: a distribution package puts
    ``blockMesh`` on the path but the binaries only run with the OpenFOAM environment sourced, so the probe sources
    the ``bashrc`` (:mod:`.openfoam.environment`) and reports what it provides. The one case it stages is
    :data:`~.contract.OPAQUE` (``SolverCase.case_directory``): the case files are OpenFOAM's own, and rewriting them
    from a neutral description would be a mesher, not a contract.
    """

    name = "openfoam"
    kind = "openfoam"
    converged_markers = (r"^End\s*$",)
    divergence_markers = (r"Min\([^)]*\) = (?:nan|[-+]?inf)", r"Max\([^)]*\) = (?:nan|[-+]?inf)",
                          r"FOAM FATAL[^\n]*(?:Floating point|divergen)", r"sigFpe", r"bounding \w+.*= nan")

    def __init__(self, workdir: str | Path | None = None, *, solver: str = "icoFoam", bashrc: str | None = None,
                 commands: Sequence[str] | None = None, edits: Sequence[dict] | None = None,
                 fields: Sequence[str] = ("p", "U"), case_subdir: str = "case",
                 patterns: Sequence[str] | None = None, environ: dict[str, str] | None = None,
                 log_name: str = "solver.log"):
        super().__init__(workdir, log_name=log_name)
        from .openfoam.environment import FOAM_BASHRCS, find_bashrc

        self.solver = solver
        self.edits = list(edits or ())
        self.commands = list(commands) if commands is not None else None
        self.fields = tuple(fields)
        self.case_subdir = case_subdir
        self.environ = environ
        self.bashrc = find_bashrc(bashrc, patterns=FOAM_BASHRCS if patterns is None else tuple(patterns),
                                  environ=environ)

    @property
    def case_dir(self) -> Path:
        return self.workdir / self.case_subdir

    def _requirements(self) -> Sequence[Requirement]:
        from .openfoam.environment import probe

        environ = self.environ if self.environ is not None else dict(os.environ)
        executables = tuple(dict.fromkeys(("blockMesh", self.solver)))
        loaded = probe(self.bashrc, executables, environ=environ) if self.bashrc else None
        found = (loaded or {}).get("executables", {})

        def solver_paths() -> tuple[bool, str, str]:
            if self.bashrc and loaded is not None and not loaded["sourced"]:
                return (False, f"the bashrc {self.bashrc} did not load: {loaded['error']}",
                        "source it by hand and fix what it reports, or point FOAM_BASHRC at a working one")
            missing = [e for e in executables if not found.get(e) and not shutil.which(e)]
            if missing:
                return (False, f"OpenFOAM not found ({', '.join(missing)} not on PATH and no bashrc)",
                        "install OpenFOAM, or set FOAM_BASHRC (or the step's bashrc) to its etc/bashrc")
            detail = ", ".join(f"{e}={found.get(e) or shutil.which(e)}" for e in executables)
            return True, detail, ""

        def environment() -> tuple[bool, str, str]:
            if self.bashrc or environ.get("WM_PROJECT_DIR"):
                return True, (f"bashrc {self.bashrc}" if self.bashrc
                              else f"WM_PROJECT_DIR={environ['WM_PROJECT_DIR']}"), ""
            return (False, f"{self.solver} is on PATH but the OpenFOAM environment is not loaded and no bashrc was "
                           "found: source it, or set FOAM_BASHRC", "source the OpenFOAM bashrc before running")

        def version() -> Requirement:
            v = (loaded or {}).get("version") or environ.get("WM_PROJECT_VERSION")
            if not v and self.bashrc:                  # the release directory names its own version: OpenFOAM-v2606
                v = Path(self.bashrc).parent.parent.name
            if v:
                return Requirement(VERSION, "OpenFOAM version", True, str(v))
            return Requirement(VERSION, "OpenFOAM version", True, "not read (the environment is not loaded here)",
                               verified=False)

        return (self._probe(EXECUTABLE, "blockMesh and " + self.solver, solver_paths),
                self._probe(ENVIRONMENT, "OpenFOAM environment", environment),
                self._probe(VERSION, "OpenFOAM version", version),
                Requirement(LICENCE, "OpenFOAM licence", True, "none: OpenFOAM is free software (GPL)"))

    def prepare(self, case: SolverCase) -> Path:
        if case.physics not in (OPAQUE, INCOMPRESSIBLE_FLOW):
            raise CaseUnsupported(f"openfoam stages a case directory ({OPAQUE}) or an incompressible flow case "
                                  f"({INCOMPRESSIBLE_FLOW}), not {case.physics!r}")
        if case.source is None:
            raise CaseUnsupported("openfoam needs case.source: the directory holding the OpenFOAM case")
        src = Path(case.source)
        if not src.is_dir():
            raise CaseUnsupported(f"the OpenFOAM case {src} does not exist")
        from .openfoam.campaign import edit_file

        shutil.rmtree(self.case_dir, ignore_errors=True)
        shutil.copytree(src, self.case_dir)
        for e in case.parameters.get("edits", self.edits):
            edit_file(self.case_dir / e["file"], e["key"], str(e["value"]))
        self.solver = case.parameters.get("solver", self.solver)
        self.commands = list(case.parameters.get("commands", self.commands or ["blockMesh", self.solver]))
        self.case = case
        return self.workdir

    def _start(self, timeout: float | None) -> RunHandle:
        import shlex

        line = "cd " + shlex.quote(str(self.case_dir)) + " && " + " && ".join(
            f"{c} > log.{c.split()[0]} 2>&1" for c in self.commands)
        if self.bashrc:
            line = f"source {shlex.quote(self.bashrc)} >/dev/null 2>&1 && " + line
        argv = ["bash", "-c", line]
        process = self._spawn(argv)
        return RunHandle.new(self.name, self.workdir, argv=argv, log_path=self.log_path, process=process)

    def _solver_log(self) -> str:
        p = self.case_dir / f"log.{self.solver.split()[0]}"
        return p.read_text(errors="replace") if p.is_file() else ""

    def _progress(self, log: str) -> dict[str, Any]:
        from .openfoam.campaign import parse_solver_log

        health = parse_solver_log(self._solver_log())
        return {k: health[k] for k in ("last_time", "steps", "courant_max") if health.get(k) is not None}

    def _classify(self, handle: RunHandle, returncode: int | None) -> SolverStatus:
        from .openfoam.campaign import parse_solver_log

        solver_log = self._solver_log()
        health = parse_solver_log(solver_log) if solver_log else {}
        times = sorted((d.name for d in self.case_dir.glob("*") if d.is_dir() and _is_number(d.name)), key=float)
        return classify_run(backend=self.name, log=handle.read_log() + "\n" + solver_log, returncode=returncode,
                            converged=self.converged_markers, divergence=self.divergence_markers,
                            artefacts={f"log.{self.solver.split()[0]}": bool(solver_log),
                                       "a time directory": bool(times)},
                            elapsed_s=handle.elapsed_s,
                            progress={k: health[k] for k in ("last_time", "steps", "courant_max")
                                      if health.get(k) is not None},
                            version=self.version())

    def _results(self) -> SolverResults:
        import numpy as np

        from .openfoam.campaign import parse_solver_log
        from .openfoam.field_reader import openfoam_case_to_upd

        health = parse_solver_log(self._solver_log())
        sample = openfoam_case_to_upd(str(self.case_dir), fields=self.fields)
        fields = {k: np.asarray(v) for k, v in dict(sample.state).items() if not k.startswith("_")}
        return SolverResults(backend=self.name, version=self.version(), fields=fields,
                             scalars={k: health[k] for k in ("last_time", "steps") if health.get(k) is not None}
                             | ({"courant_max": health["courant_max"]} if health.get("courant_max") else {}),
                             coords={k: np.asarray(v) for k, v in (sample.domain.get("coords") or {}).items()},
                             provenance={"physics_domain": "cfd", "case_dir": str(self.case_dir),
                                         "solver": self.solver, "time_dir": sample.provenance.get("time_dir"),
                                         "status": "converged"})


# ---------------------------------------------------------------------------------------------- CalculiX
class CalculiXBackend(ExternalSolver):
    """Implicit structural FEM through ``ccx``: the native binary when it is on ``PATH``, else the bundled Debian image.

    The deck comes from a neutral :meth:`~.contract.SolverCase.cantilever` through ``calculix.inp``; the answer comes
    back from ``calculix.frd``. A run whose ``.frd`` holds non-finite displacements is reported ``diverged`` even when
    the log looks clean -- a NaN field is what a diverged structural solve leaves behind, and calling it converged
    because the process exited 0 is how a wrong number reaches a report.
    """

    name = "calculix"
    kind = "calculix"
    converged_markers = (r"^\s*Job finished\s*$", r"^\s*\*END\b")
    divergence_markers = (r"\*ERROR[^\n]*(?:no convergence|does not converge|divergen)",
                          r"increment size[^\n]*(?:smaller than|below) (?:the )?minimum",
                          r"\bno convergence\b", r"^\s*nan\s*$")

    def __init__(self, workdir: str | Path | None = None, *, job: str = "case", threads: int = 1,
                 backend: str | None = None, image: str | None = None, platform: str | None = None,
                 log_name: str = "solver.log"):
        super().__init__(workdir, log_name=log_name)
        self.job = job
        self.threads = threads
        self.backend_kind = backend
        self.image = image
        self.platform = platform

    @property
    def frd(self) -> Path:
        return self.workdir / f"{self.job}.frd"

    def _requirements(self) -> Sequence[Requirement]:
        from .calculix.runner import IMAGE, ccx_backend

        def ccx() -> tuple[bool, str, str]:
            kind = self.backend_kind or ccx_backend()
            if kind == "native":
                return True, str(shutil.which("ccx")), ""
            if kind == "docker":
                return True, f"docker (the bundled image {IMAGE})", ""
            return (False, "CalculiX (ccx) not on PATH and Docker is not available",
                    "install CalculiX (apt install calculix-ccx, or conda install -c conda-forge calculix) or start "
                    "Docker")

        def version() -> Requirement:
            """Informational: a ``ccx`` that runs but will not say its version is still a usable solver, so an
            unreadable version never counts as missing -- it is reported unverified, and gives no provenance version."""
            kind = self.backend_kind or ccx_backend()
            if kind == "native":
                p = subprocess.run(["ccx", "-v"], capture_output=True, text=True, timeout=15)
                m = re.search(r"Version\s+([\w.\-]+)", p.stdout + p.stderr)
                if m:
                    return Requirement(VERSION, "CalculiX version", True, m.group(1))
                return Requirement(VERSION, "CalculiX version", True,
                                   f"not read (ccx -v printed {_first_line(p.stdout + p.stderr, 'nothing')!r})",
                                   verified=False)
            if kind == "docker":
                from .calculix.runner import _docker_env, image_version

                tag = self.image or IMAGE
                inspected = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, env=_docker_env())
                if inspected.returncode != 0:
                    return Requirement(VERSION, "CalculiX version", True, f"{tag} (built on the first run)",
                                       verified=False)
                v = image_version(tag, self.platform)
                return Requirement(VERSION, "CalculiX version", True, v or f"{tag} (version unread)",
                                   verified=bool(v))
            return Requirement(VERSION, "CalculiX version", True, "not read (no CalculiX to ask)", verified=False)

        return (self._probe(EXECUTABLE, "ccx", ccx),
                self._probe(VERSION, "CalculiX version", version),
                Requirement(LICENCE, "CalculiX licence", True,
                            "none: CalculiX is free software (GPL); the bundled image carries Debian's calculix-ccx"))

    def prepare(self, case: SolverCase) -> Path:
        self._require_case_supported(case, physics=(LINEAR_ELASTICITY,), dimension=3)
        from .calculix.inp import cantilever_hex_mesh, write_inp

        g, m = case.geometry, case.mesh
        element = case.parameters.get("element_type", "C3D20R")
        model = cantilever_hex_mesh(g["L"], g["b"], g["h"], nx=m.get("nx", 20), ny=m.get("ny", 2),
                                    nz=m.get("nz", 2), element_type=element)
        model.E = float(case.material.get("E", 210e9))
        model.nu = float(case.material.get("nu", 0.3))
        model.boundary = []
        for bc in case.boundary_conditions:
            if bc.get("type") != "clamp":
                raise CaseUnsupported(f"calculix takes a 'clamp' boundary condition, not {bc.get('type')!r}")
            dofs = bc.get("dofs", (0, 1, 2))
            model.boundary.append(("FIXED", min(dofs) + 1, max(dofs) + 1, 0.0))
        model.cloads = []
        for load in case.loads:
            if load.get("type") != "force":
                raise CaseUnsupported(f"calculix takes a nodal 'force' load, not {load.get('type')!r}")
            value = list(load["value"])
            tip = model.nsets["TIP"]
            for dof, component in enumerate(value, start=1):
                if component:
                    model.cloads += [(str(n), dof, component / len(tip)) for n in tip]
        self.job = case.parameters.get("job", self.job)
        write_inp(model, str(self.workdir / f"{self.job}.inp"))
        self.case = case
        return self.workdir

    def _start(self, timeout: float | None) -> RunHandle:
        from .calculix.runner import ccx_command

        argv = ccx_command(str(self.workdir), self.job, threads=self.threads, backend=self.backend_kind,
                           image=self.image, platform=self.platform)
        process = self._spawn(argv, env={"OMP_NUM_THREADS": str(self.threads)})
        return RunHandle.new(self.name, self.workdir, argv=argv, log_path=self.log_path, process=process)

    def _progress(self, log: str) -> dict[str, Any]:
        steps = re.findall(r"increment\s+(\d+)", log)
        return {"increment": int(steps[-1])} if steps else {}

    def _classify(self, handle: RunHandle, returncode: int | None) -> SolverStatus:
        status = classify_run(backend=self.name, log=handle.read_log(), returncode=returncode,
                              converged=self.converged_markers, divergence=self.divergence_markers,
                              artefacts={f"{self.job}.frd": self.frd.is_file()}, elapsed_s=handle.elapsed_s,
                              progress=self._progress(handle.read_log()), version=self._known_version())
        if not status.ok:
            return status
        nonfinite = self._nonfinite_displacement()
        if nonfinite:
            return SolverStatus(SolverState.DIVERGED, self.name, self._known_version(), returncode, {},
                                (nonfinite,), "the result holds non-finite displacements", handle.elapsed_s)
        return status

    def _nonfinite_displacement(self) -> str | None:
        """The first non-finite value the .frd holds in a component that carries data, as evidence, or None.

        CalculiX declares components it does not write (``DISP``'s fourth, ``ALL``, is the resultant cgx computes),
        and the reader fills those with NaN; a column with no finite value at all is therefore skipped -- unless the
        whole block is like that, which is a result with nothing in it and is reported as diverged.
        """
        import numpy as np

        from .calculix.frd import read_frd

        try:
            results = read_frd(str(self.frd))
        except Exception as exc:                      # noqa: BLE001 - an unreadable .frd is a result to report
            return f"the .frd could not be read: {type(exc).__name__}: {exc}"
        for step in results.steps:
            for name, values in step.items():
                arr = np.asarray(values, dtype=float)
                if not arr.size:
                    continue
                arr2 = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
                written = np.isfinite(arr2).any(axis=0)
                if written.any():
                    bad = int((~np.isfinite(arr2[:, written])).sum())
                    if bad:
                        return f"{name} holds {bad} non-finite values"
                elif not np.isfinite(arr2).any():
                    return f"{name} holds no finite value at all"
        return None

    def _results(self) -> SolverResults:
        import numpy as np

        from .calculix.frd import read_frd

        results = read_frd(str(self.frd))
        disp = results.field("DISP") if any("DISP" in s for s in results.steps) else None
        fields: dict[str, Any] = {}
        scalars: dict[str, float | int] = {"nodes": int(len(results.node_ids))}
        if disp is not None:
            fields["U"] = np.asarray(disp)
            scalars["max_displacement"] = float(np.linalg.norm(np.asarray(disp)[:, :3], axis=1).max())
        if any("STRESS" in s for s in results.steps):
            fields["S"] = np.asarray(results.field("STRESS"))
        coords = np.asarray(results.coords)
        return SolverResults(backend=self.name, version=self.version(), fields=fields, scalars=scalars,
                             coords={n: coords[:, i] for i, n in enumerate(("x", "y", "z")[:coords.shape[1]])},
                             units={"U": "m", "S": "Pa", "max_displacement": "m"},
                             provenance={"physics_domain": "structural", "job": self.job,
                                         "deck": str(self.workdir / f"{self.job}.inp"), "status": "converged"})


# ---------------------------------------------------------------------------------------------- in-process FEM
class SolidFEMBackend(ExternalSolver):
    """3-D linear elasticity in-process (``numerical_solvers.solid_fem``: 8-node hexahedra with incompatible modes, the
    element Abaqus calls C3D8I), with no executable and no licence to check.

    It is the second half of the cross-backend test: the same ``SolverCase.cantilever`` as :class:`CalculiXBackend`,
    with a different element, a different solver and the same schema out -- so the contract is shown to be a layer and
    not a renaming of one program.
    """

    name = "pinneapple-solid-fem"
    kind = "solid-fem"
    converged_markers = (r"^solved: \d+ free degrees of freedom$",)
    divergence_markers = (r"^the solve raised\b", r"non-finite")

    def _requirements(self) -> Sequence[Requirement]:
        import numpy
        import scipy

        import pinneapple

        return (Requirement(MODULE, "numpy", True, numpy.__version__),
                Requirement(MODULE, "scipy", True, scipy.__version__),
                Requirement(VERSION, "PINNeAPPle version", True, getattr(pinneapple, "__version__", "installed")),
                Requirement(EXECUTABLE, "solver executable", True, "none: it solves in this process"),
                Requirement(LICENCE, "licence", True, "none: the element is our own implementation"))

    def prepare(self, case: SolverCase) -> Path:
        self._require_case_supported(case, physics=(LINEAR_ELASTICITY,), dimension=3)
        from ..numerical_solvers.solid_fem import box_mesh

        g, m = case.geometry, case.mesh
        # the same box as the CalculiX deck (y and z from 0, not centred), so both backends see one geometry
        self._mesh = box_mesh(g["L"], g["b"], g["h"], m.get("nx", 20), m.get("ny", 2), m.get("nz", 2),
                              origin=(0.0, g["b"] / 2, g["h"] / 2))
        self.case = case
        return self.workdir

    def _start(self, timeout: float | None) -> RunHandle:
        import numpy as np

        from ..numerical_solvers.solid_fem import SolidFEM

        case = self.case
        mesh = self._mesh
        try:
            fem = SolidFEM(mesh, E=float(case.material.get("E", 210e9)), nu=float(case.material.get("nu", 0.3)))
            for bc in case.boundary_conditions:
                if bc.get("type") != "clamp":
                    raise CaseUnsupported(f"the in-process backend takes a 'clamp', not {bc.get('type')!r}")
                side = {"x-": ("x", 0.0), "x+": ("x", case.geometry["L"])}[bc.get("side", "x-")]
                fem.fix(mesh.nodes_on(**{side[0]: side[1]}), dofs=tuple(bc.get("dofs", (0, 1, 2))))
            for load in case.loads:
                if load.get("type") != "force":
                    raise CaseUnsupported(f"the in-process backend takes a nodal 'force', not {load.get('type')!r}")
                fem.load_face(mesh.face_nodes("x+"), tuple(load["value"]))
            result = fem.solve()
        except Exception as exc:                      # noqa: BLE001 - it is the run's log that carries the failure
            self.log_path.write_text(f"the solve raised {type(exc).__name__}: {exc}\n")
            status = classify_run(backend=self.name, log=self.log_path.read_text(), returncode=1,
                                  converged=self.converged_markers, divergence=self.divergence_markers,
                                  version=self._known_version())
            return RunHandle.new(self.name, self.workdir, log_path=self.log_path, status=status)
        self._result = result
        free = int(3 * len(mesh.nodes) - len(fem._fixed))
        finite = bool(np.isfinite(result.u).all())
        self.log_path.write_text(
            f"solved: {free} free degrees of freedom\n"
            f"max displacement {result.max_displacement:.10g} m\n"
            + ("" if finite else "non-finite values in the displacement field\n"))
        status = classify_run(backend=self.name, log=self.log_path.read_text(), returncode=0,
                              converged=self.converged_markers, divergence=self.divergence_markers,
                              artefacts={"displacement field": bool(np.isfinite(result.u).all())},
                              elapsed_s=0.0, progress={"free_dofs": free}, version=self._known_version())
        return RunHandle.new(self.name, self.workdir, log_path=self.log_path, status=status)

    def _classify(self, handle: RunHandle, returncode: int | None) -> SolverStatus:
        return classify_run(backend=self.name, log=handle.read_log(), returncode=returncode,
                            converged=self.converged_markers, divergence=self.divergence_markers,
                            elapsed_s=handle.elapsed_s, version=self._known_version())

    def _results(self) -> SolverResults:
        import numpy as np

        res = self._result
        nodes = np.asarray(res.mesh.nodes)
        return SolverResults(backend=self.name, version=self.version(),
                             fields={"U": np.asarray(res.u), "von_mises": np.asarray(res.von_mises)},
                             scalars={"max_displacement": float(res.max_displacement),
                                      "nodes": int(len(nodes)), "elements": int(len(res.mesh.elements))},
                             coords={n: nodes[:, i] for i, n in enumerate(("x", "y", "z"))},
                             units={"U": "m", "von_mises": "Pa", "max_displacement": "m"},
                             provenance={"physics_domain": "structural", "element": "C3D8I equivalent",
                                         "status": "converged"})


# ---------------------------------------------------------------------------------------------- FEniCSx / DOLFINx
class FEniCSxBackend(ExternalSolver):
    """Plane elasticity through ``numerical_solvers.fenics_bridge`` (DOLFINx, or legacy FEniCS when that is what the
    machine has).

    The bridge forms plane stress/strain with a body force and Dirichlet boundaries -- no traction term -- so the
    neutral case it takes is :meth:`~.contract.SolverCase.plane_cantilever`, whose uniform load becomes the body force
    ``q / h``. Its labels come from the bridge's own supported kinds: anything else is
    :class:`~.contract.CaseUnsupported` rather than a guess.
    """

    name = "fenicsx"
    kind = "fenicsx"
    converged_markers = (r"^solved:",)
    divergence_markers = (r"^the solve raised\b", r"non-finite", r"DIVERGED_(?:MAX_IT|BREAKDOWN|DIVERGED_ITS)")

    _PDES = {(LINEAR_ELASTICITY, "plane_stress"): "linear_elasticity_plane_stress",
             (LINEAR_ELASTICITY, "plane_strain"): "linear_elasticity_plane_strain",
             (HEAT_TRANSFER, None): "heat_equation_steady"}

    def _requirements(self) -> Sequence[Requirement]:
        found = None
        for name in ("dolfinx", "dolfin"):
            req = _module_requirement(name, fix="conda install -c conda-forge fenics-dolfinx mpich pyvista "
                                                "(or build FEniCSx from source)")
            if req.ok:
                found = req
                break
        if found is None:
            return (Requirement(MODULE, "dolfinx (or legacy dolfin)", False,
                                "neither dolfinx nor dolfin is importable",
                                "conda install -c conda-forge fenics-dolfinx mpich pyvista"),
                    Requirement(LICENCE, "FEniCS licence", True, "none: FEniCS is free software (LGPL)"))
        return (found, Requirement(VERSION, "FEniCS version", True, found.detail),
                Requirement(LICENCE, "FEniCS licence", True, "none: FEniCS is free software (LGPL)"))

    def prepare(self, case: SolverCase) -> Path:
        if case.physics == LINEAR_ELASTICITY and case.dimension == 2:
            kind = self._PDES[(LINEAR_ELASTICITY, case.parameters.get("model", "plane_stress"))]
        elif case.physics == HEAT_TRANSFER:
            kind = self._PDES[(HEAT_TRANSFER, None)]
        else:
            raise CaseUnsupported(f"fenicsx runs the bridge's plane elasticity (2-D) or steady heat, not "
                                  f"{case.physics!r} in {case.dimension}-D")
        g, m = case.geometry, case.mesh
        bounds = (0.0, 0.0, g.get("L", 1.0), g.get("h", g.get("H", 1.0)))
        conditions = {}
        for i, bc in enumerate(case.boundary_conditions):
            side = {"x-": "left", "x+": "right", "y-": "bottom", "y+": "top"}.get(bc.get("side"), "left")
            conditions[bc.get("name", f"bc{i}")] = SimpleNamespace(type=bc.get("type", "dirichlet"), side=side,
                                                                   value=list(bc.get("value", [0.0, 0.0])))
        parameters = {"E": float(case.material.get("E", 1e6)), "nu": float(case.material.get("nu", 0.3))}
        for load in case.loads:
            if load.get("type") == "uniform_load":     # a load per length over the depth of the beam
                depth = bounds[3] - bounds[1]
                direction = load.get("direction", [0.0, -1.0])
                parameters["f"] = [float(direction[0]) * float(load["value"]) / depth,
                                   float(direction[1]) * float(load["value"]) / depth]
            else:
                raise CaseUnsupported(f"the FEniCS bridge carries a body force, not a {load.get('type')!r} load")
        parameters.update({k: v for k, v in case.parameters.items() if k in ("k", "mu")})
        self._spec = SimpleNamespace(kind=kind, domain_bounds=bounds, conditions=conditions, parameters=parameters)
        self._mesh_size = (m.get("nx", 32), m.get("ny", 32))
        self.case = case
        return self.workdir

    def _start(self, timeout: float | None) -> RunHandle:
        import numpy as np

        from ..numerical_solvers.fenics_bridge import FEnicsBridge

        try:
            bridge = FEnicsBridge(mesh_nx=self._mesh_size[0], mesh_ny=self._mesh_size[1],
                                  solver_backend="dolfinx" if _module_requirement("dolfinx", fix="").ok else "legacy")
            output = bridge.forward(self._spec)
            fields = dict(output.extras.get("fields") or {})
            if not fields:
                raise BackendMissing(f"the FEniCS bridge returned no fields: "
                                     f"{output.extras.get('error', 'no error reported')}")
            finite = all(np.isfinite(np.asarray(v)).all() for v in fields.values())
        except Exception as exc:                      # noqa: BLE001 - the run's log carries the failure
            self.log_path.write_text(f"the solve raised {type(exc).__name__}: {exc}\n")
            status = classify_run(backend=self.name, log=self.log_path.read_text(), returncode=1,
                                  converged=self.converged_markers, divergence=self.divergence_markers,
                                  version=self._known_version())
            return RunHandle.new(self.name, self.workdir, log_path=self.log_path, status=status)
        self._fields = fields
        self.log_path.write_text(f"solved: {output.extras.get('dofs')} dofs, kind {self._spec.kind}\n"
                                 + ("" if finite else "non-finite values in the solution\n"))
        status = classify_run(backend=self.name, log=self.log_path.read_text(), returncode=0,
                              converged=self.converged_markers, divergence=self.divergence_markers,
                              artefacts={"solution": finite}, elapsed_s=0.0, version=self._known_version())
        return RunHandle.new(self.name, self.workdir, log_path=self.log_path, status=status)

    def _classify(self, handle: RunHandle, returncode: int | None) -> SolverStatus:
        return classify_run(backend=self.name, log=handle.read_log(), returncode=returncode,
                            converged=self.converged_markers, divergence=self.divergence_markers,
                            elapsed_s=handle.elapsed_s, version=self._known_version())

    def _results(self) -> SolverResults:
        import numpy as np

        fields = {k: np.asarray(v, dtype=float) for k, v in self._fields.items() if not k.startswith("_")}
        scalars: dict[str, float | int] = {"dofs": int(fields.get("u", np.empty(0)).size)}
        for name in ("uy", "uz"):
            if name in fields:
                scalars["max_deflection"] = float(np.abs(fields[name]).max())
                break
        return SolverResults(backend=self.name, version=self.version(), fields=fields, scalars=scalars,
                             units={"u": "m", "max_deflection": "m"},
                             provenance={"physics_domain": "structural", "pde": self._spec.kind,
                                         "dofs_not_coordinates": True, "status": "converged"})


# ---------------------------------------------------------------------------------------------- ANSYS Fluent
class AnsysFluentBackend(ExternalSolver):
    """ANSYS Fluent in batch mode: a journal from the neutral case, ``fluent <precision> -g -i run.jou``.

    The licence is part of :meth:`check_environment` and not of the run: ANSYS is licence-gated, and a machine without
    ``ANSYSLMD_LICENSE_FILE`` (or ``LM_LICENSE_FILE``) is a machine that will fail at start, so the report says so
    before anything is launched. The version comes from the launcher's own banner, which only prints when a licensed
    session starts -- that requirement is reported as unverified, never as a pass.
    """

    name = "ansys-fluent"
    kind = "ansys-fluent"
    converged_markers = (r"^Calculation complete\.$", r"^Done\.$")
    divergence_markers = (r"Divergence detected in AMG solver", r"Floating point exception",
                          r"Divergence detected")

    def __init__(self, workdir: str | Path | None = None, *, executable: str = "fluent", precision: str = "3ddp",
                 n_procs: int = 1, extra_args: Sequence[str] = (), log_name: str = "fluent.log",
                 environ: dict[str, str] | None = None):
        super().__init__(workdir, log_name=log_name)
        self.executable = executable
        self.precision = precision
        self.n_procs = n_procs
        self.extra_args = tuple(extra_args)
        self.environ = environ

    def _requirements(self) -> Sequence[Requirement]:
        environ = self.environ if self.environ is not None else dict(os.environ)

        def executable() -> tuple[bool, str, str]:
            from .ansys.runner import _find_fluent_executable

            try:
                return True, str(_find_fluent_executable(self.executable)), ""
            except FileNotFoundError as exc:
                return False, str(exc), "install ANSYS Fluent, or pass executable='/path/to/fluent'"

        def licence() -> tuple[bool, str, str]:
            for var in ("ANSYSLMD_LICENSE_FILE", "LM_LICENSE_FILE"):
                if environ.get(var):
                    return True, f"{var}={environ[var]}", ""
            return (False, "no ANSYSLMD_LICENSE_FILE or LM_LICENSE_FILE in the environment",
                    "export ANSYSLMD_LICENSE_FILE=1055@your-licence-server, or install the licence file")

        return (self._probe(EXECUTABLE, "fluent", executable),
                Requirement(VERSION, "Fluent version", True, "read from the launcher's banner when it runs",
                            verified=False),
                self._probe(LICENCE, "ANSYS licence", licence))

    def prepare(self, case: SolverCase) -> Path:
        from .ansys.journal_builder import FluentCaseConfig, build_journal

        if case.physics not in (INCOMPRESSIBLE_FLOW, OPAQUE):
            raise CaseUnsupported(f"fluent runs a flow journal ({INCOMPRESSIBLE_FLOW}) or a case it cannot describe "
                                  f"({OPAQUE}), not {case.physics!r}")
        spec = case.parameters.get("fluent_case")
        if not spec:
            raise CaseUnsupported("fluent needs case.parameters['fluent_case']: the FluentCaseConfig fields "
                                  "(mesh/case file, models, iterations)")
        config = FluentCaseConfig(**spec)
        config.validate()
        self._journal = self.workdir / "run.jou"
        self._journal.write_text(build_journal(config))
        self._output = Path(config.output_case_file)
        self.case = case
        return self.workdir

    def _start(self, timeout: float | None) -> RunHandle:
        from .ansys.runner import _find_fluent_executable

        argv = [_find_fluent_executable(self.executable), self.precision, "-g", "-i", str(self._journal)]
        if self.n_procs > 1:
            argv.append(f"-t{self.n_procs}")
        argv += list(self.extra_args)
        process = self._spawn(argv)
        return RunHandle.new(self.name, self.workdir, argv=argv, log_path=self.log_path, process=process)

    def _classify(self, handle: RunHandle, returncode: int | None) -> SolverStatus:
        return classify_run(backend=self.name, log=handle.read_log(), returncode=returncode,
                            converged=self.converged_markers, divergence=self.divergence_markers,
                            artefacts={self._output.name: self._output.exists()}, elapsed_s=handle.elapsed_s,
                            version=self._known_version())

    def _results(self) -> SolverResults:
        import numpy as np

        from .cfd_formats.fluent_mesh_reader import fluent_mesh_to_upd

        sample = fluent_mesh_to_upd(str(self._output))
        fields = {k: np.asarray(v) for k, v in dict(sample.state).items() if not k.startswith("_")}
        return SolverResults(backend=self.name, version=self.version(), fields=fields,
                             coords={k: np.asarray(v) for k, v in (sample.domain.get("coords") or {}).items()},
                             provenance={"physics_domain": "cfd", "case_file": str(self._output),
                                         "status": "converged"})


# ---------------------------------------------------------------------------------------------- MATLAB
class MatlabBackend(ExternalSolver):
    """MATLAB in ``-batch`` mode: the neutral case is a script (or a function call with its arguments), the results are
    the ``.mat`` file it writes.

    The licence is checked by the launcher, not by us: a machine can have ``matlab`` on ``PATH`` and still fail
    checkout. That requirement is therefore reported unverified, and the launcher's refusal ("License checkout
    failed") lands in the run's log where the status carries it as evidence.
    """

    name = "matlab"
    kind = "matlab"
    converged_markers = ()                           # a script has no end marker: exit 0 and the .mat it wrote
    divergence_markers = (r"^the script raised\b", r"non-finite")

    def __init__(self, workdir: str | Path | None = None, *, matlab_executable: str = "matlab", timeout: float = 300.0,
                 log_name: str = "matlab.log"):
        super().__init__(workdir, log_name=log_name)
        self.matlab_executable = matlab_executable
        self.timeout = timeout

    def _requirements(self) -> Sequence[Requirement]:
        def executable() -> tuple[bool, str, str]:
            path = shutil.which(self.matlab_executable)
            if path:
                return True, path, ""
            return (False, f"{self.matlab_executable} not on PATH",
                    "install MATLAB and put its bin directory on PATH, or pass matlab_executable='/path/to/matlab'")

        return (self._probe(EXECUTABLE, self.matlab_executable, executable),
                Requirement(VERSION, "MATLAB version", True, "printed by the launcher under -batch", verified=False),
                Requirement(LICENCE, "MATLAB licence", True,
                            "checked by the launcher at start (a checkout failure appears in the run's log)",
                            verified=False))

    def prepare(self, case: SolverCase) -> Path:
        if not case.source:
            raise CaseUnsupported("matlab needs case.source: the .m script to run")
        script = Path(case.source)
        if not script.is_file():
            raise CaseUnsupported(f"the MATLAB script {script} does not exist")
        shutil.copy(script, self.workdir / script.name)
        self._script = self.workdir / script.name
        self._mat = self.workdir / (case.parameters.get("result_file") or f"{script.stem}.mat")
        self.case = case
        return self.workdir

    def _start(self, timeout: float | None) -> RunHandle:
        batch = f"addpath('{self._script.parent}'); {self._script.stem}"
        argv = [self.matlab_executable, "-nodesktop", "-nosplash", "-batch", batch]
        process = self._spawn(argv)
        return RunHandle.new(self.name, self.workdir, argv=argv, log_path=self.log_path, process=process)

    def _classify(self, handle: RunHandle, returncode: int | None) -> SolverStatus:
        return classify_run(backend=self.name, log=handle.read_log(), returncode=returncode,
                            converged=self.converged_markers, divergence=self.divergence_markers,
                            artefacts={self._mat.name: self._mat.is_file()}, elapsed_s=handle.elapsed_s,
                            version=self._known_version())

    def _results(self) -> SolverResults:
        import numpy as np

        from .matlab.mat_io import load_mat

        data = load_mat(self._mat)
        fields, scalars = {}, {}
        for name, value in (data or {}).items():
            arr = np.asarray(value, dtype=float) if not isinstance(value, dict) else None
            if arr is None:
                continue
            if arr.ndim >= 1 and arr.size > 1:
                fields[name] = arr
            else:
                scalars[name] = float(arr.ravel()[0]) if arr.size else 0.0
        return SolverResults(backend=self.name, version=self.version(), fields=fields, scalars=scalars,
                             provenance={"physics_domain": "script", "script": str(self._script),
                                         "result_file": str(self._mat), "status": "converged"})


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------------------------- registry
BACKEND_CLASSES: tuple[type[ExternalSolver], ...] = (OpenFOAMBackend, CalculiXBackend, FEniCSxBackend,
                                                     AnsysFluentBackend, MatlabBackend, SolidFEMBackend)

for _cls in BACKEND_CLASSES:
    register_solver(_cls, key=_cls.kind)

#: registry key -> backend class, for a caller that wants the class rather than a ready instance.
BACKENDS: dict[str, type[ExternalSolver]] = {cls.kind: cls for cls in BACKEND_CLASSES}
