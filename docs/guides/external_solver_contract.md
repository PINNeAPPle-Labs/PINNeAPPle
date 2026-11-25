# External solvers: one interface, six backends, and what is out of scope

OpenFOAM, CalculiX, ANSYS Fluent, FEniCSx and MATLAB arrived in this library one at a time, each with its own case
layout, its own health check, its own log parsing and its own idea of what "finished" means. Adding the fourth solver
meant reading the third one's code, which is the problem
[`pinneapple_simulation.external_solvers.contract`](../api/pinneapple_simulation/index.md) removes: the gap was never
coverage, it was the contract.

## The contract

| Method | Answers |
|---|---|
| `check_environment()` | what this machine is missing -- executable, version, licence, environment, Python module -- with the evidence and the fix. **Never raises.** |
| `prepare(case)` | materialises a neutral `SolverCase` (physics, geometry, material, mesh, boundary conditions, loads) into that backend's own input |
| `run()` | starts it in its own session and returns a `RunHandle` |
| `status()` | `queued` / `running` / `converged` / `diverged` / `failed` / `cancelled`, with the log lines that say so |
| `results()` | fields and scalars in one schema (`to_upd()` hands them to the rest of the library), provenance naming the backend and its version |
| `cancel()` | idempotent -- a second call reports the same state |

Two rules make it useful:

* **the environment verdict never raises.** A solver that is not installed is a paragraph with a fix, not an exception
  in the middle of a campaign:
  `raise_if_missing()` is the only thing that raises, and it raises the same text.
* **there is one definition of "finished"** (`classify_run`): a divergence marker in the log outranks the exit code,
  which outranks the artefact the solver had to write, which outranks the solver's own end marker. A run that blew up
  numerically is never reported as a mere failure -- the two are different answers because the reader does different
  things with them.

```python
from pinneapple_simulation.external_solvers import SolverCase, check_all, get_solver

for name, report in check_all().items():
    print(report.explain())                  # what each solver needs, what this machine lacks, the fix per line

solver = get_solver("calculix", workdir="run")
solver.check_environment().raise_if_missing()
solver.prepare(SolverCase.cantilever(P=1000.0))
solver.run()
print(solver.wait().explain())               # converged / diverged / failed, with the evidence
sample = solver.results().to_upd()           # the neutral scientific schema, provenance included
```

## The six backends

| key | wraps | needs |
|---|---|---|
| `openfoam` | `openfoam.campaign`, `openfoam.field_reader` | an OpenFOAM environment: the probe sources the `bashrc` (the step's, `FOAM_BASHRC`, or one found under `/usr/share/openfoam`, `/opt/openfoam*`, `/Volumes/OpenFOAM*`) and asks it for the executables and the version, because a distribution package puts `blockMesh` on `PATH` but the binaries only run with the environment loaded |
| `calculix` | `calculix.inp`, `calculix.frd`, `calculix.runner` | `ccx` on `PATH` or Docker for the bundled image (no licence -- CalculiX is GPL); a `.frd` holding non-finite displacements is reported `diverged` even when the exit code is 0 |
| `fenicsx` | `numerical_solvers.fenics_bridge` | DOLFINx 0.8 or 0.9 (see below) or legacy FEniCS; plane elasticity and steady conduction |
| `ansys-fluent` | `ansys.runner`, `ansys.journal_builder` | the `fluent` launcher **and a licence** (`ANSYSLMD_LICENSE_FILE` or `LM_LICENSE_FILE`) -- the licence is part of the environment verdict, not a surprise at start |
| `matlab` | `matlab.runner`, `matlab.mat_io` | `matlab` on `PATH`; the licence is checked by the launcher, so that requirement is reported as unverified rather than as a pass |
| `solid-fem` | `numerical_solvers.solid_fem` | nothing: it solves in this process (8-node hexahedra with incompatible modes, the element Abaqus calls C3D8I) |

The same neutral case goes through more than one backend: a 1 x 0.1 x 0.1 m steel cantilever under a 1 kN tip load
returns **0.9937** of the Timoshenko tip deflection through CalculiX (C3D20R) and **0.9894** through the in-process
FEM (C3D8I equivalent) -- same schema, provenance naming each backend and its version, 0.6 % apart.

## FEniCSx: the version is part of the environment

`numerical_solvers.fenics_bridge` calls `FunctionSpace(mesh, element)` and `dolfinx.fem.petsc.LinearProblem`, both of
which DOLFINx 0.10 changed. The backend therefore reports the release as a *requirement*:

```
[missing] dolfinx version: 0.11.0 is outside what this bridge speaks (0.8-0.9: FunctionSpace(mesh, element) and
          dolfinx.fem.petsc, both changed in 0.10) (fix: conda install -c conda-forge fenics-dolfinx=0.9 mpich ...)
```

Installed and usable are different answers, and the report says which one this machine has. With 0.9 installed, the
backend reproduces a closed form rather than another solver's numbers: for
`SolverCase.conduction` (`-k T'' = source` with two sides held), a linear exact solution comes back to solver
precision, and a sourced one converges at the second order the P1 mesh allows.

## COMSOL is out of scope, and says so

The interface does **not** pretend to cover COMSOL. There is no COMSOL implementation in this repository, and there
cannot be an honest one here: its Python coupling (`mph`, LiveLink) needs the commercial client installed and licensed,
and its model files are binary and proprietary, so nothing about it can be exercised or verified in this repository's
tests -- which is exactly the kind of claim this project does not make. A backend that cannot be run is worse than no
backend: it moves the failure from `check_environment()` to the middle of a run, which is the problem the contract
exists to fix.

Adding it later is one class and one registration, which is the point of the contract:

```python
from pinneapple_simulation.external_solvers import ExternalSolver, register_solver

class ComsolBackend(ExternalSolver):
    name = "comsol"
    kind = "comsol"
    def _requirements(self): ...     # the mph module, the licence, the client version
    def prepare(self, case): ...
    def _start(self, timeout): ...
    def _classify(self, handle, returncode): ...   # what the client's log says "finished" is
    def _results(self): ...

register_solver(ComsolBackend)
```

Until then `get_solver("comsol")` raises `KeyError` naming the backends that do exist, and `check_all()` reports those
six -- no silent absence.

## Writing a backend

Fill in five things and inherit the rest (`check_environment`, `status`, `cancel`, `wait` and the handle are written
once, in `ExternalSolver`): `_requirements`, `prepare`, `_start`, `_classify` and `_results`, plus the
`converged_markers` / `divergence_markers` that feed `classify_run`. The workflow step surface
(`pinneapple_core.workflows.adapters`) asks the same contract for its availability verdict, so a workflow step and the
library cannot disagree about whether a solver is installed.
