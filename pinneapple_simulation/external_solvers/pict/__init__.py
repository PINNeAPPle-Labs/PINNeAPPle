"""PICT: a differentiable multi-block PISO solver on CUDA, driven from PINNeAPPle.

`PICT <https://github.com/tum-pbs/PICT>`_ (Franz, Wei, Guastoni and Thuerey, *A differentiable, GPU-accelerated
multi-block PISO solver for simulation-coupled learning tasks in fluid dynamics*, Journal of Computational Physics
544 (2026) 114433, Apache-2.0) solves the incompressible Navier-Stokes equations with PISO on structured, deformed
multi-block grids. Its fields are plain PyTorch tensors and every step supports autograd through hand-written CUDA
backward kernels, which is what makes it useful here: a network can sit inside a CFD solve and be trained by
backpropagating through it.

What this wrapper is, and what it is not
----------------------------------------
It is a bridge: it builds the grid, the domain, the boundaries and the ``PISOtorch_simulation.Simulation``
configuration from a description in SI units, runs it, records the velocity and pressure fields it read back, and
returns them as a :class:`PICTResult` and a PINNeAPPle ``PhysicalSample``. It adds no physics of its own.

It is **not** a reimplementation and it does not fake a run. PICT needs Linux, an NVIDIA GPU and its own compiled
extensions (``extensions/setup.py``, CUDA 12.8 as the paper's environment); on a machine without them
``run_cavity(..., engine="pict")`` raises with the missing part named. Nothing in this repository has been executed
against PICT -- state that plainly when reporting a PICT number. What *is* checked without the solver is the grid
generator, the case validation, the reference data and the reduction of a field to the quantities the checks need
(``tests/test_pict_bridge.py``).

``engine="local"`` runs the cavity here instead, with this repository's own vorticity-streamfunction solver
(:mod:`.local_solver`, numpy/scipy, no GPU): the same case, the same result type, the same Ghia comparison, and a
result that says ``engine="local"`` so the two are never confused. ``engine="auto"`` (the default) picks PICT when it
can run and the local engine when it cannot.

Sub-modules
-----------
case       — availability, the wall-refined grid, the lid-driven cavity case and its PISOtorch domain
runner     — PICTRunConfig, run_cavity() -> PICTResult (velocity, pressure, times) and its PhysicalSample
validation — Ghia et al. (1982) centreline reference data, the checks of a cavity result, and validate()

Quick start (on a machine with PICT installed)
----------------------------------------------
>>> from pinneapple_simulation.external_solvers.pict import CavityCase, PICTRunConfig, run_cavity
>>> result = run_cavity(CavityCase(nx=32, ny=32, Re=1000), PICTRunConfig(iterations=100), engine="auto")
>>> result.engine                               # "pict" (CUDA) or "local" (this repository's CPU solver)
>>> result.velocity.shape                     # (snapshots, 2, ny, nx)
>>> checks = result.checks()                   # continuity, lid and wall boundary conditions, Ghia profile
>>> print(checks["ghia_u"]["rms"], checks["ghia_u"]["passed"])

Install: ``git clone https://github.com/tum-pbs/PICT.git``, then ``cd extensions && python setup.py install`` (see
the project's installation guide; the Python side also needs ``torch`` with CUDA).
"""
from .case import (
    CavityCase,
    PICTAvailability,
    build_cavity_domain,
    pict_availability,
    pict_available,
    wall_refined_grid,
)
from .local_solver import LocalSolverConfig, solve_cavity_local
from .runner import ENGINES, PICTResult, PICTRunConfig, run_cavity
from .validation import (
    GHIA,
    cavity_checks,
    centerline,
    compare_to_ghia,
    divergence_max,
    ghia_profile,
    validate,
)

__all__ = [
    "pict_available", "pict_availability", "PICTAvailability",
    "wall_refined_grid", "CavityCase", "build_cavity_domain",
    "PICTRunConfig", "PICTResult", "run_cavity", "ENGINES",
    "LocalSolverConfig", "solve_cavity_local",
    "GHIA", "ghia_profile", "centerline", "compare_to_ghia", "divergence_max",
    "cavity_checks", "validate",
]
