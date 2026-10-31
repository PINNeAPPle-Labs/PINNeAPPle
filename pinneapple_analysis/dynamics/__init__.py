"""Mathematical toolkit for dynamical systems: the shared foundation.

This package will hold five frameworks that operate on trajectories from any source (measurements, simulations,
closed forms, PINN / FNO / DeepONet predictions): first-passage and hitting times, dimensional analysis and
Buckingham pi, Fisher information and experimental design, critical slowing down and early-warning indicators, and
Koopman / spectral analysis. Each lands in its own pull request with its benchmarks; none is available yet.

What is here now is what they share:

* the method contract (:mod:`._contract`): :class:`MethodCard` (inputs, assumptions, configuration, outputs,
  role, failure and applicability conditions, written before a run) and :class:`DynamicsResult` (estimate,
  uncertainty, checked and unchecked assumptions, diagnostics, convergence, applicability, reproducibility);
* backend detection (:mod:`._backends`): NumPy and SciPy are the core; PyTorch and JAX are detected without being
  imported and used only where a method offers automatic differentiation;
* trajectories (:mod:`.trajectories`): :class:`Trajectory` and :class:`TrajectoryEnsemble` with irregular
  sampling, missing values and right-censored observation windows, and the adapters from arrays, tables and CSV
  files, ``pp.solve`` solutions, callables (trained networks) and PDE snapshots.

Guide: ``docs/guides/dynamics_toolkit.md``.
"""
from __future__ import annotations

from ._backends import BackendUnavailable, Capabilities, capabilities, require, to_numpy
from ._contract import (
    ROLES,
    Assumption,
    Diagnostic,
    DynamicsResult,
    MethodCard,
    Reproducibility,
    fingerprint,
    jsonable,
    make_rng,
    reproducibility,
)
from .trajectories import (
    SOURCE_KINDS,
    Trajectory,
    TrajectoryEnsemble,
    from_array,
    from_csv,
    from_model,
    from_snapshots,
    from_solution,
    from_table,
)

__all__ = [
    "Assumption", "Diagnostic", "MethodCard", "DynamicsResult", "Reproducibility", "ROLES",
    "fingerprint", "make_rng", "reproducibility", "jsonable",
    "BackendUnavailable", "Capabilities", "capabilities", "require", "to_numpy",
    "SOURCE_KINDS", "Trajectory", "TrajectoryEnsemble",
    "from_array", "from_table", "from_csv", "from_solution", "from_model", "from_snapshots",
]
