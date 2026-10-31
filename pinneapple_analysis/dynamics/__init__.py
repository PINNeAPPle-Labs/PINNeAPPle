"""Mathematical toolkit for dynamical systems.

Five frameworks that operate on trajectories from any source (measurements, simulations, closed forms, PINN / FNO /
DeepONet predictions) are being added one pull request at a time, each with its benchmarks. Available now:

* :mod:`.first_passage`: first-passage and hitting times (deterministic crossing, Kaplan-Meier from observed
  trajectories, Monte Carlo of SDEs with Brownian-bridge correction, closed forms). Guide:
  ``docs/guides/dynamics_first_passage.md``.

* :mod:`.dimensional`: dimensional analysis and Buckingham pi in exact rational arithmetic, scaling laws,
  dimensional vs empirical normalisation, curve collapse with a permutation control, similarity. Guide:
  ``docs/guides/dynamics_dimensional.md``.

Not yet available: Fisher information and experimental design, critical slowing down and early-warning indicators,
Koopman / spectral analysis.

The frameworks share:

* the method contract (:mod:`._contract`): :class:`MethodCard` (inputs, assumptions, configuration, outputs,
  role, failure and applicability conditions, written before a run) and :class:`DynamicsResult` (estimate,
  uncertainty, checked and unchecked assumptions, diagnostics, convergence, applicability, reproducibility);
* backend detection (:mod:`._backends`): NumPy and SciPy are the core; PyTorch and JAX are detected without being
  imported and used only where a method offers automatic differentiation;
* trajectories (:mod:`.trajectories`): :class:`Trajectory` and :class:`TrajectoryEnsemble` with irregular
  sampling, missing values and right-censored observation windows, and the adapters from arrays, tables and CSV
  files, ``pp.solve`` solutions, callables (trained networks) and PDE snapshots.

Guide to the shared parts: ``docs/guides/dynamics_toolkit.md``.
"""
from __future__ import annotations

from . import dimensional, first_passage
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
from .dimensional import (
    BuckinghamPiAnalyzer,
    CurveCollapseAnalyzer,
    Dimension,
    DimensionalMatrix,
    DimensionlessGroup,
    PhysicalQuantity,
    ScalingLaw,
    SimilarityMetrics,
    TrajectoryNormalizer,
)
from .first_passage import (
    SDE,
    BoundaryCondition,
    CrossingProbability,
    FirstPassageAnalyzer,
    FirstPassageResult,
    HittingTimeDistribution,
    SurvivalFunction,
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
    "first_passage", "FirstPassageAnalyzer", "HittingTimeDistribution", "CrossingProbability", "SurvivalFunction",
    "BoundaryCondition", "FirstPassageResult", "SDE",
    "dimensional", "Dimension", "PhysicalQuantity", "DimensionalMatrix", "BuckinghamPiAnalyzer", "DimensionlessGroup",
    "ScalingLaw", "TrajectoryNormalizer", "CurveCollapseAnalyzer", "SimilarityMetrics",
]
