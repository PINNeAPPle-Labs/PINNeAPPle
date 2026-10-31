"""Mathematical toolkit for dynamical systems.

Five frameworks that operate on trajectories from any source (measurements, simulations, closed forms, PINN / FNO /
DeepONet predictions), each with its benchmarks:

* :mod:`.first_passage`: first-passage and hitting times (deterministic crossing, Kaplan-Meier from observed
  trajectories, Monte Carlo of SDEs with Brownian-bridge correction, closed forms). Guide:
  ``docs/guides/dynamics_first_passage.md``.

* :mod:`.dimensional`: dimensional analysis and Buckingham pi in exact rational arithmetic, scaling laws,
  dimensional vs empirical normalisation, curve collapse with a permutation control, similarity. Guide:
  ``docs/guides/dynamics_dimensional.md``.

* :mod:`.fisher`: Fisher information (Gaussian, Poisson, any likelihood by score Monte Carlo with finite
  differences, PyTorch or JAX), Cramér-Rao bounds on parameters and on targets, identifiability, and experimental
  design (D, A, E, c, Bayesian D) built on :mod:`pinneapple_analysis.inverse_problems`. Guide:
  ``docs/guides/dynamics_fisher.md``.

* :mod:`.early_warning`: critical slowing down and early-warning indicators (rolling variance and autocorrelation,
  recovery rate, spectral indicators, detrending, surrogate significance, window / bandwidth / sampling
  sensitivity), with the indicator kept apart from any prediction. Guide: ``docs/guides/dynamics_early_warning.md``.

* :mod:`.koopman`: Koopman operator and spectral analysis (exact and TLS DMD with Gavish-Donoho rank selection,
  EDMD on dictionaries of observables including neural ones, eigenvalues, eigenfunctions, ResDMD residuals,
  generator EDMD, prediction and reconstruction). Its :class:`KoopmanOperator` is a fitted NumPy operator, distinct
  from the trainable layer ``pinneapple_neural.architectures.rom.koopman.KoopmanOperator``. Guide:
  ``docs/guides/dynamics_koopman.md``.

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

from . import dimensional, early_warning, first_passage, fisher, koopman
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
from .early_warning import (
    AutocorrelationAnalyzer,
    CriticalSlowingDown,
    EarlyWarningAnalyzer,
    RecoveryRateEstimator,
    RollingStatistics,
    StabilityIndicator,
    TransitionDiagnostics,
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
from .fisher import (
    CramerRaoBound,
    ExperimentalDesign,
    FisherInformation,
    FisherInformationMatrix,
    InformationGain,
    ObservationScheduler,
    ParameterIdentifiability,
    SamplingStrategy,
    SensitivityAnalyzer,
)
from .koopman import (
    DMD,
    EDMD,
    DynamicMode,
    KoopmanEigenfunctions,
    KoopmanEigenvalues,
    KoopmanOperator,
    KoopmanReconstruction,
    ObservableDictionary,
    SpectralAnalyzer,
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
    "fisher", "FisherInformation", "FisherInformationMatrix", "CramerRaoBound", "ParameterIdentifiability",
    "SensitivityAnalyzer", "ExperimentalDesign", "ObservationScheduler", "InformationGain", "SamplingStrategy",
    "early_warning", "EarlyWarningAnalyzer", "CriticalSlowingDown", "RollingStatistics", "AutocorrelationAnalyzer",
    "RecoveryRateEstimator", "StabilityIndicator", "TransitionDiagnostics",
    "koopman", "KoopmanOperator", "DMD", "EDMD", "ObservableDictionary", "KoopmanEigenvalues",
    "KoopmanEigenfunctions", "SpectralAnalyzer", "DynamicMode", "KoopmanReconstruction",
]
