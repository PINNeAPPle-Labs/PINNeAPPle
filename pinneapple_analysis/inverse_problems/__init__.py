"""pinneapple_inverse — Inverse Problems module for PINNeAPPle.

Provides a complete toolkit for physics-informed inverse problems:

Core components
---------------
Noise models (data-misfit terms):
  GaussianMisfit            standard L2 / Mahalanobis distance
  HuberMisfit               robust to moderate outliers
  CauchyMisfit              heavy-tailed; very robust
  StudentTMisfit            Student-t likelihood (controls tail weight)
  HeteroscedasticMisfit     per-observation noise variance

Regularization:
  TikhonovRegularizer       L2 ‖θ − θ₀‖², Mahalanobis variant
  SparsityRegularizer       L1 ‖θ‖₁ (smooth approximation)
  TotalVariationRegularizer TV(u) for spatial field parameters
  CompositeRegularizer      weighted sum of multiple regularizers
  LCurveSelector            automatic λ via L-curve + max curvature

Observation operators:
  PointObsOperator          evaluate u at sparse sensor locations
  LinearObsOperator         y = A u (projections, Radon, …)
  IntegralObsOperator       y = ∫ k(x) u(x) dx
  ComposedObsOperator       stack multiple operators

Sensitivity analysis:
  LocalSensitivity          Jacobian J = ∂G/∂θ; FIM = J^T Γ⁻¹ J
  IdentifiabilityAnalyzer   eigenanalysis of FIM; D/A/E-optimality
  GlobalSensitivity         Sobol variance-based indices (Saltelli method)

Derivative-free inversion:
  EnsembleKalmanInversion   standard EKI (Iglesias et al. 2013)
  IteratedEKI               Tikhonov-regularized EKI / TEKI (Chada et al. 2020)

High-level solver:
  InverseProblemSolver      wires all components; supports Adam / L-BFGS / EKI

Quick start
-----------
>>> import pinneapple_analysis.inverse_problems as pinv
>>> import torch

>>> # Define forward model (PINN)
>>> from pinneapple_neural.architectures import ModelRegistry
>>> model = ModelRegistry.build("VanillaPINN", in_dim=2, out_dim=1)

>>> # Observation operator: evaluate at 50 sensor points
>>> sensor_locs = torch.rand(50, 2)
>>> H = pinv.PointObsOperator(sensor_locs)

>>> # Noise model
>>> D = pinv.HuberMisfit(delta=0.3, noise_std=0.02)

>>> # Regularizer
>>> R = pinv.TikhonovRegularizer(lambda_reg=1e-3)

>>> # Solver
>>> cfg = pinv.InverseSolverConfig(method="adam", n_iters=2000, lr=5e-4)
>>> solver = pinv.InverseProblemSolver(model, H, D, R, cfg)

>>> # Synthetic observations
>>> y_obs = H(model, sensor_locs).detach() + 0.02 * torch.randn(50, 1)

>>> # Solve
>>> result = solver.solve(y_obs, sensor_locs)
>>> print(f"Final misfit: {result.final_misfit:.4e}")
"""
from __future__ import annotations

from .calibration import CalibrationResult, calibrate, residual_metrics

# Counterfactual (Shapley-value) attribution
from .counterfactual_attribution import (
    AmbiguityResult,
    CandidateCause,
    ExactShapleyAttributor,
    ShapleyAttributionResult,
    classify_ambiguity,
)

# Ensemble Kalman Inversion
from .ensemble_kalman import (
    EKIConfig,
    EKIHistory,
    EnsembleKalmanInversion,
    IteratedEKI,
)
from .estimation import FitResult, Prior, gradient_fit, least_squares_fit, metropolis

# Identifiability, calibration and the inverse studio (one core for the identifiability, calibration and inverse
# model apps): models, parameters and observations; estimators; the four identifiability methods and D-optimal
# design; calibration with intervals and held-out validation; one interface over five inversion methods; presets.
from .forward import (
    CallableModel,
    ExpressionModel,
    ForwardModel,
    Observations,
    Parameter,
    ParameterSet,
    TabulatedModel,
    jacobian,
)
from .identifiability import (
    IdentifiabilityReport,
    analyze_identifiability,
    d_optimal_design,
    profile_likelihood,
)

# Missing term identification
from .missing_term import (
    CandidateLibrary,
    NeuralTermConfig,
    NeuralTermDiscovery,
    ResidualAnalysisResult,
    ResidualAnalyzer,
    SINDyIdentifier,
    SINDyResult,
)
from .models import ArrheniusModel, LumpedCoolingModel, PinFinModel, PlateModel, RodModel

# Noise models
from .noise_models import (
    CauchyMisfit,
    DataMisfitBase,
    GaussianMisfit,
    HeteroscedasticMisfit,
    HuberMisfit,
    StudentTMisfit,
)

# Observation operators
from .obs_operator import (
    ComposedObsOperator,
    IntegralObsOperator,
    LinearObsOperator,
    ObsOperatorBase,
    PointObsOperator,
)
from .presets import get_preset, list_presets, model_from_spec

# Regularization
from .regularization import (
    CompositeRegularizer,
    LCurveResult,
    LCurveSelector,
    RegularizerBase,
    SparsityRegularizer,
    TikhonovRegularizer,
    TotalVariationRegularizer,
)

# Sensitivity analysis
from .sensitivity import (
    GlobalSensitivity,
    IdentifiabilityAnalyzer,
    IdentifiabilityResult,
    LocalSensitivity,
    LocalSensitivityResult,
    SobolResult,
)

# High-level solver
from .solver import (
    InverseProblemSolver,
    InverseSolverConfig,
    InverseSolverResult,
)
from .studio import METHODS, InverseProblem, available_methods, compare_methods, solve_inverse

__version__ = "0.1.0"

__all__ = [
    # Noise models
    "DataMisfitBase",
    "GaussianMisfit",
    "HuberMisfit",
    "CauchyMisfit",
    "StudentTMisfit",
    "HeteroscedasticMisfit",
    # Regularization
    "RegularizerBase",
    "TikhonovRegularizer",
    "SparsityRegularizer",
    "TotalVariationRegularizer",
    "CompositeRegularizer",
    "LCurveSelector",
    "LCurveResult",
    # Observation operators
    "ObsOperatorBase",
    "PointObsOperator",
    "LinearObsOperator",
    "IntegralObsOperator",
    "ComposedObsOperator",
    # Sensitivity
    "LocalSensitivity",
    "LocalSensitivityResult",
    "IdentifiabilityAnalyzer",
    "IdentifiabilityResult",
    "GlobalSensitivity",
    "SobolResult",
    # EKI
    "EKIConfig",
    "EKIHistory",
    "EnsembleKalmanInversion",
    "IteratedEKI",
    # Solver
    "InverseSolverConfig",
    "InverseSolverResult",
    "InverseProblemSolver",
    # Counterfactual (Shapley-value) attribution
    "CandidateCause",
    "ExactShapleyAttributor",
    "ShapleyAttributionResult",
    "AmbiguityResult",
    "classify_ambiguity",
    # Missing term identification
    "CandidateLibrary",
    "SINDyResult",
    "SINDyIdentifier",
    "ResidualAnalysisResult",
    "ResidualAnalyzer",
    "NeuralTermConfig",
    "NeuralTermDiscovery",
    # Identifiability, calibration, inverse studio
    "Parameter",
    "ParameterSet",
    "Observations",
    "ForwardModel",
    "CallableModel",
    "ExpressionModel",
    "TabulatedModel",
    "jacobian",
    "FitResult",
    "Prior",
    "least_squares_fit",
    "gradient_fit",
    "metropolis",
    "IdentifiabilityReport",
    "analyze_identifiability",
    "d_optimal_design",
    "profile_likelihood",
    "CalibrationResult",
    "calibrate",
    "residual_metrics",
    "PinFinModel",
    "RodModel",
    "PlateModel",
    "LumpedCoolingModel",
    "ArrheniusModel",
    "METHODS",
    "InverseProblem",
    "available_methods",
    "solve_inverse",
    "compare_methods",
    "get_preset",
    "list_presets",
    "model_from_spec",
]
