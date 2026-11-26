"""``pinneapple_pipeline``: the surrogate-model pipeline of PINNeAPPle.

One object that takes a problem from its definition to a verified, exportable surrogate, with VeriPhysics as a layer
over every step instead of a check at the end:

>>> import pinneapple as pp
>>> result = pp.SurrogatePipeline.from_preset("burgers_in_viscosity").run()      # doctest: +SKIP
>>> result.verification[6].status                                               # doctest: +SKIP
'PASS'

What each part is for:

=========================  ==================================================================================
``SurrogatePipeline``      the facade: ``run``, ``run_until``, ``resume``, ``status``, ``predict_verified``
``configs``                one dataclass per block of the pipeline, round-trips through YAML/JSON
``stages``                 the seven stages, each with ``validate``/``execute``/``verify``
``adapters``               the only place a configuration meets a module that computes something
``verification``           the ten VeriPhysics gates and the evidence each one leaves
``results``                ``VerificationResult``, ``StageRecord``, ``PipelineResult`` (and its report)
``presets``                four ready configurations
=========================  ==================================================================================

What it orchestrates (nothing here re-implements these): problem presets and condition sampling
(``pinneapple_physics.pde_environment``), the symbolic PDE parser (``pinneapple_physics.symbolic_pde``), reference
solves and comparisons (``pinneapple_physics.solving``), designs of experiments (``pinneapple_data.doe``),
model architectures and training (``pinneapple_neural``), the supervised surrogate study with its own trust checks
(``pinneapple_analysis.surrogate``), validation of a trained PINN (``pinneapple_analysis.validation.pinn``), and
model export (``pinneapple_tools.model_export``).
"""
from __future__ import annotations

from .configs import (
    ArchitectureConfig,
    BoundaryConditionConfig,
    ConfigError,
    DataConfig,
    DimensionConfig,
    ExecutionConfig,
    ExportConfig,
    LossConfig,
    ParameterSpaceConfig,
    PhysicsConfig,
    PipelineConfig,
    ProblemConfig,
    SamplingConfig,
    SolverConfig,
    SurrogateConfig,
    TrainingConfig,
    UncertaintyConfig,
    VerificationConfig,
    VisualizationConfig,
)
from .pipeline import PipelineFailed, PipelineRejected, SurrogatePipeline, SurrogatePipelineResult
from .presets import PRESETS, preset_config
from .results import (
    FAIL,
    NOT_APPLICABLE,
    NOT_EVALUATED,
    PASS,
    WARN,
    GateStatus,
    PipelineResult,
    StageRecord,
    VerificationResult,
)
from .stages import PipelineStage
from .state import RunState

__all__ = [
    # facade
    "SurrogatePipeline", "SurrogatePipelineResult", "PipelineResult", "PipelineRejected", "PipelineFailed",
    "PipelineStage", "RunState",
    # configuration
    "SurrogateConfig", "PipelineConfig", "ProblemConfig", "DimensionConfig", "PhysicsConfig", "BoundaryConditionConfig",
    "ParameterSpaceConfig", "SamplingConfig", "SolverConfig", "DataConfig", "ArchitectureConfig", "TrainingConfig",
    "LossConfig", "ExecutionConfig", "VerificationConfig", "UncertaintyConfig", "ExportConfig",
    "VisualizationConfig", "ConfigError",
    # verification
    "VerificationResult", "StageRecord", "GateStatus", "PASS", "WARN", "FAIL", "NOT_EVALUATED", "NOT_APPLICABLE",
    # presets
    "PRESETS", "preset_config",
]

# experimental (labs): see pinneapple_stability and docs/api_stability.md
from pinneapple_stability import warn_experimental as _warn_experimental  # noqa: E402

_warn_experimental(__name__)
