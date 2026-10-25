"""Request and response bodies of the PINNeAPPle Arena API (pydantic v2); the catalog descriptors come from
``pinneapple_arena.pointwise.schema``."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from pinneapple_arena.pointwise.schema import (  # noqa: F401  (the catalog descriptors, re-exported for the routes)
    ArchitectureFamily,
    ArchitectureInfo,
    ColumnHints,
    DifficultyLevel,
    Dimension,
    GeometryPreset,
    GeometryType,
    MeshInfo,
    PhysicsArea,
    PhysicsProblem,
    ProblemParameter,
)


class ExperimentStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ArchitectureFeedbackRequest(BaseModel):
    """Request body for posting architecture feedback."""
    is_positive: bool
    comment: str | None = Field(default=None, max_length=2000)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

class ProblemType(str, Enum):
    FORWARD = "forward"
    INVERSE = "inverse"


class InverseNoiseModel(str, Enum):
    GAUSSIAN = "gaussian"
    HUBER = "huber"
    CAUCHY = "cauchy"
    STUDENT_T = "student_t"


class InverseSolverMethod(str, Enum):
    ADAM = "adam"
    LBFGS = "lbfgs"
    EKI = "eki"
    TEKI = "teki"


class InverseConfig(BaseModel):
    """Configuration for PINNeAPPle inverse problem solving."""
    params: list[str] = Field(
        default_factory=list,
        description="PDE parameter names to identify (e.g. ['nu', 'k'])",
    )
    initial_guesses: dict[str, float] = Field(
        default_factory=dict,
        description="Initial guess for each unknown parameter",
    )
    n_obs: int = Field(default=200, ge=1, description="Number of observation points")
    noise_std: float = Field(default=0.01, ge=0.0, description="Observation noise standard deviation")
    method: InverseSolverMethod = InverseSolverMethod.ADAM
    n_iters: int = Field(default=1000, ge=1, description="Solver iterations")
    lambda_reg: float = Field(default=1e-4, ge=0.0, description="Regularisation strength")
    noise_model: InverseNoiseModel = InverseNoiseModel.GAUSSIAN


class ColumnMapping(BaseModel):
    """Describes how uploaded dataset columns map to coordinates / outputs."""
    coord_cols: list[str] = Field(default_factory=list)
    output_cols: list[str] = Field(default_factory=list)
    time_col: str | None = None


class ExperimentConfig(BaseModel):
    """Complete configuration for launching an arena experiment."""
    problem_id: str
    problem_type: ProblemType = ProblemType.FORWARD
    geometry_id: str | None = None
    architecture_ids: list[str]
    n_train_col: int = Field(default=10_000, ge=100)
    n_train_bc: int = Field(default=2_000, ge=10)
    n_train_ic: int = Field(default=2_000, ge=0)
    epochs: int = Field(default=5_000, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    batch_size: int = Field(default=512, ge=1)
    loss_weights: dict[str, float] = Field(
        default_factory=lambda: {"pde": 1.0, "bc": 10.0, "ic": 10.0, "data": 1.0}
    )
    custom_parameters: dict[str, Any] = Field(default_factory=dict)
    # Dataset upload (optional — used instead of generated data when provided)
    dataset_id: str | None = None
    column_mapping: ColumnMapping | None = None
    # Inverse problem configuration (only relevant when problem_type == 'inverse')
    inverse_config: InverseConfig | None = None


# ---------------------------------------------------------------------------
# Training & result schemas
# ---------------------------------------------------------------------------

class TrainingMetrics(BaseModel):
    """Snapshot of losses at a single training epoch."""
    epoch: int
    total_loss: float
    pde_loss: float
    bc_loss: float
    ic_loss: float
    data_loss: float
    val_loss: float


class ModelResult(BaseModel):
    """Training outcome for a single architecture."""
    architecture_id: str
    architecture_name: str
    training_time_s: float
    final_loss: float
    metrics: dict[str, float] = Field(
        description=(
            "Error metrics: l2_relative_error, l1_error, rmse, max_error, r2_score"
        )
    )
    converged: bool
    training_history: list[TrainingMetrics] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """Aggregated result of a completed arena experiment."""
    experiment_id: str
    problem_id: str
    geometry_id: str | None
    status: ExperimentStatus
    created_at: datetime
    completed_at: datetime | None
    model_results: list[ModelResult] = Field(default_factory=list)
    ranking: list[str] = Field(default_factory=list)
    best_model_id: str = ""
    comparison_summary: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API request / response envelopes
# ---------------------------------------------------------------------------

class ExperimentStatusResponse(BaseModel):
    """Polling response for an in-flight or completed experiment."""
    experiment_id: str
    status: ExperimentStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    current_model: str = ""
    message: str = ""
    result: ExperimentResult | None = None


class CreateExperimentRequest(BaseModel):
    """Request body for POST /experiments."""
    config: ExperimentConfig
