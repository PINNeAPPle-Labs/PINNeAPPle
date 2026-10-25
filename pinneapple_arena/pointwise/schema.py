"""Descriptors of the arena catalogs: physics problems, geometry presets and architectures (pydantic v2)."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core enumerations
# ---------------------------------------------------------------------------

class PhysicsArea(str, Enum):
    FLUID_MECHANICS = "FLUID_MECHANICS"
    SOLID_MECHANICS = "SOLID_MECHANICS"
    ELECTROMAGNETISM = "ELECTROMAGNETISM"
    QUANTUM_MECHANICS = "QUANTUM_MECHANICS"
    THERMODYNAMICS = "THERMODYNAMICS"
    WAVE_PHYSICS = "WAVE_PHYSICS"
    NONLINEAR_DYNAMICS = "NONLINEAR_DYNAMICS"


class Dimension(str, Enum):
    ONE_D = "ONE_D"
    TWO_D = "TWO_D"
    THREE_D = "THREE_D"


class DifficultyLevel(str, Enum):
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"


class ArchitectureFamily(str, Enum):
    PINN = "PINN"
    NEURAL_OPERATOR = "NEURAL_OPERATOR"
    TRANSFORMER = "TRANSFORMER"
    RECURRENT = "RECURRENT"
    CONVOLUTIONAL = "CONVOLUTIONAL"
    CONTINUOUS = "CONTINUOUS"
    GRAPH = "GRAPH"
    AUTOENCODER = "AUTOENCODER"
    RESERVOIR = "RESERVOIR"
    ROM = "ROM"
    CLASSICAL = "CLASSICAL"
    PHYSICS_AWARE = "PHYSICS_AWARE"


class GeometryType(str, Enum):
    BOX = "BOX"
    SPHERE = "SPHERE"
    CYLINDER = "CYLINDER"
    PIPE = "PIPE"
    CHANNEL = "CHANNEL"
    CAVITY = "CAVITY"
    L_BRACKET = "L_BRACKET"
    AIRFOIL = "AIRFOIL"
    ANNULUS = "ANNULUS"
    TORUS = "TORUS"


# ---------------------------------------------------------------------------
# Physics problem schemas
# ---------------------------------------------------------------------------

class ProblemParameter(BaseModel):
    """A configurable scalar parameter for a physics problem."""
    name: str
    default: float
    min_val: float
    max_val: float
    description: str
    unit: str = ""


class ColumnHints(BaseModel):
    """
    Describes the expected column layout for a dataset associated with this problem.
    Used to guide users when uploading CSV / NPZ / JSON files and to auto-populate
    the column mapping UI.
    """
    coord_cols: list[str] = Field(
        description="Spatial (and optionally temporal) coordinate columns",
    )
    output_cols: list[str] = Field(
        description="Solution / output field columns",
    )
    time_col: str | None = Field(
        default=None,
        description="Name of the time column if time-dependent",
    )
    csv_header: str = Field(
        description="Minimal CSV header example — comma-separated column names in suggested order",
    )
    notes: str = Field(
        default="",
        description="Extra guidance (e.g. complex-valued fields split as _re/_im)",
    )


class PhysicsProblem(BaseModel):
    """Full descriptor for a physics problem available in the arena."""
    id: str
    name: str
    description: str
    domain: PhysicsArea
    dimension: Dimension
    equations: list[str]
    applications: list[str]
    initial_conditions: str
    boundary_conditions: str
    parameters: list[ProblemParameter]
    difficulty: DifficultyLevel
    requires_geometry: bool = False
    pinneapple_preset: str | None = None
    tags: list[str] = Field(default_factory=list)
    latex_equations: list[str] | None = Field(
        default=None,
        description="Equations in LaTeX syntax for KaTeX rendering (optional; falls back to equations[])",
    )
    column_hints: ColumnHints | None = Field(
        default=None,
        description="Expected column layout for uploaded datasets",
    )


# ---------------------------------------------------------------------------
# Geometry schemas
# ---------------------------------------------------------------------------

class GeometryPreset(BaseModel):
    """Pre-built 3-D geometry available for 3-D experiments."""
    id: str
    name: str
    description: str
    geometry_type: GeometryType
    dimensions: dict[str, Any]
    supported_problems: list[str]
    mesh_resolution_default: int = 20
    image_placeholder: str = ""


class MeshInfo(BaseModel):
    """Summary statistics returned after meshing a geometry."""
    n_points: int
    n_elements: int
    bounds: dict[str, tuple[float, float]]
    quality_score: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Architecture schemas
# ---------------------------------------------------------------------------

class ArchitectureInfo(BaseModel):
    """Metadata for a neural architecture available in the arena."""
    id: str
    name: str
    family: ArchitectureFamily
    description: str
    detailed_description: str | None = Field(
        default=None,
        description="Extended description covering design choices, inductive biases, and known limitations",
    )
    supports_physics_loss: bool = False
    supports_3d: bool = True
    # True if this architecture is only meaningful for inverse problems
    # (e.g. inverse_pinn). These are hidden in forward experiments and
    # highlighted as recommended in inverse experiments.
    inverse_only: bool = False
    best_for: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    paper_reference: str | None = None
