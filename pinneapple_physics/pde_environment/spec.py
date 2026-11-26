from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .environment_typing import CoordNames, FieldNames
from .conditions import ConditionSpec
from .scales import ScaleSpec


@dataclass(frozen=True)
class PDETermSpec:
    """
    PDE descriptor used by the compiler.

    kind: name of PDE (e.g. laplace, poisson, burgers, navier_stokes_incompressible)
    fields: model output fields involved
    coords: coordinate names used by the PDE (e.g. ("x","y","t"))
    params: PDE parameters (e.g. nu, Re, kappa, alpha)
    meta: extra options (e.g. darcy mode)
    """
    kind: str
    fields: FieldNames
    coords: CoordNames
    params: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProblemSpec:
    """
    Complete problem specification.

    conditions: boundary/initial/data constraints
    sample_defaults: recommended sampling counts
    """
    name: str
    dim: int
    coords: CoordNames
    fields: FieldNames
    pde: PDETermSpec
    conditions: tuple[ConditionSpec, ...] = field(default_factory=tuple)

    sample_defaults: dict[str, int] = field(default_factory=dict)
    scales: ScaleSpec = field(default_factory=ScaleSpec)
    field_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    references: tuple[str, ...] = field(default_factory=tuple)
    domain_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    solver_spec: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    problem_id: str = ""  # alias for name used by quickstart / arena

    def to_system(self) -> Any:
        """The physics ontology view of this problem (#204): domain, fields, parameters, equation, conditions, solver.

        A ``ProblemSpec`` is what the compiler and the presets speak, and it keeps its shape. What changed is that the
        *concepts* it declares -- a field, a parameter, a governing equation, a boundary condition -- are now the
        ontology's types, so a dataset manifest or a model card describing the same case produces the same objects.
        Units come from ``meta["units"]`` when the problem carries them; without it they are recorded as ``"1"``
        (dimensionless) rather than guessed from the field name.
        """
        from pinneapple_core.ontology import PhysicalSystem

        return PhysicalSystem.from_problem_spec(self)
