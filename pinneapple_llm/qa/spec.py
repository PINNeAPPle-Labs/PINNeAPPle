"""Typed, serializable physical specification -- stage A of the quality pipeline.

The point of this module is to make "what was asked for" inspectable
*before* any code or twin is generated, and to keep the four kinds of
statement a physics answer quietly mixes together visibly separate:

``USER``        stated by the requester;
``SOURCE``      read from a cited reference (with ``source`` filled in);
``ASSUMPTION``  adopted by the system and marked as such;
``ESTIMATED``   inferred, never verified.

Nothing in this module invents a value to fill a gap. A field that is
absent stays absent, and :meth:`PhysicalSpecification.missing_information`
is what the pipeline reports as ``BLOCKED``/limitation instead of
silently defaulting it.

Invariants are *declarative* on purpose (``kind`` + ``params``), never
Python source to ``eval``: the specification may come from an LLM, and a
spec that can execute code is a security hole, not a feature. The
supported kinds are listed in :data:`INVARIANT_KINDS` and evaluated by
``pinneapple_llm.qa.validators.physics``.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .units import UnitError, are_compatible, parse_unit


class Provenance(str, Enum):
    USER = "user"
    SOURCE = "source"
    ASSUMPTION = "assumption"
    ESTIMATED = "estimated"


#: Invariant kinds the pipeline can actually evaluate. A kind not in this
#: set is rejected at spec-construction time rather than silently skipped.
INVARIANT_KINDS = frozenset({
    "nonnegative",          # every finite value of `target` >= 0
    "max",                  # every finite value of `target` <= params["limit"]
    "min",                  # every finite value of `target` >= params["limit"]
    "range",                # params["low"] <= value <= params["high"]
    "monotonic_nondecreasing",  # along the leading (time) axis of `target`
    "conservation",         # sum over `target` along params["axis"] is constant in time within params["rtol"]
    "unit_compatible",      # unit of `target` is dimensionally compatible with params["unit"]
    "finite",               # no NaN / inf in `target`
})

SEVERITIES = ("error", "warning")


class SpecError(ValueError):
    """Raised when a specification is structurally invalid."""


@dataclass(frozen=True)
class Quantity:
    """A number with a unit, a provenance tag, and (optionally) an uncertainty."""

    value: float
    unit: str = ""
    provenance: Provenance = Provenance.USER
    source: str = ""
    uncertainty: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.value, (int, float)):
            raise SpecError(f"quantity value must be a number, got {type(self.value).__name__}")
        if self.provenance == Provenance.SOURCE and not self.source:
            raise SpecError(
                "a SOURCE quantity must name its source -- an unsourced 'source' value is an "
                "unsourced claim, tag it ASSUMPTION or ESTIMATED instead"
            )
        if self.uncertainty is not None and self.uncertainty < 0:
            raise SpecError("uncertainty cannot be negative")
        parse_unit(self.unit)  # raises UnitError on an unknown unit

    def to_si(self) -> float:
        return parse_unit(self.unit).to_si(self.value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": float(self.value),
            "unit": self.unit,
            "provenance": self.provenance.value,
            "source": self.source,
            "uncertainty": self.uncertainty,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Quantity":
        return cls(
            value=d["value"], unit=d.get("unit", ""),
            provenance=Provenance(d.get("provenance", "user")),
            source=d.get("source", ""), uncertainty=d.get("uncertainty"),
        )


@dataclass(frozen=True)
class Invariant:
    """A physical constraint the artifact must satisfy. Declarative, not code."""

    name: str
    kind: str
    target: str
    params: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"
    description: str = ""

    def __post_init__(self):
        if self.kind not in INVARIANT_KINDS:
            raise SpecError(
                f"invariant '{self.name}': unknown kind '{self.kind}' -- supported kinds are "
                f"{sorted(INVARIANT_KINDS)}; arbitrary expressions are deliberately not supported"
            )
        if self.severity not in SEVERITIES:
            raise SpecError(f"invariant '{self.name}': severity must be one of {SEVERITIES}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "target": self.target,
            "params": dict(self.params), "severity": self.severity,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Invariant":
        return cls(
            name=d["name"], kind=d["kind"], target=d["target"],
            params=dict(d.get("params", {})), severity=d.get("severity", "error"),
            description=d.get("description", ""),
        )


@dataclass(frozen=True)
class Observation:
    """A sensor reading: where, what, when, and whether it may be used for validation."""

    sensor_id: str
    quantity: str
    unit: str
    position: Tuple[Quantity, ...]
    timestamps: Tuple[float, ...]
    time_unit: str
    values: Tuple[float, ...]
    provenance: Provenance = Provenance.USER
    holdout: bool = False  # reserved for validation, never for calibration

    def __post_init__(self):
        if len(self.values) != len(self.timestamps):
            raise SpecError(
                f"observation '{self.sensor_id}': {len(self.values)} value(s) but "
                f"{len(self.timestamps)} timestamp(s)"
            )
        for t in self.timestamps:
            if not isinstance(t, (int, float)):
                raise SpecError(f"observation '{self.sensor_id}': timestamps must be numbers")
        parse_unit(self.time_unit)
        for p in self.position:
            parse_unit(p.unit)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id, "quantity": self.quantity, "unit": self.unit,
            "position": [p.to_dict() for p in self.position],
            "timestamps": list(self.timestamps), "time_unit": self.time_unit,
            "values": list(self.values), "provenance": self.provenance.value,
            "holdout": self.holdout,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Observation":
        return cls(
            sensor_id=d["sensor_id"], quantity=d["quantity"], unit=d["unit"],
            position=tuple(Quantity.from_dict(p) for p in d.get("position", [])),
            timestamps=tuple(d.get("timestamps", [])), time_unit=d.get("time_unit", "s"),
            values=tuple(d.get("values", [])),
            provenance=Provenance(d.get("provenance", "user")), holdout=bool(d.get("holdout", False)),
        )


@dataclass(frozen=True)
class GeometrySpec:
    """Geometry, domain and coordinate system, with units made explicit."""

    kind: str
    coordinate_system: str = "cartesian"
    dimensions: Dict[str, Quantity] = field(default_factory=dict)
    hollow: bool = False
    wall_thickness: Optional[Quantity] = None
    notes: str = ""

    def __post_init__(self):
        if self.hollow and self.wall_thickness is None:
            raise SpecError(
                "geometry is declared hollow but has no wall_thickness -- a hollow body without a "
                "wall thickness is not a specification, it is a slogan"
            )
        if self.wall_thickness is not None:
            parse_unit(self.wall_thickness.unit)
        for q in self.dimensions.values():
            parse_unit(q.unit)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "coordinate_system": self.coordinate_system,
            "dimensions": {k: q.to_dict() for k, q in self.dimensions.items()},
            "hollow": self.hollow,
            "wall_thickness": None if self.wall_thickness is None else self.wall_thickness.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometrySpec":
        return cls(
            kind=d["kind"], coordinate_system=d.get("coordinate_system", "cartesian"),
            dimensions={k: Quantity.from_dict(q) for k, q in d.get("dimensions", {}).items()},
            hollow=bool(d.get("hollow", False)),
            wall_thickness=None if d.get("wall_thickness") is None else Quantity.from_dict(d["wall_thickness"]),
            notes=d.get("notes", ""),
        )


#: Fields the pipeline treats as essential for a *physics* artifact. Missing
#: ones do not raise: they are reported, so an honest "not specified" is
#: possible instead of a fabricated default.
ESSENTIAL_FIELDS = (
    "objective", "phenomenon", "geometry", "dependent_variables",
    "parameters", "governing_equations", "spatial_validity", "temporal_validity",
    "validation_metrics",
)


@dataclass
class PhysicalSpecification:
    """Stage A output: what the user asked for, expressed so it can be checked."""

    objective: str = ""
    phenomenon: str = ""
    geometry: Optional[GeometrySpec] = None
    independent_variables: List[str] = field(default_factory=list)
    dependent_variables: List[str] = field(default_factory=list)
    parameters: Dict[str, Quantity] = field(default_factory=dict)
    materials: Dict[str, Dict[str, Quantity]] = field(default_factory=dict)
    governing_equations: List[str] = field(default_factory=list)
    initial_conditions: List[str] = field(default_factory=list)
    boundary_conditions: List[str] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    spatial_validity: Dict[str, Tuple[Quantity, Quantity]] = field(default_factory=dict)
    temporal_validity: Optional[Tuple[Quantity, Quantity]] = None
    invariants: List[Invariant] = field(default_factory=list)
    numerical_method: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    missing_data: List[str] = field(default_factory=list)
    #: Names of the validators the spec itself requires for a Q2 rating. A
    #: validator absent from this list still runs if applicable; being listed
    #: here means its absence blocks the level (it can be NOT_APPLICABLE and
    #: still block, which is the point).
    required_validations: List[str] = field(default_factory=list)
    spec_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # -- derivation -------------------------------------------------------
    def missing_information(self) -> List[str]:
        """Essential fields that are absent or empty, plus declared gaps."""
        missing = list(self.missing_data)
        for name in ESSENTIAL_FIELDS:
            value = getattr(self, name)
            if value is None or (hasattr(value, "__len__") and len(value) == 0):
                missing.append(name)
        if not self.validation_metrics:
            missing.append("validation_metrics (no acceptance criteria declared)")
        return missing

    def criteria(self) -> Dict[str, Any]:
        """The acceptance criteria and invariants, as a hashable payload.

        The repair loop snapshots this before attempting corrections: a
        correction that "passes" because it relaxed a threshold, dropped an
        invariant or changed reference data is rejected rather than accepted.
        """
        return {
            "validation_metrics": self.validation_metrics,
            "invariants": [i.to_dict() for i in self.invariants],
            "required_validations": sorted(self.required_validations),
            "governing_equations": list(self.governing_equations),
        }

    def criteria_hash(self) -> str:
        return _hash_json(self.criteria())

    def content_hash(self) -> str:
        return _hash_json(self.to_dict())

    def assumptions_provenance_report(self) -> Dict[str, List[str]]:
        """Which claims are user-supplied, sourced, assumed or estimated."""
        out: Dict[str, List[str]] = {"user": [], "source": [], "assumption": [], "estimated": []}
        for name, q in sorted(self.parameters.items()):
            out[q.provenance.value].append(name)
        for name, material in sorted(self.materials.items()):
            for prop, q in sorted(material.items()):
                out[q.provenance.value].append(f"{name}.{prop}")
        for obs in self.observations:
            out[obs.provenance.value].append(f"sensor:{obs.sensor_id}")
        return out

    def validate(self) -> List[str]:
        """Structural problems with this spec (empty list means well-formed)."""
        problems: List[str] = []
        names = [i.name for i in self.invariants]
        if len(set(names)) != len(names):
            problems.append("duplicate invariant names")
        problems.extend(self._check_boundary_conditions())
        for inv in self.invariants:
            if inv.kind == "unit_compatible" and "unit" not in inv.params:
                problems.append(f"invariant '{inv.name}': unit_compatible needs params['unit']")
            if inv.kind in ("max", "min", "range") and not (
                {"limit"} & set(inv.params) or {"low", "high"} <= set(inv.params)
            ):
                problems.append(f"invariant '{inv.name}': kind '{inv.kind}' needs a limit")
            if "unit" in inv.params:
                try:
                    parse_unit(inv.params["unit"])
                except UnitError as e:
                    problems.append(f"invariant '{inv.name}': {e}")
        for name, (lo, hi) in self.spatial_validity.items():
            if not are_compatible(lo.unit, hi.unit):
                problems.append(f"spatial_validity['{name}']: bounds have incompatible units")
            elif lo.to_si() > hi.to_si():
                problems.append(f"spatial_validity['{name}']: lower bound above upper bound")
        if self.temporal_validity is not None:
            lo, hi = self.temporal_validity
            if not are_compatible(lo.unit, hi.unit):
                problems.append("temporal_validity: bounds have incompatible units")
            elif lo.to_si() >= hi.to_si():
                problems.append("temporal_validity: start must precede end")
        if self.geometry is not None and self.geometry.hollow:
            wt = self.geometry.wall_thickness
            thickness = self.geometry.dimensions.get("thickness") or self.geometry.dimensions.get("wall_thickness")
            if thickness is not None and wt is not None and are_compatible(thickness.unit, wt.unit):
                if abs(thickness.to_si() - wt.to_si()) > 1e-9 * max(1.0, abs(thickness.to_si())):
                    problems.append(
                        "geometry declares both a 'thickness' dimension and a wall_thickness that "
                        "disagree -- one object cannot have two wall thicknesses"
                    )
        length_units = [
            q.unit for q in (self.geometry.dimensions.values() if self.geometry else [])
            if _is_length(q.unit)
        ]
        for obs in self.observations:
            for p in obs.position:
                if length_units and not _is_length(p.unit):
                    problems.append(
                        f"observation '{obs.sensor_id}': position unit '{p.unit}' is not a length, "
                        f"but the geometry is in '{length_units[0]}'"
                    )
        return problems

    # -- serialization ----------------------------------------------------
    def _check_boundary_conditions(self) -> List[str]:
        """Narrow, honest check of declared boundary conditions.

        Boundary conditions are free text on purpose -- a parser strong enough
        to understand arbitrary physical prose is an LLM, and asking one to
        grade another's prose is exactly what this pipeline exists to avoid.
        What is checkable cheaply and worth checking: a condition written as
        ``"<variable> = <number> <unit>"`` must name a variable the spec
        declares, and its unit must be compatible with that variable's own
        declared unit when one exists.
        """
        problems: List[str] = []
        known = set(self.dependent_variables) | set(self.independent_variables)
        for bc in self.boundary_conditions:
            text = str(bc).strip()
            if "=" not in text:
                continue  # prose; nothing to check structurally
            lhs, rhs = text.split("=", 1)
            variable = lhs.split(":")[-1].strip()
            if known and variable and variable not in known:
                problems.append(
                    f"boundary condition '{text}' refers to '{variable}', which is not one of the "
                    f"declared variables {sorted(known)}")
                continue
            parts = rhs.strip().split()
            if not parts:
                continue
            try:
                float(parts[0])
            except ValueError:
                continue
            unit = " ".join(parts[1:]) or None
            if unit:
                try:
                    parse_unit(unit)
                except UnitError as exc:
                    problems.append(f"boundary condition '{text}': {exc}")
                    continue
                declared = self.parameters.get(variable)
                if declared is not None and not are_compatible(unit, declared.unit):
                    problems.append(
                        f"boundary condition '{text}' is in '{unit}' but '{variable}' is declared "
                        f"in '{declared.unit}'")
        return problems

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "objective": self.objective,
            "phenomenon": self.phenomenon,
            "geometry": None if self.geometry is None else self.geometry.to_dict(),
            "independent_variables": list(self.independent_variables),
            "dependent_variables": list(self.dependent_variables),
            "parameters": {k: q.to_dict() for k, q in self.parameters.items()},
            "materials": {m: {k: q.to_dict() for k, q in props.items()} for m, props in self.materials.items()},
            "governing_equations": list(self.governing_equations),
            "initial_conditions": list(self.initial_conditions),
            "boundary_conditions": list(self.boundary_conditions),
            "observations": [o.to_dict() for o in self.observations],
            "spatial_validity": {
                k: [lo.to_dict(), hi.to_dict()] for k, (lo, hi) in self.spatial_validity.items()
            },
            "temporal_validity": None if self.temporal_validity is None
            else [self.temporal_validity[0].to_dict(), self.temporal_validity[1].to_dict()],
            "invariants": [i.to_dict() for i in self.invariants],
            "numerical_method": dict(self.numerical_method),
            "validation_metrics": dict(self.validation_metrics),
            "assumptions": list(self.assumptions),
            "missing_data": list(self.missing_data),
            "required_validations": list(self.required_validations),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhysicalSpecification":
        tv = d.get("temporal_validity")
        return cls(
            spec_id=d.get("spec_id") or uuid.uuid4().hex,
            objective=d.get("objective", ""),
            phenomenon=d.get("phenomenon", ""),
            geometry=None if d.get("geometry") is None else GeometrySpec.from_dict(d["geometry"]),
            independent_variables=list(d.get("independent_variables", [])),
            dependent_variables=list(d.get("dependent_variables", [])),
            parameters={k: Quantity.from_dict(v) for k, v in d.get("parameters", {}).items()},
            materials={m: {k: Quantity.from_dict(v) for k, v in props.items()}
                      for m, props in d.get("materials", {}).items()},
            governing_equations=list(d.get("governing_equations", [])),
            initial_conditions=list(d.get("initial_conditions", [])),
            boundary_conditions=list(d.get("boundary_conditions", [])),
            observations=[Observation.from_dict(o) for o in d.get("observations", [])],
            spatial_validity={
                k: (Quantity.from_dict(v[0]), Quantity.from_dict(v[1]))
                for k, v in d.get("spatial_validity", {}).items()
            },
            temporal_validity=None if tv is None else (Quantity.from_dict(tv[0]), Quantity.from_dict(tv[1])),
            invariants=[Invariant.from_dict(i) for i in d.get("invariants", [])],
            numerical_method=dict(d.get("numerical_method", {})),
            validation_metrics=dict(d.get("validation_metrics", {})),
            assumptions=list(d.get("assumptions", [])),
            missing_data=list(d.get("missing_data", [])),
            required_validations=list(d.get("required_validations", [])),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _is_length(unit: str) -> bool:
    return parse_unit(unit).dim == (1, 0, 0, 0, 0, 0, 0)


def _hash_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_default(obj: Any):
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def quantity_map(**kwargs: Any) -> Dict[str, Quantity]:
    """``quantity_map(nu=(1e-6, "m^2/s"))`` -- small construction helper."""
    out: Dict[str, Quantity] = {}
    for name, spec in kwargs.items():
        value, unit = spec if isinstance(spec, tuple) else (spec, "")
        out[name] = Quantity(float(value), unit)
    return out
