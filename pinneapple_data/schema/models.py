"""The ``pinneapple.physics-data/1`` models: geometry, mesh, physics, fields, provenance and verification.

Everything here is a pydantic model with ``extra="forbid"`` (a misspelled key is an error, not a silently
dropped field) and JSON round-trip. Arrays are never embedded: a model holds an :class:`ArrayRef` -- a key, a
shape and a dtype -- and the arrays themselves live in :class:`~pinneapple_data.schema.bundle.PhysicsDataBundle`.
That keeps the manifest small, diffable and hashed on its own, and stops a 2 GB field from ending up in a JSON
log.

The five rules the validators enforce are the ones that go wrong in real datasets
------------------------------------------------------------------------------
1. a field with a physical meaning declares a unit that parses, or says ``"1"``;
2. a field declares ``nodal`` or ``cell`` and its shape has to agree with the mesh (``(nt, n_nodes)`` for a
   nodal field over ``n_nodes`` nodes, ``(nt, n_cells)`` for a cell field over ``n_cells`` cells);
3. every axis a field indexes is declared in :attr:`Mesh.axes`, with its coordinates and its unit, so the frame
   is recorded rather than assumed;
4. provenance names a source, a licence and the SHA-256 of every input file it was built from;
5. a verification check carries its tolerance and its result; ``passed=None`` means "not run", which is not a
   pass.

Coordinates and units
---------------------
Axis coordinates carry the unit (``x`` in ``m``, ``t`` in ``s``, ``lambda`` in ``1``); ``Geometry.units`` is the
length unit of the *geometry*, kept separate because a CAD file can be in ``mm`` while its simulation runs in
``m`` -- the exact mismatch that silently scales a drag coefficient by ``1e-3``.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic import Field as PField

from ..physical_units import UnitError, parse_unit

SCHEMA_VERSION = "1.0"
DIMENSIONLESS = frozenset({"1", "-", "", "dimensionless", "[-]"})

Location = Literal["nodal", "cell"]
Kind = Literal["grid", "unstructured", "point-cloud", "none"]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def unit_or_dimensionless(text: str) -> str:
    """Return ``text`` if it parses as a unit or means dimensionless; raise otherwise.

    ``"1"`` and ``"-"`` are the two spellings of dimensionless that appear in practice (NASA CFL3D and
    OpenFOAM respectively); both are normalised to ``"1"`` so a bundle has exactly one of them."""
    raw = (text or "").strip()
    if raw.lower() in DIMENSIONLESS:
        return "1"
    try:
        parse_unit(raw)
    except UnitError as exc:
        raise ValueError(f"'{text}' is not a unit this library knows ({exc}); use \"1\" for dimensionless") from exc
    return raw


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FileRef(_Model):
    """A file a bundle was built from, with the SHA-256 of its bytes.

    The hash is what makes "this bundle came from that download" checkable later; a path alone is not, because
    files get replaced in place (a re-downloaded PDEBench case keeps its name and changes its content)."""

    path: str = PField(min_length=1)
    sha256: str = ""
    bytes: int = 0
    role: str = "input"

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if v and (len(v) != 64 or any(c not in "0123456789abcdef" for c in v.lower())):
            raise ValueError("sha256 must be 64 hex characters (or empty when unknown)")
        return v.lower()


class ArrayRef(_Model):
    """Where an array lives inside its bundle: a key, its shape and its dtype.

    ``path`` is set only when the array is stored as its own ``.npy`` file next to the bundle instead of in
    ``arrays.npz`` (the case for a field too big to load at once)."""

    key: str = PField(min_length=1)
    shape: tuple[int, ...]
    dtype: str
    path: str | None = None

    @field_validator("shape")
    @classmethod
    def _shape(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("shape must have at least one dimension")
        if any(int(n) < 0 for n in v):
            raise ValueError("shape entries must be non-negative")
        return tuple(int(n) for n in v)


class AxisSpec(_Model):
    """One axis of a structured dataset: its name, its coordinates and the unit of those coordinates.

    For ``name="t"`` the values are physical times; for a spatial axis they are positions. The unit belongs to
    the coordinate array, so ``x`` in ``mm`` is a different dataset from ``x`` in ``m`` and says so."""

    name: str = PField(min_length=1)
    unit: str = "1"
    data: ArrayRef | None = None
    description: str = ""

    def __init__(self, name: str, unit: str = "1", data: ArrayRef | None = None, description: str = "", **kw: Any):
        """``AxisSpec("x", "m")`` -- name then unit, the order every adapter reads them in."""
        super().__init__(name=name, unit=unit, data=data, description=description, **kw)

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))


class Mesh(_Model):
    """The discretisation a field lives on: a structured grid (axis by axis) or an unstructured one.

    ``kind="grid"`` uses :attr:`axes` and needs no connectivity; ``kind="unstructured"`` uses :attr:`nodes`
    ``(n_nodes, dim)`` and :attr:`cells` ``(n_cells, nodes_per_cell)`` plus :attr:`cell_types`; ``kind="none"``
    is a dataset of independent samples (a table of parametric results) that still wants a unit-bearing
    contract.
    """

    kind: Kind = "grid"
    dimension: int = PField(default=1, ge=0, le=3)
    axes: list[AxisSpec] = PField(default_factory=list)
    nodes: ArrayRef | None = None
    cells: ArrayRef | None = None
    cell_types: list[str] = PField(default_factory=list)
    physical_groups: dict[str, list[str]] = PField(default_factory=dict)
    quality: dict[str, float] = PField(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> Mesh:
        if self.kind == "grid" and not self.axes:
            raise ValueError("a grid mesh declares at least one axis (with its coordinates)")
        if self.kind == "grid" and len({a.name for a in self.axes}) != len(self.axes):
            raise ValueError("axis names must be unique")
        if self.kind == "unstructured" and (self.nodes is None or self.cells is None):
            raise ValueError("an unstructured mesh needs nodes and cells")
        if self.kind == "unstructured" and self.cell_types and len(self.cell_types) != 1:
            # a mixed-element mesh is allowed, but then every cell has to say which type it is
            raise ValueError("a non-empty cell_types must have exactly one entry (homogeneous mesh); "
                             "per-cell types belong in the bundle's metadata")
        return self

    @property
    def axis_names(self) -> list[str]:
        return [a.name for a in self.axes]

    def axis(self, name: str) -> AxisSpec:
        for a in self.axes:
            if a.name == name:
                return a
        raise KeyError(f"axis '{name}' not in {self.axis_names}")

    @property
    def n_nodes(self) -> int | None:
        """Nodes of the mesh: rows of ``nodes``, or the product of the spatial axes of a grid (time excluded)."""
        if self.nodes is not None:
            return int(self.nodes.shape[0])
        spatial = [a for a in self.axes if a.name != "t"]
        if not spatial or any(a.data is None for a in spatial):
            return None
        return int(math.prod(a.data.shape[0] for a in spatial))  # type: ignore[union-attr]

    @property
    def n_cells(self) -> int | None:
        if self.cells is not None:
            return int(self.cells.shape[0])
        return None


class Region(_Model):
    """A named part of the geometry or the mesh (a wall, an inlet, a solid zone)."""

    name: str = PField(min_length=1)
    kind: Literal["volume", "surface", "boundary", "inlet", "outlet", "wall", "symmetry", "zone"] = "boundary"
    selector: str = ""
    description: str = ""


class Geometry(_Model):
    """The physical domain: what it is, in what units, and the files that describe it.

    ``units`` is the length unit of the geometry files themselves. A STEP part in ``mm`` solved in ``m``
    records ``units="mm"`` here and ``unit="m"`` on the axes: the conversion is then visible instead of lost.
    """

    kind: Literal["cad", "mesh", "structured-grid", "analytic", "none"] = "structured-grid"
    dimension: int = PField(default=1, ge=0, le=3)
    units: str = "m"
    files: list[FileRef] = PField(default_factory=list)
    regions: list[Region] = PField(default_factory=list)
    bounding_box: tuple[float, ...] | None = None
    description: str = ""

    _units = field_validator("units")(lambda cls, v: unit_or_dimensionless(v))

    @model_validator(mode="after")
    def _bbox(self) -> Geometry:
        if self.bounding_box is not None and len(self.bounding_box) not in (2 * self.dimension, 0):
            raise ValueError(f"bounding_box has {len(self.bounding_box)} entries; expected {2 * self.dimension} "
                             "(min_x, min_y, ..., max_x, max_y, ...)")
        return self

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        if not self.bounding_box:
            return {}
        d = self.dimension
        return {f"d{i}": (self.bounding_box[i], self.bounding_box[i + d]) for i in range(d)}


class Parameter(_Model):
    """A physical parameter of the case (a viscosity, a conductivity), with its unit and where it came from."""

    name: str = PField(min_length=1)
    value: float | int | None = None
    unit: str = "1"
    source: str = ""
    description: str = ""

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))


class BoundaryCondition(_Model):
    """One boundary condition: which field, on which region, of which kind, with what value."""

    field: str = PField(min_length=1)
    region: str = PField(min_length=1)
    kind: Literal["dirichlet", "neumann", "robin", "periodic", "wall", "inlet", "outlet", "symmetry", "open"] = (
        "dirichlet")
    value: float | int | Sequence[float] | None = None
    unit: str = "1"
    description: str = ""

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))


class InitialCondition(_Model):
    """The state a transient case starts from."""

    field: str = PField(min_length=1)
    value: float | int | Sequence[float] | None = None
    unit: str = "1"
    description: str = ""

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))


class Material(_Model):
    """A named material with its properties and their units."""

    name: str = PField(min_length=1)
    properties: dict[str, float] = PField(default_factory=dict)
    units: dict[str, str] = PField(default_factory=dict)

    @field_validator("units")
    @classmethod
    def _units(cls, v: dict[str, str]) -> dict[str, str]:
        return {k: unit_or_dimensionless(u) for k, u in v.items()}


class Physics(_Model):
    """What equation the data satisfies, with what materials, boundary and initial conditions.

    ``governing`` is the equation in plain text with its symbols defined in :attr:`symbols`; the schema does not
    pretend to parse PDEs (that is what ``pinneapple_physics.pde_environment`` does), it records the statement so
    a residual check downstream knows what to check against."""

    governing: str = ""
    symbols: dict[str, str] = PField(default_factory=dict)
    boundary_conditions: list[BoundaryCondition] = PField(default_factory=list)
    initial_conditions: list[InitialCondition] = PField(default_factory=list)
    parameters: list[Parameter] = PField(default_factory=list)
    materials: list[Material] = PField(default_factory=list)
    dimensionless: dict[str, float] = PField(default_factory=dict)
    notes: str = ""

    def parameter(self, name: str) -> Parameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"parameter '{name}' not recorded")


class Field(_Model):
    """One variable of the dataset: what it is called, its unit, and where its values live.

    ``location`` is ``"nodal"`` (one value per node of the mesh) or ``"cell"`` (one value per cell centre) and
    is not cosmetic: comparing a cell-centred pressure with a nodal reference is a half-cell shift that shows up
    as a first-order error. ``components`` names the components of a vector field (``["u", "v"]``), and the last
    axis of the array indexes them.
    """

    name: str = PField(min_length=1)
    symbol: str = ""
    unit: str = "1"
    location: Location = "nodal"
    components: list[str] = PField(default_factory=list)
    axes: list[str] = PField(default_factory=list)
    data: ArrayRef
    description: str = ""

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))

    @classmethod
    def from_array(cls, name: str, array: np.ndarray, *, unit: str = "1", location: Location = "nodal",
                   axes: Sequence[str] | None = None, components: Sequence[str] | None = None,
                   symbol: str = "", description: str = "") -> Field:
        """Build a field (and the reference its array will get in the bundle) from a numpy array."""
        arr = np.asarray(array)
        key = f"field:{name}"
        return cls(name=name, symbol=symbol, unit=unit, location=location,
                   components=list(components or []), axes=list(axes or []),
                   data=ArrayRef(key=key, shape=tuple(int(n) for n in arr.shape), dtype=str(arr.dtype)),
                   description=description)

    @property
    def is_vector(self) -> bool:
        return bool(self.components) and len(self.components) > 1


class Fields(_Model):
    """Every variable in the dataset, plus the axis names they index."""

    variables: list[Field] = PField(default_factory=list)

    def names(self) -> list[str]:
        return [f.name for f in self.variables]

    def get(self, name: str) -> Field:
        for f in self.variables:
            if f.name == name:
                return f
        raise KeyError(f"field '{name}' not in {self.names()}")

    def __contains__(self, name: str) -> bool:
        return name in self.names()


class Provenance(_Model):
    """Where the data came from, under what licence, and what it was computed with.

    The licence is a required, non-empty field on purpose: a bundle whose licence is unknown cannot be
    distributed, and the place to find that out is here, not in a review email."""

    source: str = PField(min_length=1)
    kind: Literal["dataset", "simulation", "experiment", "analytical", "synthetic"] = "dataset"
    adapter: str = ""
    adapter_version: str = ""
    solver: str = ""
    solver_version: str = ""
    config: dict[str, Any] = PField(default_factory=dict)
    inputs: list[FileRef] = PField(default_factory=list)
    license: str = PField(min_length=1)
    citation: str = ""
    url: str = ""
    created_at: str = PField(default_factory=_now)
    attributes: dict[str, Any] = PField(default_factory=dict)

    @field_validator("config", "attributes", mode="before")
    @classmethod
    def _jsonable(cls, v: Any) -> dict[str, Any]:
        return _json_safe(v) if isinstance(v, Mapping) else {}


class CheckResult(_Model):
    """One verification check and its outcome. ``passed=None`` is "not run", which is not a pass."""

    check_id: str = PField(min_length=1)
    tool: str = ""
    passed: bool | None = None
    value: float | None = None
    reference: float | None = None
    tolerance: float | None = None
    unit: str = "1"
    evidence_sha256: str = ""
    detail: str = ""

    _unit = field_validator("unit")(lambda cls, v: unit_or_dimensionless(v))

    @property
    def status(self) -> str:
        return "not run" if self.passed is None else ("passed" if self.passed else "failed")


class Verification(_Model):
    """The checks a bundle claims were run against it, and the gates it was required to pass."""

    checks: list[CheckResult] = PField(default_factory=list)
    required: list[str] = PField(default_factory=list)

    def get(self, check_id: str) -> CheckResult | None:
        for c in self.checks:
            if c.check_id == check_id:
                return c
        return None

    def status(self, check_id: str) -> str:
        c = self.get(check_id)
        return c.status if c is not None else "not run"

    @property
    def passed(self) -> bool:
        """True only when every required check ran and passed; an unrequired missing check never fails it."""
        return all(self.status(c) == "passed" for c in self.required) if self.required else all(
            c.passed is True for c in self.checks)

    @property
    def missing(self) -> list[str]:
        return [c for c in self.required if self.status(c) != "passed"]


class BundleManifest(_Model):
    """The serialisable description of a physics dataset: what it is, in what units, from where, checked how.

    Saved as ``bundle.json`` by :meth:`PhysicsDataBundle.save`; :attr:`digest` is then filled with the SHA-256 of
    the manifest and of every array, so a file edited after the fact no longer matches."""

    kind: Literal["physics-data"] = "physics-data"
    schema_version: str = SCHEMA_VERSION
    id: str = PField(min_length=1)
    title: str = ""
    description: str = ""
    geometry: Geometry = PField(default_factory=Geometry)
    mesh: Mesh = PField(default_factory=Mesh)
    physics: Physics = PField(default_factory=Physics)
    fields: Fields = PField(default_factory=Fields)
    provenance: Provenance
    verification: Verification = PField(default_factory=Verification)
    metadata: dict[str, Any] = PField(default_factory=dict)
    digest: str = ""

    MODEL: ClassVar[str] = "pinneapple.physics-data"

    @field_validator("metadata", mode="before")
    @classmethod
    def _meta(cls, v: Any) -> dict[str, Any]:
        return _json_safe(v) if isinstance(v, Mapping) else {}

    @model_validator(mode="after")
    def _fields_on_mesh(self) -> BundleManifest:
        """A field's shape has to be the shapes of the axes it names (plus one component axis if it is a vector)."""
        if self.mesh.kind != "grid":
            return self
        axes = set(self.mesh.axis_names)
        for f in self.fields.variables:
            unknown = [a for a in f.axes if a not in axes]
            if unknown:
                raise ValueError(f"field '{f.name}' indexes unknown axes {unknown}; the mesh declares {sorted(axes)}")
            if not f.axes:
                continue
            expected = []
            for name in f.axes:
                spec = self.mesh.axis(name)
                if spec.data is None:
                    raise ValueError(f"axis '{name}' has no coordinates, so field '{f.name}' cannot be laid out on it")
                expected.append(int(spec.data.shape[0]))
            body = tuple(f.data.shape[:-1]) if f.is_vector else tuple(f.data.shape)
            if body != tuple(expected):
                raise ValueError(
                    f"field '{f.name}' has shape {tuple(f.data.shape)} but axes {list(f.axes)} have shapes "
                    f"{tuple(expected)}{' (+1 component axis)' if f.is_vector else ''}")
        return self

    @property
    def axes(self) -> list[str]:
        return self.mesh.axis_names

    def json_schema_dict(self) -> dict[str, Any]:
        """JSON Schema (draft 2020-12) of the manifest, for other languages and the REST APIs."""
        return self.model_json_schema()

    def canonical_json(self, *, include_digest: bool = False) -> str:
        data = self.model_dump(mode="json", exclude={"digest"} if not include_digest else None)
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def model_digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    """JSON-safe copy of a mapping: numpy scalars/arrays to plain values, non-finite floats to strings."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def axis_names_of(manifest: BundleManifest) -> Iterable[str]:
    return (a.name for a in manifest.mesh.axes)
