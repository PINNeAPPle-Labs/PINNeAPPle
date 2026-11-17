"""Capsule forebody aeroheating (Bijective Systems): one wall, two fidelities, into the universal schema.

Laminar, perfect-gas heat flux and pressure on the heat shield of a parametric entry capsule. 1.024 capsules carry a
solution of an engineering model -- a modified-Newtonian panel method with Fay-Riddell stagnation heating and a Lees
laminar distribution, ``fidelity == "newtonian_panel"`` -- and 64 of them also carry a second-order laminar
Navier-Stokes solution from Eilmer, ``fidelity == "eilmer_second_order"``. Diameter runs from 0.8 m to 5.5 m, Mach
from 6 to 15 and altitude from 40 km to 80 km; the wall is isothermal at 300 K, the flow is at zero incidence, and
the gas is calorically perfect air.

Two properties of the file are what make it worth an adapter rather than a download link:

* **one wall for every capsule.** Every member's wall has 2.361 nodes and 4.640 triangles, and node *i* is the same
  point of the heat shield on every capsule and at both fidelities. The coordinates differ per capsule (metres, at
  real size); the *correspondence* does not. The difference between the two fidelities is therefore a subtraction at
  each node, with no interpolation between meshes -- which is what makes a multi-fidelity study of it meaningful,
  and is not true of any other public CFD corpus this library reads.
* **the file states its own units.** Every parameter, scalar and node field carries one (``W m-2``, ``kg m-2 s-1``,
  ``J kg-1``), and every parameter carries the bounds of its design space. Nothing here is guessed:
  :func:`schema_unit` translates the spelling, and those bounds are what :func:`field_channels` scales by.

The two fidelities are *runs*, not channels: each member is one shape at one flight condition, and the pair of runs
that share a member is what :func:`pair_fidelities` returns. Reading it the way the dataset is meant to be read:

    >>> from pinneapple_data.datasets import capsule_aeroheating as cap
    >>> path = cap.download("wall")                                         # 77 MB, the wall file
    >>> with cap.open_study(path) as study:
    ...     high, low = cap.pair_fidelities(study)                          # 64 rows each, same members
    ...     q_high = study.get_field("heat_flux", rows=[int(high[0])])[0]   # (2361,) W/m^2
    ...     q_low = study.get_field("heat_flux", rows=[int(low[0])])[0]     # the same nodes, the cheap model
    ...     one = cap.wall(study, int(high[0]))                             # points, triangles, every nodal field
    >>> bundle = cap.load_bundle(path, member="capsule_0066")               # one run as a PhysicsDataBundle

The cheap model is not a bad predictor: over the 64 cases its heat flux is 18.3 % off the Navier-Stokes one in
relative L2 (10.5 % to 29.7 %), its heat load 1.0 % to 15.4 % low, and its shoulder peak 11 % to 43 % low. That is
the number a surrogate trained on the expensive cases has to beat, and the reason :func:`field_samples` offers the
cheap field to the model as a channel instead of letting it meet the expensive field alone.
"""
from __future__ import annotations

import os
import re
import shutil
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["HUGGINGFACE", "RESOLVE_URL", "LICENSE", "CITATION", "FILES", "LOW_FIDELITY", "HIGH_FIDELITY", "FIDELITIES",
           "PARAMETERS", "NODE_FIELDS", "SCALARS", "Wall", "download", "open_study", "schema_unit", "pair_fidelities",
           "wall", "load_bundle", "load_table_bundle", "field_channels", "field_samples"]

HUGGINGFACE = "bijective/capsule-aeroheating"
RESOLVE_URL = f"https://huggingface.co/datasets/{HUGGINGFACE}/resolve/main"

#: The dataset card's licence, which covers the data and the figures. The reader (``bijective-gamut``) is Apache-2.0
#: and is a separate work, so the two do not mix here.
LICENSE = "CC-BY-4.0"

CITATION = ("Bijective Systems (2026), Capsule forebody aeroheating, Newtonian and Navier--Stokes CFD, version 4, "
            "https://huggingface.co/datasets/bijective/capsule-aeroheating")

#: The files the dataset publishes, and what each one is for. ``wall`` is the one the readers here use; the volume
#: file holds the flow field of the 64 CFD cases and is 470 MB.
FILES: dict[str, str] = {
    "wall": "vehicle_capsule_multifidelity.gamut",
    "wall_quad": "vehicle_capsule_multifidelity_quad.gamut",
    "volume": "vehicle_capsule_eilmer_volume.gamut",
    "newtonian": "vehicle_capsule_newtonian.gamut",
}

LOW_FIDELITY = "newtonian_panel"
HIGH_FIDELITY = "eilmer_second_order"
FIDELITIES: dict[str, str] = {
    LOW_FIDELITY: "modified Newtonian pressure, Fay-Riddell stagnation heating at the nose radius' velocity gradient "
                  "and a Lees laminar distribution elsewhere: seconds per run, and blind to how the shoulder "
                  "changes the flow over the face",
    HIGH_FIDELITY: "Eilmer, laminar Navier-Stokes, second order, on a grid fitted to the bow shock: 1.2 h to 5.4 h "
                   "per run on six cores (median 1.9 h), so only 64 of the 1.024 capsules have one",
}

#: The six inputs. ``nose_radius`` and ``shoulder_radius`` are fractions of the diameter, not lengths -- the file says
#: so in their units (``1``), and it is the easiest thing in the dataset to misread as metres.
PARAMETERS = ("diameter", "nose_radius", "forebody_angle", "shoulder_radius", "mach", "altitude")

#: The nodal fields. ``film_coefficient`` and ``recovery_enthalpy``, with the ``edge_*`` state behind them, are what
#: lets a hotter wall be estimated: ``q(T_w) ~= C_H (h_r - c_p T_w)``.
NODE_FIELDS = ("heat_flux", "pressure", "arc_length", "film_coefficient", "recovery_enthalpy", "edge_temperature",
               "edge_enthalpy", "edge_velocity", "edge_density", "edge_mach_number")

#: The per-run results.
SCALARS = ("heat_load", "stagnation_heat_flux", "peak_heat_flux", "mean_heat_flux", "shoulder_heat_flux",
           "stagnation_pressure", "forebody_pressure_drag_coefficient", "freestream_density", "freestream_pressure",
           "freestream_temperature", "freestream_velocity", "total_enthalpy", "peak_heat_flux_arc_length",
           "nose_radius_fraction")

_GROUPS = ("parameters", "labels", "scalars", "constants", "node_fields")

#: VTK's element type numbers, as ``Study.get_cells`` reports them, mapped to the schema's block type and node count.
_CELL_TYPES = {5: ("triangle", 3), 9: ("quad", 4), 12: ("hexahedron", 8), 10: ("tetra", 4)}


# ---------------------------------------------------------------------------
# downloading and opening
# ---------------------------------------------------------------------------

def download(what: str = "wall", dest: str | os.PathLike[str] = "data/capsule_aeroheating", *,
             revision: str | None = None, timeout: float = 600.0, force: bool = False) -> Path:
    """Fetch one of the dataset's files into ``dest`` and return its path.

    ``what`` is a key of :data:`FILES`. ``revision`` picks a tagged version (``"v3"``); without it the file is the
    current one. A file already on disk is kept unless ``force`` is set -- these are 77 MB to 470 MB."""
    if what not in FILES:
        raise KeyError(f"'{what}' is not one of {sorted(FILES)}")
    target = Path(dest)
    target.mkdir(parents=True, exist_ok=True)
    local = target / FILES[what]
    if local.exists() and not force:
        return local
    url = f"{RESOLVE_URL}/{FILES[what]}" + (f"?revision={revision}" if revision else "")
    with urllib.request.urlopen(url, timeout=timeout) as response, open(local, "wb") as handle:
        shutil.copyfileobj(response, handle)            # the flow-field file is 470 MB: never held in memory
    return local


def open_study(path: str | os.PathLike[str]):
    """Open one of the dataset's ``.gamut`` files, refusing a file that is not this dataset's.

    The check is on the groups and the labels the readers below rely on, not on the file name: a version of the
    dataset that drops one of them should say so here rather than raise a ``KeyError`` three functions later.
    Reading needs ``bijective-gamut`` (``pip install bijective-gamut``), imported here so the rest of the library
    works without it."""
    import gamut

    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"{file} is not there; fetch it with download() from {RESOLVE_URL}")
    study = gamut.open(str(file))
    try:
        for group in ("parameters", "labels"):
            if not group_present(study, group):
                raise ValueError(f"{file} has no '{group}' group: it is not a capsule-aeroheating study")
        if "fidelity" not in study.names("labels"):
            raise ValueError(f"{file} labels no 'fidelity', so the two fidelities cannot be told apart; this reads "
                             f"the multi-fidelity file ({FILES['wall']}), not version 1's Newtonian-only one")
    except Exception:
        study.close()
        raise
    return study


def group_present(study, name: str) -> bool:
    """Whether a study has a group (``node_fields``, ``constants``, ...) -- the format raises when it does not."""
    try:
        study.names(name)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

_EXPONENT = re.compile(r"^([A-Za-z%$]+)([-+]?\d+)$")


def schema_unit(reported: str) -> str:
    """The schema spelling of a unit the file reports.

    Gamut writes an exponent as a trailing signed number (``W m-2``, ``kg m-2 s-1``); the schema writes it with a
    caret (``W/m^2``, ``kg/(m^2*s)``). The translation is mechanical, and a spelling that still does not parse comes
    back as ``"unknown"`` -- which the units gate reports -- rather than being dropped or called dimensionless."""
    from ..schema.models import unit_or_dimensionless

    raw = (reported or "").strip()
    if not raw:
        return "unknown"
    try:
        return unit_or_dimensionless(raw)
    except ValueError:
        pass
    numerator: list[str] = []
    denominator: list[str] = []
    for token in raw.split():
        match = _EXPONENT.match(token)
        if match is None:
            numerator.append(token)
            continue
        symbol, exponent = match.group(1), int(match.group(2))
        (numerator if exponent > 0 else denominator).append(
            symbol if abs(exponent) == 1 else f"{symbol}^{abs(exponent)}")
    translated = "*".join(numerator) or "1"
    if denominator:
        translated += "/" + (denominator[0] if len(denominator) == 1 and "^" not in denominator[0]
                             else "(" + "*".join(denominator) + ")")
    try:
        return unit_or_dimensionless(translated)
    except ValueError:
        return "unknown"


def _unit_of(study, name: str) -> str:
    return schema_unit(str((study.describe(name) or {}).get("units", "")))


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def pair_fidelities(study) -> tuple[np.ndarray, np.ndarray]:
    """The rows of the two fidelities, paired by member: ``(high, low)``, both sorted by member.

    Every high-fidelity run has a low-fidelity run of the same member -- the cheap model runs on every capsule -- and
    this pairing is what makes the difference between them a subtraction at each node. A member with no
    low-fidelity run, or with two, is an error rather than something to drop: the two runs would then not be one
    shape at one flight condition."""
    labels = study.table(["member", "fidelity"])
    members = np.asarray(labels["member"], dtype=str)
    fidelity = np.asarray(labels["fidelity"], dtype=str)
    high = np.flatnonzero(fidelity == HIGH_FIDELITY)
    low = np.flatnonzero(fidelity == LOW_FIDELITY)
    if high.size == 0:
        raise ValueError(f"no run has fidelity '{HIGH_FIDELITY}': this file carries the engineering model only")
    low_of = {members[row]: int(row) for row in low}
    if len(low_of) != low.size:
        raise ValueError(f"the low-fidelity runs are not one per member: {low.size} runs, {len(low_of)} members")
    missing = [members[row] for row in high if members[row] not in low_of]
    if missing:
        raise ValueError(f"{len(missing)} high-fidelity run(s) have no low-fidelity run of the same member, e.g. "
                         f"{missing[0]}: the two cannot be paired")
    order = np.argsort(members[high], kind="stable")
    high = high[order]
    return high, np.array([low_of[members[row]] for row in high], dtype=np.int64)


@dataclass
class Wall:
    """One run's wall: the points, the triangles, and every nodal field on them."""

    row: int
    member: str
    fidelity: str
    points: np.ndarray                                   # (n_nodes, 3), m
    cells: np.ndarray                                    # (n_cells, nodes_per_cell), into ``points``
    cell_type: str
    fields: dict[str, np.ndarray] = field(default_factory=dict)      # each (n_nodes,)
    units: dict[str, str] = field(default_factory=dict)
    #: Values on the faces themselves -- Eilmer is cell-centred, so ``heat_flux_cell`` is the solver's own output and
    #: the nodal ``heat_flux`` is its area-weighted value at the nodes.
    cell_fields: dict[str, np.ndarray] = field(default_factory=dict)   # each (n_cells,)
    cell_field_units: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, float] = field(default_factory=dict)       # the six inputs of the run
    parameter_units: dict[str, str] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)          # per-run results, NaN where absent
    scalar_units: dict[str, str] = field(default_factory=dict)
    constants: dict[str, float] = field(default_factory=dict)        # what is fixed for the whole study

    @property
    def n_nodes(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    def summary(self) -> str:
        return f"{self.member}/{self.fidelity}: {self.n_nodes} nodes, {self.n_cells} {self.cell_type}s"


def wall(study, row: int) -> Wall:
    """One run's wall, read one row at a time.

    The mesh changes with the run -- every capsule has its own node coordinates -- so the file is read row by row;
    ``get_field`` across several rows is refused by the format for exactly that reason."""
    row = int(row)
    points = np.asarray(study.get_points(row), dtype=np.float64)
    types, offsets, connectivity = (np.asarray(part) for part in study.get_cells(row))
    per_cell = np.diff(offsets)
    if per_cell.size == 0 or int(per_cell.min()) != int(per_cell.max()):
        raise ValueError(f"row {row}: the wall mixes {sorted(set(per_cell.tolist()))} nodes per cell, and this reader "
                         "builds one homogeneous connectivity block")
    if len(set(types.tolist())) != 1 or int(types[0]) not in _CELL_TYPES:
        raise ValueError(f"row {row}: cell types {sorted(set(types.tolist()))} are not one known element type")
    cell_type, n_per_cell = _CELL_TYPES[int(types[0])]

    label = study.table(["member", "fidelity"], rows=[row])
    names = study.names("node_fields")
    # one row of a changing mesh reads back as (n_nodes,) on some layouts and (1, n_nodes) on others
    fields = {name: np.asarray(study.get_field(name, rows=[row]), dtype=np.float64).reshape(-1) for name in names}
    cell_names = study.names("cell_fields") if group_present(study, "cell_fields") else []
    cell_fields = {name: np.asarray(study.get_field(name, rows=[row]), dtype=np.float64).reshape(-1)
                   for name in cell_names}
    per_run = study.table(study.names("parameters") + study.names("scalars"), rows=[row])
    return Wall(row=row,
                member=str(np.asarray(label["member"]).reshape(-1)[0]),
                fidelity=str(np.asarray(label["fidelity"]).reshape(-1)[0]),
                points=points, cells=connectivity.reshape(-1, n_per_cell), cell_type=cell_type,
                fields=fields, units={name: _unit_of(study, name) for name in names},
                cell_fields=cell_fields, cell_field_units={name: _unit_of(study, name) for name in cell_names},
                parameters={name: float(per_run[name][0]) for name in study.names("parameters")},
                parameter_units={name: _unit_of(study, name) for name in study.names("parameters")},
                scalars={name: float(per_run[name][0]) for name in study.names("scalars")},
                scalar_units={name: _unit_of(study, name) for name in study.names("scalars")},
                constants={name: float(study.get_constant(name)) for name in study.names("constants")})


# ---------------------------------------------------------------------------
# bundles
# ---------------------------------------------------------------------------

def load_bundle(source: str | os.PathLike[str] | Any, *, row: int | None = None, member: str | None = None,
                license: str = LICENSE) -> Any:
    """One run's wall as a :class:`~pinneapple_data.schema.PhysicsDataBundle`.

    Give the file (or a study opened by :func:`open_study`) and either ``row`` or ``member``; with a member name and
    no row, its *high-fidelity* run is the one read when it has one, and its low-fidelity run otherwise. The six
    inputs go to ``physics.parameters`` with their units, the wall's isothermal temperature and the zero incidence to
    ``physics.boundary_conditions``, and the per-run scalars to ``metadata`` -- a bundle has a slot for parameters and
    for node fields, and none for a scalar result, which is recorded rather than bent into a field."""
    from ..schema import Field, Geometry, Mesh, Physics, PhysicsDataBundle
    from ..schema.bundle import file_refs
    from ..schema.models import ArrayRef, Block, BoundaryCondition, Parameter

    study, opened = _as_study(source)
    try:
        path = str(getattr(study, "path", ""))
        one = wall(study, _resolve_row(study, row=row, member=member))
        cells_key = f"block:{one.cell_type}"
        store: dict[str, np.ndarray] = {"mesh:nodes": one.points, cells_key: one.cells}
        fields = [Field.from_array(name, one.fields[name], unit=one.units[name], location="nodal",
                                   axes=["node"], description=f"{one.member} ({one.fidelity}): {name}")
                  for name in one.fields]
        for name in one.fields:
            store[f"field:{name}"] = one.fields[name]
        for name, values in one.cell_fields.items():          # Eilmer's own cell-centred values
            store[f"field:{name}"] = values
            fields.append(Field.from_array(name, values, unit=one.cell_field_units[name], location="cell",
                                           axes=["cell"], description=f"{one.member} ({one.fidelity}): {name}, on "
                                                                      "the faces"))
        inputs = file_refs([path]) if path and os.path.exists(path) else []
        conditions = [BoundaryCondition(field="pressure", region="wall", kind="wall"),
                      BoundaryCondition(field="velocity", region="wall", kind="wall")]
        if "wall_temperature" in one.constants:
            conditions.append(BoundaryCondition(field="temperature", region="wall", kind="dirichlet",
                                                value=float(one.constants["wall_temperature"]), unit="K"))
        return PhysicsDataBundle.from_arrays(
            store, fields=fields, axes=[],
            mesh=Mesh(kind="unstructured", dimension=3,
                      nodes=ArrayRef(key="mesh:nodes", shape=tuple(int(n) for n in one.points.shape),
                                     dtype=str(one.points.dtype)),
                      blocks=[Block(type=one.cell_type, name="wall",
                                    data=ArrayRef(key=cells_key, shape=tuple(int(n) for n in one.cells.shape),
                                                  dtype=str(one.cells.dtype)))]),
            geometry=Geometry(kind="mesh", dimension=3, units="m", files=inputs,
                              bounding_box=tuple(float(v) for v in one.points.min(axis=0))
                              + tuple(float(v) for v in one.points.max(axis=0)),
                              description="the capsule's heat shield, to mid-shoulder"),
            physics=Physics(
                governing="steady laminar Navier-Stokes of a calorically perfect gas, and the engineering model "
                          "that approximates it",
                symbols={"q": "wall heat flux", "p": "wall pressure", "h_r": "recovery enthalpy",
                         "C_H": "film coefficient", "T_w": "wall temperature"},
                parameters=[Parameter(name=name, value=value, unit=one.parameter_units.get(name, "1"),
                                      source=f"the run's inputs ({one.fidelity})")
                            for name, value in one.parameters.items()],
                boundary_conditions=conditions,
                notes=f"fidelity={one.fidelity}: {FIDELITIES.get(one.fidelity, '')}. Heating on a hotter wall: "
                      f"q(T_w) ~= C_H (h_r - c_p T_w). Constants of the whole study: {sorted(one.constants)}."),
            source=f"capsule-aeroheating {one.member} ({one.fidelity})",
            title=f"capsule {one.member}, {one.fidelity}", license=license, kind="dataset",
            adapter="pinneapple_data.datasets.capsule_aeroheating", adapter_version="1", inputs=inputs,
            citation=CITATION, url=RESOLVE_URL,
            metadata={"capsule_aeroheating": {
                "member": one.member, "fidelity": one.fidelity, "row": one.row,
                "n_nodes": one.n_nodes, "n_cells": one.n_cells, "cell_type": one.cell_type,
                "scalars": one.scalars, "scalar_units": one.scalar_units, "constants": one.constants,
                "held_fixed": ["calorically perfect air", "laminar", "300 K wall", "zero angle of attack",
                               "heat shield only, no afterbody or wake"],
                "node_i_is_the_same_point": "node i is the same point of the heat shield on every capsule and at "
                                            "both fidelities: two fidelities of one member subtract node by node",
            }})
    finally:
        if opened:
            study.close()


def load_table_bundle(source: str | os.PathLike[str] | Any, *, license: str = LICENSE) -> Any:
    """The run table as a bundle with no mesh: every scalar result of every run, and the six inputs.

    This is the tabular-regression half of the dataset -- ``heat_load`` and the rest against the six inputs, at either
    fidelity -- and the form in which the cheap and expensive runs are joined to study one against the other."""
    from ..schema import Field, Geometry, Mesh, Physics, PhysicsDataBundle
    from ..schema.bundle import file_refs
    from ..schema.models import Parameter

    study, opened = _as_study(source)
    try:
        path = str(getattr(study, "path", ""))
        table = study.table()
        rows = int(study.row_count)
        fidelity = np.asarray(table["fidelity"], dtype=str)
        store: dict[str, np.ndarray] = {}
        fields = []
        for name in study.names("scalars"):
            values = np.asarray(table[name], dtype=np.float64).reshape(rows)
            store[f"field:{name}"] = values
            fields.append(Field.from_array(name, values, unit=_unit_of(study, name), location="nodal", axes=["run"],
                                           description=f"one value per run: {name}"))
        inputs = file_refs([path]) if path and os.path.exists(path) else []
        return PhysicsDataBundle.from_arrays(
            store, fields=fields, axes=[],
            mesh=Mesh(kind="none", dimension=0), geometry=Geometry(kind="none", dimension=0),
            physics=Physics(
                governing="an engineering model and a laminar Navier-Stokes solution of the same capsules",
                parameters=[Parameter(name=name, unit=_unit_of(study, name),
                                      source="one value per run; the run's own bundle carries the value")
                            for name in study.names("parameters")],
                notes=f"{rows} runs: {int((fidelity == HIGH_FIDELITY).sum())} '{HIGH_FIDELITY}' and "
                      f"{int((fidelity == LOW_FIDELITY).sum())} '{LOW_FIDELITY}'. The two fidelities join on "
                      f"'member'."),
            source="capsule-aeroheating, run table", title="capsule aeroheating runs", license=license,
            kind="dataset", adapter="pinneapple_data.datasets.capsule_aeroheating", adapter_version="1", inputs=inputs,
            citation=CITATION, url=RESOLVE_URL,
            metadata={"capsule_aeroheating": {
                "rows": rows,
                "fidelities": {str(k): int(v) for k, v in zip(*np.unique(fidelity, return_counts=True),
                                                              strict=True)},
                "members": int(len(set(np.asarray(table["member"], dtype=str)))),
                "scalar_units": {name: _unit_of(study, name) for name in study.names("scalars")},
                "parameter_units": {name: _unit_of(study, name) for name in study.names("parameters")},
            }})
    finally:
        if opened:
            study.close()


def _as_study(source) -> tuple[Any, bool]:
    """The study behind ``source``, and whether this call has to close it."""
    if isinstance(source, (str, os.PathLike)):
        return open_study(source), True
    if not group_present(source, "parameters"):
        raise TypeError(f"{type(source).__name__} is neither a path to a capsule-aeroheating file nor an open study")
    return source, False


def _resolve_row(study, *, row: int | None, member: str | None) -> int:
    if row is None and member is None:
        raise ValueError("give the run to read: row=<int> or member='capsule_0066'")
    labels = study.table(["member", "fidelity"])
    members = np.asarray(labels["member"], dtype=str)
    fidelity = np.asarray(labels["fidelity"], dtype=str)
    if row is not None:
        if member is not None and members[int(row)] != member:
            raise ValueError(f"row {row} is {members[int(row)]}, not {member}")
        return int(row)
    matching = np.flatnonzero(members == member)
    if matching.size == 0:
        raise KeyError(f"no member '{member}' in this file; e.g. {sorted(set(members.tolist()))[:3]}")
    high = matching[fidelity[matching] == HIGH_FIDELITY]
    return int(high[0]) if high.size else int(matching[0])


# ---------------------------------------------------------------------------
# field-benchmark samples
# ---------------------------------------------------------------------------

def field_channels(study, rows: Sequence[int] | None = None
                   ) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """The six inputs as per-run channels, scaled to ``[-1, 1]`` by the bounds the file declares.

    The bounds come from the file's own description of each parameter, so a channel is scaled by its design space
    rather than by whatever the sample happened to reach. Returns the channel names, the values ``(n_rows, 6)``, and
    what the scaling was -- recorded in a sample's metadata so a prediction can be mapped back."""
    names = list(PARAMETERS)
    bounds: dict[str, tuple[float, float]] = {}
    for name in names:
        description = study.describe(name) or {}
        low, high = description.get("lower"), description.get("upper")
        if low is None or high is None or float(high) <= float(low):
            raise ValueError(f"parameter '{name}' declares no usable bounds ({low!r}, {high!r}): this scales by the "
                             "design space the file states, not by the range of the sample")
        bounds[name] = (float(low), float(high))
    picked = np.arange(study.row_count) if rows is None else np.asarray(rows, dtype=np.int64)
    table = study.table(names, rows=picked)
    columns = [2.0 * (np.asarray(table[name], dtype=np.float64) - low) / (high - low) - 1.0
               for name, (low, high) in bounds.items()]
    return names, np.stack(columns, axis=-1), {"bounds": bounds,
                                               "scaling": "2 * (x - lower) / (upper - lower) - 1"}


def field_samples(source: str | os.PathLike[str] | Any, *, field_name: str = "heat_flux",
                  fidelity: str = HIGH_FIDELITY, low_fidelity_channel: bool = True,
                  arc_length_channel: bool = True, rows: Sequence[int] | None = None) -> list[Any]:
    """One ``FieldSample`` per run: the wall's nodes, the field on them, and what a surrogate may read besides.

    A sample's ``points`` are the capsule's own wall nodes, ``values`` the requested field, ``thickness`` the arc
    length from the stagnation point (the heat shield's one intrinsic coordinate), and ``extra`` the six shape and
    flight inputs broadcast to every node -- followed, with ``low_fidelity_channel``, by the *cheap model's own value
    of the field* on the same nodes. That last channel is the multi-fidelity part: a model that reads it predicts a
    correction to the engineering model, and one built without the extra channels (``uses_extra=False``) predicts the
    field from the geometry and the flight condition alone. Both are legitimate; only the first is asked to beat the
    18.3 % the cheap model already reaches.

    ``fidelity`` selects which runs become samples -- ``HIGH_FIDELITY`` (the 64 with a Navier-Stokes solution) or
    ``LOW_FIDELITY`` (all 1.024). Requires ``pinneapple_arena.fieldbench`` for the sample type."""
    try:
        from pinneapple_arena.fieldbench import FieldSample
    except Exception as exc:                                   # pragma: no cover - depends on the installed tree
        raise ImportError(
            "field_samples returns pinneapple_arena.fieldbench samples, which this installation does not have "
            f"({exc}). Everything else in this module reads the dataset without it.") from exc

    study, opened = _as_study(source)
    try:
        labels = study.table(["member", "fidelity"])
        members = np.asarray(labels["member"], dtype=str)
        kinds = np.asarray(labels["fidelity"], dtype=str)
        if fidelity not in FIDELITIES:
            raise KeyError(f"'{fidelity}' is not one of {sorted(FIDELITIES)}")
        chosen = np.flatnonzero(kinds == fidelity) if rows is None else np.asarray(rows, dtype=np.int64)
        if chosen.size == 0:
            raise ValueError(f"no run has fidelity '{fidelity}'")
        channel_names, channels, scaling = field_channels(study, chosen)
        cheap_row: dict[str, int] = {}
        if low_fidelity_channel and fidelity == HIGH_FIDELITY:
            high_rows, low_rows = pair_fidelities(study)
            cheap_row = {str(members[row]): int(low) for row, low in zip(high_rows, low_rows, strict=True)}
        if cheap_row:
            channel_names = channel_names + [f"{field_name}_{LOW_FIDELITY}"]
        samples = []
        for row, channel in zip(chosen, channels, strict=True):
            one = wall(study, int(row))
            if field_name not in one.fields:
                raise KeyError(f"no nodal field '{field_name}'; this file has {sorted(one.fields)}")
            extra = np.broadcast_to(channel, (one.n_nodes, channel.shape[0])).astype(np.float64)
            if one.member in cheap_row:
                cheap = wall(study, cheap_row[one.member]).fields[field_name]
                extra = np.concatenate([extra, cheap.reshape(-1, 1)], axis=1)
            samples.append(FieldSample(
                id=f"{one.member}_{one.fidelity}", points=one.points,
                values=one.fields[field_name].reshape(-1, 1), category=one.fidelity,
                thickness=one.fields["arc_length"] if arc_length_channel else None, extra=extra,
                meta={"member": one.member, "fidelity": one.fidelity, "field": field_name,
                      "channels": channel_names, "channel_scaling": scaling,
                      "units": one.units.get(field_name, "unknown"),
                      "parameters": one.parameters, "scalars": one.scalars}))
        return samples
    finally:
        if opened:
            study.close()
