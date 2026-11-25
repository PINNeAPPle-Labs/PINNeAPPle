"""``StructuralRun``: one explicit-FEA run as a single object — parts, a time axis, units and provenance.

The library could read a mesh and a field at one instant, but not a *run*: ``crash_fem.CrashResult`` hands back
arrays in memory, OpenRadioss results arrive as a directory of VTK frames and reduced displacement dumps, and LS-DYNA
files arrive as a deck and a printed history. Nothing said "this is one impact, these are its parts, these are the
fields at these instants, this is what was dropped". Every benchmark rebuilt that privately, which is how provenance
gets lost. This is that object, and the four constructors that build it — from a :class:`CrashResult`, from a legacy
VTK time series (the ``anim_to_vtk`` output of the OpenRadioss campaign), from a staged OpenRadioss case directory,
and from the ASCII half of an LS-DYNA run.

The contract, in one sentence: **a solver's structural output becomes frames of the library's ``Mesh`` sharing one
connectivity, a time axis in seconds, named parts with the element indices each one owns, a unit per field, a
provenance record, and the masks a surrogate needs** — eroded elements, the contact force and the fixed nodes.

Where the fields live is the convention the whole CAE layer already uses: nodal quantities in ``Mesh.point_data``
(``displacement``, ``velocity``), element quantities in ``Mesh.cell_data`` (``von_mises``, ``plastic_strain``), and
``Mesh.points`` carrying the *deformed* geometry of that frame, so a frame is directly renderable. The undeformed
geometry is :attr:`StructuralRun.points`.

Units are never guessed. A solver's numbers are only meaningful with the system the deck was written in, so every
constructor takes ``units`` (a named system from :data:`UNIT_SYSTEMS` or a field → unit mapping) and
:meth:`StructuralRun.to_units` is the only path between systems, through ``pinneapple_data.physical_units`` — no unit
arithmetic is re-implemented here. A run built in ``mm-tonne-s`` and the same run in ``m-kg-s`` are the same object
after one call.

Honesty about what a source carries: :meth:`from_lsdyna` reads the ASCII side of a run (deck + ``nodout``) and
therefore has displacement and velocity and no stress — that is recorded in ``info["unsupported"]`` and in
``provenance``, not hidden behind an empty array that looks like a zero field.

    from pinneapple_data.cae import StructuralRun
    run = StructuralRun.from_crash_result(result, part_names={1: "beam", 2: "box"}, units="SI")
    run = StructuralRun.from_openradioss("runs/case_07", units="mm-tonne-s").to_units("SI")
    run.to_zarr("case_07.zarr")                       # and StructuralRun.from_zarr("case_07.zarr")
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import physical_units as pu
from ..provenance import build_provenance
from .model import CellBlock, Mesh

__all__ = ["CANONICAL_UNITS", "CONTAINER_KEYS", "FIELD_ALIASES", "UNIT_SYSTEMS", "Part", "StructuralRun",
           "canonical_field"]

#: the unit every field is stored in once a run has been normalised: SI, the system the library's solvers use
CANONICAL_UNITS = {"coordinates": "m", "displacement": "m", "velocity": "m/s", "von_mises": "Pa",
                   "plastic_strain": "1", "contact_force": "N", "time": "s", "energy": "J", "part_id": "1",
                   "density": "kg/m3"}

#: consistent systems an explicit deck is written in. ``mm-tonne-s`` is the one RADIOSS and LS-DYNA decks use for
#: crash work (mm, tonne, s give N and N*mm), and the reason a length and a stress cannot be compared across runs
#: until one of these is chosen.
UNIT_SYSTEMS: dict[str, dict[str, str]] = {
    "SI": dict(CANONICAL_UNITS),
    "m-kg-s": dict(CANONICAL_UNITS),
    "mm-tonne-s": {"coordinates": "mm", "displacement": "mm", "velocity": "mm/s", "von_mises": "MPa",
                   "plastic_strain": "1", "contact_force": "N", "time": "s", "energy": "N*mm", "part_id": "1",
                   "density": "t/mm3"},
    "mm-kg-ms": {"coordinates": "mm", "displacement": "mm", "velocity": "mm/ms", "von_mises": "MPa",
                 "plastic_strain": "1", "contact_force": "N", "time": "ms", "energy": "N*mm", "part_id": "1",
                 "density": "kg/mm3"},
}

SCHEMA = "pinneapple.structural_run/1"

#: ``info`` keys that describe the container a run was read from rather than the run itself; they are left out of
#: :meth:`StructuralRun.checksum`, so a run and its single-precision copy carry the same digest.
CONTAINER_KEYS = frozenset({"stored_as"})

#: how the names a solver writes map onto the canonical field names of a run. ``anim_to_vtk`` writes
#: ``VECTORS Displacement`` and a scalar for the von Mises stress under its own spelling; a run built from those files
#: has to carry them under the name the rest of this library (and every surrogate) uses.
FIELD_ALIASES = {"displacement": "displacement", "velocity": "velocity", "vonmises": "von_mises",
                 "von_mises": "von_mises", "von mises": "von_mises", "effective_stress": "von_mises",
                 "plasticstrain": "plastic_strain", "plastic_strain": "plastic_strain",
                 "effective_plastic_strain": "plastic_strain", "eroded": "eroded", "part_id": "part_id",
                 "node_id": "node_id"}


def _units_of(units: str | Mapping[str, str] | None) -> dict[str, str]:
    """The field → unit mapping of a named system, a mapping, or the canonical one."""
    if units is None:
        return dict(CANONICAL_UNITS)
    if isinstance(units, str):
        if units not in UNIT_SYSTEMS:
            raise ValueError(f"unknown unit system {units!r}: known are {sorted(UNIT_SYSTEMS)}")
        return dict(UNIT_SYSTEMS[units])
    return {str(k): str(v) for k, v in units.items()}


@dataclass(frozen=True)
class Part:
    """One part of the model: its solver id, its name, and the global cell indices it owns.

    ``elements`` is what a part-aware surrogate needs (a message passes inside a part, a part boundary is different),
    and it is the same indexing as ``Mesh.cell_data`` of every frame — so a field can be masked by part without a
    second lookup table that could go stale.
    """
    id: int
    name: str
    elements: np.ndarray
    kind: str = ""                              # "shell" / "solid" / the VTK element name
    thickness: float | None = None               # shell thickness, in the run's length unit
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.elements))

    def to_dict(self) -> dict[str, Any]:
        return {"id": int(self.id), "name": self.name, "elements": np.asarray(self.elements, dtype=np.int64).tolist(),
                "kind": self.kind, "thickness": self.thickness, "properties": dict(self.properties)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Part:
        return cls(id=int(d["id"]), name=str(d["name"]), elements=np.asarray(d["elements"], dtype=np.int64),
                   kind=str(d.get("kind", "")), thickness=d.get("thickness"),
                   properties=dict(d.get("properties") or {}))


def _parts_from_ids(ids: Sequence[int], *, names: Mapping[int, str] | None = None,
                    thickness: Mapping[int, float] | None = None,
                    kinds: Mapping[int, str] | None = None) -> list[Part]:
    """One :class:`Part` per distinct id, holding the element indices that carry it."""
    arr = np.asarray(ids, dtype=np.int64)
    out: list[Part] = []
    for pid in sorted(set(arr.tolist())):
        out.append(Part(id=int(pid), name=(names or {}).get(pid, f"part {pid}"),
                        elements=np.nonzero(arr == pid)[0].astype(np.int64),
                        kind=(kinds or {}).get(pid, ""), thickness=(thickness or {}).get(pid)))
    return out


@dataclass
class StructuralRun:
    """One structural run: frames of one mesh, a time axis, parts, units, provenance and the surrogate masks."""
    frames: list[Mesh]                              # deformed geometry + fields of each instant
    times: np.ndarray                               # (F,) in ``units["time"]``
    parts: list[Part] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=lambda: dict(CANONICAL_UNITS))
    provenance: dict[str, Any] = field(default_factory=dict)
    eroded: np.ndarray | None = None                # (F, m) bool: the element is deleted at that frame
    contact_force: np.ndarray | None = None         # (F, 3): the force on the body, in ``units["contact_force"]``
    fixed_nodes: np.ndarray | None = None           # (n,) bool: the nodes the deck held
    dropped: list[dict[str, Any]] = field(default_factory=list)   # what a normalisation step removed, and why
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("a StructuralRun needs at least one frame")
        self.times = np.asarray(self.times, dtype=np.float64).ravel()
        if len(self.times) != len(self.frames):
            raise ValueError(f"{len(self.frames)} frames against {len(self.times)} times")
        n, m = self.n_points, self.n_cells
        for k, f in enumerate(self.frames):
            if f.n_points != n:
                raise ValueError(f"frame {k} has {f.n_points} points, the first frame has {n}")
            if f.n_cells != m:
                raise ValueError(f"frame {k} has {f.n_cells} cells, the first frame has {m}")
            if not np.array_equal(f.points.shape, self.frames[0].points.shape):
                raise ValueError(f"frame {k} does not share the geometry of the first frame")
        if self.eroded is not None:
            self.eroded = np.asarray(self.eroded, dtype=bool)
            if self.eroded.shape != (len(self.frames), m):
                raise ValueError(f"eroded must be ({len(self.frames)}, {m}), got {self.eroded.shape}")
        if self.contact_force is not None:
            self.contact_force = np.asarray(self.contact_force, dtype=np.float64).reshape(len(self.frames), 3)
        if self.fixed_nodes is not None:
            self.fixed_nodes = _as_mask(self.fixed_nodes, n, "fixed_nodes")
        self.units = dict(self.units)
        self.provenance = dict(self.provenance)
        self.info = dict(self.info)

    # ------------------------------------------------------------------------------------- what a run is
    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_points(self) -> int:
        return self.frames[0].n_points

    @property
    def n_cells(self) -> int:
        return self.frames[0].n_cells

    @property
    def points(self) -> np.ndarray:
        """The undeformed geometry, ``(n, 3)`` — the first frame before its displacement."""
        return self.frames[0].points - self.frame_field("displacement", 0) \
            if "displacement" in self.frames[0].point_data else self.frames[0].points

    @property
    def fields(self) -> list[str]:
        """Every field name the run carries, nodal and elemental, in the order they first appear."""
        names: list[str] = []
        for f in self.frames:
            for k in list(f.point_data) + list(f.cell_data):
                if k not in names:
                    names.append(k)
        return names

    def data_of(self, name: str) -> str | None:
        """``"point"``/``"cell"``: where the field ``name`` lives (``None`` when the run does not carry it)."""
        for f in self.frames:
            if name in f.point_data:
                return "point"
            if name in f.cell_data:
                return "cell"
        return None

    def field(self, name: str) -> np.ndarray:
        """A field stacked over the frames: ``(F, n)``, ``(F, n, 3)``, ``(F, m)`` ... depending on where it lives."""
        where = self.data_of(name)
        if where is None:
            raise KeyError(f"the run carries no field {name!r}; it has {self.fields}")
        out = [np.asarray((f.point_data if where == "point" else f.cell_data)[name], dtype=np.float64)
               if name in (f.point_data if where == "point" else f.cell_data) else None for f in self.frames]
        shape = next(a.shape for a in out if a is not None)
        # a frame that did not carry the field is nan, never a made-up zero that would train as a measurement
        return np.stack([np.full(shape, np.nan) if a is None else a for a in out])

    def frame_field(self, name: str, k: int) -> np.ndarray | None:
        """One field of one frame, without stacking the series."""
        f = self.frames[k]
        if name in f.point_data:
            return np.asarray(f.point_data[name], dtype=np.float64)
        if name in f.cell_data:
            return np.asarray(f.cell_data[name], dtype=np.float64)
        return None

    def family(self) -> str:
        """The geometry family a split holds out — the run's own id, so whole geometries move together."""
        return str(self.provenance.get("case") or self.provenance.get("data_hash") or self.info.get("case")
                   or "unfamilied")

    def summary(self) -> dict[str, Any]:
        """What this run is, in the numbers a report can quote — the object's own view of itself."""
        return {"frames": self.n_frames, "points": self.n_points, "cells": self.n_cells,
                "elements": self.frames[0].element_counts(), "parts": {p.id: p.name for p in self.parts},
                "parts_elements": {p.id: len(p) for p in self.parts},
                "fields": self.fields, "units": dict(self.units),
                "time": [float(self.times[0]), float(self.times[-1])], "time_unit": self.units.get("time", "?"),
                "eroded_at_end": int(self.eroded[-1].sum()) if self.eroded is not None else None,
                "provenance": dict(self.provenance), "dropped": list(self.dropped), "info": dict(self.info)}

    # ------------------------------------------------------------------------------------- units
    def to_units(self, system: str | Mapping[str, str]) -> StructuralRun:
        """The same run in another unit system, through ``pinneapple_data.physical_units``.

        Every array whose field has a unit in both systems is converted (geometry, nodal and elemental fields, the
        contact force and the times); a field with no unit in one of them is left as it is and named in
        ``info["not_converted"]``, because silently rescaling a dimensionless or an unknown quantity is how a factor of
        a thousand survives a review.
        """
        target = _units_of(system)
        not_converted: list[str] = []
        new_frames: list[Mesh] = []
        for f in self.frames:
            pts = pu.convert(f.points, self.units.get("coordinates", target.get("coordinates", "m")),
                             target["coordinates"]) if "coordinates" in target else f.points
            mesh = Mesh(points=np.asarray(pts, dtype=np.float64), blocks=[CellBlock(b.type, b.conn, b.ids)
                                                                          for b in f.blocks],
                        cell_sets={k: v.copy() for k, v in f.cell_sets.items()},
                        node_sets={k: v.copy() for k, v in f.node_sets.items()},
                        units=dict(target), source=dict(f.source))
            for store, out in ((f.point_data, mesh.point_data), (f.cell_data, mesh.cell_data)):
                for name, vals in store.items():
                    src, dst = self.units.get(name), target.get(name)
                    if src is None or dst is None:
                        not_converted.append(name)
                        out[name] = np.asarray(vals, dtype=np.float64)
                    else:
                        out[name] = np.asarray(pu.convert(np.asarray(vals, dtype=np.float64), src, dst),
                                               dtype=np.float64)
            new_frames.append(mesh)
        times = self.times
        if "time" in target:
            times = np.asarray(pu.convert(self.times, self.units.get("time", target["time"]), target["time"]),
                               dtype=np.float64)
        force = self.contact_force
        if force is not None and "contact_force" in target:
            force = np.asarray(pu.convert(force, self.units.get("contact_force", target["contact_force"]),
                                          target["contact_force"]), dtype=np.float64)
        parts = [_scale_thickness(p, self.units, target) for p in self.parts]
        info = dict(self.info)
        if not_converted:
            info["not_converted"] = sorted(set(not_converted))
        return StructuralRun(frames=new_frames, times=times, parts=parts, units=dict(target),
                             provenance=dict(self.provenance), eroded=None if self.eroded is None else self.eroded.copy(),
                             contact_force=None if force is None else force,
                             fixed_nodes=None if self.fixed_nodes is None else self.fixed_nodes.copy(),
                             dropped=list(self.dropped), info=info)

    # ------------------------------------------------------------------------------------- identity
    def part_of_cell(self) -> np.ndarray:
        """``(m,)`` part id of every cell, in the frames' own cell order (``0`` where no part claims it)."""
        out = np.zeros(self.n_cells, dtype=np.int64)
        for p in self.parts:
            out[np.asarray(p.elements, dtype=np.int64)] = p.id
        return out

    def checksum(self) -> str:
        """SHA-256 over the arrays and the metadata: two runs with the same checksum carry the same numbers.

        The digest covers every frame's points and fields, the masks and a canonical rendering of the metadata, in a
        fixed order, so it changes when a field does and does not change when a dict is written with other keys in
        another order. It describes what the run *is*: the keys in :data:`CONTAINER_KEYS` describe the file it was
        read from, so a run and its Zarr copy carry the same checksum whatever precision the copy was stored at.
        """
        h = hashlib.sha256()
        for k, f in enumerate(self.frames):
            h.update(f"frame {k}\n".encode())
            for name, a in [("points", f.points)] + sorted(f.point_data.items()) + sorted(f.cell_data.items()):
                h.update(name.encode())
                h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
        h.update(np.ascontiguousarray(self.times, dtype=np.float64).tobytes())
        for name, a in (("eroded", self.eroded), ("contact_force", self.contact_force),
                        ("fixed_nodes", self.fixed_nodes)):
            h.update(name.encode())
            h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes() if a is not None
                     else b"none")
        info = {k: v for k, v in self.info.items() if k not in CONTAINER_KEYS}
        h.update(_canonical({"units": self.units, "parts": [p.to_dict() for p in self.parts],
                             "provenance": self.provenance, "info": info, "dropped": self.dropped}).encode())
        return h.hexdigest()

    # ------------------------------------------------------------------------------------- constructors
    @classmethod
    def from_crash_result(cls, result: Any, *, part_names: Mapping[int, str] | None = None,
                          thickness: Mapping[int, float] | None = None,
                          fixed_nodes: np.ndarray | None = None, units: str | Mapping[str, str] | None = None,
                          provenance: Mapping[str, Any] | None = None,
                          case: str | None = None) -> StructuralRun:
        """The one-call conversion of ``crash_fem.CrashResult`` into the run container.

        Everything the crash solver measures has a home here: the displacement and velocity per node, the von Mises
        stress and equivalent plastic strain per element, the eroded mask, the contact force and the times. The
        underside of the envelope is carried too — ``out_of_envelope`` and its reason go into the provenance, because a
        run outside its verified range is still a run, and a dataset that drops that fact trains on it unknowingly.
        """
        u = np.asarray(result.displacement, dtype=np.float64)
        vel = np.asarray(result.velocity, dtype=np.float64) if result.velocity is not None else None
        vm = np.asarray(result.von_mises, dtype=np.float64)
        ep = np.asarray(result.plastic_strain, dtype=np.float64)
        times = np.asarray(result.times, dtype=np.float64)
        nodes = np.asarray(result.mesh.nodes, dtype=np.float64)
        block = CellBlock("hexahedron", np.asarray(result.mesh.elements, dtype=np.int64))
        info = dict(result.info or {})
        units_map = _units_of(units)
        frames: list[Mesh] = []
        for k, t in enumerate(times.tolist()):
            point = {"displacement": u[k]}
            if vel is not None:
                point["velocity"] = vel[k]
            frames.append(Mesh(points=nodes + u[k], blocks=[block], point_data=point,
                               cell_data={"von_mises": vm[k], "plastic_strain": ep[k]}, units=dict(units_map),
                               source={"format": "in-memory (CrashResult)", "solver": "pinneapple-crash-fem",
                                       "time": t, "frame": k}))
        prov = dict(provenance) if provenance else build_provenance(
            "pinneapple-crash-fem", seed="deterministic",
            mesh={"points": int(len(nodes)), "cells": int(len(block.conn)), "type": "hexahedron"},
            data_hash=_hash_of(info),
            case=case or info.get("case"),
            out_of_envelope=bool(getattr(result, "out_of_envelope", False)),
            envelope_reason=getattr(result, "envelope_reason", None),
            dt=float(getattr(result, "dt", 0.0)),
            frames=int(len(times)))
        return cls(frames=frames, times=times,
                   parts=_parts_from_ids(np.asarray(result.parts, dtype=np.int64), names=part_names,
                                         thickness=thickness, kinds=None),
                   units=units_map, provenance=prov,
                   eroded=np.asarray(result.eroded, dtype=bool),
                   contact_force=(np.asarray(result.contact_force, dtype=np.float64)
                                  if result.contact_force is not None else None),
                   fixed_nodes=fixed_nodes, info=dict(info, source="CrashResult"))

    @classmethod
    def from_vtk_series(cls, source: Iterable[str | Path] | str | Path, *,
                        units: str | Mapping[str, str] | None = None,
                        times: Sequence[float] | None = None,
                        part_names: Mapping[int, str] | None = None,
                        provenance: Mapping[str, Any] | None = None,
                        field_units: Mapping[str, str] | None = None) -> StructuralRun:
        """A legacy-VTK time series (a directory or the files) as one run — the generic and the OpenRadioss path.

        The series must share one connectivity: frames of different meshes are not a run but a set of runs, and the
        error says which file disagrees rather than aligning them silently. Times come from the ``TIME`` value of each
        file when it has one, else from ``times``, else from the file order.
        """
        from .vtk_series import read_vtk_series

        got = read_vtk_series(source)
        frames = [g.mesh for g in got]
        uni = _units_of(units)
        if field_units:
            uni.update(dict(field_units))
        t = np.array([g.time for g in got], dtype=np.float64)
        if np.isnan(t).any() and times is not None:
            t = np.asarray(times, dtype=np.float64)
        elif np.isnan(t).any():
            t = np.arange(len(frames), dtype=np.float64)
        fixed = _fixed_connectivity(frames, [g.path for g in got])
        parts: list[Part] = []
        ids = got[0].part_ids
        if ids is not None:
            parts = _parts_from_ids(np.asarray(ids, dtype=np.int64), names=part_names)
        for f in frames:
            f.units = dict(uni)
        prov = dict(provenance) if provenance else build_provenance(
            "vtk-series", seed="deterministic", data_hash=_hash_of([g.path for g in got] + [fixed]),
            mesh={"points": frames[0].n_points, "cells": frames[0].n_cells,
                  "type": ",".join(sorted(frames[0].element_counts()))},
            frames=len(frames), files=len(got))
        return cls(frames=frames, times=t, parts=parts, units=uni, provenance=prov,
                   info={"source": "VTK time series", "files": [g.path for g in got],
                         "connectivity": fixed,
                         "frames_with_time": sum(1 for g in got if g.time is not None)})

    @classmethod
    def from_openradioss(cls, case_dir: str | Path, *, run_name: str | None = None,
                         units: str | Mapping[str, str] | None = "mm-tonne-s",
                         part_names: Mapping[int, str] | None = None,
                         provenance: Mapping[str, Any] | None = None,
                         thickness: Mapping[int, float] | None = None) -> StructuralRun:
        """A staged OpenRadioss case (what ``external_solvers.openradioss.runner`` leaves on disk) as one run.

        The campaign keeps ``frames/frame_000.vtk`` (the full first frame: points, cells, node and part ids) and one
        ``frames/disp_XXX.txt`` + ``frames/time_XXX.txt`` per output step with the nodal displacement — the reduction
        the runner does inside the container so a long run does not fill a disk. Both are read here: the geometry from
        the full frame, the kinematics from the reduced dumps, and any scalar the full frames carry (a campaign that
        keeps more than frame 0 keeps its fields with it). Elemental fields only exist where a full frame exists, so
        ``info["fields_partial"]`` counts the frames that carry each one and ``info["fields_missing"]`` lists the ones no
        frame carries — a run reduced to displacement says exactly that.

        ``units`` defaults to ``mm-tonne-s`` because that is the system a crash deck is written in and a RADIOSS deck
        states it only in prose, never in a header the reader could trust: the caller states the system, and
        :meth:`to_units` is how the result is brought to the canonical one.
        """
        from pinneapple_simulation.external_solvers.openradioss import read_case_displacements

        from .vtk_series import read_vtk_frame

        root = Path(case_dir)
        frames_dir = root / "frames" if (root / "frames").is_dir() else root
        full = sorted(frames_dir.glob("frame_*.vtk"))
        if not full:
            raise FileNotFoundError(f"{root}: no frames/frame_*.vtk — the case has not been run "
                                    "(external_solvers.openradioss.run_openradioss_case writes them)")
        first = read_vtk_frame(full[0])
        disp_files = sorted(frames_dir.glob("disp_*.txt"))
        uni = _units_of(units)
        if disp_files:
            times, disp = read_case_displacements(frames_dir)
            times = np.asarray(times, dtype=np.float64)
            disp = np.asarray(disp, dtype=np.float64)
        else:
            times = np.array([first.time if first.time is not None else 0.0])
            disp = np.zeros((1, first.mesh.n_points, 3))
        if disp.shape[1] != first.mesh.n_points:
            raise ValueError(f"the displacement dumps carry {disp.shape[1]} nodes and the frame has "
                             f"{first.mesh.n_points}: they are not the same case")
        # the full frames that exist carry their own scalars; frame k uses one when the file is there
        by_index = {int(Path(p).stem.split("_")[-1]): p for p in full}
        frames: list[Mesh] = []
        carried: dict[str, int] = {}                  # field -> the number of frames that carry it
        for k in range(len(times)):
            src = by_index.get(k) or by_index.get(k + 1)
            src_frame = read_vtk_frame(src) if src is not None else None
            point = {"displacement": disp[k]}
            cell: dict[str, np.ndarray] = {}
            if src_frame is not None:
                for name, vals in src_frame.mesh.point_data.items():
                    alias = canonical_field(name)
                    if alias != "displacement":        # the reduced dump is the displacement of every frame
                        point[alias] = np.asarray(vals, dtype=np.float64)
                cell = {canonical_field(k2): np.asarray(v, dtype=np.float64)
                        for k2, v in src_frame.mesh.cell_data.items()}
            for name in list(point) + list(cell):
                carried[name] = carried.get(name, 0) + 1
            frames.append(Mesh(points=first.mesh.points + disp[k],
                               blocks=[CellBlock(b.type, b.conn, b.ids) for b in first.mesh.blocks],
                               point_data=point, cell_data=cell, units=dict(uni),
                               source={"format": "OpenRadioss (ANIM->VTK)", "file": str(src) if src else "",
                                       "time": float(times[k]), "frame": k}))
        parts: list[Part] = []
        if first.part_ids is not None:
            kinds = {int(p): b.type for p, b in zip(sorted(set(first.part_ids.tolist())), first.mesh.blocks,
                                                    strict=False)}
            parts = _parts_from_ids(np.asarray(first.part_ids, dtype=np.int64), names=part_names,
                                    thickness=thickness, kinds=kinds)
        # only the full frames carry the element fields; a campaign that kept one frame of six has them there alone
        partial = {n: c for n, c in sorted(carried.items()) if c < len(times)}
        prov = dict(provenance) if provenance else build_provenance(
            "openradioss", seed="deterministic",
            data_hash=_hash_of([str(p) for p in full] + [str(p) for p in disp_files]),
            mesh={"points": int(first.mesh.n_points), "cells": int(first.mesh.n_cells),
                  "type": ",".join(sorted(first.mesh.element_counts()))},
            case=run_name or root.name, solver_version=_radioss_version(root), frames=len(times),
            fields_partial=partial)
        return cls(frames=frames, times=times, parts=parts, units=uni, provenance=prov,
                   info={"source": "OpenRadioss staged case", "case": str(root),
                         "full_frames": [str(p) for p in full], "reduced_frames": len(disp_files),
                         "fields_partial": partial,
                         "fields_missing": sorted(n for n in ("von_mises", "plastic_strain", "eroded")
                                                  if n not in carried)})

    @classmethod
    def from_lsdyna(cls, directory: str | Path, *, deck: str | Path | None = None,
                    units: str | Mapping[str, str] | None = None,
                    provenance: Mapping[str, Any] | None = None) -> StructuralRun:
        """The ASCII half of an LS-DYNA run: the keyword deck (geometry, parts, thickness) and ``nodout``.

        This path carries displacement and velocity and **no** element field: ``d3plot`` is binary and no licence here
        can check a reader of it, ``glstat``/``elout`` are not read. ``info["unsupported"]`` says exactly that, and the
        guide repeats it — a run built here looks like a run, so the object has to say which fields it could not have.
        A node the deck has and ``nodout`` does not print keeps ``nan`` as its displacement and its undeformed position
        in the frames; it is counted in ``info["nodes_outside_nodout"]``.
        """
        from .lsdyna import deck_mesh, read_lsdyna_directory

        got = read_lsdyna_directory(directory, deck=deck)
        mesh0 = deck_mesh(got.deck)
        uni = _units_of(units)
        hist = got.history
        disp = hist.displacement()
        vel = hist.velocity()
        if disp is None:
            raise ValueError(f"{directory}: nodout carries no x/y/z displacement, so there is no run to build")
        # map the history's node ids onto the deck's rows (the deck is the geometry of record)
        deck_ids = np.asarray(mesh0.source["node_ids"], dtype=np.int64)
        row_of = {int(n): k for k, n in enumerate(deck_ids.tolist())}
        target = np.array([row_of.get(int(n), -1) for n in hist.node_ids.tolist()], dtype=np.int64)
        keep = target >= 0
        if not keep.any():
            raise ValueError(f"{directory}: no nodout node id appears in the keyword deck — they are not the same run")
        # the nodes the deck has and nodout does not print: no measured displacement, and counted as such
        measured = np.zeros(len(deck_ids), dtype=bool)
        measured[target[keep]] = True
        outside = int((~measured).sum())
        u = np.full((len(hist.times), len(deck_ids), 3), np.nan)
        u[:, target[keep]] = disp[:, keep]
        v = None
        if vel is not None:
            v = np.full_like(u, np.nan)
            v[:, target[keep]] = vel[:, keep]
        # a deck node that nodout does not print has no measured displacement: its field is nan, and its position in
        # the frame stays its undeformed one (a zero would claim the node did not move, which is a measurement it
        # never made). ``info["nodes_outside_nodout"]`` counts them.
        frames: list[Mesh] = []
        for k, t in enumerate(hist.times.tolist()):
            point = {"displacement": u[k]}
            if v is not None:
                point["velocity"] = v[k]
            frames.append(Mesh(points=mesh0.points + np.where(np.isnan(u[k]), 0.0, u[k]),
                               blocks=[CellBlock(b.type, b.conn, b.ids) for b in mesh0.blocks],
                               point_data=point, cell_data={}, units=dict(uni),
                               source={"format": "LS-DYNA ASCII (nodout)", "time": t, "frame": k}))
        ids = mesh0.cell_data["part_id"].astype(np.int64)
        parts = _parts_from_ids(ids, names=got.deck.part_names,
                                thickness={p: got.deck.thickness_of_part(p) for p in got.deck.part_ids},
                                kinds={p: got.deck.section_kind.get(got.deck.part_section.get(p, -1), "")
                                       for p in got.deck.part_ids})
        prov = dict(provenance) if provenance else build_provenance(
            "lsdyna-ascii", seed="deterministic", data_hash=_hash_of(got.deck.summary()),
            mesh={"points": int(mesh0.n_points), "cells": int(mesh0.n_cells),
                  "type": ",".join(sorted(mesh0.element_counts()))},
            deck_title=got.deck.title, unsupported=got.unsupported,
            nodes_outside_nodout=outside)
        return cls(frames=frames, times=np.asarray(hist.times, dtype=np.float64), parts=parts, units=uni,
                   provenance=prov,
                   info={"source": "LS-DYNA ASCII", "directory": str(directory),
                         "unsupported": list(got.unsupported), "deck": got.deck.summary(),
                         "nodes_outside_nodout": outside})

    # ------------------------------------------------------------------------------------- export
    def to_vtk_series(self, out_dir: str | Path, *, precision: str = "single",
                      prefix: str = "frame") -> list[Path]:
        """Write the run as a legacy-VTK series — the format :meth:`from_vtk_series` reads back."""
        from .vtk_series import write_vtk_series

        return write_vtk_series(self.frames, out_dir, times=self.times, precision=precision, prefix=prefix,
                                part_ids=self.part_of_cell() if self.parts else None)

    def to_zarr(self, path: str | Path, *, dtype: str = "float64") -> Path:
        """Write the run to a Zarr group and return the path; :meth:`from_zarr` reads it back.

        The frames are stacked (one array per field, ``first dimension = frame``), the connectivity is stored once
        because the contract is that every frame shares it, and the metadata (units, parts, provenance, masks, what was
        dropped) travels as one JSON attribute.

        ``dtype="float64"`` is the default and the round trip is **bit-identical**: every array comes back with the
        same bytes and :meth:`checksum` matches. ``dtype="float32"`` halves the store and is the option a campaign uses
        when a dataset outgrows a disk; it is documented, not silent — the stored precision is written into
        ``attrs["stored_as"]`` and, on the way back, into ``info["stored_as"]``, and the numbers differ in the eighth
        significant digit rather than agreeing. The connectivity and the masks stay integer/boolean in both cases.
        """
        if dtype not in ("float64", "float32"):
            raise ValueError("dtype must be 'float64' or 'float32'")
        store = np.dtype(dtype)
        import zarr

        from ..zarr_store import _create_array

        root = Path(path)
        group = zarr.open_group(str(root), mode="w")
        names = self.fields
        point_fields = [n for n in names if self.data_of(n) == "point"]
        cell_fields = [n for n in names if self.data_of(n) == "cell"]
        _create_array(group, "frame_points", np.stack([f.points for f in self.frames]).astype(store))
        _create_array(group, "times", self.times.astype(store))
        point_group = _subgroup(group, "point")
        for name in point_fields:
            _create_array(point_group, name, self.field(name).astype(store))
        cell_group = _subgroup(group, "cell")
        for name in cell_fields:
            _create_array(cell_group, name, self.field(name).astype(store))
        block_types: list[str] = []
        for k, b in enumerate(self.frames[0].blocks):
            _create_array(_subgroup(group, f"block_{k}"), "conn", np.asarray(b.conn, dtype=np.int64))
            block_types.append(b.type)
        for name, a in (("eroded", self.eroded), ("contact_force", self.contact_force),
                        ("fixed_nodes", self.fixed_nodes)):
            if a is not None:
                _create_array(group, name, np.asarray(a) if a.dtype == bool else np.asarray(a, dtype=store))
        digest = self.checksum()
        group.attrs["meta"] = _canonical({
            "schema": SCHEMA, "units": self.units, "parts": [p.to_dict() for p in self.parts],
            "provenance": self.provenance, "dropped": self.dropped, "info": self.info,
            "points": int(self.n_points), "cells": int(self.n_cells), "frames": int(self.n_frames),
            "blocks": block_types,
            "cell_sets": {k: v.tolist() for k, v in self.frames[0].cell_sets.items()},
            "node_sets": {k: v.tolist() for k, v in self.frames[0].node_sets.items()},
            "checksum": digest, "stored_as": dtype})
        group.attrs["checksum"] = digest
        return root

    @classmethod
    def from_zarr(cls, path: str | Path) -> StructuralRun:
        """Read a run written by :meth:`to_zarr`."""
        import zarr

        group = zarr.open_group(str(Path(path)), mode="r")
        meta = json.loads(group.attrs["meta"])
        times = np.asarray(group["times"][:], dtype=np.float64)
        frame_points = np.asarray(group["frame_points"][:], dtype=np.float64)
        blocks = [CellBlock(str(t), np.asarray(group[f"block_{k}/conn"][:], dtype=np.int64))
                  for k, t in enumerate(meta.get("blocks") or [])]
        point_series = _arrays_of(group, "point")
        cell_series = _arrays_of(group, "cell")
        frames = [Mesh(points=frame_points[i], blocks=[CellBlock(b.type, b.conn, b.ids) for b in blocks],
                       point_data={n: a[i] for n, a in point_series.items()},
                       cell_data={n: a[i] for n, a in cell_series.items()},
                       units=dict(meta["units"]), source={"format": "Zarr (StructuralRun)", "frame": i,
                                                          "time": float(times[i])})
                  for i in range(len(times))]
        eroded = np.asarray(group["eroded"][:], dtype=bool) if "eroded" in group else None
        force = np.asarray(group["contact_force"][:], dtype=np.float64) if "contact_force" in group else None
        fixed = np.asarray(group["fixed_nodes"][:], dtype=bool) if "fixed_nodes" in group else None
        return cls(frames=frames, times=times, parts=[Part.from_dict(p) for p in meta["parts"]],
                   units=dict(meta["units"]), provenance=dict(meta["provenance"]),
                   eroded=eroded, contact_force=force, fixed_nodes=fixed,
                   dropped=list(meta.get("dropped") or []),
                   info=dict(meta.get("info") or {}, stored_as=meta.get("stored_as", "float64")))


# ------------------------------------------------------------------------------------------------------- helpers
def canonical_field(name: str) -> str:
    """The canonical name of a field a solver wrote (``Displacement`` and ``von Mises`` included), or the name."""
    return FIELD_ALIASES.get(str(name).strip().lower().replace(" ", "_"), str(name))


def _as_mask(nodes: Any, n: int, what: str) -> np.ndarray:
    """A boolean mask of ``n`` entries from either a mask or a list of node indices — the convention CrashSolver uses."""
    arr = np.asarray(nodes)
    if arr.dtype == bool and arr.size == n:
        return arr.ravel().copy()
    mask = np.zeros(n, dtype=bool)
    idx = arr.ravel()
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError(f"{what}: node index {idx.max()} is outside 0..{n - 1}")
    if idx.size and not np.issubdtype(idx.dtype, np.integer):
        if not np.allclose(idx, np.rint(idx)):
            raise ValueError(f"{what}: a list of node indices must be whole numbers")
        idx = np.rint(idx)
    mask[idx.astype(np.int64)] = True
    return mask


def _subgroup(group: Any, name: str) -> Any:
    """A child group of a Zarr group, created when absent (Zarr 2 and 3 differ in the spelling, not in the meaning)."""
    if name in group:
        return group[name]
    if hasattr(group, "require_group"):
        return group.require_group(name)
    return group.create_group(name)


def _arrays_of(group: Any, name: str) -> dict[str, np.ndarray]:
    """Every array of a child group, read as ``float64`` — the fields of a Zarr-stored run."""
    try:
        child = group[name]
    except (KeyError, TypeError):
        return {}
    keys = list(child.array_keys()) if hasattr(child, "array_keys") else list(child.keys())
    return {k: np.asarray(child[k][:], dtype=np.float64) for k in keys}


def _canonical(obj: Any) -> str:
    """A deterministic text rendering of a metadata tree, for hashing and for storing in an attribute."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def _hash_of(obj: Any) -> str:
    """SHA-256 of a metadata tree — the ``data_hash`` a provenance record carries."""
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()


def _fixed_connectivity(frames: Sequence[Mesh], names: Sequence[str]) -> str:
    """The hash of the connectivity, after checking every frame agrees with the first (the run's own contract)."""
    ref = frames[0]
    h = hashlib.sha256()
    for b in ref.blocks:
        h.update(b.type.encode())
        h.update(np.ascontiguousarray(np.asarray(b.conn, dtype=np.int64)).tobytes())
    for k, f in enumerate(frames[1:], start=1):
        same = len(f.blocks) == len(ref.blocks) and all(
            a.type == b.type and a.conn.shape == b.conn.shape and np.array_equal(a.conn, b.conn)
            for a, b in zip(f.blocks, ref.blocks, strict=True))
        if not same:
            where = names[k] if k < len(names) else f"frame {k}"
            raise ValueError(f"{where} does not share the connectivity of the first frame: a run has one mesh and "
                             "many instants, so frames with different cells are different runs")
    return h.hexdigest()


def _scale_thickness(part: Part, src: Mapping[str, str], dst: Mapping[str, str]) -> Part:
    """A part with its shell thickness converted when both systems have a length unit."""
    if part.thickness is None or "coordinates" not in src or "coordinates" not in dst:
        return part
    t = float(pu.convert(part.thickness, src["coordinates"], dst["coordinates"]))
    return Part(id=part.id, name=part.name, elements=part.elements, kind=part.kind, thickness=t,
                properties=dict(part.properties))


def _radioss_version(root: Path) -> str | None:
    """The solver version a staged case records in its logs, when it records one (``OpenRadioss`` banner)."""
    for name in ("engine.log", "starter.log"):
        p = root / name
        if p.is_file():
            for line in p.read_text(encoding="latin-1", errors="replace").splitlines()[:40]:
                low = line.lower()
                if "version" in low and any(ch.isdigit() for ch in line):
                    return line.strip()[:120]
    return None
