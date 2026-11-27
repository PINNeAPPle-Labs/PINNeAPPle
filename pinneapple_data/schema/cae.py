"""CAE ⇄ bundle: what a solver wrote becomes a ``PhysicsDataBundle``, and a bundle leaves as VTU.

Every solver this repository drives already produces a neutral :class:`pinneapple_data.cae.model.Mesh` -- an OpenFOAM
case through ``cae.foam``, a CalculiX ``.frd``, an Abaqus deck, a VTU, a Gmsh mesh, a point table with fields. Until
now that mesh could be inspected, checked and rendered, but it could not become *data*: it had no licence, no
provenance, no declared unit per field and no record of what was checked. This module closes that loop in both
directions:

``bundle_from_mesh``
    a CAE ``Mesh`` plus the licence and the source becomes a bundle: nodes and **mixed** cell blocks (a volume of
    hexahedra or tetrahedra beside a surface of triangles is the normal case, and the boundary must not be dropped
    to fit a single connectivity array), the point fields as ``nodal`` and the cell fields as ``cell`` with the unit
    each one already carries, the named groups (patches, ELSETs, face zones) as ``physical_groups``, and the
    solver, format and time step as provenance.
``bundle_to_mesh``
    the inverse, so a bundle goes back through the CAE tools that read a ``Mesh`` -- quality metrics, the VTU/HDF5/
    Parquet exporters in :mod:`pinneapple_data.cae.dataset`, the 3-D views.
``export_bundle``
    one call to write a bundle as VTU (the binary XML the existing writer produces), HDF5, NPZ, Parquet, CSV, JSON
    or the PINNeAPPle UPD archive, using the exporters that already exist rather than a second copy of them.

What is deliberately *not* invented: the licence. A solver output has whatever licence its case does, and this
module asks for it instead of writing ``unknown`` into a file somebody will redistribute.

    >>> from pinneapple_data.schema.cae import bundle_from_mesh, export_bundle
    >>> bundle = bundle_from_mesh(mesh, source="CalculiX cantilever, step 1", license="CC BY 4.0")
    >>> export_bundle(bundle, "cantilever.vtu")          # the same field data the solver wrote
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..cae.model import CellBlock, Mesh
from .bundle import PhysicsDataBundle, file_refs
from .models import ArrayRef, Block, Field, Geometry, Physics, Region
from .models import Mesh as MeshSpec

__all__ = ["bundle_from_mesh", "bundle_to_mesh", "bundle_from_files", "bundle_from_path", "export_bundle",
           "EXPORT_FORMATS"]

EXPORT_FORMATS = ("vtu", "hdf5", "npz", "parquet", "csv", "json", "pinneapple")

def bundle_from_mesh(mesh: Mesh, *, source: str, license: str, title: str = "",
                     kind: str = "simulation", inputs: Sequence[str] = (), citation: str = "",
                     url: str = "", notes: str = "", units: Mapping[str, str] | None = None,
                     geometry_units: str | None = None) -> PhysicsDataBundle:
    """A CAE ``Mesh`` as a :class:`PhysicsDataBundle`.

    ``units`` overrides the unit per field (``{"DISP": "mm", "STRESS": "MPa"}``): a CalculiX deck stores results in
    the deck's own system, and the reader can only report "model units", which the schema records as ``unknown``
    rather than as dimensionless. With the override the bundle passes the ``units`` gate; without it, that gate
    fails and says which fields nobody declared -- a verdict the reviewer needs, not a conversion invented here.
    A unit this library cannot parse is always a hard error."""
    arrays: dict[str, np.ndarray] = {"mesh:nodes": np.asarray(mesh.points, dtype=np.float64)}
    for i, block in enumerate(mesh.blocks):
        arrays[f"block:{i}"] = np.asarray(block.conn, dtype=np.int64)
    for name, block in mesh.boundary_blocks:
        arrays[f"group:{name}"] = np.asarray(block.conn, dtype=np.int64)

    fields: list[Field] = []
    for location, store in (("nodal", mesh.point_data), ("cell", mesh.cell_data)):
        for name, values in store.items():
            arr = np.asarray(values)
            components = []
            if arr.ndim == 2 and arr.shape[1] > 1:
                components = list(mesh.source.get("components", {}).get(name)
                                  or [f"c{i}" for i in range(arr.shape[1])])
            arrays[f"field:{name}"] = arr
            reported = str(mesh.units.get(name, "1"))
            unit = (units or {}).get(name) or _schema_unit(reported)
            fields.append(Field.from_array(
                name, arr, unit=unit, location=location, components=components,
                description=f"{location} values as {mesh.source.get('format', 'the CAE reader')} wrote them"
                            + ("" if unit == reported else f"; the source recorded '{reported}'")))

    spec = _mesh_spec(mesh, arrays)
    regions = [Region(name=name, kind="boundary", description="named group of the source mesh")
               for name, _ in mesh.boundary_blocks]
    metadata = {"cae": {
        "format": str(mesh.source.get("format", "")), "mesh_kind": mesh.kind, "dimension": int(mesh.dim),
        "points": int(mesh.n_points), "cells": int(mesh.n_cells),
        "source": {k: v for k, v in mesh.source.items() if isinstance(v, (str, int, float, bool))},
    }}
    return PhysicsDataBundle.from_arrays(
        arrays, fields=fields, mesh=spec,
        physics=Physics(notes=notes or "CAE result: the reader records the fields, not the equation that produced "
                                       "them"),
        source=source, license=license, title=title or source, kind=kind,
        geometry=Geometry(kind="mesh", dimension=int(max(1, mesh.dim)),
                          units=geometry_units or _schema_unit(str(mesh.source.get("length_unit", "m"))),
                          regions=regions,
                          description=str(mesh.source.get("format", ""))),
        inputs=file_refs(inputs), citation=citation, url=url,
        adapter="pinneapple_data.schema.cae", adapter_version="1", metadata=metadata)


def _schema_unit(reported: str) -> str:
    """The schema spelling of a reported unit: "model units (length)" is ``unknown``, anything else is itself."""
    from .models import UNKNOWN_UNIT, unit_or_dimensionless

    try:
        return unit_or_dimensionless(reported)
    except ValueError:
        return UNKNOWN_UNIT


def _mesh_spec(mesh: Mesh, arrays: dict[str, np.ndarray]) -> MeshSpec:
    """The schema mesh of a CAE ``Mesh``: nodes, one block per element group, or the faces of a polyhedral mesh."""
    if mesh.poly is not None:
        return _poly_spec(mesh, arrays)
    blocks = [Block(type=block.type, name=f"block{i}",
                    data=ArrayRef(key=f"block:{i}", shape=tuple(int(n) for n in block.conn.shape),
                                  dtype=str(np.asarray(block.conn).dtype)))
              for i, block in enumerate(mesh.blocks)]
    blocks += [Block(type=block.type, name=name, boundary=True,
                     data=ArrayRef(key=f"group:{name}", shape=tuple(int(n) for n in block.conn.shape),
                                   dtype=str(np.asarray(block.conn).dtype)))
               for name, block in mesh.boundary_blocks]
    groups = {name: [str(int(i)) for i in np.asarray(idx).ravel()[:100_000]]
              for name, idx in {**mesh.node_sets, **{f"cells:{k}": v for k, v in mesh.cell_sets.items()}}.items()}
    return MeshSpec(kind="unstructured", dimension=int(max(1, mesh.dim)),
                    nodes=ArrayRef(key="mesh:nodes", shape=tuple(int(n) for n in mesh.points.shape),
                                   dtype=str(np.asarray(mesh.points).dtype)),
                    blocks=blocks, physical_groups=groups)


def _poly_spec(mesh: Mesh, arrays: dict[str, np.ndarray]) -> MeshSpec:
    """OpenFOAM's face-based mesh, kept as faces (expanding a polyhedron into a cell type would be a lossy guess)."""
    from ..cae.model import PolyMesh
    from .models import Patch, Polyhedral

    poly: PolyMesh = mesh.poly
    arrays["poly:face_nodes"] = np.asarray(poly.face_nodes, dtype=np.int64)
    arrays["poly:face_offsets"] = np.asarray(poly.face_offsets, dtype=np.int64)
    arrays["poly:owner"] = np.asarray(poly.owner, dtype=np.int64)
    arrays["poly:neighbour"] = np.asarray(poly.neighbour, dtype=np.int64)

    def ref(key: str) -> ArrayRef:
        arr = arrays[key]
        return ArrayRef(key=key, shape=tuple(int(n) for n in arr.shape), dtype=str(arr.dtype))

    return MeshSpec(
        kind="unstructured", dimension=3,
        nodes=ArrayRef(key="mesh:nodes", shape=tuple(int(n) for n in mesh.points.shape),
                       dtype=str(np.asarray(mesh.points).dtype)),
        poly=Polyhedral(face_nodes=ref("poly:face_nodes"), face_offsets=ref("poly:face_offsets"),
                        owner=ref("poly:owner"), neighbour=ref("poly:neighbour"), n_cells=int(poly.n_cells),
                        patches=[Patch(name=str(p.get("name", "")), type=str(p.get("type", "")),
                                       start_face=int(p.get("startFace", 0)), n_faces=int(p.get("nFaces", 0)))
                                 for p in poly.patches]),
        physical_groups={str(p.get("name", "")): [] for p in poly.patches},
    )


def bundle_to_mesh(bundle: PhysicsDataBundle) -> Mesh:
    """The CAE ``Mesh`` a bundle describes (unstructured bundles only), for the CAE tools downstream."""
    spec = bundle.manifest.mesh
    if spec.kind != "unstructured" or spec.nodes is None:
        raise ValueError(f"bundle {bundle.manifest.id} has a '{spec.kind}' mesh: only an unstructured mesh goes back "
                         "to a CAE Mesh, because a structured grid carries no cell-to-node connectivity")
    points = np.asarray(bundle.array(spec.nodes.key), dtype=np.float64)
    if spec.poly is not None:
        from ..cae.model import PolyMesh
        poly = PolyMesh(
            face_nodes=np.asarray(bundle.array(spec.poly.face_nodes.key), dtype=np.int64),
            face_offsets=np.asarray(bundle.array(spec.poly.face_offsets.key), dtype=np.int64),
            owner=np.asarray(bundle.array(spec.poly.owner.key), dtype=np.int64),
            neighbour=np.asarray(bundle.array(spec.poly.neighbour.key), dtype=np.int64),
            n_cells=int(spec.poly.n_cells),
            patches=[{"name": p.name, "type": p.type, "startFace": p.start_face, "nFaces": p.n_faces}
                     for p in spec.poly.patches])
        variables = bundle.manifest.fields.variables
        return Mesh(
            points=points, poly=poly,
            point_data={f.name: np.asarray(bundle.field_array(f.name)) for f in variables if f.location == "nodal"},
            cell_data={f.name: np.asarray(bundle.field_array(f.name)) for f in variables if f.location == "cell"},
            units={f.name: f.unit for f in variables},
            source={"format": f"bundle {bundle.manifest.id}", "length_unit": bundle.manifest.geometry.units})
    blocks: list[CellBlock] = []
    groups: list[tuple[str, CellBlock]] = []
    for name, etype, ref, is_boundary in spec.all_cells():
        block = CellBlock(type=etype, conn=np.asarray(bundle.array(ref.key), dtype=np.int64))
        if is_boundary:
            groups.append((name or "boundary", block))
        else:
            blocks.append(block)
    variables = bundle.manifest.fields.variables
    return Mesh(
        points=points, blocks=blocks, boundary_blocks=groups,
        point_data={f.name: np.asarray(bundle.field_array(f.name)) for f in variables if f.location == "nodal"},
        cell_data={f.name: np.asarray(bundle.field_array(f.name)) for f in variables if f.location == "cell"},
        units={f.name: f.unit for f in variables},
        source={"format": f"bundle {bundle.manifest.id}",
                "length_unit": bundle.manifest.geometry.units,
                "components": {f.name: list(f.components) for f in variables if f.components}},
    )


def bundle_from_files(files: Mapping[str, bytes], *, source: str, license: str, **kwargs: Any) -> PhysicsDataBundle:
    """Read an uploaded CAE case (a dict of path -> bytes, as the API receives it) into a bundle."""
    from ..cae.io import read_any
    return bundle_from_mesh(read_any(dict(files)), source=source, license=license, **kwargs)


def bundle_from_path(path: str | os.PathLike[str], *, license: str, source: str = "",
                     **kwargs: Any) -> PhysicsDataBundle:
    """Read one file or a folder (an OpenFOAM case directory) from disk into a bundle."""
    target = Path(path)
    if target.is_dir():
        files = {str(item.relative_to(target)): item.read_bytes()
                 for item in sorted(target.rglob("*")) if item.is_file()}
        inputs: tuple[str, ...] = ()
    else:
        files = {target.name: target.read_bytes()}
        inputs = (str(target),)
    return bundle_from_files(files, source=source or f"CAE case {target.name}", license=license,
                             inputs=inputs, **kwargs)


def export_bundle(bundle: PhysicsDataBundle, path: str | os.PathLike[str], fmt: str = "vtu") -> Path:
    """Write ``bundle`` as VTU (binary XML), HDF5, NPZ, Parquet, CSV, JSON or the UPD archive."""
    from ..cae.dataset import build_dataset, export
    fmt = fmt.lower()
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unknown export format '{fmt}'; known: {', '.join(EXPORT_FORMATS)}")
    blob, _media_type, ext = export(build_dataset(bundle_to_mesh(bundle)), fmt)
    target = Path(path)
    if not target.suffix:
        target = target.with_suffix("." + ext)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    return target
