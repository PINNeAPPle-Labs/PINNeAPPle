"""XDMF: read and write the mesh and the fields in one file, without loss (#115).

XDMF is an XML description of a mesh plus its data items, with the numbers either inline in the XML or in an HDF5
file next to it. It is what ParaView's "Xdmf3" reader, FEniCS and many solvers exchange, and the format a simulation
result is usually handed over in.

This module writes **XDMF3 with inline XML data** (``Format="XML"``): one file, no companion HDF5, and float64
values written with 17 significant digits, which is the round-trip precision of a double -- so a field read back is
the field written, bit for bit, and the round-trip test asserts equality rather than a tolerance. The layout is the
one ParaView and meshio understand::

    <Xdmf Version="3.0"><Domain>
      <Grid Name="pinneapple" GridType="Collection" CollectionType="Spatial">   <!-- one child per cell block -->
        <Grid Name="block0"><Topology TopologyType="Tetrahedron">...</Topology>
          <Geometry GeometryType="XYZ">...</Geometry>
          <Attribute Name="T" Center="Node" AttributeType="Scalar">...</Attribute>
          <Attribute Name="sigma" Center="Cell" AttributeType="Tensor">...</Attribute>
        </Grid>
      </Grid></Domain></Xdmf>

A mesh with one cell block is written as a plain ``<Grid>``; several blocks become the collection above, each
carrying the same geometry and its own connectivity and cell values, which is how a block-structured result stays
one file.

**Not supported, and refused with a sentence instead of half-read:**

- an XDMF whose data lives in HDF5 (``Format="HDF"`` or ``"Binary"``) -- the common ParaView output. Reading it
  needs the HDF5 companion file, which the meshio path in ``pinneapple_data.cae.io`` already handles when meshio is
  installed; this reader is for the self-contained file this module writes.
- ``Polygon``/``Polyhedron`` topologies (variable nodes per element) and time series (several ``Grid`` steps under
  one domain): neither has a place in the ``Mesh`` container, which holds one snapshot.

``MESHIO_EXT`` in ``io.py`` already lists ``.xdmf``, so an uploaded file of this kind is routed here when meshio is
absent; the two readers agree on the same block/field structure.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from .model import CellBlock, Mesh

# VTK element name (what a CellBlock carries) <-> XDMF3 topology type, for the linear types
VTK_TO_XDMF = {"vertex": "Polyvertex", "line": "Polyline", "triangle": "Triangle", "quad": "Quadrilateral",
               "tetra": "Tetrahedron", "hexahedron": "Hexahedron", "wedge": "Wedge", "pyramid": "Pyramid"}
XDMF_TO_VTK = {v: k for k, v in VTK_TO_XDMF.items()}
TENSOR_SIZES = {3: "Vector", 9: "Tensor"}


class XdmfError(ValueError):
    """A file this reader cannot read, with the reason in the message."""


def _numbers(values: np.ndarray) -> str:
    """The payload of a DataItem: integers as they are, floats at round-trip precision (``%.17g``)."""
    flat = np.asarray(values).reshape(-1)
    if flat.dtype.kind in "iu":
        return " ".join(str(int(v)) for v in flat)
    return " ".join(f"{v:.17g}" for v in flat.astype(np.float64))


def _data_item(parent: ET.Element, values: np.ndarray, number_type: str, precision: int) -> None:
    dims = " ".join(str(int(d)) for d in np.shape(values)) or "1"
    item = ET.SubElement(parent, "DataItem", {"Dimensions": dims, "NumberType": number_type,
                                             "Precision": str(precision), "Format": "XML"})
    item.text = _numbers(values)


def _attribute(parent: ET.Element, name: str, values: Any, center: str) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr, kind = arr.reshape(-1, 1), "Scalar"
    else:
        kind = TENSOR_SIZES.get(arr.shape[1], "Matrix")
    node = ET.SubElement(parent, "Attribute", {"Name": name, "AttributeType": kind, "Center": center})
    _data_item(node, arr, "Float", 8)


def write_xdmf(mesh: Mesh, path: str | Path) -> Path:
    """Write a ``Mesh`` (points, cell blocks, point and cell data) as one self-contained XDMF3 file.

    Raises ``XdmfError`` for what the promise does not cover: a polyhedral (OpenFOAM) mesh, a block type with no
    XDMF topology, a cell field whose length does not match the mesh's cells, and a second data dimension the
    attribute does not carry.
    """
    if getattr(mesh, "poly", None) is not None:
        raise XdmfError("a polyhedral (OpenFOAM) mesh has faces, not cell blocks; XDMF here writes blocks -- "
                        "convert it to a block mesh first, or write the faces as triangles")
    if not mesh.blocks:
        raise XdmfError("the mesh has no cell blocks, so there is no topology to write")
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise XdmfError(f"points must be (n, 3), got {points.shape}")
    n_cells = mesh.n_cells
    for name, values in (mesh.cell_data or {}).items():
        if len(values) != n_cells:
            raise XdmfError(f"cell field {name!r} has {len(values)} rows but the mesh has {n_cells} cells")
    for name, values in (mesh.point_data or {}).items():
        if len(values) != len(points):
            raise XdmfError(f"point field {name!r} has {len(values)} rows but the mesh has {len(points)} points")

    root = ET.Element("Xdmf", {"Version": "3.0"})
    domain = ET.SubElement(root, "Domain")
    blocks = [(i, b) for i, b in enumerate(mesh.blocks)]
    collection = len(blocks) > 1
    parent = ET.SubElement(domain, "Grid", {"Name": "pinneapple", "GridType": "Collection",
                                            "CollectionType": "Spatial"}) if collection else domain
    cell_offset = 0
    for index, block in blocks:
        kind = block.linear if hasattr(block, "linear") else block.type
        if kind not in VTK_TO_XDMF:
            raise XdmfError(f"block {index} is {kind!r}, which has no XDMF3 topology here; known: "
                            f"{', '.join(sorted(VTK_TO_XDMF))}")
        grid = ET.SubElement(parent, "Grid", {"Name": f"block{index}", "GridType": "Uniform"})
        topology = ET.SubElement(grid, "Topology", {"TopologyType": VTK_TO_XDMF[kind],
                                                    "NumberOfElements": str(len(block.conn))})
        _data_item(topology, np.asarray(block.conn, dtype=np.int64), "Int", 8)
        geometry = ET.SubElement(grid, "Geometry", {"GeometryType": "XYZ"})
        _data_item(geometry, points, "Float", 8)
        for name, values in (mesh.point_data or {}).items():
            _attribute(grid, name, values, "Node")
        for name, values in (mesh.cell_data or {}).items():
            field = np.asarray(values, dtype=np.float64)
            field = field[cell_offset:cell_offset + len(block.conn)] if field.ndim > 1 else \
                field[cell_offset:cell_offset + len(block.conn)]
            _attribute(grid, name, field, "Cell")
        cell_offset += len(block.conn)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)                                        # Python 3.9+: the file is read by humans too
    target.write_bytes(b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="utf-8") + b"\n")
    return target


def _read_data_item(node: ET.Element) -> np.ndarray:
    fmt = (node.get("Format") or "XML").upper()
    if fmt != "XML":
        where = (node.text or "").strip()
        raise XdmfError(f'this XDMF keeps its data outside the XML (Format="{node.get("Format")}", pointing at '
                        f"{where or 'no file'}); read it with meshio (pinneapple_data.cae.io does that when it is "
                        "installed), or write it with write_xdmf so the file is self-contained")
    text = (node.text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.float64)
    dtype = np.int64 if (node.get("NumberType") or "Float").lower().startswith(("int", "uint")) else np.float64
    values = np.array(text.split(), dtype=dtype)
    dims = [int(d) for d in (node.get("Dimensions") or "").split()]
    return values.reshape(dims) if dims else values


def read_xdmf(source: str | Path | bytes) -> Mesh:
    """Read an XDMF3 file written with inline XML data back into a ``Mesh``.

    The inverse of :func:`write_xdmf`: one cell block when the file has one ``Grid``, several when it has a
    collection, with the fields at the nodes and at the cells.
    """
    try:
        root = ET.fromstring(source if isinstance(source, bytes) else Path(source).read_bytes())
    except ET.ParseError as e:
        raise XdmfError(f"not an XML document: {e}") from e
    if root.tag != "Xdmf":
        raise XdmfError(f"the root element is <{root.tag}>, not <Xdmf>")

    grids: list[ET.Element] = []

    def collect(node: ET.Element) -> None:
        for child in node:
            if child.tag == "Grid":
                if (child.get("GridType") or "").startswith("Collection"):
                    collect(child)
                else:
                    grids.append(child)
            elif child.tag == "Domain":
                collect(child)

    collect(root)
    if not grids:
        raise XdmfError("the file has no <Grid> with a topology; an XDMF whose data is all in HDF5 is not this "
                        "reader's job")
    if len({child.get("Name") for child in grids}) == 1 and len(grids) > 1:
        raise XdmfError("the file has several Grid steps with the same name: a time series has no place in one "
                        "Mesh; read the step you want")

    points: np.ndarray | None = None
    blocks: list[CellBlock] = []
    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, list[np.ndarray]] = {}
    for grid in grids:
        geometry = grid.find("Geometry")
        topology = grid.find("Topology")
        if geometry is None or topology is None:
            raise XdmfError("a <Grid> is missing its <Geometry> or <Topology>")
        geom_item = geometry.find("DataItem")
        if geom_item is None:
            raise XdmfError("the <Geometry> has no <DataItem>")
        this_points = np.atleast_2d(_read_data_item(geom_item)).astype(np.float64)
        if this_points.shape[1] == 2:                                  # a 2-D mesh: z is zero
            this_points = np.column_stack([this_points, np.zeros(len(this_points))])
        if points is None:
            points = this_points
        elif not np.array_equal(points, this_points):
            raise XdmfError("the grids carry different points; a Mesh holds one point array, so this file cannot "
                            "be read as one mesh without losing which block each point belongs to")

        topo_type = topology.get("TopologyType") or ""
        if topo_type not in XDMF_TO_VTK:
            raise XdmfError(f"topology {topo_type!r} is not one of the supported linear types: "
                            f"{', '.join(sorted(XDMF_TO_VTK))}")
        conn = np.atleast_2d(_read_data_item(topology.find("DataItem"))).astype(np.int64)
        blocks.append(CellBlock(XDMF_TO_VTK[topo_type], conn))

        for attr in grid.findall("Attribute"):
            name = attr.get("Name") or f"field{len(point_data) + len(cell_data)}"
            values = _read_data_item(attr.find("DataItem"))
            center = (attr.get("Center") or "Node").lower()
            if center == "node":
                point_data[name] = values.reshape(len(values), -1)[:, 0] if values.ndim > 1 and \
                    values.shape[1] == 1 else values
            elif center == "cell":
                cell_data.setdefault(name, []).append(values)
            else:
                raise XdmfError(f'attribute {name!r} is at Center="{attr.get("Center")}", which is neither Node '
                                "nor Cell; this reader does not interpolate")

    mesh = Mesh(points=points if points is not None else np.zeros((0, 3)), blocks=blocks,
                source={"format": "xdmf", "file": os.path.basename(source) if not isinstance(source, bytes) else ""})
    mesh.point_data.update(point_data)
    for name, parts in cell_data.items():
        merged = [p.reshape(len(p), -1) if p.ndim > 1 else p.reshape(-1, 1) for p in parts]
        stacked = np.concatenate(merged)
        mesh.cell_data[name] = stacked[:, 0] if stacked.shape[1] == 1 else stacked
    return mesh
