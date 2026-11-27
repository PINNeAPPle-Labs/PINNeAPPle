"""CGNS (CFD General Notation System) writer, through the CGNS/HDF5 mapping (#115).

This is the other half of ``cgns_reader.py``: what that module reads, this one writes. A CGNS/HDF5 file is a plain
HDF5 file where every CGNS node is one HDF5 group carrying a ``label`` attribute (the SIDS node type) and a
``type`` attribute (the CGNS data-type code, ``I8``/``R8``/``C1``/``MT``), and, for a leaf, one dataset named
literally ``" data"`` -- leading space included -- holding the payload. The layout written here is the one the
reference implementation produces for a single unstructured zone:

    CGNSLibraryVersion  (R4)
    Base                (CGNSBase_t, I4 [cell dim, phys dim])
      Zone1             (Zone_t, I8 [[n vertices], [n elements], [0]])
        ZoneType        (C1 "Unstructured")
        GridCoordinates (MT)
          CoordinateX/Y/Z  (DataArray_t, R8)
        Elements        (Elements_t, I4 [element type, 0])
          ElementRange  (IndexRange_t, I8 [first, last])
          ElementConnectivity (DataArray_t, I8, 1-based)
        FlowSolution    (MT, + GridLocation when not Vertex)
          <field>       (DataArray_t, R8)

The tree was matched node for node against ``tests/fixtures/cfd_formats/real_cgns_mll.cgns``, which was written by
a C program linked against the official CGNS Mid-Level Library, and the files this writer produces pass the CGNS
project's own ``cgnscheck`` validator (see ``tests/test_cgns_writer.py``).

Covered: one unstructured zone, ``TETRA_4``/``HEXA_8``/``PENTA_6``/``PYRA_5``/``TRI_3``/``QUAD_4``/``BAR_2``/
``NODE`` elements, and vertex or cell-centre flow solutions. **Not** covered, and not pretended: multiple zones per
base, structured (``SIDS``) zones, boundary conditions, and the ADF (non-HDF5) backend -- CGNS 3.x files. A file
written here is CGNS 3.4+/HDF5, which is what ParaView, SU2 and the CGNS tools read.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

# CGNS element type codes (SIDS: the first value of Elements_t's data)
ELEMENT_CODES = {
    "NODE": 2,
    "BAR_2": 3,
    "TRI_3": 5,
    "QUAD_4": 7,
    "TETRA_4": 10,
    "PYRA_5": 12,
    "PENTA_6": 14,
    "HEXA_8": 17,
}
# how many nodes each element code holds, to check the connectivity the caller passed
ELEMENT_NODES = {"NODE": 1, "BAR_2": 2, "TRI_3": 3, "QUAD_4": 4, "TETRA_4": 4, "PYRA_5": 5, "PENTA_6": 6,
                 "HEXA_8": 8}
# VTK element names (what a Mesh.CellBlock carries) -> the CGNS code, for the linear types
VTK_TO_CGNS = {"vertex": "NODE", "line": "BAR_2", "triangle": "TRI_3", "quad": "QUAD_4", "tetra": "TETRA_4",
               "hexahedron": "HEXA_8", "wedge": "PENTA_6", "pyramid": "PYRA_5"}
LIBRARY_VERSION = 4.52


def write_cgns(path: str | Path, points: Any, cells: Any, fields: Mapping[str, Any] | None = None, *,
               cell_type: str = "TETRA_4", base_name: str = "Base", zone_name: str = "Zone1",
               location: str = "Vertex") -> Path:
    """Write one unstructured zone to a CGNS/HDF5 file, 1-based connectivity included.

    Parameters
    ----------
    points : (n, 3) node coordinates, written as ``R8`` so they round-trip without loss.
    cells : (m, k) connectivity, **0-based** like every other array in this library; the 1-based numbering CGNS
        requires is applied here, in one place.
    fields : ``{name: values}`` at the nodes (``location="Vertex"``, the default) or the cells
        (``location="CellCenter"``); ``(n,)`` for a scalar, ``(n, k)`` for a vector, flattened as CGNS stores it.
    cell_type : one of :data:`ELEMENT_CODES`; the whole zone is one element block, which is what the reader reads.

    Raises ``ValueError`` for what would otherwise become a file a solver rejects: an unknown element type, a
    connectivity matrix whose column count does not match it, node indices outside the point array, non-finite
    coordinates, and a field with the wrong length for the location it claims.
    """
    import h5py

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (n, 3), got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points contain NaN or Inf; a CGNS file cannot carry them and a solver will not guess")
    if cell_type not in ELEMENT_CODES:
        raise ValueError(f"unknown element type {cell_type!r}; known: {', '.join(ELEMENT_CODES)}")
    conn = np.asarray(cells, dtype=np.int64)
    if conn.ndim == 1 and ELEMENT_NODES[cell_type] == 1:
        conn = conn.reshape(-1, 1)
    if conn.ndim != 2 or conn.shape[1] != ELEMENT_NODES[cell_type]:
        raise ValueError(f"{cell_type} needs {ELEMENT_NODES[cell_type]} nodes per element, got shape {conn.shape}")
    if conn.size and (conn.min() < 0 or conn.max() >= len(pts)):
        raise ValueError(f"connectivity refers to nodes outside 0..{len(pts) - 1}; it is 0-based here and becomes "
                         "1-based in the file")
    n_per = len(pts) if location == "Vertex" else len(conn)
    stored: dict[str, np.ndarray] = {}
    for name, values in (fields or {}).items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if len(arr) != n_per:
            raise ValueError(f"field {name!r} has {len(arr)} values but the zone has {n_per} "
                             f"{'nodes' if location == 'Vertex' else 'cells'}")
        stored[name] = arr
    if location not in ("Vertex", "CellCenter"):
        raise ValueError(f"location must be 'Vertex' or 'CellCenter', got {location!r}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as f:
        f.attrs.create("label", "Root Node of HDF5 File", dtype=_string_type())
        f.attrs.create("name", "HDF5 MotherNode", dtype=_string_type())
        f.attrs.create("type", "MT", dtype=_string_type(3))
        f.attrs.create("flags", np.array([1], dtype=np.int32))
        # what the reference writer leaves at the root: the floating-point format and the HDF5 library
        f.create_dataset(" format", data=np.frombuffer(b"IEEE_LITTLE_32 ", dtype=np.int8))
        f.create_dataset(" hdf5version", data=np.frombuffer(_hdf5_version(), dtype=np.int8))

        lib = _node(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t", "R4")
        lib.create_dataset(" data", data=np.array([LIBRARY_VERSION], dtype=np.float32))

        base = _node(f, base_name, "CGNSBase_t", "I4")
        base.create_dataset(" data", data=np.array([3, 3], dtype=np.int32))

        zone = _node(base, zone_name, "Zone_t", "I8")
        zone.create_dataset(" data", data=np.array([[len(pts)], [len(conn)], [0]], dtype=np.int64))
        zone_type = _node(zone, "ZoneType", "ZoneType_t", "C1")
        zone_type.create_dataset(" data", data=np.frombuffer(b"Unstructured", dtype=np.int8))

        coords = _node(zone, "GridCoordinates", "GridCoordinates_t", "MT")
        for axis, name in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
            _node(coords, name, "DataArray_t", "R8").create_dataset(" data", data=pts[:, axis])

        elements = _node(zone, "Elements", "Elements_t", "I4")
        elements.create_dataset(" data", data=np.array([ELEMENT_CODES[cell_type], 0], dtype=np.int32))
        element_range = _node(elements, "ElementRange", "IndexRange_t", "I8")
        element_range.create_dataset(" data", data=np.array([1, max(len(conn), 0)], dtype=np.int64))
        connectivity = _node(elements, "ElementConnectivity", "DataArray_t", "I8")
        connectivity.create_dataset(" data", data=(conn + 1).reshape(-1).astype(np.int64))

        if stored or location != "Vertex":
            solution = _node(zone, "FlowSolution", "FlowSolution_t", "MT")
            if location != "Vertex":
                loc = _node(solution, "GridLocation", "GridLocation_t", "C1")
                loc.create_dataset(" data", data=np.frombuffer(location.encode(), dtype=np.int8))
            for name, arr in stored.items():
                _node(solution, name, "DataArray_t", "R8").create_dataset(" data", data=arr.reshape(-1))
    return target


def _string_type(size: int = 33) -> Any:
    """A fixed-size ASCII attribute type, NULL-terminated -- the types ``libcgns`` writes.

    The size and the padding are not cosmetic. ``label`` and ``name`` are 33 bytes; ``type`` is exactly
    ``len(value) + 1``. The reference library's HDF5 layer reads these attributes into fixed buffers, and a file
    whose ``type`` attribute was 33 bytes crashed ``cgnscheck`` with a segfault -- a first version of this writer
    did exactly that, which is how the sizes were pinned down.
    """
    import h5py

    kind = h5py.h5t.C_S1.copy()
    kind.set_size(size)
    kind.set_strpad(h5py.h5t.STR_NULLTERM)
    return h5py.Datatype(kind)


def _node(parent: Any, name: str, label: str, cgns_type: str) -> Any:
    """One CGNS node: an HDF5 group with the four attributes ``libcgns`` writes for every node.

    ``label`` (the SIDS node type), ``type`` (the data-type code), ``name`` (the node's own name, which the
    reference library reads back and dereferences: a node without it crashed ``cgnscheck`` with a segfault rather
    than reporting a missing attribute) and ``flags`` (``[1]``, int32 -- "this node was created", in the mapping's
    own terms).
    """
    group = parent.create_group(name)
    group.attrs.create("label", label, dtype=_string_type())
    group.attrs.create("type", cgns_type, dtype=_string_type(len(cgns_type) + 1))
    group.attrs.create("name", name, dtype=_string_type())
    group.attrs.create("flags", np.array([1], dtype=np.int32))
    return group


def _hdf5_version() -> bytes:
    """The `` hdf5version`` leaf the reference writer leaves at the root, padded the way it pads it."""
    import h5py

    return f"HDF5 Version {h5py.version.hdf5_version}".ljust(32).encode()


def mesh_to_cgns(path: str | Path, mesh: Any, *, cells: Iterable[int] | None = None, **kwargs: Any) -> Path:
    """Write a ``pinneapple_data.cae.Mesh`` (its points, one cell block and its point fields) as one CGNS zone.

    ``cells`` selects which block of the mesh to write -- the first by default. A mesh with a polyhedral
    (OpenFOAM) representation has no single element type and is refused with that sentence rather than silently
    written as something else.
    """
    if getattr(mesh, "poly", None) is not None:
        raise ValueError("a polyhedral (OpenFOAM) mesh has no single CGNS element type; write the blocks you mean "
                         "with write_cgns(point, cells, cell_type=...)")
    block = mesh.blocks[0] if cells is None else mesh.blocks[int(cells)]
    vtk_type = block.linear if hasattr(block, "linear") else block.type
    if vtk_type not in VTK_TO_CGNS:
        raise ValueError(f"the block is {vtk_type!r}, which has no linear CGNS counterpart; known: "
                         f"{', '.join(sorted(VTK_TO_CGNS))}")
    fields = {name: values[:, 0] if getattr(values, "ndim", 1) == 2 and values.shape[1] == 1 else values
              for name, values in (mesh.point_data or {}).items()}
    return write_cgns(path, mesh.points, block.corners, fields, cell_type=VTK_TO_CGNS[vtk_type], **kwargs)
