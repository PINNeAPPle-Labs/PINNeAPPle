"""Read the surface field of an Ashton-family case, decoding only the arrays asked for.

The surface files are VTK XML -- ``.vtp`` (PolyData) for the two OpenFOAM datasets, ``.vtu``
(UnstructuredGrid) for WindsorML -- written by ``foamToVTK`` and by Volcano Platforms' writer as *inline*
binary: every ``DataArray`` is base64 in the XML body, prefixed with a ``UInt64`` byte count, and the file is
one long text document. There is no index to seek with, so "partial read" means something specific here:
a 612 MB DrivAerML boundary file holds ~15 arrays, one of which is the field being benchmarked, and this
reader decodes that one and *discards the text of the others as it goes by* rather than materialising them.
The bytes still come off the disk (or off the network, if the caller streams the file); what is saved is the
decode and the memory, which is where the cost of the 8.8 M-point surface actually is.

    >>> from pinneapple_data.ashton.vtk_xml import read_surface
    >>> s = read_surface("run_1/boundary_1.vtp", arrays=["Cp"])   # points, connectivity, Cp; nothing else
    >>> s.n_points, s.n_cells                                      # AhmedML run 1, as the file states it
    (1131049, 1101574)

Compressed blocks (``compressor="vtkZLibDataCompressor"``, which is how the SHIFT surfaces are written) are read
too: the block header and every block are inflated as VTK wrote them.

What is deliberately *not* supported: ``format="appended"`` (the payload after ``<AppendedData>`` would have to be
seeked, and no dataset here writes it) and any compressor other than zlib. Both raise rather than return half a
field.
The writer below is used by the synthetic fixture (:mod:`pinneapple_data.ashton.fixture`) and produces the
same layout in ASCII or inline binary, so the reader is exercised on what it claims to read.
"""
from __future__ import annotations

import base64
import re
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np

__all__ = ["VtkFormatError", "Surface", "read_surface", "write_surface", "face_area_normals",
           "node_area_normals", "surface_quadrature"]

_CHUNK = 1 << 20                 # how much of the file is in hand at a time
_MAX_TAG = 4096                  # a VTK XML tag longer than this is not a tag
_ARRAY_OPEN = b"<DataArray"
_ARRAY_CLOSE = b"</DataArray>"

_NUMERIC = {"Float32": "<f4", "Float64": "<f8", "Int8": "i1", "Int16": "<i2", "Int32": "<i4", "Int64": "<i8",
            "UInt8": "u1", "UInt16": "<u2", "UInt32": "<u4", "UInt64": "<u8"}

_ATTR = re.compile(r"""([A-Za-z_][\w:.-]*)\s*=\s*(?:'([^']*)'|"([^"]*)")""")

#: The arrays that are the mesh, not data on it: they come back as ``connectivity``/``offsets``/``cell_types``.
_TOPOLOGY = ("connectivity", "offsets", "types")
#: VTK writes the geometry arrays without a ``Name``: they are identified by the element they sit in. A reader that
#: insists on the attribute skips the points of every real file and returns a surface with no geometry -- which is what
#: the first SHIFT case did before this map existed.
_IMPLICIT = {"Points": "Points", "Connectivity": "connectivity", "Offsets": "offsets", "Types": "types"}
#: the compressors this reader understands; anything else is named rather than decoded wrongly
_COMPRESSORS = ("vtkZLibDataCompressor",)
_B64_RUN = re.compile(rb"=+")           # the end of one base64 chunk


class VtkFormatError(ValueError):
    """The file is not the VTK XML this reader claims to read."""


@dataclass
class Surface:
    """One case's surface: the points, the cells as connectivity and offsets, and the arrays that were read.

    ``connectivity``/``offsets`` are kept as the raw pair (``offsets[j]:offsets[j + 1]`` slices the vertices of
    cell ``j``) rather than as a rectangular array, because WindsorML's immersed-boundary surface mixes
    triangles and quads and flattening it would mean inventing padding.
    """

    points: np.ndarray                                                       # (n_points, 3), float64
    connectivity: np.ndarray                                                 # (n_cell_points,), int64
    offsets: np.ndarray                                                      # (n_cells + 1,), int64
    grid: str = "polydata"                                                   # polydata | unstructuredgrid
    cell_types: np.ndarray | None = None                                     # (n_cells,), int64, .vtu only
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)
    field_data: dict[str, float] = field(default_factory=dict)               # <FieldData>, e.g. TimeValue
    skipped: tuple[str, ...] = ()                                            # arrays present but not decoded
    #: Every array *name* the file holds, whether or not it was decoded: what a partial read still has to
    #: know, so that asking for ``["Cp"]`` can be resolved before the file is read a second time.
    available_points: tuple[str, ...] = ()
    available_cells: tuple[str, ...] = ()
    bytes_decoded: int = 0

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.offsets.size - 1)

    @property
    def cell_sizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def cells_per_shape(self) -> dict[int, int]:
        """How many cells have 3 vertices, 4 vertices, ... -- the check that says what this surface is made of."""
        sizes, counts = np.unique(self.cell_sizes, return_counts=True)
        return {int(s): int(c) for s, c in zip(sizes, counts, strict=True)}

    def cells_of(self, size: int) -> np.ndarray:
        """The cells with ``size`` vertices, as an ``(m, size)`` array (the ones the force integral can hold)."""
        rows = np.flatnonzero(self.cell_sizes == int(size))
        if rows.size == 0:
            return np.zeros((0, size), dtype=np.int64)
        return self.connectivity[self.offsets[rows][:, None] + np.arange(size)[None, :]]

    def summary(self) -> str:
        return (f"{self.grid}: {self.n_points} points, {self.n_cells} cells "
                f"({', '.join(f'{c}x{s}' for s, c in self.cells_per_shape.items())}), "
                f"point arrays {sorted(self.point_data)}, cell arrays {sorted(self.cell_data)}")


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _attrs(tag: bytes) -> dict[str, str]:
    text = tag.decode("latin-1")
    return {k: (a or b) for k, a, b in _ATTR.findall(text)}


def _decode_compressed(raw: bytes, attrs: dict[str, str], header_type: str, compressor: str) -> np.ndarray:
    """Decode a compressed inline payload: a block header, then the blocks, each one compressed on its own.

    The header is ``[nBlocks, blockSize, lastBlockSize, compressedSize x nBlocks]``, and the third word is the trap:
    it is the size of the final, partial block, and a reader that does not expect it takes the last block's size for
    the first compressed size, reads every block one place off, and finds a payload that does not add up. The word is
    absent in some writers, so both layouts are tried and the one whose compressed sizes sum to exactly the bytes
    that are left is used -- that sum is the only claim about the layout that the file itself can confirm.

    A block whose compressed size equals its uncompressed size was written raw (compression did not pay for it) and
    is taken as it stands; everything else is inflated. Read as if it were an uncompressed array, a compressed
    ``DataArray`` yields an empty or garbled field rather than an error, which is why the misread is caught here:
    the SHIFT surfaces are written this way by default.
    """
    if compressor not in _COMPRESSORS:
        raise VtkFormatError(f"compressor {compressor!r} is not supported here; this reader decodes "
                             f"{', '.join(_COMPRESSORS)} (add it with the same block layout)")
    dtype = np.dtype(_NUMERIC.get(attrs.get("type", "Float32"), "<f4"))
    header = np.dtype("<u8") if header_type.lower() == "uint64" else np.dtype("<u4")
    h = header.itemsize

    def word(index: int) -> int:
        return int(np.frombuffer(raw, dtype=header, count=1, offset=index * h)[0])

    if len(raw) < 2 * h:
        raise VtkFormatError("a compressed DataArray shorter than its own header")
    n_blocks, block_size = word(0), word(1)
    if n_blocks <= 0:
        return np.zeros(0, dtype=dtype)
    for extra in (h, 0):
        start = 2 * h + extra
        if start + n_blocks * h > len(raw):
            continue
        sizes = np.frombuffer(raw, dtype=header, count=n_blocks, offset=start).astype(np.int64)
        if start + n_blocks * h + int(sizes.sum()) != len(raw):
            continue
        last = word(2) if extra else block_size
        wanted = [block_size] * (n_blocks - 1) + [last if 0 < last <= block_size else block_size]
        blocks, pos = [], start + n_blocks * h
        for take, want in zip((int(v) for v in sizes), wanted, strict=True):
            block = raw[pos:pos + take]
            pos += take
            if len(block) != take:
                raise VtkFormatError("a compressed block is shorter than its declared size: the file is truncated")
            out = block if take == want else zlib.decompress(block)
            if len(out) != want:
                raise VtkFormatError(f"a block decompressed to {len(out)} bytes, not the {want} declared")
            blocks.append(out)
        payload = b"".join(blocks)
        if len(payload) % dtype.itemsize:
            raise VtkFormatError(f"{len(payload)} bytes of {dtype} data is not a whole number of values")
        return np.frombuffer(payload, dtype=dtype).copy()
    raise VtkFormatError(f"a compressed DataArray declares {n_blocks} blocks of {block_size} bytes, and the "
                         f"{len(raw)} bytes present hold neither block-header layout")


def _decode_numeric(raw: bytes, attrs: dict[str, str], header_type: str,
                    compressor: str = "") -> np.ndarray:
    """Decode one ``DataArray`` payload: inline base64 blocks, each prefixed by its byte count."""
    dtype = np.dtype(_NUMERIC.get(attrs.get("type", "Float32"), "<f4"))
    if not raw:
        return np.zeros(0, dtype=dtype)
    if compressor:
        return _decode_compressed(raw, attrs, header_type, compressor)
    header = np.dtype("<u8") if header_type.lower() == "uint64" else np.dtype("<u4")
    blocks, pos = [], 0
    while pos + header.itemsize <= len(raw):
        size = int(np.frombuffer(raw, dtype=header, count=1, offset=pos)[0])
        pos += header.itemsize
        if size <= 0 or pos + size > len(raw):
            break
        blocks.append(raw[pos:pos + size])
        pos += size
    if not blocks:                                    # a writer that omitted the block header
        blocks = [raw]
    return np.frombuffer(b"".join(blocks), dtype=dtype)


def _b64(text: str) -> bytes:
    """Decode VTK's inline base64, which is *several* padded streams laid end to end.

    VTK encodes the block header of a compressed array as one base64 stream and the block data as another, and writes
    them one after the other, so a ``==`` sits in the *middle* of the payload -- and ``base64.b64decode`` stops at the
    first padding it meets, which is how a 6.1 MB pressure array arrived at the decoder as 796 bytes. Whitespace is
    not data; each ``=`` run closes a chunk, and the chunk is decoded with its padding so it keeps its byte alignment.

    The text is cut at the first ``<`` because a ``DataArray`` element ends with its own ``<InformationKey>`` children
    when the writer records a range, and their ``location="vtkDataArray"`` attributes carry ``=`` that belong to the
    XML, not to the payload.
    """
    clean = re.sub(rb"\s+", b"", text.split("<", 1)[0].encode("latin-1"))
    out, start = [], 0
    for pad in _B64_RUN.finditer(clean):
        out.append(base64.b64decode(clean[start:pad.end()]))
        start = pad.end()
    if start < len(clean):
        tail = clean[start:]
        out.append(base64.b64decode(tail + b"=" * (-len(tail) % 4)))
    return b"".join(out)


def _decode(text: str, attrs: dict[str, str], header_type: str, compressor: str = "") -> np.ndarray:
    fmt = (attrs.get("format") or "ascii").lower()
    if fmt == "ascii":
        values = np.fromstring(text, sep=" ") if text.strip() else np.zeros(0, dtype=np.float64)  # noqa: DEP001
        dtype = np.dtype(_NUMERIC.get(attrs.get("type", "Float32"), "<f4"))
        return values.astype(dtype, copy=False)
    if fmt == "binary":
        return _decode_numeric(_b64(text), attrs, header_type, compressor)
    if fmt == "appended":
        raise VtkFormatError("VTK 'appended' data is not read here: neither dataset writes it, and seeking into "
                             "<AppendedData> would need a second file format")
    raise VtkFormatError(f"unknown DataArray format {fmt!r}")


def _read_chunks(stream: IO[bytes], wanted: set[str] | None) -> dict[str, object]:
    """One pass over the file: every wanted DataArray decoded, the rest skipped without being held."""
    seen: dict[str, object] = {"arrays": {}, "point_arrays": {}, "cell_arrays": {}, "field_data": {},
                               "skipped": [], "bytes": 0, "piece": {}, "root": {}, "names": {}}
    header_type = "UInt64"
    compressor = ""
    stack: list[bytes] = []
    buf, pos, base = b"", 0, 0

    def _refill(*, keep_from: int | None = None) -> bool:
        """Read one more chunk into the window; False at end of file.

        ``keep_from`` is an **absolute** file offset: everything from there on is kept (an array being decoded is held
        whole), and with ``keep_from=None`` everything already scanned is dropped -- which is what makes skipping an
        array the cost of the read and not of the memory.

        It has to be absolute. As a buffer index it is only correct for the first refill: once the window has been
        trimmed, the same number means a different place, and a second refill of a payload that spans several chunks
        drops part of it. That is not hypothetical -- an 8.5 MB inline array (the first SHIFT surface) was handed to
        the decoder starting 96 KB into itself, and the block header was read as 3.7 billion blocks.
        """
        nonlocal buf, pos, base
        chunk = stream.read(_CHUNK)
        if not chunk:
            return False
        target = (base + pos) if keep_from is None else keep_from
        cut = max(0, target - base)
        buf, pos, base = buf[cut:] + chunk, pos - cut, base + cut
        return True

    def _find(tag: bytes, *, keep_from: int | None) -> int:
        """Position of ``tag`` at or after ``pos``, reading more of the file until it is found; -1 at EOF."""
        absolute = None if keep_from is None else base + keep_from      # fixed once: the window moves under it
        while True:
            found = buf.find(tag, pos)
            if found >= 0:
                return found
            if not _refill(keep_from=absolute):
                return -1

    while True:
        start = buf.find(b"<", pos)
        if start < 0:
            if not _refill(keep_from=None):
                return seen
            continue
        end = buf.find(b">", start)
        if end < 0:
            if not _refill(keep_from=base + start):
                return seen
            continue
        if end - start > _MAX_TAG:                       # not a tag: a stray '<' in the text
            pos = start + 1
            continue
        tag = buf[start:end + 1]
        pos = end + 1
        if tag.startswith(b"</"):
            name = tag[2:-1].strip()
            if stack and stack[-1] == name:
                stack.pop()
            elif name in stack:
                while stack and stack.pop() != name:
                    pass
            continue
        name = tag[1:].split(b" ", 1)[0].split(b">", 1)[0].split(b"/", 1)[0].strip()
        if name == b"VTKFile":
            attrs = _attrs(tag)
            header_type = attrs.get("header_type", "UInt64")
            compressor = attrs.get("compressor", "")
            seen["root"] = attrs
        elif name == b"Piece":
            seen["piece"] = _attrs(tag)
        if tag.endswith(b"/>"):
            continue
        stack.append(name)
        if name != b"DataArray":
            continue
        attrs = _attrs(tag)
        parent = stack[-2] if len(stack) >= 2 else b""
        array_name = attrs.get("Name") or attrs.get("name") or _IMPLICIT.get(parent.decode(errors="ignore"), "")
        if wanted is None or array_name in wanted:
            close = _find(_ARRAY_CLOSE, keep_from=pos)   # the payload is held whole: it is being decoded
            if close < 0:
                raise VtkFormatError("the file ends inside a <DataArray>: it is truncated")
            text = buf[pos:close].decode("latin-1")
            seen["bytes"] = int(seen["bytes"]) + len(text)
            values = _decode(text, attrs, header_type, compressor)
            components = int(attrs.get("NumberOfComponents", "1") or 1)
            if components > 1 and values.size % components == 0:
                values = values.reshape(-1, components)
            target = {"PointData": "point_arrays", "CellData": "cell_arrays", "FieldData": "field_data",
                      "Points": "point_arrays", "Polys": "cell_arrays", "Cells": "cell_arrays",
                      "Connectivity": "cell_arrays"}.get(parent.decode(), "arrays")
            if target == "field_data":
                seen["field_data"][array_name] = float(np.asarray(values).reshape(-1)[0]) if values.size else 0.0
            else:
                seen[target][array_name] = values
                seen["arrays"][array_name] = values
            pos = close + len(_ARRAY_CLOSE)
        else:
            seen["skipped"].append(array_name)
            # skipped: the bytes go by and are dropped, keeping only the overlap a split tag needs
            close = _find(_ARRAY_CLOSE, keep_from=max(0, pos - len(_ARRAY_CLOSE) + 1))
            if close < 0:
                raise VtkFormatError("the file ends inside a <DataArray>: it is truncated")
            pos = close + len(_ARRAY_CLOSE)
        seen["names"][array_name] = parent.decode()
        stack.pop()


def read_surface(path: str | Path, *, arrays: Iterable[str] | None = None) -> Surface:
    """Read one case's surface, decoding ``arrays`` (and always the points and the cell connectivity).

    ``arrays=None`` reads every array in the file. Naming the ones needed is the point of the module: the rest
    are skipped as the scan passes them, so a 612 MB file asking for ``["Cp"]`` decodes a third of what it
    would otherwise. A named array that is not in the file raises -- silently returning fewer fields would make
    a benchmark compare the wrong thing.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"{file} is not there")
    wanted = None if arrays is None else set(arrays) | {"Points", "connectivity", "offsets", "types"}
    with open(file, "rb") as stream:
        seen = _read_chunks(stream, wanted)
    root = seen["root"] or {}
    kind = str(root.get("type", "")).lower()
    if kind not in ("polydata", "unstructuredgrid"):
        raise VtkFormatError(f"{file}: VTK type {root.get('type')!r} is not a surface (PolyData or "
                             "UnstructuredGrid)")
    points = np.asarray(seen["point_arrays"].get("Points", np.zeros((0, 3))), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise VtkFormatError(f"{file}: the Points array is {points.shape}, not (n, 3)")
    connectivity = np.asarray(seen["cell_arrays"].get("connectivity", np.zeros(0)), dtype=np.int64).reshape(-1)
    offsets = np.asarray(seen["cell_arrays"].get("offsets", np.zeros(0)), dtype=np.int64).reshape(-1)
    if offsets.size == 0 and connectivity.size:
        raise VtkFormatError(f"{file}: cells without offsets: this reader slices a cell with offsets[j]:offsets[j+1]")
    # VTK writes `<Offsets>` as the *end* of every cell, so it holds one entry per cell and starts at the first
    # cell's size; a start-offset array holds one more entry and starts at zero. Read as start offsets, an end-offset
    # array gives every cell its neighbour's vertices and drops the last one -- the SHIFT surface lost 1 of its
    # 3,197,220 triangles that way, and with it every point area on the body.
    if offsets.size and int(offsets[0]) != 0 and int(offsets[-1]) == connectivity.size:
        offsets = np.concatenate(([0], offsets))
    if connectivity.size and int(connectivity.max()) >= points.shape[0]:
        raise VtkFormatError(f"{file}: connectivity reaches point {int(connectivity.max())} but the file has "
                             f"{points.shape[0]}")
    if arrays is not None:
        missing = sorted(set(arrays) - set(seen["arrays"]) - {"Points", "connectivity", "offsets", "types"})
        if missing:
            raise KeyError(f"{file} has no array(s) {missing}; it holds {sorted(seen['arrays'])}")
    cell_types = seen["cell_arrays"].get("types")
    point_data = {k: v for k, v in seen["point_arrays"].items() if k != "Points"}
    names = seen["names"]
    available_points = tuple(sorted(n for n, parent in names.items()
                                    if parent == "PointData" and n != "Points"))
    available_cells = tuple(sorted(n for n, parent in names.items()
                                   if parent in ("CellData", "Polys", "Cells") and n not in _TOPOLOGY))
    return Surface(points=points, connectivity=connectivity, offsets=offsets, grid=kind,
                   cell_types=None if cell_types is None else np.asarray(cell_types, dtype=np.int64).reshape(-1),
                   point_data=point_data,
                   cell_data={k: v for k, v in seen["cell_arrays"].items() if k not in _TOPOLOGY},
                   field_data=dict(seen["field_data"]),
                   skipped=tuple(sorted(set(seen["skipped"]))), available_points=available_points,
                   available_cells=available_cells, bytes_decoded=int(seen["bytes"]))


# ---------------------------------------------------------------------------
# geometry: area and normal per face, and the same folded onto the nodes
# ---------------------------------------------------------------------------

def face_area_normals(surface: Surface) -> tuple[np.ndarray, np.ndarray]:
    """Area and unit outward normal of every cell, by Newell's method (which is exact for planar polygons).

    The winding of the file decides the sign; a mesh whose winding is inconsistent would give a normal that
    points inward on some faces, which is the one thing the force integral cannot detect on its own -- so
    :func:`surface_quadrature` reports the closed-body check (the integrated normal should be ~0).
    """
    points = surface.points
    areas = np.zeros(surface.n_cells, dtype=np.float64)
    normals = np.zeros((surface.n_cells, 3), dtype=np.float64)
    for size in surface.cells_per_shape:
        rows = np.flatnonzero(surface.cell_sizes == size)
        cells = surface.cells_of(size)
        if cells.size == 0:
            continue
        v = points[cells]                                              # (m, size, 3)
        cross = np.cross(v, np.roll(v, -1, axis=1)).sum(axis=1)         # (m, 3) = 2 * area * normal
        area = 0.5 * np.linalg.norm(cross, axis=1)
        areas[rows] = area
        normals[rows] = cross / np.maximum(2.0 * area, 1e-30)[:, None]
    return areas, normals


def surface_quadrature(surface: Surface) -> dict[str, float]:
    """What the surface is, numerically: total area, the closed-body normal check, and the cell mix."""
    areas, normals = face_area_normals(surface)
    total = float(areas.sum())
    closed = float(np.linalg.norm((areas[:, None] * normals).sum(axis=0)) / max(total, 1e-30))
    return {"n_points": surface.n_points, "n_cells": surface.n_cells, "area": total,
            "integrated_normal_ratio": closed}


def node_area_normals(surface: Surface) -> tuple[np.ndarray, np.ndarray]:
    """Area and unit normal per point: each face's area vector shared out among its vertices.

    This is the form WindsorML's per-point quadrature weights are used in: the card publishes one weight per
    point, so a pressure on a point acts over that weight along the node's normal. The weight returned here is
    the *magnitude* of the node's summed area vector -- not the sum of the incident faces' areas -- because
    those two are only equal where the incident faces are parallel. Having the magnitude and the unit normal
    from the same vector is what makes ``sum_i w_i n_i`` equal ``sum_f A_f n_f``; using the area sum instead
    biases every edge node, which on a coarse surface is worth tens of percent of the drag.
    """
    areas, normals = face_area_normals(surface)
    node_vec = np.zeros((surface.n_points, 3), dtype=np.float64)
    for size in surface.cells_per_shape:
        rows = np.flatnonzero(surface.cell_sizes == size)
        cells = surface.cells_of(size)
        if cells.size == 0:
            continue
        share = ((areas[rows, None] * normals[rows]) / size)[:, None, :]
        np.add.at(node_vec, cells, np.repeat(share, size, axis=1))
    norm = np.linalg.norm(node_vec, axis=1)
    return norm, node_vec / np.maximum(norm, 1e-30)[:, None]


# ---------------------------------------------------------------------------
# writing (the synthetic fixture)
# ---------------------------------------------------------------------------

def _encode(values: np.ndarray, vtk_type: str, fmt: str, header_type: str) -> str:
    flat = np.ascontiguousarray(np.asarray(values).reshape(-1))
    if fmt == "ascii":
        if np.issubdtype(flat.dtype, np.integer):
            return " ".join(str(int(v)) for v in flat)
        return " ".join(f"{float(v):.9g}" for v in flat)
    payload = flat.astype(_NUMERIC[vtk_type]).tobytes()
    header = np.dtype("<u8" if header_type.lower() == "uint64" else "<u4")
    block = np.array([len(payload)], dtype=header).tobytes() + payload
    return base64.b64encode(block).decode("ascii")


def _vtk_type(dtype) -> str:
    dtype = np.dtype(dtype)
    for name, code in _NUMERIC.items():
        if np.dtype(code) == dtype:
            return name
    return "Float64"


def write_surface(path: str | Path, *, points: np.ndarray, connectivity: np.ndarray, offsets: np.ndarray,
                  point_data: dict[str, np.ndarray] | None = None, cell_data: dict[str, np.ndarray] | None = None,
                  grid: str = "polydata", cell_types: np.ndarray | None = None, fmt: str = "ascii",
                  time: float | None = None) -> Path:
    """Write a surface in the same VTK XML layout (ASCII or inline binary), for the synthetic fixture."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    header_type = "UInt64"
    point_arrays = dict(point_data or {})
    cell_arrays = dict(cell_data or {})
    lines = ["<?xml version='1.0'?>",
             f"<VTKFile type='{'PolyData' if grid == 'polydata' else 'UnstructuredGrid'}' version='0.1' "
             f"byte_order='LittleEndian' header_type='{header_type}'>",
             f"  <{'PolyData' if grid == 'polydata' else 'UnstructuredGrid'}>"]
    if time is not None:
        lines += ["    <FieldData>",
                  f"      <DataArray type='Float32' Name='TimeValue' NumberOfTuples='1' format='{fmt}'>"
                  f"{_encode(np.array([time], dtype=np.float32), 'Float32', fmt, header_type)}</DataArray>",
                  "    </FieldData>"]
    lines.append(f"    <Piece NumberOfPoints='{points.shape[0]}' NumberOfCells='{len(offsets) - 1}'>")

    def array(name: str, values: np.ndarray, components: int = 1) -> str:
        vtk_type = _vtk_type(np.asarray(values).dtype)
        comp = f" NumberOfComponents='{components}'" if components > 1 else ""
        return (f"<DataArray type='{vtk_type}' Name='{name}'{comp} format='{fmt}'>"
                f"{_encode(values, vtk_type, fmt, header_type)}</DataArray>")

    lines.append("      <Points>")
    lines.append(f"        {array('Points', points.astype(np.float32) if fmt == 'binary' else points, 3)}")
    lines.append("      </Points>")
    if point_arrays:
        lines.append("      <PointData>")
        for name, values in point_arrays.items():
            arr = np.asarray(values)
            lines.append(f"        {array(name, arr, arr.shape[1] if arr.ndim == 2 else 1)}")
        lines.append("      </PointData>")
    if cell_arrays:
        lines.append("      <CellData>")
        for name, values in cell_arrays.items():
            arr = np.asarray(values)
            lines.append(f"        {array(name, arr, arr.shape[1] if arr.ndim == 2 else 1)}")
        lines.append("      </CellData>")
    if grid == "polydata":
        lines += ["      <Polys>",
                  f"        {array('connectivity', np.asarray(connectivity, dtype=np.int64))}",
                  # VTK's `<Offsets>` is the *end* of each cell, so the leading zero of a start-offset array is
                  # not written: the fixture then goes through the same branch a real file does
                  f"        {array('offsets', np.asarray(offsets, dtype=np.int64)[1:])}",
                  "      </Polys>"]
    else:
        types = np.asarray(cell_types if cell_types is not None else
                           np.full(len(offsets) - 1, 5, dtype=np.int64), dtype=np.int64)
        lines += ["      <Cells>",
                  f"        {array('connectivity', np.asarray(connectivity, dtype=np.int64))}",
                  f"        {array('offsets', np.asarray(offsets, dtype=np.int64)[1:])}",
                  f"        {array('types', types)}",
                  "      </Cells>"]
    lines += ["    </Piece>",
              f"  </{'PolyData' if grid == 'polydata' else 'UnstructuredGrid'}>",
              "</VTKFile>", ""]
    file.write_text("\n".join(lines), encoding="ascii")
    return file
