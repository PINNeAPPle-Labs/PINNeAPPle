"""The DrivAerNet++ surface contract: read the released surface, and score the one field it is judged on.

The FluidsBench spec for this dataset is the cheapest of the four open 3-D ones — a single scalar, the surface pressure
`p`, at every native point of the geometry, weighted by the point's dual area. Two metrics are bound to it
(`surface_pressure_rel_l2`, area-weighted, and `surface_pressure_equal_entity_rel_l2`, uniform) and they are the whole
composite, aggregated per geometry and then macro-averaged over the designs.

What this module deliberately does *not* do is invent a way to get the data. The release is fifteen family ZIPs behind
a Harvard Dataverse guestbook (doi:10.7910/DVN/K7PWNJ, 212 GB, CC BY-NC 4.0), and the per-case scoring support ships
without coordinates — so a run needs the geometry, and the geometry needs whoever holds the account to accept the
guestbook. Everything below is the part that can be written and tested before that: the VTK reader, the support reader
and the exact reductions.

Verified against a real case (`E_S_WWC_WM_008`): 511 063 points, 474 686 polygons, dual areas summing to 32.35 m²,
`p` from −1 853.8 to 449.9 m²/s² at a dynamic head of 450 m²/s².
"""
from __future__ import annotations

import numpy as np

__all__ = ["read_legacy_polydata", "load_support", "surface_pressure_metrics", "aggregate_metrics"]

#: the type names VTK's legacy writer puts in its headers, and the big-endian dtype each one means
_TYPES = {"bit": None, "char": ">i1", "signed_char": ">i1", "unsigned_char": ">u1", "short": ">i2",
          "unsigned_short": ">u2", "int": ">i4", "unsigned_int": ">u4", "long": ">i8", "unsigned_long": ">u8",
          "float": ">f4", "double": ">f8", "vtktypeint64": ">i8", "vtktypeuint64": ">u8"}


def _dtype(token: bytes) -> str:
    name = token.decode().lower()
    if name not in _TYPES or _TYPES[name] is None:
        raise ValueError(f"unsupported VTK array type '{name}'")
    return _TYPES[name]


def read_legacy_polydata(path: str) -> dict[str, np.ndarray]:
    """Points and point-data arrays of a binary VTK legacy ``POLYDATA`` file.

    The release's surfaces are written by VTK's C++ writer: big-endian binary, one ``POINTS`` block, then
    ``POINT_DATA`` with its arrays. Anything else — ASCII, compressed, cell data ahead of point data — raises instead of
    being guessed at, because a silent misread of a 500 000-point surface is worse than a refusal.
    """
    with open(path, "rb") as handle:
        blob = handle.read()
    if b"DATASET POLYDATA" not in blob[:512]:
        raise ValueError(f"{path}: not a VTK legacy POLYDATA file")
    if b"BINARY" not in blob[:512]:
        raise ValueError(f"{path}: only the release's binary form is supported")

    out: dict[str, np.ndarray] = {}
    position, total = 0, len(blob)
    while position < total:
        end = blob.find(b"\n", position)
        if end < 0:
            break
        header = blob[position:end].strip()
        position = end + 1
        fields = header.split()
        if not fields:
            continue
        keyword = fields[0].upper()

        if keyword == b"POINTS":
            count, dtype = int(fields[1]), _dtype(fields[2])
            width = np.dtype(dtype).itemsize
            out["points"] = np.frombuffer(blob, dtype=dtype, count=3 * count,
                                          offset=position).reshape(count, 3).astype(np.float64)
            position += 3 * count * width
        elif keyword in (b"SCALARS", b"COLOR_SCALARS"):
            # SCALARS <name> <type> [<components>] then a LOOKUP_TABLE line the values follow
            name, dtype = fields[1].decode(), _dtype(fields[2])
            components = int(fields[3]) if len(fields) > 3 else 1
            lookup = blob.find(b"\n", position)
            if blob[position:lookup].strip().upper().startswith(b"LOOKUP_TABLE"):
                position = lookup + 1
            count = len(out["points"])
            values = np.frombuffer(blob, dtype=dtype, count=count * components,
                                   offset=position).reshape(count, components).astype(np.float64)
            out[name] = values[:, 0] if components == 1 else values
            position += count * components * np.dtype(dtype).itemsize
        elif keyword in (b"VECTORS", b"NORMALS"):
            name, dtype = fields[1].decode(), _dtype(fields[2])
            count = len(out["points"])
            out[name] = np.frombuffer(blob, dtype=dtype, count=3 * count,
                                      offset=position).reshape(count, 3).astype(np.float64)
            position += 3 * count * np.dtype(dtype).itemsize
        elif keyword == b"POINT_DATA":
            continue                                  # the section header itself carries nothing
        elif keyword == b"POLYGONS":
            count = int(fields[1])
            integers = np.dtype(">i4").itemsize
            # POLYGONS <cells> <size> : <size> integers (each cell is n, i0..in-1)
            out["polygon_count"] = np.array([count])
            position += int(fields[2]) * integers
        elif keyword in (b"CELL_DATA", b"LINES", b"TRIANGLE_STRIPS", b"VERTICES"):
            break                                     # nothing scored lives past this
    if "points" not in out:
        raise ValueError(f"{path}: no POINTS section")
    return out


def load_support(path: str) -> dict[str, np.ndarray]:
    """One case's scoring support: the truth, the area weights and the profile selections."""
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    for required in ("p", "dual_area_m2"):
        if required not in arrays:
            raise ValueError(f"{path}: the support has no '{required}' array")
    return arrays


def surface_pressure_metrics(prediction, truth, dual_area) -> dict[str, float]:
    """The two published reductions, as percentages: area-weighted and uniform relative L2.

    ``100·sqrt(Σ w (p̂ − p)² / Σ w p²)``, which is what ``relative_l2_percent`` means in the spec; the weighting is the
    point's dual area for the first and 1 for the second. Both are per geometry, so this is one case's answer."""
    truth = np.asarray(truth, float)
    error = np.asarray(prediction, float) - truth
    if error.shape != truth.shape:
        raise ValueError(f"prediction {error.shape} does not match the support's {truth.shape} points")
    area = np.asarray(dual_area, float)
    weighted = float(np.sqrt(np.sum(area * error ** 2) / np.sum(area * truth ** 2)))
    uniform = float(np.sqrt(np.sum(error ** 2) / np.sum(truth ** 2)))
    return {"surface_pressure_rel_l2": 100.0 * weighted, "surface_pressure_equal_entity_rel_l2": 100.0 * uniform}


def aggregate_metrics(per_case: list[dict[str, float]]) -> dict[str, float]:
    """The macro average over geometries — the aggregation the spec declares for both metrics."""
    if not per_case:
        raise ValueError("no case to aggregate")
    names = per_case[0]
    return {name: float(np.mean([case[name] for case in per_case])) for name in names}
