"""OpenFOAM readers: polyMesh (points, faces, owner, neighbour, boundary) and volume fields of a time directory.

ASCII and binary files, plain or gzipped (``.gz``), ``faceList`` and ``faceCompactList``. Decomposed cases
(``processor*``) are not reassembled. Nothing is executed: ``#include``/``#codeStream`` in field files are reported by
the metadata extractor, here only the data lists are read.
"""
from __future__ import annotations

import gzip
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import Mesh, PolyMesh

_DIMS = ("kg", "m", "s", "K", "mol", "A", "cd")
_NAMED_UNITS = {(1, -1, -2, 0, 0, 0, 0): "Pa", (0, 2, -2, 0, 0, 0, 0): "m2/s2", (0, 1, -1, 0, 0, 0, 0): "m/s",
                (0, 0, 0, 1, 0, 0, 0): "K", (0, 2, -1, 0, 0, 0, 0): "m2/s", (0, 0, -1, 0, 0, 0, 0): "1/s",
                (1, -3, 0, 0, 0, 0, 0): "kg/m3", (1, -1, -1, 0, 0, 0, 0): "Pa.s", (0, 0, 0, 0, 0, 0, 0): "1",
                (1, 2, -3, 0, 0, 0, 0): "W", (1, 0, -3, 0, 0, 0, 0): "W/m2", (0, 1, 0, 0, 0, 0, 0): "m"}


def _text(b: bytes) -> str:
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    return b.decode("latin-1")


def header(text: str) -> Dict[str, str]:
    m = re.search(r"FoamFile\s*\{(.*?)\}", text, re.S)
    out: Dict[str, str] = {}
    if m:
        for k, v in re.findall(r"(\w+)\s+([^;]+);", m.group(1)):
            out[k] = v.strip().strip('"')
    return out


def _arch(h: Dict[str, str]) -> Tuple[np.dtype, np.dtype]:
    a = h.get("arch", "")
    lab = np.dtype("<i8") if "label=64" in a else np.dtype("<i4")
    sca = np.dtype("<f4") if "scalar=32" in a else np.dtype("<f8")
    if "MSB" in a:
        lab, sca = lab.newbyteorder(">"), sca.newbyteorder(">")
    return lab, sca


_LIST = re.compile(r"(?:\s|//[^\n]*|/\*.*?\*/)*(\d+)\s*([({])", re.S)


def _body_start(text: str) -> int:
    m = re.search(r"FoamFile\s*\{.*?\}", text, re.S)
    return m.end() if m else 0


def _list_at(text: str, pos: int, binary: bool, dtype: np.dtype, width: int) -> Tuple[np.ndarray, int]:
    """Read ``N ( ... )`` (or ``N{value}``) starting at ``pos``; returns (array, position after it)."""
    m = _LIST.match(text, pos)
    if not m:
        raise ValueError("expected a list")
    n = int(m.group(1))
    if m.group(2) == "{":                                             # uniform list: N{value}
        end = text.index("}", m.end())
        v = np.array(text[m.end():end].replace("(", " ").replace(")", " ").split(), dtype=dtype)
        return np.tile(v, (n, 1)).reshape(n, width) if width > 1 else np.full(n, v[0], dtype=dtype), end + 1
    start = m.end()
    if binary:
        nbytes = n * width * dtype.itemsize
        raw = text[start:start + nbytes].encode("latin-1")
        arr = np.frombuffer(raw, dtype=dtype).astype(dtype.newbyteorder("="))
        end = start + nbytes + 1
    else:
        if width == 1:
            end_inner = text.index(")", start)
        else:                                                         # "(x y z)\n...(x y z)\n)": first "))"
            mm = re.compile(r"\)\s*\)").search(text, start) if n else re.compile(r"\s*\)").match(text, start)
            end_inner = (mm.start() + 1) if n else mm.start()
        end = text.index(")", end_inner) + 1 if width > 1 and n else end_inner + 1
        s = text[start:end_inner].replace("(", " ").replace(")", " ")
        arr = np.array(s.split(), dtype=dtype) if n else np.zeros(0, dtype)
        if len(arr) != n * width:
            raise ValueError(f"list of {n} entries has {len(arr) / max(width, 1):g}")
    return (arr.reshape(n, width) if width > 1 else arr), end


def read_points(b: bytes) -> np.ndarray:
    t = _text(b)
    h = header(t)
    _, sca = _arch(h)
    arr, _ = _list_at(t, _body_start(t), h.get("format", "ascii") == "binary", sca, 3)
    return np.asarray(arr, float)


def read_labels(b: bytes) -> np.ndarray:
    t = _text(b)
    h = header(t)
    lab, _ = _arch(h)
    arr, _ = _list_at(t, _body_start(t), h.get("format", "ascii") == "binary", lab, 1)
    return np.asarray(arr, np.int64)


def read_faces(b: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Faces as (flat node list, offsets)."""
    t = _text(b)
    h = header(t)
    lab, _ = _arch(h)
    binary = h.get("format", "ascii") == "binary"
    pos = _body_start(t)
    if "Compact" in h.get("class", "") or binary:
        offsets, pos = _list_at(t, pos, binary, lab, 1)
        flat, _ = _list_at(t, pos, binary, lab, 1)
        return np.asarray(flat, np.int64), np.asarray(offsets, np.int64)
    m = _LIST.match(t, pos)
    n = int(m.group(1))
    end = t.rindex(")")
    tok = np.array(t[m.end():end].replace("(", " ").replace(")", " ").split(), dtype=np.int64)
    counts = np.empty(n, np.int64)
    flat_idx = np.empty(len(tok) - n, np.int64)
    i = k = 0
    for f in range(n):                                               # tokens: n0 a b c ... n1 a b c ...
        c = tok[i]
        counts[f] = c
        flat_idx[k:k + c] = tok[i + 1:i + 1 + c]
        i += c + 1
        k += c
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return flat_idx[: offsets[-1]], offsets


def read_boundary(b: bytes) -> List[Dict[str, Any]]:
    t = _text(b)
    body = t[_body_start(t):]
    out = []
    for name, blk in re.findall(r"([\w.:\-]+)\s*\{([^{}]*)\}", body):
        d = dict(re.findall(r"(\w+)\s+([^;]+);", blk))
        if "nFaces" in d:
            out.append({"name": name, "type": d.get("type", "patch").strip(), "nFaces": int(d["nFaces"]),
                        "startFace": int(d["startFace"]), "inGroups": d.get("inGroups", "")})
    return out


def _poly_dir(fs: Dict[str, bytes], root: str) -> Optional[str]:
    for cand in (root + "constant/polyMesh/", root + "polyMesh/", root):
        if any(k.startswith(cand + "points") for k in fs) and any(k.startswith(cand + "owner") for k in fs):
            return cand
    return None


def _get(fs: Dict[str, bytes], base: str) -> Optional[bytes]:
    return fs.get(base) if base in fs else fs.get(base + ".gz")


def read_polymesh(fs: Dict[str, bytes], root: str = "") -> Mesh:
    d = _poly_dir(fs, root)
    if d is None:
        raise ValueError("no constant/polyMesh (points, faces, owner, neighbour, boundary) in the upload; "
                         "run blockMesh/snappyHexMesh first or include the polyMesh folder")
    pts = read_points(_get(fs, d + "points"))
    flat, offs = read_faces(_get(fs, d + "faces"))
    own = read_labels(_get(fs, d + "owner"))
    nb_b = _get(fs, d + "neighbour")
    nei = read_labels(nb_b) if nb_b is not None else np.zeros(0, np.int64)
    bnd = read_boundary(_get(fs, d + "boundary")) if _get(fs, d + "boundary") is not None else []
    n_cells = int(max(own.max(initial=-1), nei.max(initial=-1)) + 1)
    poly = PolyMesh(flat, offs, own, nei, n_cells, bnd)
    m = Mesh(points=pts, poly=poly, source={"format": "OpenFOAM polyMesh", "dir": d})
    for z in ("cellZones",):
        zb = _get(fs, d + z)
        if zb is not None:
            try:
                m.cell_sets.update(_zones(_text(zb)))
            except Exception:                                         # zones are optional, never fatal
                pass
    return m


def _zones(t: str) -> Dict[str, np.ndarray]:
    out = {}
    for name, blk in re.findall(r"(\w+)\s*\{(.*?)\}", t[_body_start(t):], re.S):
        m = re.search(r"cellLabels\s+List<label>\s+(\d+)\s*\((.*?)\)", blk, re.S)
        if m:
            out[name] = np.array(m.group(2).split(), dtype=np.int64)
    return out


# ------------------------------------------------------------------------------------------------- fields
def dims_unit(dims: str) -> Optional[str]:
    v = re.findall(r"-?\d+(?:\.\d+)?", dims or "")
    if len(v) < 5:
        return None
    e = tuple(int(float(x)) for x in (v + ["0"] * 7)[:7])
    if e in _NAMED_UNITS:
        return _NAMED_UNITS[e]
    num = [f"{u}{abs(p) if abs(p) != 1 else ''}" for u, p in zip(_DIMS, e) if p > 0]
    den = [f"{u}{abs(p) if abs(p) != 1 else ''}" for u, p in zip(_DIMS, e) if p < 0]
    return (".".join(num) or "1") + ("/" + ".".join(den) if den else "")


def read_field(b: bytes, n_cells: int) -> Tuple[np.ndarray, Optional[str], str]:
    """internalField of a vol*Field: (values per cell, unit, class)."""
    t = _text(b)
    h = header(t)
    cls = h.get("class", "")
    width = {"volScalarField": 1, "volVectorField": 3, "volSymmTensorField": 6, "volTensorField": 9,
             "volSphericalTensorField": 1}.get(cls)
    if width is None:
        raise ValueError(f"not a volume field ({cls or 'no class'})")
    _, sca = _arch(h)
    dm = re.search(r"dimensions\s*(\[[^\]]*\])", t)
    unit = dims_unit(dm.group(1)) if dm else None
    m = re.search(r"internalField\s+(uniform|nonuniform)\s*", t)
    if not m:
        raise ValueError("no internalField")
    if m.group(1) == "uniform":
        end = t.index(";", m.end())
        v = np.array(t[m.end():end].replace("(", " ").replace(")", " ").split(), dtype=float)
        arr = np.tile(v, (n_cells, 1)) if width > 1 else np.full(n_cells, v[0])
    else:
        lm = re.compile(r"List<\w+>\s*").match(t, m.end())
        pos = lm.end() if lm else m.end()
        arr, _ = _list_at(t, pos, h.get("format", "ascii") == "binary", sca, width)
        arr = np.asarray(arr, float)
        if len(arr) != n_cells:
            raise ValueError(f"field has {len(arr)} values for {n_cells} cells")
    return arr, unit, cls


def time_dirs(fs: Dict[str, bytes], root: str = "") -> List[str]:
    """Numeric time directories under the case root, ascending."""
    names = set()
    for k in fs:
        if not k.startswith(root):
            continue
        part = k[len(root):].split("/")
        if len(part) >= 2:
            try:
                float(part[0])
                names.add(part[0])
            except ValueError:
                pass
    return sorted(names, key=float)


def read_time(fs: Dict[str, bytes], mesh: Mesh, root: str = "", time: Optional[str] = None,
              only: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load the volume fields of one time directory (latest by default) into ``mesh.cell_data``."""
    times = time_dirs(fs, root)
    if not times:
        return {"time": None, "fields": [], "skipped": []}
    t = time if time is not None else times[-1]
    loaded, skipped = [], []
    for k in sorted(fs):
        if not k.startswith(f"{root}{t}/") or k.count("/") != root.count("/") + 1:
            continue
        name = k.split("/")[-1].removesuffix(".gz")
        if only and name not in only:
            continue
        try:
            arr, unit, cls = read_field(fs[k], mesh.n_cells)
        except Exception as e:                                        # not a vol field, or a uniform/patch file
            skipped.append({"file": k, "reason": str(e)[:120]})
            continue
        mesh.cell_data[name] = arr
        if unit:
            mesh.units[name] = unit
        loaded.append(name)
    mesh.source.update(time=float(t), times=[float(x) for x in times])
    return {"time": float(t), "fields": loaded, "skipped": skipped}
