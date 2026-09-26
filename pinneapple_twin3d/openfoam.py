"""OpenFOAM case -> Twin3D scene: wall patches as surfaces, per-face values over time.

Built for the PINNeAPPle-CFD erosion maps (erosion rate per wall face, possibly at several
times), but generic: any per-face quantity keyed by the global OpenFOAM face id, or boundary
fields read from the case's time directories.

    from pinneapple_twin3d.openfoam import scene_from_case
    sc = scene_from_case("case/", face_values={"erosion_rate": {1057: 0.8, 1058: 1.3}},
                         units={"erosion_rate": "mm/year"})
    sc.export("out/twin")

Faces keep their own value (vertices are duplicated per face, fan-triangulated), so a per-face
map is shown exactly as computed, with no smoothing across faces. ASCII and binary polyMesh
files are both read (binary through ``pinneapple_simulation.external_solvers.openfoam``).
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .scene import Scene

_HEADER_END = re.compile(rb"^\s*//\s*\*.*$", re.M)
FaceValues = Union[Mapping[int, float], Sequence[Mapping[int, float]], np.ndarray]


def _body(path: str) -> Tuple[bytes, bool]:
    with open(path, "rb") as f:
        data = f.read()
    binary = re.search(rb"format\s+binary\s*;", data[:2000]) is not None
    return data, binary


def _strip_foamfile(data: bytes) -> str:
    text = data.decode("utf-8", "replace")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return re.sub(r"FoamFile\s*\{.*?\}", " ", text, count=1, flags=re.S)


def read_points(case: str) -> np.ndarray:
    path = os.path.join(case, "constant", "polyMesh", "points")
    data, binary = _body(path)
    if binary:
        from pinneapple_simulation.external_solvers.openfoam import binary_reader as b
        return b.read_vector_field_points(data, path)
    text = _strip_foamfile(data)
    m = re.search(r"(\d+)\s*\(", text)
    n = int(m.group(1))
    nums = re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", text[m.end():])
    pts = np.array(nums[:n], dtype=float)
    if len(pts) != n:
        raise ValueError(f"{path}: expected {n} points, read {len(pts)}")
    return pts


def read_faces(case: str) -> List[np.ndarray]:
    path = os.path.join(case, "constant", "polyMesh", "faces")
    data, binary = _body(path)
    if binary:
        from pinneapple_simulation.external_solvers.openfoam import binary_reader as b
        offsets, indices = b.read_face_compact_list(data, path)
        return [indices[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    text = _strip_foamfile(data)
    m = re.search(r"(\d+)\s*\(", text)
    n = int(m.group(1))
    faces = [np.array(g.split(), dtype=np.int64) for g in re.findall(r"\d+\s*\(([\d\s]+)\)", text[m.end():])[:n]]
    if len(faces) != n:
        raise ValueError(f"{path}: expected {n} faces, read {len(faces)}")
    return faces


def read_boundary(case: str) -> Dict[str, Dict[str, object]]:
    """``{patch: {"type": str, "startFace": int, "nFaces": int}}`` in file order."""
    text = _strip_foamfile(open(os.path.join(case, "constant", "polyMesh", "boundary"), "rb").read())
    out: Dict[str, Dict[str, object]] = {}
    for name, body in re.findall(r"([A-Za-z_][\w.:-]*)\s*\{([^{}]*)\}", text):
        typ = re.search(r"\btype\s+(\w+)\s*;", body)
        start = re.search(r"\bstartFace\s+(\d+)\s*;", body)
        nf = re.search(r"\bnFaces\s+(\d+)\s*;", body)
        if typ and start and nf:
            out[name] = {"type": typ.group(1), "startFace": int(start.group(1)), "nFaces": int(nf.group(1))}
    return out


def time_directories(case: str) -> List[Tuple[float, str]]:
    out = []
    for name in os.listdir(case):
        try:
            t = float(name)
        except ValueError:
            continue
        if os.path.isdir(os.path.join(case, name)):
            out.append((t, os.path.join(case, name)))
    return sorted(out)


def read_boundary_field(path: str, patch: str, n_faces: int) -> Optional[np.ndarray]:
    """Per-face values of ``patch`` in a volScalarField/volVectorField file (vectors -> magnitude)."""
    text = _strip_foamfile(open(path, "rb").read())
    bf = text.find("boundaryField")
    m = re.search(re.escape(patch) + r"\s*\{", text[bf:]) if bf >= 0 else None
    if not m:
        return None
    start = bf + m.end()
    depth, i = 1, start
    while depth and i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    body = text[start:i - 1]
    vm = re.search(r"\bvalue\s+(uniform|nonuniform)\s+", body)
    if not vm:
        return None
    rest = body[vm.end():]
    if vm.group(1) == "uniform":
        vec = re.match(r"\(\s*([^)]*)\)", rest)
        v = np.array(vec.group(1).split(), float) if vec else np.array([float(rest.split(";")[0])])
        return np.full(n_faces, float(np.linalg.norm(v)) if v.size > 1 else float(v[0]))
    lm = re.match(r"List<(\w+)>\s*(\d+)\s*\(", rest)
    if not lm:
        return None
    payload = rest[lm.end():]
    if lm.group(1) == "scalar":
        vals = np.array(payload.split(")")[0].split(), float)
    else:
        vals = np.linalg.norm(np.array(re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)",
                                                  payload)[: int(lm.group(2))], float), axis=1)
    return vals[:n_faces]


def _patch_surface(points, faces, start, n):
    """Duplicate vertices per face (exact per-face colours); fan-triangulate polygons."""
    verts, tris, owner = [], [], []
    for local in range(n):
        f = faces[start + local]
        base = len(verts)
        verts.extend(points[f])
        for k in range(1, len(f) - 1):
            tris.append([base, base + k, base + k + 1])
        owner.extend([local] * len(f))
    return np.asarray(verts, float), np.asarray(tris, np.int64), np.asarray(owner, np.int64)


def _as_steps(values: FaceValues, start: int, n: int, fill: float) -> np.ndarray:
    """-> (T, n) array of per-face values for one patch."""
    if isinstance(values, np.ndarray):
        arr = values if values.ndim == 2 else values[None, :]
        return arr[:, start:start + n].astype(float)
    steps = [values] if isinstance(values, Mapping) else list(values)
    out = np.full((len(steps), n), fill, float)
    for s, mapping in enumerate(steps):
        for fid, v in mapping.items():
            if start <= int(fid) < start + n:
                out[s, int(fid) - start] = float(v)
    return out


def scene_from_case(case: str, *, title: Optional[str] = None, patches: Optional[Sequence[str]] = None,
                    face_values: Optional[Dict[str, FaceValues]] = None, units: Optional[Dict[str, str]] = None,
                    boundary_fields: Sequence[str] = (), times: Optional[Sequence[float]] = None,
                    time_unit: str = "s", fill: float = 0.0, length_unit: str = "m",
                    source: str = "") -> Scene:
    """Build a Twin3D scene from an OpenFOAM case.

    Parameters
    ----------
    patches : patch names to show; default = every ``wall`` patch.
    face_values : ``{field: values}`` keyed by *global* face id. ``values`` is a mapping
        ``{face_id: value}`` (one instant), a sequence of such mappings (one per time step), or an
        array of shape (n_faces_total,) / (T, n_faces_total). Faces without a value get ``fill``.
    boundary_fields : field names read from every time directory (patch values; vectors -> |v|).
    times : time values for the steps of ``face_values`` (default: 0..T-1); ignored when only
        ``boundary_fields`` are given (then the case's time directories are used).
    """
    points, faces, boundary = read_points(case), read_faces(case), read_boundary(case)
    names = list(patches) if patches else [n for n, p in boundary.items() if p["type"] == "wall"]
    missing = [n for n in names if n not in boundary]
    if missing:
        raise KeyError(f"patches {missing} not in {case}/constant/polyMesh/boundary ({list(boundary)})")
    units = dict(units or {})
    face_values = dict(face_values or {})

    tdirs = time_directories(case) if boundary_fields else []
    n_steps = max([_as_steps(v, 0, 0, fill).shape[0] if not isinstance(v, np.ndarray) else
                   (v.shape[0] if v.ndim == 2 else 1) for v in face_values.values()] + [len(tdirs) if tdirs else 1])
    if tdirs and not face_values:
        step_times = [t for t, _ in tdirs]
    else:
        step_times = list(times) if times is not None else list(range(n_steps))
    sc = Scene(title or os.path.basename(os.path.abspath(case)), length_unit=length_unit,
               times=step_times if n_steps > 1 else None, time_unit=time_unit,
               source=source or f"OpenFOAM case {os.path.abspath(case)}")
    for name in names:
        p = boundary[name]
        start, n = int(p["startFace"]), int(p["nFaces"])
        v, f, owner = _patch_surface(points, faces, start, n)
        sc.add_part(name, v, f, group=str(p["type"]))
        for field, vals in face_values.items():
            per_face = _as_steps(vals, start, n, fill)
            sc.add_field(name, field, per_face[:, owner] if per_face.shape[0] > 1 else per_face[0, owner],
                         unit=units.get(field, ""))
        for field in boundary_fields:
            series = []
            for _, d in tdirs:
                fp = os.path.join(d, field)
                series.append(read_boundary_field(fp, name, n) if os.path.isfile(fp) else None)
            if all(s is None for s in series):
                continue
            arr = np.array([s if s is not None else np.full(n, np.nan) for s in series])
            sc.add_field(name, field, arr[:, owner] if arr.shape[0] > 1 else arr[0, owner], unit=units.get(field, ""))
    return sc
