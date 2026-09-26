"""Build and export a 3D digital-twin scene: parts, fields over time, sensors.

Export layout (one folder, static files, served by anything)::

    scene.json      manifest: parts, fields, time steps, sensors, units, provenance
    geometry.glb    all parts as glTF 2.0 binary meshes (opens in Blender, CAD viewers, three.js)
    fields.bin      float32 field values, offsets listed in scene.json
    index.html,
    viewer.js       the bundled web viewer (see ``viewer/``), added by ``export(with_viewer=True)``

Geometry goes to glTF because it is the interchange format every 3D tool reads; fields stay
out of the glTF (a transient field is a (time, vertex) array, which glTF has no standard slot
for) and are addressed per part and per time step in the manifest. OpenUSD export is on the
roadmap (``ROADMAP.md`` §11) for the CAD -> sim -> twin thread.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

VIEWER_DIR = os.path.join(os.path.dirname(__file__), "viewer")


@dataclass
class Part:
    name: str
    vertices: np.ndarray  # (V, 3) float
    faces: np.ndarray  # (F, 3) int
    group: str = ""
    color: Sequence[float] = (0.75, 0.77, 0.8)
    fields: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> (V,) or (T, V)


@dataclass
class Sensor:
    id: str
    position: Sequence[float]
    label: str = ""
    unit: str = ""
    quantity: str = ""
    series: Optional[np.ndarray] = None  # (T,) values aligned with the scene time steps
    envelope: Optional[Sequence[float]] = None  # (min, max) validity / alarm range


class Scene:
    """A digital-twin scene. Add parts, fields and sensors, then ``export`` to a folder."""

    def __init__(self, title: str, *, length_unit: str = "m", times: Optional[Sequence[float]] = None,
                 time_unit: str = "s", source: str = ""):
        self.title = title
        self.length_unit = length_unit
        self.times = None if times is None else np.asarray(times, dtype=float)
        self.time_unit = time_unit
        self.source = source
        self.parts: List[Part] = []
        self.sensors: List[Sensor] = []
        self.field_units: Dict[str, str] = {}

    # -- building ------------------------------------------------------
    def add_part(self, name: str, vertices, faces, *, group: str = "", color=(0.75, 0.77, 0.8)) -> Part:
        v = np.asarray(vertices, dtype=np.float32)
        f = np.asarray(faces, dtype=np.uint32)
        if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"part '{name}': vertices must be (V, 3) and faces (F, 3)")
        if f.size and int(f.max()) >= len(v):
            raise ValueError(f"part '{name}': face index out of range")
        if any(p.name == name for p in self.parts):
            raise ValueError(f"duplicate part name '{name}'")
        part = Part(name, v, f, group, tuple(color))
        self.parts.append(part)
        return part

    def add_trimesh(self, name: str, mesh, **kw) -> Part:
        """Add a ``trimesh.Trimesh`` (e.g. from ``trimesh.load('part.stl')``)."""
        return self.add_part(name, mesh.vertices, mesh.faces, **kw)

    def add_field(self, part: str, name: str, values, *, unit: str = "") -> None:
        """Per-vertex field: shape (V,) for steady, (T, V) for transient (T = len(times))."""
        p = self._part(part)
        a = np.asarray(values, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        if a.ndim != 2 or a.shape[1] != len(p.vertices):
            raise ValueError(f"field '{name}' on '{part}': expected (V,) or (T, V) with V={len(p.vertices)}")
        if a.shape[0] > 1 and (self.times is None or a.shape[0] != len(self.times)):
            raise ValueError(f"field '{name}' has {a.shape[0]} time steps but the scene has "
                             f"{0 if self.times is None else len(self.times)}")
        p.fields[name] = a
        if unit:
            self.field_units[name] = unit

    def add_sensor(self, id: str, position, *, label: str = "", unit: str = "", quantity: str = "",
                   series=None, envelope=None) -> Sensor:
        if any(s.id == id for s in self.sensors):
            raise ValueError(f"duplicate sensor id '{id}'")
        ser = None if series is None else np.asarray(series, dtype=float)
        if ser is not None and (self.times is None or len(ser) != len(self.times)):
            raise ValueError(f"sensor '{id}': series must have one value per scene time step")
        s = Sensor(id, tuple(float(c) for c in position), label or id, unit, quantity, ser,
                   None if envelope is None else (float(envelope[0]), float(envelope[1])))
        self.sensors.append(s)
        return s

    def _part(self, name: str) -> Part:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(f"unknown part '{name}'")

    # -- export --------------------------------------------------------
    def export(self, folder: str, *, with_viewer: bool = True, usd: bool = False) -> str:
        """Write scene.json + geometry.glb + fields.bin (+ viewer, + scene.usda with ``usd=True``).

        Returns the scene.json path.
        """
        if not self.parts:
            raise ValueError("scene has no parts")
        os.makedirs(folder, exist_ok=True)
        _write_glb(os.path.join(folder, "geometry.glb"), self.parts)

        field_meta, offset = [], 0
        with open(os.path.join(folder, "fields.bin"), "wb") as fb:
            for p in self.parts:
                for name, arr in p.fields.items():
                    data = np.ascontiguousarray(arr, dtype="<f4")
                    fb.write(data.tobytes())
                    finite = data[np.isfinite(data)]
                    field_meta.append({
                        "part": p.name, "name": name, "unit": self.field_units.get(name, ""),
                        "steps": int(data.shape[0]), "count": int(data.shape[1]),
                        "offset": offset, "min": float(finite.min()) if finite.size else 0.0,
                        "max": float(finite.max()) if finite.size else 0.0,
                    })
                    offset += data.nbytes

        lo = np.min([p.vertices.min(0) for p in self.parts], axis=0)
        hi = np.max([p.vertices.max(0) for p in self.parts], axis=0)
        manifest = {
            "format": "pinneapple-twin3d/1",
            "title": self.title,
            "source": self.source,
            "length_unit": self.length_unit,
            "time_unit": self.time_unit,
            "times": [] if self.times is None else [float(t) for t in self.times],
            "bounds": [lo.tolist(), hi.tolist()],
            "geometry": "geometry.glb",
            "fields_file": "fields.bin",
            "parts": [{"name": p.name, "group": p.group, "color": list(p.color), "mesh_index": i,
                       "vertices": int(len(p.vertices)), "triangles": int(len(p.faces))}
                      for i, p in enumerate(self.parts)],
            "fields": field_meta,
            "sensors": [{"id": s.id, "label": s.label, "unit": s.unit, "quantity": s.quantity,
                         "position": list(s.position),
                         "series": None if s.series is None else [float(v) for v in s.series],
                         "envelope": None if s.envelope is None else list(s.envelope)}
                        for s in self.sensors],
        }
        path = os.path.join(folder, "scene.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=1)
        if usd:
            from .usd import export_usd
            export_usd(self, os.path.join(folder, "scene.usda"))
        if with_viewer:
            for name in os.listdir(VIEWER_DIR):
                shutil.copy(os.path.join(VIEWER_DIR, name), os.path.join(folder, name))
        return path


def _write_glb(path: str, parts: List[Part]) -> None:
    """Minimal glTF 2.0 binary writer: one mesh + node per part (POSITION, NORMAL, indices)."""
    bin_chunks, views, accessors, meshes, nodes, materials = [], [], [], [], [], []
    offset = 0

    def add_view(data: bytes, target: int) -> int:
        nonlocal offset
        pad = (-len(data)) % 4
        bin_chunks.append(data + b"\x00" * pad)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        offset += len(data) + pad
        return len(views) - 1

    for i, p in enumerate(parts):
        v = np.ascontiguousarray(p.vertices, dtype="<f4")
        idx = np.ascontiguousarray(p.faces.reshape(-1), dtype="<u4")
        n = _vertex_normals(v, p.faces)
        pv = add_view(v.tobytes(), 34962)
        nv = add_view(n.tobytes(), 34962)
        iv = add_view(idx.tobytes(), 34963)
        accessors.append({"bufferView": pv, "componentType": 5126, "count": len(v), "type": "VEC3",
                          "min": v.min(0).tolist(), "max": v.max(0).tolist()})
        accessors.append({"bufferView": nv, "componentType": 5126, "count": len(v), "type": "VEC3"})
        accessors.append({"bufferView": iv, "componentType": 5125, "count": int(idx.size), "type": "SCALAR"})
        materials.append({"name": p.name, "pbrMetallicRoughness": {
            "baseColorFactor": list(p.color) + [1.0], "metallicFactor": 0.1, "roughnessFactor": 0.7},
            "doubleSided": True})
        meshes.append({"name": p.name, "primitives": [{
            "attributes": {"POSITION": 3 * i, "NORMAL": 3 * i + 1}, "indices": 3 * i + 2, "material": i}]})
        nodes.append({"name": p.name, "mesh": i})

    blob = b"".join(bin_chunks)
    gltf = {"asset": {"version": "2.0", "generator": "pinneapple_twin3d"}, "scene": 0,
            "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
            "materials": materials, "accessors": accessors, "bufferViews": views,
            "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


def _vertex_normals(v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.zeros_like(v, dtype=np.float64)
    if faces.size:
        tri = v[faces.astype(np.int64)]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for k in range(3):
            np.add.at(n, faces[:, k].astype(np.int64), fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return np.ascontiguousarray(n / norm, dtype="<f4")
