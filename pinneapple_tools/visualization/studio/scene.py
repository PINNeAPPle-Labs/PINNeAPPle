"""A 3D scene for any geometry: surfaces with materials and per-vertex fields, streamlines, slice planes.

Build it from an STL/OBJ file, a ``pinneapple_data.cae`` mesh (any CFD/FEA result), plain arrays or an existing list of
parts, map a field onto it (CFD faces, FEA nodes, sensor points), then export it (glTF with the fields as vertex
attributes, OpenUSD, STL), render it in Blender (:mod:`.blender`) or open it in the browser (:mod:`.web`).

    from pinneapple_tools.visualization.studio import Scene
    sc = Scene.from_file("part.stl")
    sc.map_field("cp", points, cp)              # nearest-neighbour inverse-distance mapping onto the vertices
    sc.save("part.glb")                          # .glb / .usda / .stl

Coordinates are engineering axes with z up (metres or the model's unit). ``axes="aircraft"`` keeps the convention of
``pinneapple_design.aero`` (x aft, y right, z up) for its viewers.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# sRGB colours (converted to linear on export), PBR parameters
MATERIALS: Dict[str, Dict[str, Any]] = {
    "paint": {"color": [0.93, 0.94, 0.95], "metallic": 0.0, "roughness": 0.28, "clearcoat": 1.0},
    "metal": {"color": [0.72, 0.74, 0.77], "metallic": 0.85, "roughness": 0.32},
    "aluminium": {"color": [0.80, 0.81, 0.83], "metallic": 1.0, "roughness": 0.3},
    "copper": {"color": [0.85, 0.50, 0.32], "metallic": 1.0, "roughness": 0.3},
    "steel": {"color": [0.60, 0.61, 0.63], "metallic": 1.0, "roughness": 0.4},
    "plastic": {"color": [0.15, 0.17, 0.2], "metallic": 0.0, "roughness": 0.55},
    "pcb": {"color": [0.05, 0.32, 0.16], "metallic": 0.0, "roughness": 0.5},
    "concrete": {"color": [0.62, 0.62, 0.60], "metallic": 0.0, "roughness": 0.9},
    "glass": {"color": [0.02, 0.035, 0.05], "metallic": 0.0, "roughness": 0.02, "clearcoat": 1.0},
    "grey": {"color": [0.62, 0.65, 0.69], "metallic": 0.0, "roughness": 0.6},
}

AXES = {   # engineering axes -> glTF (y up); Blender's importer maps glTF back to z up
    "z_up": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),        # (x, y, z) -> (x, z, -y)
    "aircraft": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),     # (x aft, y right, z up) -> (y, z, x)
}


@dataclass
class Surface:
    """One triangulated surface. ``fields``: name -> one value per vertex."""
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    material: str = "grey"
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    group: str = ""                         # free tag ("propeller" parts spin in the aero viewer)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, float).reshape(-1, 3)
        self.faces = np.asarray(self.faces, np.int64).reshape(-1, 3)


@dataclass
class Lines:
    """Polylines (e.g. streamlines), each with optional per-point values (e.g. speed)."""
    points: List[np.ndarray]
    values: Optional[List[np.ndarray]] = None
    name: str = "streamlines"
    label: str = ""


@dataclass
class Slice:
    """A rectangular plane with a value grid: ``origin`` + s*``u`` + t*``v`` for s, t in [0, 1]. ``grid[j, i]``:
    row j along v, column i along u; NaN = no value (inside a body)."""
    name: str
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    grid: np.ndarray
    label: str = ""


@dataclass
class Scene:
    surfaces: List[Surface] = field(default_factory=list)
    lines: List[Lines] = field(default_factory=list)
    slices: List[Slice] = field(default_factory=list)
    axes: str = "z_up"
    title: str = "PINNeAPPle scene"
    labels: Dict[str, str] = field(default_factory=dict)     # field name -> label with unit (for colour bars)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_arrays(cls, vertices, faces, name: str = "body", material: str = "grey", **kw) -> "Scene":
        return cls([Surface(name, vertices, faces, material)], **kw)

    @classmethod
    def from_file(cls, path: str, material: str = "grey", **kw) -> "Scene":
        """STL (binary or ASCII, one surface per solid), OBJ, or any mesh/result ``pinneapple_data.cae`` reads
        (OpenFOAM case directory or zip, Gmsh, VTK, CalculiX .frd, Abaqus .inp ...)."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".stl":
            return cls([Surface(n, V, F, material) for n, V, F in read_stl(path)], **kw)
        if ext == ".obj":
            V, F = read_obj(path)
            return cls([Surface(os.path.splitext(os.path.basename(path))[0], V, F, material)], **kw)
        from pinneapple_data.cae import read_any
        if os.path.isdir(path):
            files = {}
            for root, _, fs in os.walk(path):
                for f in fs:
                    p = os.path.join(root, f)
                    files[os.path.relpath(p, path).replace(os.sep, "/")] = open(p, "rb").read()
        else:
            from pinneapple_data.cae.upload import expand
            files = expand([(os.path.basename(path), open(path, "rb").read())])
        return cls.from_mesh(read_any(files, fields=True), material=material, **kw)

    @classmethod
    def from_mesh(cls, mesh, fields: Optional[Dict[str, Any]] = None, material: str = "grey", name: str = "body",
                  **kw) -> "Scene":
        """The boundary surface of a ``pinneapple_data.cae`` mesh (volume or surface elements, polyhedral OpenFOAM
        meshes) with its cell or point fields on the vertices: vectors as magnitude, 6-component tensors as von
        Mises. ``fields`` defaults to the mesh's own (``mesh.cell_data`` / ``mesh.point_data``)."""
        from pinneapple_data.cae.compare import surface
        s = surface(mesh, max_tris=2_000_000)
        if s is None:
            raise ValueError("no surface in this mesh")
        V = np.asarray(s["points"], float).reshape(-1, 3)
        F = np.asarray(s["tris"], np.int64).reshape(-1, 3)
        surf = Surface(name, V, F, material)
        tri_cell = np.asarray(s["tri_cell"])
        cells = np.asarray(s["cells"])
        vidx = np.asarray(s["vertex_index"])
        fl = fields if fields is not None else _mesh_fields(mesh)
        for k, (where, a) in fl.items():
            a = scalar_of(np.asarray(a, float))
            if where == "point" and len(a) == mesh.n_points:
                surf.fields[k] = a[vidx]
            elif where == "cell" and len(a) == mesh.n_cells:
                per_tri = a[cells][tri_cell]                     # cell value -> its triangles -> mean at vertices
                acc, cnt = np.zeros(len(V)), np.zeros(len(V))
                for j in range(3):
                    np.add.at(acc, F[:, j], per_tri)
                    np.add.at(cnt, F[:, j], 1)
                surf.fields[k] = acc / np.maximum(cnt, 1)
        return cls([surf], **kw)

    @classmethod
    def from_parts(cls, parts: Sequence[Any], axes: str = "aircraft", **kw) -> "Scene":
        """From objects with name/vertices/faces/material (e.g. ``pinneapple_design.aero`` parts)."""
        return cls([Surface(p.name, p.vertices, p.faces, p.material, group=getattr(p, "group", ""))
                    for p in parts], axes=axes, **kw)

    # ------------------------------------------------------------------ fields
    def map_field(self, name: str, points, values, *, k: int = 4, mirror_y: bool = False,
                  surfaces: Optional[Sequence[str]] = None, label: str = "") -> "Scene":
        """Put ``values`` known at ``points`` (CFD wall faces, FEA nodes, sensors) on every vertex by inverse-distance
        weighting of the ``k`` nearest. ``mirror_y``: the data covers y >= 0 only (half model with a symmetry plane)."""
        from scipy.spatial import cKDTree
        P = np.asarray(points, float).reshape(-1, 3)
        vals = np.asarray(values, float).ravel()
        tree = cKDTree(P)
        kk = min(k, len(P))
        for s in self.surfaces:
            if surfaces is not None and s.name not in surfaces:
                continue
            q = s.vertices.copy()
            if mirror_y:
                q[:, 1] = np.abs(q[:, 1])
            d, i = tree.query(q, k=kk)
            if kk == 1:
                d, i = d[:, None], i[:, None]
            w = 1 / (d + 1e-9 * max(1.0, float(np.ptp(P)))) ** 2
            s.fields[name] = (vals[i] * w).sum(1) / w.sum(1)
        if label:
            self.labels[name] = label
        return self

    def field_names(self) -> List[str]:
        return list(dict.fromkeys(k for s in self.surfaces for k in s.fields))

    def field_range(self, name: str, pct: Tuple[float, float] = (1, 99)) -> Tuple[float, float]:
        a = np.concatenate([s.fields[name] for s in self.surfaces if name in s.fields])
        return float(np.nanpercentile(a, pct[0])), float(np.nanpercentile(a, pct[1]))

    def add_lines(self, points, values=None, name: str = "streamlines", label: str = "") -> "Scene":
        self.lines.append(Lines([np.asarray(p, float) for p in points],
                                None if values is None else [np.asarray(v, float) for v in values], name, label))
        return self

    def add_slice(self, name, origin, u, v, grid, label: str = "") -> "Scene":
        self.slices.append(Slice(name, np.asarray(origin, float), np.asarray(u, float), np.asarray(v, float),
                                 np.asarray(grid, float), label))
        return self

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        V = np.concatenate([s.vertices for s in self.surfaces])
        return V.min(0), V.max(0)

    # ------------------------------------------------------------------ export
    def save(self, path: str) -> str:
        """By extension: .glb (fields as vertex attributes ``_NAME``), .usda, .stl, .json (lines and slices)."""
        ext = os.path.splitext(path)[1].lower()
        data = {".glb": self.to_glb, ".usda": lambda: self.to_usda().encode(), ".stl": self.to_stl,
                ".json": lambda: json.dumps(self.extras()).encode()}[ext]()
        with open(path, "wb") as f:
            f.write(data)
        return path

    def to_glb(self, materials: Optional[Dict[str, Dict[str, Any]]] = None) -> bytes:
        return write_glb(self.surfaces, AXES[self.axes], {**MATERIALS, **(materials or {})},
                         generator="PINNeAPPle pinneapple_tools.visualization.studio")

    def to_usda(self, materials: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        return write_usda(self.surfaces, {**MATERIALS, **(materials or {})}, self.title)

    def to_stl(self) -> bytes:
        return write_stl(self.surfaces, self.title)

    def extras(self) -> Dict[str, Any]:
        """Lines and slices as JSON (scene axes, same as the surfaces), with the field labels."""
        r = lambda a, nd=4: np.round(np.asarray(a, float), nd).tolist()  # noqa: E731
        return {"axes": self.axes, "title": self.title, "labels": self.labels, "fields": self.field_names(),
                "lines": [{"name": L.name, "label": L.label, "points": [r(p) for p in L.points],
                           "values": None if L.values is None else [r(v) for v in L.values]} for L in self.lines],
                "slices": [{"name": s.name, "label": s.label, "origin": r(s.origin), "u": r(s.u), "v": r(s.v),
                            "grid": [[None if not np.isfinite(t) else round(float(t), 5) for t in row] for row in s.grid]}
                           for s in self.slices]}


# ---------------------------------------------------------------------- helpers
def scalar_of(a: np.ndarray) -> np.ndarray:
    """One value per entry: itself, the magnitude of a vector, the von Mises stress of a 6-component tensor
    (xx, yy, zz, xy, yz, zx)."""
    a = np.asarray(a, float)
    if a.ndim == 1:
        return a
    if a.shape[1] == 6:
        xx, yy, zz, xy, yz, zx = a.T
        return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy ** 2 + yz ** 2 + zx ** 2))
    return np.linalg.norm(a, axis=1)


def _mesh_fields(mesh) -> Dict[str, Tuple[str, np.ndarray]]:
    out = {}
    for where, d in (("point", getattr(mesh, "point_data", None)), ("cell", getattr(mesh, "cell_data", None))):
        for k, v in (d or {}).items():
            out.setdefault(k, (where, v))
    return out


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = np.zeros_like(V, dtype=float)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def read_stl(path: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """(name, vertices, faces) per solid; duplicate vertices merged."""
    raw = open(path, "rb").read()
    solids: List[Tuple[str, np.ndarray]] = []
    if len(raw) >= 84 and 84 + 50 * struct.unpack("<I", raw[80:84])[0] == len(raw):
        n = struct.unpack("<I", raw[80:84])[0]
        rec = np.frombuffer(raw, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n, offset=84)
        name = raw[:80].decode("latin-1").strip().replace("solid", "").strip() or os.path.splitext(os.path.basename(path))[0]
        solids.append((name, rec["v"].astype(float)))
    else:
        name, tri, cur = "body", [], []
        for line in raw.decode("latin-1").splitlines():
            t = line.split()
            if not t:
                continue
            if t[0] == "solid":
                name, tri = (" ".join(t[1:]) or "body"), []
            elif t[0] == "vertex":
                cur.append([float(x) for x in t[1:4]])
                if len(cur) == 3:
                    tri.append(cur)
                    cur = []
            elif t[0] == "endsolid":
                solids.append((name, np.array(tri, float).reshape(-1, 3, 3)))
                tri = []
        if tri:
            solids.append((name, np.array(tri, float).reshape(-1, 3, 3)))
    out = []
    for name, T in solids:
        P = T.reshape(-1, 3)
        key = np.round(P / (np.ptp(P) + 1e-30) * 1e7).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        out.append((name, P[first], inv.reshape(-1, 3)))
    return out


def read_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    V, F = [], []
    for line in open(path, encoding="latin-1"):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            idx = [int(s.split("/")[0]) for s in t[1:]]
            idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            F += [[idx[0], idx[j], idx[j + 1]] for j in range(1, len(idx) - 1)]
    return np.array(V, float), np.array(F, np.int64)


def write_glb(surfaces, T: np.ndarray, materials: Dict[str, Dict[str, Any]], generator: str = "PINNeAPPle",
              groups: Optional[Dict[str, List[int]]] = None) -> bytes:
    """glTF 2.0 binary with PBR materials (clearcoat, transmission, emissive). Every per-vertex field becomes a
    float attribute ``_NAME`` (upper case; three.js reads it back as ``_name``). ``groups``: node name -> indices of
    surfaces to parent under one node (e.g. a propeller that spins)."""
    mats = list(dict.fromkeys(s.material for s in surfaces))
    buf = bytearray()
    views, accessors, meshes, nodes = [], [], [], []

    def add(arr: np.ndarray, target: int, comp: int, typ: str, minmax=False):
        nonlocal buf
        while len(buf) % 4:
            buf += b"\0"
        off = len(buf)
        data = arr.tobytes()
        buf += data
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), "target": target})
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            acc["min"] = arr.min(0).tolist()
            acc["max"] = arr.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    for s in surfaces:
        V = (s.vertices @ T.T).astype(np.float32)
        N = (vertex_normals(s.vertices, s.faces) @ T.T).astype(np.float32)
        attrs = {"POSITION": add(V, 34962, 5126, "VEC3", True), "NORMAL": add(N, 34962, 5126, "VEC3")}
        for an, vals in getattr(s, "fields", {}).items():
            attrs["_" + an.upper().lstrip("_")] = add(np.asarray(vals, np.float32), 34962, 5126, "SCALAR")
        idx = add(s.faces.astype(np.uint32).ravel(), 34963, 5125, "SCALAR")
        meshes.append({"name": s.name, "primitives": [{"attributes": attrs, "indices": idx, "material": mats.index(s.material)}]})
        nodes.append({"name": s.name, "mesh": len(meshes) - 1})
    materials_out = []
    for m in mats:
        st = materials.get(m, MATERIALS["grey"])
        lin = [round(c ** 2.2, 5) for c in st["color"]]                  # glTF colours are linear; the tables are sRGB
        mat = {"name": m, "pbrMetallicRoughness": {"baseColorFactor": lin + [st.get("alpha", 1.0)],
                                                   "metallicFactor": st.get("metallic", 0.0),
                                                   "roughnessFactor": st.get("roughness", 0.5)}}
        ext = {}
        if st.get("clearcoat"):
            ext["KHR_materials_clearcoat"] = {"clearcoatFactor": st["clearcoat"], "clearcoatRoughnessFactor": 0.05}
        if st.get("transmission"):
            ext["KHR_materials_transmission"] = {"transmissionFactor": st["transmission"]}
            mat["alphaMode"] = "BLEND"
        if st.get("emissive"):
            mat["emissiveFactor"] = [round(c ** 2.2, 5) for c in st["emissive"]]
            ext["KHR_materials_emissive_strength"] = {"emissiveStrength": 4.0}
        if ext:
            mat["extensions"] = ext
        materials_out.append(mat)
    children = {i for ix in (groups or {}).values() for i in ix}
    top = [i for i in range(len(nodes)) if i not in children]
    for gname, ix in (groups or {}).items():
        if ix:
            nodes.append({"name": gname, "children": list(ix)})
            top.append(len(nodes) - 1)
    gltf = {"asset": {"version": "2.0", "generator": generator},
            "scene": 0, "scenes": [{"nodes": top}], "nodes": nodes, "meshes": meshes, "materials": materials_out,
            "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(buf)}],
            "extensionsUsed": ["KHR_materials_clearcoat", "KHR_materials_transmission", "KHR_materials_emissive_strength"]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    while len(buf) % 4:
        buf += b"\0"
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(buf), 0x004E4942) + bytes(buf)
    return out


def write_usda(surfaces, materials: Dict[str, Dict[str, Any]], title: str = "scene", root: str = "Scene") -> str:
    """OpenUSD text: one Mesh per surface with UsdPreviewSurface materials; z up, metres (Omniverse, usdview).
    Per-vertex fields become ``primvars:<name>`` (float, vertex interpolation)."""
    L = ['#usda 1.0', '(', f'    doc = "{title}: PINNeAPPle"', '    metersPerUnit = 1', '    upAxis = "Z"',
         f'    defaultPrim = "{root}"', ')', '', f'def Xform "{root}"', '{', '    def Scope "Looks"', '    {']
    for m in dict.fromkeys(s.material for s in surfaces):
        st = materials.get(m, MATERIALS["grey"])
        c = [round(v ** 2.2, 5) for v in st["color"]]                     # UsdPreviewSurface colours are linear
        L += [f'        def Material "{m}"', '        {',
              f'            token outputs:surface.connect = </{root}/Looks/{m}/Surface.outputs:surface>',
              '            def Shader "Surface"', '            {', '                uniform token info:id = "UsdPreviewSurface"',
              f'                color3f inputs:diffuseColor = ({c[0]}, {c[1]}, {c[2]})',
              f'                float inputs:metallic = {st.get("metallic", 0.0)}', f'                float inputs:roughness = {st.get("roughness", 0.5)}',
              f'                float inputs:clearcoat = {st.get("clearcoat", 0.0)}', f'                float inputs:opacity = {st.get("alpha", 1.0)}',
              '                token outputs:surface', '            }', '        }']
    L += ['    }']
    for s in surfaces:
        nm = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in s.name) or "mesh"
        if nm[0].isdigit():
            nm = "m_" + nm
        pts = ", ".join(f"({x:.5f}, {y:.5f}, {z:.5f})" for x, y, z in s.vertices)
        N = vertex_normals(s.vertices, s.faces)
        nrm = ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in N)
        L += [f'    def Mesh "{nm}"', '    {', f'        int[] faceVertexCounts = [{", ".join(["3"] * len(s.faces))}]',
              f'        int[] faceVertexIndices = [{", ".join(map(str, s.faces.ravel()))}]',
              f'        point3f[] points = [{pts}]', f'        normal3f[] normals = [{nrm}] (', '            interpolation = "vertex"', '        )']
        for k, vals in getattr(s, "fields", {}).items():
            pv = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in k)
            L += [f'        float[] primvars:{pv} = [{", ".join(f"{v:.6g}" for v in np.asarray(vals, float))}] (',
                  '            interpolation = "vertex"', '        )']
        L += ['        uniform token subdivisionScheme = "none"', f'        rel material:binding = </{root}/Looks/{s.material}>', '    }']
    L += ['}', '']
    return "\n".join(L)


def write_stl(surfaces, solid: str = "scene") -> bytes:
    """Binary STL of all surfaces."""
    tris = np.concatenate([s.vertices[s.faces] for s in surfaces]).astype(np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tris
    return solid.encode().ljust(80, b" ")[:80] + struct.pack("<I", len(tris)) + rec.tobytes()
