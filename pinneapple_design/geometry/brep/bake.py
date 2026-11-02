"""Bake exact bodies to plain meshes (#1118): STL / OBJ / glTF / USD files, PINNeAPPle ``MeshData``, a gmsh volume mesh,
and a mesh-quality report against the exact B-rep.

* :func:`bake` writes one or more bodies (or their :class:`~.tessellate.TessMesh`) to ``.stl`` (binary), ``.obj``,
  ``.glb`` / ``.gltf`` (glTF 2.0), ``.usda`` (text, no dependency) or ``.usd`` / ``.usdc`` (with ``usd-core``).
  Body names are kept where the format has them (OBJ objects, glTF nodes, USD prims); optional per-vertex colours
  (a diagnostic such as mean curvature or zebra) go to OBJ vertex colours, glTF ``COLOR_0`` and USD
  ``displayColor``. Units: STL and OBJ carry none, so they are written in millimetres by default (what slicers and
  most CAD tools assume); glTF is metres by specification and USD records ``metersPerUnit``.
* :func:`to_meshdata` hands a body to the rest of PINNeAPPle (boundary sampling for PINNs, the immersed solvers);
  :func:`volume_mesh` meshes its interior with gmsh through :func:`pinneapple_design.geometry.mesh.generate.generate_mesh`.
* :func:`mesh_quality` reports watertightness, self-intersections, the smallest angle, the edge aspect ratio, the
  chordal error measured against the exact faces, and the volume and area against the kernel's.
"""
from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .step import MM, CadBody
from .tessellate import TessMesh, tessellate

FORMATS = (".stl", ".obj", ".glb", ".gltf", ".usda", ".usd", ".usdc")
_UNITS = {"m": 1.0, "mm": 1e3, "cm": 1e2, "inch": 1.0 / 0.0254}


def _items(source, quality: str, mesher: str, names=None) -> list[tuple[str, TessMesh, CadBody | None]]:
    seq = source if isinstance(source, (list, tuple)) else [source]
    if names is not None and len(names) != len(seq):
        raise ValueError("give one name per body")
    out = []
    for i, s in enumerate(seq):
        if isinstance(s, CadBody):
            out.append((s.name or f"body_{i}", tessellate(s, quality, mesher), s))
        elif isinstance(s, TessMesh):
            out.append((f"body_{i}", s, None))
        else:
            raise TypeError(f"bake takes CadBody or TessMesh objects, not {type(s).__name__}")
    if names is not None:
        out = [(str(n), m, b) for n, (_, m, b) in zip(names, out, strict=True)]
    return out


def _colours(colors, n_vertices: int) -> np.ndarray | None:
    """Per-vertex RGB in [0, 1] from RGB rows or from scalars (mapped with viridis over their range)."""
    if colors is None:
        return None
    c = np.asarray(colors, float)
    if c.ndim == 1:
        import matplotlib
        lo, hi = np.nanmin(c), np.nanmax(c)
        c = matplotlib.colormaps["viridis"]((c - lo) / max(hi - lo, 1e-300))[:, :3]
    if c.shape != (n_vertices, 3):
        raise ValueError(f"colors must be ({n_vertices},) scalars or ({n_vertices}, 3) RGB, got {c.shape}")
    return np.clip(c, 0.0, 1.0)


def _safe(name: str) -> str:
    s = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name) or "body"
    return s if not s[0].isdigit() else "_" + s


def _write_stl(path: Path, items, scale: float) -> None:
    tris = np.vstack([m.vertices[m.triangles] for _, m, _ in items]) * scale            # (m, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = n, tris
    header = ("pinneapple brep: " + ", ".join(nm for nm, _, _ in items)).encode("ascii", "replace")[:80]
    with open(path, "wb") as f:
        f.write(header.ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        f.write(rec.tobytes())


def _write_obj(path: Path, items, scale: float, cols) -> None:
    lines, off = ["# pinneapple brep bake"], 1
    for (name, m, _), c in zip(items, cols, strict=True):
        lines.append(f"o {_safe(name)}")
        V = m.vertices * scale
        if c is None:
            lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in V]
        else:
            lines += [f"v {x:.9g} {y:.9g} {z:.9g} {r:.4f} {g:.4f} {b:.4f}" for (x, y, z), (r, g, b) in zip(V, c, strict=True)]
        lines += [f"f {a + off} {b + off} {d + off}" for a, b, d in m.triangles]
        off += len(V)
    path.write_text("\n".join(lines) + "\n")


def _gltf(items, cols) -> tuple[dict, bytes]:
    blob, views, accessors, meshes, nodes = bytearray(), [], [], [], []

    def add(arr: np.ndarray, target: int, comp: int, typ: str, minmax: bool = False) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes, "target": target})
        blob.extend(arr.tobytes())
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": int(len(arr)), "type": typ}
        if minmax:
            acc["min"], acc["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    for (name, m, _), c in zip(items, cols, strict=True):
        attrs = {"POSITION": add(m.vertices.astype("<f4"), 34962, 5126, "VEC3", True)}
        if c is not None:
            attrs["COLOR_0"] = add(c.astype("<f4"), 34962, 5126, "VEC3")
        idx = add(m.triangles.astype("<u4").ravel(), 34963, 5125, "SCALAR")
        meshes.append({"name": name, "primitives": [{"attributes": attrs, "indices": idx, "mode": 4}]})
        nodes.append({"name": name, "mesh": len(meshes) - 1})
    doc = {"asset": {"version": "2.0", "generator": "pinneapple brep bake"}, "scene": 0,
           "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
           "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    return doc, bytes(blob)


def _write_gltf(path: Path, items, cols) -> None:
    doc, blob = _gltf(items, cols)
    if path.suffix == ".gltf":
        import base64
        doc["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(blob).decode()
        path.write_text(json.dumps(doc))
        return
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


def _usda(items, cols, meters_per_unit: float, scale: float) -> str:
    def arr(a, fmt):
        return "[" + ", ".join(fmt(r) for r in a) + "]"

    out = ['#usda 1.0', '(', '    defaultPrim = "World"', f'    metersPerUnit = {meters_per_unit:.9g}',
           '    upAxis = "Z"', ')', '', 'def Xform "World"', '{']
    for (name, m, _), c in zip(items, cols, strict=True):
        V = m.vertices * scale
        out += [f'    def Mesh "{_safe(name)}"', '    {',
                f'        int[] faceVertexCounts = [{", ".join(["3"] * len(m.triangles))}]',
                f'        int[] faceVertexIndices = {arr(m.triangles.ravel(), str)}',
                f'        point3f[] points = {arr(V, lambda p: f"({p[0]:.9g}, {p[1]:.9g}, {p[2]:.9g})")}',
                '        uniform token subdivisionScheme = "none"']
        if c is not None:
            out += [f'        color3f[] primvars:displayColor = {arr(c, lambda p: f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")} (',
                    '            interpolation = "vertex"', '        )']
        out += ['    }']
    out += ['}', '']
    return "\n".join(out)


def bake(source: CadBody | TessMesh | Sequence[CadBody | TessMesh], path: str, *, quality: str = "normal",
         mesher: str = "auto", unit: str | None = None, colors: Any = None,
         names: Sequence[str] | None = None) -> dict[str, Any]:
    """Write bodies (meshed at ``quality`` with ``mesher``) or ready meshes to a mesh file; the format follows the
    suffix of ``path`` (one of :data:`FORMATS`).

    ``unit``: length unit of the file, ``m``, ``mm``, ``cm`` or ``inch`` (default: ``mm`` for STL and OBJ, ``m``
    for glTF, which is metres by specification, and for USD, which records it). ``colors``: per-vertex scalars or
    RGB rows for one body, or a list with one entry (or ``None``) per body; STL has no colours. ``names``: one per body,
    replacing the body names (meshes have none: ``body_0``, ``body_1``, ...).

    Returns the path, format, unit, and the vertex, triangle and body counts."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in FORMATS:
        raise ValueError(f"unknown mesh format {ext!r}; use one of {', '.join(FORMATS)}")
    items = _items(source, quality, mesher, names)
    unit = unit or ("mm" if ext in (".stl", ".obj") else "m")
    if unit not in _UNITS:
        raise ValueError(f"unit must be one of {', '.join(_UNITS)}")
    if ext in (".glb", ".gltf") and unit != "m":
        raise ValueError("glTF is in metres by specification")
    scale = _UNITS[unit]
    per = colors if isinstance(colors, list) and len(items) > 1 else [colors] + [None] * (len(items) - 1)
    if len(per) != len(items):
        raise ValueError("give one colors entry per body")
    cols = [_colours(c, len(m.vertices)) for c, (_, m, _) in zip(per, items, strict=True)]
    if ext == ".stl" and any(c is not None for c in cols):
        raise ValueError("STL has no colours; use .obj, .glb or .usda")
    p.parent.mkdir(parents=True, exist_ok=True)
    if ext == ".stl":
        _write_stl(p, items, scale)
    elif ext == ".obj":
        _write_obj(p, items, scale, cols)
    elif ext in (".glb", ".gltf"):
        _write_gltf(p, items, cols)
    else:
        text = _usda(items, cols, 1.0 / scale, scale)
        if ext == ".usda":
            p.write_text(text)
        else:
            try:
                from pxr import Sdf, Usd
            except ImportError as e:
                raise ImportError('binary USD needs usd-core (pip install usd-core); .usda needs nothing') from e
            layer = Sdf.Layer.CreateAnonymous(".usda")
            layer.ImportFromString(text)
            Usd.Stage.Open(layer).Export(str(p))
    return {"path": str(p), "format": ext[1:], "unit": unit, "bodies": [n for n, _, _ in items],
            "n_vertices": int(sum(len(m.vertices) for _, m, _ in items)),
            "n_triangles": int(sum(len(m.triangles) for _, m, _ in items))}


def to_meshdata(source: CadBody | TessMesh, *, quality: str = "normal", mesher: str = "auto"):
    """A body (or its mesh) as :class:`pinneapple_design.geometry.core.mesh.MeshData` in metres, with face normals."""
    from pinneapple_design.geometry.core.mesh import MeshData

    m = tessellate(source, quality, mesher) if isinstance(source, CadBody) else source
    V, T = m.vertices, m.triangles
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    return MeshData(V.copy(), T.copy(), normals=n)


def volume_mesh(source: CadBody | TessMesh, h: float, *, quality: str = "normal", out_dir: str | None = None,
                **kw) -> dict[str, Any]:
    """Mesh the interior with gmsh: bake to STL in metres, then
    :func:`pinneapple_design.geometry.mesh.generate.generate_mesh` (same keywords and returned report)."""
    import tempfile

    from pinneapple_design.geometry.mesh.generate import generate_mesh

    with tempfile.TemporaryDirectory() as tmp:
        stl = bake(source, str(Path(tmp) / "body.stl"), quality=quality, unit="m")["path"]
        return generate_mesh(stl, h=h, scale=1.0, out_dir=out_dir, **kw)


# ---------------------------------------------------------------------------------------------- quality report
def _overlap_pairs(lo: np.ndarray, hi: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Index pairs (i < j) whose axis-aligned boxes overlap, by sweep and prune along x."""
    order = np.argsort(lo[:, 0], kind="stable")
    lo, hi = lo[order], hi[order]
    end = np.searchsorted(lo[:, 0], hi[:, 0], side="right")          # boxes j > i with lo_j.x <= hi_i.x
    counts = np.maximum(end - np.arange(len(lo)) - 1, 0)
    cum = np.r_[0, np.cumsum(counts)]
    cuts = np.unique(np.r_[0, np.searchsorted(cum, np.arange(chunk, cum[-1], chunk)), len(lo)])
    out = []
    for s, e in zip(cuts[:-1], cuts[1:], strict=True):
        i = np.repeat(np.arange(s, e), counts[s:e])
        j = i + 1 + np.arange(len(i)) - (cum[i] - cum[s])
        ok = np.all((lo[j, 1:] <= hi[i, 1:]) & (lo[i, 1:] <= hi[j, 1:]), axis=1)
        out.append(np.c_[order[i[ok]], order[j[ok]]])
    return np.vstack(out) if out else np.zeros((0, 2), np.int64)


def _segments_cross(p, q, a, b, c, eps) -> np.ndarray:
    """Whether segment pq crosses triangle abc strictly (rows are independent cases)."""
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    dp, dq = ((p - a) * n).sum(1), ((q - a) * n).sum(1)
    cross = (dp * dq < 0) & (np.abs(dp) > eps * nn) & (np.abs(dq) > eps * nn)
    t = dp / np.where(cross, dp - dq, 1.0)
    x = p + t[:, None] * (q - p)
    inside = np.ones(len(p), bool)
    for u, v in ((a, b), (b, c), (c, a)):
        inside &= (np.cross(v - u, x - u) * n).sum(1) > eps * nn * np.linalg.norm(v - u, axis=1)
    return cross & inside


def self_intersections(mesh: TessMesh) -> int:
    """Pairs of triangles that share no vertex and cross each other (coplanar overlaps are not counted)."""
    V, T = mesh.vertices, mesh.triangles
    tri = V[T]
    pairs = _overlap_pairs(tri.min(1), tri.max(1))
    if not len(pairs):
        return 0
    shared = (T[pairs[:, 0]][:, :, None] == T[pairs[:, 1]][:, None, :]).any(axis=(1, 2))
    pairs = pairs[~shared]
    eps = 1e-12 * float(np.linalg.norm(np.ptp(V, axis=0)))
    hit = np.zeros(len(pairs), bool)
    for s, o in ((0, 1), (1, 0)):
        A, B = tri[pairs[:, s]], tri[pairs[:, o]]
        for k in range(3):
            hit |= _segments_cross(A[:, k], A[:, (k + 1) % 3], B[:, 0], B[:, 1], B[:, 2], eps)
    return int(hit.sum())


def mesh_quality(mesh: TessMesh, body: CadBody | None = None, *, max_samples: int = 20_000,
                 seed: int = 0) -> dict[str, Any]:
    """Quality of a triangle mesh, and of how well it carries the exact body when ``body`` is given.

    Always: watertight, Euler characteristic, self-intersecting triangle pairs, smallest angle [deg], edge aspect
    ratio (longest / shortest edge of a triangle; max and mean). With ``body``: the chordal error, the largest
    distance [m] from triangle centroids and edge midpoints to the exact face the triangle belongs to (at most
    ``max_samples`` points, drawn at random when there are more), its ratio to the mesh's chord tolerance, and the
    relative volume and area errors against the kernel's exact values."""
    V, T = mesh.vertices, mesh.triangles
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    L = np.stack([np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1)], 1)

    def ang(p, q, r):
        u, v = q - p, r - p
        return np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), (u * v).sum(1)))

    min_angle = np.minimum(np.minimum(ang(a, b, c), ang(b, c, a)), ang(c, a, b))
    ar = L.max(1) / np.maximum(L.min(1), 1e-300)
    rep: dict[str, Any] = {
        "n_vertices": int(len(V)), "n_triangles": int(len(T)), "watertight": mesh.is_watertight(),
        "euler_characteristic": mesh.euler_characteristic(), "self_intersections": self_intersections(mesh),
        "min_angle_deg": float(min_angle.min()), "mean_min_angle_deg": float(min_angle.mean()),
        "aspect_ratio_max": float(ar.max()), "aspect_ratio_mean": float(ar.mean()),
        "area_m2": mesh.area(), "volume_m3": mesh.volume(),
    }
    if body is not None:
        pts = np.vstack([(a + b + c) / 3, (a + b) / 2, (b + c) / 2, (c + a) / 2])
        fid = np.tile(mesh.face_id, 4)
        if len(pts) > max_samples:
            pick = np.random.default_rng(seed).choice(len(pts), max_samples, replace=False)
            pts, fid = pts[pick], fid[pick]
        d = np.array([body.faces[f].shape.distance_to((p / MM).tolist()) for p, f in zip(pts, fid, strict=True)]) * MM
        rep.update({
            "chordal_error_max_m": float(d.max()), "chordal_error_mean_m": float(d.mean()),
            "chord_tolerance_m": mesh.chord_tolerance_m,
            "chordal_error_over_tolerance": float(d.max() / mesh.chord_tolerance_m) if mesh.chord_tolerance_m else None,
            "volume_error_rel": float(mesh.volume() / body.volume_m3 - 1) if body.volume_m3 else None,
            "area_error_rel": float(mesh.area() / body.area_m2 - 1) if body.area_m2 else None,
            "samples": int(len(pts)),
        })
    return rep
