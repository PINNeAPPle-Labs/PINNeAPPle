"""Visualization-only mesh LOD with explicit physics-data isolation.

The source mesh and field arrays remain authoritative. LODs are separate GLBs, never
substituted for the simulation mesh or for fields.bin.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _triangles(vertices, faces):
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("expected vertices (V,3) and faces (F,3)")
    if not np.isfinite(v).all() or (f.size and (f.min() < 0 or f.max() >= len(v))):
        raise ValueError("non-finite vertices or invalid face indices")
    return v, f


def _compact(vertices, faces):
    indices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return vertices[indices], inverse.reshape(-1, 3).astype(np.uint32)


def _simplify(vertices, faces, target_faces):
    """Use an optional quadric decimator; refuse to fake decimation if absent."""
    if len(faces) <= target_faces:
        return vertices.copy(), faces.astype(np.uint32)
    try:
        import fast_simplification
    except ImportError as exc:
        raise ImportError(
            "LOD decimation requires fast-simplification: pip install 'pinneapple[geom_simplify]'"
        ) from exc
    out_v, out_f = fast_simplification.simplify(
        vertices.astype(np.float64), faces.astype(np.int32),
        target_count=int(target_faces),
    )
    return np.asarray(out_v, np.float32), np.asarray(out_f, np.uint32)


def build_visual_lods(scene, folder, *, ratios=(0.5, 0.2),
                      max_relative_bounds_error=0.02):
    """Export geometry_lodN.glb and a measurable manifest for a Scene.

    This is an offline visualization pipeline. Source geometry.glb / fields.bin
    and all simulation / sensor metadata are untouched. LODs are recommended
    only for parts with no per-vertex physical fields, to avoid mismatched indices.
    Ratios are relative to original triangle counts and must decrease.
    """
    from .scene import Part, _write_glb

    folder = Path(folder)
    ratios = tuple(float(x) for x in ratios)
    if not ratios or any(not (0 < x < 1) for x in ratios):
        raise ValueError("ratios must be strictly between 0 and 1")
    if any(a <= b for a, b in zip(ratios, ratios[1:])):
        raise ValueError("ratios must be strictly decreasing")
    if not 0 <= max_relative_bounds_error <= 1:
        raise ValueError("max_relative_bounds_error must be in [0, 1]")
    if not (folder / "scene.json").exists():
        raise FileNotFoundError("export the authoritative Scene before building LODs")

    original = json.loads((folder / "scene.json").read_text())
    manifest = []
    source_faces = sum(len(p.faces) for p in scene.parts)
    for level, ratio in enumerate(ratios, 1):
        visual_parts, detail = [], []
        for part in scene.parts:
            v, f = _triangles(part.vertices, part.faces)
            # Preserve field topology, rather than silently detaching physics from vertices.
            if part.fields or len(f) < 8:
                new_v, new_f = v.copy(), f.astype(np.uint32)
                reason = "field_topology" if part.fields else "small_mesh"
            else:
                target = max(4, int(len(f) * ratio))
                new_v, new_f = _simplify(v, f, target)
                if not len(new_f) or not len(new_v):
                    raise ValueError(f"decimator produced an empty mesh for {part.name}")
                extent = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
                error = float(np.max(np.abs(np.stack([v.min(0)-new_v.min(0), v.max(0)-new_v.max(0)]))))
                if error > max_relative_bounds_error * max(extent, 1e-12):
                    new_v, new_f = v.copy(), f.astype(np.uint32)
                    reason = "bounds_guard"
                else:
                    reason = "simplified"
            visual_parts.append(Part(part.name, new_v, new_f, part.group, part.color))
            detail.append({"part": part.name, "source_triangles": int(len(f)),
                           "triangles": int(len(new_f)), "reason": reason})
        name = f"geometry_lod{level}.glb"
        _write_glb(str(folder / name), visual_parts)
        manifest.append({"level": level, "ratio_requested": ratio, "file": name,
                         "triangles": sum(x["triangles"] for x in detail), "parts": detail})
    original["visual_lods"] = manifest
    original["visual_lod_policy"] = "visual-only; original mesh and physical fields remain authoritative"
    (folder / "scene.json").write_text(json.dumps(original, indent=2))
    return {"source_triangles": source_faces, "levels": manifest}
