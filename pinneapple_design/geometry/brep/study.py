"""One call from a STEP file to everything the apps show of it (#1120): the bodies, their mesh quality, curvature,
zebra and highlight lines, edge continuity, the baked files and the ``MeshData`` arrays.

    from pinneapple_design.geometry.brep.study import step_study
    r = step_study("bracket.step", quality="normal", out_dir="out")

With ``out_dir`` it writes, per body ``k``:

* ``body_<k>.glb`` + ``body_<k>.json``: a field view (schema ``pinneapple-field-view/v1``, the format of the apps'
  3-D viewer) with mean and Gaussian curvature, zebra stripes and highlight lines per vertex, and the edges as
  G0 / G1 / G2 markers;
* ``body_<k>.npz``: ``MeshData`` arrays in metres (vertices, faces, normals, face_id) for a CFD/FEM or PINN study;

and the bake of all bodies, ``<file stem>.<format>`` for each of ``bake_formats``. The returned report is plain
JSON: per body its size, volume, area, face types, mesh quality against the exact body, continuity counts and
curvature ranges, plus the files written.
"""
from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .bake import bake, mesh_quality
from .diagnostics import edge_continuity, exact_curvature, highlight_lines, refine, zebra
from .step import MM, read_step
from .tessellate import tessellate

MAX_INPUT_MB = 100.0           # default size limit of the STEP file; max_input_mb or PINNEAPPLE_STEP_MAX_INPUT_MB


def _edge_points(body, edges: list[dict[str, Any]], level: str, per_edge: int = 16) -> np.ndarray:
    shapes = body.shape.edges()
    pts = [[list(shapes[e["edge"]].point_at(t)) for t in np.linspace(0.0, 1.0, per_edge)]
           for e in edges if e["level"] == level]
    return np.asarray(pts, float).reshape(-1, 3) * MM if pts else np.zeros((0, 3))


def _surface(mesh):
    from pinneapple_data.cae.model import CellBlock, Mesh
    return Mesh(points=np.asarray(mesh.vertices, float), blocks=[CellBlock("triangle", np.asarray(mesh.triangles))])


def step_study(geometry: str, quality: str = "normal", mesher: str = "auto", stripes: int = 16,
               refine_view: bool = True, max_faces: int = 150_000, bake_formats: Sequence[str] = ("stl",),
               max_input_mb: float | None = None, name: str | None = None,
               out_dir: str | None = None) -> dict[str, Any]:
    """Read a STEP file and study every body in it (see the module docstring for the files).

    Parameters
    ----------
    geometry : str
        path of a ``.step`` / ``.stp`` file
    quality : str
        mesh quality preset: draft, normal or fine
    mesher : str
        auto, fast or kernel
    stripes : int
        zebra stripe pairs [-]
    refine_view : bool
        split triangles for the view, so that planar faces have vertices to carry stripes and highlight lines
    max_faces : int
        largest view per body, in triangles [-]
    bake_formats : list[str]
        mesh files of all bodies: stl, obj, glb, gltf, usda
    max_input_mb : float
        largest STEP file accepted, checked before reading [MB] (default: PINNEAPPLE_STEP_MAX_INPUT_MB, else 100)
    name : str
        stem of the baked files (default: the STEP file's; an app passes the uploaded file's name)
    out_dir : str
        folder for the files (nothing is written without it)

    Returns
    -------
    bodies : list[dict]
        per body: name, size_m, volume_m3, area_m2, face_counts, solid, valid, mesh (quality report), continuity
        (edges per level), curvature (H and K ranges [1/m, 1/m2]), view and meshdata (file names)
    files : list[str]
        files written in out_dir
    """
    p = Path(geometry)
    if p.suffix.lower() not in (".step", ".stp"):
        raise ValueError(f"step_study reads .step / .stp files, not {p.suffix or 'a file without extension'}")
    limit = float(max_input_mb if max_input_mb is not None else os.environ.get("PINNEAPPLE_STEP_MAX_INPUT_MB",
                                                                               MAX_INPUT_MB))
    mb = p.stat().st_size / 2**20
    if mb > limit:
        raise ValueError(f"the STEP file is {mb:.1f} MB, above the {limit:g} MB limit")
    bodies = read_step(str(p))
    out = Path(out_dir) if out_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    rows, meshes = [], []
    for k, b in enumerate(bodies):
        tm = tessellate(b, quality, mesher)
        meshes.append(tm)
        q = mesh_quality(tm, b)
        edges = edge_continuity(b)
        levels = Counter(e["level"] or "free" for e in edges)
        view = refine(tm, max_triangles=max_faces) if refine_view else tm
        c = exact_curvature(b, view)
        row = {"name": b.name or f"body_{k}", "size_m": [float(x) for x in b.size_m], "volume_m3": b.volume_m3,
               "area_m2": b.area_m2, "face_counts": b.face_counts(), "solid": b.is_solid, "valid": b.valid,
               "mesh": q, "continuity": dict(sorted(levels.items())),
               "curvature": {"H_min_per_m": float(c["H"].min()), "H_max_per_m": float(c["H"].max()),
                             "K_min_per_m2": float(c["K"].min()), "K_max_per_m2": float(c["K"].max())}}
        if out is not None:
            from pinneapple_tools.visualization.studio.fieldview import field_view

            z = zebra(view, c["normal"], n_stripes=stripes)
            h = highlight_lines(view, c["normal"])
            markers = []
            for level, detail in (("G0", "edges where faces meet at an angle"),
                                  ("G1", "tangent edges (curvature jumps)"),
                                  ("G2", "curvature-continuous edges")):
                P = _edge_points(b, edges, level)
                if len(P):
                    markers.append({"name": f"{level} edges ({levels[level]})", "kind": "region", "points": P,
                                    "detail": detail, "label": False})
            fv = field_view(_surface(view), [
                {"name": "H", "values": c["H"], "label": "Mean curvature H", "unit": "1/m"},
                {"name": "K", "values": c["K"], "label": "Gaussian curvature K", "unit": "1/m2"},
                {"name": "zebra", "values": z, "label": f"Zebra ({stripes} stripes)"},
                {"name": "highlight", "values": h, "label": "Highlight lines"}],
                markers=markers, max_faces=max_faces, title=f"{row['name']}: surface quality",
                notes=[f"{q['n_triangles']} triangles ({tm.mesher}, {quality}); watertight {q['watertight']}; "
                       f"chordal error {q['chordal_error_over_tolerance']:.2f} x tolerance",
                       "Curvature from the exact surface; zebra and highlight lines from its normals."])
            (out / f"body_{k}.glb").write_bytes(fv.glb())
            import json
            (out / f"body_{k}.json").write_text(json.dumps(fv.to_dict()))
            V, T = tm.vertices, tm.triangles
            n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
            np.savez(out / f"body_{k}.npz", vertices=V, faces=T, normals=n, face_id=tm.face_id)
            row.update(view=f"body_{k}.glb", meshdata=f"body_{k}.npz")
            files += [f"body_{k}.glb", f"body_{k}.json", f"body_{k}.npz"]
        rows.append(row)
    if out is not None and meshes:
        for fmt in bake_formats:
            stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (name or p.stem)) or "part"
            fname = f"{stem}.{str(fmt).lower().lstrip('.')}"
            bake(meshes, str(out / fname), names=[r["name"] for r in rows])
            files.append(fname)
    return {"file": p.name, "units": "m", "quality": quality, "n_bodies": len(rows), "bodies": rows, "files": files}
