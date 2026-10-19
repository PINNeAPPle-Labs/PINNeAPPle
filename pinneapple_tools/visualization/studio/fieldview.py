"""Fields on a mesh for the apps' 3-D views: the boundary surface of a ``pinneapple_data.cae`` mesh with several
fields (values and errors), optional deformed shapes and marked regions, as a glTF plus a small JSON description.

    from pinneapple_tools.visualization.studio.fieldview import field_view
    fv = field_view(mesh, [
        {"name": "fem", "values": vm_fem, "label": "FEM von Mises", "unit": "MPa", "group": "vm", "deform": "fem"},
        {"name": "rom", "values": vm_rom, "label": "ROM von Mises", "unit": "MPa", "group": "vm", "deform": "rom"},
        {"name": "err", "values": abs(vm_rom - vm_fem), "label": "|ROM - FEM|", "unit": "MPa", "kind": "error"}],
        deform={"fem": u_fem, "rom": u_rom},
        markers=[{"name": "clamp", "kind": "bc", "points": mesh.points[clamp]}])
    fv.glb()          # bytes for /shared/fieldview3d.js (GLTFLoader): fields as vertex attributes _F0, _F1, ...
    fv.meta           # fields with their colour ranges, deformations with an automatic scale, markers, sizes

Colour ranges are computed on the surface values that are shown. Fields of one ``group`` share a range (a prediction
and its reference on the same scale). Errors run from 0 to their 99th percentile and say when the scale is clipped.
Above ``max_faces`` triangles the surface is decimated (vertex clustering, fields averaged): see the size guide in
docs/core_concepts/studio.md (interactive up to about 500 k triangles; the default keeps a view light).
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .scene import AXES, Scene

__all__ = ["FieldView", "field_view", "VIEW_MAX_FACES", "colour_range", "subsample", "length_factor",
           "displacement_factor"]

VIEW_MAX_FACES = 250_000        # 260 k triangles: 7.3 MB glTF, 0.5 s field switch in the size guide's worst case
MAX_MARKER_POINTS = 400


def colour_range(values, kind: str = "value") -> dict[str, Any]:
    """``{"lo", "hi", "max", "clipped"}``: values span min..max; errors span 0..99th percentile (``clipped`` when
    the maximum is above it)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"lo": 0.0, "hi": 1.0, "max": None, "clipped": False}
    mx = float(v.max())
    if kind == "error":
        hi = float(np.percentile(v, 99))
        hi = hi if hi > 0 else (mx if mx > 0 else 1.0)
        return {"lo": 0.0, "hi": hi, "max": mx, "clipped": bool(mx > hi * (1 + 1e-9))}
    lo = float(v.min())
    if mx == lo:
        d = abs(lo) * 0.01 or 1.0
        return {"lo": lo - d, "hi": mx + d, "max": mx, "clipped": False}
    return {"lo": lo, "hi": mx, "max": mx, "clipped": False}


_DISP = re.compile(r"^(d|u|disp|displacement|displacements)$", re.I)


def length_factor(unit: str | None, to: str = "m") -> float | None:
    """Factor from ``unit`` to ``to`` when ``unit`` is a length (``mm`` -> m: 1e-3), else None."""
    from pinneapple_data.physical_units import try_parse_unit
    a, b = try_parse_unit(unit), try_parse_unit(to)
    if a is None or b is None or a.dim != (1, 0, 0, 0, 0, 0) or b.dim != a.dim:
        return None
    return a.factor / b.factor


def displacement_factor(name: str, unit: str | None, components: int, mesh_unit: str = "m") -> float | None:
    """If a nodal vector field is a displacement, the factor that converts it to the mesh's length unit: its unit is
    a length, or it has no unit and is named like one (``D``, ``U``, ``disp``, ``displacement``). Else None."""
    if components not in (2, 3):
        return None
    f = length_factor(unit, mesh_unit) if unit else None
    if f is not None:
        return f
    return 1.0 if not unit and _DISP.match(name or "") else None


def subsample(points, n: int = MAX_MARKER_POINTS) -> np.ndarray:
    """At most ``n`` of the points, evenly spread over the array (markers do not need every node)."""
    P = np.asarray(points, float).reshape(-1, 3)
    if len(P) <= n:
        return P
    return P[np.linspace(0, len(P) - 1, n).round().astype(int)]


@dataclass
class FieldView:
    scene: Scene
    meta: dict[str, Any]

    def glb(self) -> bytes:
        return self.scene.to_glb()

    def to_dict(self) -> dict[str, Any]:
        return self.meta


def _r(a, nd: int = 6) -> list:
    return np.round(np.asarray(a, float), nd).tolist()


def field_view(mesh, fields: Sequence[Mapping[str, Any]], *, deform: Mapping[str, Any] | None = None,
               markers: Sequence[Mapping[str, Any]] = (), max_faces: int | None = VIEW_MAX_FACES,
               title: str = "", axes: str = "z_up", notes: Sequence[str] = (), length_unit: str = "") -> FieldView:
    """Build the view. ``fields``: dicts with ``name``, ``values`` (one per point or per cell, vectors as magnitude,
    6-component tensors as von Mises), optional ``where`` ("point" / "cell", from the length), ``label``, ``unit``,
    ``kind`` ("value" or "error"), ``group`` (shared colour range), ``deform`` (a key of ``deform``) and ``note``.
    ``deform``: key -> nodal displacement (n_points, 3) in the mesh's length unit. ``length_unit``: the mesh's length unit (shown with the displacements). ``markers``: dicts with ``name``,
    ``kind`` ("bc", "load", "hotspot", "region"), ``points`` (k, 3), optional ``detail``, ``centre`` (where the label
    goes; default: the member nearest the mean), ``label`` (False: points only) and ``fields`` (names of the fields it belongs to; default: shown with every field)."""
    from .scene import scalar_of
    n_pts, n_cells = int(mesh.n_points), int(mesh.n_cells)
    spec: dict[str, Any] = {}
    names: list[dict[str, Any]] = []
    for i, f in enumerate(fields):
        a = scalar_of(np.asarray(f["values"], float))
        where = f.get("where") or ("point" if len(a) == n_pts else "cell" if len(a) == n_cells else None)
        if where is None or len(a) != (n_pts if where == "point" else n_cells):
            raise ValueError(f"field {f['name']!r}: {len(a)} values for {n_pts} points and {n_cells} cells")
        key = f"f{i}"
        spec[key] = (where, a)
        names.append({"key": key, "name": str(f["name"]), "label": str(f.get("label") or f["name"]),
                      "unit": str(f.get("unit") or ""), "kind": str(f.get("kind") or "value"),
                      "group": str(f.get("group") or f["name"]), "deform": f.get("deform"), "where": where,
                      "note": str(f.get("note") or ""), "max_all": float(np.nanmax(np.abs(a))) if np.isfinite(a).any() else None})
    T = AXES[axes]
    dkeys: dict[str, dict[str, Any]] = {}
    for j, (k, u) in enumerate((deform or {}).items()):
        U = np.asarray(u, float).reshape(n_pts, -1)
        U = np.c_[U, np.zeros((n_pts, 3 - U.shape[1]))] if U.shape[1] < 3 else U[:, :3]
        Ug = U @ T.T                                            # in the viewer's (glTF) axes
        for c in range(3):
            spec[f"u{j}{'xyz'[c]}"] = ("point", Ug[:, c])
        dkeys[str(k)] = {"key": f"u{j}", "max": float(np.linalg.norm(U, axis=1).max()) if n_pts else 0.0,
                         "unit": length_unit}
    sc = Scene.from_mesh(mesh, fields=spec, title=title or "field view", axes=axes)
    n0 = sc.n_faces()
    if max_faces and n0 > max_faces:
        sc.decimate(max_faces)
    srf = sc.surfaces[0]
    lo, hi = sc.bounds()
    size = float(np.max(hi - lo)) or 1.0
    groups: dict[str, list[float]] = {}
    for f in names:
        r = colour_range(srf.fields[f["key"]], f["kind"])
        f.update(r)
        if f["kind"] != "error":
            g = groups.setdefault(f["group"], [np.inf, -np.inf])
            g[0], g[1] = min(g[0], r["lo"]), max(g[1], r["hi"])
    for f in names:                                              # one scale per group
        if f["kind"] != "error" and f["group"] in groups:
            f["lo"], f["hi"] = groups[f["group"]]
    for d in dkeys.values():
        d["auto_scale"] = float(0.1 * size / d["max"]) if d["max"] > 0 else 1.0
    from scipy.spatial import cKDTree
    tree = cKDTree(srf.vertices)
    mk = []
    for m in markers:
        P = np.asarray(m["points"], float).reshape(-1, 3)
        if not len(P):
            continue
        sub = subsample(P)
        # the label sits on the given centre, else on the member nearest the mean (a U-shaped set's mean is outside it)
        centre = np.asarray(m["centre"], float) if m.get("centre") is not None else \
            P[int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))]
        mk.append({"name": str(m["name"]), "kind": str(m.get("kind") or "region"), "detail": str(m.get("detail") or ""),
                   "n": int(len(P)), "points": _r(sub), "centre": _r(centre),
                   # nearest surface vertex of each point and of the label: the page moves them with the deformed shape
                   "vertices": tree.query(sub)[1].astype(int).tolist(), "anchor": int(tree.query(centre)[1]),
                   "bbox_min": _r(P.min(0)), "bbox_max": _r(P.max(0)),
                   "fields": [str(x) for x in m["fields"]] if m.get("fields") else None,
                   "label": bool(m.get("label", True))})
    meta = {"schema": "pinneapple-field-view/v1", "title": sc.title, "axes": axes, "fields": names,
            "deform": dkeys, "markers": mk, "bounds": {"min": _r(lo), "max": _r(hi), "size": size},
            "faces": sc.n_faces(), "faces_original": n0, "decimated": sc.n_faces() < n0, "max_faces": max_faces,
            "points": n_pts, "cells": n_cells, "notes": list(notes)}
    return FieldView(sc, meta)
