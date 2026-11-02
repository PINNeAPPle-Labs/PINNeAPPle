"""The part of the extension that needs no Blender: the library, the body cache, diagnostic colours and the
``MeshData`` export. Everything here works on numpy arrays and :class:`TessMesh` objects."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DIAGNOSTICS = {
    "MEAN": "Mean curvature H",
    "GAUSS": "Gaussian curvature K",
    "DEVIATION": "Mesh curvature error |H_mesh - H_exact|",
    "ZEBRA": "Zebra stripes (fixed eye)",
    "HIGHLIGHT": "Highlight lines",
}


def brep():
    """The library's ``brep`` package: the installed PINNeAPPle, or the copy the build put next to this file."""
    try:
        from pinneapple_design.geometry import brep as b
    except ImportError:
        from .vendor import brep as b  # type: ignore[no-redef]
    return b


# ------------------------------------------------------------------------------------------------ the body cache
_BODIES: dict[str, list] = {}


def bodies(path: str) -> list:
    """The exact bodies of a STEP file, read once per session."""
    key = str(Path(path).resolve())
    if key not in _BODIES:
        _BODIES[key] = brep().read_step(key)
    return _BODIES[key]


def clear_cache() -> None:
    _BODIES.clear()


# ------------------------------------------------------------------------------------------------------ colours
_VIRIDIS = np.array([[0.267, 0.005, 0.329], [0.229, 0.322, 0.546], [0.128, 0.567, 0.551], [0.369, 0.789, 0.383],
                     [0.993, 0.906, 0.144]])
_DIVERGING = np.array([[0.230, 0.299, 0.754], [0.865, 0.865, 0.865], [0.706, 0.016, 0.150]])


def _lut(t: np.ndarray, table: np.ndarray) -> np.ndarray:
    t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0) * (len(table) - 1)
    i = np.minimum(t.astype(int), len(table) - 2)
    f = (t - i)[:, None]
    return table[i] * (1 - f) + table[i + 1] * f


def colours(values: np.ndarray, mode: str = "sequential", clip_percentile: float = 98.0) -> np.ndarray:
    """RGB in [0, 1] per value. ``sequential`` maps the [min, p] range on a viridis ramp; ``diverging`` centres zero
    (blue negative, red positive) and scales by the ``clip_percentile`` of |value|; ``gray`` maps [0, 1] to grey."""
    v = np.asarray(values, float)
    if mode == "gray":
        g = np.clip(v, 0, 1)
        return np.stack([g, g, g], 1)
    if mode == "diverging":
        s = float(np.nanpercentile(np.abs(v), clip_percentile)) or 1.0
        return _lut(0.5 + 0.5 * v / s, _DIVERGING)
    lo, hi = float(np.nanmin(v)), float(np.nanpercentile(v, clip_percentile))
    return _lut((v - lo) / max(hi - lo, 1e-300), _VIRIDIS)


def diagnostic(body, mesh, mode: str, *, stripes: int = 16, eye=None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Per-vertex values, their RGB colours and a summary for one of :data:`DIAGNOSTICS` on ``mesh`` (a
    :class:`TessMesh` of ``body``). Curvatures are in 1/m and 1/m^2."""
    D = brep().diagnostics
    if mode not in DIAGNOSTICS:
        raise ValueError(f"unknown diagnostic {mode!r}; one of {', '.join(DIAGNOSTICS)}")
    c = D.exact_curvature(body, mesh)
    if mode == "MEAN":
        v = c["H"]
        rgb = colours(v, "diverging")
    elif mode == "GAUSS":
        v = c["K"]
        rgb = colours(v, "diverging")
    elif mode == "DEVIATION":
        v = np.abs(D.discrete_curvature(mesh)["H"] - c["H"])
        rgb = colours(v, "sequential")
    elif mode == "ZEBRA":
        v = D.zebra(mesh, c["normal"], n_stripes=stripes, eye=eye)
        rgb = colours(v, "gray")
    else:
        v = D.highlight_lines(mesh, c["normal"])
        rgb = colours(v, "gray")
    summary = {"mode": mode, "label": DIAGNOSTICS[mode], "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
               "n_vertices": int(len(v))}
    return v, rgb, summary


# ------------------------------------------------------------------------------------------ to the rest of PINNeAPPle
def write_meshdata(mesh, path: str, name: str = "") -> dict[str, Any]:
    """Write the arrays of ``MeshData`` (vertices and faces in metres, face normals) and the body face of each
    triangle to ``path`` (``.npz``), with a ``.json`` summary next to it. Load with
    ``MeshData(**{k: d[k] for k in ("vertices", "faces", "normals")})`` where ``d = np.load(path)``."""
    V, T = np.asarray(mesh.vertices, float), np.asarray(mesh.triangles, np.int64)
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, vertices=V, faces=T, normals=n, face_id=np.asarray(mesh.face_id, np.int64))
    summary = {"name": name, "n_vertices": int(len(V)), "n_faces": int(len(T)), "units": "m",
               "bbox_m": [V.min(0).tolist(), V.max(0).tolist()], "watertight": bool(mesh.is_watertight()),
               "volume_m3": float(mesh.volume()), "area_m2": float(mesh.area())}
    p.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    return {"path": str(p), **summary}
