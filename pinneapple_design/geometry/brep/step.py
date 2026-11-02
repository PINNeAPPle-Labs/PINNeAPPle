"""Read STEP files into exact B-rep bodies (#1115).

Every solid or shell of the file becomes a :class:`CadBody` in SI units (metres): its name, the kernel's exact
volume, area and bounding box, and one :class:`CadFace` per face with the surface type and its parameters
(plane origin and normal; cylinder, cone, sphere and torus axis and radii; the cone half-angle). The kernel object
stays on the body (``body.shape``) for meshing and surface diagnostics (#1116, #1117).

Backend: SMLib through ``smlibpy`` (``pip install "pinneapple[brep]"``). It reads AP203 / AP214 / AP242 B-rep data,
applies assembly placements, keeps names and colours, heals invalid bodies and works in millimetres; this module
converts to metres. ``backend="auto"`` picks the first installed backend.
"""
from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MM = 1e-3                      # the kernel's length unit, in metres

_BACKENDS = ("smlibpy",)


def available_backends() -> list[str]:
    """The installed exact-B-rep backends, in the order ``backend="auto"`` tries them."""
    out = []
    for name in _BACKENDS:
        try:
            __import__(name)
            out.append(name)
        except Exception:  # noqa: BLE001  (an optional dependency may fail to import for any reason)
            pass
    return out


def _require(backend: str) -> Any:
    if backend == "auto":
        found = available_backends()
        if not found:
            raise ImportError('no exact B-rep backend: pip install "pinneapple[brep]" (smlibpy). For STEP -> mesh '
                              "without one, use pinneapple_design.geometry.io.step.step_to_mesh (gmsh).")
        backend = found[0]
    if backend != "smlibpy":
        raise ValueError(f"unknown backend {backend!r}; available: {list(_BACKENDS)}")
    import smlibpy
    return smlibpy


@dataclass
class CadFace:
    """One face of a body: surface type, area and the parameters of its surface (SI units)."""
    index: int
    kind: str                         # plane | cylinder | cone | sphere | torus | bspline | ... (kernel's name)
    area_m2: float
    params: dict[str, Any] = field(default_factory=dict)
    shape: Any = field(default=None, repr=False)


@dataclass
class CadBody:
    """One solid or shell of a STEP file, exact, in metres."""
    name: str
    backend: str
    volume_m3: float
    area_m2: float
    bbox_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    faces: list[CadFace]
    n_edges: int
    n_vertices: int
    is_solid: bool
    valid: bool
    shape: Any = field(default=None, repr=False)

    def face_counts(self) -> dict[str, int]:
        """Number of faces of each surface type."""
        return dict(Counter(f.kind for f in self.faces))

    @property
    def size_m(self) -> np.ndarray:
        """Bounding-box extents (dx, dy, dz) [m]."""
        lo, hi = (np.asarray(v) for v in self.bbox_m)
        return hi - lo

    def summary(self) -> dict[str, Any]:
        """Plain-data description (JSON-ready)."""
        return {"name": self.name, "backend": self.backend, "volume_m3": self.volume_m3, "area_m2": self.area_m2,
                "bbox_m": [list(self.bbox_m[0]), list(self.bbox_m[1])], "faces": len(self.faces),
                "face_counts": self.face_counts(), "edges": self.n_edges, "vertices": self.n_vertices,
                "is_solid": self.is_solid, "valid": self.valid}


def _vec(v: Any, scale: float = 1.0) -> list[float]:
    return [float(c) * scale for c in v]


def _attr(obj: Any, name: str) -> Any:
    """A face attribute that the kernel only defines for some surface types (else None)."""
    try:
        v = getattr(obj, name)
        return v() if callable(v) else v
    except (ValueError, AttributeError, RuntimeError):
        return None


def _face_params(f: Any, kind: str) -> dict[str, Any]:
    p: dict[str, Any] = {}
    if kind == "plane":
        pl = _attr(f, "plane")
        if pl is not None:
            p["origin_m"] = _vec(pl.origin, MM)
            p["normal"] = _vec(pl.z_dir)
        return p
    ax = _attr(f, "axis")
    if ax is not None and hasattr(ax, "direction"):
        p["axis_origin_m"] = _vec(ax.origin, MM)
        p["axis_direction"] = _vec(ax.direction)
    if kind in ("cylinder", "sphere"):
        r = _attr(f, "radius")
        if r is not None:
            p["radius_m"] = float(r) * MM
    elif kind == "cone":
        r, a = _attr(f, "radius"), _attr(f, "cone_angle")
        if r is not None:
            p["reference_radius_m"] = float(r) * MM
        if a is not None:
            p["half_angle_deg"] = float(a)
    elif kind == "torus":
        R, r = _attr(f, "major_radius"), _attr(f, "radius")
        if R is not None:
            p["major_radius_m"] = float(R) * MM
        if r is not None:
            p["minor_radius_m"] = float(r) * MM
    return p


def _body(shape: Any, index: int, backend: str) -> CadBody:
    faces = []
    for i, f in enumerate(shape.faces()):
        kind = str(f.geom_type).lower()
        faces.append(CadFace(i, kind, float(f.area) * MM * MM, _face_params(f, kind), f))
    bb = shape.bbox()
    lo = _vec(bb.min, MM) if hasattr(bb, "min") else _vec(bb[0], MM)
    hi = _vec(bb.max, MM) if hasattr(bb, "max") else _vec(bb[1], MM)
    vol = float(getattr(shape, "volume", 0.0) or 0.0)
    valid = _attr(shape, "is_valid")
    name = getattr(shape, "name", None) or f"body_{index}"
    return CadBody(name=str(name), backend=backend, volume_m3=vol * MM**3, area_m2=float(shape.area) * MM**2,
                   bbox_m=(tuple(lo), tuple(hi)), faces=faces, n_edges=len(shape.edges()),
                   n_vertices=len(shape.vertices()), is_solid=type(shape).__name__ == "Part" and vol > 0,
                   valid=bool(valid) if valid is not None else True, shape=shape)


def read_step(path: str | os.PathLike[str], *, backend: str = "auto", heal: bool = True) -> list[CadBody]:
    """Every solid and shell of a STEP file as an exact :class:`CadBody` in metres.

    Parameters
    ----------
    path : str
        The STEP file (.step / .stp).
    backend : {"auto", "smlibpy"}
        Exact B-rep kernel; "auto" picks the first installed.
    heal : bool
        Let the kernel repair bodies that come out invalid (valid ones are left alone).
    """
    sm = _require(backend)
    res = sm.import_step(os.fspath(path), heal=heal)
    shapes = list(res) if isinstance(res, (list, tuple)) or type(res).__name__ == "ShapeList" else [res]
    return [_body(s, i, "smlibpy") for i, s in enumerate(shapes)]


def write_step(shapes: Any, path: str | os.PathLike[str], *, unit: str = "mm", name: str | None = None) -> None:
    """Write kernel shapes (or :class:`CadBody` objects) to an exact STEP AP214 file.

    ``unit`` is the length unit written into the file ("mm", "cm", "m", "inch", ...); the geometry is the same.
    """
    sm = _require("auto")
    items = shapes if isinstance(shapes, Sequence) and not isinstance(shapes, str) else [shapes]
    raw = [s.shape if isinstance(s, CadBody) else s for s in items]
    sm.export_step(raw if len(raw) > 1 else raw[0], os.fspath(path), unit=unit, name=name)
