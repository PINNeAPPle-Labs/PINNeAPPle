"""Registry, parameter specs and the :class:`Geometry` object returned by :func:`build`.

A library entry is a :class:`GeometrySpec`: a builder ``fn(p, res) -> list[Body]`` (``p`` the physical parameters
with defaults filled in, ``res`` a resolution multiplier set by the level of detail), the documented parameters with
ranges, the documented overall dimensions, the reference frame, the sources of the real-world numbers and the
intended physics uses. Entries register themselves on import (``entries/``).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import meshops

#: vertex grid [m] applied by :func:`build` (reproducible files across platforms)
QUANTUM = 1e-7

#: resolution multiplier per level of detail (the builders scale their sampling with it)
LODS: dict[str, float] = {"light": 1.0, "fine": 2.5}

FRAME_VEHICLE = ("right-handed, metres: x aft from the nose (the free-stream direction), y to starboard (right), "
                 "z up; origin on the centre plane at the nose, z = 0 on the ground for ground vehicles")


@dataclass(frozen=True)
class Param:
    """One physical parameter: default, unit, admissible range (inclusive), what it controls."""
    name: str
    default: Any
    unit: str = ""
    range: tuple[float, float] | None = None
    doc: str = ""
    choices: tuple[Any, ...] | None = None

    def check(self, value):
        if self.choices is not None:
            if value not in self.choices:
                raise ValueError(f"{self.name}={value!r}: choose one of {self.choices}")
            return value
        if isinstance(self.default, bool):
            return bool(value)
        if isinstance(self.default, int) and not isinstance(self.default, bool):
            value = int(round(value))
        else:
            value = float(value)
        if self.range is not None and not (self.range[0] <= value <= self.range[1]):
            raise ValueError(f"{self.name}={value} {self.unit} is outside the documented range {self.range}")
        return value

    def as_dict(self) -> dict[str, Any]:
        d = {"name": self.name, "default": self.default, "unit": self.unit, "doc": self.doc}
        if self.range is not None:
            d["range"] = list(self.range)
        if self.choices is not None:
            d["choices"] = list(self.choices)
        return d


@dataclass(frozen=True)
class Source:
    """Where a real-world number comes from. ``estimate=True`` flags a value chosen by eye or by engineering
    judgement (no published figure)."""
    what: str
    ref: str
    estimate: bool = False


@dataclass
class Body:
    """One closed surface (watertight, outward normals) or, with ``closed=False``, an open sheet that is drawn only.

    ``materials`` is one material name, or one name per face (a livery, windows, lights on one closed skin).
    ``aero=True`` bodies form the physics shape (CFD / FEA); the others are details drawn only."""
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    materials: Any = "grey"
    group: str = ""
    aero: bool = True
    closed: bool = True

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if not isinstance(self.materials, str):
            self.materials = np.asarray(self.materials, dtype=object)
            if len(self.materials) != len(self.faces):
                raise ValueError(f"{self.name}: {len(self.materials)} face materials for {len(self.faces)} faces")

    def face_materials(self) -> np.ndarray:
        if isinstance(self.materials, str):
            return np.full(len(self.faces), self.materials, dtype=object)
        return self.materials


@dataclass
class GeometrySpec:
    name: str
    title: str
    category: str
    description: str
    builder: Callable[[dict[str, Any], float], list[Body]]
    params: Sequence[Param]
    dimensions: dict[str, tuple[float, float]]           # "length_x": (value_m, relative tolerance)
    frame: str
    sources: Sequence[Source]
    physics: Sequence[str]
    materials: Any = field(default_factory=dict)          # dict, or a callable returning one (lazy imports)
    tags: Sequence[str] = ()
    camera: dict[str, float] = field(default_factory=dict)  # thumbnail view: azimuth, elevation (deg)
    max_triangles: dict[str, int] = field(default_factory=lambda: {"light": 20000, "fine": 400000})
    legacy: str = ""                                       # the original class / function this entry wraps
    self_intersection_exempt: dict[str, str] = field(default_factory=dict)   # body -> why it is tolerated

    def material_table(self) -> dict[str, dict[str, Any]]:
        return dict(self.materials() if callable(self.materials) else self.materials)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def resolve(self, overrides: dict[str, Any]) -> dict[str, Any]:
        known = {p.name: p for p in self.params}
        bad = sorted(set(overrides) - set(known))
        if bad:
            raise TypeError(f"{self.name}: unknown parameter(s) {bad}; parameters: {sorted(known)}")
        out = self.defaults()
        for k, v in overrides.items():
            out[k] = known[k].check(v)
        return out

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name, "title": self.title, "category": self.category, "description": self.description,
            "units": "m", "frame": self.frame,
            "dimensions": {k: {"value_m": v[0], "tolerance": v[1]} for k, v in self.dimensions.items()},
            "parameters": [p.as_dict() for p in self.params],
            "sources": [{"what": s.what, "ref": s.ref, "estimate": s.estimate} for s in self.sources],
            "estimates": [s.what for s in self.sources if s.estimate],
            "physics": list(self.physics), "tags": list(self.tags),
            "lods": {k: {"resolution": LODS[k], "max_triangles": self.max_triangles[k]} for k in LODS},
            "legacy": self.legacy, "known_issues": dict(self.self_intersection_exempt),
        }


REGISTRY: dict[str, GeometrySpec] = {}


def register(spec: GeometrySpec) -> GeometrySpec:
    if spec.name in REGISTRY:
        raise ValueError(f"geometry '{spec.name}' is already registered")
    for s in spec.dimensions:
        if s not in ("length_x", "width_y", "height_z"):
            raise ValueError(f"{spec.name}: dimension keys are length_x, width_y, height_z (got {s})")
    REGISTRY[spec.name] = spec
    return spec


def _ensure_loaded() -> None:
    from . import entries  # noqa: F401  (imports every entry module, which registers itself)


def get_spec(name: str) -> GeometrySpec:
    _ensure_loaded()
    if name not in REGISTRY:
        raise KeyError(f"unknown geometry '{name}'; available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def params_hash(name: str, params: dict[str, Any], lod: str) -> str:
    blob = json.dumps({"name": name, "lod": lod, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ============================================================================================ the built object
@dataclass
class Geometry:
    """A built library geometry: closed bodies with materials, in metres, in the documented frame."""
    name: str
    bodies: list[Body]
    params: dict[str, Any]
    lod: str
    spec: GeometrySpec

    # ------------------------------------------------------------------ views of the geometry
    @property
    def title(self) -> str:
        return self.spec.title

    def n_triangles(self) -> int:
        return int(sum(len(b.faces) for b in self.bodies))

    def mesh(self, aero_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """All bodies concatenated (each stays a separate closed shell): ``(V, F)``."""
        bs = [b for b in self.bodies if b.aero or not aero_only]
        return meshops.concat([(b.vertices, b.faces) for b in bs])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        V = np.vstack([b.vertices for b in self.bodies])
        return V.min(0), V.max(0)

    def dimensions(self) -> dict[str, float]:
        lo, hi = self.bounds()
        d = hi - lo
        return {"length_x": float(d[0]), "width_y": float(d[1]), "height_z": float(d[2])}

    def volume(self, aero_only: bool = False) -> float:
        """Sum of the enclosed volumes of the closed bodies (overlaps counted twice; see :meth:`solid`)."""
        return float(sum(meshops.volume(b.vertices, b.faces) for b in self.bodies
                         if b.closed and (b.aero or not aero_only)))

    def area(self) -> float:
        return float(sum(meshops.area(b.vertices, b.faces) for b in self.bodies))

    def solid(self, aero_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Boolean union of the closed bodies: one watertight, self-intersection-free surface for meshing
        (needs ``manifold3d``)."""
        bs = [(b.vertices, b.faces) for b in self.bodies if b.closed and (b.aero or not aero_only)]
        return meshops.union(bs)

    def check(self) -> dict[str, Any]:
        """Quality report: per body watertight / consistently wound / outward, degenerate faces; overall bounds."""
        rep = {"bodies": {}, "n_triangles": self.n_triangles(), "dimensions": self.dimensions()}
        for b in self.bodies:
            rep["bodies"][b.name] = meshops.quality(b.vertices, b.faces, closed=b.closed)
        rep["watertight"] = all(r["watertight"] for n, r in rep["bodies"].items()
                                if self._body(n).closed)
        rep["outward"] = all(r["volume"] > 0 for n, r in rep["bodies"].items() if self._body(n).closed)
        return rep

    def _body(self, name: str) -> Body:
        return next(b for b in self.bodies if b.name == name)

    # ------------------------------------------------------------------ outputs
    def surfaces(self):
        """Studio surfaces, one per (body, material)."""
        from pinneapple_tools.visualization.studio.scene import Surface
        out = []
        for b in self.bodies:
            fm = b.face_materials()
            for m in dict.fromkeys(fm):
                sel = fm == m
                V, F = meshops.compact(b.vertices, b.faces[sel])
                nm = b.name if len(set(fm)) == 1 else f"{b.name}__{m}"
                out.append(Surface(nm, V, F, str(m), group=b.group or b.name))
        return out

    def scene(self):
        """A :class:`pinneapple_tools.visualization.studio.Scene` (glTF / USD / STL, Blender, browser viewer)."""
        from pinneapple_tools.visualization.studio.scene import MATERIALS, Scene
        MATERIALS.update({k: v for k, v in self.spec.material_table().items() if k not in MATERIALS})
        return Scene(self.surfaces(), axes="z_up", title=self.spec.title)

    def export(self, path: str, fmt: str | None = None) -> str:
        """Write ``.stl`` (binary), ``.obj``, ``.glb`` (PBR materials), ``.usda`` / ``.usd`` or ``.step`` (faceted
        B-rep, one solid per closed body)."""
        from .exporters import export
        return export(self, path, fmt)

    def render(self, path: str | None = None, size: tuple[int, int] = (960, 640), **kw):
        """Software-rendered thumbnail (PNG / JPG); returns the PIL image."""
        from .render import render_geometry
        return render_geometry(self, path, size=size, **kw)

    def __repr__(self) -> str:
        d = self.dimensions()
        return (f"Geometry('{self.name}', lod={self.lod}, bodies={len(self.bodies)}, triangles={self.n_triangles()}, "
                f"L x W x H = {d['length_x']:.3g} x {d['width_y']:.3g} x {d['height_z']:.3g} m)")


def build(name: str, lod: str = "light", **params) -> Geometry:
    """Build a library geometry. ``lod`` = "light" (the committed files) or "fine" (Hub files); every other keyword
    is a physical parameter (see :func:`info`). The defaults reproduce the stored files."""
    spec = get_spec(name)
    if lod not in LODS:
        raise ValueError(f"lod must be one of {sorted(LODS)}")
    p = spec.resolve(params)
    bodies = spec.builder(dict(p), LODS[lod])
    for b in bodies:
        if len(np.unique(b.faces)) < len(b.vertices):          # drop unreferenced vertices (unused cap centres)
            b.vertices, b.faces = meshops.compact(b.vertices, b.faces)
        if b.closed:
            b.vertices, b.faces = meshops.orient_outward(b.vertices, b.faces)
        # snap to a 0.1 um grid: last-bit differences between NumPy / SciPy builds vanish, so every machine writes
        # the same bytes as the manifest (the grid is far below any feature of the library); "+ 0.0" turns -0 into +0:
        # whether a coordinate that rounds to zero keeps its sign depends on the SIMD path NumPy takes on the CPU, and
        # the text and STL writers keep the sign
        b.vertices = np.round(b.vertices / QUANTUM) * QUANTUM + 0.0
    names = [b.name for b in bodies]
    if len(set(names)) != len(names):
        raise RuntimeError(f"{name}: duplicate body names")
    return Geometry(name, bodies, p, lod, spec)


def names() -> list[str]:
    _ensure_loaded()
    return sorted(REGISTRY)


def info(name: str) -> dict[str, Any]:
    return get_spec(name).info()


def summary_table() -> str:
    """Markdown table of the library (name, title, size, uses)."""
    rows = ["| name | geometry | L x W x H (m) | physics |", "|---|---|---|---|"]
    for n in names():
        s = REGISTRY[n]
        d = s.dimensions
        dims = " x ".join(f"{d[k][0]:.4g}" for k in ("length_x", "width_y", "height_z") if k in d)
        rows.append(f"| `{n}` | {s.title} | {dims} | {'; '.join(s.physics[:2])} |")
    return "\n".join(rows)


__all__ = ["Param", "Source", "Body", "GeometrySpec", "Geometry", "register", "build", "names", "info", "get_spec",
           "LODS", "FRAME_VEHICLE", "params_hash", "summary_table"]
