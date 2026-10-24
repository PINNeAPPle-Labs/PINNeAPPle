"""Exact geometry with NVIDIA USD BRep (the SMLib NURBS B-rep kernel, ``pip install usd-brep``), measured for
PINNeAPPle: mass properties of the exact solid, curvature of every face, tessellations at several chord tolerances
with their measured deviation from the exact surfaces, and B-rep USD export.

    part = PartRecipe().box("plate", (0, 0, 0), (0.1, 0.06, 0.01)) \\
                       .cylinder("hole", (0.05, 0.03, -0.005), 0.01, 0.02).difference("bracket", "plate", "hole")
    rep = evaluate_part(part, chord_tolerances=[1e-4, 1e-5], density=2700.0, usd_brep_path="bracket_brep.usda")
    rep.volume, rep.min_radius, rep.tessellations[-1]["max_deviation"], rep.mesh(-1)   # (V, F) of the finest mesh

A part is a recipe (a list of operations on named solids), so it is data: it can be stored with a result,
regenerated, varied by a design study, or written by a model service. The kernel runs in a worker process
(:mod:`pinneapple_tools.nvidia._brep_worker`) in whichever interpreter has ``usd_brep``: the current one if it
imports, else ``$PINNEAPPLE_BREP_PYTHON`` (the ``usd-brep`` wheel is built for CPython 3.12 only:
``python3.12 -m venv /opt/brep && /opt/brep/bin/pip install usd-brep numpy`` and point the variable at
``/opt/brep/bin/python``).

Checks it gives a trust report: the solid is manifold; mesh volume and area converge to the exact ones; the
largest distance from a mesh triangle to the exact surface (chord error) is below the tolerance asked; the mesh
size resolves the smallest radius of curvature of the part (holes, fillets).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["PartRecipe", "BRepReport", "evaluate_part", "brep_python", "brep_available", "geometry_checks",
           "planar_face_mesh", "vertex_normals", "mesh_thickness", "point_mesh_distance"]

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_brep_worker.py")


def brep_python() -> str | None:
    """The interpreter that has ``usd_brep``: this one, else ``$PINNEAPPLE_BREP_PYTHON`` (if it exists)."""
    if importlib.util.find_spec("usd_brep") is not None:
        return sys.executable
    p = os.environ.get("PINNEAPPLE_BREP_PYTHON")
    return p if p and os.path.exists(p) else None


def brep_available() -> bool:
    return brep_python() is not None


@dataclass
class PartRecipe:
    ops: list[dict[str, Any]] = field(default_factory=list)
    result: str | None = None

    def _add(self, **op) -> PartRecipe:
        if any(o["name"] == op["name"] for o in self.ops):
            raise ValueError(f"solid name {op['name']!r} is already used")
        self.ops.append(op)
        self.result = op["name"]
        return self

    def box(self, name, origin, size):
        return self._add(op="box", name=name, origin=list(map(float, origin)), size=list(map(float, size)))

    def cylinder(self, name, origin, radius, height):
        """Axis along +z from ``origin``."""
        return self._add(op="cylinder", name=name, origin=list(map(float, origin)), radius=float(radius),
                         height=float(height))

    def sphere(self, name, origin, radius):
        return self._add(op="sphere", name=name, origin=list(map(float, origin)), radius=float(radius))

    def union(self, name, a, b):
        return self._add(op="union", name=name, a=a, b=b)

    def difference(self, name, a, b):
        return self._add(op="difference", name=name, a=a, b=b)

    def translate(self, name, src, vector):
        return self._add(op="translate", name=name, src=src, vector=list(map(float, vector)))

    def rotate(self, name, src, point, axis, angle_deg):
        return self._add(op="rotate", name=name, src=src, point=list(map(float, point)),
                         axis=list(map(float, axis)), angle_deg=float(angle_deg))

    def fillet_edges_near(self, name, src, points, radius):
        """Round the edges closest to ``points`` with ``radius``."""
        return self._add(op="fillet_edges_near", name=name, src=src, points=[list(map(float, p)) for p in points],
                         radius=float(radius))

    def extrude_polygon(self, name, points, direction, distance):
        """A closed planar polygon (3-D points) swept along ``direction`` by ``distance``."""
        return self._add(op="extrude_polygon", name=name, points=[list(map(float, p)) for p in points],
                         direction=list(map(float, direction)), distance=float(distance))

    def to_dict(self) -> dict[str, Any]:
        return {"ops": self.ops, "result": self.result}

    @classmethod
    def from_dict(cls, d) -> PartRecipe:
        return cls(list(d["ops"]), d.get("result"))


@dataclass
class BRepReport:
    raw: dict[str, Any]
    meshes: list[tuple[np.ndarray, np.ndarray]]
    seconds: float = 0.0

    def __getattr__(self, k):
        if k in ("raw", "meshes"):
            raise AttributeError(k)
        try:
            return self.raw[k]
        except KeyError:
            raise AttributeError(k) from None

    @property
    def min_radius(self) -> float:
        """Smallest radius of curvature over the faces (inf for a part made only of planes)."""
        k = max((c["max_curvature"] for c in self.raw["curvature"]), default=0.0)
        return 1.0 / k if k > 0 else float("inf")

    def mesh(self, i: int = -1) -> tuple[np.ndarray, np.ndarray]:
        return self.meshes[i]


def evaluate_part(recipe: PartRecipe, *, chord_tolerances=(1e-3, 1e-4), density: float = 1.0,
                  angle_deg: float = 25.0, max_edge_length: float = 0.0, usd_brep_path: str | None = None,
                  max_samples: int = 4000, vertex_curvature: bool = False, timeout: float = 600.0) -> BRepReport:
    """Build ``recipe`` with USD BRep and measure it (see the module docstring). ``vertex_curvature``: the exact
    surface curvature at every vertex of the finest mesh (``rep.vertex_curvature``, 1/m) and the B-rep face each
    vertex lies on (``rep.vertex_face``)."""
    import time
    py = brep_python()
    if py is None:
        raise ImportError("USD BRep is not available: pip install usd-brep (CPython 3.12 wheel), or set "
                          "PINNEAPPLE_BREP_PYTHON to a Python 3.12 that has it")
    req = {"recipe": recipe.to_dict(), "options": {"chord_tolerances": list(map(float, chord_tolerances)),
                                                   "density": float(density), "angle_deg": float(angle_deg),
                                                   "max_edge_length": float(max_edge_length),
                                                   "max_samples": int(max_samples),
                                                   "vertex_curvature": bool(vertex_curvature),
                                                   "usd_brep_path": os.path.abspath(usd_brep_path)
                                                   if usd_brep_path else None}}
    with tempfile.TemporaryDirectory() as d:
        rq, out = os.path.join(d, "req.json"), os.path.join(d, "out.npz")
        with open(rq, "w") as f:
            json.dump(req, f)
        t0 = time.perf_counter()
        r = subprocess.run([py, "-I", WORKER, rq, out], capture_output=True, text=True, timeout=timeout, cwd=d)
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            tail = [ln for ln in r.stderr.splitlines() if "scriptModuleLoader" not in ln and "pxr" not in ln]
            raise RuntimeError("USD BRep worker failed:\n" + "\n".join(tail[-20:]))
        data = np.load(out)
        raw = json.loads(bytes(data["report"]).decode())
        meshes = [(data[f"V{j}"], data[f"F{j}"]) for j in range(len(raw["tessellations"]))]
        if "vertex_curvature" in data:
            raw["vertex_curvature"] = data["vertex_curvature"]
            raw["vertex_face"] = data["vertex_face"]
    return BRepReport(raw, meshes, dt)


def geometry_checks(rep: BRepReport, *, element_size: float | None = None, points_per_radius: float = 4.0,
                    volume_rtol: float = 1e-3) -> list[dict[str, Any]]:
    """Trust checks of the geometry handed to a solver or a surrogate (the finest tessellation): manifold solid,
    chord error below the tolerance asked, mesh volume and area against the exact solid, and (with
    ``element_size``) the element size against the smallest radius of curvature."""
    t = rep.tessellations[-1]
    out = [{"name": "B-rep is a manifold solid", "value": None, "limit": None, "passed": bool(rep.manifold)},
           {"name": "mesh is a manifold solid", "value": None, "limit": None, "passed": bool(t["manifold"])},
           {"name": "chord error <= tolerance", "value": t["max_deviation"], "limit": t["chord_tolerance"],
            "passed": t["max_deviation"] <= t["chord_tolerance"] * (1 + 1e-9),
            "detail": "largest distance from a triangle centroid to the exact surface"},
           {"name": "mesh volume vs exact", "value": abs(t["mesh_volume"] - rep.volume) / rep.volume,
            "limit": volume_rtol, "passed": abs(t["mesh_volume"] - rep.volume) / rep.volume <= volume_rtol},
           {"name": "mesh area vs exact", "value": abs(t["mesh_area"] - rep.area) / rep.area,
            "limit": volume_rtol, "passed": abs(t["mesh_area"] - rep.area) / rep.area <= volume_rtol}]
    if element_size is not None and np.isfinite(rep.min_radius):
        need = rep.min_radius * 2 * np.pi / (4 * points_per_radius)
        out.append({"name": f"element size resolves the smallest radius ({rep.min_radius * 1e3:.2f} mm)",
                    "value": float(element_size), "limit": float(need), "passed": element_size <= need,
                    "detail": f"{points_per_radius:g} elements per quarter circle"})
    return out


def planar_face_mesh(V, F, *, axis: int = 2, value: float, atol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """The triangles of a tessellation lying in the plane ``x[axis] = value`` (e.g. the top face of a plate), as a
    2-D mesh ``(V2 (n, 2), F2 (m, 3))`` in the other two coordinates, renumbered."""
    V, F = np.asarray(V, float), np.asarray(F, int)
    on = np.all(np.abs(V[F][:, :, axis] - value) <= atol, axis=1)
    Fp = F[on]
    used = np.unique(Fp)
    idx = -np.ones(len(V), int)
    idx[used] = np.arange(len(used))
    keep = [a for a in range(3) if a != axis]
    return V[used][:, keep], idx[Fp]


def vertex_normals(V, F) -> np.ndarray:
    """Area-weighted unit vertex normals (outward for a consistently oriented closed mesh)."""
    V, F = np.asarray(V, float), np.asarray(F, int)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], n)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)


def mesh_thickness(V, F, *, normals=None, eps: float = 1e-7, chunk: int = 256) -> np.ndarray:
    """Wall thickness at every vertex of a closed mesh: distance along the inward normal to the first triangle hit
    (Moller-Trumbore ray casting); ``inf`` where the ray leaves without a hit."""
    V, F = np.asarray(V, float), np.asarray(F, int)
    N = vertex_normals(V, F) if normals is None else np.asarray(normals, float)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = B - A, C - A
    out = np.full(len(V), np.inf)
    for s0 in range(0, len(V), chunk):
        o = V[s0:s0 + chunk, None, :] - N[s0:s0 + chunk, None, :] * eps
        d = -N[s0:s0 + chunk, None, :]
        pv = np.cross(d, e2[None])
        det = np.einsum("tij,ij->ti", pv, e1)
        ok = np.abs(det) > 1e-18
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - A[None]
        u = np.einsum("tij,tij->ti", tv, pv) * inv
        qv = np.cross(tv, e1[None])
        v = np.einsum("tij,tij->ti", d, qv) * inv
        t = np.einsum("ij,tij->ti", e2, qv) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps)
        t = np.where(hit, t, np.inf)
        out[s0:s0 + chunk] = t.min(1)
    return out


def point_mesh_distance(P, V, F, chunk: int = 128) -> np.ndarray:
    """Distance from each point to the nearest triangle vertex or centroid (a cheap upper bound on the distance to
    the mesh, used to sample the deviation of a mesh from points of an exact surface)."""
    Q = np.r_[np.asarray(V, float), np.asarray(V, float)[np.asarray(F, int)].mean(1)]
    out = np.empty(len(P))
    for s0 in range(0, len(P), chunk):
        out[s0:s0 + chunk] = np.linalg.norm(np.asarray(P)[s0:s0 + chunk, None] - Q[None], axis=2).min(1)
    return out
