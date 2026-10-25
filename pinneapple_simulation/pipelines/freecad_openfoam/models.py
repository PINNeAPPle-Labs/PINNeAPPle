"""Parametric FreeCAD templates the pipeline ships, with their default patch rules and a NumPy twin for tests.

Each template is a FreeCAD document whose dimensions are expressions of a spreadsheet ``Params`` (built by
``freecad_scripts/fc_job.py``; :func:`build_document` writes it) and a :class:`Model` here: parameter defaults and
bounds (mm, degrees), the solid to export, the rules that name its faces, and :meth:`Model.fake`, the same faces in
NumPy (:class:`~.cad.FakeFreeCAD`).

=============  ========  ==================================================================  ================
template       flow      geometry                                                            patches
=============  ========  ==================================================================  ================
pipe_bend      internal  inlet leg, bend of centre-line radius Rc through ``angle``, outlet  inlet, outlet,
                         leg; diameter D                                                     walls
heated_pipe    internal  straight pipe: unheated entry L_entry, then heated L_heated          inlet, outlet,
                                                                                             wall_entry,
                                                                                             wall_heated
spheroid       external  spheroid of length L along the flow and diameter D (L = D: sphere)  body
=============  ========  ==================================================================  ================
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Model", "MODELS", "build_document"]


@dataclass
class Model:
    name: str
    title: str
    flow: str                                            # internal | external
    solid: str
    parameters: dict[str, dict[str, Any]]                # alias -> {default, min, max, unit, description}
    patches: dict[str, Any]
    fake: Callable[[dict[str, float], float], list[dict[str, Any]]] = field(repr=False, default=None)

    def defaults(self) -> dict[str, float]:
        return {k: float(v["default"]) for k, v in self.parameters.items()}


# ----------------------------------------------------------------------------------------------- NumPy twin
def _n_around(r: float, tol: float) -> int:
    return max(24, int(math.ceil(math.pi / math.acos(max(-1.0, 1.0 - tol / max(r, 1e-12))))))


def _orient(P: np.ndarray, T: np.ndarray, outward: np.ndarray) -> np.ndarray:
    """Flip the triangles whose normal points against ``outward`` (one vector per triangle)."""
    n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    flip = np.einsum("ij,ij->i", n, outward) < 0
    T = T.copy()
    T[flip] = T[flip][:, [0, 2, 1]]
    return T


def _face(i: int, kind: str, P: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    cen = ((a + b + c) / 3 * ar[:, None]).sum(0) / ar.sum()
    return {"index": i, "type": kind, "area": float(ar.sum()), "centre": cen.tolist(), "points": P.tolist(),
            "triangles": T.tolist()}


def _tube(centres: np.ndarray, tangents: np.ndarray, r: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Lateral surface through rings of radius r around a planar (x-y) centre line: points, triangles."""
    z = np.array([0.0, 0.0, 1.0])
    phi = 2 * np.pi * np.arange(n) / n
    rings = []
    for c, t in zip(centres, tangents, strict=True):
        e = np.cross(z, t)
        rings.append(c + r * (np.cos(phi)[:, None] * z + np.sin(phi)[:, None] * e))
    P = np.concatenate(rings)
    T = []
    for k in range(len(rings) - 1):
        for j in range(n):
            a, b = k * n + j, k * n + (j + 1) % n
            c, d = (k + 1) * n + (j + 1) % n, (k + 1) * n + j
            T += [[a, b, c], [a, c, d]]
    T = np.asarray(T)
    ring = np.repeat(np.arange(len(rings) - 1), 2 * n)
    out = P[T].mean(1) - 0.5 * (centres[ring] + centres[ring + 1])
    return P, _orient(P, T, out)


def _cap(c: np.ndarray, t: np.ndarray, r: float, n: int, outward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P, _ = _tube(np.array([c, c]), np.array([t, t]), r, n)
    P = np.vstack([c, P[:n]])
    T = np.array([[0, 1 + j, 1 + (j + 1) % n] for j in range(n)])
    return P, _orient(P, T, np.repeat(outward[None], n, 0))


def _line(a: np.ndarray, b: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    m = max(2, int(math.ceil(np.linalg.norm(b - a) / h)) + 1)
    s = np.linspace(0.0, 1.0, m)[:, None]
    t = (b - a) / np.linalg.norm(b - a)
    return a + s * (b - a), np.repeat(t[None], m, 0)


def _fake_pipe_bend(p: dict[str, float], tol: float) -> list[dict[str, Any]]:
    r, Rc, ang, Lin, Lout = p["D"] / 2, p["Rc"], math.radians(p["angle"]), p["L_in"], p["L_out"]
    n = _n_around(r, tol)
    h = 2 * np.pi * r / n * 2
    c1, t1 = _line(np.array([-Lin, 0, 0.0]), np.zeros(3), h)
    m = max(4, _n_around(Rc + r, tol) * ang / (2 * np.pi))
    th = np.linspace(0, ang, int(math.ceil(m)) + 1)
    c2 = np.c_[Rc * np.sin(th), Rc - Rc * np.cos(th), 0 * th]
    t2 = np.c_[np.cos(th), np.sin(th), 0 * th]
    end, d = c2[-1], t2[-1]
    c3, t3 = _line(end, end + Lout * d, h)
    faces = []
    for i, (kind, (P, T)) in enumerate([("Cylinder", _tube(c1, t1, r, n)), ("Toroid", _tube(c2, t2, r, n)),
                                        ("Cylinder", _tube(c3, t3, r, n)),
                                        ("Plane", _cap(c1[0], t1[0], r, n, -t1[0])),
                                        ("Plane", _cap(c3[-1], t3[-1], r, n, t3[-1]))]):
        faces.append(_face(i, kind, P, T))
    return faces


def _fake_heated_pipe(p: dict[str, float], tol: float) -> list[dict[str, Any]]:
    r, Le, Lh = p["D"] / 2, p["L_entry"], p["L_heated"]
    n = _n_around(r, tol)
    h = 2 * np.pi * r / n * 2
    x = np.array([1.0, 0, 0])
    c1, t1 = _line(np.zeros(3), Le * x, h)
    c2, t2 = _line(Le * x, (Le + Lh) * x, h)
    return [_face(i, k, P, T) for i, (k, (P, T)) in enumerate(
        [("Cylinder", _tube(c1, t1, r, n)), ("Cylinder", _tube(c2, t2, r, n)),
         ("Plane", _cap(c1[0], x, r, n, -x)), ("Plane", _cap(c2[-1], x, r, n, x))])]


def _fake_spheroid(p: dict[str, float], tol: float) -> list[dict[str, Any]]:
    a, b = p["L"] / 2, p["D"] / 2
    n = _n_around(b, tol)
    m = max(12, n // 2)
    th = np.linspace(0, np.pi, m + 1)[1:-1]
    phi = 2 * np.pi * np.arange(n) / n
    P = [[-a, 0, 0]] + [[-a * math.cos(t), b * math.sin(t) * math.cos(f), b * math.sin(t) * math.sin(f)]
                        for t in th for f in phi] + [[a, 0, 0]]
    P = np.asarray(P)
    T = [[0, 1 + (j + 1) % n, 1 + j] for j in range(n)]
    for k in range(len(th) - 1):
        for j in range(n):
            q, w = 1 + k * n + j, 1 + k * n + (j + 1) % n
            T += [[q, w, w + n], [q, w + n, q + n]]
    last = 1 + (len(th) - 1) * n
    T += [[len(P) - 1, last + j, last + (j + 1) % n] for j in range(n)]
    T = np.asarray(T)
    c = P[T].mean(1)
    out = c / np.array([a * a, b * b, b * b])
    return [_face(0, "Sphere" if abs(a - b) < 1e-12 else "BSplineSurface", P, _orient(P, T, out))]


MODELS: dict[str, Model] = {
    "pipe_bend": Model(
        "pipe_bend", "Pipe bend (inlet leg, bend, outlet leg)", "internal", "Fluid",
        {"D": {"default": 10.0, "min": 2.0, "max": 200.0, "unit": "mm", "description": "pipe diameter"},
         "Rc": {"default": 40.0, "min": 10.0, "max": 400.0, "unit": "mm", "description": "bend radius (centre line)"},
         "angle": {"default": 180.0, "min": 15.0, "max": 180.0, "unit": "deg", "description": "bend angle"},
         "L_in": {"default": 150.0, "min": 10.0, "max": 2000.0, "unit": "mm", "description": "inlet leg length"},
         "L_out": {"default": 30.0, "min": 5.0, "max": 2000.0, "unit": "mm", "description": "outlet leg length"}},
        {"inlet": "plane and abs(cx + L_in) < 1e-3 * D and abs(cy) < 1e-3 * D", "outlet": "plane",
         "walls": "rest"}, _fake_pipe_bend),
    "heated_pipe": Model(
        "heated_pipe", "Straight pipe with an unheated entry and a heated section", "internal", "Fluid",
        {"D": {"default": 10.0, "min": 2.0, "max": 200.0, "unit": "mm", "description": "pipe diameter"},
         "L_entry": {"default": 60.0, "min": 5.0, "max": 2000.0, "unit": "mm",
                     "description": "unheated entry length (hydrodynamic development)"},
         "L_heated": {"default": 100.0, "min": 5.0, "max": 2000.0, "unit": "mm", "description": "heated length"}},
        {"inlet": "plane and cx < 1e-3 * D", "outlet": "plane",
         "wall_heated": {"where": "cylinder and cx > L_entry", "kind": "wall"},
         "wall_entry": {"where": "rest", "kind": "wall"}}, _fake_heated_pipe),
    "spheroid": Model(
        "spheroid", "Spheroid in a free stream (L = D: sphere)", "external", "Body",
        {"D": {"default": 10.0, "min": 0.1, "max": 5000.0, "unit": "mm", "description": "diameter across the flow"},
         "L": {"default": 10.0, "min": 0.1, "max": 20000.0, "unit": "mm", "description": "length along the flow"}},
        {"body": "rest"}, _fake_spheroid),
}


def build_document(template: str, output: str | Path, parameters: dict[str, float] | None = None,
                   backend: str = "auto") -> dict[str, Any]:
    """Write the parametric document of ``template`` (defaults overridden by ``parameters``) with FreeCAD, or a
    fake document with :class:`~.cad.FakeFreeCAD` (``backend="fake"``, or ``"auto"`` without FreeCAD)."""
    from .cad import get_backend
    m = MODELS[template]
    values = {**m.defaults(), **{k: float(v) for k, v in (parameters or {}).items()}}
    b = get_backend(backend)
    out = Path(output)
    if b.name == "fake" and out.suffix.lower() == ".fcstd":
        out = out.with_suffix(".fake.json")
    return {**b.build(template, values, out), "backend": b.name}
