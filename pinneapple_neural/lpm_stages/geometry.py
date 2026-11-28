"""Stage 2: the body comes from a mesh somebody brought, and the validity gate decides which ones may be solved.

Every family in :mod:`pinneapple_neural.lpm_families` generates its geometry from a closed form, which is what makes
its field exact and also what limits it: a client's part is not a formula. :mod:`pinneapple_design.geometry.morph`
already deforms **any** triangle surface by FFD while keeping the topology, and its ``check_surface`` refuses the
deformations a solver could not represent. This module puts the two together: a
:class:`~pinneapple_neural.lpm_pipeline.FieldFamily` whose design space is a ``morph.DesignSpace`` over a mesh the
caller supplied, whose ``validate`` is the morph gate, and whose points and normals come from the morphed surface.

What it does not do: it does not solve the field. The caller brings that too — a solver run, a closed form, an
interpolated table — as one function of ``(spec, points, normals, surface)``. A design space is a *shape* space plus a
*reference*, and this stage supplies the first half for an arbitrary body while keeping the second half explicit,
because a family whose field came from nowhere would make the whole held-out number meaningless.

The validity gate is also available on its own (:func:`validity_gate`): the share of the declared box that survives the
gate is a property of the space, not of the fit, and a family that accepts one draw in twenty is a family whose
sampler will spend its budget re-drawing.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..lpm_pipeline import DesignSpec, FieldFamily, Reference

__all__ = ["MorphFamily", "morph_family", "validity_gate"]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit normal per vertex: each face's cross product shared out among its three vertices.

    Area-weighted rather than a plain average of face directions, because the faces around a vertex on a
    non-uniform mesh are not equal evidence about where "outward" is.
    """
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face = np.cross(v1 - v0, v2 - v0)                      # 2 * area * normal
    out = np.zeros_like(vertices, dtype=np.float64)
    for k in range(3):
        np.add.at(out, faces[:, k], face)
    return _unit(out)


class MorphFamily(FieldFamily):
    """A design space over a supplied surface: FFD parameters in, a gated body out, and the caller's field on it.

    ``space`` is a :class:`pinneapple_design.geometry.morph.DesignSpace`; every one of its parameters becomes a
    geometry parameter of this family, in the order the space names them, so a card can say which control point moved
    and how far. The points the model sees are the surface's own **vertices**, subsampled evenly to ``n_points``: no
    resampling and no interpolation, so two designs with the same parameters give the same points, which is what makes
    the dataset reproducible.
    """

    dim = 3

    def __init__(self, space: Any, field: Callable[..., np.ndarray], *, name: str, title: str, reference: Reference,
                 n_points: int = 256, tolerance: float = 0.05, field_names: Sequence[str] = ("cp",),
                 field_units: Sequence[str] = ("-",), field_scale: Sequence[float] = (1.0,), domain: str = "",
                 characteristic_length_m: float = 1.0) -> None:
        self.space = space
        self._field = field
        self.name = name
        self.title = title
        self.domain = domain
        self.reference = reference
        self.n_points = int(n_points)
        self.tolerance = float(tolerance)
        self.field_names = tuple(field_names)
        self.field_units = tuple(field_units)
        self.field_scale = tuple(field_scale)
        self.characteristic_length_m = float(characteristic_length_m)
        lo, hi = space.bounds
        self.geometry_params = {str(name_): (float(lo[i]), float(hi[i])) for i, name_ in enumerate(space.names)}

    # ---------------------------------------------------------------------------------- the geometry
    def _vector(self, geometry: dict[str, float]) -> np.ndarray:
        return np.asarray([float(geometry[name]) for name in self.space.names], dtype=np.float64)

    def sample_geometry(self, rng: np.random.Generator) -> dict[str, float]:
        return {name: float(rng.uniform(*bounds)) for name, bounds in self.geometry_params.items()}

    def geometry_from_unit(self, geometry: dict[str, float]) -> dict[str, float]:
        """Clip a sampler's draw into the declared box: a point outside it is not a member of the space."""
        out = {}
        for name, value in geometry.items():
            lo, hi = self.geometry_params[name]
            out[name] = float(np.clip(value, lo, hi))
        return out

    def validate(self, spec: DesignSpec) -> str:
        """The morph gate's verdict, as the family's own: ``""`` when the body may be solved."""
        report = self.space.check(self._vector(spec.geometry))
        if report.ok:
            return ""
        reasons = "; ".join(str(r) for r in getattr(report, "reasons", []) or [])
        return f"the morph validity gate refused the body: {reasons or 'no reason given'}"

    def _sampled(self, spec: DesignSpec) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        """The built surface, the sampled points and normals, and the indices those points came from."""
        surface = self.space.build(self._vector(spec.geometry))
        vertices, faces = surface.as_tuple()
        keep = np.unique(np.linspace(0, len(vertices) - 1, min(self.n_points, len(vertices))).round().astype(int))
        normals = vertex_normals(vertices, faces)
        return surface, np.asarray(vertices[keep], dtype=np.float64), np.asarray(normals[keep], dtype=np.float64), keep

    def points_and_normals(self, spec: DesignSpec) -> tuple[np.ndarray, np.ndarray]:
        _surface, points, normals, _keep = self._sampled(spec)
        return points, normals

    def field(self, spec: DesignSpec) -> np.ndarray:
        """The caller's reference, evaluated at exactly the points and normals the model will see."""
        surface, points, normals, keep = self._sampled(spec)
        values = np.asarray(self._field(spec, points, normals, surface), dtype=np.float64)
        if values.shape != (keep.size, len(self.field_names)):
            raise ValueError(f"{self.name}: the field returned {values.shape}, expected "
                             f"{(keep.size, len(self.field_names))} — one row per sampled point")
        return values

    def describe(self) -> dict[str, Any]:
        described = super().describe()
        described["geometry_source"] = {
            "kind": "morph.DesignSpace over a supplied surface",
            "n_variables": int(self.space.dim),
            "n_faces": int(len(self.space.faces)),
            "keep_out_boxes": len(getattr(self.space, "keep_out", []) or []),
            "gate": dict(getattr(self.space, "gate", {}) or {}),
            "note": "the points are the surface's own vertices, subsampled evenly; nothing is resampled",
        }
        return described


def morph_family(space: Any, field: Callable[..., np.ndarray], *, name: str, title: str,
                 reference: Reference, **kwargs: Any) -> MorphFamily:
    """A family over an existing ``morph.DesignSpace``. ``field`` is ``f(spec, points, normals, surface)``."""
    return MorphFamily(space, field, name=name, title=title, reference=reference, **kwargs)


def validity_gate(family: MorphFamily, n: int = 16, *, seed: int = 0, method: str = "lhs") -> dict[str, Any]:
    """The ``validity`` gate: how much of the declared box the morph gate lets through, measured.

    Not a pass/fail on the model — on the **space**. A design space that accepts a quarter of its draws costs the
    campaign four times what its card says, and a space that accepts none of them is a bounds declaration that
    disagrees with the geometry, which no amount of training fixes.
    """
    draws = family.space.sample(int(n), seed=seed, method=method)
    accepted, reasons = 0, {}
    for x in draws:
        report = family.space.check(x)
        if report.ok:
            accepted += 1
        else:
            for reason in getattr(report, "reasons", []) or []:
                key = str(reason)[:120]
                reasons[key] = reasons.get(key, 0) + 1
    share = accepted / max(len(draws), 1)
    top = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return {"check": "validity", "status": "passed" if share >= 0.5 else "failed", "value": share,
            "tolerance": 0.5,
            "detail": f"{accepted}/{len(draws)} draws survive the morph gate; the most common refusals are "
                      f"{[f'{k} ({v})' for k, v in top] or 'none'}",
            "rows": [{"reason": k, "count": v} for k, v in top]}
