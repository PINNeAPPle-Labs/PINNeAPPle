"""Parametric CAD variation from an existing trusted CadQuery solid.

CadQuery is optional. Input is a trusted in-process CAD object; no arbitrary
Python expressions, remote STEP loading or user-supplied code are evaluated.
Every variant must be re-meshed and re-solved before physical field comparison.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping


def parameter_sweep(builder: Callable[..., object], variants: Mapping[str, Mapping[str, float]],
                    *, max_variants: int = 32) -> dict[str, object]:
    if not callable(builder) or not 1 <= len(variants) <= max_variants:
        raise ValueError("invalid CAD builder or variant count")
    result = {}
    for name, parameters in variants.items():
        if not isinstance(name, str) or not name or len(name) > 80:
            raise ValueError("invalid CAD variant name")
        if len(parameters) > 32 or any(
            not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v)
            for v in parameters.values()
        ):
            raise ValueError("invalid CAD parameters")
        # Caller controls builder, and owns any geometric validity checks.
        solid = builder(**dict(parameters))
        if solid is None:
            raise ValueError(f"CAD builder returned no solid for {name}")
        result[name] = solid
    return result


def cadquery_mesh_artifact(shape, *, linear_deflection: float = 0.1,
                           angular_deflection: float = 0.5, label: str = "") -> dict:
    """Triangulate a trusted CadQuery Workplane/Shape into dashboard geometry."""
    from .scientific_artifacts import surface_artifact
    if not (0 < linear_deflection <= 10 and 0 < angular_deflection <= 3.14):
        raise ValueError("invalid tessellation parameters")
    solid = shape.val() if hasattr(shape, "val") else shape
    vertices, faces = solid.tessellate(linear_deflection, angular_deflection)
    return surface_artifact([tuple(float(x) for x in v.toTuple()) for v in vertices],
                            [tuple(int(i) for i in face) for face in faces],
                            label=label)
