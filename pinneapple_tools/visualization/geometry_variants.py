"""Deterministic geometry variants from an existing measured reference mesh.

Supports affine translation and scaling only; this is not a CAD kernel.
No field values are changed or claimed to be valid for the new geometry.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Mapping


def affine_variant(artifact: dict, *, scale=(1.0, 1.0, 1.0),
                   translation=(0.0, 0.0, 0.0), label: str = "") -> dict:
    if len(scale) != 3 or len(translation) != 3:
        raise ValueError("expected 3D affine parameters")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (*scale, *translation)):
        raise ValueError("non-finite geometry parameter")
    if any(v <= 0 for v in scale):
        raise ValueError("scale must be positive")
    if artifact.get("scalars"):
        raise ValueError("existing physical fields cannot be reused after geometry changes")
    points = artifact.get("vertices", [])
    if not points or len(points) > 12000:
        raise ValueError("invalid reference mesh")
    result = deepcopy(artifact)
    result["vertices"] = [[float(p[i]) * scale[i] + translation[i] for i in range(3)]
                          for p in points]
    result["label"] = label[:120]
    result["kind"] = "mesh"
    result["scalars"] = []
    result["field_name"] = None
    result["field_unit"] = None
    return result
