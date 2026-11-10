"""Export measured triangular surfaces for Physics Dashboard Studio.

No interpolation, unit conversion, or synthetic physical values is performed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


def surface_artifact(vertices: Iterable, triangles: Iterable, *,
                     scalars: Iterable | None = None, coordinate_unit: str = "m",
                     field_unit: str | None = None, field_name: str | None = None,
                     label: str = "") -> dict:
    points = [tuple(float(v) for v in p) for p in vertices]
    faces = [tuple(int(i) for i in face) for face in triangles]
    values = [] if scalars is None else [float(v) for v in scalars]
    if not 1 <= len(points) <= 12000 or not 1 <= len(faces) <= 24000:
        raise ValueError("surface size outside dashboard transport bounds")
    if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError("invalid vertex")
    if any(len(f) != 3 or any(i < 0 or i >= len(points) for i in f) for f in faces):
        raise ValueError("invalid triangle connectivity")
    if values and (len(values) != len(points) or not all(math.isfinite(v) for v in values)):
        raise ValueError("scalar values must match vertex count")
    return {"kind": "scalar_field" if values else "mesh",
            "vertices": points, "triangles": faces, "scalars": values,
            "coordinate_unit": coordinate_unit, "field_unit": field_unit,
            "field_name": field_name, "label": label}


def publish_surface(artifact: dict, *, endpoint: str, api_key: str,
                    source: str, resource_id: str, channel: str, timeout: float = 10) -> dict:
    """Opt-in HTTPS upload; endpoint is API origin, not an arbitrary artifact URL."""
    from urllib.parse import urlsplit
    from urllib.request import Request, urlopen
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("HTTPS API origin required")
    if not api_key:
        raise ValueError("API key required")
    payload = json.dumps({"source": source, "resource_id": resource_id,
                          "channel": channel, "artifact": artifact}, allow_nan=False).encode()
    if len(payload) > 1_500_000:
        raise ValueError("artifact too large for JSON transport")
    request = Request(endpoint.rstrip("/") + "/v1/dashboards/artifacts",data=payload,
                      headers={"Content-Type": "application/json", "X-API-Key": api_key},
                      method="PUT")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(8192))


def from_meshio(mesh, *, field_name: str | None = None, coordinate_unit: str = "m") -> dict:
    """Adapt a meshio Mesh with triangle cells and optional point_data field."""
    triangles = []
    for block in mesh.cells:
        if block.type == "triangle":
            triangles.extend(block.data.tolist())
    if not triangles:
        raise ValueError("meshio mesh has no triangular surface cells")
    values = None
    if field_name is not None:
        if field_name not in mesh.point_data:
            raise ValueError("unknown point field")
        field = mesh.point_data[field_name]
        if getattr(field, "ndim", 1) != 1:
            raise ValueError("only scalar point fields are supported")
        values = field.tolist()
    return surface_artifact(mesh.points[:, :3], triangles, scalars=values,
                            field_name=field_name, coordinate_unit=coordinate_unit)
