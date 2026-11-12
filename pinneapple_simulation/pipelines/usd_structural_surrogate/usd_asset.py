"""Fetch an NVIDIA USD asset and read the mounting interface out of its mesh.

The account of what NVIDIA's public USD library actually contains, and why this module reads a *mount* rather than
a benchmark part, is in the package docstring. What matters here: the numbers the design family is anchored to are
**measured from the asset**, never typed in, and the asset's identity (URL and sha256) travels with them.

Measured from ``Props/Mounts/ur10_mount.usd``:

* ``plate_diameter`` -- twice the largest radius of the round base plate, from the vertices on its bottom face;
* ``plate_thickness`` -- the spacing of the two most populated z levels of that plate;
* ``height``, ``extent`` -- the bounding box.

Units: the asset declares ``metersPerUnit = 1.0`` and is modelled at ~270 units across, which is 270 mm for a
robot base plate and 270 m for nothing. The numbers are read as millimetres, and ``units_inferred`` records that
the declaration was not taken at face value.
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["NVIDIA_UR10_MOUNT", "UsdMesh", "fetch_asset", "read_mesh", "mount_interface", "RECORDED_INTERFACE"]

# NVIDIA Isaac Sim / Omniverse content bucket. Public, anonymous, no key.
_BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
NVIDIA_UR10_MOUNT = f"{_BUCKET}/Props/Mounts/ur10_mount.usd"

# What mount_interface() measures on that asset, kept so the family can be built offline and so a change in the
# asset (or in the measurement) is visible as a diff rather than as a silent shift of every design.
RECORDED_INTERFACE: dict[str, Any] = {
    "plate_diameter_mm": 271.9,
    "plate_thickness_mm": 6.0,
    "height_mm": 515.0,
    "up_axis": "Z",
    "source": NVIDIA_UR10_MOUNT,
}


@dataclass
class UsdMesh:
    """The triangle mesh of a USD stage, flattened, with the stage's identity and units."""
    points: np.ndarray                 # (n, 3) float64, in the asset's own units
    face_counts: np.ndarray            # (m,) int, vertices per face (triangles and n-gons as stored)
    face_indices: np.ndarray           # (sum(face_counts),) int, flat indices into points
    meters_per_unit: float
    up_axis: str
    path: str
    sha256: str = ""

    @property
    def triangles(self) -> np.ndarray:
        """(t, 3) corner indices, fan-triangulating any n-gon (USD meshes are convex per face)."""
        out = []
        start = 0
        for c in self.face_counts:
            if c == 3:
                out.append(self.face_indices[start:start + 3])
            elif c > 3:
                fan = [(0, k, k + 1) for k in range(1, c - 1)]
                idx = self.face_indices[start:start + c]
                out.extend(idx[list(f)] for f in fan)
            start += int(c)
        return np.asarray(out, int) if out else np.zeros((0, 3), int)

    def info(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "points": int(len(self.points)),
                "faces": int(len(self.face_counts)), "triangles": int(len(self.triangles)),
                "meters_per_unit": self.meters_per_unit, "up_axis": self.up_axis}


def fetch_asset(url: str = NVIDIA_UR10_MOUNT, dest: str | Path = "assets/ur10_mount.usd",
                timeout: float = 120.0) -> Path:
    """Download ``url`` to ``dest`` once (a cached copy is reused), returning the path. No credentials: the Isaac
    content bucket is served anonymously."""
    p = Path(dest)
    if p.exists() and p.stat().st_size > 0:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(p)
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_mesh(path: str | Path) -> UsdMesh:
    """Flatten every ``UsdGeom.Mesh`` of the stage into one triangle soup. Needs ``usd-core`` (``pip install
    usd-core``), whose ``pxr`` module is the reference OpenUSD reader."""
    try:
        from pxr import Usd, UsdGeom
    except ImportError as e:                                        # pragma: no cover - exercised by the guide
        raise ImportError("reading a USD stage needs OpenUSD: pip install usd-core") from e
    p = Path(path)
    stage = Usd.Stage.Open(str(p))
    if stage is None:
        raise ValueError(f"not a readable USD stage: {p}")
    chunks: list[np.ndarray] = []
    counts: list[int] = []
    indices: list[int] = []
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        cnt = m.GetFaceVertexCountsAttr().Get()
        idx = m.GetFaceVertexIndicesAttr().Get()
        if not pts or not cnt or not idx:
            continue
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        P = np.array([xf.Transform(pt) for pt in pts], float)
        chunks.append(P)
        counts.extend(int(c) for c in cnt)
        indices.extend(int(i) + offset for i in idx)
        offset += len(P)
    if not chunks:
        raise ValueError(f"no UsdGeom.Mesh with points and faces in {p}")
    points = np.concatenate(chunks)
    return UsdMesh(points=points, face_counts=np.asarray(counts, int), face_indices=np.asarray(indices, int),
                   meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0),
                   up_axis=str(UsdGeom.GetStageUpAxis(stage)), path=str(p), sha256=_sha256(p))


def mount_interface(mesh: UsdMesh) -> dict[str, Any]:
    """The mounting interface, measured from the mesh: plate diameter, plate thickness and overall height (mm).

    The base plate is the set of vertices sharing the two most populated z levels of the model (a flat plate has
    its vertices on its two faces, and has more of them than any other pair of levels in this asset). The diameter
    is twice the largest radius among those vertices, the thickness the gap between the levels.
    """
    P = np.asarray(mesh.points, float)
    if len(P) == 0:
        raise ValueError("empty mesh")
    z = np.round(P[:, 2], 3)
    levels, counts = np.unique(z, return_counts=True)
    top = levels[np.argsort(counts)[-2:]]                     # the two most populated levels
    lo, hi = float(top.min()), float(top.max())
    plate = P[np.isin(z, top)]
    radius = float(np.hypot(plate[:, 0], plate[:, 1]).max())
    extent = (P.max(0) - P.min(0)).tolist()
    return {"plate_diameter_mm": round(2.0 * radius, 2), "plate_thickness_mm": round(hi - lo, 3),
            "height_mm": round(float(extent[2]), 2), "extent_mm": [round(float(e), 2) for e in extent],
            "up_axis": mesh.up_axis, "meters_per_unit": mesh.meters_per_unit,
            "units_inferred": "mm" if mesh.meters_per_unit == 1.0 else "as declared",
            "source": mesh.path, "sha256": mesh.sha256, "points": int(len(P))}


def probe(url: str = NVIDIA_UR10_MOUNT, dest: str | Path = "assets/ur10_mount.usd") -> dict[str, Any]:
    """``fetch_asset`` + ``read_mesh`` + ``mount_interface`` in one call, for a manifest."""
    p = fetch_asset(url, dest)
    iface = mount_interface(read_mesh(p))
    iface["url"] = url
    return iface
