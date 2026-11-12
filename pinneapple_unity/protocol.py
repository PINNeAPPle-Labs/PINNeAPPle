"""Wire format shared by the Unity runtime, the exporter and the bridge (``pinneapple-unity/1``).

One protocol, three lanes:

* **file lane** -- a twin folder exported by :func:`pinneapple_unity.export_for_unity` and read by the Unity
  package (``scene.json`` + ``geometry.glb`` + ``fields.bin``, the layout ``pinneapple_twin3d`` already writes);
* **link lane** -- ``pinneapple_link.json``: what a scene authored in Unity *is* (parts, units, sensors, physics)
  and what it *asks* PINNeAPPle to compute (``jobs``);
* **live lane** -- HTTP/JSON between the Unity runtime and :mod:`pinneapple_unity.bridge`.

Axis convention: Unity is Y-up left-handed, :mod:`pinneapple_twin3d` is Z-up right-handed, glTF is Y-up. The
rotation between them is a -90 deg turn about X, ``(x, y, z) -> (x, z, -y)`` (the same one Blender's glTF exporter
uses), written as a quaternion in :data:`Z_UP_TO_Y_UP`.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROTOCOL = "pinneapple-unity/1"
LINK_FILE = "pinneapple_link.json"          # next to the glTF a Unity scene exports
UNITY_FILE = "pinneapple_unity.json"        # next to scene.json in an exported twin
DATA_DIR = "Assets/TwinData"                # where an exported twin lands inside a Unity project
PACKAGE_DIR = "Assets/PinNeAPPleTwin"       # where the C# runtime lands

#: -90 deg about X as a quaternion ``(w, x, y, z)``: Z-up geometry into a Y-up world.
Z_UP_TO_Y_UP = (0.7071067811865476, -0.7071067811865476, 0.0, 0.0)

_AXES = ("x", "y", "z")


def z_up_to_y_up(vertices) -> np.ndarray:
    """``(x, y, z) -> (x, z, -y)``: Z-up points in a Y-up frame (Unity, glTF)."""
    v = np.asarray(vertices, float).reshape(-1, 3)
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)


def y_up_to_z_up(vertices) -> np.ndarray:
    """``(x, y, z) -> (x, -z, y)``: the inverse of :func:`z_up_to_y_up`."""
    v = np.asarray(vertices, float).reshape(-1, 3)
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)


def axis_index(axis: str) -> int:
    a = str(axis).strip().lower()
    if a not in _AXES:
        raise ValueError(f"axis must be one of {_AXES}, not {axis!r}")
    return _AXES.index(a)


def local_coordinate(vertices, axis: str = "auto", length: float | None = None,
                     flip: bool = False) -> tuple[np.ndarray, float]:
    """Each vertex's position ``u`` along a part's own axis, plus the length used.

    ``axis="auto"`` takes the axis of the part's bounding box with the longest extent -- the direction a beam, a
    fin or a slab is laid out along. ``u`` runs from 0 at the low end (the root, the hot wall) to ``length``;
    ``flip=True`` measures from the high end instead, for a part whose root or hot wall sits there.
    """
    v = np.asarray(vertices, float).reshape(-1, 3)
    if len(v) == 0:
        raise ValueError("a part needs vertices")
    if str(axis).lower() in ("auto", ""):
        k = int(np.argmax(np.ptp(v, axis=0)))
    else:
        k = axis_index(axis)
    span = float(np.ptp(v[:, k])) or 1.0
    L = float(length) if length else span
    u = (v[:, k] - v[:, k].min()) / span * L
    if flip:
        u = L - u
    return u, L


# ------------------------------------------------------------------------------------ link (Unity-authored scene)
@dataclass
class LinkPart:
    name: str
    node: str = ""
    group: str = ""
    color: tuple[float, float, float] = (0.75, 0.77, 0.8)
    axis: str = "auto"
    length: float | None = None


@dataclass
class LinkJob:
    job: str
    part: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Link:
    """``pinneapple_link.json``: a Unity scene's parts and the jobs PINNeAPPle should run on them."""

    title: str = "Unity scene"
    geometry: str = "scene.glb"
    length_unit: str = "m"
    time_unit: str = "s"
    up_axis: str = "y"
    parts: list[LinkPart] = field(default_factory=list)
    jobs: list[LinkJob] = field(default_factory=list)
    sensors: list[dict[str, Any]] = field(default_factory=list)
    physics: dict[str, Any] = field(default_factory=dict)
    times: list[float] = field(default_factory=list)
    protocol: str = PROTOCOL
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Link:
        if not isinstance(d, dict):
            raise ValueError("pinneapple_link.json must be a JSON object")
        proto = d.get("protocol", PROTOCOL)
        if proto != PROTOCOL:
            raise ValueError(f"link protocol {proto!r} is not {PROTOCOL!r}")
        parts = [LinkPart(str(p["name"]), str(p.get("node", "")), str(p.get("group", "")),
                          tuple(float(c) for c in p.get("color", (0.75, 0.77, 0.8))),
                          str(p.get("axis", "auto")),
                          None if p.get("length") is None else float(p["length"]))
                 for p in d.get("parts", [])]
        jobs = [LinkJob(str(j["job"]), str(j.get("part", parts[0].name if parts else "")), dict(j.get("params", {})))
                for j in d.get("jobs", [])]
        return cls(title=str(d.get("title", "Unity scene")), geometry=str(d.get("geometry", "scene.glb")),
                   length_unit=str(d.get("length_unit", "m")), time_unit=str(d.get("time_unit", "s")),
                   up_axis=str(d.get("up_axis", "y")).lower(), parts=parts, jobs=jobs,
                   sensors=list(d.get("sensors", [])), physics=dict(d.get("physics", {})),
                   times=[float(t) for t in d.get("times", [])], protocol=proto, raw=dict(d))

    @classmethod
    def load(cls, folder: str) -> Link:
        path = folder if folder.endswith(".json") else os.path.join(folder, LINK_FILE)
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol, "title": self.title, "geometry": self.geometry,
            "length_unit": self.length_unit, "time_unit": self.time_unit, "up_axis": self.up_axis,
            "times": list(self.times),
            "parts": [{"name": p.name, "node": p.node, "group": p.group, "color": list(p.color),
                       "axis": p.axis, "length": p.length} for p in self.parts],
            "sensors": list(self.sensors),
            "jobs": [{"job": j.job, "part": j.part, "params": j.params} for j in self.jobs],
            "physics": dict(self.physics),
        }

    def save(self, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, LINK_FILE)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
            f.write("\n")
        return path

    def part(self, name: str) -> LinkPart:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(f"no part {name!r} in the link (have {[p.name for p in self.parts]})")


# -------------------------------------------------------------------------------------------- field payloads
def encode_field(values, encoding: str = "base64", *, unit: str = "") -> dict[str, Any]:
    """One field over the wire: ``(T, V)`` float32, base64 (default, compact) or plain JSON numbers."""
    v = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    if v.ndim != 2:
        raise ValueError("a field payload is (steps, count)")
    out: dict[str, Any] = {"steps": int(v.shape[0]), "count": int(v.shape[1]), "unit": unit}
    if encoding == "json":
        out.update(encoding="json", values=v.tolist())
    elif encoding == "base64":
        out.update(encoding="base64", dtype="f4", data=base64.b64encode(v.tobytes()).decode("ascii"))
    else:
        raise ValueError(f"encoding must be 'base64' or 'json', not {encoding!r}")
    return out


def decode_field(payload: dict[str, Any]) -> np.ndarray:
    """The ``(T, V)`` float32 array of an :func:`encode_field` payload."""
    steps, count = int(payload["steps"]), int(payload["count"])
    if payload.get("encoding", "base64") == "json":
        return np.asarray(payload["values"], dtype="<f4").reshape(steps, count)
    buf = base64.b64decode(payload["data"])
    return np.frombuffer(buf, dtype="<f4").reshape(steps, count)
