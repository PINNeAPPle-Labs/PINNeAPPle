"""The USD lane, native: what the twin writes is a stage Unity's USD Toolkit (and Isaac Sim) opens.

PINNeAPPle already writes OpenUSD text (``pinneapple_twin3d.usd``): every part a ``Mesh``, every field a per-vertex
``primvars:<name>`` whose values are **time samples**, ``UsdPhysics`` rigid bodies / colliders / materials, and the
provenance (title, source, units, the real times) in the layer metadata. This module is the Python side of that
hand-off:

* :func:`stage_summary` -- read a ``.usda`` **without any dependency** and report what an importer will see: the
  layer's up axis and units, the prim types, the apiSchemas (``PhysicsCollisionAPI``, ...), which attributes carry
  time samples, and which primvars carry a unit. It is the same information ``UsdImporter.GetPreviewInfo`` prints in
  Unity before committing to an import.
* :func:`validate` -- open the stage with Pixar's OpenUSD (``pip install usd-core``, the library the Unity USD
  Toolkit wraps) and list what does not check out: stages that do not open, meshes without points, time samples
  outside the layer's range, physics rels pointing at prims that are not there.
* :func:`from_usd` -- read a stage authored elsewhere (Unity, Isaac Sim, Omniverse) into a
  :class:`pinneapple_twin3d.Scene`, so the jobs can run on it and the result goes back out as USD. Needs ``pxr``.

The text path (``stage_summary``) is what the tests run everywhere; the ``pxr`` paths skip when it is not installed.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any


def _layer_block(text: str) -> str:
    """The header of an .usda: everything before the first prim."""
    i = text.find("def ")
    return text[:i if i > 0 else len(text)]


def stage_summary(path: str) -> dict[str, Any]:
    """What a Unity (or Isaac Sim) import of this stage will find, read from the text with no dependency.

    ``{"format", "up_axis", "meters_per_unit", "start", "end", "prims": {type: n}, "api_schemas": {...},
    "time_sampled": [attribute names], "primvars": [{"name", "unit", "prim"}], "metadata": {...},
    "references": n, "problems": [...]}``
    """
    with open(path) as f:
        text = f.read()
    if not text.startswith("#usda") and not text.startswith("#usd"):
        raise ValueError(f"{path}: not an OpenUSD file (no #usda or #usd header)")
    head = _layer_block(text)
    summary: dict[str, Any] = {"format": text.splitlines()[0].strip(), "problems": []}
    fields = {"upAxis": ("up_axis", str), "metersPerUnit": ("meters_per_unit", float),
              "kilogramsPerUnit": ("kilograms_per_unit", float), "startTimeCode": ("start_time_code", float),
              "endTimeCode": ("end_time_code", float), "timeCodesPerSecond": ("time_codes_per_second", float)}
    for key, (name, cast) in fields.items():
        m = re.search(rf"^\s*{key}\s*=\s*\"?([^\n\"]+)\"?", head, re.M)
        if m:
            try:
                summary[name] = cast(m.group(1).strip())
            except ValueError:
                summary["problems"].append(f"layer metadata {key} is not a number: {m.group(1)!r}")
    prims: dict[str, int] = {}
    for kind in re.findall(r"^\s*def (\w+) ", text, re.M):
        prims[kind] = prims.get(kind, 0) + 1
    summary["prims"] = prims
    schemas: dict[str, int] = {}
    for s in re.findall(r'"(\w+API)"', text):
        schemas[s] = schemas.get(s, 0) + 1
    for s in ("PhysicsScene",):
        n = len(re.findall(rf"def {s} ", text))
        if n:
            schemas[s] = n
    summary["api_schemas"] = schemas
    summary["time_sampled"] = sorted(set(re.findall(r"^\s*[\w\[\]:\.]+ (\w[\w:]*?)\.timeSamples", text, re.M)))
    # primvars, and the unit each one declares in its customData (a few lines below the attribute line)
    primvars = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\s+\w+\[\] (primvars:\w+)", line)
        if not m:
            continue
        unit = ""
        for follower in lines[i + 1:i + 6]:
            u = re.search(r'string unit = "([^"]*)"', follower)
            if u:
                unit = u.group(1)
                break
        known = next((p for p in primvars if p["name"] == m.group(1)), None)
        if known is None:
            primvars.append({"name": m.group(1), "unit": unit})
        elif unit and not known["unit"]:
            known["unit"] = unit                          # the timeSamples block repeats the name without a unit
    summary["primvars"] = primvars
    meta = {}
    for key in ("pinneapple_format", "title", "source", "length_unit", "time_unit"):
        m = re.search(rf'string {key} = "([^"]*)"', head)
        if m:
            meta[key] = m.group(1)
    summary["metadata"] = meta
    summary["references"] = len(re.findall(r"@[^@\n]+@", text))
    if "def Mesh" not in text:
        summary["problems"].append("no Mesh prim: an importer would build an empty scene")
    if text.count("point3f[] points") != prims.get("Mesh", 0):
        summary["problems"].append("some Mesh prims have no points array")
    if summary.get("end_time_code", 0) < summary.get("start_time_code", 0):
        summary["problems"].append("endTimeCode before startTimeCode")
    return summary


def validate(path: str) -> list[str]:
    """Open the stage with Pixar's OpenUSD and list what does not check out (``pip install usd-core``).

    Raises ``ImportError`` when ``pxr`` is missing, so a caller can tell "not installed" from "invalid".
    """
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: F401  (UsdPhysics registers the schemas)

    stage = Usd.Stage.Open(path)
    if stage is None:
        return [f"{path}: the stage does not open"]
    problems: list[str] = []
    root = stage.GetDefaultPrim()
    if not root:
        problems.append("no defaultPrim (an importer has to guess the root)")
    start, end = stage.GetStartTimeCode(), stage.GetEndTimeCode()
    n_mesh = n_tris = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr()
            counts = mesh.GetFaceVertexCountsAttr()
            if not points.Get() or not (counts.Get() and sum(counts.Get()) % 3 == 0):
                problems.append(f"{prim.GetPath()}: mesh without points or with non-triangular topology")
                continue
            n_mesh += 1
            n_tris += len(counts.Get())
        for attr in prim.GetAttributes():
            if "primvars:" in attr.GetName():
                times = attr.GetTimeSamples()
                if times and (min(times) < start or max(times) > end):
                    problems.append(f"{attr.GetPath()}: time samples {min(times)}..{max(times)} outside "
                                    f"the layer range {start}..{end}")
    if n_mesh == 0:
        problems.append("no UsdGeom.Mesh in the stage")
    # physics rels must point at prims that exist
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            if "physics" in rel.GetName():
                for target in rel.GetTargets():
                    if not stage.GetPrimAtPath(target):
                        problems.append(f"{rel.GetPath()}: target {target} does not exist")
    if stage.GetPrimAtPath("/Twin/physicsScene") and not stage.GetPrimAtPath("/Twin/physicsScene").IsA(
            UsdPhysics.Scene):
        problems.append("/Twin/physicsScene is not a PhysicsScene")
    return problems


def inspect(path: str) -> dict[str, Any]:
    """``stage_summary`` plus what OpenUSD itself reports, when ``pxr`` is available."""
    out = stage_summary(path)
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(path)
        if stage is not None:
            out["openusd"] = {
                "version": str(Usd.GetVersion()),
                "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
                "prims": sum(1 for _ in stage.Traverse()),
                "time_range": [stage.GetStartTimeCode(), stage.GetEndTimeCode()],
                "problems": validate(path),
            }
    except ImportError:
        out["openusd"] = None
    return out


def from_usd(path: str, *, times: list[float] | None = None):
    """Read a stage authored elsewhere into a :class:`pinneapple_twin3d.Scene`.

    Every ``UsdGeom.Mesh`` becomes a part (its ``primvars:*`` with one value per point become fields), the stage's
    time samples become the scene's time steps, ``UsdPhysics`` colliders/materials come along, and the layer's up
    axis is honoured (a Z-up stage, which is what Isaac Sim writes, needs no change; a Y-up one is turned). Needs
    ``pxr`` (``pip install usd-core``).
    """
    import numpy as np
    from pxr import Usd, UsdGeom

    from pinneapple_twin3d import Scene

    stage = Usd.Stage.Open(path)
    if stage is None:
        raise ValueError(f"{path}: the stage does not open")
    up = UsdGeom.GetStageUpAxis(stage)
    units = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    start = stage.GetStartTimeCode()
    time_codes = sorted({t for prim in stage.Traverse() for attr in prim.GetAttributes()
                         for t in attr.GetTimeSamples()})
    steps = times if times is not None else (time_codes or [start])
    scene = Scene(os.path.basename(path), length_unit="m", times=np.asarray(steps, float) if len(steps) > 1 else None,
                  time_unit="s", source=f"OpenUSD: {os.path.basename(path)}")

    def to_zup(points: np.ndarray) -> np.ndarray:
        if up == UsdGeom.Tokens.y:
            return np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)
        return points

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts:
            continue
        v = to_zup(np.asarray(points, float) * units)
        tri = np.asarray(indices, int).reshape(-1, 3)
        part = scene.add_part(prim.GetName(), v, tri, group=str(prim.GetParent().GetName()))
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if not name.startswith("primvars:") or len(attr.GetTimeSamples()) == 0:
                continue
            field = name.split(":", 1)[1]
            values = []
            for t in steps:
                got = attr.Get(t if len(steps) > 1 else Usd.TimeCode.Default())
                if got is None:
                    continue
                arr = np.asarray(got, float)
                if arr.ndim == 1 and arr.size == len(v):
                    values.append(arr)
            if values:
                scene.add_field(part.name, field, np.asarray(values, float),
                                unit=_primvar_unit(attr))
        if prim.HasAPI("PhysicsCollisionAPI") or prim.HasAPI("PhysicsRigidBodyAPI"):
            scene.add_physics(part.name, rigid_body=prim.HasAPI("PhysicsRigidBodyAPI"),
                              collider=prim.HasAPI("PhysicsCollisionAPI"),
                              approximation=str(mesh.GetPrim().GetAttribute("physics:approximation").Get()
                                                or "convexHull"))
    for prim in stage.Traverse():
        if prim.IsA("PhysicsScene"):
            g = prim.GetAttribute("physics:gravityDirection").Get()
            m = prim.GetAttribute("physics:gravityMagnitude").Get()
            if g is not None and m is not None:
                vec = np.asarray(g, float) * float(m)
                scene.set_gravity(vec)
            break
    return scene


def _primvar_unit(attr) -> str:
    """The ``unit`` an exported primvar declares in its customData, or ""."""
    try:
        data = attr.GetCustomData() or {}
        if "unit" in data:
            return str(data["unit"])
    except Exception:  # noqa: BLE001 - a missing customData is not an error
        pass
    return ""


def manifest_entry() -> dict[str, Any]:
    """The ``Packages/manifest.json`` line that installs Unity's USD Toolkit (Unity 6.4+)."""
    return {"com.unity.usd-toolkit": "https://github.com/Unity-Technologies/com.unity.usd-toolkit.git#v0.8.1-exp.1"}


def write_manifest(project: str) -> str:
    """Add the USD Toolkit to a Unity project's ``Packages/manifest.json`` (idempotent); returns the path."""
    path = os.path.join(project, "Packages", "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: not a Unity project (no Packages/manifest.json)")
    with open(path) as f:
        man = json.load(f)
    man.setdefault("dependencies", {}).update(manifest_entry())
    with open(path, "w") as f:
        json.dump(man, f, indent=2)
        f.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    """``python -m pinneapple_unity.usd``: inspect a stage, or install the Unity USD Toolkit in a project."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m pinneapple_unity.usd",
                                 description="OpenUSD helpers for the Unity SDK: inspect a stage, "
                                             "or add Unity's USD Toolkit to a project")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="what a Unity / Isaac Sim import of this stage will find")
    s.add_argument("stage")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("validate", help="open the stage with usd-core and list what is wrong")
    s.add_argument("stage")
    s = sub.add_parser("manifest", help="add com.unity.usd-toolkit to <project>/Packages/manifest.json")
    s.add_argument("project")
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        out = inspect(a.stage) if a.json else stage_summary(a.stage)
        print(json.dumps(out, indent=1, default=str) if a.json else _print_summary(out))
        return 0
    if a.cmd == "validate":
        try:
            problems = validate(a.stage)
        except ImportError:
            print("validate needs Pixar's OpenUSD: pip install usd-core")
            return 2
        for problem in problems:
            print(problem)
        print(f"{len(problems)} problem(s)" if problems else f"{a.stage}: the stage opens and checks out")
        return 1 if problems else 0
    print(write_manifest(a.project))
    return 0


def _print_summary(s: dict[str, Any]) -> str:
    lines = [f"{s['format']}   up axis {s.get('up_axis', '?')}   {s.get('meters_per_unit', '?')} m per unit   "
             f"frames {s.get('start_time_code', '?')}..{s.get('end_time_code', '?')}"]
    lines.append("prims: " + ", ".join(f"{k} x{v}" for k, v in sorted(s["prims"].items())))
    lines.append("apiSchemas: " + ", ".join(f"{k} x{v}" for k, v in sorted(s["api_schemas"].items())))
    lines.append("time-sampled: " + (", ".join(s["time_sampled"]) or "none"))
    lines.append("primvars: " + ", ".join(f"{p['name']}{' [' + p['unit'] + ']' if p['unit'] else ''}"
                                          for p in s["primvars"]))
    if s["metadata"]:
        lines.append("provenance: " + json.dumps(s["metadata"]))
    if s["problems"]:
        lines.append("problems: " + "; ".join(s["problems"]))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.exit(main())
