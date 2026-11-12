"""The Unity -> PINNeAPPle -> OpenUSD lane: read a scene a team authored in Unity, run the physics it asks for,
write the result back as an OpenUSD layer (and as a twin Unity can reopen).

    from pinneapple_unity import from_unity, apply_link
    scene, link = from_unity("out/gripper_unity")     # geometry.glb + pinneapple_link.json
    reports = apply_link(scene, link)                 # the jobs of the link, with their checks
    scene.export("out/gripper_twin", usd=True)        # scene.json + glb + fields + scene.usda

or in one call::

    from pinneapple_unity import unity_to_usd
    unity_to_usd("out/gripper_unity", "out/gripper_twin", usd=True)

What the link carries (``pinneapple_link.json``, protocol ``pinneapple-unity/1``): parts with their names and units,
the time steps the scene plays, the sensors, the ``UsdPhysics`` assignment, and the **jobs** -- the physics to run.
Geometry stays a glTF (Unity is Y-up); :func:`from_unity` rotates it into the Z-up frame the twin and USD use.

:func:`emulate_unity_export` writes the two files the Unity editor menu writes (a Y-up ``.glb`` plus the link), so
the whole lane can be tested and demonstrated on a machine without Unity installed.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .jobs import JobRequest, run_job
from .protocol import LINK_FILE, Link, LinkJob, LinkPart, y_up_to_z_up, z_up_to_y_up

#: Names of the parts a link may leave out: the geometry's own node names are used then.
_DEFAULT_COLOR = (0.75, 0.77, 0.8)


def _part_of(scene, name: str):
    """The scene's part ``name``."""
    for p in scene.parts:
        if p.name == name:
            return p
    raise KeyError(f"scene has no part {name!r} (have {[p.name for p in scene.parts]})")


def _match(link_parts: list[LinkPart], names: list[str]) -> dict[str, str]:
    """``link part name -> glTF node name``: by node, then by name, then by order, case-insensitively last."""
    out: dict[str, str] = {}
    taken: set[str] = set()
    for i, p in enumerate(link_parts):
        for cand in (p.node, p.name):
            if cand and cand in names and cand not in taken:
                out[p.name] = cand
                taken.add(cand)
                break
        else:
            lower = {n.lower(): n for n in names if n not in taken}
            hit = lower.get((p.node or p.name).lower())
            if hit:
                out[p.name] = hit
                taken.add(hit)
            elif i < len(names) and names[i] not in taken:
                out[p.name] = names[i]            # no names at all in the glTF: the export order is the link order
                taken.add(names[i])
    return out


def from_unity(folder: str, *, geometry: str | None = None) -> tuple[Any, Link]:
    """Read a Unity-exported folder as a :class:`pinneapple_twin3d.Scene` (Z-up) plus its :class:`Link`.

    Geometry comes from the link's glTF (node transforms applied by the glTF reader); the link supplies the part
    names, colours, groups, sensors, time steps and ``UsdPhysics`` assignment. A part in the geometry that the link
    does not name is still imported, under its node name.
    """
    from pinneapple_tools.visualization.studio.scene import read_gltf
    from pinneapple_twin3d import Scene
    from pinneapple_twin3d.scene import PhysicsMaterial

    link = Link.load(folder)
    glb = geometry or os.path.join(folder, link.geometry)
    surfaces = read_gltf(glb, axes="as_is")
    if not surfaces:
        raise ValueError(f"{glb}: no triangle geometry")
    times = np.asarray(link.times, float) if link.times else None
    scene = Scene(link.title, length_unit=link.length_unit, times=times, time_unit=link.time_unit,
                  source=f"Unity: {os.path.basename(os.path.abspath(folder))}")
    mapped = _match(link.parts, [s.name for s in surfaces])
    by_node = {s.name: s for s in surfaces}
    for part in link.parts:
        node = mapped.get(part.name)
        if node is None:
            raise KeyError(f"part {part.name!r} of the link has no geometry in {os.path.basename(glb)} "
                           f"(nodes: {[s.name for s in surfaces]})")
        s = by_node[node]
        v = y_up_to_z_up(s.vertices) if link.up_axis == "y" else s.vertices
        scene.add_part(part.name, v, s.faces, group=part.group, color=part.color)
    named = set(mapped.values())
    for s in surfaces:
        if s.name not in named:
            v = y_up_to_z_up(s.vertices) if link.up_axis == "y" else s.vertices
            scene.add_part(s.name, v, s.faces, group="geometry")

    for s in link.sensors:
        pos = np.asarray(s.get("position", (0.0, 0.0, 0.0)), float).reshape(3)
        if link.up_axis == "y":
            pos = y_up_to_z_up(pos)[0]
        scene.add_sensor(str(s.get("id", "sensor")), pos, label=str(s.get("label", "")),
                         unit=str(s.get("unit", "")), quantity=str(s.get("quantity", "")),
                         series=s.get("series"), envelope=s.get("envelope"))

    physics = link.physics or {}
    if physics.get("gravity") is not None:
        g = np.asarray(physics["gravity"], float).reshape(3)
        scene.set_gravity(y_up_to_z_up(g)[0] if link.up_axis == "y" else g)
    for name, spec in dict(physics.get("parts", {})).items():
        if name not in [p.name for p in scene.parts]:
            continue
        mat = spec.get("material")
        material = None
        if mat:
            material = PhysicsMaterial(str(mat.get("name", f"{name}_contact")),
                                       float(mat.get("static_friction", 0.5)),
                                       float(mat.get("dynamic_friction", 0.5)),
                                       float(mat.get("restitution", 0.0)),
                                       None if mat.get("density") is None else float(mat["density"]))
        scene.add_physics(name, rigid_body=bool(spec.get("rigid_body", True)),
                          collider=bool(spec.get("collider", True)), kinematic=bool(spec.get("kinematic", False)),
                          mass=None if spec.get("mass") is None else float(spec["mass"]),
                          density=None if spec.get("density") is None else float(spec["density"]),
                          approximation=str(spec.get("approximation", "convexHull")), material=material)
    return scene, link


def apply_link(scene, link: Link) -> list[dict[str, Any]]:
    """Run every job of the link on ``scene`` and attach the returned fields to their parts.

    Returns one report per job (fields, checks, decision); a job that raises is reported, not swallowed -- the
    report then carries ``error`` and the decision ``FAILED``.
    """
    times = [float(t) for t in (scene.times if scene.times is not None else [0.0])]
    reports: list[dict[str, Any]] = []
    for job in link.jobs:
        part = _part_of(scene, job.part)
        req = JobRequest(job=job.job, part=job.part, vertices=part.vertices, times=times, params=dict(job.params),
                         faces=part.faces)
        try:
            res = run_job(req)
        except Exception as exc:  # noqa: BLE001 - reported to the caller as a failed job
            reports.append({"job": job.job, "part": job.part, "decision": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                            "checks": []})
            continue
        for name, (values, unit) in res.fields.items():
            scene.add_field(job.part, name, values, unit=unit)
        rep = res.report()
        rep["fields"] = sorted(res.fields)
        reports.append(rep)
        _attach_sensor_series(scene, link, job.part, res, job.params)
    return reports


def _attach_sensor_series(scene, link: Link, part: str, res, job_params: dict[str, Any]) -> None:
    """A job that names a sensor in ``params["sensor"]`` gives it the field's series at the nearest vertex.

    Explicit only: the link says which sensor reads which job, so nothing is guessed, and an authored series is
    never overwritten (a job that names a sensor with a series of its own is reported, not silently replaced).
    """
    sid = job_params.get("sensor")
    if not sid or scene.times is None or not res.fields:
        return
    try:
        sensor = next(x for x in scene.sensors if x.id == str(sid))
        if sensor.series is not None:
            return
        part_obj = _part_of(scene, part)
        k = int(np.argmin(np.linalg.norm(part_obj.vertices - np.asarray(sensor.position, float), axis=1)))
    except (StopIteration, KeyError, ValueError):
        return
    name = next(iter(res.fields))
    values, unit = res.fields[name]
    sensor.series = np.asarray(values[:, k], float)
    if unit and not sensor.unit:
        sensor.unit = unit


def unity_to_usd(folder: str, out: str, *, usd: bool = True, with_viewer: bool = False) -> dict[str, Any]:
    """The whole lane in one call: Unity folder -> (jobs) -> twin folder, with OpenUSD when ``usd=True``.

    Returns ``{"scene": ..., "reports": [...], "decision": ..., "usd": path|None, "manifest": path}``.
    """
    scene, link = from_unity(folder)
    reports = apply_link(scene, link)
    man = scene.export(out, with_viewer=with_viewer, usd=usd)
    decisions = [r["decision"] for r in reports]
    decision = ("NO JOBS" if not decisions
                else "FAILED" if "FAILED" in decisions
                else "REJECT" if "REJECT" in decisions else "APPROVED")
    return {"scene": scene, "link": link, "reports": reports, "decision": decision, "manifest": man,
            "usd": os.path.join(out, "scene.usda") if usd else None,
            "fields": sorted({n for r in reports for n in r.get("fields", [])})}


# ------------------------------------------------------------------------------------ stand-in for the Unity editor
def emulate_unity_export(scene, folder: str, *, jobs=None, up_axis: str = "y", physics: dict | None = None,
                         sensors: bool = True) -> Link:
    """Write the two files the Unity editor menu writes: ``geometry.glb`` (Y-up) and ``pinneapple_link.json``.

    Stands in for Unity when there is no Unity installed -- the same format the C# ``PinNeappleExport`` menu item
    writes, so the ingest path, the demos and the tests exercise the real files. ``jobs`` is a list of
    ``{"job": ..., "part": ..., "params": {...}}`` (default: none -- geometry only).
    """
    from pinneapple_tools.visualization.studio.scene import AXES, Surface, write_glb

    os.makedirs(folder, exist_ok=True)
    up = str(up_axis).lower()
    if up not in ("y", "z"):
        raise ValueError("up_axis must be 'y' (Unity, glTF) or 'z'")
    surfaces, mats = [], {}
    for i, p in enumerate(scene.parts):
        v = z_up_to_y_up(p.vertices) if up == "y" else np.asarray(p.vertices, float)
        surfaces.append(Surface(p.name, v, p.faces, material=f"part{i}"))
        mats[f"part{i}"] = {"color": list(p.color), "linear": True, "metallic": 0.1, "roughness": 0.7,
                            "double_sided": True}
    glb_name = "geometry.glb"
    with open(os.path.join(folder, glb_name), "wb") as f:
        f.write(write_glb(surfaces, AXES["as_is"], mats, generator="pinneapple_unity (Unity stand-in)"))
    link = Link(title=scene.title, geometry=glb_name, length_unit=scene.length_unit, time_unit=scene.time_unit,
                up_axis=up,
                parts=[LinkPart(p.name, p.name, p.group, tuple(float(c) for c in p.color)) for p in scene.parts],
                jobs=[LinkJob(str(j["job"]), str(j["part"]), dict(j.get("params", {}))) for j in (jobs or [])],
                sensors=[{"id": s.id, "label": s.label, "unit": s.unit, "quantity": s.quantity,
                          "position": list(s.position),
                          "series": None if s.series is None else [float(x) for x in s.series]}
                         for s in (scene.sensors if sensors else [])],
                physics=dict(physics or {}),
                times=[] if scene.times is None else [float(t) for t in scene.times])
    link.save(folder)
    with open(os.path.join(folder, "README.md"), "w") as f:
        f.write(f"# {scene.title} -- Unity export (stand-in)\n\n"
                f"`{glb_name}` (Y-up glTF) + `{LINK_FILE}` (parts, units, sensors, jobs). In a real Unity project the "
                "same two files come from `Window > PINNeAPPle > Export Scene Link`.\n")
    return link


def read_link(folder: str) -> dict[str, Any]:
    """The raw ``pinneapple_link.json`` of a folder (for inspection and for the demo's printouts)."""
    with open(os.path.join(folder, LINK_FILE)) as f:
        return json.load(f)
