"""Export a Twin3D scene as OpenUSD (``.usda`` text), for Omniverse / Isaac Sim / usdview.

Why USD next to glTF (``ROADMAP.md`` §11, NVIDIA GTC 2026 CAD -> sim -> twin thread): glTF has
no standard slot for a field that changes over time, USD does. Here every part is a ``Mesh``,
every field a per-vertex ``primvars:<name>`` whose values are **time samples** (one per scene
time step), and every sensor a ``Sphere`` with its series, unit and alarm envelope as custom
attributes. Provenance (title, source, units, real time values) goes into the layer metadata so
the meaning of the data survives the hand-off.

Physics (``UsdPhysics``): parts given ``Scene.add_physics`` become rigid bodies and/or colliders
(``PhysicsRigidBodyAPI``, ``PhysicsMassAPI``, ``PhysicsCollisionAPI``, ``PhysicsMeshCollisionAPI``),
with contact materials (``PhysicsMaterialAPI``) bound through ``material:binding:physics``;
``Scene.set_gravity`` adds a ``PhysicsScene``. Recorded rigid motion (``Scene.add_motion``) is written
as time-sampled ``xformOp:translate`` / ``xformOp:orient`` -- with ``kinematic=True`` the body replays
it, so the same file opens as a playback in usdview and as a live scene in Omniverse / Isaac Sim.

Written as plain text: no ``pxr`` dependency to export. ``usd-core`` (``pip install usd-core``)
can open the result; the tests use it when available.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from .scene import Scene

_METERS = {"m": 1.0, "mm": 0.001, "cm": 0.01, "km": 1000.0, "in": 0.0254, "ft": 0.3048}


def _ident(name: str, used: set) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name) or "prim"
    if s[0].isdigit():
        s = "_" + s
    base, k = s, 1
    while s in used:
        k += 1
        s = f"{base}_{k}"
    used.add(s)
    return s


def _floats(a: Iterable[float]) -> str:
    return ", ".join(f"{float(x):.7g}" for x in a)


def _vec3(a: np.ndarray) -> str:
    return ", ".join(f"({x:.7g}, {y:.7g}, {z:.7g})" for x, y, z in np.asarray(a, float))


def _q(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def export_usd(scene: Scene, path: str) -> str:
    n_steps = 1 if scene.times is None else len(scene.times)
    times = [] if scene.times is None else [float(t) for t in scene.times]
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Twin"',
        f"    metersPerUnit = {_METERS.get(scene.length_unit, 1.0)}",
        "    kilogramsPerUnit = 1",
        '    upAxis = "Z"',
        "    startTimeCode = 0",
        f"    endTimeCode = {max(n_steps - 1, 0)}",
        "    timeCodesPerSecond = 1",
        "    customLayerData = {",
        f"        string pinneapple_format = {_q('pinneapple-twin3d/1')}",
        f"        string title = {_q(scene.title)}",
        f"        string source = {_q(scene.source)}",
        f"        string length_unit = {_q(scene.length_unit)}",
        f"        string time_unit = {_q(scene.time_unit)}",
        f"        double[] time_values = [{_floats(times)}]",
        "    }",
        ")",
        "",
        'def Xform "Twin" (',
        '    kind = "assembly"',
        ")",
        "{",
    ]
    used: set = set()
    used.update({"PhysicsMaterials", "physicsScene"})
    if getattr(scene, "gravity", None) is not None:
        g = np.asarray(scene.gravity, float)
        mag = float(np.linalg.norm(g))
        d = g / mag if mag > 0 else np.array([0.0, 0.0, -1.0])
        lines += ['    def PhysicsScene "physicsScene"', "    {",
                  f"        vector3f physics:gravityDirection = ({_floats(d)})",
                  f"        float physics:gravityMagnitude = {mag:.7g}", "    }"]
    materials = {}
    for part in scene.parts:
        ph = getattr(part, "physics", None)
        if ph is not None and ph.material is not None:
            materials.setdefault(ph.material.name, ph.material)
    mat_path = {}
    if materials:
        lines += ['    def Scope "PhysicsMaterials"', "    {"]
        mused: set = set()
        for mname, m in materials.items():
            ident = _ident(mname, mused)
            mat_path[mname] = f"/Twin/PhysicsMaterials/{ident}"
            lines += [f'        def Material "{ident}" (', '            prepend apiSchemas = ["PhysicsMaterialAPI"]',
                      "        )", "        {",
                      f"            float physics:staticFriction = {m.static_friction:.7g}",
                      f"            float physics:dynamicFriction = {m.dynamic_friction:.7g}",
                      f"            float physics:restitution = {m.restitution:.7g}"]
            if m.density is not None:
                lines.append(f"            float physics:density = {m.density:.7g}")
            lines += ["        }"]
        lines += ["    }"]
    for part in scene.parts:
        name = _ident(part.name, used)
        counts = ", ".join(["3"] * len(part.faces))
        ph = getattr(part, "physics", None)
        schemas = []
        if ph is not None:
            if ph.rigid_body:
                schemas += ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
            if ph.collider:
                schemas += ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
            if ph.material is not None:
                schemas += ["MaterialBindingAPI"]
        if schemas:
            lines += [f'    def Mesh "{name}" (',
                      "        prepend apiSchemas = [" + ", ".join(_q(x) for x in schemas) + "]",
                      "    )"]
        else:
            lines.append(f'    def Mesh "{name}"')
        lines += [
            "    {",
            f"        custom string pinneapple:name = {_q(part.name)}",
            f"        custom string pinneapple:group = {_q(part.group)}",
            f"        int[] faceVertexCounts = [{counts}]",
            f"        int[] faceVertexIndices = [{', '.join(str(int(i)) for i in part.faces.reshape(-1))}]",
            f"        point3f[] points = [{_vec3(part.vertices)}]",
            '        uniform token subdivisionScheme = "none"',
            f"        color3f[] primvars:displayColor = [({_floats(part.color)})]",
        ]
        if ph is not None:
            if ph.rigid_body:
                lines += ["        bool physics:rigidBodyEnabled = 1",
                          f"        bool physics:kinematicEnabled = {int(ph.kinematic)}"]
                if ph.velocity is not None:
                    lines.append(f"        vector3f physics:velocity = ({_floats(ph.velocity)})")
                if ph.angular_velocity is not None:
                    lines.append(f"        vector3f physics:angularVelocity = ({_floats(ph.angular_velocity)})")
                if ph.mass is not None:
                    lines.append(f"        float physics:mass = {ph.mass:.7g}")
                if ph.density is not None:
                    lines.append(f"        float physics:density = {ph.density:.7g}")
                if ph.center_of_mass is not None:
                    lines.append(f"        point3f physics:centerOfMass = ({_floats(ph.center_of_mass)})")
            if ph.collider:
                lines += ["        bool physics:collisionEnabled = 1",
                          f"        uniform token physics:approximation = {_q(ph.approximation)}"]
            if ph.material is not None:
                lines.append(f"        rel material:binding:physics = <{mat_path[ph.material.name]}>")
        motion = getattr(part, "motion", None)
        if motion is not None:
            lines.append("        double3 xformOp:translate.timeSamples = {")
            lines += [f"            {k}: ({_floats(motion[k, :3])})," for k in range(len(motion))]
            lines += ["        }", "        quatf xformOp:orient.timeSamples = {"]
            lines += [f"            {k}: ({_floats(motion[k, 3:])})," for k in range(len(motion))]
            lines += ["        }", '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]']
        for fname, arr in part.fields.items():
            pv = "primvars:" + _ident(fname, set())
            unit = scene.field_units.get(fname, "")
            meta = ['            interpolation = "vertex"',
                    "            customData = {",
                    f"                string unit = {_q(unit)}",
                    f"                string field = {_q(fname)}",
                    "            }",
                    "        )"]
            if arr.shape[0] == 1:  # steady field: value + metadata on one declaration
                lines += [f"        float[] {pv} = [{_floats(arr[0])}] ("] + meta
            else:  # transient field: metadata, then one time sample per scene step
                lines += [f"        float[] {pv} ("] + meta
                lines.append(f"        float[] {pv}.timeSamples = {{")
                lines += [f"            {k}: [{_floats(arr[k])}]," for k in range(arr.shape[0])]
                lines.append("        }")
        lines.append("    }")
    for s in scene.sensors:
        name = _ident("sensor_" + s.id, used)
        lines += [
            f'    def Sphere "{name}"',
            "    {",
            "        double radius = 0.01",
            f"        double3 xformOp:translate = ({_floats(s.position)})",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]',
            f"        custom string pinneapple:sensor_id = {_q(s.id)}",
            f"        custom string pinneapple:label = {_q(s.label)}",
            f"        custom string pinneapple:unit = {_q(s.unit)}",
            f"        custom string pinneapple:quantity = {_q(s.quantity)}",
        ]
        if s.envelope is not None:
            lines.append(f"        custom double2 pinneapple:envelope = ({_floats(s.envelope)})")
        if s.series is not None:
            lines.append("        custom double pinneapple:value.timeSamples = {")
            lines += [f"            {k}: {float(v):.7g}," for k, v in enumerate(s.series)]
            lines.append("        }")
        lines.append("    }")
    lines.append("}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
