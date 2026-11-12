"""Unity -> PINNeAPPle -> OpenUSD: a scene a team placed in Unity gets its physics and leaves as a USD layer.

    python examples/unity/02_unity_to_pinneapple_to_openusd.py

The lane, in the order the files travel:

    Unity           Assets/TwinLink/scene.glb + pinneapple_link.json     (parts, units, sensors, UsdPhysics, jobs)
    PINNeAPPle      the jobs of the link, run and checked              (thermal + structural)
    OpenUSD         scene.usda: time-sampled fields, UsdPhysics, provenance for Omniverse / Isaac Sim

No Unity installed here: `emulate_unity_export` writes exactly the two files the editor menu writes
(`PINNeAPPle > Export Selection as Scene Link`), so this script exercises the real format on any machine. Set
`FOLDER` to a folder your Unity wrote to run the same pipeline on it.

Prints the link, the check table of every job, what went into the USD layer, and the twin re-opened as a scene.
"""
from __future__ import annotations

import os
import re

import numpy as np

from pinneapple_twin3d import Scene
from pinneapple_twin3d.primitives import sweep
from pinneapple_unity import (
    apply_link,
    emulate_unity_export,
    export_for_unity,
    from_unity,
    read_link,
    scene_from_twin,
    unity_to_usd,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "_out", "bracket_unity")      # <- a Unity export folder, real or emulated
TWIN = os.path.join(HERE, "_out", "bracket_twin")
TIMES = [0.0, 60.0, 300.0, 900.0, 1800.0, 3600.0]

PLATE = {"length": 0.2, "width": 0.08, "thickness": 0.006}
BEAM = {"length": 0.25, "width": 0.05, "thickness": 0.01}


def _section(a: float, b: float):
    """Rectangle section of a swept part, in the sweep's own (u, w) plane."""
    return np.array([(-b / 2, -a / 2), (b / 2, -a / 2), (b / 2, a / 2), (-b / 2, a / 2)], dtype=float)


def _path(start, length: float, n: int = 41):
    s = np.linspace(0.0, length, n)
    return np.stack([start[0] + s, np.full(n, start[1]), np.full(n, start[2])], axis=1)


def unity_scene() -> Scene:
    """What the team modelled in Unity: a plate and the cantilevered bracket welded to it."""
    scene = Scene("Bracket: plate + cantilever", length_unit="m", times=TIMES, time_unit="s",
                  source="a Unity scene (emulated by examples/unity/02_unity_to_pinneapple_to_openusd.py)")
    v, f = sweep(_section(PLATE["thickness"], PLATE["width"]), _path((0.0, 0.0, 0.0), PLATE["length"]))
    scene.add_part("plate", v, f, group="structure", color=(0.72, 0.75, 0.80))
    v, f = sweep(_section(BEAM["thickness"], BEAM["width"]), _path((0.0, 0.1, 0.0), BEAM["length"]))
    scene.add_part("bracket", v, f, group="structure", color=(0.80, 0.72, 0.55))
    scene.add_sensor("TC-01", (0.1, 0.0, PLATE["thickness"] / 2), label="TC-01 plate, mid length", unit="C")
    return scene


def link_jobs() -> list[dict]:
    """What the scene asks PINNeAPPle to compute -- this list is the whole contract of the lane."""
    return [
        {"job": "slab_transient", "part": "plate",
         "params": {"alpha": 1e-5, "t_left": 120.0, "t_right": 20.0, "t0": 20.0,
                    "axis": "x", "length": PLATE["length"], "sensor": "TC-01"}},
        {"job": "cantilever_stress", "part": "bracket",
         "params": {"width": BEAM["width"], "thickness": BEAM["thickness"], "load": 40.0,
                    "length": BEAM["length"], "axis": "x", "yield_mpa": 250.0}},
    ]


def link_physics() -> dict:
    return {
        "gravity": [0.0, -9.81, 0.0],
        "parts": {
            "plate": {"rigid_body": False, "collider": True,
                      "material": {"name": "steel_contact", "static_friction": 0.4,
                                   "dynamic_friction": 0.3, "restitution": 0.0}},
            "bracket": {"rigid_body": True, "collider": True, "approximation": "convexHull",
                        "material": {"name": "steel_contact", "static_friction": 0.4,
                                     "dynamic_friction": 0.3, "restitution": 0.0}},
        },
    }


def main() -> None:
    # 1. the Unity side: geometry (glTF, Y-up) + the link (Z-up conversion declared, jobs, UsdPhysics)
    if not os.path.exists(os.path.join(SOURCE, "pinneapple_link.json")):
        emulate_unity_export(unity_scene(), SOURCE, jobs=link_jobs(), physics=link_physics())
        print(f"emulated the Unity export in {os.path.relpath(SOURCE, HERE)} "
              f"(geometry.glb + pinneapple_link.json; in Unity this is the PINNeAPPle menu item)")
    link = read_link(SOURCE)
    print("\nlink:", link["title"], "| up axis", link["up_axis"], "| length unit", link["length_unit"])
    print("  parts:", [(p["name"], p["axis"]) for p in link["parts"]])
    print("  jobs: ", [(j["job"], j["part"]) for j in link["jobs"]])
    print("  physics:", list(link["physics"]["parts"]), "gravity", link["physics"]["gravity"])

    # 2. PINNeAPPle: read the scene (Y-up -> Z-up), run the jobs, keep the checks
    scene, parsed = from_unity(SOURCE)
    print(f"\nread back: {len(scene.parts)} part(s), {len(scene.times)} step(s), "
          f"unit {scene.length_unit}, up axis was '{parsed.up_axis}'")
    for part in scene.parts:
        lo, hi = part.vertices.min(0), part.vertices.max(0)
        print(f"  {part.name:8s} {len(part.vertices):5d} vertices   bbox {np.round(lo, 4)} .. {np.round(hi, 4)}")
    reports = apply_link(scene, parsed)
    for rep in reports:
        print(f"\n{rep['job']} on {rep['part']}: {rep['decision']}")
        for c in rep["checks"]:
            mark = {True: "PASS", False: "FAIL", None: "n/a "}[c["passed"]]
            value = "" if c["value"] is None else f"  value {c['value']:.4g}"
            print(f"   {mark}  {c['name']}{value}")
    sensor = scene.sensors[0]
    print(f"\nsensor {sensor.id}: {np.round(sensor.series, 1)} {sensor.unit}")

    # 3. OpenUSD: the same scene with the fields as time samples and the physics as UsdPhysics
    out = unity_to_usd(SOURCE, TWIN)
    usda = open(out["usd"]).read()
    print(f"\nUSD layer: {os.path.relpath(out['usd'], HERE)}  ({len(usda) // 1024} kB), decision {out['decision']}")
    print("  layer metadata:", ", ".join(re.findall(r"^\s+(?:string|double\[\]) (\w+)",
                                                   _layer_block(usda), re.M)))
    for key in ("timeSamples", "PhysicsCollisionAPI", "PhysicsMaterialAPI", "PhysicsScene", "primvars:temperature",
                "primvars:bending_stress", "customLayerData"):
        print(f"  {key:26s} {usda.count(key)}")
    print("  endTimeCode:", re.search(r"endTimeCode = (\d+)", usda).group(1),
          "| upAxis:", re.search(r'upAxis = "(\w)"', usda).group(1))

    # 4. and back into Unity: the twin reopened, then written as a project folder with the computed fields
    again = scene_from_twin(TWIN)
    print(f"\nre-opened the twin: {len(again.parts)} part(s), fields "
          f"{sorted({n for p in again.parts for n in p.fields})}, "
          f"{len(again.times)} step(s), units {again.field_units}")
    back = export_for_unity(again, os.path.join(HERE, "_out", "bracket_back_in_unity"))
    print(f"back in Unity: {os.path.relpath(back.folder, HERE)} ({back.summary()}), "
          f"verify: {back.problems or 'every file checks out'}")

    print("""
What this demonstrates
  * a scene authored in Unity is physics-ready: the link says what is a part, what is a sensor, what is a collider
  * the jobs run here, with their checks, and the decision travels with the field (APPROVED / REJECT)
  * the OpenUSD layer carries the field as time samples, the parts as UsdPhysics colliders with a contact material,
    and the provenance (title, source, units, the real times) in its layer metadata, so the meaning survives the
    hand-off to Omniverse / Isaac Sim / usdview
  * the loop closes: the USD layer, the twin and the Unity folder all carry the same computed fields, so the
    scene the team modelled comes back with the physics on it""")


def _layer_block(text: str) -> str:
    """The header block of an .usda (everything before the first prim), for printing."""
    i = text.find("def Xform")
    return text[:i if i > 0 else 2000]


if __name__ == "__main__":
    main()
