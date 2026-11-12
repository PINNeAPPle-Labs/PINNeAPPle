"""Export a :class:`pinneapple_twin3d.Scene` as a folder a Unity project imports as it is.

    from pinneapple_twin3d.demo import pipe_bend_scene
    from pinneapple_unity import export_for_unity

    exp = export_for_unity(pipe_bend_scene(), "out/bend_unity")   # <- drag out/bend_unity/Assets into Unity
    print(exp.verify())                                           # [] or the list of problems

Layout (a Unity project root, so the ``Assets`` folder can be dropped on an existing project)::

    Assets/PinNeAPPleTwin/          the C# runtime, the two unlit field shaders, the editor menu, docs
    Assets/TwinData/scene.json      parts, fields, time steps, sensors, units (pinneapple-twin3d/1)
    Assets/TwinData/geometry.glb    one mesh per part
    Assets/TwinData/fields.bin      float32 field values and rigid motions, offsets in scene.json
    Assets/TwinData/pinneapple_unity.json   up axis, units, the rotation Unity applies, provenance
    README.md                       what to do in Unity, in four steps

The data files are exactly the ones the web viewer reads (:meth:`pinneapple_twin3d.Scene.export`): one layout,
several front ends. The scene is Z-up (`pinneapple_twin3d` convention) and Unity is Y-up, so
``pinneapple_unity.json`` records the -90 deg rotation about X that the runtime applies to the root.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .protocol import DATA_DIR, PACKAGE_DIR, PROTOCOL, UNITY_FILE, Z_UP_TO_Y_UP

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_SOURCE = os.path.join(HERE, "unity", "Assets", "PinNeAPPleTwin")

_DATA_README = """# Twin data (PINNeAPPle)

`scene.json` + `geometry.glb` + `fields.bin` are a PINNeAPPle twin, the same three files the web viewer reads.
`pinneapple_unity.json` tells the Unity runtime how to read them (up axis, units, rotation, provenance).

Do not edit these files by hand: re-export from Python (`pinneapple_unity.export_for_unity`) or, for a Unity-authored
scene, run the jobs with `pinneapple_unity.from_unity` and let it write a new twin next to this one.
"""


@dataclass
class UnityExport:
    """What :func:`export_for_unity` wrote (and what :meth:`verify` found wrong, if anything)."""

    folder: str
    data_dir: str
    package_dir: str
    scene_json: str
    unity_json: str
    parts: int = 0
    fields: list[str] = field(default_factory=list)
    steps: int = 1
    problems: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def verify(self) -> list[str]:
        """Re-read what was written and list every inconsistency (empty when the folder is sound)."""
        self.problems = verify_unity_folder(self.folder)
        return self.problems

    def summary(self) -> str:
        return (f"{self.parts} part(s), {len(self.fields)} field(s) {self.fields}, {self.steps} step(s) -> "
                f"{self.folder}")


def export_for_unity(scene, folder: str, *, usd: bool = False, overwrite: bool = True) -> UnityExport:
    """Write ``scene`` as a Unity-importable project folder (see the module docstring for the layout)."""
    if not getattr(scene, "parts", None):
        raise ValueError("the scene has no parts")
    if os.path.isdir(folder) and os.listdir(folder) and not overwrite:
        raise FileExistsError(f"{folder} is not empty (pass overwrite=True to replace the twin files)")
    data_dir = os.path.join(folder, DATA_DIR)
    package_dir = os.path.join(folder, PACKAGE_DIR)
    os.makedirs(folder, exist_ok=True)
    scene_json = scene.export(data_dir, with_viewer=False, usd=usd)

    if not os.path.isdir(PACKAGE_SOURCE):
        raise FileNotFoundError(f"the Unity package sources are missing: {PACKAGE_SOURCE}")
    if os.path.isdir(package_dir):
        shutil.rmtree(package_dir)
    shutil.copytree(PACKAGE_SOURCE, package_dir)

    meta = {
        "protocol": PROTOCOL,
        "title": scene.title,
        "source": scene.source,
        "length_unit": scene.length_unit,
        "time_unit": scene.time_unit,
        "up_axis": "z",                                     # pinneapple_twin3d scenes are Z-up
        "unity": {"up_axis": "y", "up_rotation": "Euler(-90, 0, 0)", "up_rotation_quaternion_wxyz": list(Z_UP_TO_Y_UP),
                  "length_unit": "m", "meters_per_unit": 1.0},
        "data": {"scene": "scene.json", "geometry": "geometry.glb", "fields": "fields.bin"},
        "package": PACKAGE_DIR,
        "counts": {"parts": len(scene.parts), "sensors": len(scene.sensors),
                   "steps": 1 if scene.times is None else len(scene.times)},
    }
    unity_json = os.path.join(data_dir, UNITY_FILE)
    with open(unity_json, "w") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    with open(os.path.join(data_dir, "PINNEAPPLE.md"), "w") as f:
        f.write(_DATA_README)

    fields = sorted({name for p in scene.parts for name in p.fields})
    exp = UnityExport(folder=os.path.abspath(folder), data_dir=os.path.abspath(data_dir),
                      package_dir=os.path.abspath(package_dir), scene_json=os.path.abspath(scene_json),
                      unity_json=os.path.abspath(unity_json), parts=len(scene.parts), fields=fields,
                      steps=1 if scene.times is None else len(scene.times), info=meta)
    with open(os.path.join(folder, "README.md"), "w") as f:
        f.write(_folder_readme(exp, scene))
    exp.verify()
    return exp


def _folder_readme(exp: UnityExport, scene) -> str:
    sensors = ", ".join(f"`{s.id}`" for s in scene.sensors) or "none"
    return f"""# {scene.title} -- PINNeAPPle twin for Unity

{exp.parts} part(s), field(s) {", ".join(f"`{n}`" for n in exp.fields) or "none"}, {exp.steps} time step(s),
sensors: {sensors}.

## Four steps

1. Create a Unity project (2021.3 LTS or newer; HDRP/URP/Built-in all work) and copy this folder's `Assets` into the
   project's `Assets` (or point Unity at this folder with `-projectPath`).
2. Add an empty GameObject and give it the **PINNeAPPle Twin View** component
   (`Assets/PinNeAPPleTwin/Runtime/PinNeappleTwinView.cs`).
3. Set its **Twin Folder** to `Assets/TwinData` (the default) and press Play: the parts appear, the field is
   coloured per vertex, the time slider walks the steps, the sensor markers carry their series and alarm envelope.
4. Pick the field in the component's **Field** dropdown (or leave it on *colourize the first field*).

## What the runtime gives you

* `PinneappleTwinView` -- loads `scene.json` + `geometry.glb` + `fields.bin`, builds one GameObject per part,
  animates the field over `scene.times`, applies rigid motions and per-step visibility.
* `PinneappleField` -- vertex-colour material driving the colormap (turbo / viridis), with the legend on screen.
* `PinneappleBridgeClient` -- sends this twin to a PINNeAPPle bridge (`python -m pinneapple_unity serve`) and
  streams back a computed field with its checks; the protocol is `pinneapple-unity/1`.
* `Window > PINNeAPPle` -- check the data folder, open its folder, export a `pinneapple_link.json` for the
  Unity -> PINNeAPPle -> OpenUSD lane.

Checked at export: {len(exp.problems)} problem(s) in the data files.
"""


# --------------------------------------------------------------------------------------------------- checking
def _read_glb_counts(path: str) -> tuple[int, int]:
    """(meshes, vertices) of a .glb, read from its JSON chunk without a glTF library."""
    with open(path, "rb") as f:
        head = f.read(20)
        if len(head) < 20 or head[:4] != b"glTF":
            raise ValueError(f"{path}: not a binary glTF (no 'glTF' magic)")
        version, _length = int.from_bytes(head[4:8], "little"), int.from_bytes(head[8:12], "little")
        if version != 2:
            raise ValueError(f"{path}: glTF version {version}, expected 2")
        chunk_len, chunk_type = int.from_bytes(head[12:16], "little"), head[16:20]
        if chunk_type != b"JSON":
            raise ValueError(f"{path}: the first chunk is {chunk_type!r}, expected b'JSON'")
        gl = json.loads(f.read(chunk_len))
    accessors = gl.get("accessors", [])
    verts = sum(int(accessors[p["attributes"]["POSITION"]]["count"])
                for m in gl.get("meshes", []) for p in m.get("primitives", []))
    return len(gl.get("meshes", [])), verts


def scene_from_twin(folder: str):
    """Re-open an exported twin folder as a :class:`pinneapple_twin3d.Scene` (parts, fields, sensors, times, physics).

    The inverse of :func:`export_for_unity` on the data files: read ``scene.json`` + ``geometry.glb`` + ``fields.bin``
    back into a scene, so a twin can be extended (more jobs, more steps, a render) without the script that built it.
    ``UsdPhysics`` assignment is not read back -- it is not in ``scene.json``; a twin written from a Unity link keeps
    it in the link, which :func:`pinneapple_unity.from_unity` reads.
    """
    from pinneapple_tools.visualization.studio.scene import read_gltf
    from pinneapple_twin3d import Scene

    man, _verts, fields = read_unity_twin(folder)
    data_dir = os.path.join(folder, DATA_DIR) if os.path.isdir(os.path.join(folder, DATA_DIR)) else folder
    surfaces = read_gltf(os.path.join(data_dir, man.get("geometry", "geometry.glb")), axes="as_is")
    times = np.asarray(man.get("times", []), float)
    scene = Scene(str(man.get("title", "")), length_unit=str(man.get("length_unit", "m")),
                  times=times if times.size else None, time_unit=str(man.get("time_unit", "s")),
                  source=str(man.get("source", "")))
    for i, p in enumerate(man.get("parts", [])):
        k = int(p.get("mesh_index", i))
        s = surfaces[k] if k < len(surfaces) else surfaces[i]
        scene.add_part(str(p["name"]), s.vertices, s.faces, group=str(p.get("group", "")),
                       color=tuple(p.get("color", (0.75, 0.77, 0.8))))
    for (part, name), values in fields.items():
        scene.add_field(part, name, values, unit="")
    for name, unit in _field_units(man).items():
        scene.field_units[name] = unit
    for s in man.get("sensors", []):
        scene.add_sensor(str(s["id"]), s.get("position", (0, 0, 0)), label=str(s.get("label", "")),
                         unit=str(s.get("unit", "")), quantity=str(s.get("quantity", "")),
                         series=s.get("series"), envelope=s.get("envelope"))
    return scene


def _field_units(man: dict[str, Any]) -> dict[str, str]:
    """The unit each field name carries, from the per-part field entries of a manifest."""
    out: dict[str, str] = {}
    for e in man.get("fields", []):
        if e.get("unit"):
            out.setdefault(str(e["name"]), str(e["unit"]))
    return out


def verify_unity_folder(folder: str) -> list[str]:
    """Every inconsistency in an exported twin folder: manifests, GLB structure, `fields.bin` offsets, the package."""
    problems: list[str] = []
    data_dir = os.path.join(folder, DATA_DIR) if os.path.isdir(os.path.join(folder, DATA_DIR)) else folder
    scene_path = os.path.join(data_dir, "scene.json")
    unity_path = os.path.join(data_dir, UNITY_FILE)
    if not os.path.exists(scene_path):
        return [f"{data_dir}: no scene.json"]
    with open(scene_path) as f:
        man = json.load(f)
    if man.get("format") != "pinneapple-twin3d/1":
        problems.append(f"scene.json: format {man.get('format')!r}, expected 'pinneapple-twin3d/1'")
    glb = os.path.join(data_dir, man.get("geometry", "geometry.glb"))
    try:
        meshes, verts = _read_glb_counts(glb)
    except (OSError, ValueError) as exc:
        problems.append(f"geometry: {exc}")
        meshes = verts = -1
    parts = man.get("parts", [])
    if meshes >= 0 and meshes != len(parts):
        problems.append(f"geometry.glb has {meshes} mesh(es), scene.json {len(parts)} part(s)")
    if verts >= 0 and verts != sum(int(p.get("vertices", 0)) for p in parts):
        problems.append(f"geometry.glb has {verts} vertices, scene.json sums to "
                        f"{sum(int(p.get('vertices', 0)) for p in parts)}")
    bin_path = os.path.join(data_dir, man.get("fields_file", "fields.bin"))
    if not os.path.exists(bin_path):
        problems.append("fields.bin is missing")
    else:
        size = os.path.getsize(bin_path)
        need = 0
        nverts = {p["name"]: int(p.get("vertices", 0)) for p in parts}
        for e in man.get("fields", []):
            need = max(need, int(e["offset"]) + 4 * int(e["steps"]) * int(e["count"]))
        for e in man.get("motions", []):
            need = max(need, int(e["offset"]) + 4 * int(e["steps"]) * 7 * nverts.get(e["part"], 0))
        for e in man.get("visibility", []):
            need = max(need, int(e["offset"]) + 4 * int(e["steps"]) * nverts.get(e["part"], 0))
        if size < need:
            problems.append(f"fields.bin is {size} bytes, the manifest needs {need}")
    if not os.path.exists(unity_path):
        problems.append(f"{UNITY_FILE} is missing (up axis and units for the Unity runtime)")
    else:
        with open(unity_path) as f:
            meta = json.load(f)
        if meta.get("protocol") != PROTOCOL:
            problems.append(f"{UNITY_FILE}: protocol {meta.get('protocol')!r}, expected {PROTOCOL!r}")
        if meta.get("up_axis") not in ("y", "z"):
            problems.append(f"{UNITY_FILE}: up_axis {meta.get('up_axis')!r}")
    package = os.path.join(folder, PACKAGE_DIR)
    for rel in ("Runtime/PinNeappleTwinView.cs", "Runtime/PinNeappleTwinData.cs", "Editor/PinNeappleMenu.cs",
                "Runtime/PinNeAPPleTwin.Runtime.asmdef", "README.md"):
        if not os.path.exists(os.path.join(package, rel)):
            problems.append(f"the Unity package is incomplete: {PACKAGE_DIR}/{rel} is missing")
    return problems


def read_unity_twin(folder: str) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    """The manifest, the part vertices and the fields of an exported twin folder (what the Unity runtime reads).

    Returns ``(manifest, vertices, fields)`` with ``fields[(part, name)] = (T, V) float32``. This is the Python
    mirror of the C# loader, so an export can be checked without Unity.
    """
    data_dir = os.path.join(folder, DATA_DIR) if os.path.isdir(os.path.join(folder, DATA_DIR)) else folder
    with open(os.path.join(data_dir, "scene.json")) as f:
        man = json.load(f)
    from pinneapple_tools.visualization.studio.scene import read_gltf

    surfaces = read_gltf(os.path.join(data_dir, man.get("geometry", "geometry.glb")), axes="as_is")
    verts = np.concatenate([s.vertices for s in surfaces], axis=0)
    raw = np.fromfile(os.path.join(data_dir, man.get("fields_file", "fields.bin")), dtype="<f4")
    fields: dict[tuple[str, str], np.ndarray] = {}
    for e in man.get("fields", []):
        n = int(e["steps"]) * int(e["count"])
        off = int(e["offset"]) // 4
        fields[(e["part"], e["name"])] = raw[off:off + n].reshape(int(e["steps"]), int(e["count"]))
    return man, verts, fields
