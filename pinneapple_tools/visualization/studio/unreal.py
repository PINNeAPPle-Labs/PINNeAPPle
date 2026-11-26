"""Unreal Engine bridge: a :class:`~.scene.Scene` exported as a folder Unreal opens, with the field on the surface.

    from pinneapple_tools.visualization.studio import Scene, export_unreal
    sc = Scene.from_file("cantilever.frd")
    export_unreal(sc, "ue/cantilever", field="STRESS")      # -> ue/cantilever/

The folder holds ``geometry.glb`` (PBR materials, every field as a float vertex attribute ``_NAME`` and the chosen
one also as ``COLOR_0`` vertex colours), ``scene.usda`` (the USD Importer plugin, UE 5.4+), ``fields.bin`` (the raw
values, offsets in ``manifest.json``), ``manifest.json`` (surfaces, fields, labels, colour range) and
``PinneAPPleImport.py``, which runs inside the editor and does the rest -- import, vertex-colour material, actors,
lights, framed cine camera -- with the steps in the ``README.md`` written next to it.

Why glTF + USD and not Datasmith
--------------------------------
A ``.udatasmith`` file is written by the Datasmith *exporter* and read by the Datasmith importer, plugins that ship
with a full editor install and carry Unreal-version-specific binary assets. glTF is read by the Interchange glTF
importer built into UE 5.x and USD by the USD Importer plugin, and both are formats this side writes with ``numpy``
alone -- the same reason :mod:`.blender` writes PLY rather than Alembic.

Why vertex colours for the field
--------------------------------
glTF has no standard slot for a named per-vertex field, so the ``_NAME`` float attributes three.js reads back
(``_name``) are ignored by Unreal's importer, which reads ``COLOR_0``. The field is therefore baked into ``COLOR_0``
through the same colormap the Blender render and the browser viewer use, so the colours in Unreal match the ones in a
report; the raw numbers stay in ``fields.bin`` for a material of one's own.

The export side is pure Python and covered by the tests. ``PinneAPPleImport.py`` only runs inside a real Unreal
install, so it is not exercised by the suite: every step it takes is attempted and logged separately, and a step the
editor refuses does not abort the rest.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .compose import cmap_color, default_range
from .scene import AXES, MATERIALS, Scene, write_glb

#: the ``manifest.json`` schema tag
FORMAT = "pinneapple.unreal/1"

#: above this many vertices the glb is refused: Unreal's importer slows down badly on one huge mesh, and the file
#: is written in a single buffer
MAX_VERTICES = 6_000_000

AXES_NOTE = {"z_up": "z up, metres", "as_is": "as exported, unchanged",
             "aircraft": "aircraft axes (x aft, y right, z up)"}


def _linear(c: np.ndarray) -> np.ndarray:
    """glTF ``COLOR_0`` is linear; the colormaps (and the Blender/browser paths) are sRGB."""
    return np.clip(np.asarray(c, float), 0.0, 1.0) ** 2.2


def color_range(scene: Scene, field: str, rng: Optional[Sequence[float]] = None,
                cmap: str = "jet") -> Tuple[float, float]:
    """The colour range for ``field``: ``rng`` if given, else the scene's own values, symmetric about zero for a
    diverging ``cmap`` (as :func:`~.compose.default_range` does for the renders)."""
    if rng is not None:
        lo, hi = float(rng[0]), float(rng[1])
        if not hi > lo:
            raise ValueError(f"field_range must be (low, high) with high > low, got {tuple(rng)!r}")
        return lo, hi
    vals = [np.asarray(s.fields[field], float) for s in scene.surfaces if field in s.fields]
    if not vals:
        raise KeyError(f"no surface carries the field {field!r} (fields: {scene.field_names()})")
    return default_range(np.concatenate(vals), cmap)


def vertex_colors(scene: Scene, field: str, *, cmap: str = "jet",
                  rng: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
    """Per-surface ``(V, 3)`` colours for ``field`` on the ``cmap`` scale, **sRGB in [0, 1]** (the glTF writer turns
    them into linear). A surface without the field takes its material's base colour, so the whole assembly stays
    visible; a non-finite value takes the low end of the scale."""
    lo, hi = color_range(scene, field, rng, cmap)
    span = (hi - lo) or 1.0
    out: Dict[str, np.ndarray] = {}
    for s in scene.surfaces:
        v = s.fields.get(field)
        if v is None:
            base = np.asarray(MATERIALS.get(s.material, MATERIALS["grey"])["color"][:3], float)
            out[s.name] = np.tile(base, (len(s.vertices), 1))
            continue
        t = (np.asarray(v, float) - lo) / span
        t = np.where(np.isfinite(t), np.clip(t, 0.0, 1.0), 0.0)
        uniq, inv = np.unique(np.round(t, 6), return_inverse=True)   # cmap_color is scalar: sample unique values
        out[s.name] = np.array([cmap_color(cmap, float(u)) for u in uniq], float)[inv]
    return out


def _manifest(scene: Scene, field: Optional[str], cmap: str, rng: Tuple[float, float],
              files: Dict[str, str]) -> Dict:
    """What the editor script and any material of one's own read back: the surfaces, where each field's values sit
    in ``fields.bin``, and the colour range the vertex colours were baked with."""
    offsets: Dict[str, Dict[str, Dict[str, int]]] = {}
    at = 0
    for s in scene.surfaces:
        offsets[s.name] = {}
        for name, vals in s.fields.items():
            n = int(np.asarray(vals).size)
            offsets[s.name][name] = {"offset": at, "count": n}
            at += n
    return {
        "format": FORMAT,
        "scene": {"title": scene.title, "axes": scene.axes,
                  "axes_note": AXES_NOTE.get(scene.axes, scene.axes),
                  "surfaces": len(scene.surfaces),
                  "vertices": int(sum(len(s.vertices) for s in scene.surfaces)),
                  "faces": int(scene.n_faces()), "labels": scene.labels},
        "field": {"name": field, "cmap": cmap, "range": [rng[0], rng[1]],
                  "label": scene.labels.get(field, field) if field else ""},
        "fields": {name: {"label": scene.labels.get(name, name)} for name in scene.field_names()},
        "surfaces": [{"name": s.name, "material": s.material, "group": s.group,
                      "vertices": int(len(s.vertices)), "faces": int(len(s.faces)),
                      "fields": offsets[s.name]} for s in scene.surfaces],
        "fields_bin": {"file": files["fields"], "dtype": "float32",
                       "order": "surface by surface, field by field, vertex order as in the glb"},
        "files": files,
    }


_IMPORT_SCRIPT = '''"""PINNeAPPle -> Unreal Engine import, generated by pinneapple_tools.visualization.studio.unreal.

Run it inside the editor (Python Editor Script Plugin enabled):
    Window > Output Log, switch the console input to "Python", then
        exec(open(r"{script}").read())

Every step is attempted and logged on its own, so a step this Unreal version refuses does not abort the rest."""

import json
import os

import unreal

HERE = r"{here}"
CONFIG = {config}


def _log(msg):
    unreal.log("[PINNeAPPle] " + str(msg))


def _warn(msg):
    unreal.log_warning("[PINNeAPPle] " + str(msg))


def _asset_tools():
    return unreal.AssetToolsHelpers.get_asset_tools()


def _spawn(obj, location=(0.0, 0.0, 0.0)):
    """Spawn into the current level across editor versions (subsystem in 5.x, EditorLevelLibrary before)."""
    loc = unreal.Vector(*location)
    rot = unreal.Rotator(0.0, 0.0, 0.0)
    try:
        return unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_object(obj, loc, rot)
    except Exception:
        return unreal.EditorLevelLibrary.spawn_actor_from_object(obj, loc, rot)


def make_material():
    """Colour by vertex: COLOR_0 as base colour, plus a little emission so the low end of the scale still reads."""
    path = CONFIG["material"]
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        _log("material already there: " + path)
        return path
    folder, name = path.rsplit("/", 1)
    try:
        mat = _asset_tools().create_asset(name, folder, unreal.MaterialFactoryNew())
        lib = unreal.MaterialEditingLibrary
        vc = lib.create_material_expression(mat, unreal.MaterialExpressionVertexColor, -600, 0)
        lib.connect_material_property(vc, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
        mul = lib.create_material_expression(mat, unreal.MaterialExpressionMultiply, -300, 200)
        mul.set_editor_property("const_b", 0.35)
        lib.connect_material_expressions(vc, "RGB", mul, "A")
        lib.connect_material_property(mul, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        rough = lib.create_material_expression(mat, unreal.MaterialExpressionConstant, -300, 340)
        rough.set_editor_property("r", 0.42)
        lib.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
        lib.recompile_material(mat)
        unreal.EditorAssetLibrary.save_asset(path)
        _log("material built: " + path)
        return path
    except Exception as exc:
        _warn("could not build the vertex-colour material ({{}}) -- assign one by hand".format(exc))
        return None


def import_glb():
    src = os.path.join(HERE, CONFIG["files"]["geometry"])
    if not os.path.isfile(src):
        raise RuntimeError("geometry not found: " + src)
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", src)
    task.set_editor_property("destination_path", CONFIG["content_path"])
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    _asset_tools().import_asset_tasks([task])
    paths = list(task.get_editor_property("imported_object_paths") or [])
    _log("imported {{}} object(s)".format(len(paths)))
    return paths


def place(paths, material):
    """One actor per imported static mesh, the vertex-colour material on every slot."""
    actors = []
    mat = unreal.EditorAssetLibrary.load_asset(material) if material else None
    for p in paths:
        obj = unreal.EditorAssetLibrary.load_asset(p)
        if not isinstance(obj, unreal.StaticMesh):
            continue
        try:
            actor = _spawn(obj)
            if mat is not None:
                comp = actor.get_editor_property("static_mesh_component")
                for i in range(len(obj.get_editor_property("static_materials"))):
                    comp.set_material(i, mat)
            actors.append(actor)
        except Exception as exc:
            _warn("could not place {{}}: {{}}".format(p, exc))
    if not actors:
        _warn("no static mesh came out of the import -- check that the glTF/Interchange importer is enabled")
    return actors


def frame_camera(actors):
    """A cine camera on an iso direction, far enough back to hold the whole assembly."""
    if not actors:
        return None
    try:
        lo = unreal.Vector(1e12, 1e12, 1e12)
        hi = unreal.Vector(-1e12, -1e12, -1e12)
        for a in actors:
            origin, extent = a.get_actor_bounds(False)
            lo = unreal.Vector(min(lo.x, origin.x - extent.x), min(lo.y, origin.y - extent.y),
                               min(lo.z, origin.z - extent.z))
            hi = unreal.Vector(max(hi.x, origin.x + extent.x), max(hi.y, origin.y + extent.y),
                               max(hi.z, origin.z + extent.z))
        centre = (lo + hi) / 2.0
        radius = max((hi - lo).length(), 1.0)
        where = unreal.Vector(centre.x - radius * 1.6, centre.y - radius * 1.6, centre.z + radius * 0.9)
        cam = _spawn(unreal.CineCameraActor, (where.x, where.y, where.z))
        rot = unreal.MathLibrary.find_look_at_rotation(where, centre)
        cam.set_actor_rotation(rot, False)
        try:
            cam.camera_component.set_editor_property("current_focal_length", 35.0)
        except Exception:
            pass
        try:
            unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).set_level_viewport_camera_info(where, rot)
        except Exception:
            pass
        return cam
    except Exception as exc:
        _warn("camera framing skipped: {{}}".format(exc))
        return None


def cinematic_setup():
    """Black stage, one hard key light, fog, bloom, vignette: what makes the field read like a hero shot."""
    try:
        sun = _spawn(unreal.DirectionalLight, (0.0, 0.0, 800.0))
        sun.set_actor_rotation(unreal.Rotator(-38.0, 145.0, 0.0), False)
        comp = sun.get_editor_property("light_component")
        comp.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        comp.set_editor_property("intensity", 4.5)
        comp.set_editor_property("light_color", unreal.LinearColor(1.0, 0.96, 0.9, 1.0))
        _spawn(unreal.SkyLight)
        fog = _spawn(unreal.ExponentialHeightFog)
        fog.get_editor_property("component").set_editor_property("fog_density", 0.008)
    except Exception as exc:
        _warn("lighting skipped: {{}}".format(exc))
    try:
        pp = _spawn(unreal.PostProcessVolume)
        pp.set_editor_property("unbound", True)
        settings = pp.get_editor_property("settings")
        for field, value in (("override_bloom_intensity", True), ("bloom_intensity", 0.55),
                             ("override_vignette_intensity", True), ("vignette_intensity", 0.45),
                             ("override_auto_exposure_min_brightness", True), ("auto_exposure_min_brightness", 0.6),
                             ("override_auto_exposure_max_brightness", True), ("auto_exposure_max_brightness", 1.2)):
            try:
                settings.set_editor_property(field, value)
            except Exception:
                pass
        pp.set_editor_property("settings", settings)
    except Exception as exc:
        _warn("post-process skipped: {{}}".format(exc))


def main():
    _log("PINNeAPPle import from " + HERE)
    material = make_material()
    paths = import_glb()
    actors = place(paths, material)
    if CONFIG.get("cinematic"):
        cinematic_setup()
    frame_camera(actors)
    _log("done: {{}} surface(s); field \\'{{}}\\' ({{}}) as vertex colours".format(
        len(actors), CONFIG["field"]["name"], CONFIG["field"]["label"]))


main()
'''


def import_script(folder: str, config: Dict) -> str:
    """The editor-side half as text: a self-contained script for Unreal's Python Editor Script Plugin, bound to the
    ``folder`` it sits in and the ``config`` written into ``manifest.json``'s ``unreal`` block."""
    here = os.path.abspath(folder)
    return _IMPORT_SCRIPT.format(script=os.path.join(here, "PinneAPPleImport.py"), here=here,
                                 config=json.dumps(config, indent=4))


def _readme(scene: Scene, cfg: Dict) -> str:
    f = cfg["field"]
    if f.get("name"):
        head = (f"**{f['label']}** on the {f['cmap']} scale, {f['range'][0]:.4g} to {f['range'][1]:.4g}")
        geom = (f"| `geometry.glb` | geometry and PBR materials; the field in `COLOR_0` (what Unreal reads) and in "
                f"`_{f['name'].upper()}` attributes (what three.js reads) |")
    else:
        head = "No field: the scene's own PBR materials."
        geom = "| `geometry.glb` | geometry and PBR materials |"
    return f"""# {scene.title} -- Unreal Engine

Written by `pinneapple_tools.visualization.studio.unreal`. Geometry: {cfg['scene']['axes_note']}.
{head}.

| file | what it is |
|---|---|
{geom}
| `scene.usda` | the same scene in OpenUSD, for the USD Importer plugin (UE 5.4+) |
| `fields.bin` | the raw values, float32, in the order `manifest.json` describes |
| `manifest.json` | surfaces, fields, labels, colour range and the offsets into `fields.bin` |
| `PinneAPPleImport.py` | the script that runs **inside** the editor |
| `README.md` | this file |

## In Unreal

1. **Edit > Plugins**: enable the *Python Editor Script Plugin* and a glTF importer (*Interchange*, *glTF Importer*).
2. **Window > Output Log**, switch the console input to `Python` and run:

   ```python
   exec(open(r"{cfg['files']['script']}").read())
   ```

3. The script imports `geometry.glb` into `{cfg['content_path']}`, builds the vertex-colour material
   `{cfg['material']}`, places one actor per mesh in the current level and, with `cinematic`, lights the stage and
   frames a cine camera.

The editor half is not covered by this repository's test suite -- it needs a real Unreal install. A step that fails
is logged in the Output Log with the `[PINNeAPPle]` prefix and the rest of the script continues; the export half
(geometry, field colours, manifest, sidecar) is tested.
"""


def export_unreal(scene: Scene, out_dir: str, *, field: Optional[str] = None, cmap: str = "jet",
                  field_range: Optional[Sequence[float]] = None, usd: bool = True, cinematic: bool = True,
                  content_path: str = "/Game/PINNeAPPle", material: Optional[str] = None) -> str:
    """Write ``out_dir`` as a folder Unreal imports; returns the folder path.

    ``field``: the field baked into ``COLOR_0`` (default: the first the scene carries; a scene without fields still
    exports, in its PBR materials). ``cmap``/``field_range``: the colour scale, the same one the Blender render and
    the browser viewer take. ``usd``: also write ``scene.usda``. ``cinematic``: have the generated editor script set
    up lights, fog, post-process and a framed camera. ``content_path``/``material``: where that script puts the
    assets (defaults under ``/Game/PINNeAPPle``).

    Raises ``KeyError`` for an unknown field and ``ValueError`` for an empty scene or one above
    :data:`MAX_VERTICES` (decimate first with :meth:`Scene.decimate`).
    """
    if not scene.surfaces:
        raise ValueError("the scene has no surfaces to export")
    total = int(sum(len(s.vertices) for s in scene.surfaces))
    if total > MAX_VERTICES:
        raise ValueError(f"{total:,} vertices is above the {MAX_VERTICES:,} limit for one Unreal import; "
                         f"decimate the scene first: scene.decimate({MAX_VERTICES:,})")

    names = scene.field_names()
    if field is None:
        field = names[0] if names else None
    elif field not in names:
        raise KeyError(f"no surface carries the field {field!r} (fields: {names})")

    os.makedirs(out_dir, exist_ok=True)
    here = os.path.abspath(out_dir)
    files = {"geometry": "geometry.glb", "script": "PinneAPPleImport.py", "manifest": "manifest.json",
             "fields": "fields.bin", "readme": "README.md"}
    if usd:
        files["usd"] = "scene.usda"

    colors = vertex_colors(scene, field, cmap=cmap, rng=field_range) if field else None
    glb = write_glb(scene.surfaces, AXES[scene.axes], {**MATERIALS, **scene.materials},
                    generator="PINNeAPPle (Unreal)",
                    colors=None if colors is None else {k: _linear(v) for k, v in colors.items()})
    with open(os.path.join(here, files["geometry"]), "wb") as fh:
        fh.write(glb)
    if usd:
        with open(os.path.join(here, files["usd"]), "w", encoding="utf-8") as fh:
            fh.write(scene.to_usda())

    # every field of every surface, surface by surface: what a material of one's own reads back
    with open(os.path.join(here, files["fields"]), "wb") as fh:
        for s in scene.surfaces:
            for name in s.fields:
                fh.write(np.ascontiguousarray(np.asarray(s.fields[name], np.float32)).tobytes())

    rng = color_range(scene, field, field_range, cmap) if field else (0.0, 1.0)
    manifest = _manifest(scene, field, cmap, rng, files)
    cfg = {"content_path": content_path,
           "material": material or f"{content_path.rstrip('/')}/M_PINNeAPPleVertexColor",
           "cinematic": bool(cinematic), "field": manifest["field"], "files": files, "scene": manifest["scene"]}
    manifest["unreal"] = {"content_path": cfg["content_path"], "material": cfg["material"],
                          "cinematic": cfg["cinematic"], "script": files["script"]}
    with open(os.path.join(here, files["manifest"]), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(here, files["script"]), "w", encoding="utf-8") as fh:
        fh.write(import_script(here, cfg))
    with open(os.path.join(here, files["readme"]), "w", encoding="utf-8") as fh:
        fh.write(_readme(scene, cfg))
    return here
