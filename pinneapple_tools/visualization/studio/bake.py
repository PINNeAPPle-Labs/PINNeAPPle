"""Baking the procedural material shaders into texture maps.

A shader (see :mod:`.pbr`) is detail Blender renders and no other tool can read, and it costs evaluation time in every
frame. Baking it once turns the same look into three images, which the Blender renderer, the glTF export and the
browser viewer all read the same way:

    from pinneapple_tools.visualization.studio import bake_materials
    maps = bake_materials(scene, "maps/")            # {"painted_steel": {"base_color_map": ..., ...}}
    scene.materials.update({k: {**scene.materials.get(k, {}), **v} for k, v in maps.items()})

The signals come out of Cycles' own bake passes, not out of reading one node of the tree, because a shader such as
``painted_metal`` is two BSDFs mixed: a diffuse bake of it would return only the paint (a metal's diffuse albedo is
black), and the worn-through bare metal would vanish.

- **base colour** -- the DIFFUSE albedo plus the GLOSSY tint, added: the two layers of a mixed shader and the metal
  it exposes, in one image;
- **roughness** -- the ROUGHNESS pass, which Cycles evaluates on the real shader, mixes included;
- **normal** -- the tangent-space NORMAL pass, which picks up the bump nodes.

They are written against the same box-projected UVs the exporters use (:func:`.scene.box_uvs`), so a baked map lands
the same way in the render, in the .glb and in the browser.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

MAP_KEYS = ("base_color_map", "roughness_map", "normal_map")


def _target_node(mat, image):
    """The active image node a bake writes into, and the only one that looks like a bake target."""
    nt = mat.node_tree
    for n in [n for n in nt.nodes if n.bl_idname == "ShaderNodeTexImage" and n.name.startswith("bake_")]:
        nt.nodes.remove(n)
    node = nt.nodes.new("ShaderNodeTexImage")
    node.name = "bake_target"
    node.image = image
    node.select = True
    nt.nodes.active = node
    return node


def _bake_pass(bpy, bs, group, size: int, bake_type: str, extra: dict | None = None, tag: str = ""):
    """Run one bake over ``group`` and return its pixels as an (h, w, 3) float array (linear, bottom row first as
    Blender stores them -- every pass is read and written the same way, so the orientation round-trips)."""
    import numpy as np

    image = bpy.data.images.new(f"bake_{tag}", width=int(size), height=int(size), alpha=False)
    image.colorspace_settings.name = "Non-Color"           # the passes are read as numbers, not as a picture
    for o in group:
        for m in o.data.materials:
            if m is not None:
                _target_node(m, image)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in group:
        o.select_set(True)
    bpy.context.view_layer.objects.active = group[0]
    bpy.ops.object.bake(type=bake_type, **(extra or {}))
    arr = np.array(image.pixels[:], dtype=np.float32).reshape(int(size), int(size), 4)[..., :3]
    bpy.data.images.remove(image)
    return arr


def _save(bpy, arr, path: str, colorspace: str, name: str) -> str:
    """Write a float array as a PNG with ``colorspace`` (``sRGB`` re-encodes a linear base colour; ``Non-Color``
    stores the numbers as they are, which is what a roughness or a normal map needs)."""
    import numpy as np

    h, w = arr.shape[:2]
    image = bpy.data.images.new(name, width=w, height=h, alpha=False)
    image.colorspace_settings.name = colorspace
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = np.clip(arr, 0.0, 1.0)
    image.pixels = rgba.ravel().tolist()
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
    return path


def _bake_and_export(scene, out_dir: str, **kwargs):
    """The entry the bootstrap calls: bake, and return the maps together with the UV layout they were baked on, so
    the calling process can put the same layout back on its own surfaces (a headless bake happens in another
    interpreter; without this the render here would use box UVs and the map would land elsewhere)."""
    import numpy as np

    maps = bake_materials(scene, out_dir, **kwargs)
    return {"maps": maps,
            "uv": [None if s.uv is None else np.asarray(s.uv, float).ravel().tolist() for s in scene.surfaces]}


def bake_materials_headless(scene, out_dir: str, *, exe: str | None = None, timeout: float = 3600.0,
                            **kwargs) -> dict[str, dict[str, str]]:
    """:func:`bake_materials` on a machine where ``bpy`` is not importable: the scene goes to a ``.glb``, the Blender
    binary bakes it, and the map paths come back. ``kwargs`` go to :func:`bake_materials` and must be
    JSON-serialisable (``materials``, ``size``, ``samples``, ``margin``)."""
    import json
    import subprocess
    import tempfile

    from .blender import _BOOT, blender_executable
    from .scene import Scene

    if exe is None:
        exe = blender_executable()
    if exe is None:
        try:
            import bpy  # noqa: F401
        except ImportError:
            raise RuntimeError("bake_materials_headless needs Blender: set $BLENDER, put `blender` on PATH, or "
                               "`pip install bpy`") from None
        return bake_materials(scene, out_dir, **kwargs)

    work = tempfile.mkdtemp(prefix="pinneapple-bake-")
    sc = scene if isinstance(scene, Scene) else Scene.from_file(scene)
    glb = os.path.join(work, "scene.glb")
    sc.save(glb)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cfg_path, result = os.path.join(work, "bake.json"), os.path.join(work, "maps.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump({"root": root, "scene": glb, "out": os.path.abspath(out_dir), "kwargs": kwargs,
                   "materials": sc.materials, "labels": dict(getattr(sc, "labels", {}) or {}),
                   "module": "pinneapple_tools.visualization.studio.bake", "entry": "_bake_and_export",
                   "result": result}, fh)
    script = os.path.join(work, "bake_bootstrap.py")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(_BOOT)
    proc = subprocess.run([exe, "--background", "--python", script, "--", cfg_path],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not os.path.exists(result):
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-8:])
        raise RuntimeError(f"Blender failed to bake (exit {proc.returncode}):\n{tail}")
    with open(result, encoding="utf-8") as fh:
        (maps, uv) = (lambda r: (r["maps"], r.get("uv") or []))(json.load(fh))
    import numpy as np

    for srf, uvs in zip(sc.surfaces, uv, strict=False):
        if uvs:
            srf.uv = np.asarray(uvs, float).reshape(-1, 2)
    return maps


def _unwrap_group(bpy, pairs, angle: float, margin: float) -> None:
    """Unwrap the objects of one material together and write the layout back on their surfaces: unwrapped as a group
    they pack into one square, so a bake into a single image does not collide, and the surfaces carry the layout the
    bake used, which is what the render and the exports then follow."""
    import math

    import numpy as np

    objs = [o for o, _ in pairs]
    if not objs:
        return
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(float(angle)), island_margin=float(margin))
    bpy.ops.object.mode_set(mode="OBJECT")
    for o, s in pairs:
        uv = np.empty(len(o.data.loops) * 2, np.float32)
        o.data.uv_layers.active.data.foreach_get("uv", uv)
        s.uv = uv.reshape(-1, 2)


def unwrap_scene(scene, *, angle: float = 66.0, margin: float = 0.02) -> object:
    """Give every surface of ``scene`` its own UV layout (needs ``bpy``): Blender's smart projection, packed into the
    square and stored back on ``Surface.uv`` per face corner. This is what a closed or strongly curved body needs --
    the box projection of :func:`.scene.box_uvs` folds its islands onto each other -- and the returned scene carries
    the layout, so :func:`~.blender.render`, the glTF and the USD all agree with it."""
    import bpy

    from .blender import _build_objects, _check_bpy
    from .scene import Scene

    _check_bpy(bpy)
    sc = Scene.from_file(scene) if isinstance(scene, str) else scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bs = bpy.context.scene
    objs = _build_objects(bpy, bs, sc.surfaces, sc.axes, materials=getattr(sc, "materials", None), force_uv=True)
    _unwrap_group(bpy, list(zip(objs, sc.surfaces, strict=False)), angle, margin)
    return sc


def bake_materials(scene, out_dir: str, *, materials: Sequence[str] | None = None, size: int = 1024,
                   samples: int = 16, margin: int = 8, uv: str = "smart", log=None) -> dict[str, dict[str, str]]:
    """Bake every material whose entry names a ``shader`` (or the ones listed in ``materials``) into
    ``out_dir/<material>_<signal>.png``. Needs ``bpy``. Returns ``{material: {"base_color_map": path, ...}}``, ready
    to merge into ``scene.materials``: the entries keep their flat numbers, which stay the fallback and what the
    exporters use when a map is missing.

    ``uv``: "smart" (default) unwraps each material's surfaces with Blender's projection, packed as a group, and
    stores the layout on ``Surface.uv`` -- what a curved or closed body needs; "box" keeps the box projection of
    :func:`.scene.box_uvs`, which is enough for a plate, a duct or a housing and needs no unwrap pass."""
    import bpy
    import numpy as np

    from .blender import _build_objects, _check_bpy, _set_device, _smooth
    from .scene import MATERIALS, Scene

    _check_bpy(bpy)
    sc = Scene.from_file(scene) if isinstance(scene, str) else scene
    table = {**MATERIALS, **(getattr(sc, "materials", None) or {})}
    wanted = list(materials) if materials is not None else [m for m, st in table.items() if st.get("shader")]
    wanted = [m for m in wanted if table.get(m)]
    if not wanted:
        return {}
    os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bs = bpy.context.scene
    bs.render.engine = "CYCLES"
    _set_device(bpy, bs, "cpu")                            # a bake is a shading pass: the CPU is the safe choice
    bs.cycles.samples = max(1, int(samples))
    bs.render.bake.margin = int(margin)
    bs.render.bake.use_selected_to_active = False
    objs = _build_objects(bpy, bs, sc.surfaces, sc.axes, materials=getattr(sc, "materials", None), force_uv=True)
    _smooth(objs)

    by_mat: dict[str, list] = {}
    pairs_by_mat: dict[str, list] = {}
    for o, s in zip(objs, sc.surfaces, strict=False):
        if s.material in wanted:
            by_mat.setdefault(s.material, []).append(o)
            pairs_by_mat.setdefault(s.material, []).append((o, s))
    if str(uv).lower() == "smart":
        for pairs in pairs_by_mat.values():
            _unwrap_group(bpy, pairs, 66.0, 0.02)

    say = log or (lambda *_: None)
    out: dict[str, dict[str, str]] = {}
    for name, group in by_mat.items():
        diffuse = _bake_pass(bpy, bs, group, size, "DIFFUSE", {"pass_filter": {"COLOR"}}, tag=f"{name}_d")
        glossy = _bake_pass(bpy, bs, group, size, "GLOSSY", {"pass_filter": {"COLOR"}}, tag=f"{name}_g")
        base = np.clip(diffuse + glossy, 0.0, 1.0)         # paint and the metal it wears through to
        paths = {"base_color_map": os.path.join(os.path.abspath(out_dir), f"{name}_base_color.png")}
        _save(bpy, base, paths["base_color_map"], "sRGB", f"{name}_base_color")
        say(f"baked {name} base_color_map -> {paths['base_color_map']}")

        rough = _bake_pass(bpy, bs, group, size, "ROUGHNESS", tag=f"{name}_r")
        grey = np.repeat(rough.mean(axis=2)[..., None], 3, axis=2)
        paths["roughness_map"] = os.path.join(os.path.abspath(out_dir), f"{name}_roughness.png")
        _save(bpy, grey, paths["roughness_map"], "Non-Color", f"{name}_roughness")
        say(f"baked {name} roughness_map -> {paths['roughness_map']}")

        normal = _bake_pass(bpy, bs, group, size, "NORMAL", {"normal_space": "TANGENT"}, tag=f"{name}_n")
        paths["normal_map"] = os.path.join(os.path.abspath(out_dir), f"{name}_normal.png")
        _save(bpy, normal, paths["normal_map"], "Non-Color", f"{name}_normal")
        say(f"baked {name} normal_map -> {paths['normal_map']}")
        out[name] = paths
    return out
