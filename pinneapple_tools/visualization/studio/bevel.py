"""Bevel: the chamfer that a builder cannot add beside a body, because it changes the body itself.

Everything in :mod:`.detail` is a surface put next to the geometry -- a bolt stands on a face, a weld runs along a
seam. A chamfer is not that: it is the same body with its edges cut off, and doing it properly needs the mesh's
topology. So this module is the one place in the studio that edits the mesh instead of adding to it, through the bevel
modifier of the renderer that already draws the scene.

    from pinneapple_tools.visualization.studio import bevel
    sc = bevel.bevel_scene(sc, angle=35, width=0.0015, segments=2)

A machined edge is never perfectly sharp: at photographic distance the highlight along it is how a viewer tells a
machined part from an extruded one, and it is the first thing that looks wrong in a figure of a casting.

Two limits worth knowing before using it:

* the bevel rebuilds the mesh, so per-vertex fields do not survive it -- do it before mapping a result onto the body,
  or pass ``drop_fields=True`` and accept that they are gone;
* it works on welded geometry, so the surfaces are welded first (``weld``), which is what a mesh that came back from
  glTF needs: the exporter splits vertices at every seam.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

#: The values a machined edge of a part measured in metres usually wants: a 35° limit catches the edges that are
#: meant to be edges (not the facets of a curved wall), 1.5 mm is a real chamfer, and two segments is a chamfer with a
#: slight bevel rather than a razor line.
DEFAULTS = {"angle": 35.0, "width": 0.0015, "segments": 2, "weld": 1e-5}


def bevel_scene(scene, *, angle: float = DEFAULTS["angle"], width: float = DEFAULTS["width"],
                segments: int = DEFAULTS["segments"], weld: float = DEFAULTS["weld"], clamp: bool = True,
                drop_fields: bool = False, exe: str | None = None, timeout: float = 3600.0,
                log=None):
    """The same scene with every sharp edge chamfered, as a :class:`~.scene.Scene` you can render and export.

    ``angle`` is the limit in degrees: only edges sharper than it are cut, so the facets of a curved wall stay flat
    surfaces instead of turning into a bead. ``width`` is the chamfer in scene units, ``segments`` how many faces it is
    cut into. ``weld`` merges vertices closer than it before the bevel -- without that, the seams the glTF exporter
    split would bevel as open boundaries and leave a groove."""
    from .scene import Scene

    if width <= 0:
        raise ValueError("a bevel needs a width greater than zero")
    if int(segments) < 1:
        raise ValueError("a bevel needs at least one segment")
    if not 0.0 < float(angle) < 180.0:
        raise ValueError("the bevel angle is a limit in degrees, between 0 and 180")
    sc = scene if isinstance(scene, Scene) else Scene.from_file(scene)
    with_fields = [s.name for s in sc.surfaces if s.fields]
    if with_fields and not drop_fields:
        raise ValueError(f"a bevel rebuilds the mesh, so the per-vertex fields on {with_fields[:3]} would be lost: "
                         f"bevel before mapping them, or pass drop_fields=True")
    say = log or (lambda *_: None)

    if exe is None:
        from .blender import blender_executable
        exe = blender_executable()
    if exe is None:
        try:
            import bpy  # noqa: F401
        except ImportError:
            raise RuntimeError("bevel_scene needs Blender: set $BLENDER, put `blender` on PATH, or "
                               "`pip install bpy`") from None
        return _bevel_with_bpy(sc, angle=angle, width=width, segments=segments, weld=weld, clamp=clamp)

    work = tempfile.mkdtemp(prefix="pinneapple-bevel-")
    src, dst = os.path.join(work, "in.glb"), os.path.join(work, "out.glb")
    sc.save(src)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cfg_path, result = os.path.join(work, "bevel.json"), os.path.join(work, "counts.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump({"root": root, "scene": src, "out": dst, "result": result,
                   "materials": sc.materials, "labels": dict(getattr(sc, "labels", {}) or {}),
                   "module": "pinneapple_tools.visualization.studio.bevel", "entry": "_bevel_and_export",
                   "kwargs": {"angle": angle, "width": width, "segments": int(segments), "weld": weld,
                              "clamp": bool(clamp)}}, fh)
    from .blender import _BOOT
    script = os.path.join(work, "bevel_bootstrap.py")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(_BOOT)
    proc = subprocess.run([exe, "--background", "--python", script, "--", cfg_path],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not os.path.exists(result):
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-8:])
        raise RuntimeError(f"Blender failed to bevel (exit {proc.returncode}):\n{tail}")
    with open(result, encoding="utf-8") as fh:
        counts = json.load(fh)
    out = Scene.from_file(dst, axes=sc.axes)
    out.materials, out.labels, out.title = dict(sc.materials), dict(sc.labels), sc.title
    say(f"bevel: {counts['objects']} surfaces, {counts['faces_before']} -> {counts['faces_after']} faces, "
        f"width {width}, {segments} segment(s)")
    return out


def _bevel_with_bpy(scene, *, angle, width, segments, weld, clamp):
    """The in-process path, for a machine where ``bpy`` is importable."""
    import bpy

    from .blender import _build_objects
    objs = _build_objects(bpy, bpy.context.scene, scene.surfaces, scene.axes, scene.materials)
    before = _apply(objs, angle, width, segments, weld, clamp)
    from .scene import Scene
    work = tempfile.mkdtemp(prefix="pinneapple-bevel-")
    dst = os.path.join(work, "out.glb")
    for sel in bpy.context.view_layer.objects:      # only ours: this runs inside somebody else's session
        sel.select_set(sel in objs)
    bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=True)
    out = Scene.from_file(dst, axes=scene.axes)
    out.materials, out.labels, out.title = dict(scene.materials), dict(scene.labels), scene.title
    return out


def _apply(objs, angle, width, segments, weld, clamp) -> int:
    """Weld, chamfer and keep the faces: returns the triangle count afterwards."""
    import math

    import bmesh
    import bpy

    for o in objs:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        if weld > 0:
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=float(weld))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(o.data)
        bm.free()
        o.data.update()
        mod = o.modifiers.new("pinneapple_bevel", "BEVEL")
        mod.limit_method = "ANGLE"
        mod.angle_limit = math.radians(float(angle))
        mod.width = float(width)
        mod.segments = int(segments)
        mod.use_clamp_overlap = bool(clamp)
        mod.harden_normals = False
        bpy.context.view_layer.objects.active = o
        for sel in bpy.context.view_layer.objects:
            sel.select_set(sel is o)
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return sum(len(o.data.polygons) for o in objs)


def _bevel_and_export(scene, out_path: str, *, angle=DEFAULTS["angle"], width=DEFAULTS["width"],
                      segments=DEFAULTS["segments"], weld=DEFAULTS["weld"], clamp=True) -> dict:
    """The entry the bootstrap calls: build, weld, bevel, export, and report the counts."""
    import bpy

    from .blender import _build_objects

    bpy.ops.wm.read_factory_settings(use_empty=True)     # the startup cube would be exported with the scene
    objs = _build_objects(bpy, bpy.context.scene, scene.surfaces, scene.axes, scene.materials)
    faces_before = sum(len(o.data.polygons) for o in objs)
    faces_after = _apply(objs, angle, width, segments, weld, clamp)
    bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLB", use_selection=False)
    return {"objects": len(objs), "faces_before": faces_before, "faces_after": faces_after}


__all__ = ["bevel_scene", "DEFAULTS"]
