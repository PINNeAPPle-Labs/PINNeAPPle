"""Solids and a real boolean: cutting a hole instead of painting one.

:func:`~.manufacture.Body.bore` used to draw a throat, because a shadow with a bright lip is what a hole looks like at
photographic distance. This module is the other answer: a closed solid, a cutter, and a kernel that subtracts one from
the other. The difference matters the moment the camera moves -- a drawn hole has no other side, no interior wall and no
change of silhouette.

What the kernel needs, and what this module therefore insists on:

* **closed operands.** A boolean on an open shell does not fail loudly, it produces a plausible mess. :func:`is_closed`
  checks the edge manifold before anything is cut, and the functions here build closed solids by construction.
* **a kernel.** Blender's exact solver, driven headless -- the same route :mod:`.bevel` takes. There is no bundled mesh
  kernel in this package; ``trimesh`` with ``manifold3d`` does this too, and if you have it the shapes here are ordinary
  ``Surface`` objects it can eat.

    from pinneapple_tools.visualization.studio import solid
    body = solid.solid_drum([0, 0, 0], radius=0.11, height=0.09, n=96)
    cut = solid.boolean(body_scene, [solid.solid_drum([0.05, 0, 0.05], 0.02, 0.2, n=32)], op="difference")
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence

import numpy as np

from .scene import Scene, Surface


# ------------------------------------------------------------------ topology
def is_closed(surface: Surface) -> bool:
    """True when every edge of the mesh is shared by exactly two triangles -- the manifold a boolean needs."""
    F = np.asarray(surface.faces, np.int64).reshape(-1, 3)
    if len(F) == 0:
        return False
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def open_edges(surface: Surface) -> int:
    """How many edges are not shared by two triangles: zero for a closed solid, and the number to look at when a
    boolean goes wrong."""
    F = np.asarray(surface.faces, np.int64).reshape(-1, 3)
    if len(F) == 0:
        return 0
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int(np.sum(counts != 2))


# ------------------------------------------------------------------ closed primitives
def _ring(centre, axis, radius: float, n: int):
    c = np.asarray(centre, float).reshape(3)
    a = np.asarray(axis, float).reshape(3)
    a = a / (np.linalg.norm(a) or 1.0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u) or 1.0
    v = np.cross(a, u)
    ph = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return c + radius * (np.cos(ph)[:, None] * u + np.sin(ph)[:, None] * v)


def solid_drum(centre, radius: float, height: float, *, radius_top: float | None = None, n: int = 64,
               axis=(0.0, 0.0, 1.0), name: str = "drum", material: str = "grey") -> Surface:
    """A closed cylinder: wall, bottom and top. The workhorse of a real cut -- a boss, a pad, a body in the rough.

    ``radius_top`` gives the wall a taper, which is how a casting's draft lives on a solid instead of on a shell."""
    c = np.asarray(centre, float).reshape(3)
    a = np.asarray(axis, float).reshape(3)
    a = a / (np.linalg.norm(a) or 1.0)
    m = int(n)
    r1 = float(radius) if radius_top is None else float(radius_top)
    bottom, top = _ring(c, a, float(radius), m), _ring(c + a * float(height), a, r1, m)
    V = np.vstack([bottom, top])
    F: list[list[int]] = []
    for i in range(m):
        j = (i + 1) % m
        F += [[i, j, m + j], [i, m + j, m + i]]                # the wall
    for k in range(1, m - 1):                                  # the caps, fanned from their first ring vertex:
        F.append([m, m + k, m + k + 1])                        # no vertex sits on the axis, which is where a
        F.append([0, k + 1, k])                                # centred cutter would meet a degenerate fan
    return Surface(name, V, np.asarray(F, np.int64), material)


def solid_box(centre, size, *, name: str = "box", material: str = "grey") -> Surface:
    """A closed box: a cutter for a slot, a flat, a window."""
    c = np.asarray(centre, float).reshape(3)
    s = np.asarray(size, float).reshape(3) * 0.5
    V = np.array([[c[0] + sx * s[0], c[1] + sy * s[1], c[2] + sz * s[2]]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return Surface(name, V, np.asarray(F, np.int64), material)


def circle_profile(centre, radius: float, *, n: int = 48, normal=(0.0, 0.0, 1.0), phase: float = 0.0):
    """One station of a loft: a ring of ``n`` points. Any set of ``(n, 3)`` points is a profile, so a station can also be
    a rounded rectangle, a section off a drawing, or the outline of a duct."""
    return _ring(centre, normal, float(radius), int(n))


def rounded_rect_profile(centre, width: float, height: float, *, n: int = 48, roundness: float = 0.35,
                         normal=(0.0, 0.0, 1.0)):
    """A station that is not a circle: a rectangle with rounded corners as a superellipse. A duct transition between a
    round inlet and a rectangular outlet is the shape no primitive in this package gives, and it is everywhere in
    ventilation and in pump casings."""
    c = np.asarray(centre, float).reshape(3)
    a = np.asarray(normal, float).reshape(3)
    a = a / (np.linalg.norm(a) or 1.0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u) or 1.0
    v = np.cross(a, u)
    e = max(0.05, min(1.0, float(roundness)))
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    x = np.sign(ct) * np.abs(ct) ** e * 0.5 * float(width)
    y = np.sign(st) * np.abs(st) ** e * 0.5 * float(height)
    return c + x[:, None] * u + y[:, None] * v


def loft(stations, *, caps: bool = True, name: str = "loft", material: str = "grey") -> Surface:
    """The skin through a sequence of profiles: an organic body out of sections, which is how a duct transition, a
    volute or a pump casing is drawn -- and the shape no primitive here can make.

    Every station is an ``(n, 3)`` array with the same ``n``. The interpolation between stations is linear, which is
    honest: a station is a measurement, and a spline through measurements invents the shape in between. Put the stations
    where the geometry bends and linear is invisible; ``caps`` closes the two ends so the result is a solid a boolean
    can cut."""
    S = [np.asarray(s, float).reshape(-1, 3) for s in stations]
    if len(S) < 2:
        raise ValueError("a loft needs at least two stations")
    m = len(S[0])
    if any(len(s) != m for s in S):
        raise ValueError("every station needs the same number of points, or there is nothing to loft between")
    V = np.vstack(S)
    F: list[list[int]] = []
    for k in range(len(S) - 1):
        o, p = k * m, (k + 1) * m
        for i in range(m):
            j = (i + 1) % m
            F += [[o + i, o + j, p + j], [o + i, p + j, p + i]]
    if caps:
        for base, flip in ((0, True), ((len(S) - 1) * m, False)):
            for k in range(1, m - 1):
                F.append([base, base + k + 1, base + k] if flip else [base, base + k, base + k + 1])
    return Surface(name, V, np.asarray(F, np.int64), material)


def cutter_cylinder(a, b, radius: float, *, n: int = 48, name: str = "cutter", material: str = "cutter") -> Surface:
    """The tool of a bore: a closed cylinder from ``a`` to ``b``, long enough to pass through the part."""
    A, B = np.asarray(a, float).reshape(3), np.asarray(b, float).reshape(3)
    d = B - A
    L = float(np.linalg.norm(d))
    if L <= 0:
        raise ValueError("a cutter needs a length")
    return solid_drum(A, float(radius), L, n=n, axis=d / L, name=name, material=material)


# ------------------------------------------------------------------ the kernel
def boolean(scene, cutters: Sequence[Surface], *, op: str = "difference", exe: str | None = None,
            timeout: float = 1800.0, log=None) -> Scene:
    """``scene`` with every closed ``cutter`` combined into it (``difference``, ``union`` or ``intersect``).

    Blender's exact solver does the work, headless, so the result is a real mesh: an opening with an interior wall and a
    silhouette that changes with the camera. Open shells are refused before anything runs -- a boolean on them does not
    fail loudly, it makes a mess that looks like a result."""
    if op not in ("difference", "union", "intersect"):
        raise ValueError(f"op={op!r}: difference, union or intersect")
    sc = scene if isinstance(scene, Scene) else Scene.from_file(scene)
    cutters = list(cutters)
    if not cutters:
        raise ValueError("a boolean needs at least one cutter")
    for c in cutters:
        if not is_closed(c):
            raise ValueError(f"cutter {c.name!r} is not closed ({open_edges(c)} open edges): a boolean on an open "
                             f"shell makes a mess, not a hole")
    open_targets = [s.name for s in sc.surfaces if not is_closed(s)]
    if open_targets:
        raise ValueError(f"the body has open shells {open_targets[:3]}: close them first (a boolean needs solids on "
                         f"both sides)")
    if exe is None:
        from .blender import blender_executable
        exe = blender_executable()
    if exe is None:
        raise RuntimeError("boolean needs Blender: set $BLENDER, put `blender` on PATH, or `pip install bpy`")

    work = tempfile.mkdtemp(prefix="pinneapple-bool-")
    src, dst = os.path.join(work, "in.glb"), os.path.join(work, "out.glb")
    sc.save(src)
    cfg_path, result = os.path.join(work, "bool.json"), os.path.join(work, "counts.json")
    payload = {"cutters": [{"name": c.name, "vertices": np.asarray(c.vertices, float).tolist(),
                            "faces": np.asarray(c.faces, np.int64).tolist()} for c in cutters],
               "op": op}
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump({"root": os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                       os.path.abspath(__file__))))),
                   "scene": src, "out": dst, "result": result, "materials": sc.materials,
                   "labels": dict(getattr(sc, "labels", {}) or {}),
                   "module": "pinneapple_tools.visualization.studio.solid", "entry": "_boolean_and_export",
                   "kwargs": payload}, fh)
    from .blender import _BOOT
    script = os.path.join(work, "bool_bootstrap.py")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(_BOOT)
    proc = subprocess.run([exe, "--background", "--python", script, "--", cfg_path],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not os.path.exists(result):
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-8:])
        raise RuntimeError(f"Blender failed to run the boolean (exit {proc.returncode}):\n{tail}")
    with open(result, encoding="utf-8") as fh:
        counts = json.load(fh)
    out = Scene.from_file(dst, axes=sc.axes)
    out.materials, out.labels, out.title = dict(sc.materials), dict(sc.labels), sc.title
    say = log or (lambda *_: None)
    say(f"boolean {op}: {counts['faces_before']} -> {counts['faces_after']} faces, {len(cutters)} cutter(s)")
    out.boolean_counts = counts
    return out


def _boolean_and_export(scene, out_path: str, *, cutters, op="difference") -> dict:
    """The entry the bootstrap calls: build the body and the tools, cut, drop the tools, export."""
    import bmesh
    import bpy

    from .blender import _build_objects

    bpy.ops.wm.read_factory_settings(use_empty=True)
    targets = _build_objects(bpy, bpy.context.scene, scene.surfaces, scene.axes, scene.materials)
    tools = []
    for c in cutters:
        surf = Surface(c["name"], np.asarray(c["vertices"], float), np.asarray(c["faces"], np.int64), "cutter")
        tools += _build_objects(bpy, bpy.context.scene, [surf], scene.axes, {"cutter": {"color": [1, 0, 1]}})
    for o in targets + tools:                       # weld and orient: an exact boolean wants a clean manifold
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-9)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(o.data)
        bm.free()
        o.data.update()
    faces_before = sum(len(o.data.polygons) for o in targets)
    for t in targets:
        bpy.context.view_layer.objects.active = t
        for tool in tools:
            mod = t.modifiers.new("pinneapple_boolean", "BOOLEAN")
            mod.operation = op.upper() if op != "intersect" else "INTERSECT"
            mod.object = tool
            mod.solver = "EXACT"
            bpy.ops.object.modifier_apply(modifier=mod.name)
    for tool in tools:
        bpy.data.objects.remove(tool, do_unlink=True)
    faces_after = sum(len(o.data.polygons) for o in targets)
    bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLB", use_selection=False)
    return {"faces_before": faces_before, "faces_after": faces_after, "objects": len(targets)}


__all__ = ["is_closed", "open_edges", "solid_drum", "solid_box", "cutter_cylinder", "boolean",
           "loft", "circle_profile", "rounded_rect_profile"]
