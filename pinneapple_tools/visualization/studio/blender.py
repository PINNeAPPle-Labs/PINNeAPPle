"""Blender Cycles renders of any :class:`~.scene.Scene` (``pip install bpy``, or run inside Blender).

    from pinneapple_tools.visualization.studio import Scene, render
    render(scene, "cp.jpg", field="cp")                          # skin coloured by a field, jet scale, colour bar
    render(scene, "flow.jpg", lines=True)                        # streamlines coloured by their values
    render(scene, "slice.jpg", slice="wake", view="back")        # a slice plane (NaN = transparent)
    render(scene, "hero.jpg", background="sky")                  # the materials, under a physical sky

``view``: "iso" (default), "iso_back", "iso_low", "front", "back", "side", "side_left", "top", "bottom", or a
direction (dx, dy, dz) from the centre towards the camera, in scene axes. Names assume x is the object's length
(the flow direction for a CFD scene) and z is up: "front" looks at the x-min end. The colour bar title comes from
``scene.labels[field]`` unless ``title`` is given.
"""
from __future__ import annotations

import math
import os
import tempfile
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .colormap import JET, colorbar, jet

VIEWS = {   # direction from the centre to the camera; x is the object's length / the flow direction, z up
    "iso": (-1.0, -1.1, 0.75), "iso_back": (1.0, -1.1, 0.75), "iso_low": (-1.0, -1.1, -0.4),
    "front": (-1.0, 0.0, 0.12), "back": (1.0, 0.0, 0.12), "side": (0.0, -1.0, 0.12), "side_left": (0.0, 1.0, 0.12),
    "top": (0.001, 0.0, 1.0), "bottom": (0.001, 0.0, -1.0),
}


def _lin(c):
    return tuple(v ** 2.2 for v in c)                     # Blender colour sockets are linear


def _to_blender(P: np.ndarray, axes: str) -> np.ndarray:
    """Scene axes -> Blender (z up). The glTF importer undoes AXES[axes]; this matches it for lines and slices."""
    P = np.asarray(P, float)
    if axes == "aircraft":
        return np.c_[P[:, 1], -P[:, 0], P[:, 2]]
    return P


def render(scene, out: str, *, field: Optional[str] = None, field_range: Optional[Tuple[float, float]] = None,
           lines: Union[bool, str] = False, line_range: Optional[Tuple[float, float]] = None,
           slice: Optional[str] = None, slice_range: Optional[Tuple[float, float]] = None,
           view: Union[str, Sequence[float]] = "iso", distance: float = 1.0, lens: float = 50,
           background: str = "studio", samples: int = 96, size: Tuple[int, int] = (1600, 900),
           title: Optional[str] = None, colorbar_on: bool = True, lo_txt: str = "", hi_txt: str = "",
           sun_elevation: float = 40.0, ghost: Optional[bool] = None) -> str:
    """Render ``scene`` (a Scene or a .glb path) to ``out`` (.png or .jpg). Returns ``out``.

    field: colour the surfaces by this per-vertex field (surfaces without it are grey). lines: True (all) or the
    name of a Lines set, coloured by their values. slice: name of a Slice. background: "studio" (light gradient,
    true colours) or "sky" (Nishita sky, filmic). ghost: draw the surfaces translucent (default: when a slice is
    shown)."""
    import bpy
    from mathutils import Vector

    from .scene import Scene
    if isinstance(scene, str):
        glb, axes, sc = scene, "z_up", None
    else:
        sc = scene
        axes = sc.axes
        glb = os.path.join(tempfile.mkdtemp(), "scene.glb")
        sc.save(glb)
    studio = background == "studio"

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bs = bpy.context.scene
    bs.render.engine = "CYCLES"
    bs.cycles.device = "CPU"
    bs.cycles.samples = samples
    bs.cycles.use_denoising = True
    bs.render.resolution_x, bs.render.resolution_y = size
    names = [i.identifier for i in bs.view_settings.bl_rna.properties["view_transform"].enum_items]
    bs.view_settings.view_transform = "Standard" if studio else ("AgX" if "AgX" in names else "Filmic")
    bs.view_settings.look = "None"
    bs.view_settings.exposure = 0.0 if (studio and (field or slice)) else (-0.5 if studio else -0.3)

    bpy.ops.import_scene.gltf(filepath=glb)
    objs = [o for o in bs.objects if o.type == "MESH"]
    mins = Vector((min(min((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
    maxs = Vector((max(max((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
    centre, size_l = (mins + maxs) / 2, (maxs - mins).length

    def ramp_material(name: str, attr: str, lo: float, hi: float):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        a = nt.nodes.new("ShaderNodeAttribute")
        a.attribute_name = attr
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = lo, hi
        rp = nt.nodes.new("ShaderNodeValToRGB")
        el = rp.color_ramp.elements
        el[0].position, el[0].color = 0.0, (*_lin(jet(0.0)), 1)
        el[1].position, el[1].color = 1.0, (*_lin(jet(1.0)), 1)
        for t, _ in JET[1:-1]:
            e = el.new(t)
            e.color = (*_lin(jet(t)), 1)
        nt.links.new(a.outputs["Fac"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], rp.inputs["Fac"])
        nt.links.new(rp.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(rp.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 0.35               # readable in shadow
        bsdf.inputs["Roughness"].default_value = 0.45
        return mat

    def plain(name, rgb, alpha=1.0):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*rgb, 1)
        if alpha < 1:
            b.inputs["Alpha"].default_value = alpha
            if hasattr(m, "blend_method"):
                m.blend_method = "BLEND"
        return m

    bar = None
    if field:
        attr = "_" + field.upper().lstrip("_")
        lo, hi = field_range or (sc.field_range(field) if sc else (0.0, 1.0))
        fm, grey = ramp_material(field, attr, lo, hi), plain("grey", (0.35, 0.37, 0.4))
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(fm if attr in o.data.attributes else grey)
        bar = (title or (sc.labels.get(field, field) if sc else field), lo, hi)

    show_ghost = ghost if ghost is not None else bool(slice)
    if show_ghost and not field:
        gm = plain("ghost", (0.72, 0.75, 0.78), 0.35)
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(gm)

    if lines and sc is not None:
        sets = [L for L in sc.lines if lines is True or L.name == lines]
        allv = np.concatenate([np.concatenate(L.values) for L in sets if L.values]) if any(L.values for L in sets) else None
        lo, hi = line_range or ((float(np.nanpercentile(allv, 2)), float(np.nanpercentile(allv, 98))) if allv is not None else (0, 1))
        nb = 24
        mats = []
        for j in range(nb):
            m = bpy.data.materials.new(f"line{j}")
            m.use_nodes = True
            b = m.node_tree.nodes["Principled BSDF"]
            c = _lin(jet((j + 0.5) / nb)) if allv is not None else (0.05 ** 2.2, 0.75 ** 2.2, 1.0)
            b.inputs["Base Color"].default_value = (*c, 1)
            b.inputs["Emission Color"].default_value = (*c, 1)
            b.inputs["Emission Strength"].default_value = 1.6
            mats.append(m)
        k = 0
        for L in sets:
            for li, P0 in enumerate(L.points):
                P = _to_blender(P0, axes)
                S = np.asarray(L.values[li], float) if L.values else None
                keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9 * size_l]
                P = P[keep]
                S = S[keep] if S is not None else None
                if len(P) < 2:
                    continue
                cu = bpy.data.curves.new(f"line{k}", "CURVE")
                cu.dimensions, cu.bevel_depth, cu.bevel_resolution = "3D", 0.0011 * size_l, 2
                for m in mats:
                    cu.materials.append(m)
                step = 3 if S is not None else len(P)
                for i0 in range(0, len(P) - 1, step):
                    seg = P[i0:i0 + step + 1]
                    sp = cu.splines.new("POLY")
                    sp.points.add(len(seg) - 1)
                    for i, p in enumerate(seg):
                        sp.points[i].co = (p[0], p[1], p[2], 1)
                    if S is not None:
                        t = (np.mean(S[i0:i0 + step + 1]) - lo) / ((hi - lo) or 1)
                        sp.material_index = int(min(nb - 1, max(0, t * nb)))
                ob = bpy.data.objects.new(f"line{k}", cu)
                bs.collection.objects.link(ob)
                k += 1
        if bar is None and allv is not None:
            lab = next((L.label for L in sets if L.label), "")
            bar = (title or lab or "value along the lines", lo, hi)

    if slice and sc is not None:
        s = next(x for x in sc.slices if x.name == slice)
        lo, hi = slice_range or (float(np.nanpercentile(s.grid, 2)), float(np.nanpercentile(s.grid, 98)))
        from PIL import Image
        g = np.asarray(s.grid, float)
        rgba = np.zeros(g.shape + (4,), np.uint8)
        ok = np.isfinite(g)
        t = np.clip((np.where(ok, g, lo) - lo) / ((hi - lo) or 1), 0, 1)
        lut = np.array([jet(v) for v in np.linspace(0, 1, 256)])
        rgba[..., :3] = (lut[(t * 255).astype(int)] * 255).astype(np.uint8)
        rgba[..., 3] = np.where(ok, 240, 0)
        png = os.path.join(tempfile.mkdtemp(), "slice.png")
        Image.fromarray(rgba[::-1]).save(png)                    # row 0 of the grid is at the bottom (s = 0 side)
        C = _to_blender(np.array([s.origin, s.origin + s.u, s.origin + s.u + s.v, s.origin + s.v]), axes)
        me = bpy.data.meshes.new("slice")
        me.from_pydata([tuple(p) for p in C], [], [(0, 1, 2, 3)])
        uv = me.uv_layers.new()
        for li, (u, v) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
            uv.data[li].uv = (u, v)
        m = bpy.data.materials.new("slice")
        m.use_nodes = True
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(png)
        tex.interpolation = "Closest" if min(g.shape) < 60 else "Linear"
        em = nt.nodes.new("ShaderNodeEmission")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        outn = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0
        nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
        nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
        nt.links.new(em.outputs["Emission"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], outn.inputs["Surface"])
        me.materials.append(m)
        ob = bpy.data.objects.new("slice", me)
        bs.collection.objects.link(ob)
        if bar is None:
            bar = (title or s.label or slice, lo, hi)

    world = bpy.data.worlds.new("world")
    bs.world = world
    world.use_nodes = True
    wn = world.node_tree
    if studio:
        grad, tc, mp = wn.nodes.new("ShaderNodeTexGradient"), wn.nodes.new("ShaderNodeTexCoord"), wn.nodes.new("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value = (0, math.radians(-90), 0)
        cr = wn.nodes.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].color = (0.62, 0.66, 0.72, 1)
        cr.color_ramp.elements[1].color = (0.95, 0.96, 0.98, 1)
        wn.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        wn.links.new(mp.outputs["Vector"], grad.inputs["Vector"])
        wn.links.new(grad.outputs["Fac"], cr.inputs["Fac"])
        wn.links.new(cr.outputs["Color"], wn.nodes["Background"].inputs["Color"])
        wn.nodes["Background"].inputs["Strength"].default_value = 0.9
        sun = bpy.data.lights.new("sun", "SUN")
        sun.energy, sun.angle = 2.2, math.radians(8)
        so = bpy.data.objects.new("sun", sun)
        so.rotation_euler = (math.radians(35), math.radians(-20), math.radians(30))
        bs.collection.objects.link(so)
    else:
        sky = wn.nodes.new("ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_elevation, sky.sun_rotation = math.radians(sun_elevation), math.radians(215)
        sky.altitude = 2000
        bg = wn.nodes["Background"]
        bg.inputs["Strength"].default_value = 0.12
        wn.links.new(sky.outputs["Color"], bg.inputs["Color"])

    d = _to_blender(np.asarray(VIEWS[view] if isinstance(view, str) else view, float)[None], axes)[0]
    d = d / np.linalg.norm(d)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = lens
    cam = bpy.data.objects.new("cam", cam_data)
    bs.collection.objects.link(cam)
    cam_data.sensor_fit = "AUTO"                         # the lens angle spans the longer side of the image
    t_long = math.tan(0.5 * cam_data.angle)
    t_short = t_long * min(size) / max(size)
    tx, ty = (t_long, t_short) if size[0] >= size[1] else (t_short, t_long)
    # distance at which every corner of the bounding box is inside the frame
    fwd = -d
    upw = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, upw)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c0 = np.array(centre)
    pts = []                                             # the visible geometry, not its bounding box
    for o in objs:
        mw = np.array(o.matrix_world)
        co = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        if len(co) > 4000:
            co = co[:: len(co) // 4000]
        pts.append(co @ mw[:3, :3].T + mw[:3, 3])
    if slice and sc is not None:                         # frame the whole slice plane, not only the body
        sl = next(x for x in sc.slices if x.name == slice)
        pts.append(_to_blender(np.array([sl.origin, sl.origin + sl.u, sl.origin + sl.u + sl.v, sl.origin + sl.v]), axes))
    k = np.concatenate(pts) - c0
    # aim at the middle of the projected extent, then back off until every vertex is in frame (perspective)
    shift = right * 0.5 * (np.max(k @ right) + np.min(k @ right)) + up * 0.5 * (np.max(k @ up) + np.min(k @ up))
    centre = centre + Vector(tuple(shift))
    k = k - shift
    lat = np.maximum(np.abs(k @ right) / tx, np.abs(k @ up) / ty)
    fit = float(np.max(k @ d + lat)) * 1.15 * distance
    cam.location = centre + Vector(tuple(d * fit))
    look = centre - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    bs.camera = cam

    if out.lower().endswith((".jpg", ".jpeg")):
        bs.render.image_settings.file_format = "JPEG"
        bs.render.image_settings.quality = 92
    bs.render.filepath = os.path.abspath(out)
    bpy.ops.render.render(write_still=True)
    if bar and colorbar_on:
        colorbar(out, bar[0], bar[1], bar[2], None, lo_txt, hi_txt)
    return out
