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

from .colormap import colorbar, jet

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


MIN_BPY = (4, 2)          # oldest Blender-as-a-module release these renderers are tested on (4.2 LTS and 5.x)


def _check_bpy(bpy) -> None:
    v = tuple(bpy.app.version[:2])
    if v < MIN_BPY:
        raise RuntimeError(f"pinneapple studio renders need Blender / bpy >= {MIN_BPY[0]}.{MIN_BPY[1]} "
                           f"(found {bpy.app.version_string}); pip install 'bpy>=4.2'")


def _set_input(node, names, value) -> bool:
    """Set the first input of ``node`` among ``names`` that exists (socket names differ between Blender versions,
    e.g. "Coat Weight" (4.x/5.x) and "Clearcoat" (3.x)). Returns whether one was found."""
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def _table_material(bpy, name: str, st: dict):
    """Principled BSDF material from an entry of ``MATERIALS`` (sRGB colour -> linear unless ``"linear"``, metallic,
    roughness, clearcoat, transmission, alpha, emissive), as the glTF export of the same table describes it."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    col = [c if st.get("linear") else c ** 2.2 for c in st.get("color", (0.62, 0.65, 0.69))]
    b.inputs["Base Color"].default_value = (*col, 1.0)
    b.inputs["Metallic"].default_value = float(st.get("metallic", 0.0))
    b.inputs["Roughness"].default_value = float(st.get("roughness", 0.5))
    if st.get("clearcoat"):
        _set_input(b, ("Coat Weight", "Clearcoat"), float(st["clearcoat"]))
        _set_input(b, ("Coat Roughness", "Clearcoat Roughness"), 0.05)
    if st.get("transmission"):
        _set_input(b, ("Transmission Weight", "Transmission"), float(st["transmission"]))
    if st.get("alpha", 1.0) < 1.0:
        b.inputs["Alpha"].default_value = float(st["alpha"])
        if hasattr(m, "blend_method"):
            m.blend_method = "BLEND"
    if st.get("emissive"):
        _set_input(b, ("Emission Color", "Emission"), (*[c ** 2.2 for c in st["emissive"]], 1.0))
        _set_input(b, ("Emission Strength",), 4.0)
    if st.get("double_sided") is False and hasattr(m, "use_backface_culling"):
        m.use_backface_culling = True
    return m


def _build_objects(bpy, bs, surfaces, axes: str, materials: dict | None = None, names: Sequence[str] | None = None):
    """One mesh object per surface, built directly (no glTF import): vertices in Blender axes, triangles, the
    per-vertex fields as FLOAT point attributes ``_NAME`` (upper case, as the glTF export names them), a material per
    material name from ``MATERIALS`` + ``materials``. Returns the objects in surface order."""
    from .scene import MATERIALS
    table = {**MATERIALS, **(materials or {})}
    B = blender_axes(axes)
    mats, out = {}, []
    for i, s in enumerate(surfaces):
        V = (np.asarray(s.vertices, float) @ B.T).astype(np.float32)
        F = np.asarray(s.faces, np.int32).reshape(-1, 3)
        nm = names[i] if names is not None else s.name
        me = bpy.data.meshes.new(nm)
        me.vertices.add(len(V))
        me.vertices.foreach_set("co", V.ravel())
        me.loops.add(3 * len(F))
        me.loops.foreach_set("vertex_index", F.ravel())
        me.polygons.add(len(F))
        me.polygons.foreach_set("loop_start", np.arange(0, 3 * len(F), 3, dtype=np.int32))
        try:                                                   # read-only (derived from loop_start) on newer Blender
            me.polygons.foreach_set("loop_total", np.full(len(F), 3, np.int32))
        except (AttributeError, TypeError, RuntimeError):
            pass
        me.update(calc_edges=True)
        for fname, vals in getattr(s, "fields", {}).items():
            a = np.asarray(vals, np.float32).ravel()
            if len(a) != len(V):
                continue
            at = me.attributes.new("_" + fname.upper().lstrip("_"), "FLOAT", "POINT")
            at.data.foreach_set("value", a)
        if s.material not in mats:
            mats[s.material] = _table_material(bpy, s.material, table.get(s.material, MATERIALS["grey"]))
        me.materials.append(mats[s.material])
        ob = bpy.data.objects.new(nm, me)
        bs.collection.objects.link(ob)
        out.append(ob)
    return out


def render(scene, out: str, *, field: Optional[str] = None, field_range: Optional[Tuple[float, float]] = None,
           lines: Union[bool, str] = False, line_range: Optional[Tuple[float, float]] = None,
           slice: Optional[str] = None, slice_range: Optional[Tuple[float, float]] = None,
           view: Union[str, Sequence[float]] = "iso", distance: float = 1.0, lens: float = 50,
           background: str = "studio", samples: int = 96, size: Tuple[int, int] = (1600, 900),
           title: Optional[str] = None, colorbar_on: bool = True, lo_txt: str = "", hi_txt: str = "",
           sun_elevation: float = 40.0, ghost: Optional[bool] = None,
           surface_fields: dict | None = None) -> str:
    """Render ``scene`` (a Scene or a .glb path) to ``out`` (.png or .jpg). Returns ``out``.

    field: colour the surfaces by this per-vertex field (surfaces without it are grey). lines: True (all) or the
    name of a Lines set, coloured by their values. slice: name of a Slice. background: "studio" (light gradient,
    true colours), "sky" (Nishita sky, filmic) or a lunar preset of :mod:`.lighting` ("lunar_south_pole",
    "lunar_low_sun", "lunar_day": black sky, one Sun lamp with the Sun's angular size). ghost: draw the surfaces translucent (default: when a slice is
    shown). surface_fields: {surface name: (field, lo, hi, cmap, title)} colours that surface by its own field
    on ``cmap`` (one of :data:`.compose.CMAPS`: "jet", or a diverging "rdbu" / "coolwarm" / "bwr") and adds a
    second colour bar (bottom left) for it."""
    import bpy
    from mathutils import Vector

    from .scene import Scene
    _check_bpy(bpy)
    sc = Scene.from_file(scene) if isinstance(scene, str) else scene
    axes = sc.axes
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

    objs = _build_objects(bpy, bs, sc.surfaces, axes, getattr(sc, "materials", None))
    _smooth(objs)
    mins = Vector((min(min((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
    maxs = Vector((max(max((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
    centre, size_l = (mins + maxs) / 2, (maxs - mins).length

    def ramp_material(name: str, attr: str, lo: float, hi: float, cmap: str = "jet"):
        return _ramp_material(bpy, name, attr, lo, hi, cmap)

    def plain(name, rgb, alpha=1.0):
        return _plain(bpy, name, rgb, alpha)

    bar = None
    if field:
        attr = "_" + field.upper().lstrip("_")
        lo, hi = field_range or (sc.field_range(field) if sc else (0.0, 1.0))
        fm, grey = ramp_material(field, attr, lo, hi), plain("grey", (0.35, 0.37, 0.4))
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(fm if attr in o.data.attributes else grey)
        bar = (title or (sc.labels.get(field, field) if sc else field), lo, hi)

    from .compose import stops
    bar2 = None
    for sname, (sfield, slo, shi, scale, stitle) in (surface_fields or {}).items():
        sattr = "_" + sfield.upper().lstrip("_")
        smat = ramp_material(f"{sname}_{sfield}", sattr, slo, shi, scale)
        for o in objs:
            if o.name.split(".")[0] == sname and sattr in o.data.attributes:
                o.data.materials.clear()
                o.data.materials.append(smat)
        bar2 = (stitle or sfield, slo, shi, stops(scale))

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

    if background.startswith("lunar"):                  # black sky, one hard Sun (lighting.PRESETS)
        from .lighting import blender_world
        blender_world(background, bpy)
    else:
        _world(bpy, bs, studio, sun_elevation)

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
    cam_data.clip_start, cam_data.clip_end = fit * 1e-3, fit * 1e3     # small models: the default 0.1 m cuts them
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
    if bar2 and colorbar_on:
        colorbar(out, bar2[0], bar2[1], bar2[2], None, corner="bottom-left", stops=bar2[3])
    return out


def _ramp_material(bpy, name: str, attr: str, lo: float, hi: float, cmap: str = "jet"):
    """Principled material coloured on ``cmap`` (jet, coolwarm, bwr) by the mesh attribute ``attr`` mapped from
    lo..hi."""
    from .compose import stops
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
    st = stops(cmap)
    rgb = lambda c: (*_lin(tuple(v / 255 for v in c)), 1)       # noqa: E731
    el[0].position, el[0].color = 0.0, rgb(st[0][1])
    el[1].position, el[1].color = 1.0, rgb(st[-1][1])
    for t, c in st[1:-1]:
        e = el.new(t)
        e.color = rgb(c)
    nt.links.new(a.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], rp.inputs["Fac"])
    nt.links.new(rp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(rp.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.35               # readable in shadow
    bsdf.inputs["Roughness"].default_value = 0.45
    return mat


def _plain(bpy, name, rgb, alpha=1.0):
    """Principled material of one linear colour (translucent when alpha < 1)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
        if hasattr(m, "blend_method"):
            m.blend_method = "BLEND"
    return m


def _world(bpy, bs, studio, sun_elevation):
    """Studio gradient or Nishita sky world with a Sun lamp."""
    import math
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
        kinds = [i.identifier for i in sky.bl_rna.properties["sky_type"].enum_items]
        sky.sky_type = "NISHITA" if "NISHITA" in kinds else "MULTIPLE_SCATTERING"        # renamed in Blender 5
        sky.sun_elevation, sky.sun_rotation = math.radians(sun_elevation), math.radians(215)
        sky.altitude = 2000
        bg = wn.nodes["Background"]
        bg.inputs["Strength"].default_value = 0.12
        wn.links.new(sky.outputs["Color"], bg.inputs["Color"])


# ---------------------------------------------------------------------- animations
_GLTF_TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])     # glTF y up -> Blender z up


def blender_axes(axes: str) -> np.ndarray:
    """Rotation matrix scene axes -> Blender coordinates (what the glTF export + import does to the surfaces)."""
    from .scene import AXES
    return _GLTF_TO_BLENDER @ AXES[axes]


def _arrow_meshes(bpy, n: int = 12):
    """Unit cylinder (radius 1, z from 0 to 1) and unit cone (base radius 1 at z = 0, tip at z = 1)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[np.cos(a), np.sin(a)]
    cyl_v = [(x, y, 0.0) for x, y in ring] + [(x, y, 1.0) for x, y in ring] + [(0, 0, 0), (0, 0, 1)]
    cyl_f = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    cyl_f += [(2 * n, (i + 1) % n, i) for i in range(n)] + [(2 * n + 1, n + i, n + (i + 1) % n) for i in range(n)]
    cone_v = [(x, y, 0.0) for x, y in ring] + [(0, 0, 1), (0, 0, 0)]
    cone_f = [(i, (i + 1) % n, n) for i in range(n)] + [(n + 1, (i + 1) % n, i) for i in range(n)]
    out = []
    for name, V, F in (("arrow_shaft", cyl_v, cyl_f), ("arrow_head", cone_v, cone_f)):
        me = bpy.data.meshes.new(name)
        me.from_pydata(V, [], F)
        me.update()
        out.append(me)
    return out


def _hud_overlay(path: str, lines: Sequence[str]) -> None:
    """Burn the head-up lines (first one bold) into the top-left corner of the image at ``path``."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(path).convert("RGB")
    W, H = im.size
    fs = max(8, int(round(H * 0.032)))
    try:
        f = ImageFont.truetype("DejaVuSansMono.ttf", fs)
        fb = ImageFont.truetype("DejaVuSansMono-Bold.ttf", fs)
    except OSError:
        f = fb = ImageFont.load_default()
    d = ImageDraw.Draw(im, "RGBA")
    pad, lh = max(3, fs // 2), int(fs * 1.35)
    w = max(d.textlength(t, font=fb if i == 0 else f) for i, t in enumerate(lines))
    d.rounded_rectangle([pad, pad, pad * 3 + w, pad * 3 + lh * len(lines) - (lh - fs)], radius=pad,
                        fill=(15, 23, 42, 190))
    for i, t in enumerate(lines):
        d.text((pad * 2, pad * 2 + i * lh), t, fill=(248, 250, 252), font=fb if i == 0 else f)
    im.save(path)


def _encode(pngs: Sequence[str], out: str, fps: float, gif: bool = False, gif_fps: float = 12,
            gif_width: int = 480, crf: int = 27) -> str:
    """MP4 (H.264, yuv420p, even size) or GIF (palette) of a PNG sequence with the ffmpeg binary; without it, the
    MP4 goes through Blender's encoder (:func:`.particles.encode_mp4`) and the GIF through PIL."""
    import shutil
    import subprocess
    exe = shutil.which("ffmpeg")
    if exe is None:
        if gif:
            from PIL import Image
            ims = [Image.open(p).convert("RGB") for p in pngs]
            ims = [im.resize((gif_width, int(im.height * gif_width / im.width))) if im.width > gif_width else im
                   for im in ims]
            ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
            return out
        from .particles import encode_mp4
        return encode_mp4(list(pngs), out, fps=int(round(fps)))
    import tempfile as _tf
    seq = _tf.mkdtemp()                                         # numbered links: ffmpeg's image2 pattern
    for i, p in enumerate(pngs):
        os.symlink(os.path.abspath(p), os.path.join(seq, f"f_{i:06d}.png"))
    cmd = [exe, "-y", "-loglevel", "error", "-framerate", f"{fps}", "-i", os.path.join(seq, "f_%06d.png")]
    if gif:
        vf = (f"fps={min(fps, gif_fps)},scale='min({gif_width},iw)':-2:flags=lanczos,split[a][b];"
              "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
        cmd += ["-vf", vf, "-loop", "0", out]
    else:
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-r", f"{fps}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", str(crf), "-preset", "slow", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    shutil.rmtree(seq, ignore_errors=True)
    return out


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[j:j + 2], 16) / 255 for j in (0, 2, 4))


def _wisp_material(bpy, name: str, rgb_lin, alpha: float = 1.0, strength: float = 1.2):
    """Emission mixed with transparency: a glowing tube that reads as a soft wisp when ``alpha`` < 1."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em, tr = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeBsdfTransparent")
    mix, out = nt.nodes.new("ShaderNodeMixShader"), nt.nodes.new("ShaderNodeOutputMaterial")
    em.inputs["Color"].default_value = (*rgb_lin, 1)
    em.inputs["Strength"].default_value = strength
    mix.inputs["Fac"].default_value = float(min(1.0, max(0.0, alpha)))
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    if alpha < 1 and hasattr(m, "blend_method"):
        m.blend_method = "BLEND"
    return m


def _bucket_materials(bpy, prefix: str, cmap: str, alpha: float, nb: int = 24, strength: float = 1.2):
    """``nb`` wisp materials evenly over the colour map (a tube's segments pick theirs by value)."""
    from .compose import cmap_color
    return [_wisp_material(bpy, f"{prefix}{j}", _lin(cmap_color(cmap, (j + 0.5) / nb)), alpha, strength)
            for j in range(nb)]


def _fill_curve(cu, polylines, values, lo: float, hi: float, nb: int, step: int = 1):
    """Replace the splines of curve ``cu`` by ``polylines`` (lists of (n, 3) points); with ``values`` each run of
    ``step`` segments is its own spline whose material index follows the mean value on the colour map."""
    cu.splines.clear()
    for li, P in enumerate(polylines):
        S = None if values is None else values[li]
        if len(P) < 2:
            continue
        st = step if S is not None else len(P)
        for i0 in range(0, len(P) - 1, st):
            seg = P[i0:i0 + st + 1]
            sp = cu.splines.new("POLY")
            sp.points.add(len(seg) - 1)
            sp.points.foreach_set("co", np.c_[seg, np.ones(len(seg))].astype(np.float32).ravel())
            if S is not None:
                t = (float(np.nanmean(S[i0:i0 + st + 1])) - lo) / ((hi - lo) or 1)
                sp.material_index = int(min(nb - 1, max(0, (t if np.isfinite(t) else 0.0) * nb)))


def _sparkline(path: str, t, y, k: int, label: str, unit: str = "", fmt: str = "{:.2f}") -> None:
    """Inset chart (bottom-left) of the series y(t) drawn up to frame k, with a light axis, the full time range,
    the series' min/max and the current value."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(path).convert("RGB")
    W, H = im.size
    fs = max(9, int(round(H * 0.03)))
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", fs)
        fb = ImageFont.truetype("DejaVuSans-Bold.ttf", fs)
    except OSError:
        f = fb = ImageFont.load_default()
    d = ImageDraw.Draw(im, "RGBA")
    pad = max(4, fs // 2)
    bw, bh = int(0.30 * W), int(0.24 * H)
    x0, y0 = pad, H - bh - pad
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=pad, fill=(15, 23, 42, 170))
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(y)
    lo, hi = (float(np.nanmin(y)), float(np.nanmax(y))) if ok.any() else (0.0, 1.0)
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    cur = y[k]
    title = f"{label}: {fmt.format(cur) if np.isfinite(cur) else '-'}{(' ' + unit) if unit else ''}"
    d.text((x0 + pad, y0 + pad // 2), title, fill=(248, 250, 252), font=fb)
    px0, px1 = x0 + pad + int(2.6 * fs), x0 + bw - pad
    py0, py1 = y0 + pad + int(1.5 * fs), y0 + bh - pad - fs
    X = lambda v: px0 + (v - t[0]) / ((t[-1] - t[0]) or 1) * (px1 - px0)   # noqa: E731
    Y = lambda v: py1 - (v - lo) / (hi - lo) * (py1 - py0)                 # noqa: E731
    d.line([(px0, py1), (px1, py1)], fill=(203, 213, 225, 150), width=1)
    d.line([(px0, py0), (px0, py1)], fill=(203, 213, 225, 150), width=1)
    if lo < 0 < hi:
        d.line([(px0, Y(0.0)), (px1, Y(0.0))], fill=(203, 213, 225, 90), width=1)
    for v in (lo, hi):
        s = fmt.format(v)
        d.text((px0 - d.textlength(s, font=f) - 3, Y(v) - fs // 2), s, fill=(203, 213, 225), font=f)
    d.text((px0, py1 + 1), f"{t[0]:.0f}", fill=(203, 213, 225), font=f)
    s = f"{t[-1]:.0f} s"
    d.text((px1 - d.textlength(s, font=f), py1 + 1), s, fill=(203, 213, 225), font=f)
    pts = [(X(a), Y(b)) for a, b in zip(t[:k + 1], y[:k + 1], strict=False) if np.isfinite(b)]
    if len(pts) > 1:
        d.line(pts, fill=(251, 191, 36), width=max(1, fs // 7))
    if pts:
        r = max(2, fs // 5)
        d.ellipse([pts[-1][0] - r, pts[-1][1] - r, pts[-1][0] + r, pts[-1][1] + r], fill=(251, 191, 36))
    im.save(path)


def _haze(bpy, bs, lo, hi, density: float, blob: float, clear_obj=None, clear: tuple[float, float] | None = None):
    """A box from ``lo`` to ``hi`` filled with a Principled Volume whose density is ``density`` modulated by a
    Noise texture (blobs of about ``blob``): cloud-like haze. ``clear_obj`` / ``clear=(r0, r1)``: no haze within r0
    of that object (e.g. the Empty of the tracked body), full haze beyond r1, so the body, the camera and the lines
    near them stay crisp. Volume bounces are kept at zero (cheap)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    me = bpy.data.meshes.new("haze")
    me.from_pydata(V, [], F)
    m = bpy.data.materials.new("haze")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    mr = nt.nodes.new("ShaderNodeMapRange")
    noise.inputs["Scale"].default_value = 1.0 / max(blob, 1e-9)
    noise.inputs["Detail"].default_value = 3.0
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.35, 0.75
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.15 * density, density
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
    if clear_obj is not None and clear:
        tco = nt.nodes.new("ShaderNodeTexCoord")
        tco.object = clear_obj
        ln = nt.nodes.new("ShaderNodeVectorMath")
        ln.operation = "LENGTH"
        cz = nt.nodes.new("ShaderNodeMapRange")
        cz.inputs["From Min"].default_value, cz.inputs["From Max"].default_value = clear
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        nt.links.new(tco.outputs["Object"], ln.inputs[0])
        nt.links.new(ln.outputs["Value"], cz.inputs["Value"])
        nt.links.new(mr.outputs["Result"], mul.inputs[0])
        nt.links.new(cz.outputs["Result"], mul.inputs[1])
        nt.links.new(mul.outputs["Value"], vol.inputs["Density"])
    else:
        nt.links.new(mr.outputs["Result"], vol.inputs["Density"])
    vol.inputs["Color"].default_value = (0.92, 0.94, 0.97, 1)
    vol.inputs["Anisotropy"].default_value = 0.3
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])
    me.materials.append(m)
    bs.collection.objects.link(bpy.data.objects.new("haze", me))
    cy = bs.cycles
    for attr, v in (("volume_bounces", 0), ("volume_step_rate", 4.0), ("volume_max_steps", 64)):
        if hasattr(cy, attr):
            setattr(cy, attr, v)


COLORBAR_KINDS = ("field", "vectors", "trails", "lines")


def _smooth(objs, angle_deg: float = 35.0) -> None:
    """Smooth shading with sharp edges above ``angle_deg`` (the glTF normals are averaged over shared vertices, so
    they are dropped: creases such as trailing edges stay sharp, curved skins shade smoothly)."""
    for o in objs:
        me = o.data
        if "custom_normal" in me.attributes:
            me.attributes.remove(me.attributes["custom_normal"])
        try:
            me.shade_smooth()
            me.set_sharp_from_angle(angle=math.radians(angle_deg))
        except (AttributeError, TypeError, RuntimeError):
            for p in me.polygons:
                p.use_smooth = True


def _ground_material(bpy, m, kind: str) -> None:
    """Rebuild material ``m`` as procedural "asphalt" (patchy dark grey, rough, fine bump) or "grass" (patches of
    greens, bump), in metres of object space."""
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out, b = nt.nodes.new("ShaderNodeOutputMaterial"), nt.nodes.new("ShaderNodeBsdfPrincipled")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    big, fine = nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeTexNoise")
    ramp, rough, bump = nt.nodes.new("ShaderNodeValToRGB"), nt.nodes.new("ShaderNodeMapRange"), nt.nodes.new("ShaderNodeBump")
    for n in (big, fine):
        nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
    if kind == "asphalt":
        big.inputs["Scale"].default_value, big.inputs["Detail"].default_value = 0.06, 6.0
        fine.inputs["Scale"].default_value, fine.inputs["Detail"].default_value = 2.5, 8.0
        c0, c1 = (0.030, 0.031, 0.033), (0.085, 0.086, 0.090)
        r0, r1, strength = 0.72, 0.95, 0.12
    else:
        big.inputs["Scale"].default_value, big.inputs["Detail"].default_value = 0.012, 5.0
        fine.inputs["Scale"].default_value, fine.inputs["Detail"].default_value = 0.8, 6.0
        c0, c1 = (0.035, 0.075, 0.018), (0.11, 0.16, 0.045)
        r0, r1, strength = 0.85, 1.0, 0.3
    el = ramp.color_ramp.elements
    el[0].position, el[0].color = 0.3, (*c0, 1)
    el[1].position, el[1].color = 0.7, (*c1, 1)
    nt.links.new(big.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = r0, r1
    nt.links.new(fine.outputs["Fac"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], b.inputs["Roughness"])
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = 0.02
    nt.links.new(fine.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])


def _tube(P, r: float, nseg: int = 6):
    """Vertices and quads of a tube of radius r along the polyline P (n, 3): ``nseg`` vertices per point."""
    P = np.asarray(P, float)
    T = np.gradient(P, axis=0) if len(P) > 1 else np.array([[1.0, 0, 0]])
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-30
    up = np.where(np.abs(T[:, 2:3]) > 0.9, [[1.0, 0, 0]], [[0, 0, 1.0]])
    N = np.cross(T, up)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-30
    Bn = np.cross(T, N)
    a = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    V = P[:, None] + r * (np.cos(a)[None, :, None] * N[:, None] + np.sin(a)[None, :, None] * Bn[:, None])
    i, j = np.meshgrid(np.arange(len(P) - 1), np.arange(nseg), indexing="ij")
    F = np.stack([i * nseg + j, i * nseg + (j + 1) % nseg, (i + 1) * nseg + (j + 1) % nseg, (i + 1) * nseg + j], -1)
    return V.reshape(-1, 3), F.reshape(-1, 4)


def _trail_material(bpy, name: str, rgb_lin, alpha: float, cmap: str | None = None, lo: float = 0.0,
                    hi: float = 1.0, strength: float = 1.5):
    """Emission (a colour, or ``cmap`` over the point attribute "tval" from lo..hi) mixed with transparency by
    ``alpha`` x the point attribute "tfade": a glowing trail that fades towards its tail."""
    from .compose import stops
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em, tr = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeBsdfTransparent")
    mix, out = nt.nodes.new("ShaderNodeMixShader"), nt.nodes.new("ShaderNodeOutputMaterial")
    em.inputs["Strength"].default_value = strength
    if cmap:
        av, mr, rp = nt.nodes.new("ShaderNodeAttribute"), nt.nodes.new("ShaderNodeMapRange"), nt.nodes.new("ShaderNodeValToRGB")
        av.attribute_type, av.attribute_name = "GEOMETRY", "tval"
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = lo, hi
        st = stops(cmap)
        el = rp.color_ramp.elements
        rgb = lambda c: (*_lin(tuple(v / 255 for v in c)), 1)       # noqa: E731
        el[0].position, el[0].color = 0.0, rgb(st[0][1])
        el[1].position, el[1].color = 1.0, rgb(st[-1][1])
        for t, c in st[1:-1]:
            el.new(t).color = rgb(c)
        nt.links.new(av.outputs["Fac"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], rp.inputs["Fac"])
        nt.links.new(rp.outputs["Color"], em.inputs["Color"])
    else:
        em.inputs["Color"].default_value = (*rgb_lin, 1)
    af, mul = nt.nodes.new("ShaderNodeAttribute"), nt.nodes.new("ShaderNodeMath")
    af.attribute_type, af.attribute_name = "GEOMETRY", "tfade"
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = float(alpha)
    nt.links.new(af.outputs["Fac"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    if hasattr(m, "blend_method"):
        m.blend_method = "BLEND"
    return m


def _set_tube(me, P, vals, fade, r: float, nseg: int = 6) -> None:
    """Rebuild mesh ``me`` as a tube along P with the point attributes "tval" and "tfade" (one value per point)."""
    me.clear_geometry()
    if len(P) < 2:
        return
    V, F = _tube(P, r, nseg)
    me.from_pydata(V.tolist(), [], F.tolist())
    for nm, a in (("tval", vals), ("tfade", fade)):
        at = me.attributes.get(nm) or me.attributes.new(nm, "FLOAT", "POINT")
        at.data.foreach_set("value", np.repeat(np.asarray(a, np.float32), nseg))
    me.shade_smooth()
    me.update()


def render_animation(scene, out_dir: str, *, frames: int | Sequence[int] | None = None,
                     size: tuple[int, int] = (960, 540), samples: int = 16, camera: str = "chase",
                     camera_offset: Sequence[float] | None = None, lens: float = 35, field: str | None = None,
                     hud: bool = True, hud_keys: Sequence[str] | None = None, sparkline: str | None = None,
                     fps: float = 24, gif: bool = False, mp4: bool = True,
                     sun_elevation: float = 35.0, ground: bool | float | None = None,
                     background: str = "sky", vectors: bool = True, trails: bool | dict = True, lines: bool = True,
                     colorbars: bool | dict | Sequence[str] = True, colorbar_on: bool = True,
                     motion_blur: bool | float = False, dof: float | None = None, haze: float | None = None,
                     exposure: float | None = None, crf: int = 27, smooth: bool = True,
                     name: str = "animation", log=None, _render_mask: Sequence[bool] | None = None) -> dict:
    """Render ``scene.animation`` (an :class:`~.animation.Animation`) frame by frame with Cycles (CPU, denoised),
    burn the HUD into each PNG and encode an MP4 (and a GIF). Returns ``{"frames": [png, ...], "mp4": path | None,
    "gif": path | None}``.

    frames: None (every frame), an int n (the animation resampled to n evenly spaced frames) or frame indices.
    camera: "chase" (behind the tracked body along its direction of travel, or at ``camera_offset`` in the body
    frame), "chase_low" (behind and a little aside of the body, the camera kept low over the ground, looking up at it
    as it climbs away: a runway shot;
    ``camera_offset`` in the body frame also works), "follow" (at ``camera_offset`` in scene axes from the body,
    default from the 3/4 view), "side" (alongside the path, tracking the body), "tower" (fixed at ``camera_offset``
    in scene axes, default beside the middle of the path, looking at the body) or "fixed" (the whole scene and path
    in the 3/4 view). ``lens`` in mm.
    field: colour the surfaces by this static per-vertex field; animated surface fields are always shown.
    trails: False, True, or ``{"radius": r, "alpha": a, "seconds": s}`` overriding every trail's tube radius,
    opacity and time window (defaults: 0.003 x the tracked body's size, ``add_trail``'s alpha and trail_seconds);
    a trail is the path of its last seconds behind the body, fading to transparent at its tail.
    lines: draw the scene's static Lines as translucent glowing tubes (brighter when there is haze).
    colorbars: True / False, a dict ``{"field": bool, "vectors": bool, "trails": bool, "lines": bool}`` or the
    kinds to show; ``colorbar_on=False`` turns them all off.
    hud_keys: only these HUD entries in the caption; sparkline: a HUD key drawn as an inset chart (bottom-left).
    motion_blur: True (a 1/30 s shutter in scene time) or the shutter in seconds of scene time (not of the clip,
    which may run faster than real time); the tracked bodies and the camera are keyframed at every rendered frame,
    so the ground blurs in a chase view while the tracked body stays sharp. dof: f-stop, focused on the tracked body each frame.
    haze: cloud-like volumetric haze, ``haze`` = optical depth over ten camera-to-body distances (0.5 to 2 reads
    as light haze; costs render time). exposure: view exposure (default -0.3 sky, -0.2 studio).
    ground: True for a grey plane under the scene and path, or a height (scene up axis) for its level.
    background: "sky" (physical sky with its sun disc lighting the scene) or "studio" (light gradient and a lamp).
    Materials named "asphalt" and "grass" become procedural (colour patches, roughness variation, bump); surfaces
    whose material contains "marking" are lifted (and "grass" lowered) a hair so coplanar paint stays crisp.
    smooth: smooth shading with sharp edges above 35 degrees. crf: H.264 quality of the MP4 (higher = smaller).
    Materials missing from ``MATERIALS`` and ``scene.materials`` render grey; their names are logged."""
    import bpy
    from mathutils import Quaternion, Vector

    from .compose import cmap_color, default_range, draw_colorbars
    _check_bpy(bpy)
    anim = scene.animation
    if anim is None:
        raise ValueError("scene.animation is None: build an Animation and set scene.animation first")
    if camera not in ("chase", "chase_low", "follow", "side", "tower", "fixed"):
        raise ValueError(f"camera={camera!r}: chase, chase_low, follow, side, tower or fixed")
    if isinstance(frames, int):
        anim = anim.resample(frames)
        frames = None
    ks = list(range(anim.n_frames)) if frames is None else [int(k) for k in frames]
    os.makedirs(out_dir, exist_ok=True)
    B = blender_axes(scene.axes)
    toB = lambda P: np.asarray(P, float) @ B.T                         # noqa: E731
    qB = lambda q: Quaternion((q[0], *(B @ np.asarray(q[1:], float))))  # noqa: E731
    if isinstance(colorbars, dict):
        cb_on = {k: bool(colorbars.get(k, True)) for k in COLORBAR_KINDS}
    elif isinstance(colorbars, bool):
        cb_on = dict.fromkeys(COLORBAR_KINDS, colorbars)
    else:
        cb_on = {k: k in colorbars for k in COLORBAR_KINDS}
    if not colorbar_on:
        cb_on = dict.fromkeys(COLORBAR_KINDS, False)
    bars = []                                                  # (kind, title, lo, hi, cmap)

    studio = background == "studio"
    missing = scene.unknown_materials() if hasattr(scene, "unknown_materials") else []
    if missing:
        (log or print)(f"render_animation: materials not in MATERIALS / scene.materials (rendered grey): {missing}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    bs = bpy.context.scene
    bs.render.engine = "CYCLES"
    bs.cycles.device = "CPU"
    bs.cycles.samples = max(1, int(samples))
    bs.cycles.use_denoising = True
    try:
        bs.cycles.denoiser = "OPENIMAGEDENOISE"
    except (TypeError, AttributeError):
        pass
    if hasattr(bs.render, "use_persistent_data"):
        bs.render.use_persistent_data = True
    bs.render.resolution_x, bs.render.resolution_y = size
    bs.render.resolution_percentage = 100
    bs.render.fps = max(1, int(round(fps)))
    names = [i.identifier for i in bs.view_settings.bl_rna.properties["view_transform"].enum_items]
    bs.view_settings.view_transform = "Standard" if studio else ("AgX" if "AgX" in names else "Filmic")
    bs.view_settings.exposure = exposure if exposure is not None else (-0.3 if not studio else -0.2)
    bs.render.image_settings.file_format = "PNG"
    if motion_blur:                                            # shutter in scene seconds -> in (resampled) frames
        tk = anim.times[ks] if len(ks) > 1 else anim.times
        dt = float(np.median(np.diff(tk))) if len(tk) > 1 else 1.0
        shutter_s = 1.0 / 30.0 if motion_blur is True else float(motion_blur)
        bs.render.use_motion_blur = True
        bs.render.motion_blur_shutter = float(min(1.0, max(1e-3, shutter_s / dt)))
    bs.frame_start, bs.frame_end = 1, max(1, len(ks))

    # the surfaces, built directly (object i is surface i, named s0000, s0001, ...)
    objs = dict(enumerate(_build_objects(bpy, bs, scene.surfaces, scene.axes, getattr(scene, "materials", None),
                                         names=[f"s{i:04d}" for i in range(len(scene.surfaces))])))
    if smooth:
        _smooth([o for i, o in objs.items() if scene.surfaces[i].group in anim.tracks])
    for m in bpy.data.materials:                               # procedural ground
        base = m.name.split(".")[0]
        if base in ("asphalt", "grass") and m.node_tree is not None:
            _ground_material(bpy, m, base)

    # extent of the static scene and of the paths (for the ground, the fixed camera and the clip planes)
    static = [toB(s.vertices) for i, s in enumerate(scene.surfaces) if s.group not in anim.tracks]
    lo_p, hi_p = anim.bounds()
    pts = np.concatenate(static + [toB(np.array([lo_p, hi_p]))]) if (static or anim.tracks or anim.trails) else np.zeros((1, 3))
    s_lo, s_hi = pts.min(0), pts.max(0)
    extent = float(np.linalg.norm(s_hi - s_lo)) or 1.0
    ground_z = float(np.concatenate(static)[:, 2].min()) if static else None
    decal = max(0.01, 2e-6 * extent)                           # coplanar paint over asphalt over grass
    for i, o in objs.items():
        mat = scene.surfaces[i].material
        if scene.surfaces[i].group not in anim.tracks and ("marking" in mat or "grass" in mat):
            o.location.z += decal if "marking" in mat else -decal

    # tracks: the surfaces of each group under an Empty
    empties, radius = {}, {}
    for g in anim.tracks:
        e = bpy.data.objects.new(f"track_{g}", None)
        e.rotation_mode = "QUATERNION"
        bs.collection.objects.link(e)
        empties[g] = e
        V = [toB(s.vertices) for s in scene.surfaces if s.group == g]
        V = np.concatenate(V) if V else np.zeros((1, 3))
        radius[g] = max(1e-6, 0.5 * float(np.linalg.norm(V.max(0) - V.min(0))))
        for i, s in enumerate(scene.surfaces):
            if s.group == g and i in objs:
                objs[i].parent = e
    fg = anim.follow_group if anim.follow_group in anim.tracks else next(iter(anim.tracks), None)
    body_r = radius.get(fg, extent / 20)
    body_size = 2 * body_r

    # colours: a static field, then the animated per-vertex fields
    if field:
        attr = "_" + field.upper().lstrip("_")
        lo, hi = scene.field_range(field)
        fm, grey = _ramp_material(bpy, field, attr, lo, hi), _plain(bpy, "grey", (0.35, 0.37, 0.4))
        for o in objs.values():
            o.data.materials.clear()
            o.data.materials.append(fm if attr in o.data.attributes else grey)
        bars.append(("field", scene.labels.get(field, field), lo, hi, "jet"))
    afields = []
    for sname, sf in anim.surface_fields.items():
        idx = next((i for i, s in enumerate(scene.surfaces) if s.name == sname), None)
        if idx is None or idx not in objs:
            continue
        o = objs[idx]
        me = o.data
        nv = len(me.vertices)
        if nv == len(scene.surfaces[idx].vertices):
            order = None
        else:                                                   # the importer split or merged vertices: nearest
            from scipy.spatial import cKDTree
            co = np.empty(nv * 3)
            me.vertices.foreach_get("co", co)
            order = cKDTree(toB(scene.surfaces[idx].vertices)).query(co.reshape(-1, 3))[1]
        at = me.attributes.new("anim_field", "FLOAT", "POINT")
        me.materials.clear()
        cm = sf.get("cmap", "jet")
        me.materials.append(_ramp_material(bpy, f"anim_{sname}", "anim_field", *sf["range"], cmap=cm))
        afields.append((sf, at, order, me))
        bars.append(("field", sf["label"] or sname, *sf["range"], cm))

    # arrows: one shaft and one head object per arrow, parented to the track when attached
    arrows = []
    if vectors and anim.vectors:
        shaft_me, head_me = _arrow_meshes(bpy)
        placeholder = _plain(bpy, "arrow", (0.8, 0.3, 0.05))
        shaft_me.materials.append(placeholder)
        head_me.materials.append(placeholder)
        for vi, vs in enumerate(anim.vectors):
            L = float(np.nanmax(np.linalg.norm(vs["vectors"], axis=2))) or 1.0
            base = _hex_rgb(vs["color"])
            for i in range(vs["origins"].shape[1]):
                m = _plain(bpy, f"arrow{vi}_{i}", _lin(base))
                m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 0.4
                pair = []
                for kind, me in (("shaft", shaft_me), ("head", head_me)):
                    ob = bpy.data.objects.new(f"arrow{vi}_{i}_{kind}", me)
                    ob.rotation_mode = "QUATERNION"
                    if vs["attach"] in empties:
                        ob.parent = empties[vs["attach"]]
                    bs.collection.objects.link(ob)
                    pair.append(ob)
                arrows.append((vs, i, pair, m, L))
            if vs["values"] is not None and vs["range"] is not None and vs["range"][1] - vs["range"][0] > 1e-12:
                bars.append(("vectors", vs["label"] or vs["name"], *vs["range"], vs.get("cmap", "jet")))
    # the arrow meshes are shared: each object gets its own material through an object-linked slot
    for _, _, pair, m, _ in arrows:
        for ob in pair:
            ob.material_slots[0].link = "OBJECT"
            ob.material_slots[0].material = m

    # trails: thin tubes along the last seconds of the path, coloured point by point when they carry values,
    # fading to transparent at the tail
    trail_objs = []
    if trails:
        ov = trails if isinstance(trails, dict) else {}
        for ti, tr in enumerate(anim.trails):
            r = ov.get("radius") or tr.get("radius") or 0.003 * body_size
            alpha = ov.get("alpha", tr.get("alpha", 1.0))
            if tr.get("values") is not None:
                cm = tr.get("cmap", "jet")
                rng = tr.get("range") or default_range(tr["values"], cm)
                m = _trail_material(bpy, f"trail_{tr['name']}", None, alpha, cm, *rng)
                bars.append(("trails", tr.get("label") or tr["name"], *rng, cm))
            else:
                m = _trail_material(bpy, f"trail_{tr['name']}", _lin(_hex_rgb(tr["color"])), alpha)
            me = bpy.data.meshes.new(f"trail_{tr['name']}")
            me.materials.append(m)
            ob = bpy.data.objects.new(f"trail_{tr['name']}", me)
            bs.collection.objects.link(ob)
            trail_objs.append((ob, ti, toB(tr["points"]), r, ov.get("seconds")))

    # the scene's static Lines (streamlines, gust wisps): soft translucent tubes
    if lines and scene.lines:
        for li, L in enumerate(scene.lines):
            cm = getattr(L, "cmap", "jet") or "jet"
            alpha = float(getattr(L, "alpha", 1.0))
            r = getattr(L, "radius", None) or (0.003 * body_size if anim.tracks else 0.0011 * extent)
            P = [toB(p) for p in L.points]
            S = [np.asarray(v, float) for v in L.values] if L.values else None
            cu = bpy.data.curves.new(f"lines{li}", "CURVE")
            cu.dimensions, cu.bevel_depth, cu.bevel_resolution = "3D", r, 2
            if S is not None:
                allv = np.concatenate(S)
                rng = getattr(L, "color_range", None) or (
                    default_range(allv, cm) if cm != "jet" else
                    (float(np.nanpercentile(allv, 1)), float(np.nanpercentile(allv, 99))))
                mats = _bucket_materials(bpy, f"lines{li}_", cm, alpha, strength=2.5 if haze else 1.0)
                bars.append(("lines", L.label or L.name, *rng, cm))
            else:
                rng, mats = (0.0, 1.0), [_wisp_material(bpy, f"lines{li}", (0.05 ** 2.2, 0.75 ** 2.2, 1.0), alpha,
                                                        2.5 if haze else 1.2)]
            for m in mats:
                cu.materials.append(m)
            _fill_curve(cu, P, S, rng[0], rng[1], len(mats), step=2)
            bs.collection.objects.link(bpy.data.objects.new(f"lines{li}", cu))

    if ground is not None and ground is not False:
        # Blender z is the scene's up axis; the automatic level sits a little below everything (no z-fighting with
        # a runway or floor surface of the scene)
        z = float(s_lo[2]) - 1e-3 * extent if ground is True else float(ground)
        c = 0.5 * (s_lo + s_hi)
        h = 1.5 * extent
        me = bpy.data.meshes.new("ground")
        me.from_pydata([(c[0] - h, c[1] - h, z), (c[0] + h, c[1] - h, z), (c[0] + h, c[1] + h, z),
                        (c[0] - h, c[1] + h, z)], [], [(0, 1, 2, 3)])
        gm = _plain(bpy, "ground", (0.30, 0.31, 0.30))
        gm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.95
        me.materials.append(gm)
        bs.collection.objects.link(bpy.data.objects.new("ground", me))
        ground_z = z if ground_z is None else min(ground_z, z)

    _world(bpy, bs, studio, sun_elevation)
    if not studio:                                             # a Sun lamp matching the sky, for crisp shadows
        sun = bpy.data.lights.new("sun_lamp", "SUN")
        sun.energy, sun.angle = 3.0, math.radians(1.5)
        so = bpy.data.objects.new("sun_lamp", sun)
        so.rotation_euler = (math.radians(90 - sun_elevation), 0, math.radians(215 + 90))
        bs.collection.objects.link(so)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = lens
    cam_data.sensor_fit = "AUTO"
    cam = bpy.data.objects.new("cam", cam_data)
    cam.rotation_mode = "QUATERNION"
    bs.collection.objects.link(cam)
    bs.camera = cam
    if dof:
        cam_data.dof.use_dof = True
        cam_data.dof.aperture_fstop = float(dof)
    half = 0.5 * cam_data.angle * min(size) / max(size)       # half angle across the shorter image side
    chase_d = body_r / math.tan(half) * 1.7

    def body_at(k):
        if fg is None:
            return 0.5 * (s_lo + s_hi), np.eye(3)
        M = anim.pose(fg, k)
        V = [s.vertices for s in scene.surfaces if s.group == fg]
        c_body = 0.5 * (np.concatenate(V).min(0) + np.concatenate(V).max(0)) if V else np.zeros(3)
        return B @ (M[:3, :3] @ c_body + M[:3, 3]), B @ M[:3, :3] @ B.T

    def travel(k, win=3):
        P = toB(anim.tracks[fg]["positions"]) if fg else np.zeros((anim.n_frames, 3))
        a, b = max(0, k - win), min(anim.n_frames - 1, k + win)
        v = P[b] - P[a]
        v[2] = 0.0
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 * extent else np.array([1.0, 0.0, 0.0])

    path = toB(anim.tracks[fg]["positions"]) if fg else pts
    p_dir = path[-1] - path[0]
    p_dir[2] = 0
    p_dir = p_dir / np.linalg.norm(p_dir) if np.linalg.norm(p_dir) > 1e-9 * extent else np.array([1.0, 0.0, 0.0])
    side_n = np.cross([0.0, 0.0, 1.0], p_dir)
    tower = (B @ np.asarray(camera_offset, float) if camera_offset is not None and camera == "tower"
             else 0.5 * (path.min(0) + path.max(0)) - side_n * max(4 * chase_d, 0.25 * extent) + np.array([0, 0, chase_d * 0.3]))
    iso = _to_blender(np.asarray(VIEWS["iso"], float)[None], "z_up")[0]
    iso /= np.linalg.norm(iso)

    def camera_at(k):
        """(location, look-at point, body centre) of the camera at frame k."""
        target, Rb = body_at(k)
        look = None
        if camera in ("chase", "chase_low") and camera_offset is not None:
            loc = target + Rb @ (B @ np.asarray(camera_offset, float))
        elif camera == "chase":
            loc = target - travel(k) * chase_d * 0.92 + np.array([0, 0, chase_d * 0.35])
        elif camera == "chase_low":
            fwd = travel(k)
            dl = 0.7 * chase_d                                 # closer than "chase": the body fills more of the frame
            loc = target - fwd * dl * 0.95 + np.cross([0.0, 0.0, 1.0], fwd) * 0.3 * dl + np.array([0, 0, -0.12 * dl])
            if ground_z is not None:                           # stays low over the runway while the body climbs away
                h = max(0.0, float(target[2] - ground_z))
                loc[2] = ground_z + 0.02 * dl + 0.05 * body_size + 0.3 * h
                dist = float(np.linalg.norm(target - loc))
                # aim a little below the body so the horizon stays in frame (the body sits above the centre)
                look = target - np.array([0, 0, min(0.45 * h, 0.5 * math.tan(half) * dist)])
        elif camera == "follow":
            loc = target + (B @ np.asarray(camera_offset, float) if camera_offset is not None else iso * chase_d)
        elif camera == "side":
            loc = target - side_n * chase_d + np.array([0, 0, 0.15 * chase_d])
        elif camera == "tower":
            loc = tower
        else:                                                  # fixed
            target = 0.5 * (s_lo + s_hi)
            loc = target + iso * 0.5 * extent / math.tan(half) * 0.8
        body = target if camera != "fixed" else body_at(k)[0]
        return np.asarray(loc, float), np.asarray(target if look is None else look, float), np.asarray(body, float)

    # pass 1: keyframe the bodies and the camera at every rendered frame (Blender frame j + 1), so motion blur
    # sees the motion between frames
    cams, q_prev = [], None
    for j, k in enumerate(ks):
        f = j + 1
        for g, e in empties.items():
            tr = anim.tracks[g]
            e.location = Vector(tuple(B @ tr["positions"][k]))
            e.rotation_quaternion = qB(tr["quaternions"][k])
            e.keyframe_insert("location", frame=f)
            e.keyframe_insert("rotation_quaternion", frame=f)
        loc, target, body = camera_at(k)
        q = Vector(tuple(target - loc)).to_track_quat("-Z", "Y")
        if q_prev is not None and q.dot(q_prev) < 0:
            q.negate()
        q_prev = q
        cam.location, cam.rotation_quaternion = Vector(tuple(loc)), q
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        d = max(float(np.linalg.norm(body - loc)), 1e-6)       # focus on the body, not on the aim point
        if dof:
            cam_data.dof.focus_distance = d
            cam_data.dof.keyframe_insert("focus_distance", frame=f)
        fwd = (target - loc) / max(float(np.linalg.norm(target - loc)), 1e-9)
        cams.append((loc, d, fwd))
    dmax = max(c[1] for c in cams)
    cam_data.clip_start = max(min(c[1] for c in cams) * 1e-3, 1e-5)
    cam_data.clip_end = max(dmax * 1e3, 3 * extent + dmax)

    if haze:
        C = np.array([c[0] for c in cams] + list(path))
        dref = float(np.median([c[1] for c in cams]))
        pad = np.array([20 * dref, 20 * dref, 0.0])
        lo = C.min(0) - pad
        hi = C.max(0) + pad
        lo[2] = (ground_z if ground_z is not None else C[:, 2].min() - 5 * dref) - 0.01 * dref
        hi[2] = C[:, 2].max() + 8 * dref
        _haze(bpy, bs, lo, hi, float(haze) / (10 * dref), blob=4 * dref, clear_obj=empties.get(fg),
              clear=(1.2 * dref, 5.0 * dref))                  # clear around the body: it and nearby lines stay crisp

    pngs = []
    for j, k in enumerate(ks):
        if _render_mask is not None and not _render_mask[j]:
            continue
        bs.frame_set(j + 1)
        for vs, i, (shaft, head), m, L in arrows:
            o, v = vs["origins"][k, i], vs["vectors"][k, i]
            o, v = B @ o, B @ v                                  # attached arrows: body coordinates under the Empty
            n = float(np.linalg.norm(v))
            show = n > 1e-9 * L
            for ob in (shaft, head):
                ob.hide_render = not show
            if not show:
                continue
            u = v / n
            q = Vector((0.0, 0.0, 1.0)).rotation_difference(Vector(tuple(u)))
            h, r = min(0.22 * L, 0.45 * n), min(0.022 * L, 0.25 * n)
            shaft.location, shaft.rotation_quaternion, shaft.scale = Vector(tuple(o)), q, (r, r, n - h)
            head.location, head.rotation_quaternion, head.scale = Vector(tuple(o + u * (n - h))), q, (2.6 * r, 2.6 * r, h)
            if vs["values"] is not None and vs["range"] is not None:
                lo, hi = vs["range"]
                tt = (vs["values"][k, i] - lo) / (hi - lo) if hi - lo > 1e-12 else 0.5      # no range: neutral
                c = _lin(cmap_color(vs.get("cmap", "jet"), tt))
                b = m.node_tree.nodes["Principled BSDF"]
                b.inputs["Base Color"].default_value = (*c, 1)
                b.inputs["Emission Color"].default_value = (*c, 1)
        for ob, ti, P, r, sec in trail_objs:
            idx, fade = anim.trail_window(ti, k, sec)
            # a trail never runs through the camera: keep only its newest run of points at least 0.8 x the body
            # distance in front of it; where it was cut, it fades out over what is left
            cl, cd, cf = cams[j]
            ahead = (P[idx] - cl) @ cf > 0.8 * cd
            n_keep = len(idx) if ahead.all() else int(np.argmin(ahead[::-1]))
            if n_keep < len(idx):
                idx, fade = idx[len(idx) - n_keep:], fade[len(fade) - n_keep:] * np.linspace(0.0, 1.0, n_keep)
            vals = anim.trails[ti].get("values")
            _set_tube(ob.data, P[idx], np.zeros(len(idx)) if vals is None else vals[idx], fade, r)
        for sf, at, order, me in afields:
            vals = np.asarray(sf["values"][k], float)
            vals = np.where(np.isfinite(vals), vals, sf["range"][0])
            at.data.foreach_set("value", (vals if order is None else vals[order]).astype(np.float32))
            me.update()
        bpy.context.view_layer.update()
        path_png = os.path.join(out_dir, f"frame_{j:05d}.png")
        bs.render.filepath = os.path.abspath(path_png)
        bpy.ops.render.render(write_still=True)
        shown = [b[1:] for b in bars if cb_on.get(b[0], True)]
        if shown:
            draw_colorbars(path_png, shown)
        if hud:
            _hud_overlay(path_png, anim.hud_lines(k, keys=hud_keys))
        if sparkline:
            h = next((h for h in anim.hud if h["key"] == sparkline), None)
            if h is None:
                raise KeyError(f"sparkline={sparkline!r}: no HUD entry with that key ({[x['key'] for x in anim.hud]})")
            _sparkline(path_png, anim.times, h["values"], k, h["label"], h["unit"], h["fmt"])
        pngs.append(path_png)
        if log and (j + 1) % 10 == 0:
            log(f"rendered {j + 1}/{len(ks)} frames")
    out = {"frames": pngs, "mp4": None, "gif": None}
    if mp4 and pngs:
        out["mp4"] = _encode(pngs, os.path.join(out_dir, f"{name}.mp4"), fps, crf=crf)
    if gif and pngs:
        out["gif"] = _encode(pngs, os.path.join(out_dir, f"{name}.gif"), fps, gif=True)
    return out


def render_still(scene, out_png: str, frame: int, *, size: tuple[int, int] = (1920, 1080), samples: int = 64,
                 **kw) -> str:
    """One high-quality frame of ``scene.animation`` (frame index ``frame``) as a PNG at ``out_png``; takes every
    option of :func:`render_animation` (camera, dof, motion_blur, haze, hud_keys, sparkline, trails, lines, ...).
    With motion blur the neighbouring frames are keyframed too, so the blur is that of the clip."""
    import shutil as _sh
    anim = scene.animation
    if anim is None:
        raise ValueError("scene.animation is None")
    k = int(frame) % anim.n_frames
    ks = [x for x in (k - 1, k, k + 1) if 0 <= x < anim.n_frames]
    tmp = tempfile.mkdtemp()
    kw = {**kw, "mp4": False, "gif": False}
    r = render_animation(scene, tmp, frames=ks, size=size, samples=samples, _render_mask=[x == k for x in ks], **kw)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    _sh.move(r["frames"][0], out_png)
    _sh.rmtree(tmp, ignore_errors=True)
    return out_png
