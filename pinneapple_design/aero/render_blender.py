"""Photoreal renders of an aircraft .glb with Blender Cycles (``pip install bpy`` or run inside Blender).

    python -m pinneapple_design.aero.render_blender aircraft.glb out.png --view hero --samples 96

Scenes: a physical sky (Nishita: sun, atmosphere), a ground with a soft shadow catcher feel, a camera with a long lens.
``--field cp|cf`` colours the airframe by the ``_CP`` / ``_CF`` vertex attribute (OpenFOAM skin pressure or friction)
with the usual CFD jet scale (blue low, red high) over ``--range LO HI``; ``--lines`` adds streamline tubes from an
.npz (``lines``: (n_lines, n, 3), same axes as the aircraft; with ``line_speed`` they are coloured by speed on the
same scale and mirrored to both sides).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

VIEWS = {   # camera offset from the aircraft centre (Blender axes after the glTF import), lens mm
    "hero": ((-11.0, -8.0, 1.6), 60),
    "hero_front": ((-10.5, 8.5, 1.2), 55),
    "front": ((-15.0, 0.0, 0.8), 70),
    "top": ((0.0, 0.01, 18.0), 45),
    "side": ((0.0, -18.0, 0.6), 60),
    "rear": ((9.0, -8.5, 3.2), 55),
    "flight": ((-9.0, -11.0, -1.2), 55),          # from slightly below, sky behind
    "flight_high": ((-6.0, -12.0, 4.5), 50),
    "flight_rear": ((10.0, -9.0, 2.0), 55),
    "flight_front": ((-10.5, 9.5, -1.4), 55),     # nose towards the camera, from slightly below
    "cfd": ((-9.5, 8.0, 7.5), 50),                   # front-left, above: the upper skin of the left wing
    "cfd_low": ((-9.5, 8.5, -4.0), 50),              # front-left, below
}


JET = [(0.0, (0, 0, 143)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)), (0.625, (255, 255, 0)), (0.875, (255, 0, 0)),
       (1.0, (128, 0, 0))]


def jet(t: float):
    """sRGB jet colour for t in [0, 1] (0..1 floats)."""
    t = min(1.0, max(0.0, t))
    for (a, ca), (b, cb) in zip(JET, JET[1:]):
        if t <= b:
            f = (t - a) / (b - a)
            return tuple((ca[k] + f * (cb[k] - ca[k])) / 255 for k in range(3))
    return tuple(c / 255 for c in JET[-1][1])


def _lin(c):
    return tuple(v ** 2.2 for v in c)                    # Blender colour sockets are linear


def render(glb: str, out: str, view: str = "hero", samples: int = 96, size=(1600, 900), cp: bool = False,
           lines_npz: str = None, sun_elevation: float = 38.0, ground: bool = True, transparent: bool = False,
           distance: float = 1.0, field: str = None, field_range=(-1.2, 0.8), line_range=None,
           studio: bool = False) -> str:
    """cp=True is the same as field="cp". studio=True: plain light background, a sun lamp and no tone mapping, so
    the field colours come out as on the colour bar (for CFD figures)."""
    import bpy
    import numpy as np
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.film_transparent = transparent
    sc.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = -0.3
    if studio:
        sc.view_settings.view_transform = "Standard"
        sc.view_settings.exposure = 0.0 if (field or cp) else -0.6      # painted aircraft: keep the white paint below clipping

    bpy.ops.import_scene.gltf(filepath=glb)
    objs = [o for o in sc.objects if o.type == "MESH"]
    # Blender's glTF importer converts glTF y-up back to z-up: aircraft x(aft) -> Blender ?, use bounds to frame
    mins = Vector((min(min((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
    maxs = Vector((max(max((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
    centre = (mins + maxs) / 2
    size_l = (maxs - mins).length

    field = field or ("cp" if cp else None)
    if field:
        an = "_" + field.upper()
        mat = bpy.data.materials.new(field)
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = an
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        mapr = nt.nodes.new("ShaderNodeMapRange")
        mapr.inputs["From Min"].default_value, mapr.inputs["From Max"].default_value = field_range
        nt.links.new(attr.outputs["Fac"], mapr.inputs["Value"])
        nt.links.new(mapr.outputs["Result"], ramp.inputs["Fac"])
        el = ramp.color_ramp.elements
        el[0].position, el[0].color = 0.0, (*_lin(jet(0.0)), 1)
        el[1].position, el[1].color = 1.0, (*_lin(jet(1.0)), 1)
        for t, _ in JET[1:-1]:
            e = el.new(t)
            e.color = (*_lin(jet(t)), 1)
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 0.35          # keep the colours readable in shadow
        bsdf.inputs["Roughness"].default_value = 0.45
        grey = bpy.data.materials.new("grey")
        grey.use_nodes = True
        grey.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.37, 0.4, 1)
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(mat if an in o.data.attributes else grey)

    if lines_npz:
        z = np.load(lines_npz, allow_pickle=True)
        speed = z["line_speed"] if "line_speed" in z.files else None
        nb = 24 if speed is not None else 1                        # one emissive material per colour bin
        lo, hi = line_range or ((np.nanpercentile(speed, 2), np.nanpercentile(speed, 98)) if speed is not None else (0, 1))
        mats = []
        for j in range(nb):
            m = bpy.data.materials.new(f"streamline{j}")
            m.use_nodes = True
            b = m.node_tree.nodes["Principled BSDF"]
            c = _lin(jet((j + 0.5) / nb)) if speed is not None else (0.05 ** 2.2, 0.75 ** 2.2, 1.0)
            b.inputs["Base Color"].default_value = (*c, 1)
            b.inputs["Emission Color"].default_value = (*c, 1)
            b.inputs["Emission Strength"].default_value = 1.6
            mats.append(m)
        lines = [np.asarray(L, float) for L in z["lines"]]
        speeds = [np.asarray(v, float) for v in speed] if speed is not None else [None] * len(lines)
        if speed is not None:                                      # half model: mirror to the left side
            lines, speeds = lines + [L * [1, -1, 1] for L in lines], speeds + speeds
        bev = 0.0011 * size_l
        for k, (L, S) in enumerate(zip(lines, speeds)):
            keep = np.r_[True, np.linalg.norm(np.diff(L, axis=0), axis=1) > 1e-3]   # drop repeated end points
            L = L[keep]
            S = S[keep] if S is not None else None
            cu = bpy.data.curves.new(f"sl{k}", "CURVE")
            cu.dimensions = "3D"
            cu.bevel_depth = bev
            cu.bevel_resolution = 2
            for m in mats:
                cu.materials.append(m)
            step = 3 if S is not None else len(L)
            for i0 in range(0, len(L) - 1, step):                   # short overlapping pieces, each with its colour
                seg = L[i0:i0 + step + 1]
                sp = cu.splines.new("POLY")
                sp.points.add(len(seg) - 1)
                for i, p in enumerate(seg):                         # aircraft (x aft, y right, z up) -> Blender
                    sp.points[i].co = (p[1], -p[0], p[2], 1)          # via glTF (y, z, x) -> Blender (x, -z, y)
                if S is not None:
                    t = (np.mean(S[i0:i0 + step + 1]) - lo) / (hi - lo)
                    sp.material_index = int(min(nb - 1, max(0, t * nb)))
            ob = bpy.data.objects.new(f"sl{k}", cu)
            sc.collection.objects.link(ob)

    if studio:
        world = bpy.data.worlds.new("studio")
        sc.world = world
        world.use_nodes = True
        wn = world.node_tree
        grad = wn.nodes.new("ShaderNodeTexGradient")
        tc = wn.nodes.new("ShaderNodeTexCoord")
        mp = wn.nodes.new("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value = (0, math.radians(-90), 0)    # vertical gradient
        cr = wn.nodes.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].color = (0.62, 0.66, 0.72, 1)
        cr.color_ramp.elements[1].color = (0.95, 0.96, 0.98, 1)
        wn.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        wn.links.new(mp.outputs["Vector"], grad.inputs["Vector"])
        wn.links.new(grad.outputs["Fac"], cr.inputs["Fac"])
        wn.links.new(cr.outputs["Color"], wn.nodes["Background"].inputs["Color"])
        wn.nodes["Background"].inputs["Strength"].default_value = 0.9
        sun = bpy.data.lights.new("sun", "SUN")
        sun.energy = 2.2
        sun.angle = math.radians(8)
        so = bpy.data.objects.new("sun", sun)
        so.rotation_euler = (math.radians(35), math.radians(-20), math.radians(30))
        sc.collection.objects.link(so)
        ground = False
    if not studio:                                                     # physical sky
        world = bpy.data.worlds.new("sky")
        sc.world = world
        world.use_nodes = True
        wn = world.node_tree
        sky = wn.nodes.new("ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(sun_elevation)
        sky.sun_rotation = math.radians(215)
        sky.altitude = 400 if ground else 9000
        sky.air_density, sky.dust_density = 1.0, 1.5
        bg = wn.nodes["Background"]
        bg.inputs["Strength"].default_value = 0.12
        wn.links.new(sky.outputs["Color"], bg.inputs["Color"])

    if ground:
        bpy.ops.mesh.primitive_plane_add(size=400, location=(centre.x, centre.y, mins.z - 0.002))
        g = bpy.context.active_object
        gm = bpy.data.materials.new("tarmac")
        gm.use_nodes = True
        gb = gm.node_tree.nodes["Principled BSDF"]
        noise = gm.node_tree.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 60.0
        cr = gm.node_tree.nodes.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].color = (0.16, 0.16, 0.17, 1)
        cr.color_ramp.elements[1].color = (0.30, 0.30, 0.31, 1)
        gm.node_tree.links.new(noise.outputs["Fac"], cr.inputs["Fac"])
        gm.node_tree.links.new(cr.outputs["Color"], gb.inputs["Base Color"])
        gb.inputs["Roughness"].default_value = 0.85
        g.data.materials.append(gm)

    # camera
    (dx, dy, dz), lens = VIEWS[view]
    k = size_l / 13.0 * distance
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = lens
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    cam.location = centre + Vector((dx * k, dy * k, dz * k))
    d = centre - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    cam_data.dof.use_dof = view == "hero"
    cam_data.dof.focus_distance = d.length
    cam_data.dof.aperture_fstop = 5.6

    if out.lower().endswith((".jpg", ".jpeg")):
        sc.render.image_settings.file_format = "JPEG"
        sc.render.image_settings.quality = 90
    sc.render.filepath = out
    bpy.ops.render.render(write_still=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("out")
    ap.add_argument("--view", default="hero", choices=list(VIEWS))
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--cp", action="store_true")
    ap.add_argument("--field", choices=["cp", "cf"], default=None)
    ap.add_argument("--range", nargs=2, type=float, default=(-1.2, 0.8))
    ap.add_argument("--sun", type=float, default=38.0)
    ap.add_argument("--lines", default=None)
    ap.add_argument("--no-ground", action="store_true")
    ap.add_argument("--distance", type=float, default=1.0)
    a = ap.parse_args(argv)
    render(a.glb, a.out, a.view, a.samples, (a.width, a.height), a.cp, a.lines, sun_elevation=a.sun,
           ground=not a.no_ground, distance=a.distance, field=a.field, field_range=tuple(a.range))


if __name__ == "__main__":
    main()
