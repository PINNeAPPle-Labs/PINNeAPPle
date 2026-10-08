"""Photoreal renders of an aircraft .glb with Blender Cycles (``pip install bpy`` or run inside Blender).

    python -m pinneapple_design.aero.render_blender aircraft.glb out.png --view hero --samples 96

Scenes: a physical sky (Nishita: sun, atmosphere), a ground with a soft shadow catcher feel, a camera with a long lens.
``--cp`` colours the airframe by the ``_CP`` vertex attribute (surface pressure) with a blue-white-red ramp; ``--lines``
adds streamline tubes from an .npz (``lines``: list of (n, 3) arrays, same axes as the aircraft).
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
}


def render(glb: str, out: str, view: str = "hero", samples: int = 96, size=(1600, 900), cp: bool = False,
           lines_npz: str = None, sun_elevation: float = 38.0, ground: bool = True, transparent: bool = False,
           distance: float = 1.0) -> str:
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

    bpy.ops.import_scene.gltf(filepath=glb)
    objs = [o for o in sc.objects if o.type == "MESH"]
    # Blender's glTF importer converts glTF y-up back to z-up: aircraft x(aft) -> Blender ?, use bounds to frame
    mins = Vector((min(min((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
    maxs = Vector((max(max((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
    centre = (mins + maxs) / 2
    size_l = (maxs - mins).length

    if cp:
        mat = bpy.data.materials.new("pressure")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "_CP"
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        mapr = nt.nodes.new("ShaderNodeMapRange")
        mapr.inputs["From Min"].default_value, mapr.inputs["From Max"].default_value = -1.2, 1.0
        nt.links.new(attr.outputs["Fac"], mapr.inputs["Value"])
        nt.links.new(mapr.outputs["Result"], ramp.inputs["Fac"])
        el = ramp.color_ramp.elements
        el[0].color = (0.02, 0.15, 0.6, 1)
        el[1].color = (0.75, 0.05, 0.03, 1)
        mid = el.new(0.55)
        mid.color = (0.95, 0.95, 0.95, 1)
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.35
        for o in objs:
            if "_CP" in o.data.attributes:
                o.data.materials.clear()
                o.data.materials.append(mat)

    if lines_npz:
        z = np.load(lines_npz, allow_pickle=True)
        lmat = bpy.data.materials.new("streamlines")
        lmat.use_nodes = True
        b = lmat.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (0.05, 0.75, 1.0, 1)
        b.inputs["Emission Color"].default_value = (0.05, 0.6, 1.0, 1)
        b.inputs["Emission Strength"].default_value = 2.0
        for k, L in enumerate(z["lines"]):
            L = np.asarray(L, float)
            cu = bpy.data.curves.new(f"sl{k}", "CURVE")
            cu.dimensions = "3D"
            cu.bevel_depth = 0.012
            sp = cu.splines.new("POLY")
            sp.points.add(len(L) - 1)
            for i, p in enumerate(L):                              # aircraft axes (x aft, y right, z up) -> Blender
                sp.points[i].co = (p[1], -p[0], p[2], 1)              # aircraft -> glTF (y, z, x) -> Blender (x, -z, y)
            ob = bpy.data.objects.new(f"sl{k}", cu)
            ob.data.materials.append(lmat)
            sc.collection.objects.link(ob)

    # sky
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
    ap.add_argument("--lines", default=None)
    ap.add_argument("--no-ground", action="store_true")
    ap.add_argument("--distance", type=float, default=1.0)
    a = ap.parse_args(argv)
    render(a.glb, a.out, a.view, a.samples, (a.width, a.height), a.cp, a.lines, ground=not a.no_ground,
           distance=a.distance)


if __name__ == "__main__":
    main()
