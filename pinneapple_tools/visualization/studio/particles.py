"""Particle animations: thousands of spheres coloured by a value (speed, temperature, species), inside glass or steel
equipment that can move (a turning impeller), rendered frame by frame with Blender Cycles; then composed with live
charts that draw themselves as time runs, a colour bar and a clock, into a GIF (and an MP4 when Blender's encoder
is available). Works for any Lagrangian result: stirred tanks, fluidised beds, hoppers, sprays, mixers, DEM.

    from pinneapple_tools.visualization.studio.particles import render_particle_frames, compose_video
    pngs = render_particle_frames(frames, d=3e-3, equipment=lambda k: tank.surfaces(angle[k]), out_dir="frames",
                                  field_range=(0, 0.3), view=(-1.0, -1.2, 0.55))
    compose_video(pngs, times, out="tank.gif", charts=[("Particles Top [%]", times, 100 * top),
                                                       ("Stirrer Speed [RPM]", times, rpm)],
                  colorbar=("Velocity Magnitude (m/s)", 0, 0.3))

``frames``: a list of dicts with ``x`` (n, 3) positions and ``value`` (n,) (or ``speed``), z up.
``equipment(k)``: the surfaces of frame k as (name, V, F, material) with material in "glass", "liquid", "steel",
"grey".
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence

import numpy as np

VIRIDIS = [(0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530), (0.207, 0.372, 0.553),
           (0.164, 0.471, 0.558), (0.128, 0.567, 0.551), (0.135, 0.659, 0.518), (0.267, 0.749, 0.441),
           (0.478, 0.821, 0.318), (0.741, 0.873, 0.150), (0.993, 0.906, 0.144)]


def _lin(c):
    return tuple(v ** 2.2 for v in c)


def _material(bpy, kind: str):
    m = bpy.data.materials.new(kind)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    if kind == "glass":
        b.inputs["Base Color"].default_value = (0.92, 0.95, 1.0, 1)
        b.inputs["Transmission Weight"].default_value = 1.0
        b.inputs["Roughness"].default_value = 0.04
        b.inputs["IOR"].default_value = 1.05                   # thin wall: little refraction
        b.inputs["Alpha"].default_value = 0.25
    elif kind == "liquid":
        b.inputs["Base Color"].default_value = (0.35, 0.42, 0.5, 1)
        b.inputs["Roughness"].default_value = 0.05
        b.inputs["Alpha"].default_value = 0.12
    elif kind == "steel":
        b.inputs["Base Color"].default_value = (0.62, 0.64, 0.67, 1)
        b.inputs["Metallic"].default_value = 1.0
        b.inputs["Roughness"].default_value = 0.25
    else:
        b.inputs["Base Color"].default_value = (0.16, 0.165, 0.175, 1)
        b.inputs["Roughness"].default_value = 0.8
    if hasattr(m, "blend_method") and kind in ("glass", "liquid"):
        m.blend_method = "BLEND"
    return m


def _particle_material(bpy, lo: float, hi: float, cmap=VIRIDIS):
    m = bpy.data.materials.new("particles")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_type = "GEOMETRY"
    a.attribute_name = "value"
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = lo, hi
    rp = nt.nodes.new("ShaderNodeValToRGB")
    el = rp.color_ramp.elements
    el[0].position, el[0].color = 0.0, (*_lin(cmap[0]), 1)
    el[1].position, el[1].color = 1.0, (*_lin(cmap[-1]), 1)
    for i, c in enumerate(cmap[1:-1], start=1):
        e = el.new(i / (len(cmap) - 1))
        e.color = (*_lin(c), 1)
    nt.links.new(a.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], rp.inputs["Fac"])
    nt.links.new(rp.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(rp.outputs["Color"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = 0.25
    b.inputs["Roughness"].default_value = 0.35
    b.inputs["Specular IOR Level"].default_value = 0.6
    return m


def _points_object(bpy, n: int, radius: float, material):
    """A mesh of n vertices turned into a Cycles point cloud (spheres) by geometry nodes; attribute "value" per point."""
    me = bpy.data.meshes.new("pts")
    me.vertices.add(n)
    me.attributes.new("value", "FLOAT", "POINT")
    ob = bpy.data.objects.new("particles", me)
    bpy.context.scene.collection.objects.link(ob)
    ng = bpy.data.node_groups.new("to_points", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    m2p.inputs["Radius"].default_value = radius
    sm = ng.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = material
    ng.links.new(gi.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(m2p.outputs["Points"], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], go.inputs[0])
    mod = ob.modifiers.new("to_points", "NODES")
    mod.node_group = ng
    return ob


def render_particle_frames(frames: Sequence[dict], d: float, out_dir: str, *,
                           equipment: Callable[[int], Sequence[tuple]] | None = None,
                           field: str = "speed", field_range: tuple[float, float] | None = None,
                           view: Sequence[float] = (-1.0, -1.25, 0.55), distance: float = 1.0, lens: float = 50,
                           size: tuple[int, int] = (960, 720), samples: int = 24, every: int = 1,
                           floor: bool = True, background: tuple[float, float, float] = (0.30, 0.31, 0.33),
                           log: Callable[[str], None] | None = None) -> list[str]:
    """Render every ``every``-th frame to ``out_dir/frame_XXXX.png``; the camera frames the equipment (or the
    particles) once. Returns the PNG paths."""
    import bpy
    from mathutils import Vector

    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bs = bpy.context.scene
    bs.render.engine = "CYCLES"
    bs.cycles.device = "CPU"
    bs.cycles.samples = samples
    bs.cycles.use_denoising = True
    bs.render.resolution_x, bs.render.resolution_y = size
    bs.view_settings.view_transform = "Standard"
    bs.render.film_transparent = False
    vals = [np.asarray(f.get("value", f.get(field)), float) for f in frames]
    lo, hi = field_range or (0.0, float(np.percentile(np.concatenate(vals), 99)))
    n = len(frames[0]["x"])
    pts = _points_object(bpy, n, d / 2, _particle_material(bpy, lo, hi))
    mats = {}
    eq_objs: dict[str, object] = {}

    def set_equipment(k):
        if equipment is None:
            return
        for name, V, F, kind in equipment(k):
            if kind not in mats:
                mats[kind] = _material(bpy, kind)
            if name in eq_objs:                                  # same topology: move the vertices
                eq_objs[name].data.vertices.foreach_set("co", np.asarray(V, np.float32).ravel())
                eq_objs[name].data.update()
                continue
            me = bpy.data.meshes.new(name)
            me.from_pydata([tuple(map(float, v)) for v in V], [], [tuple(map(int, f)) for f in F])
            me.materials.append(mats[kind])
            for poly in me.polygons:
                poly.use_smooth = kind != "steel" or name == "shaft"
            ob = bpy.data.objects.new(name, me)
            bs.collection.objects.link(ob)
            eq_objs[name] = ob
    set_equipment(0)
    allp = [np.asarray(frames[0]["x"], float)]
    for ob in eq_objs.values():
        co = np.empty(len(ob.data.vertices) * 3)
        ob.data.vertices.foreach_get("co", co)
        allp.append(co.reshape(-1, 3))
    P = np.concatenate(allp)
    lo3, hi3 = P.min(0), P.max(0)
    centre = 0.5 * (lo3 + hi3)
    if floor:
        z0 = float(lo3[2]) - 1e-4
        L = 6 * float(np.max(hi3 - lo3))
        me = bpy.data.meshes.new("floor")
        me.from_pydata([(centre[0] - L, centre[1] - L, z0), (centre[0] + L, centre[1] - L, z0),
                        (centre[0] + L, centre[1] + L, z0), (centre[0] - L, centre[1] + L, z0)], [], [(0, 1, 2, 3)])
        me.materials.append(_material(bpy, "grey"))
        bs.collection.objects.link(bpy.data.objects.new("floor", me))
    world = bpy.data.worlds.new("world")
    bs.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*_lin(background), 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    span = float(np.max(hi3 - lo3))
    key = bpy.data.lights.new("key", "AREA")
    key.energy, key.size = 12.0 * span ** 2 / 0.04, 1.2 * span
    ko = bpy.data.objects.new("key", key)
    ko.location = Vector(tuple(centre + span * np.array([-0.8, -1.2, 1.6])))
    ko.rotation_euler = (Vector(tuple(centre)) - ko.location).to_track_quat("-Z", "Y").to_euler()
    bs.collection.objects.link(ko)
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy, sun.angle = 0.6, math.radians(12)
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(30), math.radians(-25), math.radians(40))
    bs.collection.objects.link(so)
    dv = np.asarray(view, float)
    dv /= np.linalg.norm(dv)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = lens
    cam = bpy.data.objects.new("cam", cam_data)
    bs.collection.objects.link(cam)
    t_long = math.tan(0.5 * cam_data.angle)
    t_short = t_long * min(size) / max(size)
    tx, ty = (t_long, t_short) if size[0] >= size[1] else (t_short, t_long)
    fwd = -dv
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    k = P - centre
    lat = np.maximum(np.abs(k @ right) / tx, np.abs(k @ up) / ty)
    fit = float(np.max(k @ dv + lat)) * 1.12 * distance
    cam.location = Vector(tuple(centre + dv * fit))
    cam.rotation_euler = (Vector(tuple(centre)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    bs.camera = cam
    out = []
    for k_ in range(0, len(frames), every):
        f = frames[k_]
        x = np.asarray(f["x"], np.float32)
        pts.data.vertices.foreach_set("co", x.ravel())
        pts.data.attributes["value"].data.foreach_set("value", np.asarray(vals[k_], np.float32))
        pts.data.update()
        set_equipment(k_)
        path = os.path.join(out_dir, f"frame_{k_:04d}.png")
        bs.render.filepath = os.path.abspath(path)
        bpy.ops.render.render(write_still=True)
        out.append(path)
        if log and len(out) % 10 == 0:
            log(f"rendered {len(out)} frames")
    return out


def compose_video(images: Sequence[str], times: Sequence[float], out: str, *,
                  charts: Sequence[tuple[str, Sequence[float], Sequence[float]]] = (),
                  colorbar: tuple[str, float, float] | None = None, cmap: str = "viridis",
                  clock: str = "Time: {t:.1f}", title: str = "", duration_ms: int = 100,
                  width: float = 12.0, dpi: int = 80, mp4: bool = False, frame_dir: str | None = None) -> str:
    """One composed frame per image: the render on the left, the ``charts`` (label, t, y) on the right drawn up to
    the current time (full axes from the start, so the line grows), a colour bar over the render and a clock.
    Writes a GIF to ``out``; with ``mp4`` also ``out`` with .mp4 through Blender's FFmpeg when available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from PIL import Image
    times = np.asarray(times, float)
    first = Image.open(images[0])
    ar = first.height / first.width
    frames = []
    nc = len(charts)
    for i, img in enumerate(images):
        t = times[i]
        fig = plt.figure(figsize=(width, width * 0.55), facecolor="#6d6f73")
        axr = fig.add_axes([0.0, 0.0, 0.62, 1.0])
        axr.imshow(Image.open(img))
        axr.set_axis_off()
        axr.set_aspect("auto" if abs(ar - 0.55 * width / (0.62 * width)) < 0.05 else "equal")
        axr.text(0.04, 0.05, clock.format(t=t), transform=axr.transAxes, color="white", fontsize=15, alpha=0.9)
        if title:
            axr.text(0.04, 0.95, title, transform=axr.transAxes, color="white", fontsize=11, va="top")
        if colorbar:
            cax = fig.add_axes([0.16, 0.86, 0.26, 0.025])
            cb = fig.colorbar(cm.ScalarMappable(colors.Normalize(colorbar[1], colorbar[2]), cmap), cax=cax,
                              orientation="horizontal")
            cb.ax.set_title(colorbar[0], color="white", fontsize=11)
            cb.ax.tick_params(colors="white", labelsize=9)
            cb.ax.xaxis.set_ticks_position("top")
            cb.outline.set_visible(False)
        for c, (lab, tc, yc) in enumerate(charts):
            h = 0.78 / max(nc, 1)
            ax = fig.add_axes([0.68, 0.12 + (nc - 1 - c) * (h + 0.06 / max(nc, 1)), 0.29, h - 0.06])
            tc, yc = np.asarray(tc, float), np.asarray(yc, float)
            m = tc <= t + 1e-9
            ax.plot(tc[m], yc[m], color=["#e07b39", "#2b7bba", "#4daf4a", "#984ea3"][c % 4], lw=1.8)
            ax.set_xlim(tc.min(), tc.max())
            pad = 0.08 * (np.nanmax(yc) - np.nanmin(yc) or 1)
            ax.set_ylim(min(0.0, np.nanmin(yc)) - pad * 0.2, np.nanmax(yc) + pad)
            ax.set_facecolor("none")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("white")
            ax.tick_params(colors="white", labelsize=9)
            ax.set_ylabel(lab, color="white", fontsize=10)
            ax.set_xlabel("Time [s]", color="white", fontsize=10)
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames.append(Image.fromarray(arr))
        if frame_dir:
            os.makedirs(frame_dir, exist_ok=True)
            frames[-1].save(os.path.join(frame_dir, f"composed_{i:04d}.png"))
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=255) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=duration_ms, loop=0, optimize=True)
    if mp4 and frame_dir:
        try:
            encode_mp4(sorted(os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.startswith("composed_")),
                       os.path.splitext(out)[0] + ".mp4", fps=max(1, round(1000 / duration_ms)))
        except Exception:                                         # noqa: BLE001 - the GIF is the deliverable
            pass
    return out


def encode_mp4(pngs: Sequence[str], out: str, fps: int = 10) -> str:
    """H.264 MP4 of a PNG sequence with Blender's sequencer (no ffmpeg binary needed)."""
    import bpy
    from PIL import Image
    w, h = Image.open(pngs[0]).size
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = w - w % 2, h - h % 2
    sc.render.fps = fps
    se = sc.sequence_editor_create()
    strip = se.sequences.new_image("frames", pngs[0], 1, 1)
    for p in pngs[1:]:
        strip.elements.append(os.path.basename(p))
    sc.frame_start, sc.frame_end = 1, len(pngs)
    sc.render.image_settings.file_format = "FFMPEG"
    sc.render.ffmpeg.format = "MPEG4"
    sc.render.ffmpeg.codec = "H264"
    sc.render.ffmpeg.constant_rate_factor = "HIGH"
    sc.render.filepath = os.path.abspath(out)
    bpy.ops.render.render(animation=True)
    produced = [f for f in os.listdir(os.path.dirname(os.path.abspath(out)))
                if f.startswith(os.path.basename(out).split(".")[0]) and f.endswith(".mp4")]
    if produced and not os.path.exists(out):
        os.replace(os.path.join(os.path.dirname(os.path.abspath(out)), produced[0]), out)
    return out
