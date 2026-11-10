"""The result as a picture: a 3D scene of the passage, a Blender render, and a video.

Three layers, in the order they are worth having:

:func:`build_scene`
    a :class:`pinneapple_twin3d.Scene` — the walls and one meridian cut, triangulated, carrying the
    fields per vertex. It exports to the interactive viewer (``Scene.export``), to OpenUSD, and to
    the Studio, so the same object serves the page, the report and the render.
:func:`render`
    a real Blender (``blender --background``) Cycles render of that scene, coloured by a field. This
    is the "Blender look" asked for: physical materials, an area light, contact shadows, and the
    field mapped through a colour ramp — not a matplotlib surface.
:func:`render_video`
    an orbit around the nominal solution, one Cycles frame per step, assembled with ``ffmpeg``.

The cut is the ``sym0`` plane of the mesh, coloured by the cell it bounds, so the picture shows the
field *inside* the passage and not only on its skin. The walls are revolved to four copies for the
render: the passage is a solid of revolution and the run used one 90 deg sector, so revolving them
is exact, and the one cut plane is then unmistakably a cut.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess

import numpy as np

__all__ = ["build_scene", "render", "render_video", "FIELDS"]

#: colour scale, unit and label per field name
FIELDS = {
    "Ma": ("-", "Número de Mach"),
    "U_m_s": ("m/s", "Velocidade"),
    "p_bar": ("bar", "Pressão estática"),
    "T_K": ("K", "Temperatura estática"),
}

_BLENDER_SCRIPT = r'''
import bpy, sys, math, os
args = sys.argv[sys.argv.index("--") + 1:]
ply, out_dir, frames, orbit, samples, res, title = (args[0], args[1], int(args[2]), float(args[3]),
                                                    int(args[4]), int(args[5]), args[6])
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
try:
    bpy.ops.wm.ply_import(filepath=ply)          # Blender 4.x and later
except (AttributeError, RuntimeError):
    bpy.ops.import_mesh.ply(filepath=ply)        # Blender 3.x
ob = bpy.context.object
me = ob.data
attr = me.color_attributes[0] if len(me.color_attributes) else None
mat = bpy.data.materials.new("field")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
if attr is not None:
    col = nt.nodes.new("ShaderNodeVertexColor")
    col.layer_name = attr.name
    nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.35
me.materials.append(mat)
verts = np.array([v.co[:] for v in me.vertices]) if False else None
co = [v.co for v in me.vertices]
xs = [c[0] for c in co]; ys = [c[1] for c in co]; zs = [c[2] for c in co]
cx, cy, cz = 0.5*(min(xs)+max(xs)), 0.5*(min(ys)+max(ys)), 0.5*(min(zs)+max(zs))
r = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
# camera: three-quarter view, orbiting by 'orbit' radians per frame
bpy.ops.object.camera_add()
cam = bpy.context.object
sc.camera = cam
cam.data.lens = 70
bpy.ops.object.empty_add(location=(cx, cy, cz))
pivot = bpy.context.object
cam.parent = pivot
cam.location = (0.95*r, -0.80*r, 0.62*r)
cam.rotation_euler = (math.radians(65), 0, math.radians(48))
# three-point lighting
for loc, energy, size in (((0.9*r, -0.9*r, 1.6*r), 260, 1.0*r),
                          ((-1.3*r, -0.7*r, 1.0*r), 90, 0.9*r),
                          ((0.1*r, 1.4*r, 1.2*r), 70, 0.9*r)):
    bpy.ops.object.light_add(type='AREA', location=(cx+loc[0], cy+loc[1], cz+loc[2]))
    lamp = bpy.context.object
    lamp.data.energy = energy
    lamp.data.size = size
bpy.ops.mesh.primitive_plane_add(size=8*r, location=(cx, cy, cz - 0.55*r))
floor = bpy.context.object
fm = bpy.data.materials.new("floor"); fm.use_nodes = True
fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.09, 0.10, 0.12, 1)
fm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
floor.data.materials.append(fm)
# a title and a scale bar as text
if title:
    bpy.ops.object.text_add(location=(cx - 1.05*r, cy - 1.05*r, cz + 0.75*r))
    txt = bpy.context.object
    txt.data.body = title
    txt.data.size = 0.075*r
    txt.data.align_x = 'LEFT'
    tm = bpy.data.materials.new("txt"); tm.use_nodes = True
    tm.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (0.95, 0.97, 1.0, 1)
    tm.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 2.0
    txt.data.materials.append(tm)
sc.render.engine = 'CYCLES'
sc.cycles.samples = samples
sc.cycles.device = 'CPU'
sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = res, int(res*2/3)
sc.render.film_transparent = False
sc.world = bpy.data.worlds.new("w")
sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.045, 0.055, 0.07, 1)
sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.9
os.makedirs(out_dir, exist_ok=True)
step = 2*math.pi/max(frames, 1) if orbit > 0 else 0.0
for k in range(max(frames, 1)):
    pivot.rotation_euler = (0, 0, k*step)
    sc.render.filepath = os.path.join(out_dir, "frame_%04d.png" % k)
    bpy.ops.render.render(write_still=True)
    print("FRAME", sc.render.filepath, flush=True)
'''


def build_scene(passage, fields: dict[str, np.ndarray], title: str = "", *, revolve: int = 4):
    """A :class:`pinneapple_twin3d.Scene` of the passage: walls (revolved) and one meridian cut."""
    from pinneapple_twin3d import Scene

    from .post import patch_triangles

    sc = Scene(title or passage.title or "passage", length_unit="m", source="OpenFOAM rhoSimpleFoam")
    case = fields["case_dir"]
    wall_v, wall_t, wall_cell = patch_triangles(case, "wall")
    cut_v, cut_t, cut_cell = patch_triangles(case, "sym0")

    # revolve the walls: exact, the passage is a solid of revolution
    if revolve > 1:
        vs, ts = [], []
        for k in range(revolve):
            a = 2 * math.pi * k / revolve
            r = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
            vs.append(wall_v @ r.T)
            ts.append(wall_t + k * len(wall_v))
        wall_v, wall_t = np.vstack(vs), np.vstack(ts)
        wall_cell = np.tile(wall_cell, revolve)

    sc.add_part("passage", wall_v, wall_t, color=(0.22, 0.45, 0.62))
    sc.add_part("corte", cut_v, cut_t, color=(0.85, 0.86, 0.88))

    per_cell = {"U_m_s": fields["Um"], "Ma": fields["Ma"], "p_bar": fields["p"] / 1e5, "T_K": fields["T"]}
    for name, values in per_cell.items():
        sc.add_field("passage", "_" + name, values[wall_cell], unit=FIELDS[name][0])
        sc.add_field("corte", name, values[cut_cell], unit=FIELDS[name][0])
    return sc


def render(scene, out: str, field: str = "Ma", *, samples: int = 48, resolution: int = 1100,
           title: str | None = None, blender: str = "blender", timeout: int = 1800) -> str:
    """One Cycles frame of ``scene`` coloured by ``field``. Returns the PNG path."""
    return _blender(scene, os.path.dirname(os.path.abspath(out)) or ".", os.path.basename(out),
                    frames=1, orbit=0.0, samples=samples, resolution=resolution, field=field,
                    title=title, blender=blender, timeout=timeout)


def render_video(scene, out_mp4: str, field: str = "Ma", *, frames: int = 36, samples: int = 16,
                 resolution: int = 720, title: str | None = None, fps: int = 24,
                 blender: str = "blender", timeout: int = 7200) -> str:
    """An orbit around the solution, assembled into ``out_mp4`` with ffmpeg."""
    folder = os.path.splitext(out_mp4)[0] + "_frames"
    _blender(scene, folder, "", frames=frames, orbit=2 * math.pi, samples=samples,
             resolution=resolution, field=field, title=title, blender=blender, timeout=timeout)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return folder                       # the frames are the deliverable without ffmpeg
    subprocess.run([ffmpeg, "-y", "-framerate", str(fps), "-i", os.path.join(folder, "frame_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_mp4],
                   capture_output=True, text=True, timeout=1800, check=False)
    return out_mp4 if os.path.exists(out_mp4) else folder


def _blender(scene, out_dir: str, name: str, *, frames: int, orbit: float, samples: int,
             resolution: int, field: str, title: str | None, blender: str, timeout: int) -> str:
    """Write the scene as a coloured PLY and let a real Blender render it."""
    work = os.path.join(out_dir, "_blender")
    os.makedirs(work, exist_ok=True)
    ply = os.path.join(work, "scene.ply")
    _write_ply(scene, ply, field)
    script = os.path.join(work, "render.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write("import numpy as np\n" + _BLENDER_SCRIPT)
    label = title if title is not None else (f"{FIELDS[field][1]} [{FIELDS[field][0]}]" if field in FIELDS else field)
    lo, hi = _field_range(scene, field)
    label = f"{label}   {lo:.4g} .. {hi:.4g}"
    exe = shutil.which(blender)
    if exe is None:
        raise FileNotFoundError(f"'{blender}' not on PATH: install Blender to render the 3D pictures")
    subprocess.run([exe, "--background", "--python", script, "--", ply, work, str(frames),
                    str(orbit), str(samples), str(resolution), label],
                   capture_output=True, text=True, timeout=timeout, check=False)
    if frames == 1:
        src = os.path.join(work, "frame_0000.png")
        dst = os.path.join(out_dir, name) if name else src
        if name:
            shutil.move(src, dst)
        return dst
    # the video step wants the frames where it looks for them
    for k in range(frames):
        src = os.path.join(work, f"frame_{k:04d}.png")
        if os.path.exists(src):
            shutil.move(src, os.path.join(out_dir, f"frame_{k:04d}.png"))
    return out_dir


def _field_range(scene, field: str) -> tuple[float, float]:
    vals = [p.fields[field] for p in scene.parts if field in p.fields]
    if not vals:
        return 0.0, 1.0
    return float(np.nanmin([np.nanmin(v) for v in vals])), float(np.nanmax([np.nanmax(v) for v in vals]))


def _write_ply(scene, path: str, field: str) -> None:
    """ASCII PLY with per-vertex colours from ``field`` (viridis), readable by Blender's importer."""
    verts, faces, colors = [], [], []
    lo, hi = _field_range(scene, field)
    span = max(hi - lo, 1e-30)
    base = 0
    for part in scene.parts:
        if field not in part.fields:
            continue
        vals = np.asarray(part.fields[field], dtype=float).reshape(-1)
        rgb = _viridis((vals - lo) / span)
        verts.append(part.vertices)
        faces.append(part.faces + base)
        colors.append(rgb)
        base += len(part.vertices)
    v = np.vstack(verts)
    f = np.vstack(faces)
    c = np.vstack(colors)
    with open(path, "w", encoding="ascii") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(v)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write(f"element face {len(f)}\nproperty list uchar int vertex_index\nend_header\n")
        for (x, y, z), (r, g, b) in zip(v, c, strict=False):
            fh.write(f"{x:.6g} {y:.6g} {z:.6g} {int(r)} {int(g)} {int(b)}\n")
        for tri in f:
            fh.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def _viridis(t: np.ndarray) -> np.ndarray:
    """Viridis through a handful of anchors — no dependency on matplotlib's colormap registry."""
    anchors = np.array([[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
                        [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]],
                       dtype=float)
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    pos = t * (len(anchors) - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(anchors) - 1)
    w = (pos - i0)[:, None]
    return anchors[i0] * (1 - w) + anchors[i1] * w
