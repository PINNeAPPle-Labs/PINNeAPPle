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

__all__ = ["build_scene", "build_scene_from_case", "render", "render_video", "FIELDS"]

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
(ply, out_dir, frames, orbit, samples, res, title, cx, cy, cz, rad, height) = (
    args[0], args[1], int(args[2]), float(args[3]), int(args[4]), int(args[5]), args[6],
    float(args[7]), float(args[8]), float(args[9]), float(args[10]), int(float(args[11])))
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
try:
    bpy.ops.wm.ply_import(filepath=ply)          # Blender 4.x and later
except (AttributeError, RuntimeError):
    bpy.ops.import_mesh.ply(filepath=ply)        # Blender 3.x
ob = bpy.context.object
me = ob.data
attr = me.color_attributes[0] if len(me.color_attributes) else None
# The field is the shading: a mild emission mix keeps the colour map readable at the values it
# encodes instead of turning a dark-red region into a black one through the lighting.
mat = bpy.data.materials.new("field")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
if attr is not None:
    col = nt.nodes.new("ShaderNodeVertexColor")
    col.layer_name = attr.name
    nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Metallic"].default_value = 0.0
    nt.links.new(col.outputs["Color"], bsdf.inputs["Emission Color"])
    # A emissao carrega a cor exata do mapa; a luz so da forma. Sem isso a exposicao muda o matiz
    # e o que se le na figura deixa de ser o valor que ela codifica.
    bsdf.inputs["Emission Strength"].default_value = 1.0
me.materials.append(mat)
for p in me.polygons:
    p.use_smooth = False
r = rad
# camera: three-quarter view on the valve body, orbiting by 'orbit' radians per frame
bpy.ops.object.camera_add()
cam = bpy.context.object
sc.camera = cam
cam.data.lens = 60
bpy.ops.object.empty_add(location=(cx, cy, cz))
pivot = bpy.context.object
cam.parent = pivot
cam.location = (1.7*r, -2.1*r, 1.9*r)
cam.rotation_euler = (math.radians(56), 0, math.radians(39))
# three-point lighting, scaled with the framing distance so the exposure does not depend on the
# size of the part (the same scene must look the same on a 30 mm valve and on a 300 mm duct)
gain = 8500*r*r*(float(args[12]) if len(args) > 12 else 0.004)
for loc, energy, size in (((0.9*r, -1.0*r, 1.5*r), gain, 0.8*r),
                          ((-1.2*r, -0.8*r, 0.9*r), 0.4*gain, 0.7*r),
                          ((0.2*r, 1.3*r, 1.1*r), 0.3*gain, 0.7*r)):
    bpy.ops.object.light_add(type='AREA', location=(cx+loc[0], cy+loc[1], cz+loc[2]))
    lamp = bpy.context.object
    lamp.data.energy = energy
    lamp.data.size = size
bpy.ops.mesh.primitive_plane_add(size=9*r, location=(cx, cy, cz - 2.2*r))
floor = bpy.context.object
fm = bpy.data.materials.new("floor"); fm.use_nodes = True
fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.055, 0.06, 0.075, 1)
fm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.7
floor.data.materials.append(fm)
floor.hide_render = True
if title:
    bpy.ops.object.text_add(location=(cx - 1.15*r, cy - 0.9*r, cz + 0.95*r))
    txt = bpy.context.object
    txt.data.body = title
    txt.data.size = 0.055*r
    txt.data.align_x = 'LEFT'
    tm = bpy.data.materials.new("txt"); tm.use_nodes = True
    tm.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (0.92, 0.95, 1.0, 1)
    tm.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 1.4
    txt.data.materials.append(tm)
sc.render.engine = 'CYCLES'
sc.cycles.samples = samples
sc.cycles.device = 'CPU'
sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = res, height
sc.render.film_transparent = False
sc.view_settings.view_transform = 'Standard'      # AgX desaturates the colour map
sc.view_settings.look = 'None'
sc.world = bpy.data.worlds.new("w")
sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.035, 0.042, 0.055, 1)
sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.55
os.makedirs(out_dir, exist_ok=True)
step = 2*math.pi/max(frames, 1) if orbit > 0 else 0.0
for k in range(max(frames, 1)):
    pivot.rotation_euler = (0, 0, k*step)
    sc.render.filepath = os.path.join(out_dir, "frame_%04d.png" % k)
    bpy.ops.render.render(write_still=True)
    print("FRAME", sc.render.filepath, flush=True)
'''


def build_scene(passage, fields: dict[str, np.ndarray], title: str = "", *, revolve: int = 4,
                half_open: str = "z"):
    """A :class:`pinneapple_twin3d.Scene` of the passage: walls (revolved) and one meridian cut.

    ``revolve`` copies the sector round the axis and ``half_open`` names the axis along which the
    near half of the walls is dropped (``""`` keeps the closed body): the valve is a solid of
    revolution, so a closed shell hides the very thing the reader wants to see. Removing one half
    turns the picture into the usual cutaway — the meridian cut face, and the far shell curving
    away around it.
    """
    from pinneapple_twin3d import Scene

    from .post import patch_triangles

    sc = Scene(title or passage.title or "passage", length_unit="m", source="OpenFOAM rhoSimpleFoam")
    case = fields["case_dir"]
    wall_v, wall_t, wall_cell = patch_triangles(case, "wall")
    try:
        cut_v, cut_t, cut_cell = patch_triangles(case, "sym0")
    except (OSError, ValueError, KeyError):
        cut_v, cut_t, cut_cell = np.zeros((0, 3)), np.zeros((0, 3), dtype=int), np.zeros(0, dtype=int)
    if len(cut_t) == 0:
        revolve = 1          # malha 3-D: o corpo já é inteiro, não há setor a repetir

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

    if half_open in ("x", "y", "z"):
        axis = "xyz".index(half_open)
        centroid = wall_v[wall_t].mean(axis=1)[:, axis]
        wall_t = wall_t[centroid <= 0.0]      # the values stay per vertex, so only faces go

    sc.add_part("passage", wall_v, wall_t, color=(0.22, 0.45, 0.62))
    if len(cut_t):
        sc.add_part("corte", cut_v, cut_t, color=(0.85, 0.86, 0.88))

    per_cell = {"U_m_s": fields["Um"], "Ma": fields["Ma"], "p_bar": fields["p"] / 1e5, "T_K": fields["T"]}
    for name, values in per_cell.items():
        # com o corte, o campo das paredes leva prefixo para não colidir com o do corte; sem corte
        # (malha 3-D) ele é o único, e é ele que o render procura pelo nome
        sc.add_field("passage", "_" + name if len(cut_t) else name, values[wall_cell],
                     unit=FIELDS[name][0])
        if len(cut_t):
            sc.add_field("corte", name, values[cut_cell], unit=FIELDS[name][0])
    return sc


def frame_on_passage(passage, *, zoom: float = 1.0) -> tuple[tuple[float, float, float], float]:
    """Where the camera should look, and how wide, in metres: the **body**, not the whole duct.

    A passage is 0.3 m of straight pipe with a 70 mm valve in the middle, so a render framed on the
    bounding box is a picture of a pipe. The frame is the throttling region: its centre, and a radius
    a little wider than the chamber, which is what a reader needs to see.
    """
    if passage.meta.get("frame_x_mm") is not None:
        # um domínio 3-D não é o setor de revolução: o enquadramento vem das medidas da peça
        return ((float(passage.meta["frame_x_mm"]) * 1e-3,
                 float(passage.meta.get("frame_y_mm", 0.0)) * 1e-3, 0.0),
                float(passage.meta["frame_r_mm"]) * 1e-3 * zoom)
    spans = passage.spans
    middle = spans[len(spans) // 2]
    curtain = next((s for s in spans if s.name == "curtain"), middle)
    xc = 0.5 * (curtain.x0 + curtain.x1) * 1e-3
    r_chamber = float(passage.meta.get("r_chamber_mm", curtain.radii[-1])) * 1e-3
    length = (max(s.x1 for s in spans) - min(s.x0 for s in spans)) * 1e-3
    radius = max(2.0 * r_chamber, 0.16 * length) * zoom
    return (xc, 0.0, 0.0), radius


def render(scene, out: str, field: str = "Ma", *, samples: int = 48, resolution: int = 1100,
           title: str | None = None, blender: str = "blender", timeout: int = 1800,
           frame: tuple[tuple[float, float, float], float] | None = None,
           aspect: float = 0.62, light_gain: float = 0.004) -> str:
    """One Cycles frame of ``scene`` coloured by ``field``. Returns the PNG path."""
    return _blender(scene, os.path.dirname(os.path.abspath(out)) or ".", os.path.basename(out),
                    frames=1, orbit=0.0, samples=samples, resolution=resolution, field=field,
                    title=title, blender=blender, timeout=timeout, frame=frame, aspect=aspect,
                    light_gain=light_gain)


def render_video(scene, out_mp4: str, field: str = "Ma", *, frames: int = 36, samples: int = 16,
                 resolution: int = 720, title: str | None = None, fps: int = 24,
                 blender: str = "blender", timeout: int = 7200,
                 frame: tuple[tuple[float, float, float], float] | None = None,
                 light_gain: float = 0.004) -> str:
    """An orbit around the solution, assembled into ``out_mp4`` with ffmpeg."""
    folder = os.path.splitext(out_mp4)[0] + "_frames"
    _blender(scene, folder, "", frames=frames, orbit=2 * math.pi, samples=samples,
             resolution=resolution, field=field, title=title, blender=blender, timeout=timeout,
             frame=frame, aspect=9 / 16, light_gain=light_gain)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return folder                       # the frames are the deliverable without ffmpeg
    subprocess.run([ffmpeg, "-y", "-framerate", str(fps), "-i", os.path.join(folder, "frame_%04d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", out_mp4],
                   capture_output=True, text=True, timeout=1800, check=False)
    escrito = os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0
    if escrito:
        shutil.rmtree(folder, ignore_errors=True)   # os quadros são intermediários do vídeo
        return out_mp4
    return folder


def _blender(scene, out_dir: str, name: str, *, frames: int, orbit: float, samples: int,
             resolution: int, field: str, title: str | None, blender: str, timeout: int,
             frame: tuple[tuple[float, float, float], float] | None = None,
             aspect: float = 0.62, light_gain: float = 0.004) -> str:
    """Write the scene as a coloured PLY and let a real Blender render it."""
    work = os.path.join(out_dir, "_blender")
    os.makedirs(work, exist_ok=True)
    ply = os.path.join(work, "scene.ply")
    _write_ply(scene, ply, field)
    (cx, cy, cz), radius = frame if frame else _bounding_sphere(scene)
    script = os.path.join(work, "render.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write("import numpy as np\n" + _BLENDER_SCRIPT)
    label = title if title is not None else (f"{FIELDS[field][1]} [{FIELDS[field][0]}]" if field in FIELDS else field)
    lo, hi = _field_range(scene, field)
    label = f"{label}   {lo:.4g} .. {hi:.4g}"
    exe = shutil.which(blender)
    if exe is None:
        raise FileNotFoundError(f"'{blender}' not on PATH: install Blender to render the 3D pictures")
    height = max(int(resolution * aspect) // 2 * 2, 2)   # yuv420p needs even sizes
    subprocess.run([exe, "--background", "--python", script, "--", ply, work, str(frames),
                    str(orbit), str(samples), str(resolution), label,
                    f"{cx:.9g}", f"{cy:.9g}", f"{cz:.9g}", f"{radius:.9g}", str(height),
                    f"{light_gain:.6g}"],
                   capture_output=True, text=True, timeout=timeout, check=False)
    if frames == 1:
        src = os.path.join(work, "frame_0000.png")
        dst = os.path.join(out_dir, name) if name else src
        if name:
            shutil.move(src, dst)
            _add_colorbar(dst, field, lo, hi)
        return dst
    # the video step wants the frames where it looks for them
    for k in range(frames):
        src = os.path.join(work, f"frame_{k:04d}.png")
        if os.path.exists(src):
            shutil.move(src, os.path.join(out_dir, f"frame_{k:04d}.png"))
    return out_dir


def _bounding_sphere(scene) -> tuple[tuple[float, float, float], float]:
    v = np.vstack([p.vertices for p in scene.parts])
    lo, hi = v.min(axis=0), v.max(axis=0)
    return tuple(0.5 * (lo + hi)), float(np.max(hi - lo) / 2.0)


def _add_colorbar(path: str, field: str, lo: float, hi: float) -> None:
    """Stamp the colour scale the render encodes onto the picture, so it reads without a legend.

    A render without its scale is decoration: the colours mean nothing a reader can quote. The bar
    is drawn by matplotlib and composited with PIL, which keeps Blender's job to the 3-D alone.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        return
    unit = FIELDS.get(field, ("-", field))[0]
    fig, ax = plt.subplots(figsize=(1.25, 4.6), dpi=110, constrained_layout=True)
    grad = np.linspace(0.0, 1.0, 256)[:, None]
    ax.imshow(grad, aspect="auto", origin="lower", cmap="viridis", extent=(0, 1, lo, hi))
    ax.set_xticks([]), ax.yaxis.tick_right(), ax.yaxis.set_label_position("right")
    ax.set_ylabel(f"{FIELDS.get(field, ('-', field))[1]} [{unit}]")
    ax.set_title(field, fontsize=10)
    bar = os.path.splitext(path)[0] + "_bar.png"
    fig.savefig(bar, transparent=False, facecolor="white")
    plt.close(fig)
    try:
        img, strip = Image.open(path).convert("RGBA"), Image.open(bar).convert("RGBA")
        out = Image.new("RGBA", (img.width + strip.width, img.height), (255, 255, 255, 255))
        out.paste(img, (0, 0))
        out.paste(strip, (img.width, max((img.height - strip.height) // 2, 0)))
        out.convert("RGB").save(path)
    finally:
        if os.path.exists(bar):
            os.remove(bar)


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


def build_scene_from_case(case_dir: str, fields, title: str = "", *, times=None):
    """A :class:`pinneapple_twin3d.Scene` of a case, without needing the passage that built it.

    ``fields`` is one reading of the case, or a list of them (one per time) — then the scene carries
    a **series** and the viewer gets a time slider, which is what a transient needs: rotating the
    valve while the field evolves. The geometry comes from the case's own ``wall`` patch and the
    colour from the cell each face bounds, the same rule the Blender render uses.
    """
    from pinneapple_twin3d import Scene

    from .post import patch_triangles

    leituras = fields if isinstance(fields, (list, tuple)) else [fields]
    sc = Scene(title or "resultado", length_unit="m", source="OpenFOAM",
               times=list(times) if times is not None else None)
    wall_v, wall_t, wall_cell = patch_triangles(case_dir, "wall")
    sc.add_part("parede", wall_v, wall_t, color=(0.62, 0.68, 0.78))
    for name, chave in (("Ma", "Ma"), ("p_bar", "p"), ("T_K", "T"), ("U_m_s", "Um")):
        serie = []
        for leitura in leituras:
            valores = leitura[chave]
            if chave == "p":
                valores = valores / 1e5
            serie.append(np.asarray(valores)[wall_cell])
        sc.add_field("parede", name, np.vstack(serie), unit=FIELDS.get(name, ("-", name))[0])
    if len(leituras) > 1 and times is not None:
        sc.set_step_labels([f"t = {t * 1e3:.2f} ms" for t in times])
    return sc
