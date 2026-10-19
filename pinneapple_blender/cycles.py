"""In-process Cycles renderer: build a small scene from NumPy arrays and render it with Blender's ``bpy`` module.

``render.py`` drives a separate Blender installation through a subprocess; this module uses the ``bpy`` Python
module directly (``pip install bpy``, which bundles Blender and Cycles), so a scene can be described from data in
the same process: surfaces from height grids, tubes along curves, point clouds, translucent planes, ribbons, text,
a camera, lights. Rendering is headless on the CPU.

    from pinneapple_blender.cycles import CyclesScene, available
    if available():
        sc = CyclesScene(resolution=(960, 540), samples=64)
        sc.add_surface(X, Y, Z, colors)               # (ny, nx) grids, colors (ny, nx, 3) in 0..1
        sc.add_plane(z=0.0, size=(10, 6), color=(0.1, 0.3, 0.6), alpha=0.35)
        sc.camera(location=(8, -9, 6), look_at=(0, 0, 0))
        sc.render("landscape.png")

``render_frames`` renders a sequence (the caller changes the scene between frames through a callback) and
``frames_to_video`` assembles them with ffmpeg.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["available", "CyclesScene", "frames_to_video"]


def available() -> bool:
    """True when the ``bpy`` module can be imported."""
    try:
        import bpy  # noqa: F401
    except Exception:  # noqa: BLE001 -- any import failure means no Blender here
        return False
    return True


def _rgba(color, alpha=1.0):
    c = tuple(float(v) for v in color)
    return c[:3] + ((c[3],) if len(c) == 4 else (float(alpha),))


class CyclesScene:
    """A fresh Blender scene rendered with Cycles on the CPU.

    ``background`` is the world colour (also the ambient light, scaled by ``ambient``); ``samples`` the path-
    tracing samples per pixel (denoised when Blender's OpenImageDenoise is available).
    """

    def __init__(self, resolution=(960, 540), samples: int = 64, background=(0.97, 0.97, 0.96),
                 ambient: float = 0.6, transparent: bool = False):
        import bpy
        self.bpy = bpy
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.samples = int(samples)
        try:
            sc.cycles.use_denoising = True
        except AttributeError:  # pragma: no cover - older Blender
            pass
        sc.render.resolution_x, sc.render.resolution_y = int(resolution[0]), int(resolution[1])
        sc.render.resolution_percentage = 100
        sc.render.film_transparent = bool(transparent)
        sc.view_settings.view_transform = "Standard"
        world = bpy.data.worlds.new("world")
        world.use_nodes = True
        bg = world.node_tree.nodes["Background"]
        bg.inputs["Color"].default_value = _rgba(background)
        bg.inputs["Strength"].default_value = float(ambient)
        sc.world = world
        self.scene = sc
        self._n = 0

    # -- materials ------------------------------------------------------------
    def _name(self, base):
        self._n += 1
        return f"{base}_{self._n}"

    def material(self, color=(0.5, 0.5, 0.5), *, alpha: float = 1.0, roughness: float = 0.5,
                 emission: float = 0.0, transmission: float = 0.0, vertex_colors: str | None = None):
        """Principled BSDF material; ``vertex_colors`` names a colour attribute used as base colour."""
        bpy = self.bpy
        mat = bpy.data.materials.new(self._name("mat"))
        mat.use_nodes = True
        nt = mat.node_tree
        p = nt.nodes["Principled BSDF"]
        p.inputs["Base Color"].default_value = _rgba(color)
        p.inputs["Roughness"].default_value = float(roughness)
        p.inputs["Alpha"].default_value = float(alpha)
        if "Transmission Weight" in p.inputs:
            p.inputs["Transmission Weight"].default_value = float(transmission)
        if emission > 0:
            p.inputs["Emission Color"].default_value = _rgba(color)
            p.inputs["Emission Strength"].default_value = float(emission)
        if vertex_colors:
            attr = nt.nodes.new("ShaderNodeAttribute")
            attr.attribute_name = vertex_colors
            nt.links.new(attr.outputs["Color"], p.inputs["Base Color"])
        if alpha < 1.0:
            try:
                mat.surface_render_method = "BLENDED"
            except AttributeError:  # pragma: no cover
                mat.blend_method = "BLEND"
        return mat

    def _link(self, obj, mat=None):
        self.scene.collection.objects.link(obj)
        if mat is not None:
            obj.data.materials.append(mat)
        return obj

    # -- geometry -------------------------------------------------------------
    def add_mesh(self, vertices, faces, *, material=None, colors=None, smooth: bool = True, name: str = "mesh"):
        """A mesh from (n, 3) vertices and faces (lists of vertex indices); ``colors`` (n, 3) per vertex."""
        bpy = self.bpy
        me = bpy.data.meshes.new(self._name(name))
        me.from_pydata([tuple(map(float, v)) for v in np.asarray(vertices)], [], [list(map(int, f)) for f in faces])
        me.update()
        if smooth:
            for poly in me.polygons:
                poly.use_smooth = True
        if colors is not None:
            col = me.color_attributes.new(name="col", type="FLOAT_COLOR", domain="POINT")
            for i, c in enumerate(np.asarray(colors, float)):
                col.data[i].color = (float(c[0]), float(c[1]), float(c[2]), 1.0)
        obj = bpy.data.objects.new(me.name, me)
        return self._link(obj, material)

    def add_surface(self, X, Y, Z, colors=None, *, roughness: float = 0.55, alpha: float = 1.0, name="surface"):
        """Height surface from (ny, nx) grids; ``colors`` (ny, nx, 3) per vertex (0..1)."""
        X, Y, Z = (np.asarray(a, float) for a in (X, Y, Z))
        ny, nx = Z.shape
        verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        faces = [[j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i]
                 for j in range(ny - 1) for i in range(nx - 1)]
        mat = self.material(roughness=roughness, alpha=alpha, vertex_colors="col" if colors is not None else None)
        cols = None if colors is None else np.asarray(colors, float).reshape(-1, 3)
        return self.add_mesh(verts, faces, material=mat, colors=cols, name=name)

    def add_plane(self, z: float = 0.0, size=(10.0, 10.0), center=(0.0, 0.0), color=(0.2, 0.4, 0.7),
                  alpha: float = 0.35, roughness: float = 0.15, name="plane"):
        cx, cy = center
        sx, sy = size[0] / 2, size[1] / 2
        v = [(cx - sx, cy - sy, z), (cx + sx, cy - sy, z), (cx + sx, cy + sy, z), (cx - sx, cy + sy, z)]
        return self.add_mesh(v, [[0, 1, 2, 3]], material=self.material(color, alpha=alpha, roughness=roughness),
                             smooth=False, name=name)

    def add_tube(self, points, radius: float = 0.03, color=(0.2, 0.4, 0.8), *, emission: float = 0.0,
                 alpha: float = 1.0, resolution: int = 4, name="tube"):
        """A tube of the given radius along a polyline of (n, 3) points."""
        bpy = self.bpy
        cu = bpy.data.curves.new(self._name(name), "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = float(radius)
        cu.bevel_resolution = int(resolution)
        sp = cu.splines.new("POLY")
        pts = np.asarray(points, float)
        sp.points.add(len(pts) - 1)
        for i, p in enumerate(pts):
            sp.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
        obj = bpy.data.objects.new(cu.name, cu)
        return self._link(obj, self.material(color, alpha=alpha, emission=emission, roughness=0.35))

    def add_points(self, points, radius: float = 0.02, color=(0.5, 0.5, 0.5), *, alpha: float = 1.0,
                   name="points"):
        """A point cloud: one small icosphere instanced on every point (vertex instancing)."""
        import bmesh
        bpy = self.bpy
        me = bpy.data.meshes.new(self._name(name))
        me.from_pydata([tuple(map(float, p)) for p in np.asarray(points, float)], [], [])
        parent = bpy.data.objects.new(me.name, me)
        self.scene.collection.objects.link(parent)
        sm = bpy.data.meshes.new(self._name("sphere"))
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=float(radius))
        bm.to_mesh(sm)
        bm.free()
        sph = bpy.data.objects.new(sm.name, sm)
        self._link(sph, self.material(color, alpha=alpha, roughness=0.5))
        sph.parent = parent
        parent.instance_type = "VERTS"
        return parent

    def add_ribbon(self, x, lower, upper, *, y: float = 0.0, color=(0.2, 0.4, 0.8), alpha: float = 0.35,
                   axis: str = "xz", name="ribbon"):
        """A flat band between ``lower`` and ``upper`` along ``x`` (in the x-z plane at depth ``y``)."""
        x, lo, hi = (np.asarray(a, float) for a in (x, lower, upper))
        n = len(x)
        verts = [(x[i], y, lo[i]) for i in range(n)] + [(x[i], y, hi[i]) for i in range(n)]
        faces = [[i, i + 1, n + i + 1, n + i] for i in range(n - 1)]
        return self.add_mesh(verts, faces, material=self.material(color, alpha=alpha, roughness=0.3),
                             smooth=False, name=name)

    def add_text(self, text: str, location, *, size: float = 0.3, color=(0.1, 0.1, 0.1), rotation=(90, 0, 0),
                 align: str = "CENTER", emission: float = 0.0, name="text"):
        """Text object; ``emission`` > 0 makes it glow in its own colour (readable on dark scenes)."""
        bpy = self.bpy
        cu = bpy.data.curves.new(self._name(name), "FONT")
        cu.body = str(text)
        cu.size = float(size)
        cu.align_x = align
        obj = bpy.data.objects.new(cu.name, cu)
        obj.location = tuple(map(float, location))
        obj.rotation_euler = tuple(math.radians(a) for a in rotation)
        obj.visible_shadow = False                     # labels should not cast shadows on the scene
        return self._link(obj, self.material(color, roughness=0.8, emission=emission))

    # -- camera, lights, render -------------------------------------------------
    def camera(self, location, look_at=(0.0, 0.0, 0.0), *, lens: float = 50.0, ortho_scale: float | None = None):
        bpy = self.bpy
        from mathutils import Vector
        cam = bpy.data.cameras.new(self._name("cam"))
        cam.lens = float(lens)
        if ortho_scale is not None:
            cam.type = "ORTHO"
            cam.ortho_scale = float(ortho_scale)
        obj = bpy.data.objects.new(cam.name, cam)
        obj.location = tuple(map(float, location))
        obj.rotation_euler = (Vector(tuple(map(float, look_at))) - obj.location).to_track_quat("-Z", "Y").to_euler()
        self.scene.collection.objects.link(obj)
        self.scene.camera = obj
        return obj

    def fit_camera(self, lo, hi, *, direction=(0.0, -1.0, 0.35), ortho: bool = True, margin: float = 1.08,
                   lens: float = 50.0):
        """Point a camera along ``direction`` (from the scene towards the camera) at the box [lo, hi] and size it
        so the whole box is in view: orthographic (scale from the projected extent) or perspective (distance)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c = 0.5 * (lo + hi)
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(d, up)
        right = right / (np.linalg.norm(right) + 1e-12)
        upv = np.cross(right, d)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - c
        w = np.ptp(corners @ right)
        h = np.ptp(corners @ upv)
        aspect = self.scene.render.resolution_x / self.scene.render.resolution_y
        extent = max(w, h * aspect) * margin
        radius = float(np.linalg.norm(hi - lo))
        if ortho:
            return self.camera(c + d * radius * 2, c, ortho_scale=extent)
        dist = extent * lens / 36.0 + 0.5 * float(np.abs(corners @ d).max())
        return self.camera(c + d * dist, c, lens=lens)

    def light(self, kind: str = "SUN", *, rotation=(50, 10, 30), energy: float = 3.0, location=(0, 0, 10),
              size: float = 1.0):
        bpy = self.bpy
        li = bpy.data.lights.new(self._name("light"), kind)
        li.energy = float(energy)
        if kind == "AREA":
            li.size = float(size)
        obj = bpy.data.objects.new(li.name, li)
        obj.location = tuple(map(float, location))
        obj.rotation_euler = tuple(math.radians(a) for a in rotation)
        self.scene.collection.objects.link(obj)
        return obj

    def render(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.scene.render.image_settings.file_format = "PNG"
        self.scene.render.filepath = os.path.abspath(path)
        self.bpy.ops.render.render(write_still=True)
        return path

    def render_frames(self, directory: str, n_frames: int, update: Callable[[CyclesScene, int], None]
                      ) -> list[str]:
        """Render ``n_frames`` PNGs into ``directory``; ``update(scene, i)`` changes the scene before frame i."""
        os.makedirs(directory, exist_ok=True)
        out = []
        for i in range(int(n_frames)):
            update(self, i)
            out.append(self.render(os.path.join(directory, f"frame_{i:04d}.png")))
        return out


def frames_to_video(frames: Sequence[str], path: str, *, fps: int = 24, crf: int = 28) -> str:
    """Assemble PNG frames (named ``frame_%04d.png`` in one directory) into an H.264 MP4 with ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg is not on PATH")
    d = os.path.dirname(frames[0])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    os.path.join(d, "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
                    "-movflags", "+faststart", path], check=True)
    return path
