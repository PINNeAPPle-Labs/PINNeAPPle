"""Studio hero assets for one experiment: two stills, a turntable and a glTF, in one call (#920).

An experiment ships matplotlib slices and, at best, one hand-made render. This module turns a geometry-and-field
result -- a :class:`~.scene.Scene`, a PyVista mesh or a path it can read -- into the assets a page, a slide or the
one-page report of :mod:`pinneapple_analysis.report` needs, with a preset per physics so two experiments of the same
kind look like they belong together:

    from pinneapple_tools.visualization.studio import hero
    h = hero("case.frd", "structural")                 # or external_aero, internal_flow, conjugate_heat, electromagnetic
    h["stills"]["1920x1080"], h["gltf"], h["turntable"]["mp4"]

What it writes in ``out_dir`` (default ``<cwd>/hero/``): ``still_1920x1080.png``, ``still_3840x2160.png``,
``turntable.mp4`` (when ``ffmpeg`` is on the PATH and a frame backend exists), ``hero.glb`` (the glTF the browser
viewer reads, with the field as a vertex attribute), optionally ``viewer/index.html``, and ``hero.json`` -- the same
report as the returned dict, which is what makes the call reproducible from the folder alone.

**Physically honest by construction.** The mesh is drawn as it is (no smoothing beyond the material shading), the
colour range is the field's own and is written in ``field``, and a deformation is a stated exaggeration: ``warp``
gives the field it came from, the scale factor applied to every vertex and the measured maximum displacement, and
the same sentence goes into the colour-bar title. A result with no displacement field is not deformed and
``warp`` says so; nothing here invents geometry.

**Back ends, and what happens when one is missing.** ``render(..., quality=...)`` of :mod:`.blender` (Blender
Cycles, GPU when there is one) draws the stills when ``bpy`` imports or the Blender binary is on the PATH. Without
Blender the same scenes are drawn by PyVista off-screen with the same framing, the house colour map and a colour
bar -- and the report says ``backend: "pyvista"`` with the reason in ``fallback``, because a lower-quality picture
is fine and a picture that claims to be a Cycles render is not. With neither, no still is written, ``stills`` is
empty and ``reason`` says what to install; the glTF, which needs no renderer, is still produced.

**Turntable.** The frames orbit the subject at the preset's elevation and the clip is ``seconds`` long at ``fps``.
``turntable_backend="auto"`` draws them with PyVista (a frame is milliseconds, so a 8 s / 24 fps clip is seconds of
work) and says so; ``"cycles"`` forces the Cycles frames at the same framing, which on a laptop costs tens of
seconds per frame -- the measured time per frame goes into ``turntable["per_frame_s"]`` either way.
"""
from __future__ import annotations

import glob
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["hero", "hero_from_experiment", "PRESETS", "Preset", "KINDS"]

#: the colour-bar title of each preset, the quantity whose house map the 2-D figures use (``theme.field_cmaps``) and
#: the map the Blender ramp library actually has. The house sequential map is viridis and the Blender ramps are jet,
#: magma and the diverging three; ``magma`` is the perceptually uniform sibling of viridis, so it is what a sequential
#: house map becomes in Cycles, and ``field`` says both names rather than silently using a different scale.
_SEQUENTIAL = "magma"
_DIVERGING = "rdbu"


@dataclass(frozen=True)
class Preset:
    """One physics look: which field to colour by, on which stage, with which camera and extras."""
    kind: str
    fields: tuple[str, ...]          # candidate field names, the first one present wins
    label: str
    unit: str
    quantity: str                    # the house quantity (pinneapple_tools.visualization.theme.field_cmaps)
    cmap: str
    background: str = "dark"
    camera: str = "hero"
    view: str = "iso"
    lines: bool = False              # draw the scene's Line sets (streamlines, pathlines)
    slice: bool = False              # draw the scene's slices (a cut plane, a section)
    ghost: bool = False              # the walls translucent, the field showing through
    warp: str | None = None          # the field that deforms the geometry, when the result carries one
    floor: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "fields": list(self.fields), "label": self.label, "unit": self.unit,
                "quantity": self.quantity, "cmap": self.cmap, "background": self.background,
                "camera": self.camera, "lines": self.lines, "slice": self.slice, "ghost": self.ghost,
                "warp": self.warp, "note": self.note}


#: the presets of the issue, keyed by ``kind``; the aliases at the bottom are what a caller naturally types
PRESETS: dict[str, Preset] = {
    "external_aero": Preset(
        "external_aero", ("cp", "pressure_coefficient", "p", "pressure", "pressure_pa"),
        "surface pressure coefficient Cp", "-", "pressure_coefficient", _DIVERGING, camera="product", lines=True,
        slice=True, note="surface field, streamlines and the wake slice when the result carries them"),
    "internal_flow": Preset(
        "internal_flow", ("u", "velocity", "speed", "u_magnitude", "velocity_magnitude"), "velocity magnitude", "m/s",
        "velocity_magnitude", _SEQUENTIAL, background="studio", camera="product", slice=True, lines=True,
        ghost=True, note="a cut plane through the ducts, pathlines, and the wall translucent"),
    "conjugate_heat": Preset(
        "conjugate_heat", ("t", "temperature", "temperature_k", "temp"), "temperature", "K", "temperature",
        _SEQUENTIAL, background="studio", camera="product", slice=True,
        note="surface temperature with a solid section through the wall"),
    "structural": Preset(
        "structural", ("von_mises", "stress", "s_vm", "mises", "sigma_vm"), "von Mises stress", "Pa", "stress",
        _SEQUENTIAL, background="dark", camera="product", warp="DISP",
        note="von Mises contour on the deformed shape, with the undeformed body as a ghost"),
    "electromagnetic": Preset(
        "electromagnetic", ("e", "e_magnitude", "b", "b_magnitude", "h", "j", "field"), "field magnitude", "V/m",
        "speed", _SEQUENTIAL, background="dark", camera="hero", slice=True, lines=True,
        note="field magnitude on the planes the result carries, with the vector glyphs beside them"),
}

#: what a caller may pass as ``kind``: the preset names and the natural aliases
KINDS = {**{k: k for k in PRESETS},
         "fluids": "external_aero", "fluid": "external_aero", "aero": "external_aero", "external": "external_aero",
         "aerodynamics": "external_aero", "cfd": "external_aero",
         "internal": "internal_flow", "pipe": "internal_flow", "duct": "internal_flow",
         "heat": "conjugate_heat", "cht": "conjugate_heat", "thermal": "conjugate_heat",
         "structures": "structural", "structure": "structural", "fea": "structural", "fem": "structural",
         "solid": "structural", "mechanics": "structural",
         "em": "electromagnetic", "electromagnetics": "electromagnetic", "electromagnetic_field": "electromagnetic",
         "maxwell": "electromagnetic"}

#: the sizes the issue asks for, in the order they are rendered (the small one first: it is the one to look at)
STILLS = ((1920, 1080), (3840, 2160))

# ---------------------------------------------------------------------- the result -> a Scene
def _as_scene(result: Any, field: str | None = None):
    """A :class:`~.scene.Scene` from a Scene, a PyVista mesh or a path any reader of the studio understands."""
    from .scene import Scene

    if isinstance(result, Scene):
        return result
    if isinstance(result, (str, os.PathLike)):
        path = os.fspath(result)
        if not os.path.exists(path):
            raise FileNotFoundError(f"hero: no such result file: {path}")
        return Scene.from_file(path)
    if hasattr(result, "points") and hasattr(result, "n_points"):                 # a PyVista dataset
        return Scene.from_mesh(result)
    raise TypeError(f"hero: cannot draw a {type(result).__name__}: pass a Scene, a PyVista mesh or a path "
                    f"(stl, obj, glb, vtp, vtu, frd)")


def _displacements(result: Any, scene) -> dict[str, np.ndarray]:
    """The vector displacement fields of the result, on the scene's own vertices, when the reader kept them.

    :meth:`~.scene.Scene.from_mesh` keeps a vector as its magnitude, which is what a colour map shows and not what a
    deformation needs. A CalculiX ``.frd`` holds the components, so they are put back here -- on the vertices of the
    surface the scene was built from (``compare.surface`` gives that mapping) -- and the scene is not otherwise
    touched. Anything else returns ``{}``: no direction, no deformation, and ``warp`` says it was not deformed.
    """
    path = os.fspath(result) if isinstance(result, (str, os.PathLike)) else None
    if path is None or not path.lower().endswith(".frd") or not scene.surfaces:
        return {}
    try:
        from pinneapple_data.cae.compare import surface
        from pinneapple_data.cae.frd import read_frd

        with open(path, "rb") as fh:
            mesh = read_frd(fh.read())
        vidx = np.asarray(surface(mesh)["vertex_index"])
    except Exception:                                          # not a result this reader understands
        return {}
    out = {}
    for key, val in (getattr(mesh, "point_data", None) or {}).items():
        a = np.asarray(val, float)
        if a.ndim == 2 and a.shape[1] == 3 and a.shape[0] == getattr(mesh, "n_points", -1):
            out[key] = a[vidx]
    return out


def _surface_values(scene, name: str) -> np.ndarray | None:
    """The field ``name`` across the scene's surfaces, NaN where a surface does not carry it (a vector becomes its
    magnitude, which is what a colour map can show)."""
    vals = []
    for s in scene.surfaces:
        v = None
        for key, arr in s.fields.items():
            if key.lower() == name.lower():
                v = np.asarray(arr, float)
                break
        if v is None:
            vals.append(np.full(len(s.vertices), np.nan))
        elif v.ndim == 2 and v.shape[1] == 3:
            vals.append(np.linalg.norm(v, axis=1))
        else:
            vals.append(v.reshape(len(s.vertices)))
    out = np.concatenate(vals) if vals else np.array([])
    return out if np.isfinite(out).any() else None


def _pick_field(scene, preset: Preset, field: str | None) -> str | None:
    """The field to colour by: the one asked for, else the preset's first candidate the result actually has."""
    names = {k.lower(): k for s in scene.surfaces for k in s.fields}
    if field is not None:
        if field.lower() not in names:
            raise KeyError(f"hero: no field {field!r} in the result (have: {', '.join(sorted(names)) or 'none'})")
        return names[field.lower()]
    for cand in preset.fields:
        if cand in names:
            return names[cand]
    return next(iter(names.values())) if names else None


def _deform(scene, field_name: str, exaggerate: float, vectors: dict | None = None):
    """The scene with ``V + scale * d`` and the ghost of the undeformed body; the scale is returned, never implied."""
    from .scene import Lines, Scene, Surface

    name = field_name.lower()
    disp = None
    for key, arr in (vectors or {}).items():                   # the vector the reader kept, when it did
        if key.lower() == name or key.lower() in ("disp", "displacement", "u_disp"):
            disp = (key, np.asarray(arr, float))
            break
    for s in scene.surfaces:                                   # else a vector already on the surface
        for key, arr in s.fields.items():
            if (key.lower() == name or key.lower() in ("disp", "displacement", "u_disp")):
                a = np.asarray(arr, float)
                if a.ndim == 2 and a.shape[1] == 3:
                    disp = (key, a)
                    break
        if disp:
            break
    if disp is None or len(scene.surfaces) != 1 or len(disp[1]) != len(scene.surfaces[0].vertices):
        return scene, None
    key, d = disp
    lo, hi = scene.bounds()
    diag = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))) or 1.0
    peak = float(np.nanmax(np.linalg.norm(d, axis=1))) if len(d) else 0.0
    if peak <= 0:
        return scene, None
    scale = exaggerate * diag / peak
    surfaces, wires = [], []
    for s in scene.surfaces:
        a = np.asarray(s.fields.get(key, np.zeros((len(s.vertices), 3))), float)
        if a.shape != s.vertices.shape:
            a = np.zeros_like(s.vertices)
        surfaces.append(Surface(vertices=s.vertices + scale * a, faces=s.faces, name=s.name,
                                material=s.material, fields=dict(s.fields), group=s.group, uv=s.uv))
        wires.append(Lines(points=_silhouette(s), name="undeformed", label="undeformed shape"))
    # the undeformed body enters the picture as its wireframe: a solid shell coincident with the coloured body (which
    # is what a 5 % exaggeration makes it) would only fog the colour map, and hiding it would lose the statement
    out = Scene(surfaces=surfaces, lines=scene.lines + [w for w in wires if w.points],
                axes=scene.axes, labels=dict(scene.labels), materials=dict(scene.materials))
    return out, {"field": key, "scale_factor": round(scale, 4), "exaggeration": exaggerate,
                 "max_displacement": peak, "reference_length": diag,
                 "statement": f"deformation exaggerated x{scale:,.0f} ({100 * exaggerate:g} % of the model's "
                              f"diagonal; max |u| = {peak:.3g})"}


def _silhouette(surface, angle_deg: float = 24.0) -> list[np.ndarray]:
    """The feature edges of ``surface`` as two-point polylines: the undeformed shape drawn as a wireframe."""
    V, F = np.asarray(surface.vertices, float), np.asarray(surface.faces, np.int64)
    if not len(F):
        return []
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.where(ln == 0, 1.0, ln)
    seen: dict[tuple[int, int], list[np.ndarray]] = {}
    for tri, nv in zip(F, fn, strict=True):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(tri[a]), int(tri[b])
            seen.setdefault((min(i, j), max(i, j)), []).append(nv)
    cos_lim = math.cos(math.radians(angle_deg))
    out = []
    for (i, j), normals in seen.items():
        if len(normals) != 2 or float(np.dot(normals[0], normals[1])) < cos_lim:
            out.append(np.array([V[i], V[j]]))
    return out


# ---------------------------------------------------------------------- the back ends
def _blender_available() -> tuple[bool, str]:
    """``(can_render, why_not)``: ``bpy`` in this process, else the Blender binary through ``render_headless``."""
    try:
        import bpy  # noqa: F401
        return True, ""
    except Exception:
        from .blender import blender_executable
        exe = blender_executable()
        if exe:
            return True, ""
        return False, ("no Blender: `pip install bpy` or install Blender and put it on PATH (or set $BLENDER). "
                       "The stills fall back to PyVista off-screen")


def _pyvista_available() -> tuple[bool, str]:
    try:
        from ..pyvista_bridge import can_render
    except Exception as exc:                                    # no pyvista at all
        return False, f"PyVista is not installed ({exc})"
    return (True, "") if can_render() else (False, "PyVista is installed but no off-screen OpenGL context exists "
                                                   "(on a server: xvfb-run -a python ...)")


def _cycles_still(scene, path: str, *, size: tuple[int, int], field: str | None, clim, preset: Preset, title: str,
                  samples: int | None, device: str, view: str, floor: bool) -> dict[str, Any]:
    """One Cycles render through :func:`.blender.render` / ``render_headless`` (colour bar drawn afterwards)."""
    from .blender import blender_executable, render, render_headless

    kw: dict[str, Any] = dict(background=preset.background, camera=preset.camera, view=view, size=tuple(size),
                              device=device, floor=floor, cmap=preset.cmap, colorbar_on=False, title=title)
    if samples is not None:                                    # None: the quality preset's own sample count
        kw["samples"] = int(samples)
    if field is not None:
        body = next((s for s in scene.surfaces if field in s.fields), None)
        ghosted = body is not None and any(not s.fields for s in scene.surfaces)
        if ghosted and clim is not None:
            # `field=` gives every surface without it the plain grey default, which would paint over the
            # undeformed ghost and, at a small exaggeration, over the coloured body too. Naming the surface keeps
            # the ghost's own translucent material (_build_objects built it) and still ramps the one that has data.
            kw["surface_fields"] = {body.name: (field, float(clim[0]), float(clim[1]), preset.cmap,
                                                title.replace("\n", " · "))}
        else:
            kw["field"] = field
            if clim is not None:
                kw["field_range"] = tuple(float(x) for x in clim)
    if scene.lines and (preset.lines or any(line.name == "undeformed" for line in scene.lines)):
        kw["lines"] = True
    if preset.slice and scene.slices:
        kw["slice"] = scene.slices[0].name
    if preset.ghost:
        kw["ghost"] = True
    t0 = time.time()
    if _importable("bpy"):
        render(scene, path, **kw)
        backend = "cycles (bpy, in process)"
    else:                                                       # no Pillow inside Blender: the bar is drawn here
        render_headless(scene, path, exe=blender_executable(), **kw)
        backend = "cycles (Blender binary, headless)"
    return {"path": path, "seconds": round(time.time() - t0, 2), "backend": backend}


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _frame_camera(scene, azimuth_deg: float, elevation_deg: float):
    """Where the camera stands for a turntable frame: on a sphere around the subject, at the preset's elevation."""
    lo, hi = scene.bounds()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) or 1.0
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    direction = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return centre, centre + 2.45 * radius * direction, direction


def _pyvista_frames(scene, paths: Sequence[str], *, field: str | None, clim, size: tuple[int, int], title: str,
                    preset: Preset, dark: bool, azimuths: Sequence[float], elevation: float) -> list[str]:
    """Draw the same scene off-screen with PyVista, one frame per azimuth, with the house colour map and its bar.

    The fallback of the issue: the same subject, the same framing and the same scale as the Cycles still, at a
    raster quality a CI machine can produce. It is declared as such in the returned ``backend`` -- never passed off
    as a Cycles render.
    """
    import pyvista as pv

    from ..pyvista_bridge import BACKGROUND
    from ..style import field_cmap

    cmap = field_cmap(preset.quantity)                    # the exact map the 2-D figures use
    pl = pv.Plotter(off_screen=True, window_size=[int(size[0]), int(size[1])])
    pl.set_background(BACKGROUND)
    for s in scene.surfaces:
        faces = np.hstack([np.full((len(s.faces), 1), 3), s.faces.astype(np.int64)]).ravel()
        mesh = pv.PolyData(np.asarray(s.vertices, float), faces)
        kw: dict[str, Any] = dict(smooth_shading=True, show_scalar_bar=False)
        if field is not None:
            for key, arr in s.fields.items():
                if key.lower() != field.lower():
                    continue
                a = np.asarray(arr, float)
                v = np.linalg.norm(a, axis=1) if a.ndim == 2 and a.shape[1] == 3 else a.reshape(len(s.vertices))
                mesh.point_data[field] = v
                if clim is not None:
                    kw.update(scalars=field, cmap=cmap, clim=list(clim))
                break
        if "scalars" not in kw:                 # a surface with no field is the undeformed ghost
            kw.update(color="#9aa6b8", opacity=0.25 if not s.fields else 1.0)
        pl.add_mesh(mesh, **kw)
    for line in scene.lines:                               # streamlines / pathlines, in the same neutral ink
        for i, pts in enumerate(line.points):
            vals = line.values[i] if line.values and i < len(line.values) else None
            poly = pv.lines_from_points(np.asarray(pts, float))
            if vals is None:
                pl.add_lines(np.asarray(pts, float), color="#7fb2d8", width=2)
            else:
                poly.point_data["value"] = np.asarray(vals, float)
                pl.add_mesh(poly, scalars="value", cmap=cmap, line_width=3, show_scalar_bar=False)
    for sl in scene.slices:                                # a cut plane, as a quad with its value grid
        grid = np.asarray(sl.grid, float)
        j, i = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
        s_, t_ = i / max(grid.shape[1] - 1, 1), j / max(grid.shape[0] - 1, 1)
        pts = (np.asarray(sl.origin, float)[None, None, :] + s_[..., None] * np.asarray(sl.u, float)[None, None, :]
               + t_[..., None] * np.asarray(sl.v, float)[None, None, :])
        surf = pv.StructuredGrid(pts[..., 0], pts[..., 1], pts[..., 2])
        surf.point_data["value"] = grid.ravel(order="F")
        pl.add_mesh(surf, scalars="value", cmap=cmap, clim=list(clim) if clim else None, opacity=0.85,
                    show_scalar_bar=False)
    if field is not None:
        pl.add_scalar_bar(title=title, n_labels=4, fmt="%.3g", color="white" if dark else "black",
                          vertical=False, position_x=0.60, position_y=0.04, width=0.36, height=0.06)
    elif title:
        pl.add_text(title, font_size=10, color="white" if dark else "black")
    pl.enable_anti_aliasing("ssaa")
    written = []
    for path, az in zip(paths, azimuths, strict=True):
        centre, eye, _d = _frame_camera(scene, az, elevation)
        pl.camera_position = [tuple(float(x) for x in eye), tuple(float(x) for x in centre), (0.0, 0.0, 1.0)]
        pl.screenshot(path)
        written.append(path)
    pl.close()
    return written


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _encode_mp4(frames: Sequence[str], out: str, fps: float, crf: int = 20) -> str:
    """Encode the PNG frames to H.264 with ffmpeg; the caller has already checked that ffmpeg exists."""
    exe = _ffmpeg()
    if exe is None:
        raise RuntimeError("ffmpeg is not on the PATH: the turntable frames are written but not encoded")
    pattern = os.path.join(os.path.dirname(frames[0]), "frame_%05d.png")
    proc = subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", pattern,
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), out],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"ffmpeg failed to encode {os.path.basename(out)}: "
                           f"{(proc.stderr or '').strip().splitlines()[-3:]}")
    return out


def _machine() -> dict[str, Any]:
    try:
        from pinneapple_catalog.costs import machine

        m = dict(machine())
        return {k: m.get(k) for k in ("cpu", "cpu_count", "mem_gb", "platform") if k in m}
    except Exception:
        import platform

        return {"platform": platform.platform(), "cpu_count": os.cpu_count()}


# ---------------------------------------------------------------------- the call
def _resolve_kind(kind: str) -> Preset:
    key = str(kind).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in KINDS:
        raise ValueError(f"kind={kind!r}: one of {', '.join(sorted(PRESETS))} "
                         f"(aliases: {', '.join(sorted(set(KINDS) - set(PRESETS)))})")
    return PRESETS[KINDS[key]]


def _bar_title(label: str, unit: str, warp: dict | None) -> str:
    """What the colour bar says: the quantity with its unit, and the exaggeration when the shape was scaled."""
    t = f"{label} [{unit}]" if unit and unit not in ("-", "1") else label
    return f"{t}\n{warp['statement']}" if warp else t


def hero(result: Any, kind: str = "external_aero", *, field: str | None = None, out_dir: str | None = None,
         stills: Sequence[Sequence[int]] = STILLS, samples: int | None = None, device: str = "auto",
         backend: str = "auto", turntable: bool = True, seconds: float = 8.0, fps: float = 24.0,
         turntable_backend: str = "auto", gltf: bool = True, viewer: bool = False, unit: str | None = None,
         label: str | None = None, exaggeration: float = 0.05, colormap: str | None = None,
         name: str = "hero") -> dict[str, Any]:
    """Write the hero assets of one result: two stills, a turntable and a glTF, with the preset of ``kind``.

    ``kind`` is one of ``external_aero``, ``internal_flow``, ``conjugate_heat``, ``structural``,
    ``electromagnetic`` (or the aliases ``fluids``, ``heat``, ``structures``, ``em`` ...). ``result`` is a
    :class:`~.scene.Scene`, a PyVista dataset or a path (stl, obj, glb, vtp, vtu, frd). ``field`` overrides which
    field is coloured (default: the preset's first candidate the result has, and its range from the data).
    ``unit`` and ``label`` override what the colour bar says; ``exaggeration`` is how much of the model's diagonal
    the largest displacement is drawn at (``structural``, when the result carries one; the factor and the measured
    peak go into ``warp`` and into the bar's title).

    The returned dict -- also written as ``out_dir/hero.json`` -- says what was really produced:

    ``backend``
        ``"cycles"`` when Blender drew the stills, ``"pyvista"`` when the fallback did, ``None`` when neither could.
    ``fallback``
        the reason the stills are PyVista's rather than Cycles', or ``""``; never a claim that Cycles ran when it
        did not.
    ``stills``
        ``{"1920x1080": path, "3840x2160": path}`` for the sizes that were written (empty with no backend).
    ``turntable``
        ``{"mp4", "frames", "fps", "seconds", "backend", "per_frame_s", "reason"}``; ``mp4`` is ``None`` with the
        reason when ``ffmpeg`` is missing or no frame backend exists.
    ``gltf`` / ``viewer``
        the ``.glb`` the browser viewer reads (always written: it needs no renderer) and, with ``viewer=True``,
        the three.js page beside it.
    ``field`` / ``warp`` / ``preset`` / ``hardware`` / ``seconds``
        what was coloured, how the shape was scaled, the look that was used, the machine and the measured times.

    >>> from pinneapple_tools.visualization.studio import hero                      # doctest: +SKIP
    >>> h = hero("examples/calculix_cantilever/beam.frd", "structural")              # doctest: +SKIP
    >>> h["backend"], h["warp"]["scale_factor"], h["stills"]["1920x1080"]            # doctest: +SKIP
    """
    preset = _resolve_kind(kind)
    t_start = time.time()
    scene = _as_scene(result)
    scene_src = os.fspath(result) if isinstance(result, (str, os.PathLike)) else type(result).__name__
    warp = None
    if preset.warp:
        scene, warp = _deform(scene, preset.warp, exaggeration, _displacements(result, scene))
    fname = _pick_field(scene, preset, field)
    values = _surface_values(scene, fname) if fname else None
    clim = None
    if values is not None:
        from .compose import default_range

        clim = default_range(values, preset.cmap if not colormap else colormap)
    # the preset's quantity names its own field; a field the preset does not know is labelled by its own name, never
    # by the preset's -- a pressure field must not come out titled "velocity magnitude"
    known = fname is not None and fname.lower() in [c.lower() for c in preset.fields]
    lab = label if label is not None else (preset.label if known else (fname or preset.label))
    uni = (preset.unit if known else "") if unit is None else unit
    title = _bar_title(lab, uni, warp)

    out_dir = os.path.abspath(out_dir or os.path.join(os.getcwd(), name))
    os.makedirs(out_dir, exist_ok=True)
    result_doc: dict[str, Any] = {
        "schema": "pinneapple.hero/1", "kind": preset.kind, "preset": preset.to_dict(), "source": scene_src,
        "out_dir": out_dir, "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "field": ({"name": fname, "label": lab, "unit": uni, "range": list(clim) if clim else None,
                   "house_cmap": preset.quantity, "cmap": colormap or preset.cmap} if fname else None),
        "warp": warp, "backend": None, "fallback": "", "stills": {}, "turntable": None, "gltf": None,
        "renderers": {},
        "viewer": None, "hardware": _machine(),
        "honesty": {"smoothing": "none: the mesh is drawn as it is; shading is the material's, the shape is not",
                    "colour_range": "the field's own min..max (symmetric about zero for a diverging map)",
                    "deformation": (warp or {}).get("statement") or "not deformed: the result carries no "
                                                                    "displacement field"},
        "seconds": {}}

    # -- back end for the stills
    cycles_ok, cycles_why = _blender_available()
    pv_ok, pv_why = _pyvista_available()
    result_doc["renderers"] = {"cycles": cycles_ok, "cycles_why": cycles_why, "pyvista": pv_ok,
                               "pyvista_why": pv_why, "ffmpeg": bool(_ffmpeg())}
    want = str(backend).lower()
    if want not in ("auto", "cycles", "pyvista"):
        raise ValueError(f'backend={backend!r}: "auto", "cycles" or "pyvista"')
    use_cycles = (want == "cycles") or (want == "auto" and cycles_ok)
    if use_cycles and not cycles_ok:
        if want == "cycles":
            raise RuntimeError(f"backend='cycles' was asked for but Blender cannot render here: {cycles_why}")
        use_cycles = False
    if use_cycles:
        result_doc["backend"] = "cycles"
    elif pv_ok:
        result_doc["backend"] = "pyvista"
        result_doc["fallback"] = cycles_why or "Blender was not asked for (backend='pyvista')"
    else:
        result_doc["backend"] = None
        result_doc["fallback"] = (f"no renderer: {cycles_why}; {pv_why}. The glTF and the viewer page are still "
                                  f"written, the stills are not")

    # -- stills
    bar = title
    for size in stills:
        tag = f"{int(size[0])}x{int(size[1])}"
        path = os.path.join(out_dir, f"still_{tag}.png")
        t0 = time.time()
        if use_cycles:
            _cycles_still(scene, path, size=(int(size[0]), int(size[1])), field=fname, clim=clim,
                                 preset=preset, title=bar, samples=samples, device=device,
                                 view=preset.view, floor=preset.floor)
            if not _importable("bpy"):                    # Blender's Python has no Pillow: the bar is drawn here
                from .colormap import colorbar
                from .compose import CMAPS

                lo_hi = clim or (0.0, 1.0)
                colorbar(path, title=bar.replace("\n", " · "), lo=float(lo_hi[0]), hi=float(lo_hi[1]),
                         stops=CMAPS[preset.cmap], dark=preset.background != "studio")
            result_doc["seconds"][f"still_{tag}"] = round(time.time() - t0, 2)
            result_doc["stills"][tag] = path
        elif pv_ok:
            _pyvista_frames(scene, [path], field=fname, clim=clim, size=(int(size[0]), int(size[1])), title=bar,
                            preset=preset, dark=preset.background != "studio", azimuths=[-45.0], elevation=25.0)
            result_doc["seconds"][f"still_{tag}"] = round(time.time() - t0, 2)
            result_doc["stills"][tag] = path

    # -- glTF for the web viewer (no renderer needed)
    if gltf and scene.surfaces:
        t0 = time.time()
        gpath = os.path.join(out_dir, "hero.glb")
        scene.save(gpath)
        result_doc["gltf"] = gpath
        result_doc["seconds"]["gltf"] = round(time.time() - t0, 2)
        if viewer:
            from .web import web_viewer

            t0 = time.time()
            result_doc["viewer"] = web_viewer(scene, os.path.join(out_dir, "viewer"))
            result_doc["seconds"]["viewer"] = round(time.time() - t0, 2)

    # -- turntable
    if turntable:
        n = max(2, int(round(float(seconds) * float(fps))))
        tt: dict[str, Any] = {"mp4": None, "frames": n, "fps": float(fps), "seconds": float(seconds),
                              "backend": None, "per_frame_s": None, "reason": ""}
        tb = str(turntable_backend).lower()
        if tb not in ("auto", "cycles", "pyvista"):
            raise ValueError(f'turntable_backend={turntable_backend!r}: "auto", "cycles" or "pyvista"')
        use_pv = (tb == "pyvista") or (tb == "auto" and pv_ok)
        if tb == "auto" and not pv_ok:
            use_pv = False
        if tb == "cycles" and not cycles_ok:
            tt["reason"] = f"turntable_backend='cycles' but Blender cannot render here: {cycles_why}"
        elif not (cycles_ok or pv_ok):
            tt["reason"] = f"no frame backend: {cycles_why}; {pv_why}"
        elif _ffmpeg() is None:
            tt["reason"] = "ffmpeg is not on the PATH: the frames could be drawn but not encoded to MP4"
        else:
            t0 = time.time()
            tmp = tempfile.mkdtemp(prefix="pinneapple-turntable-")
            azimuths = [360.0 * i / n for i in range(n)]
            paths = [os.path.join(tmp, f"frame_{i:05d}.png") for i in range(n)]
            size = (int(stills[0][0]), int(stills[0][1]))
            try:
                if use_pv:
                    _pyvista_frames(scene, paths, field=fname, clim=clim, size=size, title=bar, preset=preset,
                                    dark=preset.background != "studio", azimuths=azimuths, elevation=25.0)
                    tt["backend"] = "pyvista"
                else:
                    for p, az in zip(paths, azimuths, strict=True):
                        _c, eye, _d = _frame_camera(scene, az, 25.0)
                        lo_a, hi_a = scene.bounds()
                        centre = (np.asarray(lo_a, float) + np.asarray(hi_a, float)) / 2
                        view = tuple(float(x) for x in (eye - centre))      # the direction the camera looks from
                        _cycles_still(scene, p, size=size, field=fname, clim=clim, preset=preset, title=bar,
                                      samples=max(8, min(int(samples or 24), 24)), device=device,
                                      view=view, floor=preset.floor)
                    tt["backend"] = "cycles"
                tt["per_frame_s"] = round((time.time() - t0) / n, 2)
                tt["mp4"] = _encode_mp4(paths, os.path.join(out_dir, "turntable.mp4"), float(fps))
            except (RuntimeError, OSError) as exc:
                tt["reason"] = f"{type(exc).__name__}: {exc}"
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            result_doc["seconds"]["turntable"] = round(time.time() - t0, 2)
        result_doc["turntable"] = tt

    result_doc["seconds"]["total"] = round(time.time() - t_start, 2)
    result_doc["caption"] = (f"{lab}"
                             + (f" ({uni})" if uni and uni not in ("-", "1") else "")
                             + f" on {os.path.basename(scene_src) if isinstance(scene_src, str) else 'the result'}, "
                             + (warp["statement"] if warp else "drawn at its own scale")
                             + f"; {result_doc['backend'] or 'no renderer'}.")
    with open(os.path.join(out_dir, "hero.json"), "w", encoding="utf-8") as fh:
        json.dump(result_doc, fh, indent=1, default=str)
    return result_doc


def hero_from_experiment(folder: str, kind: str | None = None, **kw: Any) -> dict[str, Any]:
    """``hero()`` for an example folder: the geometry and the field the experiment's manifest points at.

    The result file is the first of the manifest's ``files.data``/``files.viewers``/``figures`` (or a ``.glb``,
    ``.frd``, ``.vtp``, ``.vtu``, ``.stl`` beside them) the studio can read; ``kind`` defaults to the manifest's
    domain ("fluid dynamics" -> ``external_aero``, "structural mechanics" -> ``structural``, ...). The asset lands in
    ``<folder>/report/hero`` so that :func:`pinneapple_analysis.report.experiment_report` picks it up on its own.
    """
    from pinneapple_lab.manifest import load

    m = load(folder)
    dom = f"{m.domain} {m.title}".lower()
    if kind is None:
        kind = ("structural" if any(k in dom for k in ("structur", "solid", "fem", "mechan", "stress")) else
                "electromagnetic" if any(k in dom for k in ("electro", "magnet", "maxwell", "em")) else
                "conjugate_heat" if any(k in dom for k in ("heat", "thermal", "temperature")) else
                "internal_flow" if any(k in dom for k in ("internal", "duct", "pipe", "channel")) else
                "external_aero")
    cands: list[str] = []
    for key in ("viewers", "data", "figures"):
        cands += [os.path.join(folder, f) for f in ((m.files or {}).get(key) or [])]
    cands += sorted(glob.glob(os.path.join(folder, "**", "*.glb"), recursive=True))
    cands += sorted(glob.glob(os.path.join(folder, "**", "*.frd"), recursive=True))
    cands += sorted(glob.glob(os.path.join(folder, "**", "*.vtp"), recursive=True))
    for path in cands:
        if path.endswith((".glb", ".gltf", ".stl", ".vtp", ".vtu", ".frd", ".obj")):
            return hero(path, kind, out_dir=kw.pop("out_dir", os.path.join(folder, "report", "hero")), **kw)
    raise FileNotFoundError(f"hero_from_experiment: no geometry the studio can read in {folder} "
                            f"(looked at {len(cands)} files across the manifest and the folder)")
