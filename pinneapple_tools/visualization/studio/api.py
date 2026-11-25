"""Three calls a viewer needs, whatever the geometry came from: ``export_gltf``, ``viewer_html``, ``blender_render``.

    import pinneapple as pp

    glb = pp.viz.export_gltf("wing.stl", {"cp": cp})            # bytes: glTF 2.0 binary, field as a vertex attribute
    page = pp.viz.viewer_html(grid, {"cp": cp}, "viewer/")      # index.html + scene.glb: the three.js page
    pp.viz.blender_render("wing.stl", "wing.png", field="cp")   # Blender Cycles, jet scale, colour bar

The input is a mesh in any of the shapes this repository already produces -- an STL/OBJ/glTF/VTK/VTP file, a
``pinneapple_data.cae`` mesh, a PyVista grid (what the OpenFOAM reader of #60 returns), a pair of vertex/face arrays,
or a :class:`~.scene.Scene` -- with per-vertex or per-cell scalar and vector fields. Those are the parts of the
aero_optimizer's 3-D presentation that did not depend on being an aircraft (#281): the aircraft specifics (its parts,
materials, propeller group, axes) stay in ``pinneapple_design.aero.airframe``, which calls the same
:func:`~.scene.write_glb` this module does.

This module is deliberately thin: the work is the ``Scene`` (geometry, materials, fields as vertex attributes), the
page (:func:`~.web.web_viewer`) and the render (:func:`~.blender.render`). What it adds is the input normalisation,
so a caller with a mesh and an array does not have to know which of the five constructors it needs.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .fieldview import VIEW_MAX_FACES
from .scene import Scene, Surface, scalar_of

MAX_FACES = VIEW_MAX_FACES

__all__ = ["export_gltf", "viewer_html", "blender_render", "to_scene", "MAX_FACES"]

_PYVISTA_HINT = 'a PyVista mesh needs PyVista: pip install "pinneapple[pyvista]"'


# ----------------------------------------------------------------------------- the input, in whichever of its shapes
def _norm_fields(fields) -> dict:
    """``{name: values}`` or ``[{name, values, ...}]`` (the ``field_view`` spelling) -> ``{name: values}``."""
    if fields is None:
        return {}
    if isinstance(fields, Mapping):
        return dict(fields)
    out = {}
    for f in fields:
        if not isinstance(f, Mapping) or "name" not in f or "values" not in f:
            raise TypeError(f"a field is a dict with 'name' and 'values', got {type(f).__name__}: {f!r}")
        out[str(f["name"])] = f["values"]
    return out


def _is_pyvista(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "pyvista"


def _triangle_faces(raw) -> np.ndarray:
    """The triangle list of a VTK face stream ``[n, ids..., n, ids..., ...]``: quick when all are triangles."""
    raw = np.asarray(raw, np.int64)
    if raw.size == 0:
        return np.zeros((0, 3), np.int64)
    if raw.size % 4 == 0 and np.all(raw.reshape(-1, 4)[:, 0] == 3):     # every face a triangle: [3, a, b, c, ...]
        return raw.reshape(-1, 4)[:, 1:]
    tris, i = [], 0                                                   # a general stream, fan-triangulated
    while i < raw.size:
        n = int(raw[i])
        for k in range(1, n - 1):
            tris.append((raw[i + 1], raw[i + 1 + k], raw[i + 2 + k]))
        i += n + 1
    return np.asarray(tris, np.int64)


def _scene_from_pyvista(obj, fields: Mapping[str, Any], *, name: str, material: str, axes: str, title: str) -> Scene:
    """A PyVista grid or surface -> ``Scene``: its boundary triangles, the fields kept per vertex or averaged from cells.

    A volume grid of VTK polyhedra (an OpenFOAM case, #60) has no vertex-ordered surface, so its boundary is extracted
    first and a cell field is interpolated to the nodes there -- the value a coloured node should carry -- then
    triangulated. A surface whose fields are per face keeps them per face and they are averaged onto the vertices of
    each triangle: a vertex colour cannot carry a different value on two triangles that share the vertex, and averaging
    is the only honest way to say so.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(_PYVISTA_HINT) from exc
    if isinstance(obj, pv.MultiBlock):                       # the patches of an OpenFOAM boundary: one surface
        blocks = [b for b in obj if b is not None and getattr(b, "n_cells", 0)]
        if not blocks:
            raise ValueError("the MultiBlock has no blocks with cells")
        obj = blocks[0].copy() if len(blocks) == 1 else blocks[0].merge(blocks[1:])
    o = obj.copy()
    n_pts, n_cells = int(o.n_points), int(o.n_cells)
    per_cell = set()
    for key, values in fields.items():                       # attach by row count: one value per vertex or per cell
        a = np.asarray(values, float)
        a = a.reshape(len(a), -1) if a.ndim > 1 else a       # a vector field is one row per point, not one number
        if len(a) == n_pts:
            o.point_data[str(key)] = a
        elif len(a) == n_cells:
            o.cell_data[str(key)] = a
            per_cell.add(str(key))
        else:
            raise ValueError(f"field {key!r}: {len(a)} values for {n_pts} points and {n_cells} cells")
    if not isinstance(o, pv.PolyData):                       # a volume: interpolate to the nodes, then its boundary
        if per_cell:
            o = o.cell_data_to_point_data()
        try:
            o = o.extract_surface(algorithm="dataset_surface")
        except TypeError:                                    # a PyVista older than the keyword: same default
            o = o.extract_surface()
    tri = o.triangulate()
    V = np.asarray(tri.points, float).reshape(-1, 3)
    F = _triangle_faces(tri.faces)
    if not len(F):
        raise ValueError("the surface has no faces")
    surf = Surface(name, V, F, material)
    for key in fields:
        key = str(key)
        if key in tri.point_data:
            surf.fields[key] = scalar_of(np.asarray(tri.point_data[key], float))
        elif key in tri.cell_data:
            v = scalar_of(np.asarray(tri.cell_data[key], float))
            acc, cnt = np.zeros(len(V)), np.zeros(len(V))
            for j in range(3):
                np.add.at(acc, F[:, j], v)
                np.add.at(cnt, F[:, j], 1)
            surf.fields[key] = acc / np.maximum(cnt, 1)
        else:                                                # the field did not survive the conversion: say so
            raise ValueError(f"field {key!r} was lost on the way to the surface of a {type(obj).__name__}")
    return Scene([surf], axes=axes, title=title)


def to_scene(mesh, fields=None, *, material: str = "grey", name: str = "body", axes: str = "z_up",
             title: str = "") -> Scene:
    """A :class:`~.scene.Scene` from ``mesh``, with ``fields`` attached as per-vertex values.

    ``mesh`` is any of: a path (STL, OBJ, glTF/GLB, VTK/VTP, or anything ``pinneapple_data.cae`` reads), a PyVista
    grid or surface, a ``(vertices, faces)`` pair, a ``pinneapple_data.cae`` mesh, or an existing ``Scene``.
    ``fields``: ``{name: values}`` (per vertex, or per cell -- decided by the lengths) with values that may be vectors
    (their magnitude is what a colour map shows) or ``{name: {surface_name: values}}`` for a multi-surface scene.
    """
    fl = _norm_fields(fields)
    if isinstance(mesh, Scene):
        sc = mesh
    elif isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        V, F = np.asarray(mesh[0], float).reshape(-1, 3), np.asarray(mesh[1], np.int64).reshape(-1, 3)
        sc = Scene.from_arrays(V, F, name=name, material=material, axes=axes, title=title or "mesh")
    elif isinstance(mesh, (str, os.PathLike)):
        sc = Scene.from_file(str(mesh), material=material, axes=axes)
        sc.title = title or sc.title
    elif _is_pyvista(mesh):
        return _scene_from_pyvista(mesh, fl, name=name, material=material, axes=axes, title=title or "mesh")
    elif hasattr(mesh, "poly") or hasattr(mesh, "cell_data"):
        sc = Scene.from_mesh(mesh, material=material, name=name, axes=axes, title=title or "mesh")
    else:
        raise TypeError(f"cannot read a {type(mesh).__name__} as a mesh: pass a file, a PyVista object, "
                        f"(vertices, faces), a pinneapple_data.cae mesh or a Scene")
    if not fl:
        return sc
    per_surface = all(isinstance(v, Mapping) for v in fl.values())
    for key, values in fl.items():
        if per_surface:
            for s in sc.surfaces:
                if s.name in values:
                    s.fields[str(key)] = scalar_of(np.asarray(values[s.name], float))
            continue
        a = scalar_of(np.asarray(values, float))
        for s in sc.surfaces:                              # one field over the whole scene: same values per vertex
            if len(a) == len(s.vertices):
                s.fields[str(key)] = a
            elif len(sc.surfaces) == 1:
                raise ValueError(f"field {key!r}: {len(a)} values for {len(s.vertices)} vertices")
    return sc


# ----------------------------------------------------------------------------------------------- glTF, page, render
def export_gltf(mesh, fields=None, *, out: str | None = None, materials: dict | None = None,
                groups: Mapping[str, Sequence[int]] | None = None, generator: str = "PINNeAPPle",
                **scene_kw) -> bytes:
    """The mesh as glTF 2.0 binary, every field a float vertex attribute ``_NAME`` (upper case, what the viewers read).

    Returns the bytes; ``out`` also writes them to a file (``.glb``). One mesh per material, PBR materials with
    clearcoat and transmission from :data:`~.scene.MATERIALS`, so the same file opens in the browser viewer, in Blender
    and in Omniverse. A field's colour range is not baked in: the viewer computes it (``web_viewer`` writes the ranges
    in ``scene.json``), because a range baked into the geometry cannot be changed without re-exporting. ``groups``:
    node name -> indices of the surfaces to parent under one node (a propeller that spins, a set of parts that moves
    together), which is what the airliner viewer needs and why this is not just ``Scene.save``.
    """
    sc = to_scene(mesh, fields, **scene_kw)
    if groups:
        from .scene import AXES, MATERIALS, write_glb
        data = write_glb(sc.surfaces, AXES[sc.axes], {**MATERIALS, **sc.materials, **(materials or {})},
                         generator=generator, groups=dict(groups))
    else:
        data = sc.to_glb(materials)
    if not data.startswith(b"glTF"):
        raise RuntimeError("the glTF writer returned something that is not a GLB")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(data)
    return data


def viewer_html(mesh, fields=None, out_dir: str | None = None, *, single_file: bool = False,
                max_faces: int | None = MAX_FACES, title: str | None = None, **scene_kw) -> str:
    """Write the interactive three.js page for ``mesh``; returns the path of its ``index.html``.

    ``out_dir`` defaults to a temporary folder (the page is then throw-away until it is served). ``single_file=True``
    writes one self-contained page -- the geometry, the viewer and three.js inlined -- which opens from ``file://`` and
    can be mailed; without it the folder needs a local server, because browsers do not load ES modules from disk.
    Surfaces above ``max_faces`` triangles are decimated for the page (the scene itself is not changed).
    """
    from .web import web_viewer

    sc = to_scene(mesh, fields, title=title or "", **scene_kw)
    if out_dir is None:
        import tempfile
        out_dir = tempfile.mkdtemp(prefix="pinneapple_viewer_")
    return web_viewer(sc, out_dir, title=title, max_faces=max_faces, single_file=single_file)


def blender_render(mesh, out: str, *, fields=None, field: str | None = None, field_range=None, cmap: str = "jet",
                   samples: int = 64, size: Sequence[int] = (1600, 900), headless: bool | None = None,
                   **scene_kw) -> str:
    """A Blender Cycles picture of ``mesh`` from the same data as the page; returns the path written.

    ``field`` picks one of ``fields`` for the colour (the first when omitted), ``field_range`` pins the scale instead
    of taking it from the data. ``headless``: run the Blender binary instead of ``bpy`` (None decides: ``bpy`` when it
    imports, else the binary, else the reason it cannot). A machine with neither gets a RuntimeError that says so,
    never a picture that was not rendered.
    """
    from . import blender as _b

    sc = to_scene(mesh, fields, **scene_kw)
    if field is None and sc.surfaces and sc.surfaces[0].fields:
        field = next(iter(sc.surfaces[0].fields))
    if field is not None and not any(field in s.fields for s in sc.surfaces):
        raise KeyError(f"no field {field!r} in the scene (have: "
                       f"{sorted({k for s in sc.surfaces for k in s.fields})})")
    kw: dict[str, Any] = dict(samples=samples, size=tuple(size), cmap=cmap)
    if field is not None:
        kw["field"] = field
    if field_range is not None:
        kw["field_range"] = tuple(field_range)
    if headless is None:
        try:
            import bpy  # noqa: F401
            headless = False
        except Exception:
            headless = _b.blender_executable() is not None
    if not headless:
        try:
            import bpy  # noqa: F401
        except Exception as exc:
            raise RuntimeError("neither bpy nor the Blender binary is available: pip install bpy==4.2.0, or install "
                               "Blender and put it on PATH (or pass headless=False with bpy)") from exc
        return _b.render(sc, out, **kw)
    if _b.blender_executable() is None:
        raise RuntimeError("no Blender executable found: install Blender (BLENDER env var or PATH), or "
                           "pip install bpy==4.2.0 to render in-process")
    return _b.render_headless(sc, out, **kw)
