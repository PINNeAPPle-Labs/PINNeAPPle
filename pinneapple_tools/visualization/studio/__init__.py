"""PINNeAPPle studio: realistic 3D visualisation of any geometry and its physics.

    import pinneapple as pp
    sc = pp.viz.Scene.from_file("result.frd")          # or an STL, an OpenFOAM case, arrays, a CFD result
    sc.save("result.glb")                               # glTF with the fields, also .usda (Omniverse) and .stl
    pp.viz.render(sc, "stress.jpg", field="STRESS")     # Blender Cycles, jet colour scale, colour bar
    pp.viz.render_headless(sc, "stress.png", field="STRESS")   # same, through the Blender binary (no bpy needed)
    pp.viz.web_viewer(sc, "viewer/")                    # interactive three.js page: surfaces, streamlines, slices
    pp.viz.LIGHTING_PRESETS["lunar_south_pole"]         # black sky, low hard Sun (also: studio, lunar_low_sun, ...)
    pp.viz.field_view(mesh, fields, deform=..., markers=...)   # fields, errors and deformed shapes for the apps' views
    pp.viz.export_gltf("wing.stl", {"cp": cp})          # glTF/GLB bytes, the field as a _CP vertex attribute (#281)
    pp.viz.viewer_html(grid, {"cp": cp}, "viewer/")     # the three.js page for any mesh and any field
    pp.viz.blender_render("wing.stl", "wing.png", field="cp")          # Blender Cycles, jet scale, colour bar
    pp.viz.read_openfoam("case/")                       # an OpenFOAM case as a PyVista grid with its fields (#60)
    pp.viz.compare_solutions(ref, pred, ...)           # reference, surrogate and the error map, L2 in the title (#63)

Planetary surfaces (heightfield terrain, Sun shadows, Lunar-Lambert regolith): :mod:`.planetary`, :mod:`.shadows`.
"""
from .api import MAX_FACES as SCENE_MAX_FACES
from .api import blender_render, export_gltf, to_scene, viewer_html
from .colormap import JET, colorbar, jet, magma
from .compose import CMAPS, stops
from .fieldview import VIEW_MAX_FACES, FieldView, field_view
from .lighting import PRESETS as LIGHTING_PRESETS
from .lighting import LightingPreset
from .scene import (
    MATERIALS,
    Lines,
    Scene,
    Slice,
    Surface,
    disc_surface,
    read_obj,
    read_stl,
    rod_surface,
    scalar_of,
    sphere_surface,
)
from .view_cache import cached_view, prune_views, save_view, view_paths

__all__ = ["Scene", "Surface", "Lines", "Slice", "MATERIALS", "read_stl", "read_obj", "scalar_of", "JET", "jet",
           "magma", "CMAPS", "stops", "colorbar", "render", "render_headless", "blender_executable",
           "web_viewer", "serve", "turntable", "headless", "figure_caption",
           "LIGHTING_PRESETS", "LightingPreset", "Dashboard", "field_view", "FieldView", "VIEW_MAX_FACES",
           "cached_view", "view_paths", "save_view", "prune_views",
           "sphere_surface", "rod_surface", "disc_surface",
           "QUALITY", "DEVICE_BACKENDS", "HDRIS", "hdri_path", "SHADERS", "bake_materials",
           "bake_materials_headless", "benchmark", "detail", "bevel_scene", "manufacture", "solid",
           # #60: an OpenFOAM case, a .foam pointer or a VTK file as a PyVista grid
           "read_openfoam", "read_openfoam_patches", "read_grid", "read_vtk", "openfoam_times", "case_folder",
           # #63: reference, surrogate and the error map, with the L2 error as a number
           "compare_solutions", "field_error_metrics",
           # #281: a mesh and its fields -> glTF, a browser page, a Blender render
           "export_gltf", "viewer_html", "blender_render", "to_scene", "SCENE_MAX_FACES"]

_REEXPORT = {   # name -> (submodule of .visualization, attribute): imported on first use
    "read_openfoam": ("openfoam", "read_openfoam"),
    "read_openfoam_patches": ("openfoam", "read_openfoam_patches"),
    "read_grid": ("openfoam", "read_grid"),
    "read_vtk": ("openfoam", "read_vtk"),
    "openfoam_times": ("openfoam", "openfoam_times"),
    "case_folder": ("openfoam", "case_folder"),
    "compare_solutions": ("fields", "compare_solutions"),
    "field_error_metrics": ("fields", "field_error_metrics"),
}


def __getattr__(name):
    if name in _REEXPORT:                          # the readers and the comparison live one level up, not here
        import importlib
        mod, attr = _REEXPORT[name]
        return getattr(importlib.import_module(".." + mod, __name__), attr)
    if name == "Dashboard":                        # pp.viz.Dashboard: the animation-centred dashboard
        from ..dashboard import Dashboard
        return Dashboard
    if name in ("QUALITY", "DEVICE_BACKENDS", "HDRIS"):     # render() presets, backends and HDRI registry
        from . import blender as _b
        return getattr(_b, name)
    if name == "SHADERS":                                   # the procedural recipes of the material library
        from .pbr import SHADERS
        return SHADERS
    if name == "benchmark":                                 # the reference scenes and their records
        import importlib                                    # not `from . import`: that would land back here
        return importlib.import_module(".benchmark", __name__)
    if name == "detail":                                    # bolts, welds and cast seams: pp.viz.detail.hex_bolt(...)
        import importlib
        return importlib.import_module(".detail", __name__)
    if name == "manufacture":                               # a body with a manufacturing history: pp.viz.manufacture
        import importlib
        return importlib.import_module(".manufacture", __name__)
    if name == "solid":                                     # closed solids, loft and a real boolean: pp.viz.solid
        import importlib
        return importlib.import_module(".solid", __name__)
    raise AttributeError(name)


def bake_materials(*a, **kw):
    """Bake the procedural material shaders into base colour, roughness and normal maps; see
    :func:`.bake.bake_materials` (needs bpy)."""
    from .bake import bake_materials as _b
    return _b(*a, **kw)


def bake_materials_headless(*a, **kw):
    """The same through the Blender binary, for a machine without bpy; see :func:`.bake.bake_materials_headless`."""
    from .bake import bake_materials_headless as _b
    return _b(*a, **kw)


def bevel_scene(*a, **kw):
    """The same scene with its sharp edges chamfered -- a mesh operation, not a surface beside the body; see
    :func:`.bevel.bevel_scene`."""
    from .bevel import bevel_scene as _b
    return _b(*a, **kw)


def hdri_path(*a, **kw):
    """The image behind ``render(background="hdri:<name>")``; see :func:`.blender.hdri_path`."""
    from .blender import hdri_path as _h
    return _h(*a, **kw)


def turntable(*a, **kw):
    """A GIF or MP4 in which the camera orbits while the field advances in time; see
    :func:`pinneapple_tools.visualization.orbit.turntable`."""
    from ..orbit import turntable as _t
    return _t(*a, **kw)


def figure_caption(*a, **kw):
    """The caption a figure carries: field, unit, range, exaggeration, time, method and status; see
    :func:`pinneapple_tools.visualization.caption.figure_caption`. A unit is required -- ``"-"`` is how a
    dimensionless field says so."""
    from ..caption import figure_caption as _f
    return _f(*a, **kw)


def headless(*a, **kw):
    """What this machine needs to render without a display, and whether it can; see
    :func:`pinneapple_tools.visualization.headless.headless`."""
    from ..headless import headless as _h
    return _h(*a, **kw)


def render_field(*a, **kw):
    """Evaluate a model on a domain and write the picture and the VTU of the same numbers (#61); see
    :func:`pinneapple_tools.visualization.fields.render_field`."""
    from ..fields import render_field as _r
    return _r(*a, **kw)


def write_vtu(*a, **kw):
    """Write an unstructured grid in VTK XML format (#61); see
    :func:`pinneapple_tools.visualization.fields.write_vtu`."""
    from ..fields import write_vtu as _w
    return _w(*a, **kw)


def render(*a, **kw):
    """Blender Cycles render; see :func:`pinneapple_tools.visualization.studio.blender.render`."""
    from .blender import render as _r
    return _r(*a, **kw)


def render_headless(*a, **kw):
    """Blender Cycles render from the command line, for a machine without bpy; see
    :func:`pinneapple_tools.visualization.studio.blender.render_headless`."""
    from .blender import render_headless as _r
    return _r(*a, **kw)


def blender_executable(*a, **kw):
    """The Blender executable ``render_headless`` uses (or None); see :func:`.blender.blender_executable`."""
    from .blender import blender_executable as _b
    return _b(*a, **kw)


def web_viewer(*a, **kw):
    """Interactive page; see :func:`pinneapple_tools.visualization.studio.web.web_viewer`."""
    from .web import web_viewer as _w
    return _w(*a, **kw)


def serve(*a, **kw):
    """Write the viewer to a temporary folder and serve it; see :func:`.web.serve`."""
    from .web import serve as _s
    return _s(*a, **kw)
