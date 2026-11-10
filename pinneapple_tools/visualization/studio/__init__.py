"""PINNeAPPle studio: realistic 3D visualisation of any geometry and its physics.

    import pinneapple as pp
    sc = pp.viz.Scene.from_file("result.frd")          # or an STL, an OpenFOAM case, arrays, a CFD result
    sc.save("result.glb")                               # glTF with the fields, also .usda (Omniverse) and .stl
    pp.viz.render(sc, "stress.jpg", field="STRESS")     # Blender Cycles, jet colour scale, colour bar
    pp.viz.render_headless(sc, "stress.png", field="STRESS")   # same, through the Blender binary (no bpy needed)
    pp.viz.web_viewer(sc, "viewer/")                    # interactive three.js page: surfaces, streamlines, slices
    pp.viz.LIGHTING_PRESETS["lunar_south_pole"]         # black sky, low hard Sun (also: studio, lunar_low_sun, ...)
    pp.viz.field_view(mesh, fields, deform=..., markers=...)   # fields, errors and deformed shapes for the apps' views

Planetary surfaces (heightfield terrain, Sun shadows, Lunar-Lambert regolith): :mod:`.planetary`, :mod:`.shadows`.
"""
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
           "web_viewer", "serve",
           "LIGHTING_PRESETS", "LightingPreset", "Dashboard", "field_view", "FieldView", "VIEW_MAX_FACES",
           "cached_view", "view_paths", "save_view", "prune_views",
           "sphere_surface", "rod_surface", "disc_surface"]


def __getattr__(name):
    if name == "Dashboard":                        # pp.viz.Dashboard: the animation-centred dashboard
        from ..dashboard import Dashboard
        return Dashboard
    raise AttributeError(name)


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
