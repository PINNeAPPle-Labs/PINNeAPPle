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
           "web_viewer", "serve", "turntable", "headless", "figure_caption",
           "LIGHTING_PRESETS", "LightingPreset", "Dashboard", "field_view", "FieldView", "VIEW_MAX_FACES",
           "cached_view", "view_paths", "save_view", "prune_views",
           "sphere_surface", "rod_surface", "disc_surface",
           "QUALITY", "DEVICE_BACKENDS", "HDRIS", "hdri_path"]


def __getattr__(name):
    if name == "Dashboard":                        # pp.viz.Dashboard: the animation-centred dashboard
        from ..dashboard import Dashboard
        return Dashboard
    if name in ("QUALITY", "DEVICE_BACKENDS", "HDRIS"):     # render() presets, backends and HDRI registry
        from . import blender as _b
        return getattr(_b, name)
    raise AttributeError(name)


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
