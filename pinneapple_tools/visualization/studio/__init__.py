"""PINNeAPPle studio: realistic 3D visualisation of any geometry and its physics.

    import pinneapple as pp
    sc = pp.viz.Scene.from_file("result.frd")          # or an STL, an OpenFOAM case, arrays, a CFD result
    sc.save("result.glb")                               # glTF with the fields, also .usda (Omniverse) and .stl
    pp.viz.render(sc, "stress.jpg", field="STRESS")     # Blender Cycles, jet colour scale, colour bar
    pp.viz.web_viewer(sc, "viewer/")                    # interactive three.js page: surfaces, streamlines, slices
    pp.viz.LIGHTING_PRESETS["lunar_south_pole"]         # black sky, low hard Sun (also: studio, lunar_low_sun, ...)
    pp.viz.field_view(mesh, fields, deform=..., markers=...)   # fields, errors and deformed shapes for the apps' views

Planetary surfaces (heightfield terrain, Sun shadows, Lunar-Lambert regolith): :mod:`.planetary`, :mod:`.shadows`.
"""
from .colormap import JET, colorbar, jet
from .fieldview import VIEW_MAX_FACES, FieldView, field_view
from .lighting import PRESETS as LIGHTING_PRESETS
from .lighting import LightingPreset
from .scene import MATERIALS, Lines, Scene, Slice, Surface, read_obj, read_stl, scalar_of

__all__ = ["Scene", "Surface", "Lines", "Slice", "MATERIALS", "read_stl", "read_obj", "scalar_of", "JET", "jet",
           "colorbar", "render", "web_viewer", "serve", "LIGHTING_PRESETS", "LightingPreset", "Dashboard", "field_view",
           "FieldView", "VIEW_MAX_FACES"]


def __getattr__(name):
    if name == "Dashboard":                        # pp.viz.Dashboard: the animation-centred dashboard
        from ..dashboard import Dashboard
        return Dashboard
    raise AttributeError(name)


def render(*a, **kw):
    """Blender Cycles render; see :func:`pinneapple_tools.visualization.studio.blender.render`."""
    from .blender import render as _r
    return _r(*a, **kw)


def web_viewer(*a, **kw):
    """Interactive page; see :func:`pinneapple_tools.visualization.studio.web.web_viewer`."""
    from .web import web_viewer as _w
    return _w(*a, **kw)


def serve(*a, **kw):
    """Write the viewer to a temporary folder and serve it; see :func:`.web.serve`."""
    from .web import serve as _s
    return _s(*a, **kw)
