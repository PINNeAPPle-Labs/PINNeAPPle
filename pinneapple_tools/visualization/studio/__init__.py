"""PINNeAPPle studio: realistic 3D visualisation of any geometry and its physics.

    import pinneapple as pp
    sc = pp.viz.Scene.from_file("result.frd")          # or an STL, an OpenFOAM case, arrays, a CFD result
    sc.save("result.glb")                               # glTF with the fields, also .usda (Omniverse) and .stl
    pp.viz.render(sc, "stress.jpg", field="STRESS")     # Blender Cycles, jet colour scale, colour bar
    pp.viz.web_viewer(sc, "viewer/")                    # interactive three.js page: surfaces, streamlines, slices
"""
from .colormap import JET, colorbar, jet
from .scene import MATERIALS, Lines, Scene, Slice, Surface, read_obj, read_stl, scalar_of

__all__ = ["Scene", "Surface", "Lines", "Slice", "MATERIALS", "read_stl", "read_obj", "scalar_of", "JET", "jet",
           "colorbar", "render", "web_viewer", "serve"]


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
