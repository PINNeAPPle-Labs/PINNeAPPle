"""PINNeAPPle STEP: a Blender extension over :mod:`pinneapple_design.geometry.brep` (#1119).

A **PINNeAPPle** panel in the 3-D viewport sidebar:
1. import a STEP file, every body as an object at its real size, with the exact body kept for re-meshing;
2. pick the mesher (Auto, Fast, Kernel) and the quality (Draft, Normal, Fine) and re-mesh the selected objects;
3. a quality report per object (watertight, self-intersections, angles, chordal error against the exact faces);
4. diagnostic shading: curvature, zebra and highlight lines from the library as vertex colours, a real-time zebra
   material driven by the reflection vector, and a striped environment for reflections;
5. bake to STL / OBJ / glTF / USD and "Send to PINNeAPPle" (``MeshData`` arrays as ``.npz``).

Written from scratch: no code from the GPL BlenderSTEP add-on. ``bpy`` is imported only in :func:`register`, so
:mod:`.core` and :mod:`.build` import and test without Blender.
"""


def register():
    from . import addon
    addon.register()


def unregister():
    from . import addon
    addon.unregister()
