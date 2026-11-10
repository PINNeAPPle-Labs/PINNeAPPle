"""Export a trusted local Blender mesh to the scientific dashboard artifact.

Run only inside an authorized Blender Python process. No arbitrary .blend
file loading, embedded script execution, or remote asset resolution occurs.
"""
from __future__ import annotations

from .scientific_artifacts import surface_artifact


def blender_mesh_artifact(obj, *, world_space: bool = True, label: str | None = None):
    """Extract evaluated triangulated geometry from a Blender MESH object.

    The caller owns the Blender scene and dependency graph. Shape keys and
    modifiers are evaluated by Blender, not executed from an uploaded payload.
    """
    if getattr(obj, "type", None) != "MESH":
        raise ValueError("Blender object must be a mesh")
    mesh = obj.to_mesh()
    if mesh is None:
        raise ValueError("unable to evaluate Blender mesh")
    try:
        mesh.calc_loop_triangles()
        vertices = []
        for vertex in mesh.vertices:
            point = obj.matrix_world @ vertex.co if world_space else vertex.co
            vertices.append(tuple(float(x) for x in point))
        triangles = [tuple(int(i) for i in tri.vertices) for tri in mesh.loop_triangles]
        artifact = surface_artifact(vertices, triangles, label=label or str(obj.name)[:120])
        artifact["kind"] = "scene"
        return artifact
    finally:
        obj.to_mesh_clear()
