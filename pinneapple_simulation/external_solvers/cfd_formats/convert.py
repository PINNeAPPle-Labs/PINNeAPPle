"""Format dispatch for CFD / FEM / CAD files: a portable point bundle from any supported file, a triangle surface
from the formats that carry connectivity, and repair / simplification on the way out.

* :func:`detect_format` -- file suffix to format name (``.cgns``, ``.exo``/``.e``, ``.inp``, ``.msh`` (Fluent),
  ``.step``/``.stp``, ``.stl``, ``.obj``, ``.ply``).
* :func:`read_point_bundle` -- node coordinates + every field array + the reader's provenance string, for the
  volume formats (CGNS, Exodus II, Abaqus ``.inp``, Fluent ASCII mesh), through the readers of this package. The
  CGNS, Exodus and Fluent readers were written to the published specs and not yet checked against a file from
  the real writers (see the package docstring); their element connectivity is not returned, so these files give
  points, not surfaces.
* :func:`read_surface_mesh` -- a ``MeshData`` triangle surface for STL / OBJ / PLY (trimesh), STEP (CadQuery
  tessellation, optional ``cadquery``) and Abaqus ``.inp``: triangle shell elements (S3, STRI3, M3D3, ...) as they
  are, tetrahedra (C3D4, C3D10: corner nodes) reduced to their boundary faces (faces used by one tet only).
  Node ids are mapped to row indices.
* :func:`convert_surface` -- read, optionally repair (``pinneapple_design.geometry.ops.repair.repair_mesh``:
  duplicate and degenerate faces, normals) and simplify (``simplify_mesh``), and write STL / OBJ / PLY.

Moved here from the PINNeAPPle-apps ``cfd_format_converter`` product (MeshBridge), with two fixes: Abaqus
connectivity used the node ids as row indices, and a tetrahedral mesh was turned into one face per tet instead of
its boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SUFFIX_TO_FORMAT", "POINT_FORMATS", "SURFACE_FORMATS", "SURFACE_OUTPUTS", "detect_format",
           "read_point_bundle", "read_surface_mesh", "convert_surface", "tet_boundary_faces"]

SUFFIX_TO_FORMAT = {".cgns": "cgns", ".exo": "exodus", ".e": "exodus", ".inp": "abaqus", ".msh": "fluent",
                    ".step": "step", ".stp": "step", ".stl": "stl", ".obj": "obj", ".ply": "ply"}
POINT_FORMATS = ("cgns", "exodus", "abaqus", "fluent")
SURFACE_FORMATS = ("abaqus", "step", "stl", "obj", "ply")
SURFACE_OUTPUTS = ("stl", "obj", "ply")
_TRI_SHELLS = ("S3", "STRI3", "STRI65", "M3D3", "R3D3", "SFM3D3", "CPS3", "CPE3", "DS3", "SFMGAX1")
_TETS = ("C3D4", "C3D10", "DC3D4", "DC3D10")


def detect_format(filename: str) -> str:
    """Format name for ``filename``'s suffix; ``ValueError`` listing the supported ones otherwise."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUFFIX_TO_FORMAT:
        raise ValueError(f"unrecognised extension {suffix!r}; supported: {sorted(SUFFIX_TO_FORMAT)}")
    return SUFFIX_TO_FORMAT[suffix]


def read_point_bundle(path: str, fmt: str | None = None) -> dict[str, Any]:
    """``{"coords": (N, d), "fields": {name: array}, "provenance": str, "format": fmt}`` for a volume format."""
    fmt = fmt or detect_format(path)
    if fmt == "cgns":
        from .cgns_reader import cgns_to_upd as reader
    elif fmt == "exodus":
        from .exodus_reader import exodus_to_upd as reader
    elif fmt == "abaqus":
        from .abaqus_reader import abaqus_inp_to_upd as reader
    elif fmt == "fluent":
        from .fluent_mesh_reader import fluent_mesh_to_upd as reader
    else:
        raise ValueError(f"{fmt!r} is a surface/CAD format: no point bundle (use read_surface_mesh)")
    sample = reader(path)
    return {"coords": np.asarray(sample.geometry.nodes), "format": fmt, "provenance": sample.provenance,
            "fields": {k: np.asarray(v) for k, v in sample.state.items()}}


def tet_boundary_faces(tets: np.ndarray) -> np.ndarray:
    """Triangles used by exactly one tetrahedron (the boundary of a conforming tet mesh), outward ordering not
    guaranteed (repair fixes the winding)."""
    tets = np.asarray(tets, dtype=np.int64)[:, :4]
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    key = np.sort(faces, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inv.ravel()] == 1]


def _abaqus_surface(path: str):
    from pinneapple_design.geometry.core.mesh import MeshData

    from .abaqus_reader import read_abaqus_inp_mesh

    parsed = read_abaqus_inp_mesh(path)
    index = {int(nid): i for i, nid in enumerate(parsed["node_ids"])}
    tris = []
    for etype, block in parsed["elements"].items():
        e = etype.upper()
        conn = [[index[int(n)] for n in row] for row in block["connectivity"]]
        if e in _TRI_SHELLS or e.startswith("S3"):
            tris.append(np.asarray([row[:3] for row in conn], dtype=np.int64))
        elif e in _TETS:
            tris.append(tet_boundary_faces(np.asarray([row[:4] for row in conn])))
    if not tris:
        raise ValueError(f"no triangular shell or tetrahedral element block in {path} "
                         f"(found {sorted(parsed['elements'])}): no surface to build")
    return MeshData(vertices=np.asarray(parsed["coords"], dtype=np.float64), faces=np.concatenate(tris), normals=None)


def read_surface_mesh(path: str, fmt: str | None = None):
    """A ``MeshData`` triangle surface for STL / OBJ / PLY / STEP / Abaqus ``.inp``."""
    fmt = fmt or detect_format(path)
    if fmt in ("stl", "obj", "ply"):
        import trimesh

        from pinneapple_design.geometry.io.trimesh_bridge import TrimeshBridge

        # process=True merges coincident vertices: STL is a triangle soup, and duplicate faces or a
        # flipped patch can only be found once neighbouring triangles share vertices
        return TrimeshBridge().from_trimesh(trimesh.load(path, force="mesh", process=True))
    if fmt == "step":
        from pinneapple_design.geometry.gen.cadquery_gen import build_mesh_from_step

        return build_mesh_from_step(path)
    if fmt == "abaqus":
        return _abaqus_surface(path)
    raise ValueError(f"{fmt!r} files are read as points only here (node coordinates and fields): "
                     f"surfaces come from {SURFACE_FORMATS}")


def convert_surface(path: str, out_path: str, *, fmt: str | None = None, repair: bool = False,
                    simplify_target_faces: int | None = None) -> dict[str, Any]:
    """Read a surface, optionally repair and simplify it, write ``out_path`` (suffix .stl / .obj / .ply)."""
    import trimesh

    out_fmt = Path(out_path).suffix.lower().lstrip(".")
    if out_fmt not in SURFACE_OUTPUTS:
        raise ValueError(f"output must be one of {SURFACE_OUTPUTS}, not {out_fmt!r}")
    mesh = read_surface_mesh(path, fmt)
    n_in = int(len(mesh.faces))
    if repair:
        from pinneapple_design.geometry.ops.repair import repair_mesh

        mesh = repair_mesh(mesh)
    if simplify_target_faces is not None:
        from pinneapple_design.geometry.ops.simplify import simplify_mesh

        mesh = simplify_mesh(mesh, target_faces=int(simplify_target_faces))
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.export(out_path)
    return {"faces_in": n_in, "faces_out": int(len(tm.faces)), "vertices_out": int(len(tm.vertices)),
            "watertight": bool(tm.is_watertight), "winding_consistent": bool(tm.is_winding_consistent),
            "output": out_fmt}
