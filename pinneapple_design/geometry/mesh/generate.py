"""Unstructured mesh generation with gmsh: triangles in 2-D, tetrahedra (and prism layers at the walls) in 3-D, from
a parametric primitive or a geometry file, with the quality of every element.

    from pinneapple_design.geometry.mesh.generate import generate_mesh
    out = generate_mesh("cylinder", [0.5, 2.0], h=0.1, out_dir="mesh_out")
    out["n_elements"], out["sicn_min"], out["volume"]          # tetrahedra, worst element, 1.567 m3 (pi r^2 H)
    out["files"]                                              # mesh.msh, mesh.vtu, quality.glb, quality.json

Geometry (lengths in metres):

=============  =======================  =========================================================================
``box``        ``[lx, ly, lz]``         corner at the origin
``cylinder``   ``[radius, height]``     base centred at the origin, axis +z
``sphere``     ``[radius]``             centred at the origin
``rectangle``  ``[lx, ly]``             2-D, corner at the origin
``disk``       ``[radius]``             2-D, centred at the origin
a file         ``.stl``                 a closed surface (open or non-manifold edges are refused); meshed with
                                        :func:`.mesher_3d.mesh_from_surface`
               ``.step .stp .iges``     through OpenCASCADE (gmsh's OCC kernel); ``scale`` converts to metres
=============  =======================  =========================================================================

``h`` is the target element size; ``refine`` adds balls of smaller size (``{"center", "radius", "h"}``) with a smooth
transition back to ``h``; ``boundary_layer`` (``{"first_layer", "n_layers", "growth"}``) adds layers at the walls:
prisms extruded inward from the surface in 3-D, a layer of triangles along the boundary in 2-D.

Quality comes from gmsh (``minSICN``: signed inverse condition number, 1 for the ideal element, <= 0 inverted;
``gamma``: inscribed / circumscribed radius, normalised to 1 for the ideal element) and from
:func:`pinneapple_data.cae.quality.element_metrics` (``aspect_ratio`` = longest / shortest edge of each element).
The view is :func:`pinneapple_tools.visualization.studio.fieldview.field_view` with those three fields per element.
"""
from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["generate_mesh", "estimate_elements", "PRIMITIVES", "FILE_TYPES", "X_OUTPUTS"]

PRIMITIVES = {"box": (3, 3), "cylinder": (3, 2), "sphere": (3, 1), "rectangle": (2, 2), "disk": (2, 1)}
DEFAULT_DIMENSIONS = {"box": [1.0, 1.0, 1.0], "cylinder": [0.5, 1.0], "sphere": [0.5], "rectangle": [1.0, 1.0],
                      "disk": [0.5]}
FILE_TYPES = (".stl", ".step", ".stp", ".iges", ".igs")
# gmsh element type -> (VTK name, dimension, nodes)
_GMSH_TYPES = {2: ("triangle", 2, 3), 3: ("quad", 2, 4), 4: ("tetra", 3, 4), 5: ("hexahedron", 3, 8),
               6: ("wedge", 3, 6), 7: ("pyramid", 3, 5)}
SICN_WARN = 0.1        # below this an element is poor for most finite-element and finite-volume solvers
MAX_INPUT_MB = 100.0   # default size limit of a geometry file; max_input_mb or PINNEAPPLE_MESH_MAX_INPUT_MB override it
MAX_ELEMENTS = 2_000_000
# elements per (volume / h^3) and per (area / h^2), measured on gmsh meshes of boxes, cylinders, squares and disks
# (4.6-5.9 tetrahedra, 2.3-2.4 triangles; tests/test_mesh_generation.py) and rounded up: the guard errs on the safe side
TETS_PER_H3 = 6.0
TRIS_PER_H2 = 2.5

# what the tool returns, with units (read by the Studio catalog)
X_OUTPUTS = {
    "generate_mesh": [
        {"name": "dim", "type": "int", "unit": "-", "description": "dimension of the mesh (2 or 3)"},
        {"name": "n_nodes", "type": "int", "unit": "-", "description": "nodes"},
        {"name": "n_elements", "type": "int", "unit": "-", "description": "elements of the mesh's dimension"},
        {"name": "element_counts", "type": "dict", "unit": "-", "description": "elements per type (tetra, wedge, ...)"},
        {"name": "sicn_min", "type": "float", "unit": "-",
         "description": "worst signed inverse condition number (gmsh minSICN; 1 ideal, <= 0 inverted)"},
        {"name": "sicn_mean", "type": "float", "unit": "-", "description": "mean signed inverse condition number"},
        {"name": "gamma_min", "type": "float", "unit": "-",
         "description": "worst inscribed / circumscribed radius ratio (gmsh gamma; 1 ideal)"},
        {"name": "gamma_mean", "type": "float", "unit": "-", "description": "mean gamma"},
        {"name": "aspect_ratio_max", "type": "float", "unit": "-",
         "description": "largest longest-to-shortest edge ratio of an element"},
        {"name": "aspect_ratio_mean", "type": "float", "unit": "-", "description": "mean edge ratio"},
        {"name": "volume", "type": "float", "unit": "m3", "description": "volume of the 3-D mesh (null in 2-D)"},
        {"name": "area", "type": "float", "unit": "m2", "description": "area of the 2-D mesh, or of the 3-D boundary"},
        {"name": "h_min", "type": "float", "unit": "m", "description": "shortest edge"},
        {"name": "h_mean", "type": "float", "unit": "m", "description": "mean edge length"},
        {"name": "h_max", "type": "float", "unit": "m", "description": "longest edge"},
        {"name": "files", "type": "dict", "unit": "",
         "description": "paths of mesh.msh, mesh.vtu, quality.glb and quality.json (empty without out_dir)"},
        {"name": "backend", "type": "str", "unit": "", "description": "how the mesh was made"}],
    "estimate_elements": [
        {"name": "n", "type": "float", "unit": "-", "description": "elements gmsh will make (upper estimate)"}],
}


def _require_gmsh():
    try:
        import gmsh
    except ImportError:
        raise RuntimeError("generate_mesh needs gmsh, which is not installed: pip install gmsh "
                           "(or pip install \"pinneapple[cfd]\")") from None
    return gmsh


@contextlib.contextmanager
def _gmsh_session(name: str):
    gmsh = _require_gmsh()
    if gmsh.isInitialized():
        gmsh.finalize()
    try:
        gmsh.initialize(readConfigFiles=False, interruptible=False)  # no SIGINT handler: works off the main thread
    except TypeError:                                                 # gmsh < 4.11
        gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("General.NumThreads", 1)              # deterministic element order
        gmsh.model.add(name)
        yield gmsh
    finally:
        gmsh.finalize()


def _primitive(gmsh, shape: str, d: list[float], o: list[float]) -> int:
    occ = gmsh.model.occ
    x, y, z = o
    if shape == "box":
        occ.addBox(x, y, z, d[0], d[1], d[2])
    elif shape == "cylinder":
        occ.addCylinder(x, y, z, 0, 0, d[1], d[0])
    elif shape == "sphere":
        occ.addSphere(x, y, z, d[0])
    elif shape == "rectangle":
        occ.addRectangle(x, y, z, d[0], d[1])
    else:
        occ.addDisk(x, y, z, d[0], d[0])
    occ.synchronize()
    return PRIMITIVES[shape][0]


def _import_cad(gmsh, path: str, scale: float) -> int:
    try:
        ents = gmsh.model.occ.importShapes(path)
    except Exception as e:  # noqa: BLE001 - gmsh raises a plain Exception
        raise ValueError(f"{os.path.basename(path)}: could not be read through OpenCASCADE ({e})") from None
    if scale != 1.0:
        gmsh.model.occ.dilate(ents, 0, 0, 0, scale, scale, scale)
    gmsh.model.occ.synchronize()
    dim = max((d for d, _ in gmsh.model.getEntities()), default=0)
    if dim < 2:
        raise ValueError(f"{os.path.basename(path)}: no surface or volume in the file")
    return dim


def _sizes(gmsh, h: float, refine: list[dict[str, Any]]) -> None:
    opt = gmsh.option
    for k in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        opt.setNumber(f"Mesh.{k}", 0)
    if not refine:
        opt.setNumber("Mesh.MeshSizeMin", h)
        opt.setNumber("Mesh.MeshSizeMax", h)
        return
    f = gmsh.model.mesh.field
    balls = []
    for r in refine:
        c = [float(x) for x in r["center"]] + [0.0] * (3 - len(r["center"]))
        b = f.add("Ball")
        f.setNumber(b, "Radius", float(r["radius"]))
        f.setNumber(b, "Thickness", float(r.get("transition", r["radius"])))
        f.setNumber(b, "VIn", float(r["h"]))
        f.setNumber(b, "VOut", h)
        for k, v in zip(("XCenter", "YCenter", "ZCenter"), c, strict=True):
            f.setNumber(b, k, v)
        balls.append(b)
    m = f.add("Min")
    f.setNumbers(m, "FieldsList", balls)
    f.setAsBackgroundMesh(m)
    opt.setNumber("Mesh.MeshSizeMin", min(float(r["h"]) for r in refine))
    opt.setNumber("Mesh.MeshSizeMax", h)


def _bl_2d(gmsh, bl: dict[str, Any]) -> None:
    n, h1, g = int(bl["n_layers"]), float(bl["first_layer"]), float(bl.get("growth", 1.2))
    f = gmsh.model.mesh.field
    b = f.add("BoundaryLayer")
    f.setNumbers(b, "CurvesList", [t for _, t in gmsh.model.getEntities(1)])
    f.setNumber(b, "Size", h1)
    f.setNumber(b, "Ratio", g)
    f.setNumber(b, "Thickness", h1 * sum(g ** i for i in range(n)))
    f.setNumber(b, "Quads", 0)
    f.setAsBoundaryLayer(b)


def _bl_3d_volume(gmsh, vertices: np.ndarray, faces: np.ndarray, bl: dict[str, Any]) -> None:
    """Replace the model by the closed triangulated surface, extrude prism layers inward and fill the core with
    tetrahedra."""
    n, h1, g = int(bl["n_layers"]), float(bl["first_layer"]), float(bl.get("growth", 1.2))
    gmsh.clear()
    gmsh.model.add("bl")
    s = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, s, list(range(1, len(vertices) + 1)), vertices.ravel().tolist())
    gmsh.model.mesh.addElementsByType(s, 2, [], (faces + 1).ravel().tolist())
    gmsh.model.mesh.classifySurfaces(math.pi / 4, True, True, math.pi)
    gmsh.model.mesh.createGeometry()
    gmsh.option.setNumber("Geometry.ExtrudeReturnLateralEntities", 0)
    heights = np.cumsum([h1 * g ** i for i in range(n)])
    # negative heights: extrude against the outward normal, into the solid
    out = gmsh.model.geo.extrudeBoundaryLayer(gmsh.model.getEntities(2), [1] * n, (-heights).tolist(), True)
    inner = [t for d, t in out if d == 2]
    top = inner[::2] if len(inner) == 2 * len(gmsh.model.getEntities(2)) else inner
    gmsh.model.geo.synchronize()
    loop = gmsh.model.geo.addSurfaceLoop(top)
    gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()


def _generate(gmsh, dim: int, boundary_layer: dict[str, Any] | None) -> None:
    try:
        gmsh.model.mesh.generate(dim)
    except Exception as e:  # noqa: BLE001 - gmsh raises a plain Exception
        if boundary_layer:
            raise ValueError(f"gmsh could not mesh with this boundary layer ({e}): the layers, extruded inward, "
                             "overlap near sharp edges; use a thinner layer (first_layer, n_layers, growth) or a "
                             "larger h") from None
        raise RuntimeError(f"gmsh could not mesh the geometry: {e}") from None


def _surface_of_model(gmsh) -> tuple[np.ndarray, np.ndarray]:
    gmsh.model.mesh.generate(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    index = np.zeros(int(tags.max()) + 1, int)
    index[tags.astype(int)] = np.arange(len(tags))
    _, nodes = gmsh.model.mesh.getElementsByType(2)
    return np.asarray(xyz, float).reshape(-1, 3), index[np.asarray(nodes, int)].reshape(-1, 3)


def _read_stl(path: str, scale: float):
    from pinneapple_data.cae.io import read_stl
    from pinneapple_data.cae.quality import surface_checks
    m = read_stl(Path(path).read_bytes())
    chk = surface_checks(m)
    if not chk.get("watertight"):
        raise ValueError(f"{os.path.basename(path)}: the surface is not closed ({chk.get('open_edges', '?')} open "
                         f"edges, {chk.get('nonmanifold_edges', '?')} non-manifold edges); a volume mesh needs a "
                         "closed, manifold surface")
    tri = np.concatenate([b.corners for b in m.blocks if b.dim == 2])
    return np.asarray(m.points, float) * scale, tri.astype(int)


def _load_elements(gmsh, vertices: np.ndarray, cells: dict[str, np.ndarray], dim: int) -> None:
    """Put a mesh made elsewhere into the current model (discrete entity) so gmsh can rate its elements."""
    e = gmsh.model.addDiscreteEntity(dim)
    gmsh.model.mesh.addNodes(dim, e, list(range(1, len(vertices) + 1)), np.asarray(vertices, float).ravel().tolist())
    code = {v[0]: k for k, v in _GMSH_TYPES.items()}
    for t, c in cells.items():
        gmsh.model.mesh.addElementsByType(e, code[t], [], (np.asarray(c, int) + 1).ravel().tolist())


def _collect(gmsh, dim: int) -> dict[str, Any]:
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    tags = tags.astype(int)
    index = np.zeros(int(tags.max()) + 1, int)
    index[tags] = np.arange(len(tags))
    pts = np.asarray(xyz, float).reshape(-1, 3)
    blocks, sicn, gamma, measure = [], [], [], []
    for code, (name, d, k) in _GMSH_TYPES.items():
        if d != dim:
            continue
        etags, enodes = gmsh.model.mesh.getElementsByType(code)
        if not len(etags):
            continue
        blocks.append((name, index[np.asarray(enodes, int)].reshape(-1, k)))
        sicn.append(gmsh.model.mesh.getElementQualities(etags, "minSICN"))
        gamma.append(gmsh.model.mesh.getElementQualities(etags, "gamma"))
        measure.append(np.abs(gmsh.model.mesh.getElementQualities(etags, "volume")))
    if not blocks:
        raise RuntimeError(f"gmsh produced no {dim}-D elements")
    return {"points": pts, "blocks": blocks, "sicn": np.concatenate(sicn), "gamma": np.concatenate(gamma),
            "measure": float(np.concatenate(measure).sum())}


def _stats(v: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(v.min()), float(v.mean()), float(v.max())) if len(v) else (math.nan, math.nan, math.nan)


def _primitive_size(shape: str, d: list[float]) -> tuple[int, float, float]:
    """(dimension, volume or area, boundary area or perimeter) of a primitive."""
    if shape == "box":
        return 3, d[0] * d[1] * d[2], 2 * (d[0] * d[1] + d[1] * d[2] + d[0] * d[2])
    if shape == "cylinder":
        return 3, math.pi * d[0] ** 2 * d[1], 2 * math.pi * d[0] * (d[0] + d[1])
    if shape == "sphere":
        return 3, 4 / 3 * math.pi * d[0] ** 3, 4 * math.pi * d[0] ** 2
    if shape == "rectangle":
        return 2, d[0] * d[1], 2 * (d[0] + d[1])
    return 2, math.pi * d[0] ** 2, 2 * math.pi * d[0]


def estimate_elements(dim: int, measure: float, boundary: float, h: float, refine: list[dict[str, Any]],
                      boundary_layer: dict[str, Any] | None) -> float:
    """Elements gmsh will make, from the measured densities (an upper estimate within about 30 %)."""
    if dim == 3:
        n = TETS_PER_H3 * measure / h ** 3
        n += sum(TETS_PER_H3 * 4 / 3 * math.pi * float(r["radius"]) ** 3 / float(r["h"]) ** 3 for r in refine)
        if boundary_layer:                                   # one prism per wall triangle and layer
            n += TRIS_PER_H2 * boundary / h ** 2 * int(boundary_layer["n_layers"])
    else:
        n = TRIS_PER_H2 * measure / h ** 2
        n += sum(TRIS_PER_H2 * math.pi * float(r["radius"]) ** 2 / float(r["h"]) ** 2 for r in refine)
        if boundary_layer:                                   # two triangles per boundary edge and layer
            n += 2 * boundary / h * int(boundary_layer["n_layers"])
    return n


def _check_size(dim: int, measure: float, boundary: float, h: float, refine, boundary_layer, max_elements: int,
                what: str) -> None:
    n = estimate_elements(dim, measure, boundary, h, refine, boundary_layer)
    if n > max_elements:
        coef = TETS_PER_H3 if dim == 3 else TRIS_PER_H2
        h_ok = (coef * measure / max_elements) ** (1 / dim)
        raise ValueError(f"{what}: about {n:.3g} elements at h = {h:g} m, above max_elements = {max_elements:g}; "
                         f"use h >= {h_ok:.3g} m (or raise max_elements, if the memory allows)")


def _check_box(points: np.ndarray, dim: int, h: float, refine, boundary_layer, max_elements: int, what: str) -> None:
    """The element guard for a file: its bounding box stands for the solid (an upper bound)."""
    size = np.ptp(np.asarray(points, float), axis=0)
    if dim == 3:
        measure, boundary = float(np.prod(size)), 2 * float(size[0] * size[1] + size[1] * size[2] + size[0] * size[2])
    else:
        s2 = np.sort(size)[1:]
        measure, boundary = float(s2[0] * s2[1]), 2 * float(s2.sum())
    _check_size(dim, measure, boundary, h, refine, boundary_layer, max_elements, what)


def _boundary_area(points: np.ndarray, blocks: list[tuple[str, np.ndarray]]) -> float:
    """Area of the faces that belong to one element only (the boundary of a 3-D mesh)."""
    from pinneapple_data.cae.model import CELL_FACES
    faces = [c[:, list(f)] for t, c in blocks for f in CELL_FACES[t]]
    area = 0.0
    for k in (3, 4):
        fk = [f for f in faces if f.shape[1] == k]
        if not fk:
            continue
        f = np.concatenate(fk)
        _, inv, cnt = np.unique(np.sort(f, 1), axis=0, return_inverse=True, return_counts=True)
        b = f[cnt[inv.ravel()] == 1]
        p = points[b]
        tri = [(0, 1, 2)] if k == 3 else [(0, 1, 2), (0, 2, 3)]
        for i, j, m in tri:
            area += 0.5 * float(np.linalg.norm(np.cross(p[:, j] - p[:, i], p[:, m] - p[:, i]), axis=1).sum())
    return area


def _edges(points: np.ndarray, blocks: list[tuple[str, np.ndarray]]) -> np.ndarray:
    from pinneapple_data.cae.model import CELL_EDGES
    e = np.concatenate([np.sort(c[:, list(pair)], 1) for t, c in blocks for pair in CELL_EDGES[t]])
    e = np.unique(e, axis=0)
    return np.linalg.norm(points[e[:, 1]] - points[e[:, 0]], axis=1)


def generate_mesh(geometry: str = "box", dimensions: list[float] | None = None, h: float = 0.25,
                  scale: float = 1.0, refine: list[dict[str, Any]] | None = None,
                  boundary_layer: dict[str, Any] | None = None, out_dir: str | None = None,
                  origin: list[float] | None = None, max_input_mb: float | None = None,
                  max_elements: int = MAX_ELEMENTS) -> dict[str, Any]:
    """Mesh a primitive or a geometry file with gmsh and rate every element.

    Parameters
    ----------
    geometry : str
        ``box``, ``cylinder``, ``sphere``, ``rectangle`` or ``disk``, or the path of an ``.stl``, ``.step``,
        ``.stp``, ``.iges`` or ``.igs`` file
    dimensions : list[float]
        size of the primitive [m]: box [lx, ly, lz], cylinder [radius, height], sphere [radius], rectangle
        [lx, ly], disk [radius] (default: a unit box, 0.5 m radii)
    h : float
        target element size [m]
    scale : float
        factor from the file's length unit to metres (1e-3 for millimetres) [-]
    refine : list[dict]
        balls of smaller elements, ``{"center": [x, y, z], "radius": r, "h": h_local}`` (metres)
    boundary_layer : dict
        ``{"first_layer": m, "n_layers": int, "growth": ratio}``: prism layers at the walls (3-D) or triangle
        layers along the boundary (2-D)
    out_dir : str
        folder for mesh.msh, mesh.vtu, quality.glb and quality.json (nothing is written without it)
    origin : list[float]
        where the primitive's reference point (the corner of a box or rectangle, the centre of a sphere or disk, the
        base centre of a cylinder) goes [m] (default: the origin)
    max_input_mb : float
        largest geometry file accepted, checked before the file is opened [MB] (default:
        PINNEAPPLE_MESH_MAX_INPUT_MB, else 100)
    max_elements : int
        refuse, before meshing, a mesh estimated above this many elements (from the volume or area over h^d, the
        refinement balls and the wall layers; files: their bounding box) [-]

    Returns
    -------
    dim : int [-]
        dimension of the mesh
    n_nodes : int [-]
        nodes
    n_elements : int [-]
        elements of the mesh's dimension
    element_counts : dict [-]
        elements per type
    sicn_min : float [-]
        worst signed inverse condition number (1 ideal, <= 0 inverted)
    sicn_mean : float [-]
        mean signed inverse condition number
    gamma_min : float [-]
        worst inscribed / circumscribed radius ratio (1 ideal)
    gamma_mean : float [-]
        mean gamma
    aspect_ratio_max : float [-]
        largest longest-to-shortest edge ratio
    aspect_ratio_mean : float [-]
        mean edge ratio
    volume : float [m3]
        volume of the 3-D mesh (null in 2-D)
    area : float [m2]
        area of the 2-D mesh, or of the 3-D boundary
    h_min : float [m]
        shortest edge
    h_mean : float [m]
        mean edge length
    h_max : float [m]
        longest edge
    files : dict
        paths of the written files
    backend : str
        how the mesh was made
    """
    if not (isinstance(h, (int, float)) and math.isfinite(h) and h > 0):
        raise ValueError("h must be a positive length in metres")
    refine = list(refine or [])
    for r in refine:
        if not {"center", "radius", "h"} <= set(r) or float(r["h"]) <= 0 or float(r["radius"]) <= 0:
            raise ValueError("each refinement needs center, radius > 0 and h > 0")
    if boundary_layer:
        bl = boundary_layer
        if float(bl.get("first_layer", 0)) <= 0 or int(bl.get("n_layers", 0)) < 1 or float(bl.get("growth", 1.2)) < 1:
            raise ValueError("boundary_layer needs first_layer > 0, n_layers >= 1 and growth >= 1")
    o = [float(x) for x in (origin or [])] + [0.0] * (3 - len(origin or []))
    if len(o) != 3 or not all(math.isfinite(x) for x in o):
        raise ValueError("origin needs up to 3 finite coordinates (metres)")
    shape = str(geometry).strip()
    ext = os.path.splitext(shape)[1].lower()
    if shape in PRIMITIVES:
        d = [float(x) for x in (dimensions or DEFAULT_DIMENSIONS[shape])]
        if len(d) != PRIMITIVES[shape][1] or min(d) <= 0:
            raise ValueError(f"{shape} needs {PRIMITIVES[shape][1]} positive dimensions (metres), got {d}")
        _check_size(*_primitive_size(shape, d), float(h), refine, boundary_layer, int(max_elements), shape)
    elif ext in FILE_TYPES:
        if "://" in shape:
            raise ValueError("geometry must be a local file path; URLs are not fetched")
        if not os.path.isfile(shape):
            raise ValueError(f"no such file: {shape}")
        limit = float(max_input_mb if max_input_mb is not None else os.environ.get("PINNEAPPLE_MESH_MAX_INPUT_MB",
                                                                                   MAX_INPUT_MB))
        size = os.path.getsize(shape) / 1e6
        if size > limit:
            raise ValueError(f"{os.path.basename(shape)} is {size:.1f} MB, above the {limit:g} MB limit "
                             "(max_input_mb or PINNEAPPLE_MESH_MAX_INPUT_MB)")
    else:
        raise ValueError(f"geometry must be one of {sorted(PRIMITIVES)} or a {'/'.join(FILE_TYPES)} file")

    _require_gmsh()
    backend = "gmsh-occ"
    vol = None
    if ext == ".stl" and not boundary_layer:
        if refine:
            raise ValueError("refine needs a boundary_layer or a STEP/IGES file when the geometry is an STL surface")
        # the existing surface-to-volume mesher (gmsh, then tetgen); it runs its own gmsh session, so before ours
        from pinneapple_design.geometry.core.mesh import MeshData

        from .mesher_3d import mesh_from_surface
        verts, tris = _read_stl(shape, float(scale))
        _check_box(verts, 3, float(h), refine, None, int(max_elements), os.path.basename(shape))
        vol = mesh_from_surface(MeshData(vertices=verts.astype(np.float32), faces=tris.astype(np.int32)),
                                mesh_size=float(h))
        if vol is None or vol.tets is None or not len(vol.tets):
            raise RuntimeError("no volume mesher produced tetrahedra for this surface (install gmsh or tetgen)")
    with _gmsh_session("generate_mesh") as gmsh:
        gmsh.option.setNumber("Mesh.Algorithm", 6)                  # Frontal-Delaunay (2-D)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)                # Delaunay (3-D)
        gmsh.option.setNumber("Mesh.Optimize", 1)                   # gmsh's tetrahedron optimiser (Netgen is optional)
        if ext == ".stl":
            dim = 3
            if boundary_layer:
                verts, tris = _read_stl(shape, float(scale))
                _check_box(verts, 3, float(h), refine, boundary_layer, int(max_elements), os.path.basename(shape))
                _sizes(gmsh, float(h), refine)
                _bl_3d_volume(gmsh, verts, tris, boundary_layer)
                _sizes(gmsh, float(h), refine)
                _generate(gmsh, 3, boundary_layer)
                backend = "gmsh-discrete+boundary-layer"
            else:
                _load_elements(gmsh, vol.vertices, {"tetra": vol.tets}, 3)
                backend = f"mesh_from_surface ({vol.meta.get('backend', '?')})"
        else:
            dim = _primitive(gmsh, shape, d, o) if shape in PRIMITIVES else _import_cad(gmsh, shape, float(scale))
            if shape not in PRIMITIVES:                       # a file: its OCC bounding box, before meshing
                x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
                _check_box(np.array([[x0, y0, z0], [x1, y1, z1]]), dim, float(h), refine, boundary_layer,
                           int(max_elements), os.path.basename(shape))
            _sizes(gmsh, float(h), refine)
            if boundary_layer and dim == 3:
                verts, tris = _surface_of_model(gmsh)
                _bl_3d_volume(gmsh, verts, tris, boundary_layer)
                _sizes(gmsh, float(h), refine)
                backend = "gmsh-occ+boundary-layer"
            elif boundary_layer:
                _bl_2d(gmsh, boundary_layer)
                backend = "gmsh-occ+boundary-layer"
            _generate(gmsh, dim, boundary_layer)
        res = _collect(gmsh, dim)
        files: dict[str, str] = {}
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            files["msh"] = str(Path(out_dir, "mesh.msh").resolve())
            gmsh.write(files["msh"])

    from pinneapple_data.cae.model import CellBlock, Mesh
    from pinneapple_data.cae.quality import element_metrics
    points, blocks = res["points"], res["blocks"]
    mesh = Mesh(points=points, blocks=[CellBlock(t, c) for t, c in blocks])
    aspect = element_metrics(mesh)["edge_ratio"]
    mesh.cell_data.update(sicn=res["sicn"], gamma=res["gamma"], aspect_ratio=aspect)
    edges = _edges(points, blocks)
    counts = {t: int(len(c)) for t, c in blocks}
    s_sicn, s_gamma, s_ar = _stats(res["sicn"]), _stats(res["gamma"]), _stats(aspect)
    out = {"dim": int(dim), "n_nodes": int(len(points)), "n_elements": int(sum(counts.values())),
           "element_counts": counts, "sicn_min": s_sicn[0], "sicn_mean": s_sicn[1], "gamma_min": s_gamma[0],
           "gamma_mean": s_gamma[1], "aspect_ratio_max": s_ar[2], "aspect_ratio_mean": s_ar[1],
           "volume": res["measure"] if dim == 3 else None,
           "area": _boundary_area(points, blocks) if dim == 3 else res["measure"],
           "h_min": float(edges.min()), "h_mean": float(edges.mean()), "h_max": float(edges.max()),
           "files": files, "backend": backend}
    if out_dir:
        _write_outputs(mesh, out_dir, files, out)
    return out


def _write_outputs(mesh, out_dir: str, files: dict[str, str], summary: dict[str, Any]) -> None:
    import json

    from pinneapple_data.cae.dataset import build_dataset, export
    from pinneapple_tools.visualization.studio.fieldview import field_view
    data, _, _ = export(build_dataset(mesh), "vtu")
    files["vtu"] = str(Path(out_dir, "mesh.vtu").resolve())
    Path(files["vtu"]).write_bytes(data)
    fv = field_view(mesh, [
        {"name": "sicn", "values": mesh.cell_data["sicn"], "label": "SICN (1 ideal, <= 0 inverted)", "unit": "-",
         "where": "cell"},
        {"name": "gamma", "values": mesh.cell_data["gamma"], "label": "gamma (1 ideal)", "unit": "-", "where": "cell"},
        {"name": "aspect_ratio", "values": mesh.cell_data["aspect_ratio"], "label": "edge ratio (1 ideal)",
         "unit": "-", "where": "cell"}],
        title=f"mesh quality: {summary['n_elements']} elements", length_unit="m",
        notes=[f"{k} {v}" for k, v in summary["element_counts"].items()])
    files["glb"] = str(Path(out_dir, "quality.glb").resolve())
    Path(files["glb"]).write_bytes(fv.glb())
    files["view"] = str(Path(out_dir, "quality.json").resolve())
    Path(files["view"]).write_text(json.dumps(fv.meta, indent=1))
