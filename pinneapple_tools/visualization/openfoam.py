"""An OpenFOAM case, a ``.foam`` pointer or a VTK file, straight into a PyVista grid with the fields of each time (#60).

    import pinneapple as pp

    grid = pp.viz.read_openfoam("case/")                        # every field of the latest time
    pp.viz.render_png(grid, "case/p.png", scalars="p")          # a real case becomes an image in three lines

    pp.viz.read_openfoam("case/", time=0.5, fields=["p", "U"])  # one time, two fields
    pp.viz.read_openfoam("case/", all_times=True)               # one block per time directory, named t=<time>
    pp.viz.read_openfoam_patches("case/")                       # the boundary, one surface per patch
    pp.viz.read_openfoam_patches("case/", include_empty=True)   # also the empty (2-D thickness) patches
    pp.viz.read_grid("case.foam")                               # a .foam file: its folder, no VTK reader needed
    pp.viz.read_grid("results.vtu")                             # VTK files go through PyVista itself
    pp.viz.read_openfoam("examples/physics_check/pass/cases/cavity")   # a case committed in this repository

The case is read by ``pinneapple_data.cae.foam`` (ASCII and binary ``polyMesh``, gzipped or not), so the grid is the
mesh as OpenFOAM wrote it: every cell keeps its own faces (VTK polyhedra, not the hexes of a converted mesh), a volume
field is a cell array and a boundary field is a cell array on the patch surface. That is what ParaView shows, without
requiring VTK's own OpenFOAM reader -- which is missing from some builds and refuses cases that are still readable.

Install: ``pip install "pinneapple[pyvista]"``. Rendering needs an OpenGL context: see
:func:`~pinneapple_tools.visualization.pyvista_bridge.can_render` and ``docs/guides/headless_rendering.md``.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

_INSTALL_HINT = 'PyVista is not installed: pip install "pinneapple[pyvista]"'
VTK_EXT = (".vtk", ".vtu", ".vtp", ".vtm", ".vtkhdf", ".pvtu")
POLY_FILES = ("points", "faces", "owner", "neighbour", "boundary")


def _pv():
    try:
        import pyvista
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc
    return pyvista


# --------------------------------------------------------------------------------------- finding the case
def _has(d: str, name: str) -> bool:
    """Whether ``name`` or ``name.gz`` is in ``d``: a case written with ``writeCompression on`` is all ``.gz``."""
    return os.path.isfile(os.path.join(d, name)) or os.path.isfile(os.path.join(d, name + ".gz"))


def _is_polymesh(d: str) -> bool:
    return _has(d, "points") and _has(d, "owner")


def case_folder(path: str) -> str:
    """The case folder behind ``path``: the folder itself, the folder of a ``.foam`` file, or a bare ``polyMesh``."""
    p = os.path.abspath(str(path))
    if os.path.isdir(p):
        if os.path.isdir(os.path.join(p, "constant", "polyMesh")) or _is_polymesh(p):
            return p
        raise FileNotFoundError(f"{p} is not an OpenFOAM case (no constant/polyMesh and no polyMesh/points)")
    if p.lower().endswith(".foam"):                     # the empty file ParaView opens: it points at its folder
        for cand in (os.path.dirname(p), p[: -len(".foam")]):
            if os.path.isdir(cand):
                return case_folder(cand)
        raise FileNotFoundError(f"{p}: no case folder next to the .foam file")
    raise FileNotFoundError(f"{p}: not a case folder and not a .foam file")


def _polymesh_dir(root: str) -> tuple[str, str]:
    """(prefix the polyMesh files are stored under, the polyMesh folder)."""
    for prefix, d in (("constant/polyMesh/", os.path.join(root, "constant", "polyMesh")),
                      ("polyMesh/", os.path.join(root, "polyMesh")),
                      ("", root)):
        if os.path.isdir(d) and _is_polymesh(d):
            return prefix, d
    raise FileNotFoundError(f"{root}: no polyMesh (points, owner) in constant/polyMesh, polyMesh or the folder itself")


def _numeric_dirs(root: str) -> list[tuple[float, str]]:
    """``[(time, folder name)]`` for every numeric time directory, ascending."""
    out = []
    for name in os.listdir(root):
        try:
            t = float(name)
        except ValueError:
            continue
        if os.path.isdir(os.path.join(root, name)):
            out.append((t, name))
    return sorted(out)


def openfoam_times(path: str) -> list[float]:
    """The time directories of the case, ascending (``[]`` for a bare polyMesh)."""
    return [t for t, _ in _numeric_dirs(case_folder(path))]


def _time_name(root: str, time) -> str | None:
    """The time directory name for ``time`` (a number, a name or None for the latest); None if the case has none."""
    dirs = _numeric_dirs(root)
    if not dirs:
        return None
    if time is None:
        return dirs[-1][1]
    if isinstance(time, str) and not _looks_numeric(time):
        for _, name in dirs:
            if name == time:
                return name
        raise KeyError(f"{root}: no time directory {time!r} (have {[n for _, n in dirs]})")
    have = [n for _, n in dirs]
    want = float(time)
    best = min(dirs, key=lambda d: abs(d[0] - want))[1]
    if abs(float(best) - want) > 1e-9 * max(1.0, abs(want)):
        raise KeyError(f"{root}: no time directory {want!r} (have {have})")
    return best


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def _case_files(root: str, time: str | None = None) -> dict[str, bytes]:
    """The bytes ``pinneapple_data.cae.foam`` reads, keyed by path relative to the case root (one time directory)."""
    prefix, _ = _polymesh_dir(root)
    fs: dict[str, bytes] = {}
    for name in POLY_FILES:
        for cand in (name, name + ".gz"):
            p = os.path.join(root, prefix, cand)
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    fs[prefix + cand] = fh.read()
                break
    for _, name in _numeric_dirs(root):
        if time is not None and name != time:
            continue
        d = os.path.join(root, name)
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    fs[f"{name}/{f}"] = fh.read()
    return fs


def _read_mesh(path: str, time=None, fields: Sequence[str] | None = None):
    """(the ``pinneapple_data.cae.Mesh``, ``{"time", "fields", "skipped"}``) for one time directory."""
    from pinneapple_data.cae import foam

    root = case_folder(path)
    name = _time_name(root, time)
    fs = _case_files(root, name)
    mesh = foam.read_polymesh(fs, "")
    info = {"time": None, "fields": [], "skipped": []}
    if name is not None:
        info = foam.read_time(fs, mesh, "", name, only=list(fields) if fields else None)
    mesh.source["case"] = root
    return mesh, info


# ------------------------------------------------------------------------------- pure NumPy: the two conversions
def polyhedron_cells(poly) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell VTK polyhedron face streams ``[nFaces, nPts, ids..., nPts, ids..., ...]`` and their offsets.

    Every face is written once for the cell that owns it and, reversed, for the neighbour cell: a face's node order in
    ``polyMesh`` is outward from the owner and therefore inward from the neighbour, and VTK wants it outward on both.
    """
    cells: list[list[np.ndarray]] = [[] for _ in range(poly.n_cells)]
    for f in range(poly.n_faces):
        nodes = poly.face_nodes[poly.face_offsets[f]: poly.face_offsets[f + 1]]
        cells[int(poly.owner[f])].append(nodes)
        if f < poly.n_internal:
            cells[int(poly.neighbour[f])].append(nodes[::-1])
    stream: list[int] = []
    offsets = [0]
    for faces in cells:
        stream.append(len(faces))
        for face in faces:
            stream.append(len(face))
            stream.extend(int(v) for v in face)
        offsets.append(len(stream))
    return np.asarray(stream, np.int64), np.asarray(offsets, np.int64)


def patch_surface(points: np.ndarray, face_nodes: np.ndarray, face_offsets: np.ndarray, start: int, n: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One boundary patch as (vertices duplicated per face, fan triangles, the local face of each triangle).

    Duplicating the vertices keeps a per-face value exact: nothing is averaged across a face edge.
    """
    counts = np.diff(face_offsets[start: start + n + 1])
    base = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    idx = np.concatenate([face_nodes[face_offsets[f]: face_offsets[f + 1]] for f in range(start, start + n)])
    verts = np.asarray(points, float)[idx]
    tris = [np.stack([base[i] + np.zeros(c - 2, np.int64), base[i] + np.arange(1, c - 1),
                      base[i] + np.arange(2, c)], axis=1) for i, c in enumerate(counts) if c > 2]
    owner = np.repeat(np.arange(n, dtype=np.int64), np.maximum(counts - 2, 0))
    return verts, np.concatenate(tris) if tris else np.zeros((0, 3), np.int64), owner


# ------------------------------------------------------------------------------------------- PyVista assembly
def _grid_from_polyhedra(mesh):
    """``pinneapple_data.cae.Mesh`` (polyhedral) to a ``pyvista.UnstructuredGrid`` of VTK polyhedra."""
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
    from vtkmodules.vtkCommonDataModel import VTK_POLYHEDRON, vtkUnstructuredGrid

    pv = _pv()
    stream, offsets = polyhedron_cells(mesh.poly)
    ug = vtkUnstructuredGrid()
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(mesh.points, dtype=float)))
    ug.SetPoints(vtk_points)
    ids = vtkIdList()
    for c in range(len(offsets) - 1):
        a, b = int(offsets[c]), int(offsets[c + 1])
        ids.SetNumberOfIds(b - a)
        for k in range(a, b):
            ids.SetId(k - a, int(stream[k]))
        ug.InsertNextCell(VTK_POLYHEDRON, ids)
    grid = pv.wrap(ug)
    for name, values in mesh.cell_data.items():
        grid.cell_data[name] = np.asarray(values)
    for name, values in mesh.point_data.items():
        grid.point_data[name] = np.asarray(values)
    return grid


def _surface_from_patch(mesh, patch, values: dict[str, np.ndarray] | None = None):
    """One patch as a ``pyvista.PolyData``: triangles per face, the value of each face as cell data."""
    pv = _pv()
    poly = mesh.poly
    start, n = int(patch["startFace"]), int(patch["nFaces"])
    verts, tris, owner = patch_surface(mesh.points, poly.face_nodes, poly.face_offsets, start, n)
    flat = np.hstack([np.full((len(tris), 1), 3, np.int64), tris]).ravel()
    surf = pv.PolyData(verts, flat) if len(tris) else pv.PolyData(verts)
    first = start - poly.n_internal                       # boundary faces are numbered after the internal ones
    for name, arr in (values or {}).items():
        arr = np.asarray(arr)
        per_face = arr[first: first + n]
        surf.cell_data[name] = per_face[owner] if len(per_face) == n else np.full(n, np.nan)[owner]
    surf.field_data["patch"] = np.array([patch["name"]])
    surf.field_data["patch_type"] = np.array([patch["type"]])
    return surf


# ------------------------------------------------------------------------------------------------------ readers
def read_openfoam(path: str, *, time=None, fields: Sequence[str] | None = None, all_times: bool = False):
    """The volume mesh of an OpenFOAM case as a ``pyvista.UnstructuredGrid``, one cell per OpenFOAM cell.

    ``time`` is a time directory (number or name); the latest one by default. ``fields`` keeps only those volume
    fields. ``all_times=True`` returns a ``pyvista.MultiBlock`` with one block per time directory, named ``t=<time>``.
    """
    if not all_times:
        return _volume(path, time=time, fields=fields)
    blocks = _pv().MultiBlock()
    for t in openfoam_times(path):
        blocks[f"t={t:g}"] = _volume(path, time=t, fields=fields)
    return blocks


def _volume(path: str, time=None, fields: Sequence[str] | None = None):
    mesh, info = _read_mesh(path, time=time, fields=fields)
    grid = _grid_from_polyhedra(mesh)
    grid.field_data["source"] = np.array([f"OpenFOAM case {mesh.source['case']}"])
    if info.get("time") is not None:
        grid.field_data["time"] = np.array([float(info["time"])])
    return grid


def read_openfoam_patches(path: str, *, names: Sequence[str] | None = None, time=None,
                          fields: Sequence[str] | None = None, all_times: bool = False,
                          include_empty: bool = False):
    """The boundary of an OpenFOAM case as a ``pyvista.MultiBlock``: one surface per patch, one value per face.

    ``names`` keeps only those patches; every patch with faces by default. ``empty`` patches -- the two caps that make
    a 2-D case two-dimensional -- are left out unless ``include_empty=True``: they are not a surface of the geometry,
    and drawn they hide the field behind them (ParaView hides them too). Values come from the boundaryField of the time
    directory, so a wall value is the wall value and not the neighbouring cell's. With ``all_times=True`` the result is
    one block per time directory, each holding the patches.
    """
    if not all_times:
        return _patches(path, names=names, time=time, fields=fields, include_empty=include_empty)
    outer = _pv().MultiBlock()
    for t in openfoam_times(path):
        inner = _patches(path, names=names, time=t, fields=fields, include_empty=include_empty)
        if inner.n_blocks:
            outer[f"t={t:g}"] = inner
    return outer


def _patches(path: str, names=None, time=None, fields=None, include_empty: bool = False):
    mesh, _ = _read_mesh(path, time=time, fields=fields)
    poly = mesh.poly
    have = [p["name"] for p in poly.patches]
    keep = list(names) if names else have
    missing = [n for n in keep if n not in have]
    if missing:
        raise KeyError(f"{path}: patches {missing} not in {have}")
    out = _pv().MultiBlock()
    for patch in poly.patches:
        if patch["name"] not in keep or int(patch["nFaces"]) == 0:
            continue
        if patch["type"] == "empty" and not include_empty:
            continue
        out[patch["name"]] = _surface_from_patch(mesh, patch, mesh.boundary_values)
    return out


def read_vtk(path: str):
    """A VTK file (``.vtk``, ``.vtu``, ``.vtp``, ``.vtm``, ``.vtkhdf``, ``.pvtu``) read by PyVista itself."""
    return _pv().read(str(path))


def read_grid(path: str, **kwargs):
    """``read_openfoam`` for a case or a ``.foam`` file, ``read_vtk`` for a VTK file: one name for both.

    Extra keyword arguments (``time``, ``fields``, ``patches=True``, ``all_times=True``) go to the OpenFOAM reader;
    ``patches=True`` returns the boundary instead of the volume.
    """
    p = str(path)
    if p.lower().endswith(VTK_EXT):
        if kwargs:
            raise TypeError(f"{p}: a VTK file takes no OpenFOAM options ({sorted(kwargs)})")
        return read_vtk(p)
    patches = kwargs.pop("patches", False)
    if patches:
        return read_openfoam_patches(p, **kwargs)
    return read_openfoam(p, **kwargs)
