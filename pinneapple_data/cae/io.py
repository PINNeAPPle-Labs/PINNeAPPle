"""Read whatever was uploaded into a ``Mesh``: OpenFOAM case or polyMesh, Abaqus/CalculiX .inp, CalculiX .frd, STL,
Gmsh .msh, VTK (.vtk/.vtu) and the other meshio formats, and point clouds with fields (CSV, HDF5, NPZ).

Uploads are a dict of path -> bytes (zip archives already expanded, see ``pinneapple_data.simulation_metadata
.read_upload``). ``read_any`` picks the reader from the file names and the content.
"""
from __future__ import annotations

import io as _io
import os
import struct
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import foam, frd, inp
from .model import CellBlock, ELEMENTS, Mesh

MESHIO_EXT = (".msh", ".vtk", ".vtu", ".cgns", ".med", ".xdmf", ".xmf", ".obj", ".ply", ".off", ".mesh", ".bdf",
              ".nas", ".dat.nas", ".f3grid", ".su2", ".ugrid", ".h5m", ".tec", ".vol", ".wkt", ".exo", ".e")
POINT_EXT = (".csv", ".txt", ".npz", ".h5", ".hdf5", ".parquet")


class FormatError(ValueError):
    pass


def kind_of(fs: Dict[str, bytes]) -> Tuple[str, Optional[str]]:
    """(kind, main file or case root). Kinds: openfoam, inp, frd, stl, meshio, points."""
    names = list(fs)
    roots = [n[: -len("system/controlDict")] for n in names if n.endswith("system/controlDict")]
    poly = [n[: -len("constant/polyMesh/owner")] for n in names if n.removesuffix(".gz").endswith("constant/polyMesh/owner")]
    if poly:
        return "openfoam", min(poly, key=len)
    bare = [n[: -len("polyMesh/owner")] for n in names if n.removesuffix(".gz").endswith("polyMesh/owner")]
    if bare:
        return "openfoam_polymesh", min(bare, key=len) + "polyMesh/"
    if roots:
        return "openfoam", min(roots, key=len)
    low = {n: n.lower() for n in names}
    for ext, k in ((".frd", "frd"), (".inp", "inp"), (".stl", "stl")):
        hit = [n for n in names if low[n].endswith(ext)]
        if hit:
            return k, max(hit, key=lambda n: len(fs[n]))
    hit = [n for n in names if low[n].endswith(MESHIO_EXT)]
    if hit:
        return "meshio", max(hit, key=lambda n: len(fs[n]))
    hit = [n for n in names if low[n].endswith(POINT_EXT)]
    if hit:
        return "points", max(hit, key=lambda n: len(fs[n]))
    raise FormatError("no mesh found: upload an OpenFOAM case (zip) or polyMesh folder, a .msh, .vtk/.vtu, .stl, "
                      ".inp or .frd file, or a CSV/HDF5 point table")


def read_any(fs: Dict[str, bytes], fields: bool = True, time: Optional[str] = None) -> Mesh:
    k, main = kind_of(fs)
    if k == "openfoam":
        m = foam.read_polymesh(fs, main)
        m.source["root"] = main
        cd = fs.get(main + "system/controlDict")
        if cd:
            import re as _re
            mm = _re.search(rb"^\s*application\s+(\w+)\s*;", cd, _re.M)
            if mm:
                m.source["application"] = mm.group(1).decode()
        if fields:
            info = foam.read_time(fs, m, main, time)
            m.source["fields_skipped"] = info["skipped"]
    elif k == "openfoam_polymesh":
        m = foam.read_polymesh(fs, main[: -len("polyMesh/")])
    elif k == "inp":
        texts = {os.path.basename(n): b.decode("latin-1") for n, b in fs.items() if n.lower().endswith((".inp", ".msh", ".nam", ".txt"))}
        deck = inp.read_deck(fs[main].decode("latin-1"), os.path.basename(main), texts)
        m = inp.deck_mesh(deck)
        m.source["deck"] = deck
    elif k == "frd":
        m = frd.read_frd(fs[main])
    elif k == "stl":
        m = read_stl(fs[main])
    elif k == "meshio":
        m = read_meshio(main, fs[main])
    else:
        m = read_points(main, fs[main])
    m.source.setdefault("file", main)
    m.source["kind"] = k
    return m


# ------------------------------------------------------------------------------------------------ STL
def read_stl(b: bytes) -> Mesh:
    tris: np.ndarray
    if len(b) >= 84:
        n = struct.unpack("<I", b[80:84])[0]
        if 84 + 50 * n == len(b):
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            tris = np.frombuffer(b, dt, count=n, offset=84)["v"].astype(float)
            return _from_soup(tris, "STL (binary)")
    txt = b.decode("latin-1")
    if "vertex" not in txt:
        raise FormatError("not an STL file")
    v = np.array([ln.split()[1:4] for ln in txt.splitlines() if ln.strip().startswith("vertex")], float)
    return _from_soup(v.reshape(-1, 3, 3), "STL (ASCII)")


def _from_soup(tris: np.ndarray, fmt: str) -> Mesh:
    """Merge coincident vertices (STL stores each triangle's corners separately)."""
    pts = tris.reshape(-1, 3)
    scale = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    key = np.round(pts / (scale * 1e-9)).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    p = np.zeros((len(uniq), 3))
    p[inv] = pts
    return Mesh(points=p, blocks=[CellBlock("triangle", inv.reshape(-1, 3))], source={"format": fmt})


# ------------------------------------------------------------------------------------------------ meshio
def read_meshio(name: str, b: bytes) -> Mesh:
    try:
        import meshio
    except ImportError as e:                                           # pragma: no cover
        raise FormatError("reading this format needs meshio (pip install meshio)") from e
    ext = os.path.splitext(name)[1].lower()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "upload" + ext)
        with open(p, "wb") as f:
            f.write(b)
        try:
            mm = meshio.read(p)
        except Exception as e:
            raise FormatError(f"could not read {os.path.basename(name)}: {e}") from e
    blocks = [(c.type, np.asarray(c.data)) for c in mm.cells]
    known = [(t, c) for t, c in blocks if t in ELEMENTS]
    if not known:
        raise FormatError("the file has no supported elements")
    top = max(ELEMENTS[t][1] for t, _ in known)
    names = {}
    for nm, (tag, dim) in (mm.field_data or {}).items():
        names[(int(dim), int(tag))] = nm
    phys = mm.cell_data.get("gmsh:physical")
    m = Mesh(points=np.asarray(mm.points, float)[:, :3] if mm.points.shape[1] >= 3 else
             np.column_stack([mm.points, np.zeros(len(mm.points))]), source={"format": f"{ext[1:]} (meshio)"})
    cell_index = 0
    gidx = 0
    for bi, (t, conn) in enumerate(blocks):
        if t not in ELEMENTS:
            continue
        dim = ELEMENTS[t][1]
        if dim == top:
            m.blocks.append(CellBlock(t, conn.astype(np.int64)))
            if phys is not None:
                tags = np.asarray(phys[bi])
                for tg in np.unique(tags):
                    nm = names.get((dim, int(tg)), f"physical {int(tg)}")
                    m.cell_sets.setdefault(nm, np.zeros(0, np.int64))
                    m.cell_sets[nm] = np.concatenate([m.cell_sets[nm], gidx + np.nonzero(tags == tg)[0]])
            for k, arrs in mm.cell_data.items():
                if k.startswith("gmsh:") or k.startswith("medit:"):
                    continue
                a = np.asarray(arrs[bi], float)
                m.cell_data.setdefault(k, []).append(a)
            gidx += len(conn)
        elif dim == top - 1 and dim >= 1:
            if phys is not None:
                tags = np.asarray(phys[bi])
                for tg in np.unique(tags):
                    nm = names.get((dim, int(tg)), f"physical {int(tg)}")
                    m.boundary_blocks.append((nm, CellBlock(t, conn[tags == tg].astype(np.int64))))
            else:
                m.boundary_blocks.append((t, CellBlock(t, conn.astype(np.int64))))
        cell_index += len(conn)
    m.cell_data = {k: np.concatenate(v) for k, v in m.cell_data.items() if sum(len(x) for x in v) == m.n_cells}
    for k, v in (mm.point_data or {}).items():
        if not k.startswith("gmsh:"):
            m.point_data[k] = np.asarray(v, float)
    return m


# ------------------------------------------------------------------------------------------------ point tables
_COORD_NAMES = [("x", "y", "z"), ("X", "Y", "Z"), ("Points:0", "Points:1", "Points:2"), ("coordsX", "coordsY", "coordsZ"),
                ("x [m]", "y [m]", "z [m]"), ("x_m", "y_m", "z_m"), ("Points_0", "Points_1", "Points_2")]


def read_points(name: str, b: bytes) -> Mesh:
    """A table of points with field columns (CSV/TXT, Parquet, NPZ, HDF5): a point cloud mesh with point_data."""
    low = name.lower()
    if low.endswith((".csv", ".txt")):
        import pandas as pd
        df = pd.read_csv(_io.BytesIO(b), sep=None, engine="python", comment="#")
        cols = {c: c for c in df.columns}
    elif low.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(_io.BytesIO(b))
        cols = {c: c for c in df.columns}
    elif low.endswith(".npz"):
        z = np.load(_io.BytesIO(b), allow_pickle=False)
        return _points_from_arrays({k: z[k] for k in z.files}, name)
    else:
        return _points_from_hdf5(b, name)
    units: Dict[str, str] = {}
    clean: Dict[str, str] = {}
    import re
    for c in cols:
        mm = re.match(r"^\s*(.+?)\s*[\[(]\s*([^\])]+)\s*[\])]\s*$", str(c))
        base = mm.group(1) if mm else str(c)
        if mm:
            units[base] = mm.group(2)
        clean[c] = base
    df = df.rename(columns=clean)
    xyz = None
    for trip in _COORD_NAMES:
        have = [t for t in trip if t in df.columns]
        if len(have) >= 2:
            xyz = have
            break
    if xyz is None:
        raise FormatError("the table needs coordinate columns x, y (and z), or Points:0..2 as exported by ParaView")
    P = np.zeros((len(df), 3))
    for i, c in enumerate(xyz):
        P[:, i] = df[c].to_numpy(float)
    m = Mesh(points=P, source={"format": "point table", "file": name})
    vec = _group_vectors([c for c in df.columns if c not in xyz])
    for fname, comps in vec.items():
        arr = np.column_stack([df[c].to_numpy(float) for c in comps]) if len(comps) > 1 else df[comps[0]].to_numpy(float)
        if not np.issubdtype(np.asarray(arr).dtype, np.number):
            continue
        m.point_data[fname] = arr
        u = units.get(comps[0])
        if u:
            m.units[fname] = u
    return m


def _group_vectors(cols: List[str]) -> Dict[str, List[str]]:
    """'U:0','U:1','U:2' / 'U_x','U_y','U_z' / 'Ux','Uy','Uz' -> one vector field U."""
    import re
    out: Dict[str, List[str]] = {}
    used = set()
    pats = [r"^(.+?)[:_](0|1|2)$", r"^(.+?)[_]?(x|y|z)$", r"^(.+?)[_]?(X|Y|Z)$"]
    for pat in pats:
        groups: Dict[str, Dict[str, str]] = {}
        for c in cols:
            mm = re.match(pat, c)
            if mm and c not in used:
                groups.setdefault(mm.group(1), {})[mm.group(2).lower()] = c
        for base, d in groups.items():
            keys = [k for k in ("0", "1", "2", "x", "y", "z") if k in d]
            if len(keys) >= 2 and base:
                out[base] = [d[k] for k in keys]
                used.update(d.values())
    for c in cols:
        if c not in used:
            out[c] = [c]
    return out


def _points_from_arrays(arrs: Dict[str, np.ndarray], name: str) -> Mesh:
    pts = None
    for k in ("points", "coordinates", "coords", "xyz", "x"):
        if k in arrs and np.asarray(arrs[k]).ndim == 2:
            pts = np.asarray(arrs[k], float)
            break
    if pts is None and all(k in arrs for k in ("x", "y")):
        pts = np.column_stack([arrs["x"], arrs["y"], arrs.get("z", np.zeros(len(arrs["x"])))])
    if pts is None:
        raise FormatError("no 'points' / 'coordinates' array (n x 3), or x, y, z arrays")
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    m = Mesh(points=pts, source={"format": "point arrays", "file": name})
    for k, v in arrs.items():
        v = np.asarray(v)
        if k in ("points", "coordinates", "coords", "xyz", "x", "y", "z") or not np.issubdtype(v.dtype, np.number):
            continue
        if len(v) == len(pts):
            m.point_data[k] = v.astype(float)
    return m


def _points_from_hdf5(b: bytes, name: str) -> Mesh:
    import h5py
    arrs: Dict[str, np.ndarray] = {}
    units: Dict[str, str] = {}
    with h5py.File(_io.BytesIO(b), "r") as h:
        def visit(path, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim in (1, 2) and obj.size < 5e8:
                key = path.split("/")[-1]
                arrs[key] = obj[()]
                u = obj.attrs.get("unit", obj.attrs.get("units"))
                if u is not None:
                    units[key] = u.decode() if isinstance(u, bytes) else str(u)
        h.visititems(visit)
    m = _points_from_arrays(arrs, name)
    m.units.update({k: v for k, v in units.items() if k in m.point_data})
    return m
