"""Core coverage for pinneapple_data.cae (#1129, toward 1.0): the OpenFOAM, Abaqus/CalculiX, STL and point readers, the
face-based geometry, the CFD metrics, the dataset writers and the comparison helpers.

The meshes are the mesh-quality app's committed examples (OpenFOAM pitzDaily, a CalculiX cantilever, a damaged STL,
a Gmsh bracket); reference numbers come from OpenFOAM checkMesh (pitzDaily volume 1.4516e-05 m^3, as in
tests/test_cae_mesh_quality.py) or from closed forms.
"""
import io as _io
import json
import os

import numpy as np
import pytest

from pinneapple_data.cae import compare as CP
from pinneapple_data.cae import dataset as DS
from pinneapple_data.cae import foam as F
from pinneapple_data.cae import geometry as G
from pinneapple_data.cae import inp as INP
from pinneapple_data.cae import io as IO
from pinneapple_data.cae import metrics as MT
from pinneapple_data.cae.upload import expand

EX = os.path.join(os.path.dirname(__file__), "..", "..", "apps", "mesh_quality", "examples")


def _files(name):
    with open(os.path.join(EX, name), "rb") as f:
        return expand([(name, f.read())])


@pytest.fixture(scope="module")
def pitz_fs():
    return _files("openfoam_pitzDaily.zip")


@pytest.fixture(scope="module")
def pitz(pitz_fs):
    return IO.read_any(pitz_fs, fields=False)


# ------------------------------------------------------------------------------------------------ OpenFOAM
def test_polymesh_readers_agree_with_each_other(pitz_fs):
    b = pitz_fs
    assert F.header(b["constant/polyMesh/points"].decode("latin-1"))["class"] == "vectorField"
    pts = F.read_points(b["constant/polyMesh/points"])
    flat, off = F.read_faces(b["constant/polyMesh/faces"])
    owner = F.read_labels(b["constant/polyMesh/owner"])
    neigh = F.read_labels(b["constant/polyMesh/neighbour"])
    patches = F.read_boundary(b["constant/polyMesh/boundary"])
    n_faces = len(off) - 1
    assert pts.shape[1] == 3 and flat.max() == len(pts) - 1 and len(owner) == n_faces
    assert sum(p["nFaces"] for p in patches) == n_faces - len(neigh)           # every boundary face in a patch
    assert owner.max() + 1 == 12225                                              # checkMesh: 12225 cells


def test_time_directories_and_a_field(pitz_fs, pitz):
    fs = dict(pitz_fs)
    n = pitz.n_cells
    fs["10/p"] = b'FoamFile { class volScalarField; object p; }\ndimensions [0 2 -2 0 0 0 0];\n' \
        b'internalField uniform 3.5;\nboundaryField { ".*" { type zeroGradient; } }\n'
    fs["0.5/p"] = fs["10/p"]
    fs["2/p"] = fs["10/p"]
    assert F.time_dirs(fs) == ["0.5", "2", "10"]                                 # numeric, ascending
    values, unit, cls = F.read_field(fs["10/p"], n)
    assert cls == "volScalarField" and np.all(values == 3.5) and len(values) == n
    assert unit == F.dims_unit("[0 2 -2 0 0 0 0]") == "m2/s2"                  # kinematic pressure
    got = F.read_time(fs, pitz, time="10")                                       # loads into mesh.cell_data
    assert float(got["time"]) == 10.0 and "p" in got["fields"] and np.all(pitz.cell_data["p"] == 3.5)


# ------------------------------------------------------------------------------------------------ geometry
def test_face_geometry_volume_matches_checkmesh(pitz):
    fm = G.face_mesh(pitz)
    fg = G.face_geometry(fm)
    cg = G.cell_geometry(fm, fg)
    assert isinstance(fm, G.FaceMesh) and fm.n_cells == 12225
    assert cg["volumes"].sum() == pytest.approx(1.4516e-05, rel=1e-4)           # checkMesh total volume
    assert np.all(cg["volumes"] > 0) and len(fg["areas"]) == len(fm.owner)
    adj = G.cell_adjacency(fm)
    assert adj.shape == (12225, 12225) and (adj != adj.T).nnz == 0              # symmetric
    assert len(np.unique(G.regions(adj))) == 1                                    # one connected region
    sd = G.solution_dirs(fm, fg)
    assert sd is not None and list(sd) == [True, True, False]                    # 2-D case: z is 'empty'


def test_volume_elements_are_positively_oriented():
    deck = INP.read_deck(open(os.path.join(EX, "calculix_cantilever_hex.inp")).read())
    m = INP.deck_mesh(deck)
    assert G.oriented_blocks(m) == [] or all(isinstance(s, str) for s in G.oriented_blocks(m))
    for blk in m.blocks:
        assert np.all(G.signed_volume_sign(blk.type, m.points, blk.conn) > 0)
    nodes_adj = G.element_adjacency_by_nodes(m)
    assert nodes_adj.shape == (640, 640) and (nodes_adj != nodes_adj.T).nnz == 0


# ------------------------------------------------------------------------------------------------ Abaqus / CalculiX
def test_inp_deck_sets_and_mesh():
    text = open(os.path.join(EX, "calculix_cantilever_hex.inp")).read()
    deck = INP.read_deck(text)
    assert isinstance(deck, INP.Deck) and all(isinstance(b, INP.Block) for b in deck.blocks)
    nsets = INP.resolve_sets(deck, "NSET")
    assert {"FIXED", "TIP"} <= set(nsets) and len(nsets["FIXED"]) > 0
    m = INP.deck_mesh(deck)
    assert (m.n_cells, m.n_points) == (640, 1025)                                # as tests/test_cae_mesh_quality.py
    assert INP.vtk_type("C3D8") == "hexahedron" and INP.vtk_type("C3D4") == "tetra"
    assert INP.vtk_type("NOT_AN_ELEMENT") is None


# ------------------------------------------------------------------------------------------------ readers
def test_kind_and_readers_for_each_format():
    assert IO.kind_of(_files("calculix_cantilever_hex.inp"))[0] == "inp"
    stl = _files("bracket_surface_damaged.stl")
    assert IO.kind_of(stl)[0] == "stl"
    s = IO.read_stl(next(iter(stl.values())))
    assert s.n_cells > 0 and s.element_counts().get("triangle", 0) == s.n_cells
    csv = b"x,y,z,T\n0,0,0,300\n1,0,0,310\n0,1,0,320\n"
    p = IO.read_points("pts.csv", csv)
    assert p.n_points == 3 and np.allclose(np.asarray(p.point_data["T"]).ravel(), [300, 310, 320])
    with pytest.raises(IO.FormatError):
        IO.read_any({"notes.txt": b"nothing a mesh reader knows"})


def test_meshio_formats():
    pytest.importorskip("meshio")
    msh = _files("gmsh_bracket_tet.msh")
    t = IO.read_meshio("gmsh_bracket_tet.msh", next(iter(msh.values())))
    assert t.element_counts().get("tetra", 0) > 0


# ------------------------------------------------------------------------------------------------ CFD metrics
def test_wall_distance_is_the_distance_to_a_flat_wall():
    y = np.linspace(0.05, 0.95, 10)
    centres = np.c_[np.full(10, 0.5), y, np.zeros(10)]
    wall = np.c_[np.linspace(0, 1, 101), np.zeros(101), np.zeros(101)]
    d = MT.wall_distance(centres, wall, np.tile([0.0, 1.0, 0.0], (101, 1)))
    assert np.allclose(d, y, atol=1e-12)                                         # exact: the wall is the plane y = 0


def test_courant_number_follows_its_definition(pitz):
    fm = G.face_mesh(pitz)
    fg = G.face_geometry(fm)
    cg = G.cell_geometry(fm, fg)
    U = np.tile([10.0, 0.0, 0.0], (fm.n_cells, 1))
    dt = 1e-5
    co = MT.mesh_courant_number(pitz, U, dt)
    flux = np.abs(fg["areas"] @ np.array([10.0, 0.0, 0.0]))                     # areas: area vectors S_f
    ref = np.zeros(fm.n_cells)
    np.add.at(ref, fm.owner, flux)
    np.add.at(ref, fm.neighbour, flux[:len(fm.neighbour)])
    ref = 0.5 * dt * ref / cg["volumes"]                                          # Co_P = 0.5 dt sum |U.S| / V
    assert np.allclose(co, ref, rtol=1e-9)
    wd = MT.mesh_wall_distance(pitz)
    assert wd.shape == (fm.n_cells,) and np.all(wd > 0)


# ------------------------------------------------------------------------------------------------ datasets
def test_dataset_writers_produce_readable_files():
    deck = INP.read_deck(open(os.path.join(EX, "calculix_cantilever_hex.inp")).read())
    m = INP.deck_mesh(deck)
    m.point_data["T"] = np.linspace(300.0, 400.0, m.n_points)
    ds = DS.build_dataset(m)
    assert DS.quantity_of("T")[0] is not None and DS.quantity_of("p")[0] is not None
    lines = DS.write_csv(ds).decode().splitlines()
    assert lines[0].startswith("# pinneapple.physical_dataset") and lines[1].split(",")[3].startswith("T")
    assert len(lines) == 2 + m.n_points
    npz = np.load(_io.BytesIO(DS.write_npz(ds)))
    assert len(npz.files) >= 2
    js = json.loads(DS.write_json(ds))
    assert js
    assert DS.write_vtu(ds).lstrip().startswith(b"<?xml") or b"<VTKFile" in DS.write_vtu(ds)[:400]
    assert len(DS.write_pinneapple(ds)) > 0
    assert DS.preview(ds)


def _hex_dataset():
    m = INP.deck_mesh(INP.read_deck(open(os.path.join(EX, "calculix_cantilever_hex.inp")).read()))
    m.point_data["T"] = np.linspace(300.0, 400.0, m.n_points)
    return DS.build_dataset(m)


def test_hdf5_writer_starts_with_the_hdf5_signature():
    pytest.importorskip("h5py")
    assert DS.write_hdf5(_hex_dataset())[:8] == b"\x89HDF\r\n\x1a\n"


def test_parquet_writer_starts_with_the_parquet_signature():
    pytest.importorskip("pyarrow")
    assert DS.write_parquet(_hex_dataset())[:4] == b"PAR1"


# ------------------------------------------------------------------------------------------------ comparison
def test_comparison_helpers_on_two_copies_of_a_field():
    deck = INP.read_deck(open(os.path.join(EX, "calculix_cantilever_hex.inp")).read())
    a = INP.deck_mesh(deck)
    b = INP.deck_mesh(deck)
    a.point_data["T"] = a.points[:, 0].copy()
    b.point_data["T"] = b.points[:, 0].copy()
    fa = CP.fields_of(a)
    assert "T" in fa
    pts, where, w = CP.locations(a, "T")
    assert where == "point" and len(pts) == a.n_points
    assert np.allclose(CP.field_value(a, "T"), a.points[:, 0])
    vals = CP.Interpolator(a.points, a.points)(a.points[:, 0])
    assert np.allclose(vals, a.points[:, 0])                                     # identical locations: exact copy
    assert CP.surface(a) is not None
    res = CP.compare(a, b)
    assert "T" in json.dumps(CP.views(a, res))
