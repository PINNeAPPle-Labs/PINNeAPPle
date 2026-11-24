"""OpenFOAM case -> PyVista grid with fields per time (#60): the readers, the two conversions, the patches, the image.

The conversion itself (``polyhedron_cells``, ``patch_surface``, ``_read_mesh``) is NumPy and runs everywhere; the
tests that build a PyVista object say so with a skip, because PyVista is an extra (``pip install
"pinneapple[pyvista]"``). The rendering test also needs an OpenGL context and is skipped where there is none.
"""
import os

import numpy as np
import pytest

from pinneapple_tools.visualization import openfoam as of
from pinneapple_tools.visualization.openfoam import (
    case_folder,
    openfoam_times,
    patch_surface,
    polyhedron_cells,
    read_grid,
    read_openfoam,
    read_openfoam_patches,
    read_vtk,
)

HEADER = """FoamFile
{
    version     2.0;
    format      ascii;
    class       %s;
    object      %s;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def _write(path, cls, obj, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(HEADER % (cls, obj) + body)


def _block_case(root, nx=3, ny=2, nz=2):
    """A structured block of ``nx*ny*nz`` unit hexahedral cells, written the way blockMesh writes it."""
    pid = lambda i, j, k: i + (nx + 1) * (j + (ny + 1) * k)                # noqa: E731
    cid = lambda i, j, k: i + nx * (j + ny * k)                            # noqa: E731
    pts = [(i, j, k) for k in range(nz + 1) for j in range(ny + 1) for i in range(nx + 1)]
    internal, boundary = [], []            # (nodes, owner, neighbour) / (nodes, owner)
    for i in range(nx + 1):                # faces normal to x, node order gives +x
        for j in range(ny):
            for k in range(nz):
                f = [pid(i, j, k), pid(i, j + 1, k), pid(i, j + 1, k + 1), pid(i, j, k + 1)]
                if i == 0:
                    boundary.append((f[::-1], cid(0, j, k)))              # outward is -x here
                elif i == nx:
                    boundary.append((f, cid(nx - 1, j, k)))
                else:
                    internal.append((f, cid(i - 1, j, k), cid(i, j, k)))
    for j in range(ny + 1):                # faces normal to y
        for i in range(nx):
            for k in range(nz):
                f = [pid(i, j, k), pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j, k)]
                if j == 0:
                    boundary.append((f[::-1], cid(i, 0, k)))
                elif j == ny:
                    boundary.append((f, cid(i, ny - 1, k)))
                else:
                    internal.append((f, cid(i, j - 1, k), cid(i, j, k)))
    for k in range(nz + 1):                # faces normal to z
        for i in range(nx):
            for j in range(ny):
                f = [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)]
                if k == 0:
                    boundary.append((f[::-1], cid(i, j, 0)))
                elif k == nz:
                    boundary.append((f, cid(i, j, nz - 1)))
                else:
                    internal.append((f, cid(i, j, k - 1), cid(i, j, k)))
    internal.sort(key=lambda x: (x[1], x[2]))                   # OpenFOAM: internal faces first, by owner
    faces = [f for f, _, _ in internal] + [f for f, _ in boundary]
    owner = [o for _, o, _ in internal] + [o for _, o in boundary]
    neigh = [n for _, _, n in internal]
    pm = os.path.join(root, "constant", "polyMesh")
    _write(os.path.join(pm, "points"), "vectorField", "points",
           "%d\n(\n" % len(pts) + "\n".join("(%g %g %g)" % p for p in pts) + "\n)\n")
    _write(os.path.join(pm, "faces"), "faceList", "faces",
           "%d\n(\n" % len(faces) + "\n".join("4(%d %d %d %d)" % tuple(f) for f in faces) + "\n)\n")
    _write(os.path.join(pm, "owner"), "labelList", "owner",
           "%d\n(\n" % len(owner) + "\n".join(map(str, owner)) + "\n)\n")
    _write(os.path.join(pm, "neighbour"), "labelList", "neighbour",
           "%d\n(\n" % len(neigh) + "\n".join(map(str, neigh)) + "\n)\n")
    _write(os.path.join(pm, "boundary"), "polyBoundaryMesh", "boundary",
           "1\n(\n    walls\n    {\n        type            wall;\n        nFaces          %d;\n"
           "        startFace       %d;\n    }\n)\n" % (len(boundary), len(internal)))
    n_cells = nx * ny * nz
    p = [i / nx for k in range(nz) for j in range(ny) for i in range(nx)]
    _write(os.path.join(root, "0", "p"), "volScalarField", "p",
           "dimensions      [0 2 -2 0 0 0 0];\ninternalField   nonuniform List<scalar>\n%d\n(\n" % n_cells
           + "\n".join("%g" % v for v in p) + "\n)\n;\nboundaryField\n{\n    walls\n    {\n"
           "        type            zeroGradient;\n    }\n}\n")
    return str(root), n_cells, np.asarray(p)


@pytest.fixture(scope="module")
def stacked_case(tmp_path_factory):
    """Two hexahedral cells stacked in z (volume 2 m^3), one internal face, two patches, two time directories."""
    c = tmp_path_factory.mktemp("case")
    pm = c / "constant" / "polyMesh"
    pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
           (0, 0, 2), (1, 0, 2), (1, 1, 2), (0, 1, 2)]
    _write(str(pm / "points"), "vectorField", "points",
           "12\n(\n" + "\n".join(f"({x} {y} {z})" for x, y, z in pts) + "\n)\n")
    # face 0 internal (owner 0, neighbour 1); faces 1..9 "walls" (owner 0 on 1..5, owner 1 on 6..9); face 10 "outlet"
    faces = ["4(4 5 6 7)",
             "4(0 3 2 1)", "4(0 4 7 3)", "4(1 2 6 5)", "4(0 1 5 4)", "4(3 7 6 2)",
             "4(4 8 11 7)", "4(5 6 10 9)", "4(4 5 9 8)", "4(7 11 10 6)",
             "4(8 9 10 11)"]
    _write(str(pm / "faces"), "faceList", "faces", "11\n(\n" + "\n".join(faces) + "\n)\n")
    _write(str(pm / "owner"), "labelList", "owner",
           "11\n(\n0\n0\n0\n0\n0\n0\n1\n1\n1\n1\n1\n)\n")
    _write(str(pm / "neighbour"), "labelList", "neighbour", "1\n(\n1\n)\n")
    _write(str(pm / "boundary"), "polyBoundaryMesh", "boundary", """2
(
    walls
    {
        type            wall;
        nFaces          9;
        startFace       1;
    }
    outlet
    {
        type            patch;
        nFaces          1;
        startFace       10;
    }
)
""")
    # t = 0: p per cell 1 and 3, walls 10 (fixedValue), outlet zeroGradient; U as a vector field
    _write(str(c / "0" / "p"), "volScalarField", "p", """dimensions      [0 2 -2 0 0 0 0];
internalField   nonuniform List<scalar>
2
(
1
3
)
;
boundaryField
{
    walls
    {
        type            fixedValue;
        value           uniform 10;
    }
    outlet
    {
        type            zeroGradient;
    }
}
""")
    _write(str(c / "0" / "U"), "volVectorField", "U", """dimensions      [0 1 -1 0 0 0 0];
internalField   nonuniform List<vector>
2
(
(1 0 0)
(0 0 1)
)
;
boundaryField
{
    walls
    {
        type            noSlip;
    }
    outlet
    {
        type            fixedValue;
        value           uniform (2 0 0);
    }
}
""")
    # t = 1: p doubled, the walls value written per face, and the same U field
    _write(str(c / "1" / "p"), "volScalarField", "p", """dimensions      [0 2 -2 0 0 0 0];
internalField   nonuniform List<scalar>
2
(
2
6
)
;
boundaryField
{
    walls
    {
        type            fixedValue;
        value           nonuniform List<scalar>
9
(
20 21 22 23 24 25 26 27 28
)
;
    }
    outlet
    {
        type            zeroGradient;
    }
}
""")
    _write(str(c / "1" / "U"), "volVectorField", "U", """dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 2);
boundaryField
{
    walls
    {
        type            noSlip;
    }
    outlet
    {
        type            fixedValue;
        value           uniform (2 0 0);
    }
}
""")
    return str(c)


# ------------------------------------------------------------------------------- no PyVista needed
def test_the_case_is_found_from_the_folder_and_from_a_foam_pointer(stacked_case):
    assert case_folder(stacked_case) == os.path.abspath(stacked_case)
    pointer = os.path.join(stacked_case, "case.foam")
    open(pointer, "w").close()                                  # the empty file ParaView opens
    assert case_folder(pointer) == os.path.abspath(stacked_case)
    with pytest.raises(FileNotFoundError, match="not an OpenFOAM case"):
        case_folder(os.path.join(stacked_case, "constant"))


def test_time_directories_come_out_sorted_and_checked(stacked_case):
    assert openfoam_times(stacked_case) == [0.0, 1.0]
    with pytest.raises(KeyError, match="no time directory"):
        of._read_mesh(stacked_case, time=7)


def test_polyhedron_stream_lists_every_face_once_for_the_owner_and_reversed_for_the_neighbour(stacked_case):
    mesh, info = of._read_mesh(stacked_case, time=1)
    stream, offsets = polyhedron_cells(mesh.poly)
    assert len(offsets) == mesh.poly.n_cells + 1 == 3
    first = stream[offsets[0]:offsets[1]].tolist()               # cell 0: 6 faces
    second = stream[offsets[1]:offsets[2]].tolist()              # cell 1: 6 faces
    assert first[0] == second[0] == 6
    assert len(first) == len(second) == 1 + 6 * 5
    assert [4, 0, 3, 2, 1] in [first[i:i + 5] for i in (1, 6, 11, 16, 21, 26)]
    assert np.all(stream >= 0) and int(stream.max()) < len(mesh.points)
    # the shared face keeps the owner's node order on cell 0 and is reversed on cell 1
    assert first[1:6] == [4, 4, 5, 6, 7]
    assert second[1:6] == [4, 7, 6, 5, 4]


def test_patch_surface_duplicates_vertices_so_a_face_keeps_its_value(stacked_case):
    mesh, _ = of._read_mesh(stacked_case, time=1)
    poly = mesh.poly
    pb = next(p for p in poly.patches if p["name"] == "walls")
    verts, tris, owner = patch_surface(mesh.points, poly.face_nodes, poly.face_offsets,
                                       pb["startFace"], pb["nFaces"])
    assert verts.shape == (9 * 4, 3) and tris.shape == (9 * 2, 3)
    np.testing.assert_array_equal(owner, np.repeat(np.arange(9), 2))       # which face each triangle came from
    np.testing.assert_allclose(mesh.boundary_values["p"][0:9], np.arange(20.0, 29.0))
    # fan triangulation stays inside its own face: the four corners of face 3 of the patch
    tri = tris[6]
    assert set(tri.tolist()) <= set(range(12, 16))


def test_fields_of_one_time_are_read_from_the_boundary_field(stacked_case):
    mesh, info = of._read_mesh(stacked_case, time=0)
    assert info["time"] == 0.0 and sorted(info["fields"]) == ["U", "p"]
    np.testing.assert_allclose(mesh.cell_data["p"], [1.0, 3.0])
    assert mesh.cell_data["U"].shape == (2, 3)
    np.testing.assert_allclose(mesh.boundary_values["U"][9], [2.0, 0.0, 0.0])     # outlet, fixedValue
    np.testing.assert_allclose(mesh.boundary_values["U"][0:9], 0.0)              # walls, noSlip
    mesh1, _ = of._read_mesh(stacked_case, time="1")                             # by directory name
    np.testing.assert_allclose(mesh1.cell_data["p"], [2.0, 6.0])


# ------------------------------------------------------------------------------- the PyVista grids
def test_volume_grid_keeps_every_cell_its_own_faces(stacked_case):
    pv = pytest.importorskip("pyvista")            # optional extra: pip install "pinneapple[pyvista]"
    grid = read_openfoam(stacked_case)
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells == 2
    # a polyhedron count or a wrong face order would not integrate to the two unit cubes
    assert grid.volume == pytest.approx(2.0, rel=1e-12)
    np.testing.assert_allclose(sorted(grid.cell_centers().points[:, 2].tolist()), [0.5, 1.5])
    assert grid.n_points == 12 and grid.bounds == (0.0, 1.0, 0.0, 1.0, 0.0, 2.0)
    np.testing.assert_allclose(grid["p"], [2.0, 6.0])                     # the latest time (t = 1)
    assert grid["U"].shape == (2, 3)
    assert str(grid.field_data["source"][0]).endswith("OpenFOAM case " + os.path.abspath(stacked_case))
    assert float(grid.field_data["time"][0]) == 1.0
    # a slice through the polyhedra gives the values back: the grid is a real volume, not a surface
    sl = grid.slice(normal="z", origin=(0.5, 0.5, 1.5))
    assert sl.n_points > 0 and np.isfinite(sl["p"]).all()


def test_time_and_field_selection(stacked_case):
    pytest.importorskip("pyvista")
    np.testing.assert_allclose(read_openfoam(stacked_case, time=0)["p"], [1.0, 3.0])
    np.testing.assert_allclose(read_openfoam(stacked_case, time="0")["p"], [1.0, 3.0])
    only_u = read_openfoam(stacked_case, time=0, fields=["U"])
    assert "U" in only_u.array_names and "p" not in only_u.array_names
    with pytest.raises(KeyError, match="no time directory"):
        read_openfoam(stacked_case, time=7)


def test_all_times_is_one_block_per_time_directory(stacked_case):
    pv = pytest.importorskip("pyvista")
    blocks = read_openfoam(stacked_case, all_times=True)
    assert isinstance(blocks, pv.MultiBlock)
    assert list(blocks.keys()) == ["t=0", "t=1"]
    np.testing.assert_allclose(blocks["t=0"]["p"], [1.0, 3.0])


def test_patches_carry_the_wall_value_not_the_first_cell_value(stacked_case):
    pv = pytest.importorskip("pyvista")
    patches = read_openfoam_patches(stacked_case)
    assert isinstance(patches, pv.MultiBlock)
    assert list(patches.keys()) == ["walls", "outlet"]
    walls = patches["walls"]
    assert isinstance(walls, pv.PolyData)
    assert walls.n_cells == 18 and walls.n_points == 36          # 9 quads: two triangles each, vertices not shared
    np.testing.assert_allclose(walls["p"], np.repeat(np.arange(20.0, 29.0), 2))
    np.testing.assert_allclose(patches["outlet"]["p"], 6.0)       # zeroGradient: the owner cell's value
    at_zero = read_openfoam_patches(stacked_case, time=0)
    np.testing.assert_allclose(at_zero["walls"]["p"], 10.0)
    np.testing.assert_allclose(at_zero["walls"]["U"], 0.0)        # noSlip
    only = read_openfoam_patches(stacked_case, names=["outlet"])
    assert list(only.keys()) == ["outlet"]
    with pytest.raises(KeyError, match="nope"):
        read_openfoam_patches(stacked_case, names=["nope"])


def test_patches_over_all_times(stacked_case):
    pv = pytest.importorskip("pyvista")
    blocks = read_openfoam_patches(stacked_case, all_times=True)
    assert isinstance(blocks, pv.MultiBlock) and list(blocks.keys()) == ["t=0", "t=1"]
    assert list(blocks["t=1"].keys()) == ["walls", "outlet"]


def test_a_foam_pointer_reads_the_folder_next_to_it(stacked_case):
    pytest.importorskip("pyvista")
    pointer = os.path.join(stacked_case, "pointer.foam")
    open(pointer, "w").close()
    grid = read_grid(pointer)
    assert grid.n_cells == 2 and grid.volume == pytest.approx(2.0)
    assert read_grid(stacked_case, patches=True).n_blocks == 2


def test_vtk_files_go_through_pyvista_and_take_no_case_options(stacked_case, tmp_path):
    pv = pytest.importorskip("pyvista")
    grid = read_openfoam(stacked_case)
    path = str(tmp_path / "case.vtu")
    grid.save(path)
    back = read_vtk(path)
    assert isinstance(back, pv.UnstructuredGrid) and back.n_cells == 2
    assert read_grid(path).n_cells == 2
    with pytest.raises(TypeError, match="a VTK file takes no OpenFOAM options"):
        read_grid(path, time=0)


def test_three_lines_of_code_make_an_image(stacked_case, tmp_path):
    pytest.importorskip("pyvista")
    from pinneapple_tools.visualization.pyvista_bridge import can_render, render_png

    if not can_render():
        pytest.skip("no off-screen OpenGL context: run under `xvfb-run -a` or install Mesa (see headless_rendering.md)")
    out = str(tmp_path / "p.png")
    grid = read_openfoam(stacked_case)                       # 1
    render_png(grid, out, scalars="p", title="p")            # 2, 3
    assert os.path.getsize(out) > 1000

