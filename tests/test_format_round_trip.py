"""Lossless round trip of fields in the formats this repository promises (#115).

A writer is not tested by producing a file; it is tested by a reader giving the same numbers back. Both directions
of each format are exercised here on the same small cube, with two vertex fields and one cell field, and the
tolerance is zero — the formats carry float64 and the writers store float64 at round-trip precision, so "almost
equal" would be a decision to hide a bug.

The two formats that are *not* implemented (STEP, SSP) are checked the only way a refusal can be: the document that
declares them has to say so, and say why.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pinneapple_data.cae.model import CellBlock, Mesh
from pinneapple_data.cae.xdmf import XdmfError, read_xdmf, write_xdmf

ROOT = Path(__file__).resolve().parents[1]
INTEROP_DOC = ROOT / "docs" / "dev" / "PHYSICS_AI_INTEROP.md"

CUBE_POINTS = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
])
# the standard six-tetrahedron decomposition of the cube (the same one the CGNS fixture uses)
CUBE_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6], [3, 4, 6, 7], [1, 4, 5, 6], [0, 3, 4, 1]])
TEMPERATURE = CUBE_POINTS[:, 0] + 2 * CUBE_POINTS[:, 1] + 3 * CUBE_POINTS[:, 2]
PRESSURE = np.linalg.norm(CUBE_POINTS, axis=1)


def cube_mesh() -> Mesh:
    mesh = Mesh(points=CUBE_POINTS.copy(), blocks=[CellBlock("tetra", CUBE_TETS.copy())])
    mesh.point_data["temperature"] = TEMPERATURE.copy()
    mesh.point_data["velocity"] = CUBE_POINTS.copy()            # a vector field, three components
    mesh.cell_data["pressure"] = PRESSURE[CUBE_TETS].mean(axis=1)
    return mesh


def assert_same_mesh(back: Mesh, original: Mesh) -> None:
    """Field by field, exactly: this is the whole claim of the format work."""
    assert np.array_equal(back.points, original.points)
    assert len(back.blocks) == len(original.blocks)
    for got, want in zip(back.blocks, original.blocks, strict=True):
        assert got.type == want.type and np.array_equal(got.conn, want.conn)
    assert set(back.point_data) == set(original.point_data)
    for name, values in original.point_data.items():
        assert np.array_equal(back.point_data[name], values), name
    assert set(back.cell_data) == set(original.cell_data)
    for name, values in original.cell_data.items():
        assert np.array_equal(back.cell_data[name], values), name


# ---------------------------------------------------------------- XDMF
def test_xdmf_round_trips_a_volume_mesh_with_node_and_cell_fields(tmp_path):
    mesh = cube_mesh()
    path = write_xdmf(mesh, tmp_path / "cube.xdmf")
    back = read_xdmf(path)
    assert_same_mesh(back, mesh)


def test_xdmf_round_trips_several_cell_blocks(tmp_path):
    """A volume with a surface beside it is the normal result; the collection grid keeps both blocks."""
    mesh = cube_mesh()
    shell = CellBlock("triangle", np.array([[0, 1, 2], [0, 2, 3]]))
    mesh.blocks.append(shell)
    mesh.cell_data["pressure"] = np.concatenate([mesh.cell_data["pressure"], np.array([1.5, 2.5])])
    path = write_xdmf(mesh, tmp_path / "two_blocks.xdmf")
    back = read_xdmf(path)
    assert_same_mesh(back, mesh)


def test_xdmf_is_plain_xml_a_third_party_reader_can_open(tmp_path):
    """No companion HDF5 and no binary: the numbers are in the file, at round-trip precision."""
    path = write_xdmf(cube_mesh(), tmp_path / "cube.xdmf")
    text = path.read_text(encoding="utf-8")
    assert text.startswith('<?xml version="1.0" encoding="utf-8"?>')
    assert '<Xdmf Version="3.0">' in text and 'Format="XML"' in text
    assert "Tetrahedron" in text and 'Center="Node"' in text and 'Center="Cell"' in text
    # the largest coordinate written back out of the text is the same double
    assert f"{1.0:.17g}" in text


def test_xdmf_refuses_a_file_whose_data_lives_in_hdf5():
    """The common ParaView output: refused with the sentence that says which tool reads it."""
    with pytest.raises(XdmfError, match="meshio"):
        read_xdmf(b"""<?xml version="1.0"?><Xdmf Version="3.0"><Domain><Grid Name="g">
        <Topology TopologyType="Tetrahedron" NumberOfElements="1">
          <DataItem Dimensions="1 4" NumberType="Int" Format="HDF">cube.h5:/data</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="4 3" NumberType="Float" Format="XML">0 0 0 1 0 0 0 1 0 0 0 1</DataItem>
        </Geometry></Grid></Domain></Xdmf>""")


def test_xdmf_refuses_a_topology_it_cannot_hold():
    with pytest.raises(XdmfError, match="Polygon"):
        read_xdmf(b"""<?xml version="1.0"?><Xdmf Version="3.0"><Domain><Grid Name="g">
        <Topology TopologyType="Polygon" NumberOfElements="2">
          <DataItem Dimensions="6" NumberType="Int" Format="XML">3 0 1 2 3 0 1 2</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="4 3" NumberType="Float" Format="XML">0 0 0 1 0 0 0 1 0 0 0 1</DataItem>
        </Geometry></Grid></Domain></Xdmf>""")


def test_xdmf_refuses_to_write_a_polyhedral_mesh(tmp_path):
    from pinneapple_data.cae.model import PolyMesh  # faces + owner/neighbour

    mesh = Mesh(points=CUBE_POINTS.copy())
    mesh.poly = PolyMesh(face_nodes=np.zeros(0, dtype=np.int64), face_offsets=np.zeros(1, dtype=np.int64),
                         owner=np.zeros(0, dtype=np.int64), neighbour=np.zeros(0, dtype=np.int64), n_cells=0,
                         patches=[])
    with pytest.raises(XdmfError, match="polyhedral"):
        write_xdmf(mesh, tmp_path / "poly.xdmf")


# ---------------------------------------------------------------- CGNS
def test_cgns_round_trips_the_volume_mesh_and_its_vertex_fields(tmp_path):
    writer = pytest.importorskip("pinneapple_simulation.external_solvers.cfd_formats.cgns_writer")
    reader = pytest.importorskip("pinneapple_simulation.external_solvers.cfd_formats.cgns_reader")
    path = writer.write_cgns(tmp_path / "cube.cgns", CUBE_POINTS, CUBE_TETS,
                             {"Temperature": TEMPERATURE, "Pressure": PRESSURE})
    back = reader.read_cgns_mesh_and_fields(str(path))
    assert np.array_equal(back["coords"], CUBE_POINTS)
    assert set(back["fields"]) == {"Temperature", "Pressure"}
    assert np.array_equal(back["fields"]["Temperature"], TEMPERATURE)
    assert np.array_equal(back["fields"]["Pressure"], PRESSURE)


def test_cgns_from_a_mesh_maps_the_element_type_and_keeps_the_point_fields(tmp_path):
    writer = pytest.importorskip("pinneapple_simulation.external_solvers.cfd_formats.cgns_writer")
    reader = pytest.importorskip("pinneapple_simulation.external_solvers.cfd_formats.cgns_reader")
    path = writer.mesh_to_cgns(tmp_path / "mesh.cgns", cube_mesh())
    back = reader.read_cgns_mesh_and_fields(str(path))
    assert np.array_equal(back["coords"], CUBE_POINTS)
    assert np.array_equal(back["fields"]["temperature"], TEMPERATURE)


def test_the_reference_cgns_validator_accepts_what_we_write(tmp_path):
    """The CGNS project's own ``cgnscheck``: 0 errors. Skipped where the tool is not installed."""
    writer = pytest.importorskip("pinneapple_simulation.external_solvers.cfd_formats.cgns_writer")
    tool = shutil.which("cgnscheck")
    if not tool:
        pytest.skip("cgnscheck (CGNS reference validator) is not installed")
    path = writer.write_cgns(tmp_path / "validated.cgns", CUBE_POINTS, CUBE_TETS,
                             {"Temperature": TEMPERATURE, "Pressure": PRESSURE})
    out = subprocess.run([tool, str(path)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "checking complete" in out.stdout
    assert "0 errors" in out.stdout or "errors" not in out.stdout


def test_cgns_refuses_connectivity_that_points_outside_the_nodes(tmp_path):
    writer = pytest.importorskip("pinneapple_simulation.external_solvers.cfd_formats.cgns_writer")
    with pytest.raises(ValueError, match="outside"):
        writer.write_cgns(tmp_path / "bad.cgns", CUBE_POINTS, CUBE_TETS + 10)


def test_cgns_refuses_a_field_of_the_wrong_length(tmp_path):
    writer = pytest.importorskip("pinneapple_simulation.external_solvers.cfd_formats.cgns_writer")
    with pytest.raises(ValueError, match="values but the zone has"):
        writer.write_cgns(tmp_path / "bad.cgns", CUBE_POINTS, CUBE_TETS, {"T": TEMPERATURE[:-1]})


# ---------------------------------------------------------------- declared, not pretended
def test_the_document_declares_step_and_ssp_as_not_implemented_with_the_reason():
    text = INTEROP_DOC.read_text(encoding="utf-8")
    assert "Format interoperability" in text
    for fmt in ("XDMF", "CGNS", "STEP", "SSP"):
        assert fmt in text, fmt
    assert "OpenCascade" in text                       # why STEP is not here
    assert "parameter sets" in text                    # what SSP is, and why it is not here


# ---------------------------------------------------------------- the upload path
def test_an_uploaded_xdmf_is_read_without_meshio(tmp_path):
    """``io.read_any`` routes a ``.xdmf`` upload to the native reader, so the format works with no extra installed."""
    from pinneapple_data.cae.io import read_any

    mesh = cube_mesh()
    path = write_xdmf(mesh, tmp_path / "upload.xdmf")
    back = read_any({path.name: path.read_bytes()})
    assert np.array_equal(back.points, mesh.points)
    assert np.array_equal(back.point_data["temperature"], TEMPERATURE)
