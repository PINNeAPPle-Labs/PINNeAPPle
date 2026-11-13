"""CAE ⇄ bundle on real solver output: an OpenFOAM case and a CalculiX result file.

The fixtures are the ones the repository already ships (``examples/physics_check/pass/cases/cavity`` and
``examples/calculix_cantilever/beam.frd``), so this exercises the readers that produced them, not a hand-built mesh.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pinneapple_data.cae.model import CellBlock, Mesh
from pinneapple_data.schema import run_gates
from pinneapple_data.schema.cae import (
    EXPORT_FORMATS,
    bundle_from_mesh,
    bundle_from_path,
    bundle_to_mesh,
    export_bundle,
)

REPO = Path(__file__).resolve().parents[2]
CAVITY = REPO / "examples" / "physics_check" / "pass" / "cases" / "cavity"
FRD = REPO / "examples" / "calculix_cantilever" / "beam.frd"

needs_cavity = pytest.mark.skipif(not CAVITY.is_dir(), reason="the OpenFOAM cavity example is not in this checkout")
needs_frd = pytest.mark.skipif(not FRD.is_file(), reason="the CalculiX beam example is not in this checkout")


# ── OpenFOAM: a face-based mesh, kept as faces ────────────────────────────────────────────────────────────────
@needs_cavity
def test_an_openfoam_case_becomes_a_bundle_with_its_patches_and_units():
    bundle = bundle_from_path(CAVITY, license="CC BY 4.0")
    spec = bundle.manifest.mesh
    assert spec.poly is not None, "an OpenFOAM mesh is face-based and must stay that way"
    assert spec.poly.n_cells == 400 and spec.poly.n_faces > spec.poly.n_cells
    assert {p.name for p in spec.poly.patches} >= {"movingWall", "fixedWalls"}
    fields = {f.name: f for f in bundle.manifest.fields.variables}
    assert fields["U"].unit == "m/s" and fields["U"].location == "cell"
    assert fields["p"].unit == "m2/s2"                     # OpenFOAM writes kinematic pressure, and says so
    assert bundle.manifest.provenance.kind == "simulation"


@needs_cavity
def test_the_openfoam_bundle_round_trips_and_exports_vtu(tmp_path):
    bundle = bundle_from_path(CAVITY, license="CC BY 4.0")
    mesh = bundle_to_mesh(bundle)
    assert mesh.poly is not None and mesh.poly.n_cells == 400
    assert mesh.cell_data["U"].shape == (400, 3)
    out = export_bundle(bundle, tmp_path / "cavity", "vtu")
    assert out.suffix == ".vtu" and out.stat().st_size > 10_000
    assert b"UnstructuredGrid" in out.read_bytes()


@needs_cavity
def test_the_gates_report_what_an_unstructured_mesh_cannot_be_asked():
    report = run_gates(bundle_from_path(CAVITY, license="CC BY 4.0"))
    assert report.get("integrity").passed and report.get("units").passed
    assert report.get("location").passed
    assert report.get("frame").status == "not run"         # no coordinate axes: not applicable, not "passed"
    assert "no coordinate axes" in report.get("frame").detail


# ── CalculiX: model units are unknown, and the schema says so ────────────────────────────────────────────────
@needs_frd
def test_a_calculix_result_has_block_cells_and_unknown_units():
    bundle = bundle_from_path(FRD, license="CC BY 4.0")
    spec = bundle.manifest.mesh
    assert [b.type for b in spec.blocks] == ["hexahedron20"]
    assert spec.n_cells == 80
    fields = {f.name: f for f in bundle.manifest.fields.variables}
    assert {"DISP", "STRESS"} <= set(fields)
    assert fields["STRESS"].components and len(fields["STRESS"].components) == 6
    assert all(f.unit == "unknown" for f in fields.values())      # a deck in "model units" declares none
    units_gate = run_gates(bundle, gates=("units",)).get("units")
    assert units_gate.passed is False and "unknown unit" in units_gate.detail


@needs_frd
def test_declaring_the_units_makes_the_bundle_usable():
    bundle = bundle_from_path(FRD, license="CC BY 4.0",
                              units={"DISP": "mm", "STRESS": "Pa", "ERROR": "1"}, geometry_units="mm")
    assert bundle.manifest.geometry.units == "mm"
    report = run_gates(bundle, gates=("integrity", "units", "location", "provenance"))
    assert report.ok, report.summary()
    assert bundle.manifest.fields.get("DISP").unit == "mm"


@needs_frd
def test_a_unit_nobody_can_parse_is_still_an_error():
    with pytest.raises(Exception, match="not a unit"):
        bundle_from_path(FRD, license="CC BY 4.0", units={"DISP": "furlongs"})


# ── a mesh built by hand, and the exports ───────────────────────────────────────────────────────────────────
def hand_mesh() -> Mesh:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                       [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], float)
    return Mesh(points=points, blocks=[CellBlock("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))],
                boundary_blocks=[("wall", CellBlock("triangle", np.array([[0, 1, 2], [1, 3, 2]])))],
                point_data={"T": np.linspace(300.0, 400.0, 8)},
                cell_data={"p": np.array([101325.0])},
                units={"T": "K", "p": "Pa"}, source={"format": "hand-built"})


def test_a_mixed_mesh_keeps_its_boundary_blocks():
    bundle = bundle_from_mesh(hand_mesh(), source="unit test", license="CC0-1.0")
    blocks = bundle.manifest.mesh.blocks
    assert [(b.type, b.boundary) for b in blocks] == [("hexahedron", False), ("triangle", True)]
    assert bundle.manifest.mesh.n_cells == 1          # the boundary is geometry, not a cell of the volume
    assert bundle.manifest.mesh.n_nodes == 8
    back = bundle_to_mesh(bundle)
    assert back.n_cells == 1 and [n for n, _ in back.boundary_blocks] == ["wall"]


def test_a_stored_licence_travels_and_an_input_file_is_hashed(tmp_path):
    path = tmp_path / "mesh.json"
    path.write_text("{}")
    bundle = bundle_from_mesh(hand_mesh(), source="unit test", license="CC0-1.0", inputs=[str(path)])
    assert bundle.manifest.provenance.inputs[0].sha256
    with pytest.raises(ValueError, match="license is required"):
        bundle_from_mesh(hand_mesh(), source="unit test", license="")     # the licence is required


@pytest.mark.parametrize("fmt", ["vtu", "hdf5", "npz", "json", "csv"])
def test_every_export_format_writes_a_file(fmt, tmp_path):
    bundle = bundle_from_mesh(hand_mesh(), source="unit test", license="CC0-1.0")
    out = export_bundle(bundle, tmp_path / f"m_{fmt}", fmt)
    assert out.exists() and out.stat().st_size > 0


def test_an_unknown_export_format_is_refused():
    bundle = bundle_from_mesh(hand_mesh(), source="unit test", license="CC0-1.0")
    with pytest.raises(ValueError, match="unknown export format"):
        export_bundle(bundle, "/tmp/x", "step")
    assert "vtu" in EXPORT_FORMATS


def test_a_structured_bundle_cannot_pretend_to_be_a_cae_mesh():
    from pinneapple_data.schema import AxisSpec, Field, PhysicsDataBundle

    u = np.zeros((4, 8))
    bundle = PhysicsDataBundle.from_arrays(
        {"field:u": u, "axis:t": np.linspace(0, 1, 4), "axis:x": np.linspace(0, 1, 8)},
        fields=[Field.from_array("u", u, axes=("t", "x"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")], source="grid", license="CC0-1.0", kind="synthetic")
    with pytest.raises(ValueError, match="only an unstructured mesh"):
        bundle_to_mesh(bundle)
