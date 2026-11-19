from pinneapple_analysis.verification.artifact_contracts import (
    FieldContract, check_field_contracts, check_artifact_metadata, check_lineage_compatibility,
)
from pinneapple_analysis.verification.pipeline_assurance import Artifact, Status


def test_field_contract_detects_units_and_frame():
    checks = check_field_contracts("field_extraction",
        [FieldContract("stress", "Pa", 6, "global", "cell")],
        [FieldContract("stress", "MPa", 6, "local", "cell")])
    assert checks[0].status == Status.FAIL
    assert set(checks[0].evidence["mismatches"]) == {"unit", "frame"}


def test_geometry_requires_unit():
    assert check_artifact_metadata(Artifact("cad", "geometry"))[0].status == Status.FAIL
    assert check_artifact_metadata(Artifact("cad", "geometry", metadata={"length_unit": "mm"}))[0].status == Status.PASS


def test_mesh_and_boundary_coverage():
    mesh = Artifact("mesh", "mesh", metadata={"length_unit": "m", "node_count": 100, "cell_count": 200})
    assert all(c.status == Status.PASS for c in check_artifact_metadata(mesh))
    bc = Artifact("bc", "boundary_conditions",
                  metadata={"required_regions": ["wall", "inlet"], "assigned_regions": ["wall"]})
    assert check_artifact_metadata(bc)[0].status == Status.FAIL


def test_lineage_detects_undocumented_conversion():
    cad = Artifact("cad", "geometry", metadata={"length_unit": "mm", "coordinate_frame": "world"})
    mesh = Artifact("mesh", "mesh", ("cad",), {"length_unit": "m", "coordinate_frame": "world"})
    checks = check_lineage_compatibility(cad, mesh)
    assert [c.status for c in checks] == [Status.FAIL, Status.PASS]
