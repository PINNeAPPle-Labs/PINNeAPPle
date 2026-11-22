"""OpenFOAM extraction specification contracts; not executed."""
import json

import pytest

from pinneapple_simulation.external_solvers.openfoam.flux_extraction_plan import (
    write_openfoam_flux_extraction_plan,
)


def test_plan_requires_boundary_faces_and_source_terms(tmp_path):
    result = write_openfoam_flux_extraction_plan(
        tmp_path, time="10", output_path=tmp_path / "plan.json")
    assert result["execution_status"] == "not_executed"
    assert result["scientific_status"] == "plan_only_no_solver_evidence"
    assert result["required_export"]["boundary_faces"] == "mandatory"
    assert "independently" in result["required_export"]["cell_sources"]
    assert json.loads((tmp_path / "plan.json").read_text()) == result


@pytest.mark.parametrize("field", ["../phi", "a/b", ""])
def test_invalid_field_is_rejected(tmp_path, field):
    with pytest.raises(ValueError):
        write_openfoam_flux_extraction_plan(
            tmp_path, time="10", surface_field=field,
            output_path=tmp_path / "plan.json")
