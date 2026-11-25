"""CAD regression tests (requires cadquery)."""
import pytest
cq = pytest.importorskip("cadquery")
from cad_brackets import build, export_family


@pytest.mark.parametrize("family", ["L", "G", "U", "Z"])
def test_cad_is_valid_single_solid(family):
    shape = build(family)
    assert shape.val().isValid()
    assert len(shape.solids().vals()) == 1
    assert shape.val().Volume() > 0


@pytest.mark.parametrize("family", ["L", "G", "U", "Z"])
def test_step_roundtrip(family, tmp_path):
    path = tmp_path / f"{family}.step"
    info = export_family(family, path)
    assert path.exists() and path.stat().st_size > 0
    loaded = cq.importers.importStep(str(path))
    assert loaded.val().isValid()
    assert loaded.val().Volume() == pytest.approx(info["volume_mm3"], rel=1e-5)


def test_reject_invalid_parameters():
    with pytest.raises(ValueError):
        build("L", thickness=100)
    with pytest.raises(ValueError):
        build("unknown")
