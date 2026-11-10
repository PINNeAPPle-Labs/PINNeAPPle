"""Visual LOD contract: no physics field resampling and no false compression claims."""
import json
import numpy as np
import pytest

from pinneapple_twin3d import Scene
from pinneapple_twin3d.lod import build_visual_lods


def _grid(n=18):
    xy = np.array([(x, y, 0) for y in range(n) for x in range(n)], np.float32)
    faces = []
    for y in range(n-1):
        for x in range(n-1):
            a = y*n+x
            faces += [[a, a+1, a+n], [a+1, a+n+1, a+n]]
    return xy, np.asarray(faces, np.uint32)


def test_visual_lods_leave_physics_and_original_untouched(tmp_path):
    v, f = _grid()
    sc = Scene("field-backed")
    sc.add_part("plate", v, f)
    sc.add_field("plate", "temperature", v[:, 0], unit="K")
    sc.export(str(tmp_path), with_viewer=False)
    original_glb = (tmp_path / "geometry.glb").read_bytes()
    fields = (tmp_path / "fields.bin").read_bytes()
    result = build_visual_lods(sc, tmp_path, ratios=(0.5,))
    manifest = json.loads((tmp_path / "scene.json").read_text())
    assert result["levels"][0]["parts"][0]["reason"] == "field_topology"
    assert manifest["visual_lods"][0]["triangles"] == len(f)
    assert (tmp_path / "geometry_lod1.glb").exists()
    assert (tmp_path / "geometry.glb").read_bytes() == original_glb
    assert (tmp_path / "fields.bin").read_bytes() == fields


def test_invalid_ratios_rejected_before_processing(tmp_path):
    sc = Scene("empty")
    with pytest.raises(ValueError, match="strictly decreasing"):
        build_visual_lods(sc, tmp_path, ratios=(0.2, 0.5))


def test_simplification_requires_real_backend(tmp_path, monkeypatch):
    v, f = _grid()
    sc = Scene("mesh")
    sc.add_part("plate", v, f)
    sc.export(str(tmp_path), with_viewer=False)
    import pinneapple_twin3d.lod as lod
    def fail(*args, **kwargs):
        raise ImportError("backend missing")
    monkeypatch.setattr(lod, "_simplify", fail)
    with pytest.raises(ImportError, match="backend missing"):
        build_visual_lods(sc, tmp_path, ratios=(0.5,))


def test_viewer_lod_is_opt_in_and_never_replaces_field_geometry():
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] /
          "pinneapple_twin3d" / "viewer" / "viewer.js").read_text()
    assert 'params.get("lod")' in js
    assert '!m.fields.length' in js
    assert 'visualLod ? visualLod.file : m.geometry' in js


def test_multiple_lods_are_declared_for_field_backed_mesh(tmp_path):
    v, f = _grid()
    sc = Scene("field-backed")
    sc.add_part("plate", v, f)
    sc.add_field("plate", "temperature", v[:, 0])
    sc.export(str(tmp_path), with_viewer=False)
    result = build_visual_lods(sc, tmp_path, ratios=(0.5, 0.2))
    assert len(result["levels"]) == 2
    assert all(x["parts"][0]["reason"] == "field_topology" for x in result["levels"])
