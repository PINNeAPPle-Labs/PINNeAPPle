"""Offline contracts for visual compression and USD polygon triangulation."""
import json
from unittest.mock import patch

import pytest

from pinneapple_twin3d.simready_usd import _triangulate
from pinneapple_twin3d.compress import optimize_visual_assets


def test_usd_polygon_triangulation():
    faces = _triangulate([3, 4], [0, 1, 2, 3, 4, 5, 6])
    assert faces.tolist() == [[0, 1, 2], [3, 4, 5], [3, 5, 6]]


def test_usd_polygon_rejects_mismatched_indices():
    with pytest.raises(ValueError):
        _triangulate([3], [0, 1])


def test_compression_preserves_original(tmp_path):
    (tmp_path / "geometry.glb").write_bytes(b"original")
    (tmp_path / "scene.json").write_text(json.dumps({"geometry": "geometry.glb"}))

    def fake_run(cmd, **kwargs):
        assert "-cc" in cmd
        (tmp_path / "geometry_meshopt.glb").write_bytes(b"compressed")

    with patch("pinneapple_twin3d.compress.shutil.which", return_value="/usr/bin/gltfpack"), \
         patch("pinneapple_twin3d.compress.subprocess.run", side_effect=fake_run):
        report = optimize_visual_assets(tmp_path)
    assert (tmp_path / "geometry.glb").read_bytes() == b"original"
    assert report["requires_meshopt_decoder"]
    assert json.loads((tmp_path / "scene.json").read_text())["visual_compressed"][0]["file"] == "geometry_meshopt.glb"


def test_compression_rejects_traversal(tmp_path):
    (tmp_path / "scene.json").write_text(json.dumps({"geometry": "../outside.glb"}))
    with patch("pinneapple_twin3d.compress.shutil.which", return_value="/usr/bin/gltfpack"):
        with pytest.raises(ValueError):
            optimize_visual_assets(tmp_path)
