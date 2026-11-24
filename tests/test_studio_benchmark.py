"""The visual benchmark: the reference cases, the record it writes, and the deltas against a baseline."""
import json
import os

import pytest

from pinneapple_tools.visualization.studio import benchmark as BM
from pinneapple_tools.visualization.studio import blender as B


def test_every_case_points_at_a_geometry_that_ships_with_the_repository():
    root = BM.repository_root()
    for name, case in BM.CASES.items():
        path = os.path.join(root, "data", "geometries", case["geometry"], f"{case['geometry']}.light.glb")
        assert os.path.exists(path), f"{name}: {path} is not in the repository"
        assert case["materials"], f"{name} has no materials"
        assert case["render"]["background"], f"{name} has no background"


def test_every_case_uses_a_camera_the_renderer_knows():
    for name, case in BM.CASES.items():
        assert case["render"]["camera"] in B.CAMERAS, name
        for key in ("hdri_strength", "hdri_rotation"):
            if key in case["render"]:
                assert isinstance(case["render"][key], (int, float)), name


def test_compare_subtracts_every_statistic():
    a = {"x": {"mean": 10.0, "std": 2.0, "coverage": 0.5, "edge_energy": 1.0, "seconds": 4.0}}
    b = {"x": {"mean": 12.5, "std": 1.5, "coverage": 0.4, "edge_energy": 0.8, "seconds": 6.5}}
    d = BM.compare(a, b)                                   # current - baseline
    assert d["x"] == {"mean": -2.5, "std": 0.5, "coverage": 0.1, "edge_energy": 0.2, "seconds": -2.5}
    assert BM.compare(a, {}) == {}, "a case the baseline does not have is skipped, not invented"


def test_picture_stats_read_a_flat_image_and_a_busy_one(tmp_path):
    from PIL import Image
    import numpy as np

    flat = tmp_path / "flat.png"
    Image.new("RGB", (64, 48), (120, 120, 120)).save(flat)
    s = BM.picture_stats(str(flat))
    assert s["mean"] == pytest.approx(120.0, abs=1.0) and s["coverage"] == 0.0 and s["edge_energy"] < 0.01

    busy = tmp_path / "busy.png"
    a = np.zeros((48, 64, 3), np.uint8)
    a[:, 32:] = 255
    Image.fromarray(a).save(busy)
    s = BM.picture_stats(str(busy))
    assert s["coverage"] > 0.45, "half the frame differs from the corner pixel"
    assert s["edge_energy"] > 0.5


@pytest.mark.skipif(B.blender_executable() is None, reason="needs a Blender binary on this machine")
def test_a_run_writes_the_pictures_and_the_record(tmp_path):
    rec = BM.run(str(tmp_path), scenes=["heatsink_aluminium"], quality="preview", size=(160, 120), samples=2,
                 device="cpu", log=lambda *_: None)
    assert set(rec) == {"heatsink_aluminium"}
    r = rec["heatsink_aluminium"]
    assert r["size"] == [160, 120] and 0 <= r["coverage"] <= 1
    assert r["mean"] > 5 and r["edge_energy"] >= 0
    with open(tmp_path / "benchmark.json", encoding="utf-8") as fh:
        assert json.load(fh)["heatsink_aluminium"]["seconds"] == r["seconds"]
