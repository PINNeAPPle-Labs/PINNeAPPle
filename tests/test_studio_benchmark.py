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


def _case(**kw):
    base = {"quality": "preview", "size": [960, 540], "mean": 109.0, "std": 23.6, "coverage": 0.68,
            "edge_energy": 1.16, "seconds": 4.0, "png": "/tmp/x.png"}
    base.update(kw)
    return base


def test_the_gate_passes_a_record_against_itself():
    rec = {name: _case() for name in BM.CASES}
    assert BM.gate(rec, BM.baseline_of(rec, reason="same run")) == []


def test_the_gate_catches_a_picture_that_went_black():
    base = BM.baseline_of({name: _case() for name in BM.CASES}, reason="reference")
    rec = {name: _case() for name in BM.CASES}
    rec["impeller_bronze"].update(mean=2.0, std=0.4, coverage=0.0, edge_energy=0.01)
    problems = BM.gate(rec, base)
    assert any("impeller_bronze" in p and "mean" in p for p in problems)
    assert any("coverage" in p for p in problems)
    assert not any("heatsink_aluminium" in p for p in problems), "the other cases are untouched"


def test_the_gate_catches_a_picture_that_lost_its_micro_detail():
    base = BM.baseline_of({name: _case() for name in BM.CASES}, reason="reference")
    rec = {name: _case() for name in BM.CASES}
    rec["airliner_library"]["edge_energy"] = 1.16 * 0.5
    problems = BM.gate(rec, base)
    assert any("airliner_library" in p and "edge_energy" in p for p in problems)


def test_the_gate_says_so_when_a_case_is_missing_on_either_side():
    rec = {name: _case() for name in BM.CASES if name != "airliner_library"}
    full = BM.baseline_of({name: _case() for name in BM.CASES}, reason="reference")
    assert any("airliner_library: not rendered" in p for p in BM.gate(rec, full))
    assert any("no baseline" in p for p in BM.gate({name: _case() for name in BM.CASES}, {"impeller_bronze":
                                                                                          _case()}))


def test_the_time_is_not_gated():
    rec = {name: _case(seconds=900.0) for name in BM.CASES}
    assert BM.gate(rec, BM.baseline_of({name: _case() for name in BM.CASES}, reason="reference")) == [], \
        "a slow machine is not a regression in the picture"


def test_a_committed_baseline_keeps_the_statistics_and_drops_the_machine():
    b = BM.baseline_of({name: _case() for name in BM.CASES}, reason="shaders rewritten",
                       blender="Blender 5.2.1 LTS")
    assert set(b["impeller_bronze"]) == {"quality", "size", "mean", "std", "coverage", "edge_energy"}
    assert "seconds" not in b["impeller_bronze"] and "png" not in b["impeller_bronze"]
    meta = b["_meta"]
    assert meta["blender"] == "Blender 5.2.1 LTS" and meta["gate"] == BM.GATE
    assert BM.cases_of(b) == list(BM.CASES), "the bookkeeping keys are not cases"
    assert meta["recorded_at"].endswith("Z") and len(meta["recorded_at"]) == 20
    assert meta["reason"] == "shaders rewritten"


def test_a_baseline_without_a_reason_is_refused():
    with pytest.raises(ValueError, match="why"):
        BM.baseline_of({name: _case() for name in BM.CASES}, reason="   ")


def test_the_repository_ships_a_baseline_for_every_case_in_the_gate_configuration():
    path = os.path.join(BM.repository_root(), BM.BASELINE)
    assert os.path.isfile(path), f"{BM.BASELINE} is what the CI gate compares against"
    with open(path, encoding="utf-8") as fh:
        base = json.load(fh)
    assert set(BM.cases_of(base)) == set(BM.CASES), "every case, or the gate would have nothing to compare"
    for name in BM.cases_of(base):
        case = base[name]
        assert case["size"] == BM.GATE["size"], f"{name}: the baseline is not in the gate's size"
        assert case["quality"] == BM.GATE["quality"], f"{name}: the baseline is not in the gate's preset"
    assert "recorded_at" in base.get("_meta", {}), "a baseline with no date is a number nobody can judge"
    assert base["_meta"].get("reason", "").strip(), "a baseline with no reason is a decision nobody can audit"


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
