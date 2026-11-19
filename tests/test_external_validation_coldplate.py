"""Regression tests: falsification of the cold-plate physics and applicability gates."""
from __future__ import annotations

import json
from pathlib import Path

from examples.fno_coldplate.external_validation import run_controls


def test_external_validation_controls(tmp_path: Path):
    report = run_controls(tmp_path, n=32, source_levels=(0.5, 1.0, 2.0))
    assert report["status"] == "PASS"
    assert len(report["cases"]) == 3
    assert (tmp_path / "controls.json").is_file()
    assert len(report["provenance"]["artifact_sha256"]) == 9
    assert json.loads((tmp_path / "controls.json").read_text())["status"] == "PASS"
    for case in report["cases"]:
        assert case["clean_fv_reference"]["verdict"] == "PASS"
        assert case["corrupted_half_amplitude"]["verdict"] == "FAIL"
        assert case["ood_fv_reference"]["verdict"] == "FAIL"
        assert case["ood_physics_only"]["verdict"] == "PASS"
        assert "a_max" in case["ood_fv_reference"]["failed"]
        assert abs(case["corruption_relative_l2"] - 0.5) < 1e-12
