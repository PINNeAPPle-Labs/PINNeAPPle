"""run_results evidence integrity contracts; not executed."""
import json

import pytest

from pinneapple_lab.experiments.evidence_manifest import (
    build_manifest, verify_manifest, write_manifest,
)


def test_detects_changed_output(tmp_path):
    (tmp_path / "stdout.txt").write_text("first", encoding="utf-8")
    manifest = write_manifest(tmp_path)
    assert verify_manifest(tmp_path, manifest)["valid"]
    (tmp_path / "stdout.txt").write_text("changed", encoding="utf-8")
    assert not verify_manifest(tmp_path, manifest)["valid"]


def test_manifest_is_deterministic(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"loss": 0.1}))
    first = build_manifest(tmp_path)
    second = build_manifest(tmp_path)
    assert first == second
    assert first["scientific_status"] == "integrity_only_not_physics_validated"


def test_symlinks_are_rejected(tmp_path):
    (tmp_path / "result").symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError):
        build_manifest(tmp_path)
