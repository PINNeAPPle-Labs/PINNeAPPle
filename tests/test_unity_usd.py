"""The native OpenUSD lane: what the twin writes is a stage Unity's USD Toolkit (and Isaac Sim) opens.

The text reader (``stage_summary``) runs everywhere; the checks that need Pixar's OpenUSD (``validate``,
``from_usd``) skip when ``usd-core`` is not installed -- ``pip install usd-core`` makes them run.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import pinneapple_unity as pu
from pinneapple_twin3d import Scene
from pinneapple_twin3d.primitives import sweep
from pinneapple_twin3d.scene import PhysicsMaterial


def _has_pxr() -> bool:
    """Pixar's OpenUSD (``pip install usd-core``) is what the Unity USD Toolkit wraps."""
    try:
        import pxr  # noqa: F401
        return True
    except ImportError:
        return False


TIMES = [0.0, 60.0, 300.0, 900.0, 1800.0, 3600.0]


def _stage(tmp_path):
    """A twin with one field, one sensor and UsdPhysics, exported with usd=True."""
    section = [(-0.04, -0.003), (0.04, -0.003), (0.04, 0.003), (-0.04, 0.003)]
    s = np.linspace(0.0, 0.2, 21)
    path = np.stack([s, np.zeros(21), np.zeros(21)], axis=1)
    v, f = sweep(section, path)
    scene = Scene("USD test plate", length_unit="m", times=TIMES, time_unit="s", source="tests/test_unity_usd.py")
    scene.add_part("plate", v, f, color=(0.7, 0.75, 0.8))
    ramp = np.linspace(20.0, 120.0, len(v))
    scene.add_field("plate", "temperature", ramp[None, :] * np.ones((len(TIMES), 1)), unit="C")
    scene.add_sensor("TC-01", (0.1, 0.0, 0.003), unit="C", series=[20.0, 25.0, 40.0, 60.0, 70.0, 72.0])
    scene.set_gravity((0.0, 0.0, -9.81))
    scene.add_physics("plate", rigid_body=False, collider=True,
                      material=PhysicsMaterial("steel", static_friction=0.4))
    pu.export_for_unity(scene, tmp_path / "twin", usd=True)
    return str(tmp_path / "twin" / "Assets/TwinData/scene.usda")


def test_stage_summary_reports_what_unity_will_import(tmp_path):
    path = _stage(tmp_path)
    s = pu.usd.stage_summary(path)
    assert s["format"].startswith("#usda") and s["up_axis"] == "Z" and s["meters_per_unit"] == 1.0
    assert s["prims"]["Mesh"] == 1 and s["prims"]["PhysicsScene"] == 1
    assert s["api_schemas"]["PhysicsCollisionAPI"] == 1 and s["api_schemas"]["PhysicsMaterialAPI"] == 1
    assert "primvars:temperature" in s["time_sampled"]
    units = {p["name"]: p["unit"] for p in s["primvars"]}
    assert units["primvars:temperature"] == "C"
    assert s["metadata"]["pinneapple_format"] == "pinneapple-twin3d/1"
    assert s["metadata"]["length_unit"] == "m" and s["metadata"]["time_unit"] == "s"
    assert s["end_time_code"] == len(TIMES) - 1
    assert s["problems"] == []


def test_stage_summary_flags_a_stage_an_importer_cannot_use(tmp_path):
    path = tmp_path / "empty.usda"
    path.write_text('#usda 1.0\n(\n    upAxis = "Z"\n)\n\ndef Xform "Twin"\n{\n}\n')
    problems = pu.usd.stage_summary(str(path))["problems"]
    assert any("no Mesh prim" in p for p in problems), problems


@pytest.mark.skipif(not _has_pxr(), reason="usd-core is not installed (pip install usd-core)")
def test_openusd_opens_the_stage_without_complaint(tmp_path):
    path = _stage(tmp_path)
    assert pu.usd.validate(path) == []


@pytest.mark.skipif(not _has_pxr(), reason="usd-core is not installed (pip install usd-core)")
def test_from_usd_reads_the_stage_back_into_a_scene(tmp_path):
    path = _stage(tmp_path)
    scene = pu.usd.from_usd(path)
    assert [p.name for p in scene.parts] == ["plate"]
    assert len(scene.times) == len(TIMES)
    assert scene.field_units["temperature"] == "C"
    values = scene.parts[0].fields["temperature"]
    assert values.shape == (len(TIMES), len(scene.parts[0].vertices))
    assert float(values[0].max()) == pytest.approx(120.0, rel=1e-5)
    assert scene.gravity is not None and abs(abs(scene.gravity[2]) - 9.81) < 1e-3
    assert scene.parts[0].physics is not None and scene.parts[0].physics.collider
    assert scene.parts[0].physics.rigid_body is False


def test_the_toolkit_line_is_written_into_a_unity_project(tmp_path):
    (tmp_path / "Packages").mkdir()
    manifest = tmp_path / "Packages" / "manifest.json"
    manifest.write_text(json.dumps({"dependencies": {"com.unity.modules.ui": "1.0.0"}}, indent=2))
    pu.usd.write_manifest(str(tmp_path))
    pu.usd.write_manifest(str(tmp_path))                      # idempotent
    deps = json.loads(manifest.read_text())["dependencies"]
    assert deps["com.unity.usd-toolkit"].startswith("https://github.com/Unity-Technologies/com.unity.usd-toolkit.git#")
    assert deps["com.unity.modules.ui"] == "1.0.0"            # the project's own entries survive
    entry = pu.usd.manifest_entry()
    assert list(entry) == ["com.unity.usd-toolkit"]

