"""The Unity package's C# data layer, compiled and run outside Unity against a twin this repository exported.

`pinneapple_unity/unity/Assets/PinNeAPPleTwin/Runtime/{PinNeappleJson,PinNeappleGlb,PinNeappleTwinData,PinNeappleColormap}.cs`
have no UnityEngine dependency on purpose: they are the part of the package that can be proven without the engine.
The harness in `pinneapple_unity/unity/tests` reads a folder written by `export_for_unity`, verifies it, and
round-trips the geometry through the writer the editor menu uses.

Skipped when `dotnet` is not installed. Build artifacts go to a temporary folder, never into the repository.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from pinneapple_twin3d import Scene
from pinneapple_twin3d.primitives import sweep
from pinneapple_unity import emulate_unity_export, export_for_unity

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "pinneapple_unity", "unity", "tests")
DOTNET = shutil.which("dotnet")


def _twin(tmp_path):
    section = [(-0.04, -0.003), (0.04, -0.003), (0.04, 0.003), (-0.04, 0.003)]
    s = np.linspace(0.0, 0.2, 21)
    path = np.stack([s, np.zeros(21), np.zeros(21)], axis=1)
    v, f = sweep(section, path)
    times = [0.0, 60.0, 300.0, 900.0]
    sc = Scene("C# harness plate", length_unit="m", times=times, time_unit="s")
    sc.add_part("plate", v, f, color=(0.7, 0.75, 0.8))
    sc.add_field("plate", "temperature", np.linspace(20.0, 120.0, len(v))[None, :] * np.ones((len(times), 1)),
                 unit="C")
    sc.add_sensor("TC-01", (0.1, 0.0, 0.003), unit="C", series=[20.0, 25.0, 40.0, 60.0], envelope=(0.0, 150.0))
    twin = tmp_path / "twin"
    assert export_for_unity(sc, twin).problems == []
    unity = tmp_path / "unity"
    emulate_unity_export(sc, unity, jobs=[{"job": "slab_transient", "part": "plate",
                                           "params": {"alpha": 1e-5, "t_left": 120.0, "t_right": 20.0, "t0": 20.0}}])
    return twin, unity


@pytest.mark.skipif(DOTNET is None, reason="dotnet is not installed")
def test_the_csharp_data_layer_reads_what_python_exported(tmp_path):
    twin, unity = _twin(tmp_path)
    build = tmp_path / "dotnet"
    cmd = [DOTNET, "run", "--project", PROJECT,
           f"-p:BaseIntermediateOutputPath={build}{os.sep}obj{os.sep}",
           f"-p:BaseOutputPath={build}{os.sep}bin{os.sep}",
           "--", str(twin), str(unity)]
    env = {**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1"}
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env, cwd=PROJECT)
    out = done.stdout + done.stderr
    assert done.returncode == 0, out[-4000:]
    assert "OK: the data layer reads the twin" in out, out[-4000:]
    assert "FAIL " not in out, out[-4000:]
    assert "field plate/temperature [C] sampled in" in out
    assert "jobs in the link: 1" in out
    # nothing was written into the repository
    assert not os.path.exists(os.path.join(PROJECT, "bin"))
    assert not os.path.exists(os.path.join(PROJECT, "obj"))
