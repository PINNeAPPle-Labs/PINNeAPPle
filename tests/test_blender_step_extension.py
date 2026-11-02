"""The PINNeAPPle STEP Blender extension (#1119): the parts that need no Blender run everywhere; the operators run in a
Python with ``bpy`` (this one, or the one named by ``PINNEAPPLE_BPY_PYTHON``) and are skipped otherwise."""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pinneapple_blender.step_extension import build as B
from pinneapple_blender.step_extension import core

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "pinneapple_blender" / "step_extension"


def test_manifest_follows_the_extension_format():
    tomllib = pytest.importorskip("tomllib")                # Python 3.11+, as Blender's
    m = tomllib.loads((EXT / "blender_manifest.toml").read_text())
    for key in ("schema_version", "id", "version", "name", "tagline", "maintainer", "type", "blender_version_min",
                "license"):
        assert key in m, key
    assert m["type"] == "add-on" and m["id"] == "pinneapple_step" and len(m["tagline"]) <= 64
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert m["version"] == pyproject["project"]["version"]
    assert m["license"] == ["SPDX:Apache-2.0"]


def test_import_without_blender_and_no_bpy_at_module_level():
    import pinneapple_blender.step_extension as ext
    assert callable(ext.register) and callable(ext.unregister)
    assert "import bpy" not in (EXT / "core.py").read_text() and "import bpy" not in (EXT / "__init__.py").read_text()


def test_colours():
    rgb = core.colours(np.array([-2.0, 0.0, 2.0]), "diverging", clip_percentile=100)
    assert rgb.shape == (3, 3) and rgb[0, 2] > rgb[0, 0] and rgb[2, 0] > rgb[2, 2]       # blue < 0 < red
    assert np.allclose(rgb[1], 0.865)
    g = core.colours(np.array([0.0, 0.5, 1.0]), "gray")
    assert np.allclose(g[:, 0], [0, 0.5, 1]) and np.allclose(g[:, 0], g[:, 1])
    v = core.colours(np.linspace(0, 1, 11), "sequential", clip_percentile=100)
    assert np.all((v >= 0) & (v <= 1)) and np.allclose(v[0], core._VIRIDIS[0]) and np.allclose(v[-1], core._VIRIDIS[-1])


def test_diagnostic_and_meshdata_on_a_sphere(tmp_path):
    sm = pytest.importorskip("smlibpy")
    pytest.importorskip("mapbox_earcut")
    from pinneapple_design.geometry.brep.step import _body
    from pinneapple_design.geometry.core.mesh import MeshData

    b = _body(sm.Sphere(10), 0, "smlibpy")
    m = core.brep().tessellate(b, "normal")
    v, rgb, summary = core.diagnostic(b, m, "MEAN")
    assert np.allclose(v, 100.0) and rgb.shape == (len(m.vertices), 3) and summary["n_vertices"] == len(v)
    with pytest.raises(ValueError, match="unknown diagnostic"):
        core.diagnostic(b, m, "NOPE")
    info = core.write_meshdata(m, str(tmp_path / "ball.npz"), "ball")
    d = np.load(info["path"])
    md = MeshData(**{k: d[k] for k in ("vertices", "faces", "normals")})
    assert md.n_faces == m.n_triangles and info["watertight"] and (tmp_path / "ball.json").exists()


def test_build_puts_the_library_copy_and_the_manifest_in_the_zip(tmp_path):
    z = B.build(tmp_path)
    names = set(zipfile.ZipFile(z).namelist())
    assert {"__init__.py", "addon.py", "core.py", "blender_manifest.toml", "vendor/__init__.py"} <= names
    brep = ROOT / "pinneapple_design" / "geometry" / "brep"
    for f in brep.glob("*.py"):
        assert zipfile.ZipFile(z).read(f"vendor/brep/{f.name}") == f.read_bytes()
    assert not any("__pycache__" in n or "build.py" in n for n in names)


def _bpy_python():
    try:
        import bpy  # noqa: F401
        return sys.executable
    except ImportError:
        return os.environ.get("PINNEAPPLE_BPY_PYTHON")


@pytest.mark.skipif(not _bpy_python(), reason="no Python with bpy (set PINNEAPPLE_BPY_PYTHON)")
def test_every_operator_in_blender(tmp_path):
    """Install the built zip into a throw-away Blender user folder and run every operator
    (tests/blender/step_extension_smoke.py holds the assertions)."""
    z = B.build(tmp_path / "dist")
    env = {**os.environ, "HOME": str(tmp_path / "home")}
    env.pop("PYTHONPATH", None)                       # the extension must work from its own copy of the library
    r = subprocess.run([_bpy_python(), str(ROOT / "tests" / "blender" / "step_extension_smoke.py"), str(z),
                        str(tmp_path / "work")], capture_output=True, text=True, env=env, timeout=1200,
                       cwd=str(tmp_path))
    assert "PNP_SMOKE" in r.stdout and '"ok": true' in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    assert "vendor.brep" in r.stdout
