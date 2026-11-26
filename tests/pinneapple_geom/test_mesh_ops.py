import pytest

# A bare `import trimesh` here previously made the WHOLE test suite fail
# to collect (pytest aborts collection entirely on any module-level
# ImportError) for anyone without the optional `geom` extra installed --
# found via `pytest tests/` erroring out before running a single test.
# `importorskip` degrades this one file to a skip instead.
trimesh = pytest.importorskip("trimesh")
# `pinneapple_geom` was renamed to `pinneapple_design.geometry` in an
# earlier refactor; this test's imports were never updated, and the
# module-level ImportError this caused was masked for anyone without
# trimesh installed (the importorskip above short-circuited collection
# before reaching these lines) -- surfaced once trimesh was installed
# for real CAD-generation testing elsewhere in this session. The actual
# TrimeshBridge/repair_mesh/simplify_mesh APIs are otherwise unchanged
# (verified: same method/kwarg names), so this is a plain import-path
# fix, not a behavior change.
from pinneapple_design.geometry.io.trimesh_bridge import TrimeshBridge
from pinneapple_design.geometry.ops.repair import repair_mesh
from pinneapple_design.geometry.ops.simplify import simplify_mesh

def _icosphere():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    return mesh, TrimeshBridge().from_trimesh(mesh)


def test_repair_and_simplify_smoke():
    """The smoke test walks the path that always exists.

    ``fast-simplification`` and ``open3d`` live in the ``geom_simplify`` extra, which the blocking
    shards do not install, so with the default ``on_missing="error"`` this file failed on CI for a
    reason that had nothing to do with the mesh. ``on_missing="return_input"`` is the documented
    contract for "there is no decimator here", so the smoke test asks for it and still proves that
    the bridge and the repair work.
    """
    mesh, g = _icosphere()
    g2 = repair_mesh(g)
    g3 = simplify_mesh(g2, target_faces=max(20, len(mesh.faces) // 2), on_missing="return_input")
    assert g3 is not None
    assert len(g3.vertices) > 0 and len(g3.faces) > 0


def test_without_a_decimator_the_default_names_the_package_that_fixes_it():
    """The refusal is part of the contract: it says which two packages solve it, so nobody is left
    debugging a mesh that was never the problem."""
    import importlib.util

    if importlib.util.find_spec("fast_simplification") or importlib.util.find_spec("open3d"):
        pytest.skip("a decimator is installed here, so the missing-backend path cannot happen")
    _, g = _icosphere()
    with pytest.raises(RuntimeError) as caught:
        simplify_mesh(g, target_faces=20)
    msg = str(caught.value)
    assert "no available backend" in msg
    assert "fast_simplification" in msg and "open3d" in msg


def test_simplification_really_reduces_faces_when_a_decimator_is_installed():
    pytest.importorskip("fast_simplification")
    mesh, g = _icosphere()
    target = max(20, len(mesh.faces) // 2)
    g3 = simplify_mesh(g, target_faces=target)
    assert len(g3.faces) > 0
    assert len(g3.faces) < len(mesh.faces)
