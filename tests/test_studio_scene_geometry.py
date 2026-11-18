"""The generic studio pieces a 3-D figure is built from: the annotation geometry, the colour maps, and the scene
round trip that carries material names through a ``.glb`` (the cache a headless render reads)."""
import numpy as np
import pytest

from pinneapple_tools.visualization.studio import (
    CMAPS,
    disc_surface,
    magma,
    rod_surface,
    sphere_surface,
    stops,
)
from pinneapple_tools.visualization.studio import scene as S


def _bbox(surf):
    V = np.asarray(surf.vertices, float)
    return V.min(axis=0), V.max(axis=0)


def test_sphere_surface_is_a_closed_shell_around_its_centre():
    s = sphere_surface((0.0, 0.0, 1.5), 0.25, name="feed", material="feed")
    assert s.name == "feed" and s.material == "feed"
    lo, hi = _bbox(s)
    assert np.allclose(lo, [-0.25, -0.25, 1.25], atol=1e-9)
    assert np.allclose(hi, [0.25, 0.25, 1.75], atol=1e-9)
    assert np.allclose(np.linalg.norm(np.asarray(s.vertices, float) - [0.0, 0.0, 1.5], axis=1), 0.25)


def test_rod_surface_spans_and_encloses_the_segment():
    s = rod_surface((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), 0.05, name="beam axis")
    lo, hi = _bbox(s)
    assert np.allclose(lo, [-0.05, -0.05, 0.0], atol=1e-9)
    assert np.allclose(hi, [0.05, 0.05, 2.0], atol=1e-9)
    # every vertex is one radius from the axis, but the two cap centres on it
    V = np.asarray(s.vertices, float)
    r_axis = np.hypot(V[:, 0], V[:, 1])
    caps = np.isclose(r_axis, 0.0, atol=1e-12)
    assert caps.sum() == 2 and np.allclose(V[caps, 2], [0.0, 2.0])
    assert np.allclose(r_axis[~caps], 0.05, atol=1e-9)
    # a degenerate rod (a == b) must not divide by zero
    flat = rod_surface((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 0.05)
    assert np.all(np.isfinite(np.asarray(flat.vertices, float)))


def test_disc_surface_lies_in_its_plane_and_takes_the_material_of_the_quiet_zone():
    s = disc_surface((0.0, 0.3, 3.0), 0.2, normal=(0.0, 0.0, 1.0), name="quiet zone", material="quiet_zone")
    assert s.material == "quiet_zone"
    V = np.asarray(s.vertices, float)
    assert np.allclose(V[:, 2], 3.0)                                     # flat in the plane it was given
    assert np.allclose(V[0], [0.0, 0.3, 3.0])                            # the fan centre, then the rim
    assert np.allclose(np.linalg.norm(V[1:] - V[0], axis=1), 0.2, atol=1e-9)


def test_magma_is_published_to_both_the_python_and_the_browser_families():
    table = CMAPS["magma"]
    assert table is stops("magma")
    assert sorted(CMAPS) == ["bwr", "coolwarm", "jet", "magma", "rdbu"]
    assert len(table) >= 9 and table[0][0] == pytest.approx(0.0) and table[-1][0] == pytest.approx(1.0)
    dark, bright = magma(0.0), magma(1.0)
    assert sum(dark) < 0.4 and sum(bright) > 2.4                # black bottom, near-white top, as matplotlib has it
    ramp = np.array([magma(t) for t in np.linspace(0.0, 1.0, 24)])
    assert np.all((ramp >= 0.0) & (ramp <= 1.0))


def test_a_scene_written_to_glb_reads_back_with_its_material_names(tmp_path):
    sc = S.Scene(surfaces=[sphere_surface((0.0, 0.0, 0.0), 0.1, name="feed", material="feed")],
                 materials={"feed": {"color": [0.10, 0.42, 0.85], "emissive": [0.20, 0.55, 1.0], "roughness": 0.3}})
    path = sc.save(str(tmp_path / "scene.glb"))
    back = S.Scene.from_file(path)
    assert [s.material for s in back.surfaces] == ["feed"]        # not renamed to "grey"
    # a name the reader has never seen renders grey rather than failing, and is reported as unknown
    assert back.unknown_materials() == ["feed"]
    back.materials = sc.materials
    assert back.unknown_materials() == []


def test_the_browser_has_the_same_magma_table_as_python():
    import json
    import re
    from pathlib import Path

    from pinneapple_tools.visualization.studio.colormap import MAGMA
    js = Path(__file__).resolve().parents[1] / "pinneapple_tools/visualization/studio/web/studio-core.js"
    m = re.search(r"export const MAGMA = (\[\[.*?\]\]);", js.read_text(), re.S)
    assert m, "MAGMA table not found in studio-core.js"
    assert [(float(t), tuple(c)) for t, c in json.loads(m.group(1))] == list(MAGMA)
