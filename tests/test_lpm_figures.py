"""The figures a fitted design space shows: drawn from ``view.json``, by the repository's own plotting.

What these check is that the picture is *of the run that exists*: the view is written by the fit, every sample names
the design it came from, a family without a view says so instead of drawing an empty chart, and the figure is rebuilt
when the view changes rather than served from an older fit.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from benchmarks import lpm_figures, lpm_monitor
from pinneapple_neural.lpm_families import NOZZLE
from pinneapple_neural.lpm_pipeline import VIEW_SCHEMA, generate_dataset, train


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One family fitted end to end: a view, a card and a dataset in the campaign's own layout."""
    out = tmp_path_factory.mktemp("campaign")
    family = out / "pp_nozzle"
    generate_dataset(NOZZLE, family / "dataset", n_geometries=4, conditions_per_geometry=2)
    train(family / "dataset", family, epochs=10, width=24)
    return out


def test_the_fit_writes_a_view_of_what_it_did(run):
    view = json.loads((run / "pp_nozzle" / "model" / "view.json").read_text())
    assert view["schema"] == VIEW_SCHEMA
    assert view["family"] == "pp_nozzle"
    assert len(view["loss"]["epoch"]) == len(view["loss"]["loss"]) == 10
    assert view["loss"]["loss"][-1] < view["loss"]["loss"][0]          # it fitted something
    assert view["split"]["n_train"] + view["split"]["n_held_out"] == view["n_designs"]


def test_every_sample_names_the_design_it_came_from(run):
    view = json.loads((run / "pp_nozzle" / "model" / "view.json").read_text())
    assert view["samples"], "a fitted family shows at least one body"
    for sample in view["samples"]:
        assert sample["design"].startswith("pp_nozzle:")
        assert set(sample["target"]) == set(sample["pred"]) == set(view["fields"])
        assert len(sample["target"][view["fields"][0]]) == len(sample["pred"][view["fields"][0]])
        assert sample["points"], "the geometry is in the view, not fetched from the dataset at draw time"
        if sample["held_out"]:
            assert sample["relative_l2"] is not None
        else:
            assert sample["label"] == "training"


def test_the_design_space_axes_are_the_ones_that_vary(run):
    view = json.loads((run / "pp_nozzle" / "model" / "view.json").read_text())
    assert len(view["axes"]) == 2 and set(view["axes"]) <= set(view["params"])
    spans = []
    for axis in view["axes"]:
        vals = [d["values"][axis] for d in view["designs"]]
        spans.append(max(vals) - min(vals))
    assert all(span > 0 for span in spans)


def test_each_figure_is_drawn_and_then_cached(run):
    path = lpm_figures.ensure(run, "pp_nozzle", "loss")
    assert path.is_file() and path.suffix == ".png" and path.stat().st_size > 1000
    before = path.stat().st_mtime
    assert lpm_figures.ensure(run, "pp_nozzle", "loss") == path
    assert path.stat().st_mtime == before                          # the cache answered, nothing was redrawn
    os.utime(path, (1, 1))                                         # an older figure than the view gets rebuilt
    lpm_figures.ensure(run, "pp_nozzle", "loss")
    assert path.stat().st_mtime > 1


def test_every_figure_builds(run):
    built = lpm_figures.build_all(run, "pp_nozzle")
    assert set(built) == set(lpm_figures.KINDS)
    for kind, path in built.items():
        assert os.path.getsize(path) > 1000, kind


def test_the_viewer_is_the_studios_own_page(run):
    path = lpm_figures.ensure_viewer(run, "pp_nozzle")
    assert path.suffix == ".html" and path.stat().st_size > 100_000        # three.js is inlined, so it opens offline
    page = path.read_text(encoding="utf-8")
    assert 'import "pp/viewer"' in page                                  # the viewer is the studio's own module
    assert "base64," in page                                             # and the scene travels inside the page


def test_a_family_without_a_fit_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no view.json"):
        lpm_figures.ensure(tmp_path, "pp_fin", "loss")
    with pytest.raises(KeyError, match="unknown figure"):
        lpm_figures.ensure(tmp_path, "pp_fin", "nonsense")


def test_the_scene_lays_the_samples_side_by_side(run):
    """No two bodies share space: each gets its own shelf. The shelf runs along whichever axis the row took — a 2-D
    family is turned a quarter so its row crosses the face-on view, and a 3-D one keeps the widest axis."""
    view = json.loads((run / "pp_nozzle" / "model" / "view.json").read_text())
    scene = lpm_figures.scene(view)
    assert len(scene.surfaces) == len(view["samples"])
    for first, second in zip(scene.surfaces, scene.surfaces[1:], strict=False):
        a, b = first.vertices, second.vertices
        overlaps = all(a[:, k].min() <= b[:, k].max() and b[:, k].min() <= a[:, k].max() for k in range(3))
        assert not overlaps, "two bodies overlap: they were not given their own shelf"


def test_a_flat_scene_opens_face_on_and_faces_the_camera(run):
    """pp_airfoil and pp_exchanger came out blank: a planar sheet drawn with the default 3/4 view is edge-on, and a
    sheet whose faces point the other way is culled outright. Flat scenes open on the face-on preset and every
    triangle faces +z, which is the side all four presets look from."""
    import numpy as np

    view = json.loads((run / "pp_nozzle" / "model" / "view.json").read_text())
    scene = lpm_figures.scene(view)
    assert np.allclose([np.ptp(s.vertices[:, 2]) for s in scene.surfaces], 0.0), "the fixture family is planar"
    assert scene.view == "top", "a flat family opens face-on"
    for surface in scene.surfaces:
        v, f = surface.vertices, surface.faces
        normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        assert (normals[:, 2] >= -1e-12).all(), "a triangle facing away from the camera would not be drawn"


def test_a_deep_scene_keeps_the_three_quarter_view(run):
    """A 3-D body is turned off in the default view for a reason, and the fix for the flat ones must not touch it."""
    import numpy as np

    view = json.loads((run / "pp_nozzle" / "model" / "view.json").read_text())
    def _deep(s, i):
        pts = np.asarray(s["points"], dtype=float)
        pts = np.hstack([pts, np.zeros((len(pts), 1))]) if pts.shape[1] == 2 else pts
        return dict(s, points=(pts + np.array([0.0, 0.0, 1.0]) * i).tolist())

    view = dict(view, samples=[_deep(s, i) for i, s in enumerate(view["samples"])])
    assert lpm_figures.scene(view).view == "iso"


def test_a_view_without_at_points_still_colours_its_skin(run):
    """The campaign running when the flag was introduced wrote views without it; those must not come out grey."""
    import copy

    view = json.loads((run / "pp_nozzle" / "model" / "view.json").read_text())
    old = copy.deepcopy(view)
    for sample in old["samples"]:
        sample.pop("at_points", None)
    assert lpm_figures.scene(old).surfaces[0].fields, "the field is recovered from points == query"


def test_the_panel_serves_a_figure_and_refuses_one_it_has_not_got(run):
    srv = lpm_monitor.serve(run, port=0, block=False)
    host, port = srv.server_address[0], srv.server_address[1]
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/figure/pp_nozzle/prediction.png") as r:
            assert r.status == 200 and r.headers["Content-Type"] == "image/png"
            assert r.read().startswith(b"\x89PNG")
        with urllib.request.urlopen(f"http://{host}:{port}/figure/pp_nozzle/viewer.html") as r:
            assert r.status == 200 and r.headers["Content-Type"].startswith("text/html")
            assert b"<html" in r.read(2000).lower() or b"<!doctype" in r.read(2000).lower()
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"http://{host}:{port}/figure/pp_fin/loss.png")
        assert missing.value.code == 404
    finally:
        srv.shutdown()
