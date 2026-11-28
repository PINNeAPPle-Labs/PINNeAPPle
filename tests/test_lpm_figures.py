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


def test_all_four_figures_build(run):
    built = lpm_figures.build_all(run, "pp_nozzle")
    assert set(built) == set(lpm_figures.KINDS)
    for kind, path in built.items():
        assert os.path.getsize(path) > 1000, kind


def test_a_family_without_a_fit_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no view.json"):
        lpm_figures.ensure(tmp_path, "pp_fin", "loss")
    with pytest.raises(KeyError, match="unknown figure"):
        lpm_figures.ensure(tmp_path, "pp_fin", "nonsense")


def test_the_panel_serves_a_figure_and_refuses_one_it_has_not_got(run):
    srv = lpm_monitor.serve(run, port=0, block=False)
    host, port = srv.server_address[0], srv.server_address[1]
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/figure/pp_nozzle/prediction.png") as r:
            assert r.status == 200 and r.headers["Content-Type"] == "image/png"
            assert r.read().startswith(b"\x89PNG")
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"http://{host}:{port}/figure/pp_fin/loss.png")
        assert missing.value.code == 404
    finally:
        srv.shutdown()
