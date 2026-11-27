"""The campaign's screen and its progress files: what a run writes while nobody is watching it.

Three claims are worth testing and everything here is one of them:

* a fit and a dataset report **while they run** — the progress file exists mid-stage, not only at the end, and it
  says how far the stage got and how long it still needs (a screen reading only ``card.json`` shows a family as
  pending for the four hours it is fitting);
* the panel reads **files only**, so what it shows is what a run that died left behind — a family with a manifest
  and no card is "fitting", one with a card is done with the number it measured, and a progress file older than
  three minutes on a running stage is flagged stale instead of drawn as live;
* one campaign per output directory: a second process pointed at the same ``--out`` refuses to start.
"""
from __future__ import annotations

import json
import os
import urllib.request

import pytest

from benchmarks import lpm_campaign, lpm_monitor
from pinneapple_neural.lpm import LargePhysicsModel, fit
from pinneapple_neural.lpm_families import NOZZLE
from pinneapple_neural.lpm_pipeline import (
    PROGRESS_SCHEMA,
    build,
    draw,
    generate_dataset,
    train,
)


def _designs(n=3):
    return [build(NOZZLE, spec) for spec in draw(NOZZLE, n, 1, seed=0)]


def _model():
    return LargePhysicsModel(dim=NOZZLE.dim, n_fields=NOZZLE.n_fields,
                             n_params=len(NOZZLE.operating_params) + len(NOZZLE.geometry_features),
                             radii=(0.2, 0.8), width=24, n_heads=2)


# ----------------------------------------------------------------------------------------------- the progress hook


def test_fit_reports_every_epoch_and_the_last_loss_is_the_history():
    seen = []
    history = fit(_model(), _designs(), epochs=6, on_epoch=lambda e, loss: seen.append((e, loss)))
    assert [e for e, _ in seen] == [1, 2, 3, 4, 5, 6]          # counted from 1: an epoch you finished, not an index
    assert seen[-1][1] == pytest.approx(history[-1])
    assert history[0] == pytest.approx(seen[0][1])


def test_fit_without_the_hook_runs_the_same():
    """The default costs a comparison per epoch and changes nothing about what is fitted."""
    designs = _designs()
    m1, m2 = _model(), _model()
    m2.load_state_dict(m1.state_dict())                     # same weights before the step
    a = fit(m1, designs, epochs=3, seed=0)
    b = fit(m2, designs, epochs=3, seed=0, on_epoch=lambda e, loss: None)
    assert a == pytest.approx(b)


def test_the_dataset_stage_reports_what_it_has_written(tmp_path):
    manifest = generate_dataset(NOZZLE, tmp_path, n_geometries=3, conditions_per_geometry=2, progress_every_s=1e-9)
    record = json.loads((tmp_path / "progress.json").read_text())
    assert record["schema"] == PROGRESS_SCHEMA
    assert record["kind"] == "dataset"
    assert record["family"] == "pp_nozzle"
    assert record["done"] == record["total"] == manifest["n_designs"] == 6
    assert record["finished"] is True
    assert record["per_s"] > 0 and record["eta_s"] == 0.0


def test_the_fit_reports_epochs_and_the_time_still_needed(tmp_path):
    generate_dataset(NOZZLE, tmp_path / "dataset", n_geometries=3, conditions_per_geometry=1)
    train(tmp_path / "dataset", tmp_path, epochs=20, width=32, progress_every_s=1e-9)
    record = json.loads((tmp_path / "model" / "progress.json").read_text())
    assert record["kind"] == "fit"
    assert record["done"] == record["total"] == 20
    assert record["finished"] is True
    assert record["loss"] <= record["loss_first"]           # on this family and seed it comes down, and the record says so
    assert record["n_train_designs"] + record["n_holdout_designs"] == 3


def test_no_progress_file_is_written_unless_asked(tmp_path):
    """A pipeline that writes files nobody asked for is a pipeline whose output directory is a mystery."""
    generate_dataset(NOZZLE, tmp_path, n_geometries=2, conditions_per_geometry=1)
    assert not (tmp_path / "progress.json").exists()


# --------------------------------------------------------------------------------------------------- the panel's state


@pytest.fixture()
def ran(tmp_path):
    """One family run end to end, with progress on: the shape the panel actually reads."""
    out = tmp_path / "campaign"
    generate_dataset(NOZZLE, out / "pp_nozzle" / "dataset", n_geometries=3, conditions_per_geometry=1,
                     progress_every_s=1e-9)
    train(out / "pp_nozzle" / "dataset", out / "pp_nozzle", epochs=5, width=32, progress_every_s=1e-9)
    return out


def test_a_finished_family_is_done_with_the_number_it_measured(ran):
    row = lpm_monitor.family_state("pp_nozzle", ran)
    assert row["state"] == "done" and row["stage"] == "card"
    assert all(row["stages"].values())                      # draw, build, write, fit, card
    assert row["epochs"] == {"done": 5, "total": 5}
    assert row["held_out"] is not None and row["baseline"] is not None
    assert row["title"] and row["reference"]                # what it is, on the same row as how it went


def test_a_manifest_without_a_card_is_a_family_that_is_fitting(tmp_path):
    generate_dataset(NOZZLE, tmp_path / "pp_nozzle" / "dataset", n_geometries=2, conditions_per_geometry=1)
    row = lpm_monitor.family_state("pp_nozzle", tmp_path)
    assert row["stage"] == "fit" and row["state"] == "starting"
    assert row["stages"]["write"] is True and row["stages"]["fit"] is False


def test_a_progress_file_on_a_running_stage_is_live_and_an_old_one_is_stale(tmp_path):
    model = tmp_path / "pp_nozzle" / "model"
    model.mkdir(parents=True)
    (model / "progress.json").write_text(json.dumps({
        "schema": PROGRESS_SCHEMA, "kind": "fit", "family": "pp_nozzle", "done": 40, "total": 100,
        "finished": False, "per_s": 0.5, "eta_s": 120.0, "elapsed_s": 80.0, "updated_at": "2026-01-01T00:00:00+00:00"}))
    row = lpm_monitor.family_state("pp_nozzle", tmp_path)
    assert row["stage"] == "fit" and row["state"] == "running"
    assert row["progress"] == pytest.approx(0.4) and row["eta_s"] == 120.0
    assert row["stale"] is False                            # just written
    old = 1_000_000
    os.utime(model / "progress.json", (1, old))
    assert lpm_monitor.family_state("pp_nozzle", tmp_path)["stale"] is True


def test_the_runner_says_which_family_is_in_flight(ran):
    reporter = lpm_campaign.Reporter(ran, scale="full", stage="both", seed=0, order=["pp_nozzle"],
                                     estimate_s=10.0, max_hours=None, threads=4)
    assert json.loads((ran / "runner.json").read_text())["current"] is None
    reporter.on("pp_nozzle", "fit")
    s = lpm_monitor.state(ran)
    assert s["campaign"]["state"] == "running"              # this process is alive; the recorded pid is ours
    assert s["campaign"]["current"] == "pp_nozzle"
    assert s["campaign"]["current_stage"] == "fit"
    assert s["campaign"]["one_at_a_time"] is True
    assert s["campaign"]["seed"] == 0                       # a 0 that is a value, not a missing key
    reporter.finished("pp_nozzle", {"family": "pp_nozzle"})
    assert lpm_monitor.state(ran)["campaign"]["current"] is None


def test_a_campaign_that_finished_says_so(tmp_path):
    (tmp_path / "campaign.json").write_text(json.dumps({
        "scale": "smoke", "stage": "both", "seed": 0, "started_at": "2026-01-01T00:00:00+00:00",
        "seconds": 12.0, "order": ["pp_nozzle"], "estimates": {"pp_nozzle": 9.0}, "stopped": None,
        "families": [], "failures": []}))
    s = lpm_monitor.state(tmp_path)
    assert s["campaign"]["state"] == "finished"
    assert s["campaign"]["estimate_s"] == pytest.approx(9.0)
    assert s["families"][0]["name"] == "pp_nozzle"


def test_a_failure_reaches_the_screen(tmp_path):
    (tmp_path / "campaign.json").write_text(json.dumps({"order": ["pp_nozzle"], "families": [], "failures": ["pp_nozzle"]}))
    (tmp_path / "failures.json").write_text(json.dumps({"pp_nozzle": {"error": "RuntimeError: boom",
                                                                      "traceback": "..."}}))
    s = lpm_monitor.state(tmp_path)
    assert s["campaign"]["failures"] == ["pp_nozzle"]
    assert "boom" in s["failures"]["pp_nozzle"]


# --------------------------------------------------------------------------------------------------- the server


def test_the_panel_serves_the_page_and_its_state(ran):
    srv = lpm_monitor.serve(ran, port=0, block=False)
    host, port = srv.server_address[0], srv.server_address[1]
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/") as r:
            body = r.read().decode()
        assert r.status == 200 and "LPM" in body and "state.json" in body
        with urllib.request.urlopen(f"http://{host}:{port}/state.json") as r:
            payload = json.loads(r.read())
        row = next(f for f in payload["families"] if f["name"] == "pp_nozzle")
        assert row["state"] == "done"                        # with no runner.json, every family is listed
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"http://{host}:{port}/nope")
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------------------- one at a time, one machine


def test_one_campaign_per_output_directory(tmp_path):
    lock = lpm_campaign._acquire_lock(tmp_path)
    try:
        with pytest.raises(SystemExit, match="another campaign"):
            lpm_campaign._acquire_lock(tmp_path)            # this process is alive, so the lock is real
    finally:
        lock.unlink()
    lock = lpm_campaign._acquire_lock(tmp_path)              # and free once the run lets it go
    assert lock.is_file()


def test_a_lock_left_by_a_dead_process_does_not_block_the_next_run(tmp_path):
    dead = 999_999
    (tmp_path / ".campaign.lock").write_text(f"{dead}\n")
    lock = lpm_campaign._acquire_lock(tmp_path)
    assert lock.read_text().strip() == str(os.getpid())


def test_the_thread_cap_is_a_cap(monkeypatch):
    assert lpm_campaign.limit_threads(0) == 0              # 0 means "leave the pools alone"
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert lpm_campaign.limit_threads(2) == 2
    assert os.environ["OMP_NUM_THREADS"] == "2"
    import torch

    assert torch.get_num_threads() == 2
