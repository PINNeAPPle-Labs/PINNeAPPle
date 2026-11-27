"""Heated-channel dataset recipe (thermal_channel_datasets.py): actuation families, split plan, file layout."""
import json

import numpy as np
import pytest

from pinneapple_simulation.numerical_solvers import thermal_channel_datasets as tc


def test_plan_and_actuation_ranges():
    assert len(tc.PLAN) == 82 and {s for *_, s in tc.PLAN} == set(tc.SPLITS)
    assert len({(f, s) for f, s, _ in tc.PLAN}) == 82                     # every trajectory has its own seed
    for family in ("random_steps_ramps", "sinusoid_chirp", "edge_steps"):
        au, aq = tc.make_actuation(family, 7)
        assert tc.make_actuation(family, 7) == (au, aq)                   # deterministic in the seed
        ts = np.linspace(0, tc.WARMUP + tc.HORIZON, 200)
        u = [tc.eval_act(au, t, "U") for t in ts]
        q = [tc.eval_act(aq, t, "Q") for t in ts]
        assert tc.U_RANGE[0] - 1e-12 <= min(u) and max(u) <= tc.U_RANGE[1] + 1e-12
        assert tc.Q_RANGE[0] - 1e-12 <= min(q) and max(q) <= tc.Q_RANGE[1] + 1e-12
    with pytest.raises(ValueError):
        tc.make_actuation("nope", 0)


def test_generate_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "GRID", dict(nx=16, ny=4, dt=0.02))
    monkeypatch.setattr(tc, "WARMUP", 0.4)
    monkeypatch.setattr(tc, "HORIZON", 0.8)
    monkeypatch.setattr(tc, "PLAN", [("random_steps_ramps", 0, "train"), ("random_steps_ramps", 1, "train"),
                                     ("random_steps_ramps", 2, "train"), ("sinusoid_chirp", 3, "val")])
    monkeypatch.setattr(tc, "SPLITS", ("train", "val"))
    monkeypatch.setattr(tc, "SHARD_ROWS", 2)
    meta = tc.generate(tmp_path, workers=1)
    d = tmp_path / "thermal_channel_cfd"
    assert meta["files"] == ["train-00.npz", "train-01.npz", "val.npz"] and meta["rows"] == {"train": 3, "val": 1}
    assert json.loads((d / "meta.json").read_text())["files"] == meta["files"]
    with np.load(d / "train-00.npz") as z:
        assert z["train__fields"].shape == (2, 3, 4, 16, 4) and z["train__fields"].dtype == np.float32
        assert list(z["train__seed"]) == [0, 1] and np.isfinite(z["train__fields"]).all()
        assert z["train__qoi_T_out"].shape == (2, 3) and z["train__U_fine"].shape == (2, 9)
    with np.load(d / "val.npz") as z:
        assert list(z["val__family"]) == ["sinusoid_chirp"]


def test_parallel_workers_run_the_recipe_the_caller_asked_for(tmp_path, monkeypatch):
    """Two workers must solve the small grid: a spawned worker re-imports the module and used to fall back
    to the 128x32 defaults, so this ran the real recipe (minutes) instead of the test one (seconds)."""
    monkeypatch.setattr(tc, "GRID", dict(nx=8, ny=2, dt=0.05))
    monkeypatch.setattr(tc, "WARMUP", 0.2)
    monkeypatch.setattr(tc, "HORIZON", 0.4)
    monkeypatch.setattr(tc, "REC", 0.2)
    monkeypatch.setattr(tc, "PLAN", [("random_steps_ramps", 0, "train"), ("sinusoid_chirp", 1, "val")])
    monkeypatch.setattr(tc, "SPLITS", ("train", "val"))
    meta = tc.generate(tmp_path / "parallel", workers=2)
    tc.generate(tmp_path / "serial", workers=1)
    assert (meta["nx"], meta["ny"]) == (8, 2) and meta["rows"] == {"train": 1, "val": 1}
    for split in ("train", "val"):
        with np.load(tmp_path / "parallel" / "thermal_channel_cfd" / f"{split}.npz") as a, \
                np.load(tmp_path / "serial" / "thermal_channel_cfd" / f"{split}.npz") as b:
            assert a[f"{split}__fields"].shape == b[f"{split}__fields"].shape
            assert a[f"{split}__fields"].shape[3] == 8                      # the nx of the recipe, not the default 128
            np.testing.assert_array_equal(a[f"{split}__fields"], b[f"{split}__fields"])
            np.testing.assert_array_equal(a[f"{split}__seed"], b[f"{split}__seed"])
