"""Cylinder-wake recipe (cylinder_wake.py): Strouhal estimate, one short run, dataset layout."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from pinneapple_simulation.numerical_solvers import cylinder_wake as cw  # noqa: E402


def test_probe_strouhal_is_accurate_between_spectral_bins():
    t = np.arange(80) * 0.5                                        # 40 convective units, 2 samples per unit
    s = np.sin(2 * np.pi * 0.2447 * t + 0.3)
    assert cw.probe_strouhal(s, 0.5) == pytest.approx(0.2447, rel=3e-3)   # raw FFT bins are 0.025 apart


def test_short_low_re_run_is_steady_and_finite():
    r = cw.simulate(10.0, D=8, height=4, length=8, conv_time=6.0, keep_time=3.0)
    assert r["finite"] and r["regime"] == "steady" and r["strouhal"] == 0.0
    assert r["vorticity"].shape == (len(r["t"]), 64, 32) and np.all(r["vorticity"][:, r["solid"]] == 0)
    assert r["t"][0] > 2.9 and r["t"][-1] == pytest.approx(6.0)


def test_generate_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(cw, "SETUP", {**cw.SETUP, "D": 8, "height": 4, "length": 8, "conv_time": 4.0, "keep_time": 2.0})
    monkeypatch.setattr(cw, "PLAN", {"train": [10.0, 12.0], "test": [11.0]})
    monkeypatch.setattr(cw, "SHARD_ROWS", 5)
    meta = cw.generate(tmp_path, workers=1)
    d = tmp_path / "cylinder_wake_lbm"
    assert meta["rows"] == {"train": 8, "test": 4} and meta["files"] == ["train-00.npz", "train-01.npz", "test.npz"]
    assert json.loads((d / "meta.json").read_text())["nx"] == 64
    with np.load(d / "train-01.npz") as z:
        assert z["train__vorticity"].shape == (3, 64, 32) and list(z["train__Re"]) == [12.0] * 3
        assert set(z["train__regime"]) == {"steady"}
