"""PINNeAPPle Lab: runs, caching, validation status, failures, sweeps, datasets, export and catalogue."""
import json
import os

import numpy as np
import pytest

from pinneapple_lab import Experiment, LabStore, available, register, run, sweep
from pinneapple_lab.__main__ import main as cli


@register
class _Square(Experiment):
    name = "_test_square"
    description = "y = a x^2 sampled on a grid; fails validation when a < 0 and crashes when a is None."
    params = {"a": 1.0, "n": 5}
    space = {"a": (0.5, 2.0), "n": [3, 5, 7]}

    def run(self, ctx):
        if ctx.params["a"] is None:
            raise ValueError("a is required")
        with ctx.stage("compute"):
            x = np.linspace(0, 1, ctx.params["n"])
            y = ctx.params["a"] * x ** 2
        ctx.input("x", x)
        ctx.output("y", y)
        ctx.metric("y_max", float(y.max()))
        ctx.metric("loss", float(y.sum()), step=0)
        ctx.check("non_negative", bool((y >= 0).all()))
        ctx.check("max_value", value=float(y.max()), reference=ctx.params["a"], rtol=1e-12)
        ds = ctx.dataset("curves", units={"y": "m"})
        for xi, yi in zip(x, y, strict=True):
            ds.add(point=np.array([xi, yi]), a=ctx.params["a"], label="pos" if yi >= 0 else "neg")


def test_run_records_everything_and_is_cached(tmp_path):
    root = str(tmp_path)
    r = run("_test_square", {"a": 2.0}, root=root)
    assert r.status == "completed" and r.metrics["y_max"] == 2.0
    for f in ("run.json", "metrics.json", "validation.json", "log.txt", "progress.json", "inputs/x.npy",
              "outputs/y.npy", "datasets/curves/card.json"):
        assert os.path.exists(os.path.join(r.dir, f)), f
    rec = json.load(open(os.path.join(r.dir, "run.json")))
    assert rec["params"] == {"a": 2.0, "n": 5} and rec["validation"] == {"total": 2, "failed": 0}
    assert rec["stages"][0]["name"] == "compute"
    again = run("_test_square", {"a": 2.0}, root=root)
    assert again.status == "cached:completed" and again.run_id == r.run_id
    card = json.load(open(os.path.join(r.dir, "datasets/curves/card.json")))
    assert card["n_samples"] == 5 and card["schema"]["point"]["shape"] == [2]


def test_failed_validation_and_crash_are_recorded(tmp_path):
    root = str(tmp_path)
    bad = run("_test_square", {"a": -1.0}, root=root)
    assert bad.status == "failed_validation"
    crash = run("_test_square", {"a": None}, root=root)
    assert crash.status == "failed" and "a is required" in crash.error
    st = LabStore(root).status()["_test_square"]
    assert st == {"failed_validation": 1, "failed": 1}
    with pytest.raises(KeyError):
        run("_test_square", {"b": 1}, root=root)


def test_sweep_grid_and_samples_export_and_catalog(tmp_path):
    root = str(tmp_path)
    res = sweep("_test_square", grid={"a": [1.0, 2.0], "n": [3, 4]}, root=root)
    assert len(res) == 4 and all(r.ok for r in res)
    res2 = sweep("_test_square", samples=5, seed=1, root=root)
    assert len({r.run_id for r in res2}) == 5
    store = LabStore(root)
    assert len(store.runs("_test_square")) == 9
    tab = store.table("_test_square")
    assert {"param.a", "param.n", "metric.y_max"} <= set(tab.columns)
    samples = list(store.samples("_test_square", "curves"))
    assert len(samples) == 3 + 4 + 3 + 4 + sum(r_["params"]["n"] for r_ in store.runs("_test_square")[4:])
    card = store.export_dataset("_test_square", "curves", str(tmp_path / "curves.npz"))
    z = np.load(tmp_path / "curves.npz")
    assert z["point"].shape == (card["n_samples"], 2) and z["param__a"].shape == (card["n_samples"],)
    cat = store.catalog()
    text = open(cat).read()
    assert "_test_square" in text and "| run | status |" in text
    # the index can be rebuilt from the folders alone
    os.remove(store.db)
    assert LabStore(root).reindex() == 9


def test_builtin_experiments_run(tmp_path):
    root = str(tmp_path)
    assert {"oscillator", "heat_xtfc", "bondi_accretion", "accretion_flow", "cylinder_lbm"} <= set(available())
    r = run("oscillator", {"method": "rk4", "zeta": 0.2}, root=root)
    assert r.ok and r.metrics["rmse"] < 1e-6
    r = run("heat_xtfc", {"alpha": 0.2}, root=root)
    assert r.ok and r.metrics["rel_l2_error"] < 1e-3
    pytest.importorskip("numba")
    r = run("bondi_accretion", {"nr": 48, "t_end": 100.0}, root=root)
    assert r.ok and r.metrics["mdot_rel_error"] < 0.02
    r = run("accretion_flow", {"nr": 32, "ntheta": 16, "t_end": 60.0, "every": 20.0}, root=root)
    assert r.ok and r.metrics["frames"] == 4
    r = run("cylinder_lbm", {"Re": 30.0, "D": 8, "height": 4, "length": 8, "steps": 600, "save_every": 50}, root=root)
    assert r.ok and r.metrics["regime_shedding"] == 0
    assert LabStore(root).datasets("cylinder_lbm", "vorticity")[0]["n_samples"] == 6


def test_parallel_sweep_and_cli(tmp_path, capsys):
    root = str(tmp_path)
    res = sweep("oscillator", grid={"zeta": [0.1, 0.4], "method": ["rk4", "symplectic"]}, n_jobs=2, root=root)
    assert all(r.ok for r in res)
    assert cli(["--root", root, "status"]) == 0
    assert "oscillator" in capsys.readouterr().out
    assert cli(["--root", root, "run", "oscillator", "-p", "zeta=0.3"]) == 0
    assert cli(["--root", root, "report"]) == 0
    assert cli(["--root", root, "export", "oscillator", "trajectories", str(tmp_path / "osc.npz")]) == 0
    assert np.load(tmp_path / "osc.npz")["x"].shape[0] == 5
