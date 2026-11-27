"""The tabular half of a campaign: seven design variables in, the quantities out, every family on the same split."""
import csv
import json

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from pinneapple_arena.fieldbench import tabular as tab  # noqa: E402

INPUTS = ["velocity", "wp1", "wp2", "e1", "e2", "r1", "r2"]
TARGETS = ["peak_wall_force", "peak_deceleration"]


def _table(n: int = 90, seed: int = 0) -> dict:
    """A campaign whose quantities are a known function of two of the seven variables, with noise on the rest."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, len(INPUTS)))
    force = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.05 * rng.normal(size=n)
    decel = 1.0 + 0.5 * x[:, 0] * x[:, 0] + 0.05 * rng.normal(size=n)
    y = np.c_[force, decel]
    rows = [dict(zip(INPUTS, map(float, xi), strict=True)) for xi in x]
    return {"x": x, "y": y, "inputs": list(INPUTS), "targets": list(TARGETS), "rows": rows,
            "dropped_out_of_envelope": 0}


# --------------------------------------------------------------------------------------------- the table on disk
def test_load_table_reads_the_design_vector_and_reports_the_rows_it_drops(tmp_path):
    path = tmp_path / "quantities.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(INPUTS + TARGETS + ["out_of_envelope"])
        for i in range(4):
            w.writerow([i] * len(INPUTS) + [10.0 * i, 1.0 * i] + [i == 3])
    kept = tab.load_table(str(path), inputs=INPUTS, targets=TARGETS)
    assert kept["x"].shape == (3, len(INPUTS)) and kept["y"].shape == (3, len(TARGETS))
    assert kept["dropped_out_of_envelope"] == 1                     # reported, never silently
    everything = tab.load_table(str(path), inputs=INPUTS, targets=TARGETS, filter_ok=False)
    assert everything["x"].shape[0] == 4 and everything["dropped_out_of_envelope"] == 0
    assert float(everything["y"][3, 0]) == 30.0


# --------------------------------------------------------------------------------------------- the comparison
def test_every_family_is_scored_on_the_same_split_and_a_linear_target_is_recovered():
    table = _table()
    result = tab.benchmark_tabular(table, seed=42)
    assert result["n_train"] + result["n_test"] <= len(table["x"])
    assert {"ridge", "knn", "extra_trees"} <= set(result["scores"])       # the families scikit-learn provides
    for name, score in result["scores"].items():
        assert set(score["targets"]) == set(TARGETS), name
        assert set(score["mean"]) == {"r2", "mae", "rmse", "mape"}
    # peak wall force is linear in two variables: the linear and the tree families both have to find it
    assert result["scores"]["ridge"]["targets"]["peak_wall_force"]["r2"] > 0.9
    assert result["scores"]["extra_trees"]["targets"]["peak_wall_force"]["r2"] > 0.8
    # ranked by the mean R^2 over the targets, best first
    means = [result["scores"][n]["mean"]["r2"] for n in result["ranked"]]
    assert means == sorted(means, reverse=True)
    # the markdown table carries one row per scored family (plus a header and the separator)
    assert len(result["table"].splitlines()) == len(result["scores"]) + 2


def test_the_split_is_frozen_by_the_seed():
    table = _table()
    first = tab.benchmark_tabular(table, seed=7)
    second = tab.benchmark_tabular(table, seed=7)
    assert first["n_train"] == second["n_train"] and first["n_test"] == second["n_test"]
    for target in TARGETS:
        assert np.allclose(first["predictions"]["ridge"][target], second["predictions"]["ridge"][target])
    other = tab.benchmark_tabular(table, seed=8)
    assert not np.allclose(first["predictions"]["ridge"]["peak_wall_force"],
                           other["predictions"]["ridge"]["peak_wall_force"])


def test_a_family_that_crashes_is_reported_as_unavailable_not_dropped(monkeypatch):
    """A wheel compiled against another NumPy takes the interpreter down: the row has to say why it is missing."""
    def boom():
        raise RuntimeError("the booster is not usable here")

    factories = tab._factories(42)
    monkeypatch.setattr(tab, "_factories", lambda seed: {**factories, "broken": boom})
    result = tab.benchmark_tabular(_table(), seed=42)
    assert "broken" in result["unavailable"]
    assert "the booster is not usable here" in result["unavailable"]["broken"]
    assert "broken" not in result["scores"]
    assert "ridge" in result["scores"]                                   # the rest of the table survives


# --------------------------------------------------------------------------------------------- sensitivity and report
def test_the_sensitivity_names_the_variable_the_target_depends_on():
    table = _table(n=120)
    sens = tab.feature_sensitivity(table, seed=42, models=("extra_trees",))
    for target in TARGETS:
        importance = sens[target]
        assert set(importance) == set(INPUTS)
        assert sum(importance.values()) == pytest.approx(1.0, abs=1e-6)
        assert importance["velocity"] > 0.4, (target, importance)


def test_the_report_writes_the_markdown_and_the_same_numbers_as_json(tmp_path):
    result = tab.benchmark_tabular(_table(), seed=42)
    sens = tab.feature_sensitivity(_table(n=60), seed=42, models=("extra_trees",))
    path = tmp_path / "leaderboard.md"
    tab.write_table_report(result, str(path), sens)
    text = path.read_text()
    assert "| rank | model |" in text
    for name in result["scores"]:
        assert name in text
    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    assert set(payload["scores"]) == set(result["scores"])
    for name, score in result["scores"].items():
        assert payload["scores"][name]["mean"]["r2"] == pytest.approx(score["mean"]["r2"], rel=1e-9)
    assert set(payload["sensitivity"]) == set(TARGETS)


def test_available_regressors_reports_what_this_machine_can_build():
    have = tab.available_regressors()
    assert have["sklearn"] is True
    assert set(have) >= {"sklearn", "xgboost", "lightgbm", "catboost"}
