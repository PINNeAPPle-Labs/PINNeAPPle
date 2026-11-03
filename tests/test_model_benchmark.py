"""Model benchmarking (#1191): two identical models tie; a deliberately worse one ranks last, separated."""
from __future__ import annotations

import numpy as np

from pinneapple_analysis.surrogate.benchmark_workflow import benchmark_models
from pinneapple_analysis.surrogate.model_source import Predictor

TOOL = "pinneapple_physics.closed_form.ideal_reactors:pfr_conversion"
B = {"D": (0.02, 0.1), "L": (1.0, 10.0), "k": (0.01, 0.5), "Q": (1e-4, 1e-3)}


def _exact(noise=0.0, name="exact"):
    from pinneapple_physics.closed_form.ideal_reactors import pfr_conversion

    def predict(x):
        n = len(x["D"])
        y = np.array([pfr_conversion(**{k: float(x[k][i]) for k in B})["X"] for i in range(n)])
        return {"X": y + noise}
    return Predictor(name, "callable", list(B), ["X"], dict(B), predict)


def test_identical_models_tie_and_a_worse_one_is_last_and_separated():
    r = benchmark_models(families=(), predictors=[_exact(0.0, "a"), _exact(0.0, "b"), _exact(0.05, "worse")],
                         tool=TOOL, bounds=B, output="X", n_train=8, n_test=60, seeds=(0, 1))
    assert r["winner"] is None and set(r["tied_with_best"][:2]) == {"callable:a", "callable:b"}
    assert r["summary"][-1]["candidate"] == "callable:worse"
    worse = next(p for p in r["pairs"] if p["other"] == "callable:worse")
    assert worse["separated"]


def test_families_on_one_protocol():
    r = benchmark_models(families=("gp", "quadratic"), tool=TOOL, bounds=B, output="X", n_train=40, n_test=60,
                         seeds=(0,))
    assert {s["candidate"] for s in r["summary"]} == {"gp", "quadratic"}
    assert all(row["extrapolation_rel_rmse"] is not None for row in r["rows"])
    assert r["protocol"]["seeds"] == [0]


def test_flow_and_planner():
    from pinneapple_flow.planner.extensions import match
    from pinneapple_flow.registry import REGISTRY
    assert "benchmark.models" in REGISTRY.types()
    assert match("comparar modelos: qual modelo é melhor para o arrasto (benchmark)")[0][1].id == "benchmark.models"
