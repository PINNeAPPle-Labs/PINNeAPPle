"""Regression tests for pinneapple_app's experiment runner and surrogate report.

Covers the bugs fixed in core/experiment.py:
* the physics loss was never applied (compile_problem called with the wrong
  arguments inside a bare try/except) and every boundary was forced to u=0;
* reference data was used both for training and for the reported metrics;
* custom equations compiled to a ``* 0.0`` placeholder.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sympy")

from pinneapple_app.backend.core.collocation import CollocationConfig
from pinneapple_app.backend.core.data_pipeline import DataConfig, run_data_pipeline
from pinneapple_app.backend.core.experiment import (
    ExperimentConfig, ExperimentRunner, ModelRunConfig, _split_reference,
)
from pinneapple_app.backend.core.physics_adapter import parse_equation, parse_location
from pinneapple_app.backend.core.problem import (
    BoundaryConditionSpec, EquationSpec, define_custom, load_preset,
)
from pinneapple_app.backend.core.surrogate_report import convergence_section


def _run(problem, epochs, ref=None):
    col = CollocationConfig(n_interior=256, n_boundary=128, n_initial=64)
    data = run_data_pipeline(problem, col, DataConfig(use_solver=False), verbose=False)
    if ref is not None:
        data.u_ref = ref(data.x_col)
    cfg = ExperimentConfig(problem_name=problem.name, models=[ModelRunConfig("vanilla_pinn")],
                           epochs=epochs, lr=2e-3, auto_improve=False)
    res = asyncio.run(ExperimentRunner(cfg, data, problem).run())
    r = res.model_results["vanilla_pinn"]
    assert r.error is None, r.error
    return r


def test_preset_physics_and_conditions_are_applied():
    r = _run(load_preset("burgers_1d"), epochs=5)
    assert all(v > 0 for v in r.physics_loss_history)          # was always 0
    terms = {t["term"] for t in r.report["weights"]["loss_terms"]}
    assert {"pde", "ic_u_init", "bc_bc_left", "bc_bc_right"} <= terms
    names = {c["name"] for c in r.report["physics_checks"]}
    assert {"governing_equation", "initial:u_init", "boundary:bc_left"} <= names


def test_unplaceable_geometry_conditions_are_reported_not_dropped():
    r = _run(load_preset("cpu_heatsink_thermal"), epochs=2)
    failed = {c["name"] for c in r.report["physics_checks"] if c["status"] == "fail"}
    assert {"boundary:cpu_base", "boundary:fin_surfaces"} <= failed
    assert r.report["verdict"]["status"] == "not_trustworthy"


def test_reference_split_is_disjoint():
    x = np.arange(200, dtype=np.float32).reshape(100, 2)
    split, note = _split_reference(x, x[:, :1] * 2, 1, np.random.default_rng(0))
    assert note is None
    rows = lambda a: {tuple(r) for r in a}
    tr, cal, te = rows(split["X_train"]), rows(split["X_cal"]), rows(split["X_test"])
    assert not (tr & te) and not (tr & cal) and not (cal & te)
    assert len(tr) + len(cal) + len(te) == 100


def test_reference_with_wrong_columns_is_ignored_with_a_note():
    x = np.zeros((50, 2), dtype=np.float32)
    split, note = _split_reference(x, np.zeros((50, 1)), 3, np.random.default_rng(0))
    assert split == {} and "column" in note


def test_custom_problem_trains_on_real_equation_and_reports_held_out_error():
    problem = define_custom(
        "poisson", [EquationSpec("u_xx + u_yy + 2*pi**2*sin(pi*x)*sin(pi*y)")],
        [BoundaryConditionSpec("dirichlet", "", 0.0)], domain_bounds={"x": (0, 1), "y": (0, 1)},
    )
    exact = lambda X: (np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]))[:, None].astype(np.float32)
    r = _run(problem, epochs=20, ref=exact)
    assert r.physics_loss_history[0] > 1.0                     # source term is really there
    err = r.report["error"]
    assert err["available"] and err["n_test_points"] == round(0.15 * 256)
    assert r.metrics["l2_relative"] == pytest.approx(err["rel_l2_pct"] / 100)
    assert r.report["uncertainty"]["conformal"]["n_calibration"] > 0


def test_parse_equation_and_locations():
    import sympy as sp
    expr = parse_equation("u_t + u*u_x - 0.01*u_xx", ["x", "t"], ["u"])
    assert expr.has(sp.Derivative)
    with pytest.raises(ValueError, match="unknown symbol"):
        parse_equation("u_xx - nu", ["x"], ["u"])
    b = {"x": (0.0, 2.0), "y": (0.0, 1.0)}
    assert parse_location("x=2", b, ["x", "y"]) == [("x", "max")]
    assert parse_location("bottom", b, ["x", "y"]) == [("y", "min")]
    with pytest.raises(ValueError):
        parse_location("x=0.5", b, ["x", "y"])


@pytest.mark.parametrize("hist,status", [
    (list(np.geomspace(1, 1e-3, 200)) + [1e-3] * 50, "converged"),
    (list(np.geomspace(1, 1e-2, 100)), "still_improving"),
    ([1.0] * 100, "stalled"),
    ([1.0, 2.0, float("nan")], "diverged"),
])
def test_convergence_status(hist, status):
    assert convergence_section(hist)["status"] == status


def test_the_app_records_its_run_as_a_library_experiment():
    """#29: the record an app run leaves behind is ``pp.Experiment``'s own, fingerprint and all."""
    import pinneapple as pp
    from pinneapple_app.backend.core.benchmark import build_benchmark_payload

    problem = load_preset("burgers_1d")
    col = CollocationConfig(n_interior=64, n_boundary=32, n_initial=16)
    data = run_data_pipeline(problem, col, DataConfig(use_solver=False), verbose=False)
    cfg = ExperimentConfig(problem_name="burgers_1d", models=[ModelRunConfig("vanilla_pinn")],
                           epochs=3, seed=7, auto_improve=False)
    res = asyncio.run(ExperimentRunner(cfg, data, problem).run())

    record = res.experiments[0]
    assert record["method"] == "vanilla_pinn" and record["seed"] == 7
    assert record["problem_fingerprint"] == pp.Experiment("burgers_1d").problem.fingerprint()
    assert record["resumable"] is True and record["problem_source"]["preset"] == "burgers_1d"
    assert record["options"]["epochs"] == 3
    # ... and it travels in the payload the UI reads, without breaking it
    payload = build_benchmark_payload(res)
    assert payload["experiments"][0]["seed"] == 7 and payload["leaderboard"]


def test_a_custom_problem_is_recorded_without_a_fingerprint():
    """A custom problem has no preset for the library to rebuild, so the record says that instead of guessing."""
    cfg = ExperimentConfig(problem_name="__custom__", models=[ModelRunConfig("vanilla_pinn")], seed=1)
    record = ExperimentRunner(cfg, None, None).manifests()[0]
    assert record["method"] == "vanilla_pinn" and record["resumable"] is False
    assert record["problem_fingerprint"] is None and "__custom__" in record["unavailable"]
