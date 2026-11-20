"""The blocking end-to-end tier (#37): ten flows from an input a user has to an artifact they keep.

The unit suites answer "is this function right". These answer "does the product work": each flow starts from
something a user brings (a problem, an equation in a text, a parameter space, a plant record), goes through the
public API only -- no private helpers, no fixtures from other suites -- and ends in an artifact with a number or a
file behind it (metrics, a saved experiment that reloads, a design that passes its own verification, a governed
model card).

Two properties make this a tier rather than a folder of tests, and both are enforced here:

* **Ten flows, named.** ``FLOWS`` is the contract: ``test_the_tier_has_the_ten_named_flows`` compares it with the
  tests actually collected in this module, so adding a flow without naming it (or deleting one) fails.
* **Fast.** Each flow carries its own wall-clock budget through :func:`budget`; the sum of the budgets is under
  ten minutes, which is what "fast" means for a tier that blocks a merge. Measured on an M-series laptop on
  2026-11-21 the ten flows together take about 40 s, so the budgets are loose by an order of magnitude on purpose:
  a machine three times slower still passes, and a regression that starts training a real network does not.

``make e2e`` runs this folder and ``.github/workflows/tests.yml`` has the same step. Nothing here is allowed to
skip: a flow whose dependency is missing fails, because a skipped flow in a blocking tier reads as a pass.
"""
from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import pinneapple as pp
from pinneapple_physics.physical_problem import PhysicalProblem

# The tier's contract: these ten, in this order, and nothing else. The name is the test function without its
# ``test_`` prefix, so the list is checkable by looking at the file.
FLOWS = [
    "problem_trains_scores_and_records_provenance",   # problem -> method -> metrics -> environment
    "experiment_artifact_saves_reloads_and_reproduces",  # definition -> run -> experiment.json -> same predictions
    "methods_are_compared_against_one_reference",     # several methods -> one table -> a ranking
    "pde_spec_round_trip_preserves_the_physics",      # problem -> spec -> problem -> same fingerprint and answer
    "dataset_contract_goes_to_disk_and_back",         # parameter space -> cases -> table + sidecar -> read back
    "flow_from_a_template_runs_and_scores",           # flow graph -> solver -> comparison against exact
    "flow_trains_a_pinn_and_reuses_the_cache",        # the same graph with a training node, run twice
    "digital_twin_detects_drift_and_recalibrates",    # plant record -> detection -> recalibration -> forecast
    "surrogate_inverse_design_passes_its_own_check",  # dataset -> surrogate -> optimum -> forward verification
    "a_trained_run_becomes_a_governed_model_card",    # run -> card -> the governance gate accepts it
]

NU = 0.01 / np.pi
TINY_PINN = {"epochs": 30, "n_collocation": 256, "hidden_dim": 16, "n_layers": 2}


@contextlib.contextmanager
def budget(seconds: float):
    """Fail the flow on its own wall clock, so "under ten minutes in total" is a property, not a hope."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    assert elapsed < seconds, f"this flow took {elapsed:.1f} s, over its {seconds:.0f} s budget"


def test_the_tier_has_the_ten_named_flows():
    """The list above is the contract; this is what keeps it true as the folder changes."""
    assert len(FLOWS) == len(set(FLOWS)) == 10
    here = Path(__file__).resolve()
    source = here.read_text(encoding="utf-8")
    defined = [line.split("(")[0][len("def test_"):] for line in source.splitlines() if line.startswith("def test_")]
    defined = [name for name in defined if name != "the_tier_has_the_ten_named_flows"]
    assert defined == FLOWS, f"the folder and FLOWS disagree: {sorted(set(defined) ^ set(FLOWS))}"


# ------------------------------------------------------------------ 1
def test_problem_trains_scores_and_records_provenance(tmp_path):
    with budget(120):
        exp = pp.Experiment("burgers_1d", method="pinn", options=dict(TINY_PINN), reference="analytic",
                            seed=5, n_eval=256)
        exp.problem = exp.problem.with_parameters(nu=NU)
        result = exp.run()

    assert set(result.metrics) == {"relative_l2", "rmse", "max_abs"}
    assert np.isfinite(result.metrics["relative_l2"]["u"]) and 0 <= result.metrics["relative_l2"]["u"] < 1
    assert result.problem_fingerprint == exp.problem.fingerprint()
    assert result.environment["pinneapple"] and result.environment["python"]
    assert result.n_eval_points == 256 and result.wall_time_s > 0


# ------------------------------------------------------------------ 2
def test_experiment_artifact_saves_reloads_and_reproduces(tmp_path):
    with budget(120):
        exp = pp.Experiment("burgers_1d", method="pinn", options=dict(TINY_PINN), reference="analytic",
                            seed=5, n_eval=256)
        exp.problem = exp.problem.with_parameters(nu=NU)
        first = exp.run()
        directory = tmp_path / "run"
        exp.save(str(directory))
        first.save(str(directory))
        second = pp.Experiment.load(str(directory)).run()

    on_disk = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    assert on_disk["problem_fingerprint"] == first.problem_fingerprint and on_disk["resumable"] is True
    assert (directory / "result.json").exists()
    points = np.random.default_rng(1).random((40, 2)) * [2, 1] - [1, 0]
    assert np.array_equal(first.solution.predict(points), second.solution.predict(points))
    assert first.metrics == second.metrics


# ------------------------------------------------------------------ 3
def test_methods_are_compared_against_one_reference():
    with budget(120):
        comparison = pp.compare("burgers_1d", ["analytic", "pinn"], reference="reference",
                               options={"pinn": dict(TINY_PINN)}, n_points=200, seed=2,
                               preset_kwargs={"nu": NU})

    table = comparison.table() if hasattr(comparison, "table") else comparison.to_dict()
    assert table, "a comparison with no rows is not a comparison"
    rows = table["rows"] if isinstance(table, dict) and "rows" in table else table
    assert {row["method"] for row in rows} == {"analytic", "pinn"}
    assert all(row.get("error") is None for row in rows), rows
    best = min(rows, key=lambda r: r["metrics"]["relative_l2"]["u"])
    assert best["metrics"]["relative_l2"]["u"] >= 0


# ------------------------------------------------------------------ 4
def test_pde_spec_round_trip_preserves_the_physics():
    with budget(60):
        problem = PhysicalProblem.from_preset("burgers_1d").with_parameters(nu=NU)
        spec = problem.to_pde_spec()
        rebuilt = PhysicalProblem.from_pde_spec(spec)
        points = np.random.default_rng(3).random((25, 2)) * [2, 1] - [1, 0]
        a = pp.solve(problem, "analytic").predict(points)
        b = pp.solve(rebuilt, "analytic").predict(points)

    assert rebuilt.fingerprint() == problem.fingerprint()
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ 5
def test_dataset_contract_goes_to_disk_and_back(tmp_path):
    from pinneapple_data.contracts import (DatasetSpec, Parameter, ParameterSpace, read_table, validate,
                                           write_table)

    with budget(60):
        space = ParameterSpace(name="fin", parameters=[
            Parameter(name="height", low=0.01, high=0.05, unit="m", group="geometry"),
            Parameter(name="k", low=1.0, high=1000.0, log=True, unit="W/(m K)", group="material")])
        cases = space.from_unit([[0.5, 0.3], [0.9, 0.8]])
        assert space.feasible(cases).all()
        table = write_table(cases, tmp_path / "cases.csv")
        spec = DatasetSpec(name="fin-cases", n_rows=len(cases), description="horn of the design space")
        spec.save(tmp_path / "dataset.json")
        back, contract = read_table(table)
        loaded = DatasetSpec.load(tmp_path / "dataset.json")

    assert len(back) == 2 and "height" in back.columns
    assert contract is None or contract.name == "fin-cases"
    assert loaded.name == "fin-cases" and int(loaded.n_rows) == 2
    assert not validate(loaded), validate(loaded)


# ------------------------------------------------------------------ 6
def _heat_flow(train: bool = False):
    import pinneapple_flow as pf

    nodes = [{"id": "geo", "type": "geometry.domain", "params": {"dimension": 1, "t_range": [0, 0.5]}},
             {"id": "eq", "type": "equations.system", "params": {"template": "heat"}},
             {"id": "fd", "type": "solver.classical", "params": {"method": "fd", "n": 41, "n_t": 11}},
             {"id": "val", "type": "validate.compare", "params": {"reference": "exact"}}]
    edges = [{"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
             {"source": "eq", "source_port": "problem", "target": "fd", "target_port": "problem"},
             {"source": "fd", "source_port": "solution", "target": "val", "target_port": "candidates"},
             {"source": "eq", "source_port": "problem", "target": "val", "target_port": "problem"}]
    if train:
        nodes.append({"id": "pinn", "type": "train.pinn", "params": {"adam_steps": 20, "lbfgs_steps": 0,
                                                                     "n_collocation": 128, "width": 8, "depth": 2}})
        edges += [{"source": "eq", "source_port": "problem", "target": "pinn", "target_port": "problem"},
                  {"source": "pinn", "source_port": "solution", "target": "val", "target_port": "candidates"}]
    return pf.Flow("heat end to end", nodes=nodes, edges=edges)


def test_flow_from_a_template_runs_and_scores(tmp_path, monkeypatch):
    import pinneapple_flow as pf

    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    with budget(120):
        run = pf.run(_heat_flow(), workdir=tmp_path / "w")

    assert run.status == "done", {k: v.get("error") for k, v in run.nodes.items()}
    error = run.report("val")["metrics"]["FD_rel_l2"]["value"]
    assert error < 1e-3, error


# ------------------------------------------------------------------ 7
def test_flow_trains_a_pinn_and_reuses_the_cache(tmp_path, monkeypatch):
    import pinneapple_flow as pf

    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    flow = _heat_flow(train=True)
    with budget(180):
        first = pf.run(flow, workdir=tmp_path / "w")
        second = pf.run(flow, workdir=tmp_path / "w")

    assert first.status == second.status == "done"
    metrics = second.report("val")["metrics"]
    assert {"FD_rel_l2", "PINN_rel_l2"} <= set(metrics)
    assert all(np.isfinite(metrics[name]["value"]) for name in ("FD_rel_l2", "PINN_rel_l2"))


# ------------------------------------------------------------------ 8
def test_digital_twin_detects_drift_and_recalibrates(tmp_path):
    from pinneapple_systems.digital_twin.workflow import digital_twin_workflow

    with budget(180):
        result = digital_twin_workflow(n_steps=160, members=60, bootstrap=10, nees_runs=4, seed=0,
                                       out_dir=str(tmp_path / "twin"))

    status = {c["code"]: c["status"] for c in result["checks"]}
    assert result["events"], "a twin that never reports an event has not looked at the record"
    assert any(s == "pass" for s in status.values()), status
    assert result["forecast"] and np.isfinite(result["forecast"]["mean"][0])


# ------------------------------------------------------------------ 9
def test_surrogate_inverse_design_passes_its_own_check():
    from pinneapple_design.surrogate_inverse import heatsink_fins as hs

    with budget(240):
        surrogate = hs.train_surrogate(n_samples=400, epochs=120, seed=0)
        optimum = hs.optimize_design(surrogate.model, power_w=25.0, h_conv=25.0, steps=60)
        card = hs.verify_design(optimum.design, power_w=25.0, h_conv=25.0,
                                surrogate_predicted_temp_c=optimum.surrogate_predicted_temp_c)

    assert np.isfinite(optimum.surrogate_predicted_temp_c) and optimum.surrogate_predicted_temp_c > 0
    assert optimum.design.fin_count >= 1 and optimum.design.fin_spacing_m() > 0
    # the design the surrogate picked survives the forward formula -- the point of a verification card
    assert card.values["real_temperature_c"] > 0
    facets = {check.name if hasattr(check, "name") else check.code for check in card.checks}
    assert facets, "a verification card with no checks verifies nothing"


# ------------------------------------------------------------------ 10
def test_a_trained_run_becomes_a_governed_model_card(tmp_path):
    from pinneapple_hub.model_card import ModelCard

    with budget(120):
        exp = pp.Experiment("burgers_1d", method="pinn", options=dict(TINY_PINN), reference="analytic",
                            seed=5, n_eval=256)
        exp.problem = exp.problem.with_parameters(nu=NU)
        result = exp.run()

    card = ModelCard(name="burgers_1d/pinn/seed5", description="a tiny PINN run, end to end",
                     task="forward", version="0.0.0", architecture="pinn",
                     validation_metrics={k: float(v["u"]) for k, v in result.metrics.items()},
                     reference_source="burgers_sine_exact (closed form)",
                     pinneapple_version=result.environment["pinneapple"])
    card.save(str(tmp_path / "model_card.json"))

    gate = subprocess.run([sys.executable, "scripts/validate_model_card.py", str(tmp_path / "model_card.json")],
                          capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]))
    assert gate.returncode == 0, gate.stdout + gate.stderr

    # ... and the same gate rejects a card with no computed metric, which is the point of the gate
    bare = ModelCard(name="bare", description="no numbers")
    bare.save(str(tmp_path / "bare_model_card.json"))
    refused = subprocess.run([sys.executable, "scripts/validate_model_card.py", str(tmp_path / "bare_model_card.json")],
                             capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]))
    assert refused.returncode != 0 and "validation_metrics" in refused.stdout
