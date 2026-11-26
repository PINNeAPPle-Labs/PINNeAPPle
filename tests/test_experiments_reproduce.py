"""Reproducing a recorded run: the three outcomes, each one asserted (#1480).

What these tests hold is the distinction the issue is about: a re-run that produces a different number and says
nothing is the failure mode. ``reproduce`` answers with ``reproduced``, ``divergent`` or ``refused``, names what
moved, and does not dress a refusal up as a run.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

import pinneapple as pp
from pinneapple_physics.experiments import (
    DIVERGENT,
    REFUSED,
    REPRODUCED,
    Reproducer,
    register_reproducer,
    reproduce,
)

SMALL_PINN = {"epochs": 20, "n_collocation": 128, "hidden_dim": 8, "n_layers": 2}


def recorded_experiment(tmp_path, name="burgers_pinn"):
    """One real experiment, run and saved the way a user would: experiment.json plus result.json."""
    directory = tmp_path / name
    experiment = pp.Experiment("burgers_1d", method="pinn", options=dict(SMALL_PINN), seed=3,
                               reference="analytic", n_eval=128)
    experiment.save(str(directory))
    experiment.run().save(str(directory))
    return directory


def rewrite(path, change):
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload, indent=1))


def test_a_run_reproduces_on_the_recorded_environment(tmp_path):
    directory = recorded_experiment(tmp_path)
    result = reproduce(directory)
    assert result.outcome == REPRODUCED, result.reason
    assert bool(result) is True and result.reproduced and not result.divergent and not result.refused
    assert result.differences == []
    assert result.metrics and all(m.ok for m in result.metrics)
    assert {m.name for m in result.metrics} >= {"relative_l2.u", "rmse.u"}
    assert "Outcome: reproduced" in result.to_markdown()


def test_exact_asks_for_bit_identical_numbers(tmp_path):
    """The same machine, the same seed: the strong claim is available, and it is the caller who asks for it."""
    directory = recorded_experiment(tmp_path)
    result = reproduce(directory, exact=True)
    assert result.outcome == REPRODUCED, result.reason
    assert all(m.rtol == 0.0 for m in result.metrics)


def test_a_bumped_dependency_diverges_and_names_the_version(tmp_path):
    """Acceptance: not identical-looking output — the changed version is listed."""
    directory = recorded_experiment(tmp_path)

    def bump(payload):
        payload["environment"]["numpy"] = "0.0.0+fake"
    rewrite(directory / "result.json", bump)

    result = reproduce(directory)
    assert result.outcome == DIVERGENT
    assert bool(result) is False
    changed = {d.what: d for d in result.software_changes}
    assert "numpy" in changed
    assert changed["numpy"].recorded == "0.0.0+fake" and changed["numpy"].now != "0.0.0+fake"
    assert "numpy" in result.reason
    assert all(m.ok for m in result.metrics), "the metrics still agree; what moved is the environment"


def test_the_machine_is_reported_apart_from_the_software(tmp_path):
    """A different CPU is a different run, but it is not the same failure as a version bump."""
    directory = recorded_experiment(tmp_path)
    rewrite(directory / "result.json", lambda p: p["environment"].__setitem__("platform", "FreeBSD-14"))
    result = reproduce(directory)
    assert result.outcome == DIVERGENT
    assert result.software_changes == []
    assert {d.what for d in result.hardware_changes} == {"platform"}
    assert "the machine differs" in result.reason


def test_a_metric_out_of_tolerance_diverges_and_says_which(tmp_path):
    directory = recorded_experiment(tmp_path)

    def inflate(payload):
        payload["metrics"]["relative_l2"]["u"] *= 1.5
    rewrite(directory / "result.json", inflate)

    result = reproduce(directory)
    assert result.outcome == DIVERGENT
    assert result.metrics_out_of_tolerance == ["relative_l2.u"]
    assert "relative_l2.u" in result.reason
    assert any(d.kind == "metric" and d.what == "relative_l2.u" for d in result.differences)


def test_a_record_without_a_result_is_refused_not_run(tmp_path):
    """Refusal is a result: no result.json means there is nothing to compare a re-run against."""
    directory = tmp_path / "half"
    pp.Experiment("burgers_1d", options=dict(SMALL_PINN), seed=1).save(str(directory))
    result = reproduce(directory)
    assert result.outcome == REFUSED
    assert "result.json" in result.reason and str(directory) in result.reason


def test_a_definition_that_no_longer_matches_its_fingerprint_is_refused(tmp_path):
    """A record edited after the save cannot be run faithfully — it would run something that is not the record."""
    directory = recorded_experiment(tmp_path)
    rewrite(directory / "experiment.json", lambda p: p.__setitem__("problem_fingerprint", "0" * 64))
    result = reproduce(directory)
    assert result.outcome == REFUSED
    assert "cannot be rebuilt" in result.reason


def test_a_method_this_installation_does_not_have_is_refused(tmp_path):
    directory = recorded_experiment(tmp_path)
    rewrite(directory / "experiment.json", lambda p: p.__setitem__("method", "magic_solver"))
    result = reproduce(directory)
    assert result.outcome == REFUSED
    assert "magic_solver" in result.reason or "cannot be rebuilt" in result.reason


def test_a_missing_record_is_refused_with_where_it_looked(tmp_path):
    result = reproduce(tmp_path / "nao_existe")
    assert result.outcome == REFUSED and "no recorded run" in result.reason
    assert reproduce("nao_existe", root=tmp_path).outcome == REFUSED


def test_a_directory_no_kind_recognises_is_refused_naming_the_kinds(tmp_path):
    (tmp_path / "vazio").mkdir()
    result = reproduce(tmp_path / "vazio")
    assert result.outcome == REFUSED and "experiment" in result.reason


def test_the_report_carries_both_tables(tmp_path):
    directory = recorded_experiment(tmp_path)
    rewrite(directory / "result.json", lambda p: p["environment"].__setitem__("numpy", "0.0.0+fake"))
    markdown = reproduce(directory).to_markdown()
    assert "| Metric | Recorded | Now |" in markdown and "| Kind | What | Recorded | Now |" in markdown
    assert "`numpy`" in markdown


def test_a_second_kind_cannot_hijack_a_registered_one():
    """Two packages must not disagree about what "experiment" means; registering it again is refused."""
    with pytest.raises(ValueError, match="already registered"):
        register_reproducer(Reproducer("experiment", lambda d: False, lambda d: ({}, lambda: None)))


def test_the_metric_comparison_refuses_to_pass_on_a_missing_number():
    """A metric recorded and not produced is not "within tolerance": a silently absent number is a passing
    comparison that compared nothing."""
    from pinneapple_physics.experiments import MetricComparison

    assert not MetricComparison("relative_l2.u", 0.1, None, 1e-3).ok
    assert not MetricComparison("relative_l2.u", None, 0.1, 1e-3).ok
    assert MetricComparison("relative_l2.u", 0.1, 0.1 * (1 + 1e-9), 1e-6).ok


# ----------------------------------------------------------------------------------- the workflow kind
def flow_doc(**params):
    return {"schema": "pinneapple-flow/1", "name": "heat", "nodes": [
        {"id": "geo", "type": "geometry.domain", "params": {"dimension": 1}},
        {"id": "eq", "type": "equations.system", "params": {"template": "heat"}},
        {"id": "fd", "type": "solver.classical", "params": {"method": "fd", "n": 21, "n_t": 5}}],
        "edges": [{"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
                  {"source": "eq", "source_port": "problem", "target": "fd", "target_port": "problem"}],
        **params}


def run_and_save(tmp_path, document, name="heat"):
    import pinneapple_flow as pf

    flow = pf.Flow.from_dict(document)
    result = pf.run(flow, workdir=tmp_path / f"{name}-run")
    assert result.status == "done", result.nodes
    directory = tmp_path / name
    pf.reproducibility.save_run(directory, flow, result)
    return directory


def test_a_flow_run_reproduces_from_its_record(tmp_path):
    directory = run_and_save(tmp_path, flow_doc())
    assert "workflow" in pp.experiments.producers()
    result = reproduce(directory)
    assert result.outcome == REPRODUCED, result.reason
    assert result.kind == "workflow"
    assert {m.name for m in result.metrics} >= {"fd.n_unknowns", "fd.nfev"}


@pytest.mark.skipif(shutil.which("python3.11") is None, reason="needs the interpreter the suite runs on")
def test_a_flow_whose_dataset_is_gone_is_refused_naming_it(tmp_path):
    """Acceptance: the record is refused, and the sentence names what is missing instead of running on whatever is
    present."""
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "Sod.hdf5"
    with h5py.File(source, "w") as f:
        f.create_dataset("density", data=np.zeros((1, 4, 6), dtype=np.float32))
        f.create_dataset("x-coordinate", data=np.linspace(0, 1, 6, dtype=np.float32))
        f.create_dataset("t-coordinate", data=np.linspace(0, 1, 4, dtype=np.float32))
        f.attrs["case"] = "synthetic 1D"
    document = {"schema": "pinneapple-flow/1", "name": "archive", "edges": [],
                "nodes": [{"id": "pd", "type": "data.public_dataset",
                           "params": {"id": "pdebench", "arguments": {"source": str(source)}}}]}
    directory = run_and_save(tmp_path, document, name="archive")
    source.unlink()

    result = reproduce(directory)
    assert result.outcome == REFUSED
    assert "pd" in result.reason, result.reason
    assert "Sod.hdf5" in result.reason, "the refusal has to name what is missing"


def test_a_record_without_an_environment_is_not_called_a_reproduction(tmp_path):
    """A record that does not say what it ran on cannot be said to have reproduced: the numbers matching is not the
    whole claim, and the report says which half was made."""
    directory = run_and_save(tmp_path, flow_doc())
    (directory / "environment.json").unlink()
    result = reproduce(directory)
    assert result.outcome == DIVERGENT
    assert result.environment_recorded is False
    assert any("does not say which versions" in d.note for d in result.differences)
    assert all(m.ok for m in result.metrics), "the numbers did match; what is missing is the record of the machine"
    assert result.environment, "what this machine is running is still reported"
