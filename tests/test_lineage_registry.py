"""The physical data graph (#156): one lineage over the four registry stores, and the question it answers.

Given a prediction, ``Lineage.trace`` returns the problem, the geometry, the simulation runs, the dataset version,
the model version and the validation record — and names the ones nobody recorded instead of leaving them out.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from pinneapple_data.lineage import Artifact, Lineage, from_registry
from pinneapple_registry import ArtifactRegistry

PROBLEM = "pipe_flow_demo"


@pytest.fixture()
def registry(tmp_path):
    return ArtifactRegistry(str(tmp_path / "artifacts"))


def _dataset(registry, scenario="case_01", **metadata):
    coords = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    registry.datasets.save(PROBLEM, scenario, coords, {"u": coords[:, 0] ** 2}, metadata=metadata)
    return scenario


def _model(registry, **metadata):
    return registry.models.save(PROBLEM, torch.nn.Linear(2, 1), metadata=metadata)


def test_a_prediction_traces_back_to_all_six_groups(registry):
    registry.problems.save_from_preset(PROBLEM, "pipe_flow_3d", diameter=0.05)
    scenario = _dataset(registry, geometry="pipe_rev_B")
    version = _model(registry, datasets=[scenario], metrics={"rmse": 0.02})
    newest = registry.models.save(PROBLEM, torch.nn.Linear(2, 1), metadata={"datasets": [scenario]},
                                  trust_report={"status": "PASS", "checks": 4})

    g = from_registry(registry, PROBLEM, prediction="PRED-001")
    answer = g.trace("PRED-001")

    assert answer["complete"] is True and answer["missing"] == []
    assert answer["prediction"]["id"] == "PRED-001"
    assert [n["version"] for n in answer["problem"]] == [registry.problems.versions(PROBLEM)[-1]]
    assert [n["name"] for n in answer["geometry"]] == ["pipe_rev_B"]
    assert [n["name"] for n in answer["simulations"]] == [scenario]
    assert [d["id"] for d in answer["datasets"]] == [f"dataset:{PROBLEM}/{scenario}"]
    assert answer["datasets"][0]["version"], "a dataset answer without a version answers half the question"
    assert [m["version"] for m in answer["models"]] == [newest], "the prediction comes from the newest model version"
    assert [v["version"] for v in answer["validation"]] == [newest]
    assert answer["validation"][0]["derived_from"] == [f"model:{PROBLEM}@{newest}"]


def test_the_chains_run_from_the_prediction_back_to_the_problem(registry):
    registry.problems.save_from_preset(PROBLEM, "pipe_flow_3d")
    scenario = _dataset(registry, geometry="pipe_rev_B")
    version = _model(registry, datasets=[scenario])
    g = from_registry(registry, PROBLEM, prediction="PRED-002", model_version=version)

    assert g.nodes["PRED-002"].inputs == [f"model:{PROBLEM}@{version}"]
    upstream = g.upstream("PRED-002")
    kinds = [g.nodes[n].kind for n in upstream]
    assert kinds[0] == "model", "the nearest artifact of a prediction is the model that produced it"
    assert {"dataset", "geometry", "problem"} <= set(kinds), kinds
    assert g.downstream(f"dataset:{PROBLEM}/{scenario}")[0].startswith("model:")


def test_what_nobody_recorded_is_reported_as_missing(registry):
    registry.problems.save_from_preset(PROBLEM, "pipe_flow_3d")
    _dataset(registry)                                  # a dataset that names no geometry
    version = _model(registry)                          # a model whose metadata names no dataset
    g = from_registry(registry, PROBLEM, prediction="PRED-003", model_version=version)
    answer = g.trace("PRED-003")

    assert answer["complete"] is False
    # the dataset exists in the store, but the model never named it: it is not upstream of the prediction at all
    assert set(answer["missing"]) == {"geometry", "simulations", "datasets", "validation"}, answer["missing"]
    note = g.nodes[f"model:{PROBLEM}@{version}"].notes
    assert "names no dataset" in note, note


def test_a_problem_with_several_versions_keeps_all_of_them(registry):
    registry.problems.save_from_preset(PROBLEM, "pipe_flow_3d", diameter=0.05)
    registry.problems.save_from_preset(PROBLEM, "pipe_flow_3d", diameter=0.08)
    versions = registry.problems.versions(PROBLEM)
    assert len(versions) == 2
    g = from_registry(registry, PROBLEM)
    problems = [n for n in g.nodes.values() if n.kind == "problem"]
    assert sorted(p.version for p in problems) == versions
    dataset = _dataset(registry)
    g = from_registry(registry, PROBLEM)
    assert g.nodes[f"dataset:{PROBLEM}/{dataset}"].inputs == [f"problem:{PROBLEM}@{versions[-1]}"]


def test_an_experiment_run_is_linked_to_the_model_that_names_it(registry):
    registry.problems.save_from_preset(PROBLEM, "pipe_flow_3d")
    run = registry.experiments.start_run(PROBLEM, "first")
    registry.experiments.log_metrics(run, {"loss": 0.5})
    registry.experiments.finish_run(run, {"loss": 0.1})
    version = _model(registry, run_id=run)

    g = from_registry(registry, PROBLEM, prediction="PRED-004", model_version=version)
    node = f"experiment:{PROBLEM}#{run}"
    assert node in g.nodes and g.nodes[node].kind == "experiment"
    assert node in g.nodes[f"model:{PROBLEM}@{version}"].inputs
    assert str(run) in g.upstream("PRED-004") or node in g.upstream("PRED-004")


def test_a_prediction_without_a_model_says_so(registry):
    registry.problems.save_from_preset(PROBLEM, "pipe_flow_3d")
    g = from_registry(registry, PROBLEM, prediction="PRED-005")
    assert g.nodes["PRED-005"].inputs == []
    assert "no model version" in g.nodes["PRED-005"].notes
    assert {"models", "validation"} <= set(g.trace("PRED-005")["missing"])


def test_the_registry_can_be_given_as_a_path(registry):
    registry.problems.save_from_preset(PROBLEM, "pipe_flow_3d")
    g = from_registry(registry.root, PROBLEM)
    assert any(n.kind == "problem" for n in g.nodes.values())


def test_the_graph_is_consistent_and_exportable(registry):
    registry.problems.save_from_preset(PROBLEM, "pipe_flow_3d")
    scenario = _dataset(registry, geometry="pipe_rev_B")
    _model(registry, datasets=[scenario])
    g = from_registry(registry, PROBLEM, prediction="PRED-006")

    assert not [c for c in g.checks() if c["status"] == "fail"], g.checks()
    prov = g.to_prov()
    assert prov["entity"] and prov["wasDerivedFrom"]
    assert json.loads(json.dumps(g.report()))["status"] in ("PASS", "WARNING")
    assert "PRED-006" in g.to_markdown()


def test_trace_of_an_unknown_artifact_refuses():
    with pytest.raises(KeyError):
        Lineage([Artifact(id="A", kind="geometry")]).trace("NOPE")


def test_problem_and_validation_are_real_kinds():
    from pinneapple_data.lineage import KINDS

    g = Lineage([Artifact(id="P", kind="problem"), Artifact(id="V", kind="validation", inputs=["P"])])
    assert g.nodes["P"].kind == "problem" and g.nodes["V"].kind == "validation"
    assert {"problem", "validation"} <= set(KINDS)
    assert g.trace("V")["problem"][0]["id"] == "P"


def test_a_simple_chain_reads_the_new_keys():
    g = Lineage.from_json({"project": "p", "problem": "pipe_flow_3d", "geometry": "CAD-001", "simulation": "SIM-1",
                           "prediction": "PRED-1"})
    assert [a.kind for a in g.nodes.values()][0] == "problem"
    answer = g.trace("PRED-1")
    assert answer["problem"] and answer["geometry"] and answer["simulations"]
    assert answer["missing"] == ["datasets", "models", "validation"], answer["missing"]
