"""Reading a catalogued public archive inside a flow (#1891): the table, the refusals, the licence and the gates.

The node is the last step of the loop the stage map opened — stage, then the archives that feed it, then reading one
into something the rest of the flow consumes. What these tests hold is that the shape it produces is the shape the
other data nodes already produce (so nothing downstream needs a second reader), that a refusal happens **before**
anything is fetched, and that the licence is not lost on the way to the artifact.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

import pinneapple_flow as pf


def write_archive(path, *, nt=5, nx=7, n=2, fields=("density",)):
    """A real PDEBench-layout file, small enough to keep the test offline and honest: the adapter reads it."""
    rng = np.random.default_rng(0)
    data = {name: rng.random((n, nt, nx), dtype=np.float32) for name in fields}
    with h5py.File(path, "w") as f:
        for name, arr in data.items():
            f.create_dataset(name, data=arr)
        f.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, nx, dtype=np.float32))
        f.create_dataset("t-coordinate", data=np.linspace(0.0, 1.0, nt, dtype=np.float32))
        f.attrs["case"] = "synthetic 1D"
    return data


def flow_doc(**params):
    return {"schema": "pinneapple-flow/1", "name": "public dataset",
            "nodes": [{"id": "pd", "type": "data.public_dataset", "params": params}], "edges": []}


def failure(result, node="pd") -> str:
    state = result.nodes[node]
    assert state["status"] == "failed", state
    return str(state.get("error") or "")


def test_a_real_archive_becomes_the_table_the_rest_of_the_flow_reads(tmp_path):
    """The acceptance: one real archive through the node, gates passing, in the shape ``data.ingest`` produces."""
    src = write_archive(tmp_path / "Sod1D.hdf5")
    result = pf.run(flow_doc(id="pdebench", arguments={"source": str(tmp_path / "Sod1D.hdf5")}),
                    workdir=tmp_path / "run")
    assert result.status == "done", result.nodes

    data = result.load("pd", "data")
    X, Y = data["X"], data["Y"]
    assert X.shape == (5 * 7, 2) and Y.shape == (5 * 7, 1)
    # spatial first, time last — the order data.observe and data.parametric already use
    assert [str(c) for c in data["coords"]] == ["x", "t"]
    assert [str(f) for f in data["fields"]] == ["density"]
    # the field is laid out on (t, x) and the table is on (x, t): the values must be transposed, not flattened
    mine = Y[:, 0].reshape(7, 5)
    theirs = src["density"][0]
    assert np.allclose(mine, theirs.T)

    metrics = result.report("pd")["metrics"]
    assert metrics["points"] == 35 and metrics["fields"] == 1
    assert metrics["gates_failed"] == 0 and metrics["gates_passed"] >= 1

    manifest = result.load("pd", "manifest")
    assert manifest["dataset"] == "pdebench" and manifest["license"]
    assert manifest["digest"] == result.output("pd", "data")["digest"]
    assert str(data["license"][0]) == manifest["license"]
    assert manifest["gates"]["ok"] is True


def test_the_licence_travels_inside_the_artifact(tmp_path):
    """A licence that only lives in the run report is lost the moment the npz is copied somewhere else."""
    write_archive(tmp_path / "one.hdf5")
    result = pf.run(flow_doc(id="pdebench", arguments={"source": str(tmp_path / "one.hdf5")}),
                    workdir=tmp_path / "run")
    licence = [str(x) for x in result.load("pd", "data")["license"]]
    assert "CC BY 4.0" in licence[0]
    assert "PDEBench" in licence[1], "the citation has to travel with the arrays, not stay in the report"


def test_the_node_feeds_the_same_training_node_an_ingest_does(tmp_path):
    """Acceptance: it runs the same flow downstream as ``data.ingest`` — here the observations go into a PINN."""
    write_archive(tmp_path / "one.hdf5")
    doc = {"schema": "pinneapple-flow/1", "name": "public data into a PINN",
           "nodes": [{"id": "geo", "type": "geometry.domain", "params": {"dimension": 1}},
                     {"id": "eq", "type": "equations.system", "params": {"template": "heat"}},
                     {"id": "pd", "type": "data.public_dataset",
                      "params": {"id": "pdebench", "arguments": {"source": str(tmp_path / "one.hdf5")}}},
                     {"id": "nn", "type": "train.pinn",
                      "params": {"adam_steps": 0, "lbfgs_steps": 0, "n_collocation": 16, "width": 4, "depth": 1}}],
           "edges": [{"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
                     {"source": "eq", "source_port": "problem", "target": "nn", "target_port": "problem"},
                     {"source": "pd", "source_port": "data", "target": "nn", "target_port": "data"}]}
    result = pf.run(doc, workdir=tmp_path / "run")
    assert result.status == "done", result.nodes
    assert result.nodes["nn"]["status"] == "done"


def test_an_archive_without_a_reader_is_refused_before_anything_is_fetched():
    """The refusal names the formats and the homepage instead of failing later inside a download."""
    from pinneapple_data.datasets import catalog as core

    entry = next(e for e in core.CATALOG.values() if e.adapter is None)
    reason = failure(pf.run(flow_doc(id=entry.id), workdir=None))
    assert entry.id in reason and "no reader" in reason
    assert entry.homepage in reason
    assert any(fmt in reason for fmt in entry.formats), entry.formats


def test_a_non_commercial_archive_is_refused_in_commercial_mode(monkeypatch):
    """The same gate the resource catalogue uses, applied where the archive would actually be read."""
    monkeypatch.setenv("PINNEAPPLE_COMMERCIAL_MODE", "1")
    reason = failure(pf.run(flow_doc(id="flowbench", arguments={"x": "", "y": ""})))
    assert "non-commercial" in reason and "CC-BY-NC" in reason
    monkeypatch.delenv("PINNEAPPLE_COMMERCIAL_MODE")
    other = failure(pf.run(flow_doc(id="flowbench", arguments={"x": "", "y": ""})))
    assert "CC-BY-NC" not in other, "outside commercial mode the licence is not what stops it"


def test_the_arguments_the_adapter_insists_on_are_named():
    reason = failure(pf.run(flow_doc(id="pdebench")))
    assert "needs" in reason and "source" in reason


def test_an_unknown_id_says_what_can_be_read():
    reason = failure(pf.run(flow_doc(id="nao_existe")))
    assert "not a catalogued dataset" in reason and "pdebench" in reason


def test_without_an_id_the_stage_map_chooses_and_the_rule_is_recorded(tmp_path, monkeypatch):
    """The other half of #1891: the chooser's decision is in the run record, with the candidates it beat."""
    from pinneapple_data.datasets import read as R

    write_archive(tmp_path / "one.hdf5")
    real = R.choose
    calls = []

    def chooser(stage_id=None, **kw):
        picked = real("data", prefer="pdebench", **kw)      # the real rule, pointed at the archive we can read
        calls.append((stage_id, kw, picked))
        return picked

    monkeypatch.setattr(R, "choose", chooser)
    result = pf.run(flow_doc(id="", stage="data", arguments={"source": str(tmp_path / "one.hdf5")}),
                    workdir=tmp_path / "run")
    assert result.status == "done", result.nodes
    assert calls and calls[0][0] == "data", "the node has to pass its stage to the chooser"
    decisions = result.report("pd")["decisions"]
    chosen = [d for d in decisions if d["decided_by"]["id"] == "data.public_dataset.choose"]
    assert chosen, decisions
    assert chosen[0]["value"]["id"] == "pdebench"
    assert chosen[0]["value"]["candidates"], "a choice without its alternatives cannot be argued with"
    assert chosen[0]["rationale"], "the rule that picked it has to be written down"


def test_the_stage_map_sees_the_node_at_the_data_stage():
    """The loop the map opens: stage, then the archive, then reading it — the node has to land on a stage."""
    from pinneapple_flow.stage_fit import fit, node_stage

    assert node_stage("data.public_dataset") == "data"
    reported = fit([{"type": "data.public_dataset"}])
    assert [s["id"] for s in reported["stages"]] == ["data"]


def test_the_node_is_in_the_data_section_of_the_palette():
    from pinneapple_flow import palette

    entry = palette.entry(pf.REGISTRY.get("data.public_dataset"))
    assert entry["section"] == "data" and entry["curated"]
    assert entry["summary"] and entry["install"] == "", entry


@pytest.mark.skipif(os.environ.get("PINNEAPPLE_NETWORK_TESTS") != "1",
                    reason="set PINNEAPPLE_NETWORK_TESTS=1 for a live archive read")
def test_reads_a_real_pdebench_file_over_http(tmp_path):
    """One live range read: the same path the Studio takes, against DaRUS instead of a fixture."""
    url = "https://darus.uni-stuttgart.de/api/access/datafile/133138"
    result = pf.run(flow_doc(id="pdebench", arguments={"source": url, "index": 0}), workdir=tmp_path / "run")
    assert result.status == "done", result.nodes
    assert result.report("pd")["metrics"]["points"] > 0
