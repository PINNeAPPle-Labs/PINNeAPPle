"""The flow document: round trip, validation (types, ports, params, cycles), JSON Schema, catalog."""
import json

import pytest

import pinneapple_flow as pf
from pinneapple_flow.types import compatible


def small(edges=None, nodes=None):
    nodes = nodes or [{"id": "geo", "type": "geometry.domain", "params": {"dimension": 1}},
                      {"id": "eq", "type": "equations.system", "params": {"template": "heat"}},
                      {"id": "fd", "type": "solver.classical", "params": {"method": "fd", "n": 21, "n_t": 5}}]
    edges = edges if edges is not None else [
        {"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
        {"source": "eq", "source_port": "problem", "target": "fd", "target_port": "problem"}]
    return {"schema": "pinneapple-flow/1", "name": "t", "nodes": nodes, "edges": edges}


def errors(doc):
    return [i for i in pf.validate_flow(doc) if i.level == "error"]


def test_round_trip_and_valid():
    f = pf.Flow.from_dict(small())
    assert errors(f) == []
    g = pf.Flow.from_dict(json.loads(f.to_json()))
    assert g.to_dict() == f.to_dict()
    assert f.order() == ["geo", "eq", "fd"]


def test_unknown_type_param_and_bad_values():
    d = small()
    d["nodes"][2]["type"] = "solver.nonexistent"
    assert any("unknown node type" in i.message for i in errors(d))
    d = small()
    d["nodes"][2]["params"]["code"] = "import os"
    assert any("unknown parameter" in i.message for i in errors(d))
    d = small()
    d["nodes"][2]["params"]["method"] = "magic"
    assert any("must be one of" in i.message for i in errors(d))
    d = small()
    d["nodes"][2]["params"]["n"] = 10 ** 9
    assert any("<=" in i.message for i in errors(d))


def test_type_mismatch_missing_input_fan_in_and_cycle():
    d = small(edges=[{"source": "geo", "source_port": "geometry", "target": "fd", "target_port": "problem"},
                     {"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"}])
    assert any("type mismatch" in i.message for i in errors(d))
    d = small(edges=[{"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"}])
    assert any("required input 'problem'" in i.message for i in errors(d))
    d = small()
    d["nodes"].append({"id": "geo2", "type": "geometry.domain", "params": {}})
    d["edges"].append({"source": "geo2", "source_port": "geometry", "target": "eq", "target_port": "geometry"})
    assert any("takes one edge" in i.message for i in errors(d))
    d = small(nodes=[{"id": "a", "type": "viz.plot", "params": {}}, {"id": "b", "type": "viz.plot", "params": {}}],
              edges=[{"source": "a", "source_port": "figure", "target": "b", "target_port": "solution"}])
    assert any("type mismatch" in i.message for i in errors(d))


def test_bad_documents():
    with pytest.raises(pf.FlowError):
        pf.Flow.from_dict({"schema": "pinneapple-flow/99", "name": "x", "nodes": [], "edges": []})
    with pytest.raises(pf.FlowError):
        pf.Flow.from_dict({**small(), "code": "x"})
    d = small()
    d["nodes"][0]["id"] = "../etc"
    assert errors(d)


def test_port_compatibility():
    assert compatible("problem", "problem") and compatible("problem", "geometry")
    assert compatible("solution", "dataset") and compatible("metrics", "any")
    assert not compatible("figure", "solution") and not compatible("model", "problem")
    assert not compatible("nope", "any")


def test_catalog_and_json_schema():
    cat = pf.catalog()
    types = {c["type"] for c in cat}
    for t in ("geometry.domain", "equations.system", "solver.classical", "train.pinn", "surrogate.train",
              "surrogate.hpo", "inverse.discover", "inverse.calibrate", "validate.compare", "validate.lab",
              "twin.connect", "forecast.lab", "viz.field3d", "viz.render", "export.report", "uq.sensitivity",
              "train.finetune", "data.ingest", "flight.takeoff"):
        assert t in types
    for c in cat:
        assert c["params_schema"]["type"] == "object" and c["calls"]
    s = pf.flow_json_schema()
    assert s["properties"]["schema"]["const"] == "pinneapple-flow/1"
    assert set(s["required"]) == {"schema", "name", "nodes", "edges"}


def test_flight_node_degrades_gracefully():
    spec = pf.REGISTRY["flight.takeoff"]
    ok, why = spec.available()
    import importlib.util
    assert ok == (importlib.util.find_spec("pinneapple_simulation.flight") is not None)
    if not ok:
        assert "flight" in why
