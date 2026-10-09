"""Engineering Model Lineage: graph checks, auto-detection from real project folders, PROV export, API."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

from pinneapple_data.lineage import Lineage

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "model_lineage"))


def _status(g, kind):
    return [c for c in g.checks() if c["status"] == kind]


def test_simple_chain_and_queries():
    g = Lineage.from_json({"project": "p", "geometry": "CAD-001", "mesh": "MESH-014", "solver": "OpenFOAM v2312",
                           "simulation": "SIM-382", "model": "MODEL-07", "result": "PRED-883"})
    assert [a.kind for a in g.nodes.values()][:2] == ["geometry", "mesh"]
    assert "CAD-001" in g.upstream("PRED-883") and "PRED-883" in g.downstream("CAD-001")
    assert g.nodes["SIM-382"].software.startswith("OpenFOAM")
    assert not _status(g, "fail")


def test_hash_mismatch_stale_missing_cycle():
    good = hashlib.sha256(b"geometry rev A").hexdigest()
    g = Lineage.from_json({"artifacts": [
        {"id": "CAD", "kind": "geometry", "file": "cad.step", "hash": good, "timestamp": "2026-01-01T00:00:00Z"},
        {"id": "MESH", "kind": "mesh", "inputs": ["CAD"], "timestamp": "2026-01-02T00:00:00Z"},
        {"id": "SIM", "kind": "simulation", "inputs": ["MESH", "GHOST"], "timestamp": "2026-01-01T12:00:00Z"}]})
    g.nodes["CAD"].actual_hash = hashlib.sha256(b"geometry rev B").hexdigest()
    codes = {c["rule"] for c in g.checks()}
    assert {"hash_mismatch", "missing_input", "stale"} <= codes
    c = Lineage.from_json({"artifacts": [{"id": "A", "kind": "mesh", "inputs": ["B"]}, {"id": "B", "kind": "mesh", "inputs": ["A"]}]})
    assert "cycle" in {x["rule"] for x in c.checks()}


def test_prov_json_is_w3c_shaped():
    g = Lineage.from_json({"geometry": "CAD-001", "mesh": "MESH-014", "simulation": "SIM-382"})
    p = g.to_prov()
    assert {"prefix", "entity", "wasDerivedFrom"} <= set(p)
    assert any(v["prov:usedEntity"].endswith("CAD-001") for v in p["wasDerivedFrom"].values())


@pytest.fixture(scope="module")
def client():
    sys.path.insert(0, APP)
    from fastapi.testclient import TestClient
    from model_lineage.api import app
    with TestClient(app) as c:
        yield c


def test_examples_through_the_api(client):
    want = {"openfoam_pitzDaily_study.zip": "FAIL", "pinneapple_pinn_plate.zip": "PASS",
            "illustrative_digital_thread.json": "WARNING", "simple_chain.json": "PASS"}
    for name, status in want.items():
        j = client.get(f"/api/example/{name}").json()
        assert j["status"] == status, (name, j["status"], [c["title"] for c in j["checks"] if c["status"] != "info"])
        assert j["prov"]["entity"] and j["markdown"].startswith("#")
    pitz = client.get("/api/example/openfoam_pitzDaily_study.zip").json()
    fail = [c for c in pitz["checks"] if c["status"] == "fail"]
    assert len(fail) == 1 and "blockMeshDict" in fail[0]["title"]
    sims = [n for n in pitz["nodes"] if n["kind"] == "simulation"]
    assert len(sims) == 3 and all(n["version"] == "v1912" for n in sims)
    pinn = client.get("/api/example/pinneapple_pinn_plate.zip").json()
    assert pinn["summary"]["verified"] == pinn["summary"]["artifacts"] == 7


def test_upload_lineage_json(client):
    body = json.dumps({"geometry": "CAD-1", "mesh": "MESH-1", "simulation": "SIM-1"}).encode()
    j = client.post("/api/lineage", files=[("files", ("lineage.json", body, "application/json"))]).json()
    assert j["summary"]["artifacts"] == 3 and len(j["edges"]) == 2
