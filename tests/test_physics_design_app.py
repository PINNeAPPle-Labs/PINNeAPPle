"""HTTP smoke test of the Physics Design Optimizer app (apps/physics_design): comparison, verification, design files."""
from __future__ import annotations

import os
import sys
import time

import pytest

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "physics_design"))
pytest.importorskip("fastapi")
pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["PDO_DATA_DIR"] = str(tmp_path_factory.mktemp("pdo"))
    sys.path.insert(0, APP)
    from fastapi.testclient import TestClient
    from physics_design.api import app
    with TestClient(app) as c:
        yield c


def _wait(c, jid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise TimeoutError(jid)


def test_heatsink_study_compares_methods_and_verifies(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Physics Design Optimizer" in client.get("/").text
    meta = client.get("/api/meta").json()
    assert meta["methods"]["heatsink"]["gradient"]["available"]
    body = {"spec": {"preset": "heatsink", "objectives": [{"name": "mass_g"}, {"name": "pressure_drop_pa"}]},
            "methods": ["gradient", "nsga2", "random"], "budget": 60, "seeds": [0], "n_verify": 3}
    j = _wait(client, client.post("/api/studies", json=body).json()["id"])
    assert j["status"] == "done", j.get("error")
    res = j["result"]
    assert {r["method"] for r in res["table"]} == {"gradient", "nsga2", "random"}
    assert all(r["n_evals"] <= 60 for r in res["runs"])
    assert res["winner"] in {"gradient", "nsga2", "random"}
    assert 1 <= len(res["candidates"]) <= 3
    for c in res["candidates"]:
        assert c["verification"] in ("verified", "rejected by reference")
        assert c["found_by"] in {"gradient", "nsga2", "random"}
        assert float(c["n_fins"]).is_integer()                     # relaxed iterates never become candidates
    assert res["verification"]["errors"]["t_source_c"]["max_rel"] < 0.03
    stl = client.get(f"/api/studies/{j['id']}/design/1.stl").content
    n = int.from_bytes(stl[80:84], "little")
    assert len(stl) == 84 + 50 * n and n == 12 * (1 + int(res["candidates"][0]["n_fins"]))
    step = client.get(f"/api/studies/{j['id']}/design/1.step").text
    assert step.startswith("ISO-10303-21;") and step.count("FACETED_BREP(") == 1 + int(res["candidates"][0]["n_fins"])
    sr = client.get(f"/api/studies/{j['id']}/files/study.json").json()
    assert sr["kind"] == "StudyResult" and sr["study_type"] == "design"
    assert client.get(f"/api/studies/{j['id']}/design/99.stl").status_code == 404


def test_airfoil_only_surrogate_and_limits(client):
    body = {"spec": {"preset": "airfoil"}, "methods": ["gradient", "nsga2"], "budget": 40, "seeds": [0]}
    j = _wait(client, client.post("/api/studies", json=body).json()["id"])
    assert j["status"] == "done", j.get("error")
    res = j["result"]
    assert res["verification"] is None
    assert all(c["verification"] == "only surrogate" and c["outline"]["x"] for c in res["candidates"])
    assert client.get("/api/presets/airfoil/baseline.step").text.startswith("ISO-10303-21;")
    assert client.post("/api/studies", json={**body, "methods": ["bayesian"], "budget": 500}).status_code == 422
    assert client.post("/api/studies", json={**body, "methods": ["nope"]}).status_code == 422
