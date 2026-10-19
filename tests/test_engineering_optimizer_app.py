"""HTTP smoke test of the Engineering Design Optimizer app (apps/engineering_optimizer): example, your data, zoo."""
from __future__ import annotations

import io
import os
import sys
import time

import pytest

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "engineering_optimizer"))
pytest.importorskip("fastapi")
pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["EDO_DATA_DIR"] = str(tmp_path_factory.mktemp("edo"))
    sys.path.insert(0, APP)
    from engineering_optimizer.api import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


def _wait(c, jid, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise TimeoutError(jid)


def test_page_meta_and_example(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Engineering Design Optimizer" in client.get("/").text
    meta = client.get("/api/meta").json()
    assert meta["methods"]["nsga2"]["available"] and "mass_g" in meta["heatsink"]["objectives"]
    job = client.post("/api/runs", json={"spec": {"source": "preset", "preset": "heatsink"}, "budget": 400,
                                          "n_verify": 3}).json()
    j = _wait(client, job["id"])
    assert j["status"] == "done", j.get("error")
    res = j["result"]
    assert res["summary"]["n_evals"] == 400 and len(res["candidates"]) == 3
    assert {c["verification"] for c in res["candidates"]} <= {"verified", "rejected by reference"}
    assert res["verification"]["errors"]["t_source_c"]["max_rel"] < 0.03
    assert res["baseline"]["mass_g"] > min(c["mass_g"] for c in res["candidates"])
    sr = client.get(f"/api/runs/{job['id']}/files/study.json").json()
    assert sr["kind"] == "StudyResult" and any(c["name"] == "high_fidelity_verification" for c in sr["checks"])
    assert client.get(f"/api/runs/{job['id']}/files/nope.txt").status_code == 404
    assert client.post("/api/runs", json={"spec": {"preset": "heatsink", "source": "preset", "fidelity": "fvm"}}
                       ).status_code == 422
    assert client.post("/api/runs", json={"spec": {"source": "preset"}, "method": "bayesian", "budget": 900}
                       ).status_code == 422


def test_your_data_then_verify_with_uploaded_results(client):
    import pandas as pd

    from pinneapple_design.design_optimizer import presets
    csv = client.get("/api/examples/heatsink_doe.csv").content
    up = client.post("/api/uploads", files={"file": ("doe.csv", csv, "text/csv")}).json()
    cols = client.post("/api/table/inspect", json={"upload_id": up["id"]}).json()
    assert cols["rows"] == 120
    names = presets.heatsink_problem().space.names
    spec = {"source": "table", "inputs": names, "outputs": ["mass_g", "t_source_c", "pressure_drop_pa"],
            "integer": ["n_fins"], "objectives": [{"name": "mass_g"}, {"name": "pressure_drop_pa"}],
            "constraints": [{"name": "t_source_c", "upper": 85}]}
    job = client.post("/api/runs", json={"spec": spec, "upload_id": up["id"], "budget": 300,
                                          "verify_with": "none"}).json()
    j = _wait(client, job["id"])
    assert j["status"] == "done", j.get("error")
    res = j["result"]
    assert res["fit"]["held_out"]["mass_g"]["r2"] > 0.9
    assert all(c["verification"] == "only surrogate" for c in res["candidates"])
    assert client.get(f"/api/runs/{job['id']}/files/model_ref.json").json()["metrics_split"] == "test"
    cand = pd.read_csv(io.StringIO(client.get(f"/api/runs/{job['id']}/files/candidates_to_run.csv").text))
    Y = presets.HeatsinkEvaluator("closed_form")(cand.to_numpy(float))
    for k in ("mass_g", "t_source_c", "pressure_drop_pa"):
        cand[k] = Y[k]
    up2 = client.post("/api/uploads", files={"file": ("res.csv", cand.to_csv(index=False), "text/csv")}).json()
    v = client.post(f"/api/runs/{job['id']}/verify", json={"upload_id": up2["id"]}).json()
    assert len(v["rows"]) == len(cand)
    assert {r["status"] for r in v["rows"]} <= {"verified", "rejected by reference"}
    assert v["errors"]["mass_g"]["mean_rel"] < 0.2          # a GP of 96 rows, at the edge of its data


def test_zoo_model_source_and_usage(client):
    spec = {"source": "zoo", "model": "airfoil_cst_rans_mlp", "fixed": {"alpha_deg": 4},
            "objectives": [{"name": "Cd"}], "constraints": [{"name": "Cl", "lower": 0.6}]}
    j = _wait(client, client.post("/api/runs", json={"spec": spec, "budget": 200, "verify_with": "none"}).json()["id"])
    assert j["status"] == "done", j.get("error")
    assert j["result"]["summary"]["n_feasible"] > 0
    use = client.get("/api/usage").json()["units"]
    assert use["campaigns"]["used"] >= 1 and use["evaluations"]["used"] >= 200
