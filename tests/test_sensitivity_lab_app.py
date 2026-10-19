"""Engineering Sensitivity Lab (apps/sensitivity_lab): HTTP smoke test of the API on its examples and on the real job
path (shared runtime: queue, uploads, metering), with the shared contracts in and out."""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from pinneapple_analysis.sensitivity import sobol_indices  # noqa: E402
from pinneapple_analysis.sensitivity.testfunctions import (  # noqa: E402
    ishigami,
    ishigami_indices,
    ishigami_space,
)

APP = Path(__file__).resolve().parents[1] / "apps" / "sensitivity_lab"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["SEN_DATA_DIR"] = str(tmp_path_factory.mktemp("sensitivity"))
    os.environ["SEN_FREE_EVALUATIONS"] = "100000"          # the heat-sink example at N = 8192 needs ~181k
    sys.path.insert(0, str(APP))
    api = importlib.import_module("sensitivity_lab.api")
    api.rt.jobs.poll_interval = 0.02
    with TestClient(api.app) as c:
        yield c
    api.rt.jobs.stop()
    for k in ("SEN_DATA_DIR", "SEN_FREE_EVALUATIONS"):
        os.environ.pop(k, None)


def _done(c, job, timeout=300):
    t0 = time.time()
    while job["status"] not in ("done", "failed", "cancelled") and time.time() - t0 < timeout:
        time.sleep(0.05)
        job = c.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done", job.get("error")
    return job["result"]


def test_health_meta_model_info_and_cost(client):
    assert client.get("/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert {e["name"] for e in meta["examples"]} == {"ishigami", "heatsink", "dataset"}
    assert set(meta["runtime"]["job_kinds"]) == {"study", "runs", "given_data"}
    info = client.get("/api/model/zoo/heatsink_forced_mlp").json()
    assert info["differentiable"] and len(info["space"]) + len(info["fixed"]) == 12
    assert info["parameter_space"]["kind"] == "ParameterSpace"
    cfg = client.get("/api/example/ishigami/config").json()
    cost = client.post("/api/cost", json=cfg).json()
    assert cost["parts"]["sobol"] == 2048 * 8 and cost["total"] > cost["parts"]["sobol"]
    assert client.get("/").status_code == 200


def test_ishigami_example_matches_the_exact_indices(client):
    j = client.get("/api/example/ishigami").json()
    assert j["exact"]["max_abs_error_S1"] < 0.02 and j["exact"]["max_abs_error_ST"] < 0.02
    assert all(j["exact"]["S1_in_ci"]) and all(j["exact"]["ST_in_ci"])
    assert j["sobol"]["ranking"]["y"] == ["x1", "x2", "x3"]
    sr = j["study_result"]
    assert sr["kind"] == "StudyResult" and sr["study_type"] == "sensitivity"
    assert [c["status"] for c in sr["checks"]] == ["PASS"]


def test_heatsink_example_freezes_ambient_and_power_with_a_check(client):
    j = client.get("/api/example/heatsink").json()
    dec = {r["name"]: r["decision"] for r in j["freeze"]["rows"]}
    assert dec["t_ambient_c"] == dec["power_w"] == "freeze" and dec["n_fins"] == "keep"
    assert all(abs(v - 1) < 0.01 for v in j["freeze"]["check"]["variance_ratio"])
    assert "automatic" in j["derivatives"]["method"]
    assert j["study_result"]["models"][0]["uri"] == "zoo:heatsink_forced_mlp"


def test_dataset_example_reports_input_correlation(client):
    j = client.get("/api/example/dataset").json()
    assert j["n_rows"] == 3900 and abs(j["max_input_correlation"]["r"]) > 0.5
    assert j["study_result"]["checks"][0]["status"] == "WARN"


def test_study_job_with_contracts_in_and_out(client):
    from pinneapple_data.contracts import ModelRef, Parameter, ParameterSpace, StudyResult
    ps = ParameterSpace(name="ishigami", parameters=[Parameter(name=f"x{i}", low=-np.pi, high=np.pi)
                                                     for i in (1, 2, 3)])
    cfg = {"model": {"source": "builtin", "name": "ishigami"}, "parameter_space": ps.to_dict(), "n_sobol": 512,
           "morris": False, "effects": False}
    job = client.post("/api/study", data={"config": json.dumps(cfg)}).json()
    assert job["kind"] == "study" and job["status"] in ("queued", "running", "done")
    res = _done(client, job)
    StudyResult.model_validate(res["study_result"])
    np.testing.assert_allclose(res["sobol"]["ST"][0], ishigami_indices()["ST"], atol=0.12)
    # a ModelRef for a zoo model, a ParameterSpace with a constant
    ref = ModelRef(name="hs", uri="zoo:heatsink_forced_mlp", outputs=[]).to_dict()
    info = client.get("/api/model/zoo/heatsink_forced_mlp").json()
    cfg2 = {"model_ref": ref, "parameter_space": info["parameter_space"], "n_sobol": 256, "morris": False,
            "effects": False, "outputs": ["r_base_k_w"]}
    res2 = _done(client, client.post("/api/study?wait=120", data={"config": json.dumps(cfg2)}).json())
    assert res2["sobol"]["outputs"] == ["r_base_k_w"] and res2["model"]["kind"] == "zoo"
    usage = client.get("/api/usage").json()
    assert "evaluations" in json.dumps(usage)


def test_design_then_external_runs_round_trip(client):
    cfg = client.get("/api/example/ishigami/config").json()
    cfg["n_sobol"], cfg["second_order"] = 512, False
    r = client.post("/api/design", json=cfg)
    assert json.loads(r.headers["x-parameter-space"])["kind"] == "ParameterSpace"
    lines = r.text.splitlines()
    spec = lines[0].split(" ", 2)[2]
    X = np.array([[float(v) for v in ln.split(",")[1:]] for ln in lines[2:]])
    ids = np.random.default_rng(0).permutation(len(X))           # runs come back in any order
    csv = "run_id,y\n" + "\n".join(f"{i},{ishigami(X[i:i + 1])[0]}" for i in ids)
    job = client.post("/api/runs?wait=120", data={"spec": spec}, files={"runs": ("results.csv", csv.encode())}).json()
    res = _done(client, job)
    direct = sobol_indices(ishigami, ishigami_space(), n=512, n_boot=0)        # same design, run here
    np.testing.assert_allclose(res["sobol"]["ST"][0], direct.ST[0], rtol=1e-6)
    assert res["study_result"]["checks"] == [] or res["study_result"]["checks"][0]["status"] == "NOT_RUN"
    short = "run_id,y\n" + "\n".join(csv.splitlines()[1:-5])
    bad = client.post("/api/runs?wait=60", data={"spec": spec}, files={"runs": ("r.csv", short.encode())}).json()
    assert bad["status"] == "failed" and "missing" in (bad.get("error") or "")


def test_given_data_upload_and_errors(client):
    rng = np.random.default_rng(1)
    X = rng.uniform(-np.pi, np.pi, (3000, 3))
    csv = "x1,x2,x3,y\n" + "\n".join(",".join(f"{v:.6f}" for v in list(x) + [ishigami(x[None])[0]]) for x in X)
    up = client.post("/api/uploads", files={"file": ("d.csv", csv.encode())}).json()
    job = client.post("/api/given-data?wait=120",
                      data={"config": json.dumps({"outputs": ["y"], "n_boot": 20, "data_upload": up["id"]})}).json()
    res = _done(client, job)
    np.testing.assert_allclose(res["given_data"]["y"]["S1"], ishigami_indices()["S1"], atol=0.06)
    big = client.get("/api/example/heatsink/config").json()
    big["n_sobol"] = 2 ** 16
    assert client.post("/api/study", data={"config": json.dumps(big)}).status_code == 413
    assert client.post("/api/study", data={"config": json.dumps({"model": {"source": "builtin", "name": "ishigami"}})}
                       ).status_code == 422
    assert client.post("/api/study", data={"config": "{nope"}).status_code == 422
    cat = {"model": {"source": "builtin", "name": "ishigami"}, "parameter_space": {
        "kind": "ParameterSpace", "name": "s", "parameters": [{"name": "x1", "kind": "categorical", "values": ["a"]}]}}
    r = client.post("/api/study", data={"config": json.dumps(cat)})
    assert r.status_code == 422 and "categorical" in r.json()["detail"]
    # the free limit is enforced before the job is queued
    over = {**client.get("/api/example/heatsink/config").json(), "n_sobol": 8192, "effects": False}
    assert client.post("/api/study", data={"config": json.dumps(over)}).status_code == 429
