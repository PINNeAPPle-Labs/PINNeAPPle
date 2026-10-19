"""Engineering Uncertainty Lab (apps/uncertainty_lab): HTTP smoke test of the API on its examples and on the real job
path (shared runtime: queue, uploads, metering), with the shared contracts in and out."""
from __future__ import annotations

import importlib
import io
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

APP = Path(__file__).resolve().parents[1] / "apps" / "uncertainty_lab"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["UQL_DATA_DIR"] = str(tmp_path_factory.mktemp("uncertainty"))
    sys.path.insert(0, str(APP))
    api = importlib.import_module("uncertainty_lab.api")
    api.rt.jobs.poll_interval = 0.02
    with TestClient(api.app) as c:
        yield c
    api.rt.jobs.stop()
    os.environ.pop("UQL_DATA_DIR", None)


def _done(c, job, timeout=300):
    t0 = time.time()
    while job["status"] not in ("done", "failed", "cancelled") and time.time() - t0 < timeout:
        time.sleep(0.05)
        job = c.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done", job.get("error")
    return job["result"]


def test_health_meta_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert {e["name"] for e in meta["examples"]} == {"heatsink", "airfoil"}
    info = client.get("/api/model/heatsink_forced_mlp").json()
    assert info["differentiable"] and len(info["nominal"]) == 12
    assert client.get("/api/model/nope").status_code == 404
    assert client.get("/").status_code == 200


def test_heatsink_example_measures_coverage_and_checks_the_routing_with_the_engine(client):
    r = client.get("/api/example/heatsink")
    assert r.status_code == 200, r.text
    j = r.json()
    c = j["conformal"]
    assert c["n_cal"] == 600 and c["n_test"] == 900 and c["meets_target"]
    rows = {x["name"]: x for x in j["cases"]["rows"]}
    assert rows["8 m/s fan (beyond range)"]["status"] == "outside"
    assert rows["8 m/s fan (beyond range)"]["route"] == "high_fidelity"
    assert "straddles_limit" in rows["fan derated to 2.3 m/s"]["codes"]
    assert rows["nominal design"]["route"] == "surrogate" and rows["nominal design"]["verdict"] == "meets limits"
    assert j["cases"]["high_fidelity"]["n_run"] == len(rows)
    assert all(x["high_fidelity"] is not None for x in rows.values())
    p = j["propagation"]
    assert p["n"] == 4096 and 0 <= p["p_within_with_model_error"] <= p["p_within"] <= 1
    assert j["usage"]["high_fidelity_runs"] == len(rows) and j["usage"]["model_evaluations"] > 4096
    sr = j["study_result"]
    assert sr["kind"] == "StudyResult" and sr["study_type"] == "uncertainty"
    assert sr["checks"][0]["status"] == "PASS" and sr["models"][0]["uri"] == "zoo:heatsink_forced_mlp"


def test_airfoil_example_compares_the_raw_ensemble_interval(client):
    j = client.get("/api/example/airfoil").json()
    assert j["conformal"]["n_cal"] == 36 and j["conformal"]["n_test"] == 60 and j["scale"] == "ensemble"
    assert j["ensemble"]["coverage"] < j["conformal"]["coverage"]
    assert "mean_abs_pct" not in j["accuracy"]          # Cl crosses zero: no percent errors
    assert j["cases"]["high_fidelity"] is None


def test_onnx_upload_with_reference_table_and_case_file(client):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from pinneapple_hub import zoo
    from pinneapple_hub.zoo.export import export_bytes
    graph = export_bytes(zoo.load("heatsink_forced_mlp"))
    ref = client.get("/api/template/heatsink_forced_mlp.csv").text
    header = ref.splitlines()[0].split(",")
    names = header[:12]
    cases = "name," + ",".join(names) + "\n" + "upload case," + ",".join(ref.splitlines()[1].split(",")[:12]) + "\n"
    cfg = {"model": {"source": "onnx", "input_names": names, "output_names": ["r_base_k_w", "pressure_drop_pa", "r95"]},
           "output": "r_base_k_w", "alpha": 0.1, "scale": "relative", "limits": [None, 0.5]}
    r = client.post("/api/study?wait=120", data={"config": json.dumps(cfg)},
                    files={"model": ("hs.onnx", graph), "reference": ("ref.csv", ref.encode()),
                           "cases": ("cases.csv", cases.encode())})
    assert r.status_code == 200, r.text
    j = _done(client, r.json())
    assert j["conformal"]["n_cal"] == 600 and j["conformal"]["n_test"] == 900       # the split column was used
    assert j["model"]["kind"] == "onnx" and j["cases"]["rows"][0]["name"] == "upload case"
    assert abs(j["conformal"]["coverage"] - 0.9) < 0.05


def test_errors_are_reported_not_raised(client):
    bad = client.post("/api/study", data={"config": "{not json"})
    assert bad.status_code == 422
    r = client.post("/api/study", data={"config": json.dumps({"model": {"source": "zoo", "name": "nope"}})})
    assert r.status_code == 422 and "Zoo model" in r.json()["detail"]
    r = client.post("/api/study", data={"config": json.dumps({"model": {"source": "zoo", "name": "heatsink_forced_mlp"},
                                                             "scale": "ensemble"})})
    assert r.status_code == 422 and "ensemble" in r.json()["detail"]
    r = client.post("/api/study", data={"config": json.dumps({"model": {"source": "onnx"}})})
    assert r.status_code == 422


def test_template_is_a_valid_reference_table(client):
    import pandas as pd
    df = pd.read_csv(io.StringIO(client.get("/api/template/airfoil_cst_rans_mlp.csv").text))
    assert set(df["split"]) == {"cal", "test"} and np.isfinite(df["Cl"]).all()


def test_contracts_in_and_metering(client):
    from pinneapple_data.contracts import Parameter, ParameterSpace, StudyResult
    ref = client.get("/api/model/heatsink_forced_mlp").json()["model_ref"]
    assert ref["kind"] == "ModelRef" and ref["uri"] == "zoo:heatsink_forced_mlp"
    nominal = client.get("/api/example/heatsink/config").json()["nominal"]
    ps = ParameterSpace(name="operating", parameters=[Parameter(name="air_velocity_m_s", low=2.0, high=3.0),
                                                      Parameter(name="t_ambient_c", low=20, high=40)],
                        metadata={"distributions": {"air_velocity_m_s": {"dist": "normal", "mean": 2.5, "std": 0.2}}})
    cfg = {"model_ref": ref, "output": "r_base_k_w", "scale": "relative", "nominal": nominal,
           "parameter_space": ps.to_dict(), "n_propagation": 1024, "limits": [None, 0.42],
           "cases": [{"name": "nominal", **nominal}]}
    res = _done(client, client.post("/api/study?wait=120", data={"config": json.dumps(cfg)}).json())
    sr = StudyResult.model_validate(res["study_result"])
    assert sr.parameter_space.parameters[0].name == "air_velocity_m_s"
    assert res["propagation"]["uncertain"][0]["dist"] == "normal" and res["propagation"]["n"] == 1024
    units = client.get("/api/usage").json()["units"]
    assert units["studies"]["used"] >= 2 and units["propagation_samples"]["used"] >= 1024


def test_a_case_without_numbers_is_refused_before_queueing(client):
    cfg = client.get("/api/example/heatsink/config").json()
    cfg["cases"][0]["base_width_mm"] = None             # what a mis-split CSV row used to send
    r = client.post("/api/study", data={"config": json.dumps(cfg)})
    assert r.status_code == 422 and "no number for base_width_mm" in r.json()["detail"]
