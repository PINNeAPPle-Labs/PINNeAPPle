"""Physics AI Model Gateway app (apps/model_gateway): HTTP smoke test of serving, policies, publishing, metrics."""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sklearn")
pytest.importorskip("pinneapple_core.service")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ADMIN = {"X-Admin-Token": "test-admin"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["MDL_DATA_DIR"] = str(tmp_path_factory.mktemp("mdl"))
    os.environ["MDL_ADMIN_TOKEN"] = "test-admin"
    sys.path.insert(0, str(ROOT / "apps" / "model_gateway"))
    from model_gateway.api import app
    return TestClient(app)


def test_seeded_zoo_models_and_examples(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Model Gateway" in client.get("/").text and client.get("/docs").status_code == 200
    names = {m["name"] for m in client.get("/v1/models").json()["models"]}
    assert {"zoo-airfoil-cst-rans", "zoo-heatsink-forced", "zoo-heatsink-natural"} <= names
    ex = client.get("/api/meta").json()["examples"]
    r = client.post(f"/v1/models/{ex[0]['model']}/predict", json={"inputs": ex[0]["inputs"]}).json()
    assert abs(r["outputs"]["Cl"][0] - 0.6625) < 0.05 and not r["ood"][0]["ood"]          # OpenFOAM: 0.6625
    w = client.post(f"/v1/models/{ex[1]['model']}/predict", json={"inputs": ex[1]["inputs"]}).json()
    assert w["ood"][0]["outside"] == ["alpha_deg"] and w["outputs"]["Cl"][0] is not None   # warn policy
    tight = client.post(f"/v1/models/{ex[1]['model']}/predict", json={"inputs": ex[1]["inputs"], "ood_policy": "refuse"})
    assert tight.json()["outputs"]["Cl"] == [None]
    ref = client.get("/api/models/zoo-airfoil-cst-rans/model_ref.json").json()
    assert ref["kind"] == "ModelRef" and ref["inputs"][-1] == {**ref["inputs"][-1], "name": "alpha_deg", "low": -2.0}


def test_refuse_policy_cannot_be_relaxed(client):
    d = client.get("/v1/models/zoo-heatsink-forced").json()
    assert d["ood_policy"] == "refuse"
    row = {k: v[1] * 10 for k, v in d["validity_bounds"].items()}
    row.update({k: 10.0 for k in d["inputs"] if k not in row})        # inputs bounded via derived quantities
    assert client.post("/v1/models/zoo-heatsink-forced/predict", json={"inputs": row}).json()["outputs"][
        d["outputs"][0]] == [None]
    assert client.post("/v1/models/zoo-heatsink-forced/predict",
                       json={"inputs": row, "ood_policy": "warn"}).status_code == 403


def test_publishing_is_guarded_and_versions_metrics(client):
    body = {"model_ref": {"name": "copy", "uri": "zoo:heatsink_natural_mlp"}, "ood_policy": "refuse"}
    assert client.post("/api/models/ref", json=body).status_code == 403
    v1 = client.post("/api/models/ref", json=body, headers=ADMIN).json()["version"]
    v2 = client.post("/api/models/ref", json={**body, "stage": "staging"}, headers=ADMIN).json()["version"]
    d = client.get("/v1/models/copy").json()
    assert d["versions"] == [v1, v2] and client.get("/v1/models/copy/versions/production").json()["version"] == v1
    assert client.post("/api/models/ref", json={**body, "model_ref": {"name": "x", "uri": "/etc/passwd"}},
                       headers=ADMIN).status_code == 422
    zbad = io.BytesIO(b"not a zip")
    assert client.post("/api/models/surrogate", files={"file": ("m.zip", zbad)}, data={"name": "s"}).status_code == 403
    m = client.get("/v1/metrics").json()
    assert m["calls"] >= 4 and any(r["refused_rows"] >= 1 for r in m["by_version"])
    runs = client.get("/v1/models/zoo-heatsink-forced/runs").json()["runs"]
    assert any(r["error"] for r in runs) and all(r["caller"] for r in runs)


def test_onnx_publish_with_api_key(client, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    net = torch.nn.Linear(2, 1)
    path = tmp_path / "m.onnx"
    torch.onnx.export(net, (torch.zeros(1, 2),), str(path), input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "n"}, "y": {0: "n"}}, dynamo=False)
    key = client.post("/api/keys", json={"name": "ci"}, headers=ADMIN).json()["key"]
    spec = '{"inputs": ["a", "b"], "outputs": ["y"], "validity_bounds": {"a": [0, 1], "b": [0, 1]}}'
    with open(path, "rb") as f:
        r = client.post("/api/models/onnx", files={"file": ("m.onnx", f)}, data={"name": "lin", "spec": spec},
                        headers={"X-API-Key": key})
    assert r.status_code == 200, r.text
    p = client.post("/v1/models/lin/predict", json={"inputs": {"a": [0.5, 2.0], "b": 0.5}}).json()
    with torch.no_grad():
        ref = float(net(torch.tensor([[0.5, 0.5]])))
    assert p["outputs"]["y"][0] == pytest.approx(ref, rel=1e-5) and p["ood"][1]["outside"] == ["a"]


def _surrogate_zip(pkl: bytes) -> bytes:
    import hashlib
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("model/model.pkl", pkl)
        z.writestr("model/card.json", json.dumps({"model_sha256": hashlib.sha256(pkl).hexdigest()}))
    return buf.getvalue()


def test_pickled_surrogate_upload_off_by_default_and_scanned(client, monkeypatch):
    """A model.zip whose card hash matches its own pickle proves nothing: uploads are off unless the operator opts
    in, and then a pickle importing os/subprocess/eval is refused before anything loads it."""
    import pickle

    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("true",))

    z = _surrogate_zip(pickle.dumps(Evil()))
    r = client.post("/api/models/surrogate", files={"file": ("m.zip", z)}, data={"name": "evil"}, headers=ADMIN)
    assert r.status_code == 403 and "MDL_ALLOW_PICKLE_UPLOAD" in r.text
    monkeypatch.setenv("MDL_ALLOW_PICKLE_UPLOAD", "1")
    r = client.post("/api/models/surrogate", files={"file": ("m.zip", z)}, data={"name": "evil"}, headers=ADMIN)
    assert r.status_code == 422 and "refused" in r.text


@pytest.mark.parametrize("uri", ["http://127.0.0.1:8080/v1/models/x", "http://169.254.169.254/latest/meta-data",
                                 "http://localhost/v1/models/x", "file:///etc/passwd"])
def test_remote_ref_to_internal_address_refused(client, uri):
    body = {"model_ref": {"name": "r", "uri": uri}}
    assert client.post("/api/models/ref", json=body, headers=ADMIN).status_code == 422
