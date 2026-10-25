"""GovernedCard app (apps/governed_card): the source app's tests/test_app.py through the job queue (a card's metrics
equal an identically seeded direct guardrail run; audits are schema-only and say so), plus weight uploads."""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED, EPOCHS, NCOL = 1234, 40, 256


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["GCARD_DATA_DIR"] = str(tmp_path_factory.mktemp("gcard"))
    sys.path.insert(0, str(ROOT / "apps" / "governed_card"))
    from governed_card.api import app
    return TestClient(app)


def _wait(c, job, timeout=600):
    t0 = time.time()
    while job["status"] not in ("done", "failed") and time.time() - t0 < timeout:
        time.sleep(0.3)
        job = c.get(f"/v1/compliance_report/{job['id']}").json()
    return job


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="slow: two trainings (set PINNEAPPLE_SLOW=1)")
def test_card_metrics_match_a_direct_guardrail_run(client):
    from pinneapple_hub.verified_card import certify_model

    job = _wait(client, client.post("/v1/generate_verified_card", data={
        "preset": "laplace_2d", "name": "laplace-card", "train_epochs": str(EPOCHS), "n_collocation": str(NCOL),
        "seed": str(SEED)}).json())
    assert job["status"] == "done", job.get("error")
    r = job["result"]
    assert r["issued"] and r["model_card"]["validation_metrics"] and r["model_card"]["reference_source"]
    direct = certify_model("laplace_2d", "direct", train_epochs=EPOCHS, n_collocation=NCOL, seed=SEED)["card"]
    assert r["model_card"]["validation_metrics"] == direct.validation_metrics


def test_uploaded_weights_and_bad_inputs(client):
    import pinneapple_neural.architectures  # noqa: F401
    from pinneapple_neural.architectures.registry import ModelRegistry

    buf = io.BytesIO()
    torch.save(ModelRegistry.build("modified_mlp", in_dim=2, out_dim=1, hidden_dim=32, n_layers=3).state_dict(), buf)
    job = _wait(client, client.post("/v1/generate_verified_card", data={"preset": "laplace_2d", "name": "up"},
                                    files={"state_dict_file": ("w.pt", buf.getvalue(), "application/octet-stream")}).json())
    assert job["status"] == "done" and job["result"]["trained_live"] is False
    assert client.post("/v1/generate_verified_card", data={"preset": "nope", "name": "x"}).status_code == 422
    assert client.post("/v1/generate_verified_card", data={"preset": "laplace_2d", "name": "x",
                                                           "preset_kwargs_json": "[1]"}).status_code == 422


def test_audit_is_schema_only(client):
    bad = client.post("/v1/audit_card", json={"name": "broken", "architecture": "modified_mlp",
                                              "validation_metrics": {}, "reference_source": ""}).json()
    assert bad["schema_valid"] is False and any("validation_metrics is empty" in p for p in bad["schema_problems"])
    assert "cannot re-execute" in bad["limitation"]
    good = client.post("/v1/audit_card", json={"name": "good", "architecture": "modified_mlp",
                                               "validation_metrics": {"pde_residual": 0.001},
                                               "reference_source": "supplied for this test"}).json()
    assert good["schema_valid"] is True and good["schema_problems"] == []
    assert client.post("/v1/audit_card", json={"not_a_field": 1}).json()["schema_valid"] is False
