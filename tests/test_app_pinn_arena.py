"""PINNeAPPle Arena app (apps/pinn_arena): the source repository's integration tests (backend/tests/
test_integration.py), ported, plus the behaviour changed in the port: no stand-in model trained under another
architecture's name, limits, no unauthenticated reset or pickle upload, uploads, the token header, the external
catalog routes."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("jose")
pytest.importorskip("httpx")
torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Minimal config: 2 epochs, 100 collocation points, 1 architecture. Background tasks in TestClient run
# synchronously, so the experiment completes during the POST request itself.
_FAST_CONFIG = {
    "config": {
        "problem_id": "burgers_1d",
        "architecture_ids": ["vanilla_pinn"],
        "n_train_col": 100,
        "n_train_bc": 20,
        "n_train_ic": 20,
        "epochs": 2,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "loss_weights": {"pde": 1.0, "bc": 10.0, "ic": 10.0, "data": 1.0},
        "custom_parameters": {},
    }
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["PINN_ARENA_DATA_DIR"] = str(tmp_path_factory.mktemp("pinn_arena"))
    os.environ.setdefault("PINN_ARENA_FREE_EXPERIMENTS", "unlimited")
    for p in (ROOT / "apps" / "pinn_arena", ROOT / "apps" / "_shared"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from pinn_arena.api import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_experiment_store():
    """Clear the in-memory experiment cache after every test (as the source repository's conftest did)."""
    yield
    try:
        from pinn_arena.store import ExperimentStore
        ExperimentStore._experiments.clear()
    except ImportError:
        pass


# ------------------------------------------------------------------ ported from backend/tests/test_integration.py

def test_health_endpoint(client):
    assert client.get("/health").json()["status"] == "ok"
    data = client.get("/api/health").json()
    assert data["status"] == "ok" and "api_version" in data and data["torch_available"] is True


def test_list_problems_returns_catalog(client):
    data = client.get("/api/problems").json()
    assert data["total"] == 34 == len(data["problems"])
    for key in ("id", "name", "domain", "dimension"):
        assert key in data["problems"][0]


def test_get_problem_by_id(client):
    data = client.get("/api/problems/burgers_1d").json()
    assert data["id"] == "burgers_1d" and data["equations"]


def test_unknown_problem_returns_404(client):
    assert client.get("/api/problems/nonexistent_problem_xyz_000").status_code == 404


def test_dataset_preview_returns_coords(client):
    data = client.get("/api/problems/burgers_1d/dataset-preview?n_samples=50").json()
    assert data["coords"]


def test_list_architectures_returns_catalog(client):
    data = client.get("/api/architectures").json()
    assert data["total"] == 93 == len(data["architectures"])
    first = data["architectures"][0]
    assert {"id", "name", "family", "registered"} <= set(first)
    reg = {a["id"]: a["registered"] for a in data["architectures"]}
    assert reg["vanilla_pinn"] and reg["symm_upt"]          # a renamed entry resolves to its registry key


def test_create_experiment_appears_in_list(client):
    exp_id = client.post("/api/experiments", json=_FAST_CONFIG).json()["experiment_id"]
    ids = [e["experiment_id"] for e in client.get("/api/experiments").json()["experiments"]]
    assert exp_id in ids


def test_experiment_status_endpoint_shape(client):
    exp_id = client.post("/api/experiments", json=_FAST_CONFIG).json()["experiment_id"]
    data = client.get(f"/api/experiments/{exp_id}/status").json()
    assert data["experiment_id"] == exp_id
    assert data["status"] in ("PENDING", "RUNNING", "COMPLETED", "FAILED")
    assert 0.0 <= data["progress"] <= 1.0
    assert isinstance(data["model_statuses"], dict) and isinstance(data["logs"], list)


def test_cancel_pending_experiment(client):
    from pinn_arena.store import ExperimentStore
    exp_id = ExperimentStore.create({"problem_id": "burgers_1d", "architectures": ["vanilla_pinn"],
                                     "training": {"epochs": 1, "n_col": 10, "n_bc": 10, "n_ic": 0}})
    assert client.post(f"/api/experiments/{exp_id}/cancel").status_code == 200
    data = client.get(f"/api/experiments/{exp_id}/status").json()
    assert data["status"] == "FAILED" and "Cancelled" in data["message"]
    assert ExperimentStore.is_cancelled(exp_id)


def test_full_lifecycle_create_results_delete(client):
    exp_id = client.post("/api/experiments", json=_FAST_CONFIG).json()["experiment_id"]
    status = client.get(f"/api/experiments/{exp_id}/status").json()["status"]
    assert status == "COMPLETED"
    results = client.get(f"/api/experiments/{exp_id}/results").json()
    assert results["experiment_id"] == exp_id and results["problem_id"] == "burgers_1d"
    assert len(results["model_results"]) == 1 and results["ranking"] == ["vanilla_pinn"]
    m = results["model_results"][0]
    for key in ("l2_relative_error", "rmse", "r2_score"):
        assert key in m["metrics"]
    assert m["physics_loss_enabled"] is True and m["n_pde_evaluations"] > 0   # the residual really ran
    assert results["comparison_summary"]["pinneapple_version"]
    assert client.delete(f"/api/experiments/{exp_id}").status_code == 200
    assert client.get(f"/api/experiments/{exp_id}/status").status_code == 404


def test_leaderboard_returns_required_keys(client):
    data = client.get("/api/leaderboard").json()
    for key in ("total_experiments", "total_architectures", "problems_covered", "entries"):
        assert key in data
    assert isinstance(data["entries"], list)


def test_leaderboard_entries_have_required_fields(client):
    client.post("/api/experiments", json=_FAST_CONFIG)
    entries = client.get("/api/leaderboard").json()["entries"]
    assert entries
    for entry in entries:
        for field in ("rank", "architecture_id", "architecture_name", "family", "n_experiments", "win_rate",
                      "convergence_rate", "scores"):
            assert field in entry


def test_leaderboard_domain_filter_returns_valid_structure(client):
    data = client.get("/api/leaderboard", params={"domain": "FLUID_MECHANICS"}).json()
    assert isinstance(data["entries"], list)


def test_leaderboard_unknown_domain_returns_empty_entries(client):
    data = client.get("/api/leaderboard", params={"domain": "DOES_NOT_EXIST_XYZ"}).json()
    assert data["entries"] == [] and data["total_experiments"] == 0


def test_auth_me_without_token_returns_401(client):
    assert client.get("/api/auth/me").status_code == 401


def test_auth_me_with_invalid_token_returns_401(client):
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer this.is.not.a.real.jwt"}).status_code == 401
    assert client.get("/api/auth/me", headers={"X-Arena-Token": "this.is.not.a.real.jwt"}).status_code == 401


def test_experiments_list_unauthenticated_returns_empty(client):
    assert client.get("/api/experiments").json()["experiments"] == []


def test_problems_catalog_each_entry_has_required_fields(client):
    for prob in client.get("/api/problems").json()["problems"]:
        for field in ("id", "name", "domain", "dimension", "difficulty", "equations"):
            assert field in prob


def test_architectures_catalog_total_matches_list_length(client):
    data = client.get("/api/architectures").json()
    assert data["total"] == len(data["architectures"])


# ------------------------------------------------------------------------------------- changed in the port

def test_unbuildable_architecture_fails_instead_of_training_a_stand_in(client):
    """The source pipeline trained a fallback MLP under the name of any architecture pp.build_model could not
    build; here it is reported as failed, with the reason, and ranked after the real ones."""
    cfg = {"config": {**_FAST_CONFIG["config"], "architecture_ids": ["vanilla_pinn", "lstm", "koopman"]}}
    exp_id = client.post("/api/experiments", json=cfg).json()["experiment_id"]
    res = client.get(f"/api/experiments/{exp_id}/results").json()
    by_id = {m["architecture_id"]: m for m in res["model_results"]}
    assert by_id["vanilla_pinn"]["status"] == "completed"
    assert by_id["lstm"]["status"] == "failed" and "pp.build_model('lstm'" in by_id["lstm"]["error"]
    assert by_id["koopman"]["status"] == "failed" and "Not applicable" in by_id["koopman"]["error"]
    assert res["ranking"][0] == "vanilla_pinn"


def test_limits_and_validation(client):
    def post(**over):
        return client.post("/api/experiments", json={"config": {**_FAST_CONFIG["config"], **over}})
    assert post(problem_id="nope").status_code == 404
    assert post(architecture_ids=["not_an_architecture"]).status_code == 422
    assert post(epochs=10**7).status_code == 422
    assert post(architecture_ids=["vanilla_pinn"] * 50).status_code == 422


def test_reset_and_pickle_upload_routes_are_gone(client):
    assert client.post("/api/experiments/reset").status_code in (404, 405)
    files = {"file": ("m.pt", b"not a model", "application/octet-stream")}
    assert client.post("/api/architectures/upload", files=files).status_code in (404, 405, 422)


def test_uploaded_dataset_is_used_or_the_experiment_fails(client):
    import numpy as np
    x = np.linspace(-1, 1, 400)
    t = np.tile(np.linspace(0, 1, 20), 20)
    u = np.sin(np.pi * x) * np.exp(-t)
    csv = "x,t,u\n" + "\n".join(f"{a},{b},{c}" for a, b, c in zip(x, t, u, strict=True))
    meta = client.post("/api/datasets/upload", files={"file": ("d.csv", csv.encode(), "text/csv")}).json()
    assert meta["n_rows"] == 400 and meta["columns"] == ["x", "t", "u"] and meta["n_skipped_rows"] == 0
    cfg = {"config": {**_FAST_CONFIG["config"], "dataset_id": meta["dataset_id"],
                      "column_mapping": {"coord_cols": ["x", "t"], "output_cols": ["u"]}}}
    exp_id = client.post("/api/experiments", json=cfg).json()["experiment_id"]
    assert client.get(f"/api/experiments/{exp_id}/status").json()["status"] == "COMPLETED"
    # a dataset that does not exist fails the experiment (the source silently switched to generated data)
    bad = {"config": {**cfg["config"], "dataset_id": "00000000-0000-0000-0000-000000000000"}}
    exp_id = client.post("/api/experiments", json=bad).json()["experiment_id"]
    st = client.get(f"/api/experiments/{exp_id}/status").json()
    assert st["status"] == "FAILED" and "could not be used" in st["message"]
    assert client.get("/api/datasets/not-a-dataset-id").status_code == 404     # ids are UUIDs, nothing else is read


def test_weights_download(client):
    exp_id = client.post("/api/experiments", json=_FAST_CONFIG).json()["experiment_id"]
    r = client.get(f"/api/experiments/{exp_id}/models/vanilla_pinn/download")
    assert r.status_code == 200 and len(r.content) > 100
    state = torch.load(io.BytesIO(r.content), map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and state


def test_external_catalog_routes(client):
    models = client.get("/api/external-catalog").json()
    items = models["models"] if isinstance(models, dict) else models
    assert len(items) == 95
    deepxde = client.get("/api/external-catalog/models/deepxde").json()
    assert deepxde["license"] == "LGPL-2.1" and deepxde["license_source"]
    fx = client.get("/api/external-catalog/models/fluidx3d").json()
    assert fx["non_commercial"] is True and fx["install_state"] == "manual_only"
    job = client.post("/api/external-catalog/models/neuralpde-jl/infer", json={}).json()
    assert job["status"] == "done"
    assert job["result"]["status"] == "manual_install" and job["result"]["local_command"]
    assert client.post("/api/external-catalog/models/deepxde/install").status_code == 403
    assert len(client.get("/api/external-catalog/datasets").json()["datasets"]) == 1
