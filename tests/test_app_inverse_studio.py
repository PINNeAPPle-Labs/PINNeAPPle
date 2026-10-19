"""Inverse Model Studio (apps/inverse_studio): HTTP smoke test of every route the page uses."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("INVS_DATA_DIR", str(tmp_path_factory.mktemp("invs")))
    mp.delenv("INVS_USER", raising=False)
    sys.path.insert(0, str(ROOT / "apps" / "inverse_studio"))
    from inverse_studio.api import app, rt

    rt.jobs.poll_interval = 0.02
    with TestClient(app) as c:
        yield c
    mp.undo()


def _wait(c, jid):
    for _ in range(2400):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError(jid)


def _body(c, preset, methods):
    p = c.get(f"/api/presets/{preset}").json()
    return {"model": p["model_spec"], "parameters": p["parameters"], "observations": p["observations"],
            "validation": p["validation"], "methods": methods,
            "options": {"prior_std": p["prior_std"], "smoothness": p["smoothness"]}}


def test_page_meta_presets(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Inverse Model Studio" in client.get("/").text
    meta = client.get("/api/meta").json()
    assert {"rod_h_field", "fin_h", "plate_h"} <= {p["id"] for p in meta["presets"]}
    assert set(meta["methods"]) == {"least_squares", "adjoint", "autodiff", "bayesian", "pinn"}
    assert meta["heat_lab_url"].startswith("https://")
    assert client.get("/docs").status_code == 200


def test_field_reconstruction_four_methods(client):
    body = _body(client, "rod_h_field", ["least_squares", "adjoint", "autodiff", "bayesian"])
    j = _wait(client, client.post("/api/solve", json=body).json()["id"])
    assert j["status"] == "done", j.get("error")
    r = j["result"]
    assert [s["status"] for s in r["solutions"]] == ["done"] * 4
    assert all(s["truth_recovered"] for s in r["solutions"])
    adj = next(s for s in r["solutions"] if s["method"] == "adjoint")
    assert adj["derivatives"] == "adjoint gradient" and set(adj["fields"]) == {"T", "h"}
    assert set(r["truth_fields"]) == {"T", "h"}
    assert r["identifiability"]["rank"] == 4 and r["study"]["study_type"] == "inverse"
    for what in ("report.json", "study.json", "estimates.csv", "fields.csv"):
        assert client.get(f"/api/jobs/{j['id']}/download/{what}").status_code == 200
    assert client.get("/api/usage").json()["units"]["method_runs"]["used"] >= 4


def test_unavailable_methods_and_errors(client):
    body = _body(client, "cooling_h", ["least_squares", "adjoint", "pinn"])
    assert client.post("/api/methods", json=body).json()["adjoint"]["available"] is False
    r = _wait(client, client.post("/api/solve", json=body).json()["id"])["result"]
    st = {s["method"]: s["status"] for s in r["solutions"]}
    assert st == {"least_squares": "done", "adjoint": "not applicable", "pinn": "not applicable"}
    assert client.post("/api/solve", json=dict(body, methods=["nope"])).status_code == 422
    assert client.post("/api/solve", json=dict(body, model={"type": "nope"})).status_code == 422
