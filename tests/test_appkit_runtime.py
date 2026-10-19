"""apps/_shared/appkit.py install_runtime: jobs, API keys, usage and uploads over HTTP."""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "_shared"))
import appkit  # noqa: E402


def plan(params, ctx):
    ctx.progress(0.5, "sampling")
    return {"n": params["n"]}


def _app(monkeypatch, tmp_path, login=True, **env):
    monkeypatch.setenv("TST_DATA_DIR", str(tmp_path))
    if login:
        monkeypatch.setenv("TST_USER", "admin")
        monkeypatch.setenv("TST_PASSWORD", "secret")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    app = FastAPI(title="test")
    appkit.install(app, prefix="TST")
    rt = appkit.install_runtime(app, prefix="TST", name="tst", units={"cases_planned": 100},
                                upload_suffixes={".csv"})
    rt.jobs.poll_interval = 0.02
    rt.jobs.register("plan", plan)

    @app.post("/api/plan")
    def submit(request: Request, n: int = 10):
        return rt.submit(request, "plan", {"n": n}, units={"cases_planned": n})

    @app.get("/health")
    def health():
        return {"ok": True}

    return app, rt


BASIC = {"Authorization": "Basic " + base64.b64encode(b"admin:secret").decode()}


def _wait(client, job_id, headers):
    for _ in range(500):
        job = client.get(f"/api/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.01)
    raise TimeoutError


def test_login_api_keys_jobs_and_usage(monkeypatch, tmp_path):
    app, rt = _app(monkeypatch, tmp_path, TST_FREE_CASES_PLANNED="30")
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/api/runtime").status_code == 401                      # login required
    assert c.get("/api/runtime", headers=BASIC).json()["free_limits_per_month"] == {"cases_planned": 30.0}

    key = c.post("/api/keys", json={"name": "ci"}, headers=BASIC).json()
    hk = {"X-API-Key": key["key"]}
    assert c.get("/api/keys", headers=hk).status_code == 403              # a key is not admin
    assert c.get("/api/runtime", headers={"X-API-Key": "pk_bad"}).status_code == 401

    job = c.post("/api/plan?n=20", headers=hk).json()
    assert job["owner"] == f"key:{key['id']}"
    assert _wait(c, job["id"], hk)["result"] == {"n": 20}
    assert [j["id"] for j in c.get("/api/jobs", headers=hk).json()] == [job["id"]]
    assert c.get(f"/api/jobs/{job['id']}/logs", headers=hk).json()[0]["line"].startswith("attempt 1")

    r = c.post("/api/plan?n=20", headers=hk)                               # 20 + 20 > 30
    assert r.status_code == 429 and r.json()["limit"] == 30
    usage = c.get("/api/usage", headers=hk).json()
    assert usage["units"]["cases_planned"]["used"] == 20 and "totals" not in usage
    assert c.get("/api/usage", headers=BASIC).json()["totals"]["units"]["cases_planned"]["used"] == 20

    other = c.post("/api/keys", json={"name": "other"}, headers=BASIC).json()
    assert c.get(f"/api/jobs/{job['id']}", headers={"X-API-Key": other["key"]}).status_code == 404
    assert c.get(f"/api/jobs/{job['id']}", headers=BASIC).status_code == 200   # admin sees all

    assert c.delete(f"/api/keys/{key['id']}", headers=BASIC).json()["revoked"]
    assert c.get("/api/usage", headers=hk).status_code == 401
    rt.jobs.stop()


def test_public_mode_admin_token_and_uploads(monkeypatch, tmp_path):
    app, rt = _app(monkeypatch, tmp_path, login=False, TST_ADMIN_TOKEN="tok", TST_UPLOAD_MAX_MB="0.001")
    c = TestClient(app)
    assert c.post("/api/keys", json={"name": "x"}).status_code == 403     # no login: key admin needs the token
    assert c.post("/api/keys", json={"name": "x"}, headers={"X-Admin-Token": "tok"}).status_code == 200
    job = c.post("/api/plan?n=5").json()
    assert job["owner"].startswith("ip:") and _wait(c, job["id"], {})["status"] == "done"

    up = c.post("/api/uploads", files={"file": ("cases.csv", b"a,b\n1,2\n", "text/csv")})
    assert up.status_code == 200 and up.json()["size"] == 8
    assert c.post("/api/uploads", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415
    assert c.post("/api/uploads", files={"file": ("big.csv", b"x" * 5000, "text/csv")}).status_code == 413
    assert len(c.get("/api/uploads").json()) == 1
    assert c.delete(f"/api/uploads/{up.json()['id']}").json() == {"deleted": up.json()["id"]}
    rt.jobs.stop()


def test_require_api_key(monkeypatch, tmp_path):
    app, rt = _app(monkeypatch, tmp_path, login=False, TST_REQUIRE_API_KEY="1", TST_ADMIN_TOKEN="tok")
    c = TestClient(app)
    assert c.get("/api/runtime").status_code == 401
    key = rt.keys.create("k")["key"]
    assert c.get("/api/runtime", headers={"X-API-Key": key}).status_code == 200
    assert c.get("/health").status_code == 200
    rt.jobs.stop()
