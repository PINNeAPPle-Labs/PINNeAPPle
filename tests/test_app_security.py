"""Security of every app through apps/_shared/appkit.py install(): headers, CORS, body limit, rate limits, login
lockout, hashed passwords, CSRF origin check, client IP behind the proxy, no stack traces, audit trail; plus the
real apps' pages against their CSP."""
from __future__ import annotations

import base64
import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "_shared"))
import appkit  # noqa: E402

from pinneapple_security.audit import AuditLog  # noqa: E402
from pinneapple_security.passwords import hash_password  # noqa: E402
from pinneapple_security.web import inline_hashes  # noqa: E402


def _basic(user: str, pw: str) -> dict[str, str]:
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


def _app(monkeypatch, tmp_path, *, runtime=False, **env) -> FastAPI:
    for k in [k for k in list(__import__("os").environ) if k.startswith("SEC_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("SEC_DATA_DIR", str(tmp_path))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    app = FastAPI(title="sec")
    appkit.install(app, prefix="SEC")
    if runtime:
        appkit.install_runtime(app, prefix="SEC", name="sec", upload_suffixes={".csv"})

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/page")
    def page():
        return {"page": True}

    @app.post("/api/run")
    async def run(request: Request):
        return {"n": len(await request.body()), "who": appkit.principal(request)}

    @app.get("/api/boom")
    def boom():
        raise RuntimeError("secret path /srv/data/key.pem in a stack trace")

    return app


def test_security_headers_on_every_response(monkeypatch, tmp_path):
    c = TestClient(_app(monkeypatch, tmp_path))
    for path in ("/page", "/health", "/shared/charts.js", "/nope"):
        h = c.get(path).headers
        assert h["x-content-type-options"] == "nosniff", path
        assert h["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in h["content-security-policy"]
        assert h["strict-transport-security"].startswith("max-age=31536000")
        assert h["referrer-policy"] == "strict-origin-when-cross-origin"
        assert "camera=()" in h["permissions-policy"]
        assert "server" not in h


def test_no_cors_by_default_and_allow_list_when_set(monkeypatch, tmp_path):
    c = TestClient(_app(monkeypatch, tmp_path))
    r = c.get("/page", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    c = TestClient(_app(monkeypatch, tmp_path, SEC_CORS_ORIGINS="https://tool.example"))
    assert c.get("/page", headers={"Origin": "https://tool.example"}).headers["access-control-allow-origin"] == \
        "https://tool.example"
    r = c.options("/api/run", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 400 and "access-control-allow-origin" not in r.headers
    with pytest.raises(ValueError, match="credentials"):
        _app(monkeypatch, tmp_path, SEC_CORS_ORIGINS="*", SEC_CORS_CREDENTIALS="1")


def test_oversized_body_rejected_before_the_route(monkeypatch, tmp_path):
    c = TestClient(_app(monkeypatch, tmp_path, SEC_MAX_BODY_MB="1"))
    assert c.post("/api/run", content=b"x" * 1000).json()["n"] == 1000
    assert c.post("/api/run", content=b"x" * (2 << 20)).status_code == 413

    def chunks():                                               # no Content-Length: counted while streaming
        for _ in range(3):
            yield b"x" * (1 << 20)
    assert c.post("/api/run", content=chunks()).status_code == 413


def test_upload_name_traversal_and_suffix(monkeypatch, tmp_path):
    c = TestClient(_app(monkeypatch, tmp_path, runtime=True))
    r = c.post("/api/uploads", files={"file": ("../../etc/cron.d/evil.csv", b"a,b\n1,2\n")})
    assert r.status_code == 200
    up = r.json()
    stored = Path(c.app.state.runtime.uploads.path(up["id"]))
    assert stored.resolve().is_relative_to((tmp_path / "uploads").resolve()) and ".." not in stored.name
    assert c.post("/api/uploads", files={"file": ("x.sh", b"#!/bin/sh")}).status_code == 415


def test_rate_limit_returns_429_with_retry_after(monkeypatch, tmp_path):
    c = TestClient(_app(monkeypatch, tmp_path, SEC_RATE_LIMIT="3/min"))
    codes = [c.post("/api/run").status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = c.post("/api/run")
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert c.get("/page").status_code == 200                    # reads have their own, larger budget


def test_login_lockout_after_repeated_failures(monkeypatch, tmp_path):
    c = TestClient(_app(monkeypatch, tmp_path, SEC_USER="eng", SEC_PASSWORD_HASH=hash_password("s3cret", "scrypt"),
                        SEC_LOGIN_MAX_FAILURES="3"))
    assert c.get("/page").status_code == 401
    assert c.get("/page", headers=_basic("eng", "s3cret")).status_code == 200
    assert [c.get("/page", headers=_basic("eng", "guess")).status_code for _ in range(3)] == [401, 401, 401]
    r = c.get("/page", headers=_basic("eng", "s3cret"))         # right password, but locked out now
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert c.get("/health").status_code == 200
    events = [r["action"] for r in AuditLog(str(tmp_path / "audit.jsonl")).records()]
    assert events.count("login_failed") == 3 and "login_locked" in events
    assert AuditLog(str(tmp_path / "audit.jsonl")).verify()["ok"]


def test_hashed_and_legacy_plain_passwords(monkeypatch, tmp_path):
    c = TestClient(_app(monkeypatch, tmp_path, SEC_USER="eng", SEC_PASSWORD="plain-legacy"))
    assert c.get("/page", headers=_basic("eng", "plain-legacy")).status_code == 200
    users = tmp_path / "users"
    users.write_text(f"ana:{hash_password('a-pass', 'scrypt')}\n# comment\nbo:{hash_password('b-pass', 'scrypt')}\n")
    c = TestClient(_app(monkeypatch, tmp_path, runtime=True, SEC_USERS_FILE=str(users), SEC_ADMIN_USERS="ana"))
    assert c.get("/page", headers=_basic("bo", "b-pass")).status_code == 200
    assert c.get("/page", headers=_basic("bo", "a-pass")).status_code == 401
    assert c.get("/api/keys", headers=_basic("ana", "a-pass")).status_code == 200      # admin
    assert c.get("/api/keys", headers=_basic("bo", "b-pass")).status_code == 403       # user, not admin
    users.write_text("ana:plaintext\n")
    with pytest.raises(ValueError, match="not a hash"):
        _app(monkeypatch, tmp_path, SEC_USERS_FILE=str(users))


def test_cross_site_post_refused(monkeypatch, tmp_path):
    c = TestClient(_app(monkeypatch, tmp_path))
    evil = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
    assert c.post("/api/run", headers=evil).status_code == 403
    assert c.get("/page", headers=evil).status_code == 200
    assert c.post("/api/run", headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"}
                  ).status_code == 200
    assert c.post("/api/run").status_code == 200                # API clients send no Origin


def test_forwarded_for_trusted_only_from_the_proxy(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    direct = TestClient(app, client=("203.0.113.50", 1234))
    assert direct.post("/api/run", headers={"X-Forwarded-For": "198.51.100.7"}).json()["who"] == "ip:203.0.113.50"
    proxied = TestClient(app, client=("172.18.0.2", 1234))
    assert proxied.post("/api/run", headers={"X-Forwarded-For": "198.51.100.7"}).json()["who"] == "ip:198.51.100.7"


def test_no_stack_trace_to_clients(monkeypatch, tmp_path, caplog):
    c = TestClient(_app(monkeypatch, tmp_path), raise_server_exceptions=False)
    with caplog.at_level("ERROR", logger="pinneapple.apps"):
        r = c.get("/api/boom")
    assert r.status_code == 500
    body = r.text
    assert "Traceback" not in body and "key.pem" not in body and "RuntimeError" not in body
    assert r.json()["reference"] in caplog.text                 # the operator finds it in the log


def test_api_key_events_audited(monkeypatch, tmp_path):
    c = TestClient(_app(monkeypatch, tmp_path, runtime=True, SEC_ADMIN_TOKEN="tok"))
    assert c.post("/api/keys", json={"name": "ci"}).status_code == 403
    k = c.post("/api/keys", json={"name": "ci"}, headers={"X-Admin-Token": "tok"}).json()
    c.delete(f"/api/keys/{k['id']}", headers={"X-Admin-Token": "tok"})
    recs = list(AuditLog(str(tmp_path / "audit.jsonl")).records())
    assert [r["action"] for r in recs] == ["admin_denied", "api_key_created", "api_key_revoked"]
    assert k["key"] not in json.dumps(recs)                     # the secret itself is never logged


# ------------------------------------------------------------------------- the real apps and their pages
APPS = sorted(p.parent.parent.name for p in ROOT.glob("apps/*/*/static/index.html"))


@pytest.mark.parametrize("name", APPS)
def test_app_pages_fit_their_csp(name):
    """Every inline script and handler of each app's pages is allowed by hash by the CSP install() builds, and no
    page needs 'unsafe-inline' for scripts or a script from another origin."""
    pages = list((ROOT / "apps" / name).glob("*/static/**/*.html"))
    app = FastAPI()
    appkit.install(app, prefix="CSPTEST")
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(pages[0].parent)), name="static")
    csp = TestClient(app).get("/health").headers["content-security-policy"]
    script_src = csp.split("script-src ", 1)[1].split(";")[0]
    assert "'unsafe-inline'" not in script_src and "http" not in script_src
    for page in pages:
        scripts, handlers = inline_hashes(page.read_text(encoding="utf-8"))
        assert scripts | handlers <= set(script_src.split()), page


@pytest.mark.parametrize("module,prefix", [("heatsink_sizer.api", "HSS"), ("pcb_hotspot.api", "PCB")])
def test_real_app_serves_headers(module, prefix, monkeypatch):
    pytest.importorskip("torch")
    name = module.split(".")[0]
    sys.path.insert(0, str(ROOT / "apps" / name))
    try:
        sys.modules.pop(module, None)
        api = importlib.import_module(module)
        r = TestClient(api.app).get("/")
        assert r.status_code == 200 and "script-src 'self'" in r.headers["content-security-policy"]
        assert TestClient(api.app).post("/api/solve", content=b"{}", headers={
            "Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site",
            "Content-Type": "application/json"}).status_code == 403
    finally:
        sys.path.remove(str(ROOT / "apps" / name))


def _hog(params, ctx):                      # module level: the process executor imports it by name
    import subprocess
    import time
    if params.get("child"):
        subprocess.Popen(["sleep", "300"])
    if params.get("memory"):
        bytearray(params["memory"] * 2**20)
    if params.get("fail"):
        raise RuntimeError("/srv/secret/path failed")
    time.sleep(params.get("sleep", 0))
    return {"ok": True}


def _wait_job(c, job_id, timeout=60):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError


def test_job_limits_and_no_traceback_in_job_log(monkeypatch, tmp_path):
    pytest.importorskip("resource")
    app = _app(monkeypatch, tmp_path, runtime=True, SEC_JOB_EXECUTOR="process", SEC_JOB_MAX_SECONDS="3",
               SEC_JOB_MAX_MEMORY_MB="1500", SEC_RATE_LIMIT="off")
    rt = app.state.runtime
    rt.jobs.poll_interval = 0.02
    rt.jobs.register("hog", _hog)

    @app.post("/api/hog")
    def hog(request: Request, body: dict):
        return rt.submit(request, "hog", body)

    with TestClient(app) as c:
        j = _wait_job(c, c.post("/api/hog", json={"memory": 3000}).json()["id"])
        assert j["status"] == "failed"                               # 3 GB allocation above the 1.5 GB limit
        j = _wait_job(c, c.post("/api/hog", json={"sleep": 30}).json()["id"])
        assert j["status"] == "failed" and "time limit" in j["error"]
        j = _wait_job(c, c.post("/api/hog", json={"fail": True}).json()["id"])
        logs = json.dumps(c.get(f"/api/jobs/{j['id']}/logs").json())
        assert "Traceback" not in logs and "server log" in logs
    rt.jobs.stop()
