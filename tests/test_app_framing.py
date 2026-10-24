"""Framing of the apps (apps/_shared/appkit.py, pinneapple_security.web): nobody may embed an app unless
<PREFIX>_FRAME_ANCESTORS lists the origin (Flow Studio's canvas inside the Hub), and a page embeds only the origins
its app declares (frame_src)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from pinneapple_security import web  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "_shared"))
import appkit  # noqa: E402


def _client(monkeypatch, tmp_path, frame_src=None, **env):
    for k in ("FRAMETEST_FRAME_ANCESTORS", "FRAMETEST_CSP"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("FRAMETEST_DATA_DIR", str(tmp_path))
    app = FastAPI()
    appkit.install(app, prefix="FRAMETEST", frame_src=frame_src)

    @app.get("/page")
    def page():
        return {"ok": True}
    return TestClient(app)


def _csp(r):
    return dict(p.strip().split(" ", 1) for p in r.headers["content-security-policy"].split(";"))


def test_no_framing_by_default(monkeypatch, tmp_path):
    r = _client(monkeypatch, tmp_path).get("/page")
    assert r.headers["x-frame-options"] == "DENY"
    assert _csp(r)["frame-ancestors"] == "'none'" and _csp(r)["frame-src"] == "'self'"


def test_frame_ancestors_allow_list(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, FRAMETEST_FRAME_ANCESTORS="https://hub.pinneapple.org, http://127.0.0.1:8116")
    r = c.get("/page")
    assert "x-frame-options" not in r.headers                    # X-Frame-Options can not name origins
    assert _csp(r)["frame-ancestors"] == "'self' https://hub.pinneapple.org http://127.0.0.1:8116"


def test_frame_src_for_an_embedded_app(monkeypatch, tmp_path):
    r = _client(monkeypatch, tmp_path, frame_src=["https://flow.pinneapple.org"]).get("/page")
    assert _csp(r)["frame-src"] == "'self' https://flow.pinneapple.org"
    assert r.headers["x-frame-options"] == "DENY"


@pytest.mark.parametrize("bad", ["*", "https://*.pinneapple.org", "https://hub.pinneapple.org/path", "hub.pinneapple.org",
                                 "javascript:alert(1)", "data:text/html,x"])
def test_bad_frame_origins_are_refused(bad, monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        web.check_frame_origins([bad])
    with pytest.raises(ValueError):
        _client(monkeypatch, tmp_path, FRAMETEST_FRAME_ANCESTORS=bad)
