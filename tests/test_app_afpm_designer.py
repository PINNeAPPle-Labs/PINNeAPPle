"""Axial-Flux Motor Designer app (apps/afpm_designer): HTTP smoke test of the page, the estimate, the check in 3-D
and the 3-D view, on the default design (about 30 s)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["AFPM_DATA_DIR"] = str(tmp_path_factory.mktemp("afpm"))
    sys.path.insert(0, str(ROOT / "apps" / "afpm_designer"))
    from afpm_designer.api import app
    with TestClient(app) as c:
        yield c


def _wait(client, job, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.5)
    raise TimeoutError(job["id"])


def test_health_page_and_defaults(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/").text
    assert "Axial-Flux Motor Designer" in page and "Verify in 3-D" in page and "cdn.jsdelivr" not in page
    d = client.get("/api/defaults").json()["design"]
    assert d["poles"] == 8 and d["slots"] == 12 and d["topology"] == "single"


def test_bad_designs_are_refused(client):
    assert client.post("/api/estimate", json={"r_in": 0.08, "r_out": 0.085}).status_code == 422
    assert client.post("/api/estimate", json={"poles": 7}).status_code == 422
    r = client.post("/api/estimate", json={"slots": 9, "poles": 8})          # cogging period too short for the grid
    assert r.status_code == 422 and "576 steps" in r.text


def test_verify_in_3d_measures_what_the_estimate_leaves_out(client):
    j = _wait(client, client.post("/api/verify", json={}).json())
    assert j["status"] == "done", j
    r = j["result"]
    # the default design on its coarse grid: quasi-3D 6.128 N m and 1.0630 mWb, 3-D 5.417 N m and 1.0411 mWb
    assert r["cogging_pkpk_Nm"] == pytest.approx(5.4168, rel=1e-3)
    assert r["estimate_over_3d"]["cogging_pkpk"] == pytest.approx(1.1312, rel=1e-3)
    assert r["estimate_over_3d"]["tooth_linkage"] == pytest.approx(1.0210, rel=1e-3)
    meta = client.get(f"/api/view/{r['view']}").json()
    assert meta["schema"] == "pinneapple-field-view/v1" and {f["name"] for f in meta["fields"]} == {"B", "Bz"}
    glb = client.get(f"/api/view/{r['view']}.glb")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    assert client.get("/api/view/nope").status_code == 404
