"""Meshless Method Lab app (apps/meshless_lab): HTTP smoke test of the study queue, the examples, the contract and
the downloads, on the shared runtime."""
from __future__ import annotations

import json
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
    os.environ["MESHLESS_DATA_DIR"] = str(tmp_path_factory.mktemp("meshless"))
    sys.path.insert(0, str(ROOT / "apps" / "meshless_lab"))
    from meshless_lab.api import app
    return TestClient(app)


def _wait(client, job):
    assert "id" in job, job
    for _ in range(900):
        r = client.get(f"/api/jobs/{job['id']}").json()
        if r["status"] in ("done", "failed", "cancelled"):
            return r
        time.sleep(0.2)
    raise AssertionError("study did not finish")


def test_health_meta_page_and_preview(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert {e["id"] for e in m["examples"]} == {"lshape", "corner", "heat", "helmholtz"}
    assert set(m["methods"]) == {"kansa", "rbf_fd", "pinn", "fd", "fem"} and "lshape" in m["domains"]
    assert "Meshless Method Lab" in client.get("/").text and client.get("/docs").status_code == 200
    p = client.post("/api/domain", data={"spec": json.dumps({"domain": {"preset": "plate_hole"}, "h": 0.1})}).json()
    assert p["n"] > 50 and len(p["outline"]) == 2
    e = client.post("/api/expression", json={"kind": "helmholtz", "exact": "sin(3*x)", "k": 2}).json()
    assert "k = 2" in e["equation"]


def test_polygon_study_with_contract_and_downloads(client):
    spec = {"problem": {"kind": "poisson", "exact": "sin(pi*x)*sin(pi*y)"},
            "domain": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "holes": [[0.5, 0.5, 0.2]]},
            "h": [0.1, 0.05], "methods": ["rbf_fd", "fd", "fem"]}
    r = _wait(client, client.post("/api/studies", data={"spec": json.dumps(spec)}).json())
    assert r["status"] == "done", r.get("error")
    st = r["result"]["study"]
    assert st["convergence"]["fem"]["order"] == pytest.approx(2.0, abs=0.35)
    assert {w["criterion"].split(" ")[0] for w in st["wins"]} >= {"Accuracy", "Fastest"}
    sr = client.get(f"/api/jobs/{r['id']}/files/study_result.json").json()
    assert sr["kind"] == "StudyResult" and sr["study_type"] == "benchmark" and sr["checks"]
    csv = client.get(f"/api/jobs/{r['id']}/files/solutions.csv").text
    assert csv.splitlines()[0] == "x,y,reference,rbf_fd,fd,fem" and len(csv.splitlines()) == 2001


def test_point_cloud_study(client):
    sys.path.insert(0, str(ROOT))
    from pinneapple_simulation.meshless import domain
    ns = domain("square").nodes(0.05)
    csv = "x,y,boundary\n" + "\n".join(f"{x},{y},{int(b)}" for (x, y), b in zip(ns.X, ns.boundary, strict=True))
    spec = {"problem": {"kind": "poisson", "exact": "x*y + x**2"}, "methods": ["rbf_fd", "fd", "fem"]}
    r = _wait(client, client.post("/api/studies", data={"spec": json.dumps(spec)},
                                  files={"cloud": ("nodes.csv", csv.encode())}).json())
    assert r["status"] == "done", r.get("error")
    assert [s["method"] for s in r["result"]["study"]["skipped"]] == ["fd"]


def test_corner_example_runs_offline(client):
    r = _wait(client, client.post("/api/examples/corner").json())
    assert r["status"] == "done", r.get("error")
    c = r["result"]["study"]["convergence"]
    assert 1.0 < c["fem"]["order"] < 1.7 and 1.0 < c["rbf_fd"]["order"] < 1.7    # the singularity sets the order
    assert r["result"]["wall_s"] < 90


def test_bad_requests_are_refused(client):
    bad = {"problem": {"kind": "poisson", "exact": "__import__('os')"}}
    assert client.post("/api/studies", data={"spec": json.dumps(bad)}).status_code == 422
    tiny = {"problem": {"kind": "poisson", "exact": "x"}, "domain": {"preset": "square"}, "h": [0.001]}
    assert client.post("/api/studies", data={"spec": json.dumps(tiny)}).status_code == 413
    assert client.post("/api/studies", data={"spec": "{"}).status_code == 422
