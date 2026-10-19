"""Physics AI Benchmark Arena app (apps/benchmark_arena): HTTP smoke test of the routes and a private benchmark."""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sklearn")
pytest.importorskip("pinneapple_core.service")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["ARN_DATA_DIR"] = str(tmp_path_factory.mktemp("arn"))
    sys.path.insert(0, str(ROOT / "apps" / "benchmark_arena"))
    from benchmark_arena.api import app
    return TestClient(app)


def test_public_results_are_real_runs(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Benchmark Arena" in client.get("/").text and client.get("/docs").status_code == 200
    pub = client.get("/api/public").json()["benchmarks"]
    assert {b["id"] for b in pub} == {"airfoil_rans", "burgers_fields"}
    for b in pub:
        r = client.get(f"/api/public/{b['id']}").json()
        assert r["created"] and r["hardware"]["logical_cpus"] and len(r["protocol"]["seeds"]) >= 3
        assert all(not m["failures"] for m in r["methods"] if m["applicable"]), "a public run has failed methods"
        for reg, rows in r["rankings"].items():
            assert [x["error_mean"] for x in rows] == sorted(x["error_mean"] for x in rows), reg
        assert client.get(f"/api/public/{b['id']}/report.md").text.startswith("# ")
        assert client.get(f"/api/public/{b['id']}/study_result.json").json()["study_type"] == "benchmark"


def test_external_published_results(client):
    m = client.get("/api/meta").json()
    assert [e["id"] for e in m["external"]] == ["bench09"] and len(m["external"][0]["commit"]) == 40
    e = client.get("/api/external/bench09").json()
    assert {r["problem"] for r in e["rows"]} == {"burgers", "darcy", "navier_stokes", "airfoil", "inverse_heat"}
    fno = next(r for r in e["rows"] if r["problem"] == "burgers" and r["method"] == "fno" and r["set"] == "pinn_test")
    assert abs(fno["mean"] - 0.001676527301879763) < 1e-12                 # as published, unchanged


def test_private_benchmark(client):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (120, 2))
    rows = ["a,b,y,case"] + [f"{a},{b},{np.sin(3 * a) + b ** 2},c{i // 4}" for i, (a, b) in enumerate(X)]
    up = client.post("/api/uploads", files={"file": ("t.csv", io.BytesIO("\n".join(rows).encode()), "text/csv")}).json()
    body = {"upload_id": up["id"], "inputs": ["a", "b"], "outputs": ["y"], "group": "case", "methods": ["ridge", "gp"],
            "budget_s": 0.5, "seeds": 2, "extrapolation_column": "a", "constraints": ["y > -5"]}
    assert client.post("/api/benchmarks", json={**body, "seeds": 99}).status_code == 422
    assert client.post("/api/benchmarks", json={**body, "extrapolation_column": "zz"}).status_code == 422
    job = client.post("/api/benchmarks", json=body).json()
    t0 = time.time()
    while (j := client.get(f"/api/jobs/{job['id']}").json())["status"] not in ("done", "failed") and time.time() - t0 < 300:
        time.sleep(0.3)
    assert j["status"] == "done", j.get("error")
    r = j["result"]
    assert set(r["rankings"]) == {"interpolation", "few_data", "extrapolation", "geometry_change"}
    assert "a >=" in next(g for g in r["regimes"] if g["id"] == "extrapolation")["description"]
    assert client.get(f"/api/benchmarks/{job['id']}/files/report.md").text.startswith("# Private benchmark")
    assert client.get("/api/usage").json()["units"]["benchmarks"]["used"] == 1
