"""Parameter Identifiability Lab (apps/identifiability_lab): HTTP smoke test of every route the page uses."""
from __future__ import annotations

import csv
import io
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
    mp.setenv("IDENT_DATA_DIR", str(tmp_path_factory.mktemp("ident")))
    mp.delenv("IDENT_USER", raising=False)
    sys.path.insert(0, str(ROOT / "apps" / "identifiability_lab"))
    from identifiability_lab.api import app, rt

    rt.jobs.poll_interval = 0.02
    with TestClient(app) as c:
        yield c
    mp.undo()


def _wait(c, jid):
    for _ in range(1200):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError(jid)


def test_page_meta_and_presets(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Identifiability" in client.get("/").text
    meta = client.get("/api/meta").json()
    assert {"fin_h_k", "arrhenius_one_T"} <= {p["id"] for p in meta["presets"]}
    assert "pin_fin" in meta["models"]
    assert client.get("/api/presets/nope").status_code == 404
    assert client.get("/docs").status_code == 200


def test_fin_ambiguity_and_downloads(client):
    p = client.get("/api/presets/fin_h_k").json()
    job = client.post("/api/analyze", json={"model": p["model_spec"], "parameters": p["parameters"],
                                            "observations": p["observations"]}).json()
    j = _wait(client, job["id"])
    assert j["status"] == "done", j.get("error")
    r = j["result"]
    assert r["rank"] == 1 and r["ambiguities"][0]["fixed_combination"] == "h/k"
    assert r["suggestions"][0]["observable"] == "q_base"
    assert r["study"]["kind"] == "StudyResult" and r["study"]["study_type"] == "identifiability"
    for what in ("report.json", "study.json", "parameters.csv", "suggestions.csv"):
        assert client.get(f"/api/jobs/{job['id']}/download/{what}").status_code == 200
    usage = client.get("/api/usage").json()
    assert usage["units"]["analyses"]["used"] >= 1


def test_design_stage_without_values_and_validation(client):
    body = {"model": {"type": "arrhenius"},
            "parameters": [{"name": "A", "value": 2e7, "lower": 1, "upper": 1e14},
                           {"name": "Ea", "value": 75e3, "lower": 1e3, "upper": 3e5}],
            "observations": {"inputs": {"t_s": [120, 600, 2400], "T_K": [360, 360, 360]}, "sigma": [0.01] * 3}}
    j = _wait(client, client.post("/api/analyze", json=body).json()["id"])
    assert j["result"]["data"].startswith("expected")
    assert all(p["status"] == "structurally non-identifiable" for p in j["result"]["parameters"])
    bad = dict(body, model={"type": "nope"})
    assert client.post("/api/analyze", json=bad).status_code == 422
    fixed = dict(body, parameters=[dict(q, fixed=True) for q in body["parameters"]])
    assert client.post("/api/analyze", json=fixed).status_code == 422


def test_table_upload_and_parameter_space(client):
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["x_mm", "temp_C"])
    w.writerows([[10, "64,7"], [20, "54,2"], [30, "48,0"]])
    t = client.post("/api/table", files={"file": ("tc.csv", buf.getvalue().encode(), "text/csv")}).json()
    assert t["columns"] == ["x_mm", "temp_C"] and t["n_rows"] == 3
    space = {"kind": "ParameterSpace", "name": "fin",
             "parameters": [{"name": "h", "low": 1, "high": 500, "log": True, "unit": "W/m²K"},
                            {"name": "k", "kind": "constant", "value": 16}]}
    ps = client.post("/api/parameter-space", json=space).json()
    assert [(p["name"], p["fixed"]) for p in ps] == [("h", False), ("k", True)]
