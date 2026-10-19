"""Model Calibration Lab (apps/calibration_lab): HTTP smoke test of every route the page uses."""
from __future__ import annotations

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
    mp.setenv("CALIB_DATA_DIR", str(tmp_path_factory.mktemp("calib")))
    mp.delenv("CALIB_USER", raising=False)
    sys.path.insert(0, str(ROOT / "apps" / "calibration_lab"))
    from calibration_lab.api import app, rt

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


def _body(c, preset, **extra):
    p = c.get(f"/api/presets/{preset}").json()
    return {"model": p["model_spec"], "parameters": p["parameters"], "observations": p["observations"],
            "validation": p["validation"], **extra}


def test_page_meta_presets(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Calibration" in client.get("/").text
    meta = client.get("/api/meta").json()
    assert {"cooling_h", "fin_h_k", "plate_h_table"} <= {p["id"] for p in meta["presets"]}
    assert "bayesian" in meta["methods"]
    assert client.get("/docs").status_code == 200


def test_cooling_extrapolation_with_comparison_and_downloads(client):
    body = _body(client, "cooling_h", holdout={"mode": "validation"},
                 options={"bootstrap": 40, "compare": ["autodiff", "bayesian"]})
    j = _wait(client, client.post("/api/calibrate", json=body).json()["id"])
    assert j["status"] == "done", j.get("error")
    r = j["result"]
    for p in r["parameters"]:
        assert p["ci_jacobian"][0] < p["truth"] < p["ci_jacobian"][1]
    assert r["metrics"]["holdout"]["after"]["rmse"] < r["metrics"]["holdout"]["before"]["rmse"]
    assert {s["method"] for s in r["comparison"]["solutions"]} == {"least_squares", "autodiff", "bayesian"}
    assert r["curve"]["input"] == "t_s" and len(r["curve"]["after"]) == 160
    assert r["study"]["study_type"] == "calibration" and r["study"]["models"][0]["metrics_split"] == "holdout"
    for what in ("report.json", "study.json", "model_ref.json", "parameters.csv", "residuals.csv"):
        assert client.get(f"/api/jobs/{j['id']}/download/{what}").status_code == 200
    assert client.get("/api/usage").json()["units"]["method_runs"]["used"] >= 2


def test_not_estimable_and_holdout_modes(client):
    body = _body(client, "fin_h_k", holdout={"mode": "random", "fraction": 0.4}, options={"bootstrap": 0})
    body.pop("validation")
    r = _wait(client, client.post("/api/calibrate", json=body).json()["id"])["result"]
    assert all(p["ci_jacobian"] is None and p["not_estimable"] for p in r["parameters"])
    assert r["holdout_mode"] == "random" and r["metrics"]["holdout"]["after"]["n"] == 2


def test_errors_and_table(client):
    body = _body(client, "cooling_h")
    body["observations"]["values"] = None
    assert client.post("/api/calibrate", json=body).status_code == 422
    body = _body(client, "cooling_h", options={"compare": ["nope"]})
    assert client.post("/api/calibrate", json=body).status_code == 422
    csv = b"time;T\n0;94,1\n60;92,9\n120;91,5\n180;90,1\n240;89,0\n"
    t = client.post("/api/table", files={"file": ("log.csv", io.BytesIO(csv), "text/csv")}).json()
    assert t["columns"] == ["time", "T"] and t["n_rows"] == 5
    assert t["rows"]["T"][:2] == [94.1, 92.9]                       # decimal comma read as a number
