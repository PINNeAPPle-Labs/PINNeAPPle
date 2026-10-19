"""Engineering Evidence Report app (apps/evidence_report): HTTP smoke test, and the rule that a claimed pass without
its record is printed "not run" end to end (report, HTML, package)."""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("reportlab")
from fastapi.testclient import TestClient  # noqa: E402

from pinneapple_veriphysics.evidence_package import verify_package  # noqa: E402
from pinneapple_veriphysics.ledger import EvidenceLedger, sha256_bytes, utc_now  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["EVR_DATA_DIR"] = str(tmp_path_factory.mktemp("evr"))      # runtime state (keys, usage)
    os.environ["EVR_FREE_REPORTS"] = "2"
    sys.path.insert(0, os.path.join(ROOT, "apps", "evidence_report"))
    from evidence_report.api import app
    return TestClient(app)


def test_health_meta_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert "not run" in m["principle"] and "validation-lab" in m["kinds"]
    assert "Engineering Evidence Report" in client.get("/").text
    assert client.get("/docs").status_code == 200


def test_example_study_is_real_and_honest(client):
    out = client.get("/api/example").json()
    rep = out["report"]
    by = {t["check_id"]: t for t in rep["tests"]}
    assert by["grid convergence (GCI, 3 meshes)"]["status"] == "not run"           # claimed, never recorded
    assert by["grid convergence (GCI, 3 meshes)"]["source_claim"] == "passed"
    assert by["trust card FEM/PDE residual"]["status"] == "passed"
    assert by["trust card PINN/boundary condition on x_min"]["status"] == "failed"  # 200 epochs: BC error > 2 %
    comp = {r["metric"]: r for r in rep["comparison"]}
    assert comp["rel_l2_u"]["baseline"] < comp["rel_l2_u"]["candidate"]
    assert comp["rel_l2_u"]["baseline_run"].startswith("FEM")
    assert {a["kind"] for a in rep["artifacts"]} >= {"validation-lab", "metrics", "contract:StudyResult",
                                                     "contract:ModelRef", "file"}
    z = client.get("/api/example/package")
    assert z.status_code == 200 and verify_package(z.content) == []
    assert client.get("/api/example/pdf").content[:4] == b"%PDF"
    with zipfile.ZipFile(io.BytesIO(client.get("/api/example/sources").content)) as zf:
        assert "study_result.json" in zf.namelist() and "meta.json" in zf.namelist()


def test_uploads_report_pdf_and_package(client):
    led = EvidenceLedger()
    led.record("mass balance", passed=True, inputs_sha256=sha256_bytes(b"case"), outputs={"imbalance": 1e-5},
               started_at=utc_now(), finished_at=utc_now())
    claims = {"tool": "pinneapple check", "checks": [{"rule": "PC101", "target": "case", "subject": "mass",
                                                       "passed": True, "message": "ok"}]}
    files = [("files", ("ledger.json", json.dumps(led.to_dict()).encode())),
             ("files", ("ci.json", json.dumps(claims).encode())),
             ("files", ("metrics.csv", b"run,cd\nCFD fine,0.0213\nsurrogate v2,0.0220\n")),
             ("files", ("case.zip", b"PK\x05\x06" + b"\x00" * 18))]
    meta = {"title": "Drag surrogate v2", "baseline": "CFD", "candidate": "surrogate"}
    data = {"meta": json.dumps(meta), "urls": json.dumps({"case.zip": "s3://runs/42/case.zip"})}
    r = client.post("/api/report", files=files, data=data).json()
    rep = r["report"]
    st = {t["check_id"]: t["status"] for t in rep["tests"]}
    assert st == {"mass balance": "passed", "PC101/case/mass": "not run"}
    assert rep["comparison"][0]["baseline"] == pytest.approx(0.0213)
    assert next(a for a in rep["artifacts"] if a["name"] == "case.zip")["url"] == "s3://runs/42/case.zip"
    row = r["html"][r["html"].index("PC101/case/mass"):].split("</tr>")[0]
    assert "not run" in row and "class='passed'" not in row
    assert client.post("/api/report.pdf", files=files, data=data).content[:4] == b"%PDF"
    z = client.post("/api/package", files=files, data=data).content
    assert verify_package(z) == []
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        assert zf.read("sources/case.zip") == b"PK\x05\x06" + b"\x00" * 18


def test_bad_requests(client):
    assert client.post("/api/report", data={"meta": "{}"}).status_code == 422
    files = [("files", ("a.txt", b"x"))]
    assert client.post("/api/report", files=files, data={"meta": "[1]"}).status_code == 422
    assert client.get("/api/example/nope").status_code == 404


def test_reports_are_metered_and_exported_as_a_study_result(client):
    from pinneapple_data.contracts import validate

    files = [("files", ("metrics.csv", b"run,cd\nCFD,0.02\nsurrogate,0.03\n"))]
    codes, last = [], None
    for _ in range(3):
        r = client.post("/api/report", files=files, data={"meta": "{}"})
        codes.append(r.status_code)
        last = r.json() if r.status_code == 200 else last
    assert 429 in codes and codes.index(429) <= 2                   # EVR_FREE_REPORTS=2 per caller and month
    assert validate(last["study_result"]) == [] and last["study_result"]["study_type"] == "evidence"
    usage = client.get("/api/usage").json()
    assert usage and client.get("/api/runtime").json()["app"] == "evidence"
    assert client.get("/api/example").status_code == 200               # the example is not metered
