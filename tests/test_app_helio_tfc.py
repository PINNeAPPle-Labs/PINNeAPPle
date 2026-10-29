"""HelioTFC app (apps/helio_tfc): the source app's tests/test_app.py on the synthetic series (its SILSO fixture is
CC BY-NC and not shipped), an uploaded SILSO-format CSV, and the live download only with PINNEAPPLE_NETWORK=1."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pandas")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["HELIO_DATA_DIR"] = str(tmp_path_factory.mktemp("helio"))
    sys.path.insert(0, str(ROOT / "apps" / "helio_tfc"))
    from helio_tfc.api import app
    return TestClient(app)


def test_imf_summary(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/v1/imf_summary", params={"data_source": "synthetic", "max_imf": 6})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_imfs"] >= 2 and len(body["imf_statistics"]) == body["n_imfs"]
    assert body["closest_to_11yr_cycle"]["imf_index"] != 0 and "synthetic" in body["data_credit"]
    assert client.get("/v1/imf_summary", params={"data_source": "nope"}).status_code == 422


def test_full_pipeline(client):
    r = client.post("/v1/decompose_and_forecast", json={"data_source": "synthetic", "hidden_units": 30,
                                                        "forecast_horizon_frac": 0.5})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["endpoint_constraint_error"] < 1e-9 and len(b["forecast_amplitude"]) == len(b["forecast_t"])
    assert client.post("/v1/decompose_and_forecast", json={"imf_index": 99}).status_code == 422
    assert client.post("/v1/decompose_and_forecast", json={"start_date": "2020-01-01",
                                                           "end_date": "2020-01-10"}).status_code == 422


def test_upload(client):
    from pinneapple_systems.time_series import solar_cycle as sc

    csv = sc.synthetic_daily(start="2010-01-01", end="2020-12-31")[sc.SILSO_COLUMNS].to_csv(
        sep=";", header=False, index=False).encode()
    r = client.post("/v1/decompose_and_forecast_upload", files={"file": ("sn.csv", csv, "text/csv")},
                    data={"hidden_units": "20"})
    assert r.status_code == 200, r.text
    assert r.json()["n_samples"] > 4000 and "uploaded" in r.json()["data_credit"]
    bad = client.post("/v1/decompose_and_forecast_upload", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 422


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_NETWORK"), reason="network (set PINNEAPPLE_NETWORK=1)")
def test_live_silso(client):
    r = client.get("/v1/imf_summary", params={"data_source": "live_silso", "start_date": "2005-01-01",
                                              "end_date": "2025-06-30"})
    assert r.status_code == 200 and "SILSO" in r.json()["data_credit"]
