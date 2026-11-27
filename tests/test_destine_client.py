"""DestinEClient with a mocked HTTP response (no network): discovery without credentials, the
client-credentials token, the item search, and the two refusals (401 and the 403 gate)."""
import io
import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

from pinneapple_data.datasets.external_sources import DestinEClient

DT = "EO.ECMWF.DAT.D1.DT_CLIMATE.G2.BASELINE_HIST_IFS-NEMO.R1"

COLLECTIONS = {"collections": [
    {"id": DT, "title": "Climate DT - Historical - IFS-NEMO - Generation-2 - Realization-1"},
    {"id": "EO.EUM.DAT.SENTINEL-3.OL_2_WFR___", "title": "Sentinel-3 water full resolution"},
    {"id": "EO.ECMWF.DAT.D1.DT_EXTREMES.G1.HIST_IFS-NEMO.R1", "title": "Extremes DT"},
]}

COLLECTION = {
    "id": DT, "type": "Collection", "license": "CC-BY-4.0",
    "sci:doi": "10.1234/example",
    "cube:variables": {
        "Temperature(clte_pl)": {"type": "data", "unit": "K"},
        "Orography(clte_sfc)": {"type": "data", "unit": "m"},
    },
    "cube:dimensions": {"time": {"type": "temporal"}},
    "extent": {"temporal": {"interval": [["1990-01-01T00:00:00Z", "2014-12-31T23:59:59Z"]]}},
}

SEARCH = {"type": "FeatureCollection", "features": [
    {"id": "a", "bbox": [-44.0, -24.0, -42.0, -22.0], "properties": {"datetime": "2020-01-01T00:00:00Z"}},
    {"id": "b", "bbox": [-40.0, -20.0, -38.0, -18.0], "properties": {"datetime": "2020-01-02T00:00:00Z"}},
]}

TOKEN = {"access_token": "TOK", "expires_in": 600, "token_type": "Bearer"}


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def fake_http(monkeypatch):
    """Records every request. ``routes`` maps a URL substring to its payload; otherwise the one
    payload (dict/list -> JSON, bytes -> as is) answers every call. ``status`` >= 400 raises."""
    calls = []

    def install(payload=None, *, routes=None, status=200):
        def urlopen(req, timeout=None):
            body = req.data
            calls.append({"url": req.full_url, "method": req.get_method(), "headers": dict(req.headers),
                          "body": json.loads(body) if body and body[:1] == b"{" else body,
                          "form": dict(urllib.parse.parse_qsl(body.decode())) if body else {},
                          "timeout": timeout})
            if status >= 400:
                raise urllib.error.HTTPError(req.full_url, status, "error", None, None)
            out = payload
            for fragment, routed in (routes or {}).items():
                if fragment in req.full_url:
                    out = routed
                    break
            return _Resp(out if isinstance(out, bytes) else json.dumps(out).encode())
        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        return calls
    return install


@pytest.fixture(autouse=True)
def no_env_credentials(monkeypatch):
    for name in ("DESTINE_CLIENT_ID", "DESTINE_CLIENT_SECRET", "DESTINE_TOKEN"):
        monkeypatch.delenv(name, raising=False)


# -- discovery: no credentials ------------------------------------------------

def test_collections_needs_no_token(fake_http):
    calls = fake_http(COLLECTIONS)
    cols = DestinEClient().collections()
    assert len(cols) == 3
    assert calls[0]["url"] == f"{DestinEClient.STAC_URL}/collections"
    assert "Authorization" not in calls[0]["headers"]


def test_climate_dt_collections_filters_by_substring(fake_http):
    fake_http(COLLECTIONS)
    ids = [c["id"] for c in DestinEClient().climate_dt_collections()]
    assert ids == [DT]                       # DT_EXTREMES is not matched by DT_CLIMATE
    assert len(DestinEClient().collections("eum")) == 1       # case-insensitive


def test_variables_from_collection_metadata(fake_http):
    calls = fake_http(COLLECTION)
    variables = DestinEClient().variables(DT)
    assert calls[0]["url"] == f"{DestinEClient.STAC_URL}/collections/{DT}"
    assert variables["Temperature(clte_pl)"]["unit"] == "K" and len(variables) == 2
    assert "Authorization" not in calls[0]["headers"]


# -- token --------------------------------------------------------------------

def test_client_credentials_token_is_fetched_once_and_sent_as_bearer(fake_http):
    calls = fake_http(None, routes={"openid-connect/token": TOKEN, "/search": SEARCH})
    c = DestinEClient(client_id="svc", client_secret="s3cret")
    c.search(DT, limit=2)
    c.search(DT, limit=2)                    # second call reuses the cached token
    token_calls = [x for x in calls if "openid-connect/token" in x["url"]]
    assert len(token_calls) == 1 and len(calls) == 3
    assert token_calls[0]["method"] == "POST" and token_calls[0]["url"] == DestinEClient.TOKEN_URL
    assert token_calls[0]["form"] == {"grant_type": "client_credentials", "client_id": "svc",
                                      "client_secret": "s3cret"}
    searches = [x for x in calls if x["url"].endswith("/search")]
    assert all(x["headers"]["Authorization"] == "Bearer TOK" for x in searches)


def test_missing_credentials_raise_before_any_request(fake_http):
    calls = fake_http(SEARCH)
    c = DestinEClient()
    with pytest.raises(PermissionError, match="platform.destine.eu"):
        c.search(DT)
    with pytest.raises(PermissionError, match="DESTINE_CLIENT_ID"):
        c.items(DT)
    assert calls == []                       # discovery is still open, the gated path is not tried


def test_credentials_come_from_the_environment(fake_http, monkeypatch):
    calls = fake_http(None, routes={"openid-connect/token": TOKEN, "/search": SEARCH})
    monkeypatch.setenv("DESTINE_CLIENT_ID", "env-id")
    monkeypatch.setenv("DESTINE_CLIENT_SECRET", "env-secret")
    DestinEClient().search(DT)
    assert calls[0]["form"]["client_id"] == "env-id"


# -- item search and assets ---------------------------------------------------

def test_search_sends_the_stac_payload(fake_http):
    calls = fake_http(SEARCH, routes={"/search": SEARCH})
    c = DestinEClient(token="TOK")
    js = c.search(DT, bbox=(-44, -24, -42, -22), time_range="2020-01-01T00:00:00Z/2020-01-02T00:00:00Z",
                  limit=5)
    call = calls[0]
    assert call["method"] == "POST" and call["url"] == f"{DestinEClient.STAC_URL}/search"
    assert call["body"] == {"collections": [DT], "limit": 5, "bbox": [-44.0, -24.0, -42.0, -22.0],
                            "datetime": "2020-01-01T00:00:00Z/2020-01-02T00:00:00Z"}
    assert js["type"] == "FeatureCollection"


def test_search_accepts_a_bare_collection_id_and_rejects_an_empty_list(fake_http):
    fake_http(SEARCH)
    assert DestinEClient(token="TOK").search(DT)["features"]
    with pytest.raises(ValueError, match="collection id"):
        DestinEClient(token="TOK").search([])
    with pytest.raises(ValueError, match="west, south, east, north"):
        DestinEClient(token="TOK").search(DT, bbox=(-44, -24, -42))


def test_items_builds_the_get_query(fake_http):
    calls = fake_http(SEARCH)
    DestinEClient(token="TOK").items(DT, limit=3, bbox=(-44, -24, -42, -22), time_range="2020-01-01/2020-01-02")
    url = calls[0]["url"]
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    assert calls[0]["method"] == "GET"
    assert url.startswith(f"{DestinEClient.STAC_URL}/collections/{DT}/items?")
    assert query == {"limit": "3", "bbox": "-44.0,-24.0,-42.0,-22.0", "datetime": "2020-01-01/2020-01-02"}
    assert calls[0]["headers"]["Authorization"] == "Bearer TOK"


def test_download_writes_the_asset(fake_http, tmp_path):
    calls = fake_http(b"GRIB\x00\x01")
    out = tmp_path / "asset.grib"
    raw = DestinEClient(token="TOK").download("https://objects.example/a.grib", path=str(out))
    assert raw == b"GRIB\x00\x01" and out.read_bytes() == raw
    assert calls[0]["headers"]["Authorization"] == "Bearer TOK"


# -- the two refusals ---------------------------------------------------------

def test_403_names_the_upgraded_access_gate(fake_http):
    fake_http(SEARCH, status=403)
    with pytest.raises(PermissionError, match="upgraded access"):
        DestinEClient(token="TOK").search(DT)
    with pytest.raises(PermissionError, match="DT_CLIMATE"):
        DestinEClient(token="TOK").items(DT)
    with pytest.raises(PermissionError, match="Access Policy Upgrade"):
        DestinEClient(token="TOK").download("https://objects.example/a.grib")


def test_401_names_the_service_account_key(fake_http):
    fake_http(SEARCH, status=401)
    with pytest.raises(PermissionError, match="DESTINE_CLIENT_ID"):
        DestinEClient(token="TOK").search(DT)


def test_other_http_errors_propagate(fake_http):
    fake_http(SEARCH, status=500)
    with pytest.raises(urllib.error.HTTPError):
        DestinEClient(token="TOK").search(DT)


# -- response to arrays -------------------------------------------------------

def test_to_arrays_of_a_search_response():
    arr = DestinEClient.to_arrays(SEARCH)
    assert arr["datetime"].dtype == np.dtype("datetime64[s]")
    assert arr["datetime"].tolist() == np.array(["2020-01-01T00:00:00", "2020-01-02T00:00:00"],
                                                "datetime64[s]").tolist()
    assert arr["bbox"].shape == (2, 4) and arr["bbox"][1].tolist() == [-40.0, -20.0, -38.0, -18.0]


def test_to_arrays_missing_time_is_nat():
    js = {"features": [{"id": "a", "bbox": [0, 1, 2, 3], "properties": {}}]}
    arr = DestinEClient.to_arrays(js)
    assert np.isnat(arr["datetime"][0]) and arr["bbox"][0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_to_arrays_takes_the_start_of_an_interval():
    js = {"features": [{"id": "a", "bbox": [0, 1, 2, 3],
                        "properties": {"datetime": "2020-01-01T00:00:00Z/2020-01-31T23:59:59Z"}}]}
    assert DestinEClient.to_arrays(js)["datetime"][0] == np.datetime64("2020-01-01T00:00:00")


def test_to_arrays_keeps_the_footprint_of_a_3d_bbox():
    js = {"features": [{"id": "a", "bbox": [0, 1, 2, 3, 500, 900], "properties": {"datetime": "2020-01-01"}}]}
    assert DestinEClient.to_arrays(js)["bbox"][0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_to_arrays_rejects_an_empty_response():
    with pytest.raises(ValueError, match="no features"):
        DestinEClient.to_arrays({"features": []})
