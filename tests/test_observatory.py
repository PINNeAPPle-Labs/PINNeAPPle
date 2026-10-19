"""PINNeAPPle Observatory app: every API endpoint against the library it wraps, offline (local TLE file)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("skyfield_data")
pytest.importorskip("sgp4")

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "observatory"))
REF = os.path.join(os.path.dirname(__file__), "..", "pinneapple_physics", "astro", "data", "reference")
T = "2026-10-13T23:30:00Z"
SP = "lat=-23.5505&lon=-46.6333&elev=760"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient
    tle = json.load(open(os.path.join(REF, "iss_passes_reference.json")))["tle"]
    p = tmp_path_factory.mktemp("tle") / "iss.tle"
    p.write_text("\n".join(tle) + "\n")
    os.environ["OBS_TLE_FILE"] = str(p)
    os.environ["OBS_PRECOMPUTE"] = "0"
    sys.path.insert(0, APP)
    from observatory.api import app
    with TestClient(app) as c:
        yield c


def test_health_and_page(client):
    assert client.get("/health").json()["status"] == "ok"
    html = client.get("/").text
    assert "PINNeAPPle" in html and "/static/app.js" in html
    for f in ("app.js", "sky.js", "system.js", "astro.js", "style.css", "textures/earth_day.jpg", "textures/moon.jpg"):
        assert client.get(f"/static/{f}").status_code == 200, f
    assert client.get("/vendor/three/three.module.min.js").status_code == 200
    assert client.get("/vendor/three/objects/Sky.js").status_code == 200
    assert client.get("/shared/author.js").status_code == 200


def test_catalog(client):
    c = client.get("/api/catalog").json()
    assert len(c["hr"]) > 9000 and len(c["constellations"]) >= 88
    i = c["hr"].index(2491)
    assert c["name"][i] == "Sirius" and abs(c["v"][i] + 1.46) < 1e-6
    assert client.get("/api/milkyway.jpg").status_code == 200


def test_sky_matches_library(client):
    from pinneapple_physics.astro import Observer, Time, apparent
    d = client.get(f"/api/sky?{SP}&t={T}&span_h=2&step_min=10").json()
    assert d["sky_state"] == "night" and len(d["samples"]["jd"]) == 13
    p = apparent("saturn", Time.from_iso(T), Observer(-23.5505, -46.6333, 760))
    assert abs(d["now"]["saturn"]["alt_apparent"] - p.alt_apparent[0]) < 1e-4
    mid = len(d["samples"]["jd"]) // 2
    assert abs(d["samples"]["bodies"]["saturn"]["alt"][mid] - p.alt_apparent[0]) < 1e-3
    assert 0 <= d["moon"]["illuminated_fraction"] <= 1
    rts = d["almanac"]["rise_transit_set"]["sun"]
    assert rts["rise"] and rts["set"]
    assert np.allclose(np.array(d["npb"]) @ np.array(d["npb"]).T, np.eye(3), atol=1e-12)
    assert client.get(f"/api/sky?{SP}&t=1800-01-01T00:00:00Z").status_code == 400
    assert client.get("/api/sky?lat=95&lon=0").status_code == 400


def test_system(client):
    d = client.get(f"/api/system?t={T}&days=10").json()
    assert len(d["jd"]) == 21 and set(d["planets"]) >= {"earth", "mars", "jupiter"}
    r = np.linalg.norm(d["planets"]["earth"][10])
    assert 0.98 < r < 1.02
    assert abs(d["elements"]["mars"]["a_au"] - 1.5237) < 0.002


def test_analyses(client):
    s = client.get(f"/api/analysis/separation?a=moon&b=HR 6134&t={T}&{SP}").json()
    assert 0 < s["separation_deg"] < 180
    t = client.get(f"/api/analysis/target?name=saturn&t=2026-10-13T20:00:00Z&{SP}").json()
    assert t["best_window"] and t["best_window"]["max_alt"] > 30
    o = client.get(f"/api/analysis/orbit?body=mars&t={T}&{SP}").json()
    assert o["light_time_s"] > 100 and abs(o["doppler"]["beta"]) < 1e-3
    st = client.get(f"/api/analysis/star?hr=2491&t={T}&{SP}").json()
    assert st["name"] == "Sirius" and 2 < st["distance_pc"] < 3
    hr = client.get("/api/analysis/hr").json()
    assert len(hr["hr"]) > 500
    bb = client.get("/api/analysis/blackbody?T=5772").json()
    assert abs(bb["peak_nm"] - 502) < 1
    ev = client.get(f"/api/analysis/events?t=2024-03-25T00:00:00Z&days=30&{SP}").json()["events"]
    names = [e["event"] for e in ev]
    assert any("total solar eclipse" in n for n in names) and any("Full Moon" in n for n in names)
    li = client.get("/api/analysis/lensing").json()
    assert all(r["sigma"] < 1.5 for r in li["eht"])
    png = client.get("/api/analysis/lensing.png?ra=83.8&dec=-5.4&fov=40&r_cam=30&width=64")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    assert png.headers["x-disc"] in ("0", "1") and int(png.headers["x-bright-stars"]) >= 0


def test_satellites_offline_file(client):
    ref = json.load(open(os.path.join(REF, "iss_passes_reference.json")))["observers"][0]
    t = "2018-07-04T07:30:00Z"
    q = f"lat={ref['lat']}&lon={ref['lon']}"
    s = client.get(f"/api/satellites?{q}&t={t}&minutes=4&step_s=20").json()
    assert s["satellites"] and s["satellites"][0]["norad"] == 25544
    p = client.get(f"/api/passes?{q}&t=2018-07-03T12:00:00Z&days=1").json()["passes"]
    assert p and any(x["visible"] for x in p)
    refs = [e["jd_utc"] for e in ref["events"] if e["event"] == "culmination"]
    inside = [x for x in p if ref["jd_utc_start"] < x["culmination_jd"] < ref["jd_utc_start"] + ref["days"]]
    assert inside
    for x in inside:
        assert min(abs(np.array(refs) - x["culmination_jd"])) * 86400 < 2.0


def test_engine_integrators_and_validation(client):
    d = client.get("/api/engine/integrators?years=0.5&dt_hours=12").json()
    assert max(abs(v) for v in d["methods"]["yoshida4"]["rel_energy_error"]) < 1e-8
    assert max(abs(v) for v in d["methods"]["rk4"]["rel_energy_error"]) > 0
    v = client.get("/api/validation").json()
    assert "rows" in v


def test_login(client):
    from observatory.api import app
    app.state.auth_user, app.state.auth_password = "u", "p"
    try:
        assert client.get("/api/catalog").status_code == 401
        assert client.get("/health").status_code == 200
        assert client.get("/api/catalog", auth=("u", "p")).status_code == 200
    finally:
        app.state.auth_user = app.state.auth_password = None


def test_ai_endpoints(client):
    from pinneapple_physics.astro.ephemeris import AU_KM
    sb = client.get("/api/smallbodies").json()
    keys = {b["key"] for b in sb["bodies"]}
    assert {"apophis", "ceres", "phaethon"} <= keys
    tr = client.get("/api/smallbodies/track?body=apophis&t=2029-04-13T00:00:00Z&days=60").json()
    assert len(tr["pos_au"]) == 241 and 0.6 < np.linalg.norm(tr["pos_au"][120]) < 1.2
    ca = client.get("/api/smallbodies/approaches?body=apophis&start=2029-04-01T00:00:00Z&years=0.1").json()
    a = ca["approaches"][0]
    assert abs(a["dist_km"] - 0.000252172816 * AU_KM) < 700 and a["jpl"] is not None
    ml = client.get("/api/stars/ml").json()
    assert ml["lum_cv"]["accuracy"] > ml["lum_cv"]["baseline"]
    wx = client.get("/api/conditions?lat=0&lon=0").json()
    assert "error" in wx or "p_clear" in wx                          # offline: an explained error
    bh = client.get("/api/blackhole/status").json()
    assert "available" in bh
    dec = client.get("/api/sat/decay?days=10&f107=150&ap=15&B=0.005&h0=415").json()
    assert dec["verdict"]["status"] in ("trustworthy", "use_with_caution") and len(dec["h_pinn_km"]) == 200
    assert dec["h_pinn_km"][-1] < dec["h_pinn_km"][0]


def test_facts_endpoints(client):
    from pinneapple_physics.astro import facts
    c = client.get(f"/api/facts/card?id=saturn&lang=pt&{SP}&t={T}").json()
    assert c["name"] == "Saturno" and c["tonight"]["above_now"] is True
    assert [r["value"] for r in c["numbers"]] == [r["value"] for r in facts.card("saturn", "pt")["numbers"]]
    s = client.get(f"/api/facts/card?id=sat:25544&lang=en&{SP}&t={T}").json()
    assert s["id"] == "sat-iss" and "passes" in s
    r = client.get("/api/facts/card?id=nothing")
    assert r.status_code == 404 and r.headers["X-Error-Code"] == "no_card"
    assert client.get("/api/facts/search?q=cruzeiro&lang=pt").json()["results"][0]["id"] == "con-Cru"
    assert len(client.get("/api/facts/list?kind=messier").json()["cards"]) == 110
    assert client.get("/api/facts/tip?lang=en&seed=2").json()["text"]
    assert client.get("/api/facts/tip?lang=fr").status_code == 400


def test_observe_endpoints(client):
    from pinneapple_physics.astro import Observer, Time
    from pinneapple_physics.astro import observing as ob
    p = client.get(f"/api/observe/plan?target=saturn&{SP}&t={T}&step_min=10").json()
    ref = ob.visibility("saturn", Observer(-23.5505, -46.6333, 760), Time.from_iso(T), step_min=10)
    assert p["best_window"]["alt"] == pytest.approx(ref["best_window"]["alt"]) and p["target"]["id"] == "saturn"
    assert len(p["alt"]) == len(p["extinction"]["V"]) == len(p["sky_mag"]) == len(p["parallactic_deg"])
    m = client.get(f"/api/observe/plan?target=M31&{SP}&t={T}&step_min=15").json()
    assert m["target"]["ra_deg"] == pytest.approx(10.68, abs=0.05) and not m["never_rises"]
    assert client.get(f"/api/observe/plan?target=star:424&{SP}&t={T}&step_min=30").json()["never_rises"]
    assert client.get(f"/api/observe/plan?target=nowhere&{SP}&t={T}").status_code == 404
    o = client.get("/api/observe/optics?telescope=newton200&eyepiece=plossl25").json()
    assert o["magnification"] == pytest.approx(48.0)
    c = client.get("/api/observe/optics?aperture_mm=150&focal_mm=750&eyepiece=plossl10").json()
    assert c["magnification"] == pytest.approx(75.0)
    s = client.get("/api/observe/slew?from_alt=40&from_az=90&to_alt=40&to_az=270&mount=equatorial").json()
    assert s["ok"] and s["meridian_flip"]
    w = client.get(f"/api/observe/weather?{SP}&t={T}&force=cloudy").json()
    assert w["hidden"] and w["mode"] == "real"
    assert client.get(f"/api/observe/weather?{SP}&t={T}&mode=perfect").json()["hidden"] is False
    presets = client.get("/api/observe/presets").json()
    assert "sct8" in presets["telescopes"] and "hips2fits" in presets["survey_credit"]


def test_journey_endpoints(client):
    from pinneapple_physics.astro import Observer, Time, journey
    j = client.get(f"/api/journey/plan?target=mars&{SP}&t={T}").json()
    ref = journey.plan(Observer(-23.5505, -46.6333, 760), "mars", Time.from_iso(T))
    assert j["distance_km"] == pytest.approx(ref["distance_km"]) and len(j["npb"]) == 3
    s = client.get(f"/api/journey/plan?target=star:2491&{SP}&t={T}").json()
    assert s["distance_ly"] == pytest.approx(8.6, abs=0.05) and "sun_seen_from_target_mag" in s
    assert client.get(f"/api/journey/plan?target=M31&{SP}&t={T}").status_code == 404
    sc = client.get(f"/api/journey/scene?t={T}").json()
    assert len(sc["stars"]) > 500 and "mars" in sc["bodies"]
