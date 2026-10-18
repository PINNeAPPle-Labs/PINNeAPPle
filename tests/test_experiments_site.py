"""The experiments page: every experiment of the coverage list behind /api/experiments, the filters, one entry's
detail (E/F problem and next steps), seals and stamped figures, the login, the static export, and a Playwright smoke
test of the page (list, detail, an E/F card) in the preinstalled Chromium."""
import base64
import json
import os
import shutil
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from pinneapple_lab import LabStore
from pinneapple_lab import manifest as M
from pinneapple_lab.explorer import collect, coverage, filter_entries, write_static

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMIUM = next((p for p in ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome",) if os.path.exists(p)), None)


@pytest.fixture(scope="module")
def lab_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab")
    os.symlink(os.path.join(REPO, "lab", "runs"), root / "runs")
    shutil.copy(os.path.join(REPO, "lab", "curation.json"), root / "curation.json")
    return str(root)


@pytest.fixture(scope="module")
def server(lab_root):
    from pinneapple_lab.server import LabServer
    lab = LabServer(lab_root, refresh=0, user="u", password="p")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lab.handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


AUTH = {"Authorization": "Basic " + base64.b64encode(b"u:p").decode()}


def get(url, auth=True):
    req = urllib.request.Request(url, headers=AUTH if auth else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def api(server, path):
    return json.loads(get(server + path)[2])


# ---------------------------------------------------------------------------------------------- the list
def test_api_lists_every_experiment_of_the_coverage_list(server, lab_root):
    every = api(server, "/api/experiments")
    cov = api(server, "/api/coverage")
    assert {e["key"] for e in every["entries"]} == {r["key"] for r in cov} == \
        {e["key"] for e in collect(LabStore(lab_root))["entries"]}
    keys = {e["key"] for e in every["entries"]}
    for must in ("examples/flight_dynamics", "examples/enso_forecasting", "examples/near_random_forecasting",
                 "examples/black_hole_weather", "examples/modelica", "examples/electrodynamics",
                 "em_pinn/em_tm_cavity", "rom_study/cylinder_wake", "calculix_case/plate_hole",
                 "repo_results/ahmed_body", "repo_results/weather_era5"):
        assert must in keys, must
    assert every["count"] == len(keys) == sum(every["counts"].values())     # counts counted from the list
    for m in M.example_dirs():
        assert os.path.relpath(m, REPO) in keys                              # every manifest is on the page
    nc = [r for r in cov if not r["curated"]]
    assert nc and all(r["action"].startswith("python -m pinneapple_lab register examples/") for r in nc)


def test_filters(server):
    f = api(server, "/api/experiments?letter=F")
    assert f["count"] == f["counts"]["F"] > 0 and {e["tier"] for e in f["entries"]} == {"F"}
    nc = api(server, "/api/experiments?letter=none")
    assert nc["count"] == nc["counts"]["not curated"] and all(e["tier"] is None for e in nc["entries"])
    fl = api(server, "/api/experiments?domain=flight%20dynamics")
    assert [e["key"] for e in fl["entries"]] == ["examples/flight_dynamics"]
    q = api(server, "/api/experiments?q=ahmed")
    assert "repo_results/ahmed_body" in {e["key"] for e in q["entries"]}
    r = api(server, "/api/experiments?ready=1")
    assert r["count"] > 0 and all(e["ready"] for e in r["entries"])
    pinn = api(server, "/api/experiments?method=PINN")
    assert all("PINN" in e["methods"] for e in pinn["entries"])
    by_date = api(server, "/api/experiments?sort=date")["entries"]
    assert [e["date"] for e in by_date] == sorted((e["date"] for e in by_date), reverse=True)
    entries = api(server, "/api/experiments")["entries"]
    assert filter_entries(entries, letter="A,B") == [e for e in entries if e["tier"] in ("A", "B")]


def test_detail_seal_stamped_figure_and_repo_files(server):
    e = api(server, "/api/experiment?key=em_pinn/em_tm_cavity")
    assert e["tier"] == "F" and e["problem"]["text"] and e["next_steps"][0]["actions"]
    assert set(e["stages"]) == {"data_geometry", "model", "physics", "benchmark", "uncertainty", "decision"}
    st, ctype, body = get(server + e["seal_en"])
    assert ctype == "image/png" and body[:4] == b"\x89PNG"
    fig = next(m for m in e["media"] if m["kind"] == "figure")
    st, ctype, body = get(server + fig["src"])
    assert ctype == "image/jpeg" and len(body) > 1000
    flight = api(server, "/api/experiment?key=examples/flight_dynamics")
    assert flight["links"]["guide"] == "docs/guides/flight_dynamics.md" and flight["curate"]
    with pytest.raises(urllib.error.HTTPError) as err:
        get(server + "/repo/pyproject.toml")                  # only files a manifest lists are served
    assert err.value.code == 404


def test_login_is_required(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        get(server + "/experiments", auth=False)
    assert err.value.code == 401
    st, ctype, body = get(server + "/experiments")
    assert st == 200 and b"PINNeAPPle experiments" in body


# ---------------------------------------------------------------------------------------------- static export
def test_static_export_builds(tmp_path):
    from pinneapple_lab import experiment
    lab = str(tmp_path / "lab")
    ex_root = tmp_path / "examples"
    study = ex_root / "tiny"
    (study / "results").mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with experiment("examples/tiny", title="Tiny", domain="numerical methods", out_dir=str(study), lab=lab) as ex:
        ex.check("u_vs_exact", value=1e-4, max=1e-3, kind="reference")
        fig, ax = plt.subplots()
        ax.imshow(np.random.default_rng(0).random((20, 30)))
        fig.savefig(str(study / "results" / "u.png"))
        plt.close(fig)
    (ex_root / "bare").mkdir()
    M.save(M.Manifest(id="examples/bare", title="Bare"), str(ex_root / "bare"))
    out = write_static(LabStore(lab), str(tmp_path / "site"), examples=str(ex_root))
    page = open(out).read()
    data = json.loads(page.split('<script id="data" type="application/json">')[1].split("</script>")[0])
    keys = {e["key"]: e for e in data["entries"]}
    assert set(keys) == {"examples/tiny", "examples/bare"} and keys["examples/bare"]["tier"] is None
    tiny = keys["examples/tiny"]
    assert tiny["tier"] in ("A", "B", "C") and os.path.exists(tmp_path / "site" / tiny["seal_en"])
    assert all(os.path.exists(tmp_path / "site" / m["src"]) and os.path.exists(tmp_path / "site" / m["thumb"])
               for m in tiny["media"]) and tiny["media"]
    assert coverage(data) and data["counts"]["not curated"] == 1


# ---------------------------------------------------------------------------------------------- browser smoke test
@pytest.mark.skipif(CHROMIUM is None, reason="needs the preinstalled Chromium (/opt/pw-browsers)")
def test_page_in_a_browser(server):
    pw = pytest.importorskip("playwright.sync_api")
    shots = os.environ.get("PNP_SCREENSHOTS")
    with pw.sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROMIUM)
        page = b.new_page(viewport={"width": 1280, "height": 900}, extra_http_headers=AUTH)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server + "/experiments")
        page.wait_for_selector(".card")
        assert page.locator(".card").count() > 100
        if shots:
            page.screenshot(path=os.path.join(shots, "experiments_list.png"))
        page.click('[data-letter="F"]')
        assert page.locator(".card").count() == int(page.locator('[data-letter="F"] b').inner_text())
        page.click('[data-letter="F"]')
        page.goto(server + "/experiments#e=repo_results%2Fahmed_body")
        page.wait_for_selector("#detail.on h2")
        assert "Ahmed" in page.locator("#detail h2").inner_text()
        if shots:
            page.screenshot(path=os.path.join(shots, "experiment_detail.png"), full_page=True)
        page.goto(server + "/experiments#e=em_pinn%2Fem_tm_cavity")
        page.wait_for_selector("#detail.on .problem")
        assert page.locator("ol.next code").count() > 0
        if shots:
            page.screenshot(path=os.path.join(shots, "experiment_F_next_steps.png"), full_page=True)
        page.click('[data-lang="pt"]')
        assert "Problema principal" in page.locator("#detail").inner_text()
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(server + "/experiments")
        page.wait_for_selector(".card")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")   # no sideways scroll
        if shots:
            page.screenshot(path=os.path.join(shots, "experiments_mobile_pt.png"))
        b.close()
    assert errors == []
