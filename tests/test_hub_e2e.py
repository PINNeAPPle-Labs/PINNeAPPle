"""Playwright test of the PINNeAPPle Hub UI (apps/hub): home, "/" and Ctrl+K, filters, an item page (copy code,
recorded run, 3-D preview, downloads), browser inference of a zoo model, "Create your own" (blocks, code,
sandbox validation, save to My Hub), the dark theme, empty and error states, and the phone layout. Skipped when
Playwright or a Chromium build is missing (it never downloads a browser). HUB_SCREENSHOTS=<folder> saves the
screenshots used in the docs."""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pw = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[1]
CHROMIUM = [os.environ.get("PLAYWRIGHT_CHROMIUM", ""), "/opt/pw-browsers/chromium"]
SHOTS = os.environ.get("HUB_SCREENSHOTS")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    os.environ["HUB_DATA_DIR"] = str(tmp_path_factory.mktemp("hub"))
    sys.path.insert(0, str(ROOT / "apps" / "hub"))
    import uvicorn
    from hub.api import app
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(5)


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        b = None
        for exe in [c for c in CHROMIUM if c and Path(c).exists()] + [None]:
            try:
                b = p.chromium.launch(executable_path=exe, args=["--use-gl=swiftshader", "--ignore-gpu-blocklist"]) \
                    if exe else p.chromium.launch()
                break
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
        if b is None:
            pytest.skip("no Chromium build for Playwright")
        yield b
        b.close()


def _page(browser, server, **ctx):
    c = browser.new_context(**{"viewport": {"width": 1440, "height": 900}, "bypass_csp": True, **ctx})
    c.grant_permissions(["clipboard-read", "clipboard-write"], origin=server)
    pg = c.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    return pg


def _shot(pg, name, full=True):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        pg.add_style_tag(content=".hub-top { position: static !important; }")   # one header in a full-page shot
        pg.screenshot(path=str(Path(SHOTS) / f"{name}.png"), full_page=full)


def test_home_search_filters_and_shortcuts(browser, server):
    pg = _page(browser, server, color_scheme="light")
    pg.goto(server + "/")
    pg.wait_for_selector(".hero")
    assert pg.locator(".card").count() >= 6
    assert pg.locator(".cat").count() == 6
    pg.wait_for_timeout(500)
    _shot(pg, "hub_home_desktop")
    pg.keyboard.press("/")
    assert pg.evaluate("document.activeElement.id") == "q"
    pg.keyboard.type("heat sink")
    pg.keyboard.press("Enter")
    pg.wait_for_selector(".resultbar")
    assert "heat sink" in pg.inner_text(".activefilters")
    assert pg.locator(".card").count() > 0
    pg.check("input[name=kind][value=model]")
    pg.wait_for_function("location.search.includes('kind=model')")
    pg.wait_for_selector(".card")
    assert all(t == "TRAINED MODEL" for t in pg.locator(".card .badge.kind").all_inner_texts())
    pg.goto(server + "/?q=zzzz-nothing")
    pg.wait_for_selector(".state")
    assert "No matches" in pg.inner_text(".state")
    pg.goto(server + "/?kind=example&quality=A")
    pg.wait_for_selector(".card")
    _shot(pg, "hub_browse_desktop")
    pg.keyboard.press("Control+k")
    pg.wait_for_selector("#palette:not([hidden])")
    pg.keyboard.type("wind turbine")
    pg.wait_for_selector("#pres li[data-i]")
    pg.keyboard.press("Enter")
    pg.wait_for_selector("h1")
    assert "wind turbine" in pg.inner_text("h1").lower()
    pg.goto(server + "/no/such/page")
    pg.wait_for_selector(".state.err")
    assert not pg.errors, pg.errors


def test_item_page_copy_code_preview_and_downloads(browser, server):
    pg = _page(browser, server)
    pg.goto(server + "/item/geometry%3Awind_turbine_nrel5mw")
    pg.wait_for_selector("h1")
    pg.click("#copymain")
    pg.wait_for_selector("#toast.on")
    assert 'pp.geometries.load("wind_turbine_nrel5mw"' in pg.evaluate("navigator.clipboard.readText()")
    assert "recorded run passed" in pg.inner_text(".runhead").lower()
    pg.wait_for_selector("#v3d canvas")
    pg.wait_for_function("!document.querySelector('#v3d .vmsg')", timeout=30000)
    with pg.expect_download() as d:
        pg.click(".files a[download] >> nth=0")
    assert d.value.suggested_filename.endswith((".glb", ".stl"))
    _shot(pg, "hub_item_geometry")
    pg.goto(server + "/item/" + "example%3Aaccretion_flow")
    pg.wait_for_selector(".stages .stage")
    assert pg.locator(".stage").count() == 6
    assert not pg.errors, pg.errors


def test_zoo_model_runs_in_the_browser_and_on_the_server(browser, server):
    pg = _page(browser, server)
    pg.goto(server + "/item/model%3Aheatsink_forced_mlp")
    pg.wait_for_selector("#run-browser")
    pg.click("#run-browser")
    pg.wait_for_selector("#tryout .verdict", timeout=60000)
    browser_out = pg.inner_text("#tryout table")
    pg.click("#run-server")
    pg.wait_for_function("document.querySelector('#tryout .hint')?.textContent.includes('server')", timeout=60000)
    server_out = pg.inner_text("#tryout table")
    first = lambda t: [ln.split("\t")[1] for ln in t.splitlines()[1:]]  # noqa: E731
    for a, b in zip(first(browser_out), first(server_out), strict=True):
        assert float(a) == pytest.approx(float(b), rel=1e-4)
    _shot(pg, "hub_item_model")
    assert not pg.errors, pg.errors


def test_create_your_own_validate_and_save(browser, server):
    pg = _page(browser, server)
    pg.goto(server + "/create")
    pg.wait_for_selector("#c-blocks")
    assert "No blocks yet" in pg.inner_text("#c-chain")
    pg.fill("#c-name", "Rod grid")
    pg.fill("#c-filter", "domain")
    pg.click("[data-add='geometry.domain']")
    pg.fill("#c-filter", "mesh")
    assert pg.locator("[data-add='geometry.domain']").is_hidden()
    pg.click("[data-add='geometry.mesh']")
    pg.wait_for_function("document.querySelector('#c-code pre')?.textContent.includes('geometry.mesh')")
    assert "the flow validates" in pg.inner_text("#c-issues")
    pg.click("[data-tab=json]")
    assert '"schema": "pinneapple-flow/1"' in pg.inner_text("#c-code pre")
    pg.click("[data-tab=py]")
    pg.click("#c-validate")
    pg.wait_for_selector("#c-run .badge.ok", timeout=180000)
    assert "mesh done" in pg.inner_text("#c-run")
    _shot(pg, "hub_create")
    pg.click("#c-save")
    pg.wait_for_selector("h1")
    assert pg.inner_text("h1") == "Rod grid"
    pg.goto(server + "/mine")
    pg.wait_for_selector(".card")
    assert "Rod grid" in pg.inner_text(".grid")
    assert not pg.errors, pg.errors


def test_dark_theme_and_phone_layout(browser, server):
    pg = _page(browser, server, color_scheme="dark")
    pg.goto(server + "/")
    pg.wait_for_selector(".hero")
    bg = pg.evaluate("getComputedStyle(document.body).backgroundColor")
    assert bg == "rgb(11, 17, 32)"
    pg.wait_for_timeout(400)
    _shot(pg, "hub_home_dark")
    pg.click("#theme")
    assert pg.evaluate("document.documentElement.dataset.theme") == "light"
    phone = _page(browser, server, viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True,
                  has_touch=True)
    for path in ("/", "/item/model%3Aairfoil_cst_rans_mlp", "/item/geometry%3Af1_car", "/item/flow%3Aheat_1d", "/create"):
        phone.goto(server + path)
        phone.wait_for_selector("main")
        phone.wait_for_timeout(400)
        assert phone.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), path
    phone.goto(server + "/")
    phone.wait_for_selector(".hero")
    phone.wait_for_timeout(400)
    _shot(phone, "hub_home_phone")
    phone.goto(server + "/item/geometry%3Af1_car")
    phone.wait_for_selector("#v3d canvas")
    phone.wait_for_timeout(1500)
    _shot(phone, "hub_item_phone")
    assert not pg.errors and not phone.errors, pg.errors + phone.errors
