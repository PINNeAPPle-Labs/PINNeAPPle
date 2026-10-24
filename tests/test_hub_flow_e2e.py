"""Playwright round trip between the PINNeAPPle Hub (apps/hub) and Flow Studio (apps/flow_studio): run a library call
live from its Hub page, open a node as a ready flow in Flow Studio, add a node to an existing flow, publish a flow
from Flow Studio to the Hub, expose a parameter, validate and save it to My Hub, and build a tool on the Flow Studio
canvas embedded in the Hub's "Create your own". Skipped without Playwright, a
Chromium build or the built Flow Studio frontend. HUB_SCREENSHOTS=<folder> saves the screenshots."""
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
DIST = ROOT / "apps" / "flow_studio" / "frontend" / "dist" / "index.html"
CHROMIUM = [os.environ.get("PLAYWRIGHT_CHROMIUM", ""), "/opt/pw-browsers/chromium"]
SHOTS = os.environ.get("HUB_SCREENSHOTS")


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _serve(app, port):
    import uvicorn
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(300):
        if srv.started:
            break
        time.sleep(0.1)
    return srv, th


@pytest.fixture(scope="module")
def servers(tmp_path_factory):
    if not DIST.is_file():
        pytest.skip("Flow Studio frontend not built (apps/flow_studio/frontend: npm run build)")
    hub_port, fs_port = _port(), _port()
    hub_url, fs_url = f"http://127.0.0.1:{hub_port}", f"http://127.0.0.1:{fs_port}"
    keys = ("HUB_DATA_DIR", "HUB_FLOW_STUDIO_URL", "PINNEAPPLE_FLOW_HOME", "FLOW_HUB_URL", "FLOW_FRAME_ANCESTORS")
    saved = {k: os.environ.get(k) for k in keys}          # restored at teardown: later tests build their own apps
    os.environ["HUB_DATA_DIR"] = str(tmp_path_factory.mktemp("hub_rt"))
    os.environ["HUB_FLOW_STUDIO_URL"] = fs_url
    os.environ["PINNEAPPLE_FLOW_HOME"] = str(tmp_path_factory.mktemp("flow_rt"))
    os.environ["FLOW_HUB_URL"] = hub_url
    os.environ["FLOW_FRAME_ANCESTORS"] = hub_url          # the Hub embeds the Flow Studio canvas
    sys.path.insert(0, str(ROOT / "apps" / "hub"))
    from hub.api import app as hub_app

    from pinneapple_flow.studio import load_app
    started = [_serve(hub_app, hub_port), _serve(load_app(), fs_port)]
    yield hub_url, fs_url
    for srv, th in started:
        srv.should_exit = True
        th.join(5)
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def context(servers):
    with pw.sync_playwright() as p:
        b = None
        for exe in [c for c in CHROMIUM if c and Path(c).exists()] + [None]:
            try:
                b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
                break
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
        if b is None:
            pytest.skip("no Chromium build for Playwright")
        c = b.new_context(viewport={"width": 1440, "height": 900}, bypass_csp=True)
        c.errors = []
        c.on("weberror", lambda e: c.errors.append(str(e.error)))
        yield c
        b.close()


def _shot(pg, name, full_page=True):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        pg.add_style_tag(content=".hub-top { position: static !important; }")
        pg.screenshot(path=str(Path(SHOTS) / f"{name}.png"), full_page=full_page)


def test_library_call_runs_live_with_the_users_values(servers, context):
    hub_url, _ = servers
    pg = context.new_page()
    pg.goto(hub_url + "/item/node%3Aauto.systems.process_components.sn_curve_cycles_to_failure")
    pg.wait_for_selector("#node-run")
    pg.fill("#node [data-k=sigma_a]", "200")
    pg.fill("#node [data-k=A]", "12")
    pg.fill("#node [data-k=m]", "3")
    pg.click("#node-run")
    pg.wait_for_selector("#node-out .runhead .badge", timeout=180000)
    out = pg.inner_text("#node-out")
    # N_f = 10**(A - m log10(sigma_a)) = 10**(12 - 3 log10 200) = 1.25e5 cycles
    value = float(out.split("{'result': ")[1].split("}")[0])
    assert value == pytest.approx(10 ** (12 - 3 * __import__("math").log10(200)), rel=1e-9)
    _shot(pg, "hub_node_run")


def test_open_a_node_as_a_flow_and_add_it_to_a_flow(servers, context):
    hub_url, _ = servers
    pg = context.new_page()
    pg.goto(hub_url + "/item/node%3Ageometry.mesh")
    pg.wait_for_selector("#node-open")
    pg.fill("#node [data-k=n]", "41")
    with context.expect_page() as opened:
        pg.click("#node-open")
    fs = opened.value
    fs.wait_for_selector(".react-flow__node", timeout=60000)
    fs.wait_for_function("document.querySelectorAll('.react-flow__node').length === 3")
    _shot(fs, "flow_studio_from_hub")
    with context.expect_page() as added:
        pg.click("#node-add")
    fs2 = added.value
    fs2.wait_for_selector("[data-testid=pending-add]", timeout=60000)
    fs2.click("[data-testid=example-heat_1d]")
    fs2.wait_for_function("document.querySelectorAll('.react-flow__node').length === 9", timeout=60000)


def test_publish_a_flow_to_the_hub_validate_and_save(servers, context):
    hub_url, fs_url = servers
    fs = context.new_page()
    fs.goto(fs_url + "/?example=antenna_vivaldi_limits")
    fs.wait_for_selector("[data-testid=publish-hub]", timeout=60000)
    with context.expect_page() as pub:
        fs.click("[data-testid=publish-hub]")
    pg = pub.value
    pg.wait_for_selector("[data-expose]", state="attached", timeout=60000)
    box = pg.locator("[data-expose]").first
    name = box.get_attribute("data-expose").split("|")[1]
    box.locator("xpath=ancestor::details").evaluate("d => d.open = true")
    box.check()
    pg.wait_for_function(f"document.querySelector('#c-code pre')?.textContent.includes('# parameters of this tool') && "
                         f"document.querySelector('#c-code pre').textContent.includes('{name}')")
    pg.click("#c-validate")
    pg.wait_for_selector("#c-run .badge.ok", timeout=300000)
    _shot(pg, "hub_publish_from_flow_studio")
    pg.click("#c-save")
    pg.wait_for_selector("h1")
    assert "Vivaldi" in pg.inner_text("h1")
    assert "recorded run passed" in pg.inner_text(".runhead").lower()
    assert not context.errors, context.errors


def test_build_on_the_embedded_flow_studio_canvas(servers, context):
    import base64
    import json
    hub_url, _ = servers
    flow = json.loads((ROOT / "pinneapple_flow" / "flows" / "antenna_vivaldi_limits.json").read_text())
    b64 = base64.urlsafe_b64encode(json.dumps(flow).encode()).decode().rstrip("=")
    pg = context.new_page()
    pg.goto(f"{hub_url}/create?mode=canvas&flow={b64}")
    frame = pg.frame_locator("#c-frame")
    frame.locator(".react-flow__node").first.wait_for(timeout=60000)      # the Hub sent its flow to the canvas
    assert frame.locator(".react-flow__node").count() == len(flow["nodes"])
    frame.locator("text=Domain").first.dblclick()                          # edit on the canvas ...
    pg.wait_for_function("document.querySelector('#c-code pre')?.textContent.includes('geometry.domain')")
    assert f"{len(flow['nodes']) + 1} nodes" in pg.inner_text("#c-chain")  # ... and the Hub has the new flow
    _shot(pg, "hub_canvas_mode", full_page=False)                          # a full-page capture blanks the iframe
    pg.click("#c-validate")
    pg.wait_for_selector("#c-run .badge.ok", timeout=300000)
    _shot(pg, "hub_canvas_validated")
    pg.click("#c-save")
    pg.wait_for_selector("h1")
    assert not context.errors, context.errors
