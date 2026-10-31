"""Known answers for the ideal reactors, verify_against_model and the scalar twin page."""
import json
import math
import re
import sys
import types

import pytest

from pinneapple_autolab.study.scalar_twin import in_hull, surrogate_twin, training_set_hash
from pinneapple_autolab.study.verify import verify_against_model
from pinneapple_physics.closed_form import ideal_reactors as R

PFR = "pinneapple_physics.closed_form.ideal_reactors:pfr_conversion_ktau"
CSTR = "pinneapple_physics.closed_form.ideal_reactors:cstr_conversion_ktau"


def _raises(k, tau):
    raise RuntimeError("solver diverged")


_raises.x_ranges = {"k": (0, 10), "tau": (0, 1e4)}


@pytest.mark.parametrize("k,tau,X", [(0.02, 120, 0.909282), (0.1, 10, 0.632121), (0.5, 4, 0.864665)])
def test_pfr_known_answers(k, tau, X):
    assert R.pfr_conversion_ktau(k, tau)["X"] == pytest.approx(X, abs=1e-6)


def test_tube_geometry():
    out = R.pfr_conversion(D=0.0275, L=5.5, k=0.02, Q=2.722305e-5)
    assert out["V"] == pytest.approx(3.266765e-3, rel=1e-6)
    assert out["tau"] == pytest.approx(120.0, rel=1e-5)
    assert out["X"] == pytest.approx(0.909282, abs=1e-6)


def test_verify_pass_and_fail():
    ok = verify_against_model({"k": 0.02, "tau": 120.0}, 0.909282, PFR, rtol=1e-4)
    assert ok["status"] == "pass" and ok["model_value"] == pytest.approx(0.909282, abs=1e-6)
    bad = verify_against_model({"k": 0.02, "tau": 120.0}, 0.95, PFR, rtol=0.01)
    assert bad["status"] == "fail" and bad["rel_error"] == pytest.approx(0.0448, abs=1e-4)
    assert {c["name"] for c in bad["verifications"]} == {"model_runs", "inside_model_ranges", "units_agree",
                                                         "agreement"}


def test_verify_inconclusive_never_pass():
    r = verify_against_model({"k": 0.02, "tau": 120.0}, 0.909282, _raises, rtol=0.5)
    assert r["status"] == "inconclusive" and r["model_value"] is None
    out = verify_against_model({"k": 20.0, "tau": 120.0}, 1.0, PFR, rtol=0.5)
    assert out["status"] == "inconclusive"
    unit = verify_against_model({"k": 0.02, "tau": 120.0}, {"value": 0.909282, "unit": "%"}, PFR, rtol=0.5)
    assert unit["status"] == "inconclusive"
    with pytest.raises(ValueError):
        verify_against_model({"k": 0.02, "tau": 120.0}, 0.9, PFR, rtol=float("nan"))


def test_cstr_is_not_a_pfr():
    k, tau = 0.02, 120.0  # k tau = 2.4
    x_cstr = R.cstr_conversion_ktau(k, tau)["X"]
    assert x_cstr == pytest.approx(0.705882, abs=1e-6)
    assert verify_against_model({"k": k, "tau": tau}, x_cstr, PFR, rtol=0.01)["status"] == "fail"


def _data(page):
    return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', open(page).read(),
                                re.S).group(1))


def test_twin_table_is_the_model(tmp_path):
    out = surrogate_twin(PFR, [{"name": "k", "unit": "1/s", "low": 0.02, "high": 0.02, "values": [0.02]},
                               {"name": "tau", "unit": "s", "low": 10, "high": 120, "values": [10, 50, 100, 120]}],
                         {"name": "X", "unit": "-"}, str(tmp_path))
    d = _data(out["page"])
    for kt, v in zip([0.2, 1.0, 2.0, 2.4], d["value"], strict=True):
        assert abs(v - (1 - math.exp(-kt))) <= 1e-9
    assert d["half"] == [0.0] * 4 and out["manifest"]["band_label"] == "no uncertainty estimate"
    assert not any(d["inside"])  # no training set given


def test_twin_ranges_and_hull(tmp_path):
    with pytest.raises(ValueError):
        surrogate_twin(PFR, [{"name": "k", "low": 0, "high": 0.1, "values": [0.2]}], {"name": "X"}, str(tmp_path))
    samples = [{"k": 0.0, "tau": 0.0}, {"k": 0.1, "tau": 0.0}, {"k": 0.0, "tau": 100.0}]  # a triangle
    out = surrogate_twin(PFR, [{"name": "k", "low": 0, "high": 0.1, "values": [0.0, 0.1]},
                               {"name": "tau", "low": 0, "high": 100, "values": [0.0, 100.0]}],
                         {"name": "X"}, str(tmp_path), samples_used=samples)
    d = _data(out["page"])
    # nodes in row-major order: (0,0) (0,100) (0.1,0) (0.1,100); the last is inside the box, outside the hull
    assert d["inside"] == [True, True, True, False]
    html = open(out["page"]).read()
    assert "extrapolation" in html and 'type="range"' not in html  # sliders are built from the grid indices
    assert "s.min = 0; s.max = v.grid.length - 1" in html and "Math.min(Math.max(" in html


def test_twin_hash_changes_iff_training_changes(tmp_path):
    names = ["k", "tau"]
    a = [{"k": 0.01, "tau": 10.0}, {"k": 0.05, "tau": 50.0}]
    assert training_set_hash(a, names) == training_set_hash([dict(s) for s in a], names)
    b = [dict(s) for s in a]
    b[1]["tau"] = 50.000001
    assert training_set_hash(a, names) != training_set_hash(b, names)
    v = [{"name": "k", "low": 0, "high": 0.1, "values": [0.01]}, {"name": "tau", "low": 0, "high": 100, "values": [10]}]
    m1 = surrogate_twin(PFR, v, {"name": "X"}, str(tmp_path / "1"), samples_used=a)["manifest"]
    m2 = surrogate_twin(PFR, v, {"name": "X"}, str(tmp_path / "2"), samples_used=a)["manifest"]
    m3 = surrogate_twin(PFR, v, {"name": "X"}, str(tmp_path / "3"), samples_used=b)["manifest"]
    assert m1 == m2 and m1["training_set_sha256"] != m3["training_set_sha256"]
    assert {k for k in m1 if m1[k] != m3[k]} == {"training_set_sha256"}


def test_twin_bands(tmp_path):
    v = [{"name": "k", "low": 0, "high": 0.1, "values": [0.05]}, {"name": "tau", "low": 0, "high": 100,
                                                                     "values": [20]}]
    out = surrogate_twin(PFR, v, {"name": "X", "unit": "-"}, str(tmp_path), band="holdout_error", holdout_error=0.01)
    assert _data(out["page"])["half"] == [0.01] and "held-out" in out["manifest"]["band_label"]
    with pytest.raises(ValueError):
        surrogate_twin(PFR, v, {"name": "X"}, str(tmp_path), band="holdout_error")


def test_twin_offline_and_escaped(tmp_path):
    mod = types.ModuleType("_evil_mod")
    mod.f = lambda **kw: {"<img src=x onerror=alert(1)>": 0.5}
    sys.modules["_evil_mod"] = mod
    name = "<img src=x onerror=alert(1)>"
    out = surrogate_twin("_evil_mod:f", [{"name": "</script><script>alert(2)</script>", "low": 0, "high": 1,
                                          "values": [0.5]}], {"name": name}, str(tmp_path))
    html = open(out["page"]).read()
    assert "<img" not in html and "</script><script>alert(2)" not in html
    assert _data(out["page"])["value"] == [0.5]
    assert not re.search(r'(src|href)="(https?:)?//', html)  # nothing loaded from the network
    assert "default-src 'none'" in html


def test_in_hull_1d_and_degenerate():
    import numpy as np
    assert list(in_hull(np.array([[0.5], [2.0]]), np.array([[0.0], [1.0]]))) == [True, False]
    assert list(in_hull(np.array([[0.0, 0.0], [0.5, 0.5]]), np.array([[0.0, 0.0], [1.0, 1.0]]))) == [True, False]


def test_twin_page_in_a_browser(tmp_path):
    """Offline (every request blocked), sliders pinned to the grid, names shown as text."""
    sync_api = pytest.importorskip("playwright.sync_api")
    samples = [{"k<b>": 0.0, "tau": 0.0}, {"k<b>": 0.1, "tau": 0.0}, {"k<b>": 0.0, "tau": 100.0}]
    mod = types.ModuleType("_renamed_pfr")
    mod.f = lambda tau, **kw: R.pfr_conversion_ktau(kw["k<b>"], tau)
    sys.modules["_renamed_pfr"] = mod
    out = surrogate_twin("_renamed_pfr:f", [{"name": "k<b>", "unit": "1/s", "low": 0, "high": 0.1, "values": [0.0, 0.1]},
                               {"name": "tau", "unit": "s", "low": 0, "high": 100, "values": [0.0, 100.0]}],
                         {"name": "X", "unit": "-"}, str(tmp_path), samples_used=samples)
    requests = []
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no browser: {e}")
        page = browser.new_page()
        page.route("**/*", lambda route: (requests.append(route.request.url), route.continue_()
                                          if route.request.url.startswith("file:") else route.abort()))
        page.goto("file://" + out["page"])
        assert page.locator("label span").first.text_content() == "k<b> [1/s]"
        sliders = page.locator("input[type=range]")
        for j in range(2):  # past the top: the browser clamps it to the last grid index
            sliders.nth(j).evaluate("s => { s.value = 99; s.dispatchEvent(new Event('input')); }")
        assert page.locator("body").get_attribute("data-node") == "3"
        assert "extrapolation" in page.locator("#flag").text_content()
        assert float(page.locator("#value").text_content()) == pytest.approx(-math.expm1(-10.0), abs=1e-5)
        sliders.nth(0).evaluate("s => { s.value = -5; s.dispatchEvent(new Event('input')); }")
        assert page.locator("body").get_attribute("data-node") == "1"
        assert page.locator("#flag").text_content() == "inside the training set's hull"
        browser.close()
    assert all(u.startswith("file:") for u in requests)


def test_outputs_declared_with_units():
    import pinneapple_autolab.study.scalar_twin as T
    import pinneapple_autolab.study.verify as V
    from pinneapple_autolab.study.verify import output_unit
    for mod, tools in [(R, ["tube_volume", "pfr_conversion", "pfr_conversion_ktau", "cstr_conversion_ktau"]),
                       (V, ["verify_against_model"]), (T, ["surrogate_twin"])]:
        for name in tools:
            rows = mod.X_OUTPUTS[name]
            assert rows and all(set(r) == {"name", "type", "unit", "description"} for r in rows)
    assert output_unit(R.pfr_conversion, "tau") == "s"
    assert output_unit(R.tube_volume, "V") == "m3"  # no attribute: read from the module's X_OUTPUTS


def test_open_sided_range():
    def model(k, tau):
        return {"X": -math.expm1(-k * tau)}

    model.x_ranges = {"k": (0.0, None), "tau": (None, 1e4)}
    assert verify_against_model({"k": 50.0, "tau": -1.0}, -math.expm1(50.0), model, rtol=1e-9)["status"] == "pass"
    assert verify_against_model({"k": -1.0, "tau": 1.0}, 0.0, model, rtol=1.0)["status"] == "inconclusive"
