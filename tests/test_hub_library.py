"""The Hub index (pinneapple_hub.library): generated from the sources and current, every snippet has a passing
recorded run of its current code (and a sample per kind runs again here; PINNEAPPLE_HUB_RUN_SNIPPETS=1 runs all
of them), every file, figure and link it points to exists, search and filters, the composer, My Hub, the sandbox."""
from __future__ import annotations

import json
import os

import pytest

from pinneapple_hub import library as hub
from pinneapple_hub.library.sandbox import code_hash, run_snippet

INDEX = hub.load_index()
ITEMS = INDEX["items"]


def test_index_is_generated_and_current():
    from pinneapple_flow import REGISTRY
    test_nodes = {f"node:{s.type}" for s in REGISTRY.specs()     # registered by other test modules in this process
                  if getattr(s.fn, "__module__", "").startswith(("test_", "tests."))}
    fresh = hub.build()
    fresh["items"] = [i for i in fresh["items"] if i["id"] not in test_nodes]
    assert [i["id"] for i in fresh["items"]] == [i["id"] for i in ITEMS], "python -m pinneapple_hub.library build"
    for a, b in zip(fresh["items"], ITEMS, strict=True):
        assert a["code"] == b["code"], f"{a['id']}: snippet changed; rebuild and re-run the index"
        assert a.get("category") == b.get("category"), f"{a['id']}: category rules changed; rebuild the index"
    assert len({i["id"] for i in ITEMS}) == len(ITEMS)
    assert INDEX["schema"] == hub.SCHEMA


def test_every_tool_is_in_exactly_one_of_20_task_categories():
    from pinneapple_hub.library import categories as cat
    tools = [i for i in ITEMS if i["kind"] == "tool"]
    assert len(cat.CATEGORIES) == 20 and len(set(cat.SLUGS)) == 20
    assert [c["slug"] for c in INDEX["categories"]] == cat.SLUGS
    homeless = [cat.key(i) for i in tools if i.get("category") not in cat.SLUGS]
    assert not homeless, f"add a rule (or a RECIPES entry) in pinneapple_hub/library/categories.py for {homeless}"
    assert all("category" not in i for i in ITEMS if i["kind"] != "tool")
    counts = INDEX["facets"]["category"]
    assert set(counts) == set(cat.SLUGS) and sum(counts.values()) == len(tools)
    keys = {cat.key(i): i["category"] for i in tools}
    dead = [(s, p) for s, _t, _d, ps in cat.CATEGORIES for p in ps
            if not any(k.startswith(p) and v == s for k, v in keys.items())]
    assert not dead, f"rules that place no tool (a typo, or a tool that moved): {dead}"
    assert not set(cat.RECIPES) - set(keys), "RECIPES lists recipes that are no longer in the index"


def test_category_rules_pick_the_most_specific_prefix():
    from pinneapple_hub.library.categories import category_of
    node = lambda t: {"id": f"node:{t}", "kind": "tool", "subkind": "flow node", "name": t}  # noqa: E731
    lib = lambda m: {"id": "node:auto.x", "kind": "tool", "subkind": "library call", "name": "auto.x",  # noqa: E731
                     "extra": {"calls": m}}
    assert category_of(node("lib.solver.beam")) == "classical"
    assert category_of(node("lib.solver.terramechanics")) == "vehicles"      # longer prefix wins
    assert category_of(node("forecast.enso")) == "time-series" and category_of(node("forecast.lab")) == "verification"
    assert category_of(lib("pinneapple_design.geometry:naca")) == "geometry"
    assert category_of(lib("pinneapple_design.topology:simp")) == "design"
    assert category_of(node("nothing.like.this")) is None
    assert category_of({"id": "recipe:pinneapple_new__9", "kind": "tool", "subkind": "recipe"}) is None


def test_search_by_category():
    uq = hub.search(INDEX, "", category="uq")
    assert uq and all(i["category"] == "uq" for i in uq)
    assert {"node:lib.uq.sobol", "node:uq.sensitivity"} <= {i["id"] for i in uq}
    assert hub.search(INDEX, "sobol", category="geometry") == []
    assert hub.search(INDEX, "uncertainty sensitivity")[0]["category"] == "uq"    # the category title is searched

def test_every_kind_is_present_with_its_sources():
    c = INDEX["counts"]
    assert c["model"] >= 4 and c["geometry"] >= 17 and c["dataset"] >= 10 and c["flow"] >= 10
    assert c["example"] >= 60 and c["tool"] >= 100
    sub = {i.get("subkind") for i in ITEMS if i["kind"] == "tool"}
    assert {"function", "recipe", "flow node", "agent tool (MCP)"} <= sub
    for it in ITEMS:
        for k in ("id", "kind", "title", "summary", "domain", "tags", "code", "run", "files", "links", "examples"):
            assert k in it, (it["id"], k)
        assert it["code"].strip(), it["id"]


def test_every_snippet_has_a_passing_run_and_every_path_exists():
    assert hub.check(INDEX) == []


def test_quality_comes_from_the_lab_and_the_zoo():
    from pinneapple_lab.seal import load_curation
    cur = load_curation(str(hub.ROOT / "lab"))["items"]
    lab = [i for i in ITEMS if i["kind"] == "example" and i["extra"]["origin"] == "lab"]
    assert lab
    for it in lab:
        assert it["quality"]["letter"] == cur[it["name"]]["tier"]
        assert set(it["quality"]["stages"]) <= {"data_geometry", "model", "physics", "benchmark", "uncertainty",
                                                "decision"}
    for it in (i for i in ITEMS if i["kind"] == "model"):
        card = json.loads((hub.ROOT / next(f["path"] for f in it["files"] if f["path"].endswith("model_card.json")))
                          .read_text())
        assert it["quality"]["decision"] == card["trust_report"]["decisao"]
        assert {m["name"]: m["value"] for m in it["results"]["metrics"]} == {
            k: v for k, v in card["validation_metrics"].items() if isinstance(v, (int, float))}


def test_recorded_model_run_matches_a_fresh_prediction():
    """The number shown as "already run" for a model is what pp.zoo predicts on the same row."""
    import numpy as np

    import pinneapple as pp
    it = hub.get(INDEX, "model:heatsink_forced_mlp")
    row = it["results"]["reference_row"]["inputs"]
    out = pp.zoo.load("heatsink_forced_mlp").predict(**{k: [round(v, 6)] for k, v in row.items()})
    shown = it["run"]["stdout"].splitlines()[0]
    for k, v in out.items():
        assert f"'{k}': [{np.round(float(np.asarray(v).ravel()[0]), 5)}]" in shown


def _sample() -> list[dict]:
    if os.environ.get("PINNEAPPLE_HUB_RUN_SNIPPETS") == "1":
        return [i for i in ITEMS if os.environ.get("PINNEAPPLE_HUB_SLOW") == "1"
                or (i["run"].get("seconds") or 0) < hub.SLOW_S]
    seen, out = set(), []
    for it in ITEMS:
        key = (it["kind"], it.get("subkind"), it["extra"].get("origin") if it["kind"] == "example" else "")
        if key not in seen and (it["run"].get("seconds") or 0) < 30:
            seen.add(key)
            out.append(it)
    return out


@pytest.mark.parametrize("item", _sample(), ids=lambda i: i["id"])
def test_snippet_runs_again(item):
    r = run_snippet(item["code"], timeout=300)
    assert r["ok"], f"{item['id']}\n{r['stderr']}"
    assert r["code_hash"] == item["run"]["code_hash"]


def test_search_and_filters():
    hits = hub.search(INDEX, "airfoil")
    assert "model:airfoil_cst_rans_mlp" in [h["id"] for h in hits[:5]]
    assert all(h["kind"] == "geometry" for h in hub.search(INDEX, "", kind="geometry"))
    a = hub.search(INDEX, "", quality="A")
    assert a and all(hub.quality_key(h) == "A" for h in a)
    assert {hub.quality_key(h) for h in hub.search(INDEX, "", quality="A,B")} == {"A", "B"}
    assert hub.search(INDEX, "zzzz-no-such-thing") == []
    d = hub.search(INDEX, "", domain="heat transfer")
    assert d and all(h["domain"] == "heat transfer" for h in d)
    assert set(INDEX["featured"]) <= {i["id"] for i in ITEMS}


def test_compose_wires_blocks_and_its_code_runs(tmp_path):
    from pinneapple_hub.library.compose import ComposeError, compose
    out = compose({"name": "Rod grid", "blocks": [{"type": "geometry.domain", "params": {"dimension": 1}},
                                                  {"type": "geometry.mesh", "params": {"n": 41}}]})
    assert out["ok"] and out["issues"] == []
    assert [e["target_port"] for e in out["flow"]["edges"]] == ["geometry"]
    r = run_snippet(out["code"], timeout=300)
    assert r["ok"], r["stderr"]
    assert "mesh done {'nodes': 41}" in r["stdout"]
    bad = compose({"blocks": [{"type": "geometry.mesh", "params": {"n": 1}}]})
    assert not bad["ok"] and any("n must be >=" in i["message"] for i in bad["issues"])
    with pytest.raises(ComposeError):
        compose({"blocks": [{"type": "os.system"}]})
    with pytest.raises(ComposeError):
        compose({"blocks": [], "geometry": "../etc"})
    start = compose({"name": "Heat + more", "start": "heat_1d", "blocks": []})
    assert len(start["flow"]["nodes"]) == 8 and start["ok"]


def test_my_hub_round_trip(tmp_path):
    from pinneapple_hub.library.compose import compose
    from pinneapple_hub.library.myhub import MyHub
    out = compose({"name": "Rod grid", "blocks": [{"type": "geometry.domain"}]})
    mine = MyHub(tmp_path)
    run = {"ok": True, "stdout": "x", "seconds": 1.0, "code_hash": code_hash(out["code"])}
    it = mine.save(out, name="Rod grid", run=run)
    assert it["id"] == "mine:rod_grid" and it["run"]["status"] == "ok"
    assert mine.save(out, name="Rod grid", run={**run, "code_hash": "other"})["run"]["status"] == "not run"
    assert [i["id"] for i in mine.list()] == ["mine:rod_grid"]
    assert mine.get("mine:rod_grid")["flow"] == out["flow"]
    with pytest.raises(KeyError):
        mine.get("mine:../x")
    mine.delete("mine:rod_grid")
    assert mine.list() == []


def test_sandbox_time_limit_and_figures():
    r = run_snippet("import time\nwhile True: time.sleep(0.1)", timeout=2)
    assert not r["ok"] and r["timed_out"]
    r = run_snippet("import matplotlib.pyplot as plt\nplt.plot([0, 1])\nprint('drawn')", timeout=120)
    assert r["ok"] and r["stdout"] == "drawn" and r["figures"] and r["figures"][0][:4] == b"\x89PNG"
