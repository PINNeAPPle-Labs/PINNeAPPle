"""Library-call nodes (curated allowlist + introspected forms), user-defined node types (safe: only allowlisted
calls), and the coverage table of the library."""
import pytest

import pinneapple_flow as pf
from pinneapple_flow.libcalls import CALLS, custom_spec, introspect


def test_every_allowlisted_call_resolves_and_has_a_form():
    for key, call in CALLS.items():
        assert call.target.split(":")[0].startswith("pinneapple"), key
        spec = pf.REGISTRY[call.node_type or f"lib.{key}"]
        ok, why = spec.available()
        if ok:
            params, _skipped = introspect(call)
            assert spec.params_schema()["type"] == "object"
            assert all(p.kind for p in params)
        else:
            assert "needs" in why


@pytest.mark.parametrize("type_,params,key", [
    ("lib.solver.terramechanics", {"soil": "GRC-1", "slip": 0.2}, "dp_over_w"),
    ("lib.solver.beam", {}, "max_abs_deflection_m"),
    ("lib.solve.preset", {"preset": "burgers_1d", "method": "analytic"}, "method"),
    ("lib.model.architecture", {"name": "fno"}, "parameters"),
    ("lib.doe.design", {"parameters": {"a": [0, 1], "b": [1, 10]}, "n": 8}, "n_cases"),
    ("lib.export.geometry", {"name": "caarc_building", "fmt": "stl"}, "bytes"),
])
def test_library_nodes_run(tmp_path, type_, params, key):
    r = pf.run(pf.Flow("lib", nodes=[{"id": "n", "type": type_, "params": params}]), workdir=tmp_path)
    assert r.status == "done", r.nodes["n"]["error"]
    assert key in r.output("n", "result")


def test_library_node_rejects_unknown_parameters():
    issues = pf.validate_flow({"schema": "pinneapple-flow/1", "name": "x", "edges": [],
                               "nodes": [{"id": "n", "type": "lib.solver.beam", "params": {"__class__": 1}}]})
    assert any("unknown parameter" in i.message for i in issues)


def test_custom_node_types_are_safe(tmp_path):
    spec = custom_spec({"type": "custom.soft_soil", "base": "solver.terramechanics", "title": "Wheel on JSC-1A",
                        "params": {"soil": {"default": "JSC-1A", "hidden": True}, "slip": 0.3}})
    assert "soil" not in [p.name for p in spec.params]
    assert next(p for p in spec.params if p.name == "slip").default == 0.3
    with pytest.raises(ValueError, match="registered library call"):
        custom_spec({"type": "custom.evil", "base": "os:system"})
    with pytest.raises(ValueError, match="no parameter"):
        custom_spec({"type": "custom.xx", "base": "solver.beam", "params": {"cmd": "rm -rf /"}})
    with pytest.raises(ValueError):
        custom_spec({"type": "lib.solver.beam", "base": "solver.beam"})
    with pytest.raises(ValueError):
        custom_spec({"type": "custom.xx", "base": "solver.terramechanics", "params": {"soil": {"default": "moon"}}})
    flow = pf.Flow("c", nodes=[{"id": "w", "type": "custom.soft_soil", "params": {}}],
                   meta={"custom_nodes": [{"type": "custom.soft_soil", "base": "solver.terramechanics",
                                           "params": {"soil": {"default": "JSC-1A", "hidden": True}}}]})
    r = pf.run(flow, workdir=tmp_path)
    assert r.status == "done", r.nodes["w"]["error"]


def test_custom_nodes_through_the_api(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    from pinneapple_flow.studio import load_app
    app = load_app()
    with TestClient(app) as c:
        assert any(x["key"] == "solver.beam" for x in c.get("/api/library-calls").json())
        r = c.post("/api/custom-nodes", json={"type": "custom.long_beam", "base": "solver.beam",
                                              "title": "2 m beam", "params": {"L_m": {"default": 2.0,
                                                                                      "hidden": True}}})
        assert r.status_code == 200 and r.json()["type"] == "custom.long_beam"
        assert c.post("/api/custom-nodes", json={"type": "custom.bad", "base": "subprocess:run"}).status_code == 400
        types = {n["type"] for n in c.get("/api/catalog").json()["nodes"]}
        assert "custom.long_beam" in types
        assert c.get("/api/custom-nodes").json()[0]["type"] == "custom.long_beam"
        assert c.delete("/api/custom-nodes/custom.long_beam").status_code == 200
    app.state.service.shutdown()


def test_coverage_table_names_every_package():
    from pinneapple_flow.coverage import coverage, markdown, packages
    rows = coverage()
    mods = {r["module"] for r in rows}
    for p in packages():
        assert p in mods
    pkg_rows = [r for r in rows if r["module"] in packages()]
    assert all(r["nodes"] or r["reason"] for r in pkg_rows), [r["module"] for r in pkg_rows
                                                              if not (r["nodes"] or r["reason"])]
    assert "| package / module |" in markdown()


def test_file_writing_calls_write_in_the_run_folder_only(tmp_path):
    from pinneapple_flow.libcalls import introspect as intro
    assert "out_dir" not in {p.name for p in intro(CALLS["export.geometry"])[0]}   # never a flow parameter
    r = pf.run(pf.Flow("g", nodes=[{"id": "g", "type": "lib.export.geometry", "params": {"fmt": "usda"}}]),
               workdir=tmp_path)
    path = r.nodes["g"]["report"]["metrics"]["file"]
    path = path["value"] if isinstance(path, dict) else path
    assert str(tmp_path) in path and path.endswith(".usda")
    with pytest.raises(pf.FlowError):
        pf.check_flow(pf.Flow("g", nodes=[{"id": "g", "type": "lib.export.geometry", "params": {"out_dir": "/"}}]))


def test_no_public_module_is_left_unreached():
    """Every package and first-level module has a node or a reason: a new public module fails here until it gets a
    node (python -m pinneapple_flow autocalls) or a curated reason in pinneapple_flow/coverage.py."""
    from pinneapple_flow.coverage import counts, coverage
    rows = coverage()
    assert counts(rows)["unreached"] == 0, [r["module"] for r in rows if not r["nodes"] and not r["reason"]]
    assert not any(r["reason"] in (None, "None") for r in rows)


def test_generated_nodes_are_safe_and_run(tmp_path):
    from pinneapple_flow.autocalls import RISKY_NAME, RISKY_PARAM, load
    data = load()
    assert len(data["calls"]) > 100
    for c in data["calls"]:
        assert c["target"].startswith("pinneapple") and c["key"].startswith("auto.")
        assert not RISKY_NAME.search(c["target"].split(":")[1]), c["target"]
        assert not any(RISKY_PARAM.search(p["name"]) for p in c["params"]), c["target"]
        assert c["check"].startswith(("runs with its defaults", "needs input")), c["key"]
        assert pf.REGISTRY.get(c["key"]) is not None
    key = "auto.analysis.data_assimilation.twin_experiment"
    assert key in pf.REGISTRY
    r = pf.run(pf.Flow("auto", nodes=[{"id": "n", "type": key, "params": {"n": 16, "window_steps": 4}}]),
               workdir=tmp_path)
    assert r.status == "done", r.nodes["n"]["error"]
    assert r.nodes["n"]["report"]["metrics"]


def test_committed_coverage_table_is_current():
    from pathlib import Path

    from pinneapple_flow.coverage import ROOT, markdown
    assert (ROOT / "docs/flow_studio/COVERAGE.md").read_text() == markdown(), \
        "regenerate: python -m pinneapple_flow coverage -o docs/flow_studio/COVERAGE.md"
    assert Path(ROOT / "docs/flow_studio/COVERAGE.md").is_file()
