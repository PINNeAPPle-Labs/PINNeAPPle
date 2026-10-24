"""pinneapple_autolab.dashboard: live state from the session files, the page, the replay gating, the Flow node."""
import json
import os
import urllib.request

import pytest

from pinneapple_autolab import AutonomousLab, Budget, FunctionSimulator, Limit, Mission
from pinneapple_autolab.dashboard import build_dashboard, live_state, serve
from pinneapple_design.design_optimizer.study import Objective, Variable

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RECORDED = os.path.join(ROOT, "examples", "autolab_aero", "_out")


def toy(p):
    return {"drag": 1 + (p["a"] - 0.3) ** 2 + 0.5 * p["b"] ** 2, "down": 2 + p["a"] + 0.8 * p["b"]}


def small_lab(folder, fail_at=None):
    m = Mission("toy", "lower drag", [Variable("a", 0, 1), Variable("b", 0, 1)], [Objective("drag")],
                [Limit("down", lower_frac_of_baseline=1.0)], baseline={"a": 0.5, "b": 0.5}, budget=Budget(9, 1),
                acceptance={"drag": 0.03, "down": 0.03})
    sim = FunctionSimulator(toy, ["drag", "down"], fail=lambda p: "diverged" if fail_at and p["a"] > fail_at else None)
    return AutonomousLab(m, sim, str(folder), families=("gp",), nsga_budget=100, n_pool=50)


def test_live_state_reports_running_failed_and_finished(tmp_path):
    lab = small_lab(tmp_path, fail_at=0.9)
    lab.plan(4)
    lab.baseline()
    lab.ledger.event("simulate", "run002-doe: running", params={"a": 0.1, "b": 0.1}, purpose="doe", running=True)
    st = live_state(str(tmp_path))
    assert st["status"]["running"]["id"] == "run002-doe" and not st["status"]["finished"]
    lab.simulate({"a": 0.95, "b": 0.5}, "doe")                         # fails: diverged
    st = live_state(str(tmp_path))
    assert st["status"]["running"] is None and st["status"]["failed"] == ["run002-doe"]
    assert [r["ok"] for r in st["ledger"]["runs"]] == [True, False]


def test_lab_writes_a_strict_json_live_snapshot(tmp_path):
    lab = small_lab(tmp_path)
    lab.run(n_doe=4)
    snap = json.load(open(tmp_path / "snapshot_live.json"), parse_constant=lambda c: pytest.fail(f"non-JSON {c}"))
    assert snap["candidates"]["label"] and snap["limits_all"]["down"][1] is None


def test_page_and_live_server(tmp_path):
    lab = small_lab(tmp_path / "s")
    lab.run(n_doe=4)
    page = build_dashboard(str(tmp_path / "s"), str(tmp_path / "d"), replay=True)
    html = open(page).read()
    assert "Geometry" in html and "Decisions" in html and '"replay":true' in html and "/*__STATE__*/null" not in html
    srv = serve(str(tmp_path / "s"), str(tmp_path / "d2"), port=0, block=False)
    try:
        port = srv.server_address[1]
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/state.json").read())
        assert st["status"]["finished"] and len(st["ledger"]["runs"]) == 9
        assert b"Autonomous Physics Lab" in urllib.request.urlopen(f"http://127.0.0.1:{port}/").read()
    finally:
        srv.shutdown()


@pytest.mark.skipif(not os.path.exists(os.path.join(RECORDED, "events.jsonl")), reason="no recorded session")
def test_recorded_session_dashboard_gates_designs_by_their_runs(tmp_path):
    from pinneapple_autolab.domains.racecar import recorded_dashboard
    page = recorded_dashboard(RECORDED, str(tmp_path))
    html = open(page).read()
    cfg = json.loads(html.split("const CFG = ")[1].split(";\n")[0])
    summ = json.load(open(os.path.join(RECORDED, "summary.json")))
    assert cfg["designs"] == {"baseline": summ["baseline"]["id"], "best": summ["best"]["id"]}
    assert cfg["figure_runs"]["cp"] and "best_cp_top" in cfg["figures"] and cfg["limits"]["downforce_N"][1] is None


@pytest.mark.skipif(not os.path.exists(os.path.join(RECORDED, "events.jsonl")), reason="no recorded session")
def test_flow_node_recorded_session(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    import pinneapple_flow as pf
    from pinneapple_flow.registry import REGISTRY
    assert [p.name for p in REGISTRY.get("autolab.aerodynamicist").outputs] == ["report", "figure", "metrics"]
    r = pf.run({"nodes": [{"id": "aero", "type": "autolab.aerodynamicist", "params": {"source": "recorded"}}],
                "edges": []}, workdir=tmp_path / "w")
    assert r.status == "done", {k: v["error"] for k, v in r.nodes.items()}
    rep = r.report("aero")
    assert rep["metrics"]["cfd_runs"]["value"] == 20 and rep["views"]["html"]["path"].endswith("index.html")
