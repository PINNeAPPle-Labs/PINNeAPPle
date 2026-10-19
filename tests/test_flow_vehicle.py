"""Rover nodes (pinneapple_flow.nodes.vehicle): typed ports, palette sections, the run artifact's checksum, the
sandboxed dashboard route, and a small headless run of the lunar rover flow (skipped without pyvista or a display)."""
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

import pinneapple_flow as pf
from pinneapple_flow import REGISTRY, palette
from pinneapple_flow.nodes.vehicle import load_run

pytest.importorskip("pinneapple_simulation.vehicle")
FLOWS = Path(pf.__file__).parent / "flows"


def val(m):
    return m["value"] if isinstance(m, dict) else m

TYPES = {"vehicle.spec": ("problem", [], ["vehicle", "metrics"]),
         "terramechanics.soil": ("problem", [], ["soil"]),
         "scenario.site": ("problem", [], ["site"]),
         "vehicle.simulate": ("solvers", ["vehicle", "soil", "site"], ["vehicle_run", "metrics"]),
         "terramechanics.wheel_model": ("solvers", ["vehicle", "soil"], ["model", "metrics"]),
         "terramechanics.zoo_surrogate": ("physics_ai", ["model"], ["model", "metrics"]),
         "vehicle.dashboard": ("viz", ["vehicle_run", "model", "model"], ["report", "figure", "metrics"])}


def test_rover_node_types_ports_and_palette_sections():
    for t, (section, ins, outs) in TYPES.items():
        s = REGISTRY.get(t)
        assert s is not None, t
        assert [p.type for p in s.inputs] == ins and [p.type for p in s.outputs] == outs, t
        assert palette.section_of(s) == section, t
        assert "pinneapple_simulation" in s.calls or "pinneapple_hub" in s.calls
    assert "viper" in REGISTRY.get("vehicle.spec").param("preset").choices
    assert "GRC-1" in REGISTRY.get("terramechanics.soil").param("preset").choices
    assert pf.compatible("vehicle_run", "vehicle_run") and not pf.compatible("vehicle_run", "solution")


def test_example_flows_validate_and_mirror_the_script():
    for name, dur, frames in (("lunar_rover_dashboard", 45.0, 150), ("lunar_rover_dashboard_quick", 12.0, 24)):
        f = pf.Flow.load(FLOWS / f"{name}.json")
        assert not [i for i in pf.validate_flow(f) if i.level == "error"], name
        p = {n.type: n.params for n in f.nodes}
        assert p["vehicle.simulate"]["duration"] == dur and p["vehicle.simulate"]["speed"] == 0.2
        assert p["vehicle.dashboard"]["frames"] == frames
        assert p["terramechanics.soil"]["preset"] == "GRC-1" and p["scenario.site"]["scenario"] == "crater_site"
        assert {e.target for e in f.edges if e.target_port == "validation"} == {"lab"}


def test_presets_soil_site_and_wheel_model_run(tmp_path):
    fb = pf.Flow.from_dict({
        "schema": pf.SCHEMA_ID, "id": "rover-parts", "name": "rover parts", "description": "", "meta": {},
        "nodes": [{"id": "veh", "type": "vehicle.spec", "params": {"total_mass": 300.0}},
                  {"id": "soil", "type": "terramechanics.soil", "params": {"preset": "GRC-1", "c": 200.0}},
                  {"id": "wheel", "type": "terramechanics.wheel_model", "params": {}}],
        "edges": [{"source": "veh", "source_port": "vehicle", "target": "wheel", "target_port": "vehicle"},
                  {"source": "soil", "source_port": "soil", "target": "wheel", "target_port": "soil"}]})
    r = pf.run(fb, workdir=str(tmp_path), cache=False)
    assert r.status == "done", {k: v.get("error") for k, v in r.nodes.items()}
    assert val(r.report("veh")["metrics"]["total_mass_kg"]) == 300.0
    m = r.report("wheel")["metrics"]
    assert val(m["static_wheel_load_N"]) == pytest.approx(300.0 * 1.62 / 4, rel=1e-3)
    assert val(m["drawbar_pull_at_slip_0.2_N"]) > 0


def test_a_tampered_run_artifact_is_refused(tmp_path):
    p = tmp_path / "vehicle_run.pkl"
    p.write_bytes(b"not the run")
    from pinneapple_core.workflows import StepFailed
    with pytest.raises(StepFailed, match="changed"):
        load_run({"$artifact": str(p), "kind": "vehicle_run", "sha256": "0" * 64})
    with pytest.raises(StepFailed):
        load_run({"$artifact": str(tmp_path / "x.npz"), "kind": "npz"})


def test_dashboard_html_is_served_sandboxed(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path))
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "flow_studio" / "backend"))
    from app import create_app

    from pinneapple_flow.service import FlowService
    svc = FlowService(db_url=f"sqlite:///{tmp_path}/f.db")
    run = svc.workdir / "r1" / "dashboard"
    run.mkdir(parents=True)
    (run / "rover-dashboard.html").write_text("<script>1</script>")
    (run / "data.json").write_text("{}")
    c = TestClient(create_app(svc))
    r = c.get("/api/artifact/html", params={"path": str(run / "rover-dashboard.html")})
    assert r.status_code == 200
    csp = r.headers["content-security-policy"]
    assert csp.startswith("sandbox allow-scripts") and "allow-same-origin" not in csp and "connect-src 'none'" in csp
    assert r.headers["x-frame-options"] == "SAMEORIGIN"
    assert c.get("/api/artifact/html", params={"path": str(run / "data.json")}).status_code == 403
    assert c.get("/api/artifact/html", params={"path": "/etc/hostname"}).status_code == 403


@pytest.mark.skipif(importlib.util.find_spec("pyvista") is None or importlib.util.find_spec("imageio") is None
                    or not (os.environ.get("DISPLAY") or shutil.which("Xvfb")),
                    reason="the dashboard needs pyvista, imageio and a display (or Xvfb)")
def test_quick_rover_flow_smoke(tmp_path):
    f = pf.Flow.load(FLOWS / "lunar_rover_dashboard_quick.json")
    try:
        from pinneapple_hub import zoo
        zoo.load("rover_contact_grc1_viper")
    except Exception as e:  # noqa: BLE001 - the zoo download needs the network
        pytest.skip(f"model zoo unavailable: {e}")
    for n in f.nodes:                                      # smaller than --quick, so the test stays short
        if n.type == "vehicle.simulate":
            n.params["duration"] = 4.0
        if n.type == "vehicle.dashboard":
            n.params.update(frames=4, width=640, height=360, video="none")
    r = pf.run(f, workdir=str(tmp_path), cache=False)
    assert r.nodes["dashboard"]["status"] == "done", {k: v.get("error") for k, v in r.nodes.items()}
    m = r.report("simulate")["metrics"]
    assert val(m["distance_m"]) > 0 and val(m["duration_s"]) == pytest.approx(4.0, abs=0.1)
    assert Path(r.output("dashboard", "report")["$artifact"]).is_file()
    assert Path(r.output("dashboard", "figure")["$artifact"]).is_file()
    checks = {c["name"]: c for c in r.output("dashboard", "metrics")["checks"]}
    assert "horizon_no_ground_border" in checks
    assert r.output("pinn", "metrics")["checks"][0]["name"] == "zoo_pinn_same_physics"
    assert json.dumps(r.report("lab")["metrics"], default=str)
