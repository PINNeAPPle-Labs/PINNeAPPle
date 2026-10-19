"""Running flows on the workflow engine: results, cache, run-from-node, failures, the store, the CLI."""
import json
import subprocess
import sys

import numpy as np
import pytest

import pinneapple_flow as pf
from pinneapple_flow.store import FlowStore


def heat_flow(pinn=False, pdes=None):
    nodes = [{"id": "geo", "type": "geometry.domain", "params": {"dimension": 1, "t_range": [0, 0.5]}},
             {"id": "eq", "type": "equations.system", "params": {"template": "heat"} if pdes is None else {
                 "pdes": pdes, "conditions": [{"kind": "initial", "on": "t_min", "equation": "u = sin(pi*x)"}]}},
             {"id": "fd", "type": "solver.classical", "params": {"method": "fd", "n": 41, "n_t": 11}},
             {"id": "val", "type": "validate.compare", "params": {"reference": "exact"}}]
    edges = [{"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
             {"source": "eq", "source_port": "problem", "target": "fd", "target_port": "problem"},
             {"source": "fd", "source_port": "solution", "target": "val", "target_port": "candidates"},
             {"source": "eq", "source_port": "problem", "target": "val", "target_port": "problem"}]
    if pinn:
        nodes.append({"id": "pinn", "type": "train.pinn", "params": {"adam_steps": 30, "lbfgs_steps": 0,
                                                                      "n_collocation": 128, "width": 8, "depth": 2}})
        edges += [{"source": "eq", "source_port": "problem", "target": "pinn", "target_port": "problem"},
                  {"source": "pinn", "source_port": "solution", "target": "val", "target_port": "candidates"}]
    return pf.Flow("heat test", nodes=nodes, edges=edges)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    return tmp_path


def test_run_measures_against_exact_then_caches(home):
    f = heat_flow()
    r = pf.run(f, workdir=home / "w")
    assert r.status == "done", {k: v["error"] for k, v in r.nodes.items()}
    rel = r.report("val")["metrics"]["FD_rel_l2"]["value"]
    assert rel < 1e-3
    sol = r.load("fd", "solution")
    assert sol["values"].shape == (41, 11, 1)
    r2 = pf.run(f, workdir=home / "w")
    assert {v["status"] for v in r2.nodes.values()} == {"cached"}
    r3 = pf.run(f, workdir=home / "w", from_node="fd")
    st = {k: v["status"] for k, v in r3.nodes.items()}
    assert st["fd"] == "done" and st["val"] == "done" and st["geo"] == "cached"


def test_pinn_node_trains_and_reports(home):
    r = pf.run(heat_flow(pinn=True), workdir=home / "w")
    assert r.nodes["pinn"]["status"] == "done", r.nodes["pinn"]["error"]
    rep = r.report("pinn")
    assert rep["views"]["architecture"]["total_params"] > 0
    assert any(t["title"].startswith("PINN validation") for t in rep["tables"])
    assert "PINN_rel_l2" in r.report("val")["metrics"]


def test_bad_equation_fails_that_node_and_cancels_downstream(home):
    r = pf.run(heat_flow(pdes=["u_t = alpha*u_xx"]), workdir=home / "w")
    assert r.status == "failed"
    assert r.nodes["eq"]["status"] == "failed" and "alpha" in r.nodes["eq"]["error"]
    assert r.nodes["fd"]["status"] == "cancelled"


def test_invalid_flow_never_runs(home):
    d = heat_flow().to_dict()
    d["nodes"][2]["type"] = "python.exec"
    with pytest.raises(pf.FlowError):
        pf.run(d, workdir=home / "w")


def test_store_records_versions_runs_and_node_outputs(home):
    store = FlowStore(f"sqlite:///{home / 'flow.sqlite'}")
    f = heat_flow()
    assert store.save_flow(f, "first") == 1
    f.description = "edited"
    assert store.save_flow(f, "second") == 2
    assert [v["version"] for v in store.list_versions(f.id)] == [2, 1]
    assert store.get_flow(f.id, 1)["description"] == ""
    assert store.get_flow(f.id)["description"] == "edited"
    r = pf.run(f, workdir=home / "w", store=store, flow_version=2)
    rec = store.get_run(r.run_id)
    assert rec["status"] == "done" and rec["version"] == 2
    assert rec["nodes"]["val"]["report"]["metrics"]["FD_rel_l2"]["value"] < 1e-3
    assert rec["nodes"]["fd"]["outputs"]["solution"]["$artifact"].endswith("solution.npz")
    assert store.list_runs(f.id)[0]["id"] == r.run_id
    store.delete_flow(f.id)
    with pytest.raises(KeyError):
        store.get_flow(f.id)


def test_cli_run_and_validate(home, tmp_path):
    p = heat_flow().save(tmp_path / "flow.json")
    out = subprocess.run([sys.executable, "-m", "pinneapple_flow", "validate", str(p)], capture_output=True,
                         text=True)
    assert out.returncode == 0 and "valid" in out.stdout
    out = subprocess.run([sys.executable, "-m", "pinneapple_flow", "run", str(p), "--workdir", str(tmp_path / "w"),
                          "--json", str(tmp_path / "res.json"), "--quiet"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads((tmp_path / "res.json").read_text())
    assert res["status"] == "done" and res["nodes"]["val"]["status"] == "done"
    out = subprocess.run([sys.executable, "-m", "pinneapple_flow", "plan", "heat in a 2D plate", "-o",
                          str(tmp_path / "planned.json")], capture_output=True, text=True)
    assert out.returncode == 0 and "planner: heuristic" in out.stdout
    assert pf.validate_flow(json.loads((tmp_path / "planned.json").read_text())) == []


def test_surrogate_and_hpo_nodes_on_a_small_sweep(home):
    nodes = [{"id": "geo", "type": "geometry.domain", "params": {"dimension": 1, "t_range": [0, 0.2]}},
             {"id": "eq", "type": "equations.system", "params": {"template": "heat"}},
             {"id": "sweep", "type": "data.parametric", "params": {"vary": {"alpha": [0.05, 0.2]}, "n_samples": 16,
                                                                   "n": 21, "n_t": 5}},
             {"id": "sur", "type": "surrogate.train", "params": {"kind": "pod_mlp", "epochs": 20}},
             {"id": "hpo", "type": "surrogate.hpo", "params": {"kind": "mlp", "n_trials": 2, "trial_epochs": 5,
                                                               "final_epochs": 10,
                                                               "space": {"lr": [1e-3, 1e-2, "log"],
                                                                         "width": [8, 16]}}},
             {"id": "sobol", "type": "uq.sensitivity", "params": {"n": 128}}]
    edges = [{"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
             {"source": "eq", "source_port": "problem", "target": "sweep", "target_port": "problem"},
             {"source": "sweep", "source_port": "data", "target": "sur", "target_port": "data"},
             {"source": "sweep", "source_port": "data", "target": "hpo", "target_port": "data"},
             {"source": "sur", "source_port": "model", "target": "sobol", "target_port": "model"}]
    r = pf.run(pf.Flow("sur", nodes=nodes, edges=edges), workdir=home / "w")
    assert r.status == "done", {k: v["error"] for k, v in r.nodes.items()}
    m = r.report("sur")["metrics"]
    assert np.isfinite(m["test_rel_l2"]["value"])
    assert r.report("sur")["views"]["architecture"]["total_params"] == m["n_parameters"]
    assert r.report("hpo")["metrics"]["trials"] == 2
    assert "S1_alpha" in r.report("sobol")["metrics"]


@pytest.mark.parametrize("dim", [2, 3])
def test_trust_card_with_a_source_and_boundary_dirichlet(home, dim):
    """Poisson with a source term and ``u = 0`` on the whole boundary, two candidates on different grids: the trust
    card of each is computed on its own grid (regressions: the source closure was rebound to a face name; a candidate
    on another grid was reshaped to the wrong size, then judged after interpolation, which breaks its derivatives)."""
    lap = {2: "-2*pi**2*sin(pi*x)*sin(pi*y)", 3: "-3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)"}[dim]
    exact = {2: "sin(pi*x)*sin(pi*y)", 3: "sin(pi*x)*sin(pi*y)*sin(pi*z)"}[dim]
    f = pf.Flow("poisson", nodes=[
        {"id": "geo", "type": "geometry.domain", "params": {"dimension": dim, "time_dependent": False}},
        {"id": "eq", "type": "equations.system", "params": {
            "fields": ["u"], "pdes": [f"lap(u) = {lap}"], "exact": exact,
            "conditions": [{"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}]}},
        {"id": "fd", "type": "solver.classical", "params": {"method": "fd", "n": 17 if dim == 2 else 9}},
        {"id": "fem", "type": "solver.classical", "params": {"method": "fem", "n": 9 if dim == 2 else 5}},
        {"id": "val", "type": "validate.compare", "params": {"reference": "exact"}}], edges=[
        {"source": "eq", "source_port": "problem", "target": "fem", "target_port": "problem"},
        {"source": "fem", "source_port": "solution", "target": "val", "target_port": "candidates"},
        {"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
        {"source": "eq", "source_port": "problem", "target": "fd", "target_port": "problem"},
        {"source": "fd", "source_port": "solution", "target": "val", "target_port": "candidates"},
        {"source": "eq", "source_port": "problem", "target": "val", "target_port": "problem"}])
    r = pf.run(f, workdir=home / "runs")
    assert r.status == "done", r.nodes["val"]["error"]
    if dim == 2:
        cards = {k: v.get("trust_card") for k, v in r.output("val", "metrics")["candidates"].items()}
        assert cards and all(c == "APPROVED" for c in cards.values()), cards


def test_a_candidates_own_failing_check_is_not_hidden(home):
    """validate_pinn's checks travel with the PINN's solution into validate.compare (and from there to the Lab), so a
    failing own check is visible next to the error against the reference."""
    r = pf.run(heat_flow(pinn=True), workdir=home / "runs")
    assert r.status == "done"
    checks = r.output("val", "metrics")["checks"]
    own = [c for c in checks if c["name"].startswith("PINN_pinn.")]
    assert own and all(c["kind"] == "physics" for c in own)
    pinn_report = r.report("pinn")
    failed = {row[0] for t in pinn_report["tables"] for row in t["rows"] if row[1] == "fail"}
    assert {c["name"].removeprefix("PINN_") for c in own if not c["passed"]} == failed
    assert any(t["title"] == "candidates' own checks" for t in r.report("val")["tables"])


def test_lab_node_reports_the_letter_the_curation_gives_this_run(home):
    f = heat_flow()
    f.nodes.append(pf.Node(id="lab", type="validate.lab", params={}))
    f.edges.append(pf.Edge(source="val", source_port="metrics", target="lab", target_port="validation"))
    r = pf.run(f, workdir=home / "runs")
    assert r.status == "done", r.nodes["lab"]["error"]
    m = r.report("lab")["metrics"]
    tier = m["tier"]["value"] if isinstance(m["tier"], dict) else m["tier"]
    import json
    from pathlib import Path
    cur = json.loads((Path(home) / "home" / "lab" / "curation.json").read_text())
    run_id = m["lab_run_id"]["value"] if isinstance(m["lab_run_id"], dict) else m["lab_run_id"]
    assert tier == cur["runs"][run_id]["tier"] and tier in "ABCDEF"
    assert r.report("lab")["summary"].startswith(f"tier {tier}")
