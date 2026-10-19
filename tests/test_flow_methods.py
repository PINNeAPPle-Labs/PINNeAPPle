"""First-class method nodes: verification (grid convergence), data assimilation, uncertainty propagation, ROMs,
MeshGraphNet, meta-learning, the PINN benchmark arena, NSGA-II and control. Small budgets: each must run and report
what the library measured."""
import pytest

import pinneapple_flow as pf


def E(a, ap, b, bp):
    return {"source": a, "source_port": ap, "target": b, "target_port": bp}


HEAT = [{"id": "geo", "type": "geometry.domain", "params": {"dimension": 1, "t_range": [0, 0.5]}},
        {"id": "eq", "type": "equations.system", "params": {"template": "heat"}}]
GEO = [E("geo", "geometry", "eq", "geometry")]


def run(tmp_path, nodes, edges=()):
    r = pf.run(pf.Flow("m", nodes=nodes, edges=list(edges)), workdir=tmp_path, cache=False)
    assert r.status == "done", {k: v["error"] for k, v in r.nodes.items() if v["error"]}
    return r


def test_grid_convergence_finds_second_order(tmp_path):
    r = run(tmp_path, HEAT + [{"id": "n", "type": "verify.convergence", "params": {"n": 9}}],
            GEO + [E("eq", "problem", "n", "problem")])
    m = r.report("n")["metrics"]
    order = m["observed_order"]["value"] if isinstance(m["observed_order"], dict) else m["observed_order"]
    assert 1.8 < order < 2.2                         # three-point Laplacian, smooth heat problem
    assert any(c["name"] == "grid_convergence_asymptotic" for c in r.output("n", "metrics")["checks"])


def test_kalman_filter_on_a_live_stream(tmp_path):
    r = run(tmp_path, HEAT + [{"id": "s", "type": "data.stream", "params": {"t_end": 0.5, "dt": 0.005}},
                              {"id": "n", "type": "assimilate.filter", "params": {"method": "enkf", "n_ens": 50}}],
            GEO + [E("eq", "problem", "s", "problem"), E("s", "stream", "n", "series")])
    assert "innovation_rms" in r.report("n")["metrics"]
    assert r.load("n", "series")["values"].shape[1] == 4


def test_rom_on_a_solution(tmp_path):
    r = run(tmp_path, HEAT + [{"id": "fd", "type": "solver.classical", "params": {"method": "fd", "n": 41, "n_t": 41}},
                              {"id": "n", "type": "rom.fit", "params": {"ranks": [2, 4]}}],
            GEO + [E("eq", "problem", "fd", "problem"), E("fd", "solution", "n", "solution")])
    assert any(t["title"].startswith("ROMs") for t in r.report("n")["tables"])


@pytest.mark.parametrize("node", [
    {"id": "n", "type": "gnn.meshgraphnet", "params": {"n_train": 3, "n_test": 1, "n_points": 50, "steps": 30,
                                                      "rollout": 3}},
    {"id": "n", "type": "benchmark.arena", "params": {"problems": ["heat_1d"], "models": ["bench_mlp"], "epochs": 5,
                                                     "n_col": 32}},
    {"id": "n", "type": "optimize.nsga2", "params": {"budget": 60}},
    {"id": "n", "type": "control.compare", "params": {"controllers": ["pid", "rules"]}},
])
def test_standalone_method_nodes(tmp_path, node):
    r = run(tmp_path, [node])
    assert r.report("n")["summary"]


def test_meta_learning_reports_both_initialisations(tmp_path):
    r = run(tmp_path, HEAT + [{"id": "n", "type": "meta.learn", "params": {"meta_epochs": 2, "inner_steps": 2,
                                                                          "adapt_steps": 3, "width": 8}}],
            GEO + [E("eq", "problem", "n", "problem")])
    m = r.report("n")["metrics"]
    assert "loss_after_adapt_meta" in m and "loss_after_adapt_random" in m


def test_uq_propagation_on_a_surrogate(tmp_path):
    nodes = HEAT + [{"id": "sweep", "type": "data.parametric", "params": {"n_samples": 24, "vary": {
                        "alpha": [0.05, 0.2]}, "n": 21, "n_t": 11}},
                    {"id": "sur", "type": "surrogate.train", "params": {"kind": "mlp", "epochs": 30}},
                    {"id": "n", "type": "uq.propagate", "params": {"n": 256}}]
    r = run(tmp_path, nodes, GEO + [E("eq", "problem", "sweep", "problem"), E("sweep", "data", "sur", "data"),
                                    E("sur", "model", "n", "model")])
    assert "std" in r.report("n")["metrics"]


def test_first_class_names_for_external_solvers():
    for t in ("solver.calculix", "solver.fmu", "solver.newton", "surrogate.study"):
        assert t in pf.REGISTRY
        ok, why = pf.REGISTRY[t].available()
        assert ok or why.startswith("needs")


def test_transfer_from_a_trained_pinn(tmp_path):
    src = [{"id": "geo", "type": "geometry.domain", "params": {"dimension": 1, "t_range": [0, 0.5]}},
           {"id": "eq", "type": "equations.system", "params": {"template": "heat"}},
           {"id": "pinn", "type": "train.pinn", "params": {"adam_steps": 50, "lbfgs_steps": 0, "n_collocation": 128,
                                                           "width": 16, "depth": 2}},
           {"id": "eq2", "type": "equations.system", "params": {"template": "heat", "coefficients": {"alpha": 0.15}}},
           {"id": "n", "type": "transfer.pinn", "params": {"epochs": 10, "n_collocation": 64}}]
    r = run(tmp_path, src, GEO + [E("eq", "problem", "pinn", "problem"), E("geo", "geometry", "eq2", "geometry"),
                                  E("pinn", "model", "n", "model"), E("eq2", "problem", "n", "problem")])
    m = r.report("n")["metrics"]
    assert "loss_transferred" in m and "loss_from_scratch" in m
