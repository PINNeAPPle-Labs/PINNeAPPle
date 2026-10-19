"""Generic problem node: identification from text (confidence, reasoning, references), equation discovery from data,
and the automatic digital twin generated from a description."""
import numpy as np
import pytest

import pinneapple_flow as pf
from pinneapple_flow.identify import confirm, discover_pde, identify
from pinneapple_physics.symbolic_pde.text import parse_system
from pinneapple_simulation.numerical_solvers.symbolic_fd import solve_fd


def _heat_data(alpha=0.1, n=81, nt=81):
    s = parse_system(["u_t = alpha*lap(u)"], [{"kind": "initial", "equation": "u = sin(pi*x) + 0.5*sin(2*pi*x)"},
                                             {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}],
                     coords=["x", "t"], fields=["u"], coefficients={"alpha": alpha})
    r = solve_fd(s, {"x": (0, 1), "t": (0, 0.5)}, n=n, n_save=nt)
    return {"x": r.axes["x"], "t": r.t}, np.moveaxis(r.fields["u"], 0, 1)


@pytest.mark.parametrize("text,family", [
    ("a steel rod cools down over time with its ends held at 20 C", "heat"),
    ("vibrations of a guitar string fixed at both ends", "wave"),
    ("steady temperature in a plate with internal heat generation", "poisson"),
    ("pollutant transport by advection and diffusion in a channel", "advection_diffusion"),
])
def test_identify_ranks_the_right_family_with_reasoning_and_references(text, family):
    res = identify(text)
    best = res["candidates"][0]
    assert best["family"] == family
    assert 0 < best["confidence"] <= 0.95 and best["reasoning"] and best["conservation_law"]
    assert any(r["status"] == "registry" for d in best["decisions"] for r in d["references"])
    assert all(c["confidence"] <= best["confidence"] for c in res["candidates"])


def test_unrecognised_text_gives_low_confidence_and_a_warning():
    res = identify("the quarterly sales report of the bakery")
    assert res["warnings"] and res["candidates"][0]["confidence"] < 0.6


def test_discovery_recovers_heat_and_burgers():
    axes, u = _heat_data()
    f = discover_pde(axes, u, ["x", "t"])
    assert f["family"] == "heat" and f["r2"] > 0.999
    assert f["terms"] == pytest.approx({"u_xx": 0.1}, rel=0.02)
    assert confirm(identify("heat in a rod")["candidates"][0], f)["family_agrees"]
    s = parse_system(["u_t + u*u_x = nu*u_xx"], [{"kind": "initial", "equation": "u = -sin(pi*x)"},
                                                 {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}],
                     coords=["x", "t"], fields=["u"], coefficients={"nu": 0.05})
    r = solve_fd(s, {"x": (-1, 1), "t": (0, 0.5)}, n=201, n_save=101)
    f = discover_pde({"x": r.axes["x"], "t": r.t}, np.moveaxis(r.fields["u"], 0, 1), ["x", "t"])
    assert f["family"] == "burgers"
    assert f["terms"]["u_xx"] == pytest.approx(0.05, rel=0.05) and f["terms"]["u*u_x"] == pytest.approx(-1, rel=0.05)


def test_discovery_says_when_the_data_cannot_separate_terms():
    s = parse_system(["u_t = alpha*lap(u)"], [{"kind": "initial", "equation": "u = sin(pi*x)"},
                                             {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}],
                     coords=["x", "t"], fields=["u"], coefficients={"alpha": 0.1})
    r = solve_fd(s, {"x": (0, 1), "t": (0, 0.5)}, n=61, n_save=41)
    f = discover_pde({"x": r.axes["x"], "t": r.t}, np.moveaxis(r.fields["u"], 0, 1), ["x", "t"])
    assert f["warning"] and ("u", "u_xx") in [tuple(p) for p in f["collinear"]]


def test_generic_node_with_data_and_user_edit(tmp_path):
    flow = pf.Flow("g", nodes=[
        {"id": "src", "type": "problem.generic", "params": {"description": "transient heat in a rod, alpha = 0.1",
                                                            "conditions": [
                                                                {"kind": "initial", "on": "t_min",
                                                                 "equation": "u = sin(pi*x) + 0.5*sin(2*pi*x)"},
                                                                {"kind": "dirichlet", "on": "boundary",
                                                                 "equation": "u = 0"}], "t_range": [0, 0.5]}},
        {"id": "fd", "type": "solver.classical", "params": {"n": 81, "n_t": 81}},
        {"id": "gen", "type": "problem.generic", "params": {"description": "diffusion in a bar"}}],
        edges=[{"source": "src", "source_port": "problem", "target": "fd", "target_port": "problem"},
               {"source": "fd", "source_port": "solution", "target": "gen", "target_port": "data"}])
    r = pf.run(flow, workdir=tmp_path)
    assert r.status == "done", {k: v["error"] for k, v in r.nodes.items()}
    m = r.report("gen")["metrics"]
    assert "u_xx" in m["discovered_equation"] and m["discovery_r2"]["value"] > 0.999
    assert r.report("gen")["views"]["candidates"]["candidates"][0]["family"] == "heat"
    assert any(d["decided_by"]["kind"] == "user" for d in r.report("src")["decisions"])


def test_twin_generated_from_a_description(tmp_path):
    text = ("digital twin of a heated rod (1D) from 4 temperature sensors; detect if its thermal diffusivity "
            "changes")
    res = pf.plan(text, backend="heuristic")
    p = res["proposals"][0]
    types = [n["type"] for n in p["flow"]["nodes"]]
    assert types[:3] == ["problem.generic", "data.stream", "twin.auto"]
    flow = pf.Flow.from_dict(p["flow"])
    for n in flow.nodes:                                # keep the test fast: a shorter record, coarser plant
        if n.type == "data.stream":
            n.params.update(plant_n=41, every=2)
    r = pf.run(flow, workdir=tmp_path)
    assert r.status == "done", {k: v["error"] for k, v in r.nodes.items()}
    m = r.report("twin")["metrics"]
    assert m["alpha_true_final"]["value"] == pytest.approx(0.15)
    assert abs(m["alpha_final"]["value"] - 0.15) < 0.03
    assert m["alerts"] >= 1 and m["detection_delay"]["value"] >= 0
    assert r.report("view3d")["views"]["scene"]["glb"].endswith(".glb")
    ds = r.report("twin")["decisions"]
    assert {d["decided_by"]["id"] for d in ds} >= {"twin.enkf", "twin.design", "twin.parameter_state"}


def test_twin_observes_sensors_between_grid_nodes():
    """Sensors that are not on the twin's coarse grid are observed by interpolation: no alarm before the change,
    the coefficient tracked after it (a regression: nearest-node observation raised false alarms)."""
    from pinneapple_flow.twin import PDETwinModel, run_twin, simulate_sensors
    sys_ = parse_system(["u_t = alpha*lap(u)"], [{"kind": "initial", "equation": "u = sin(pi*x)"},
                                                 {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"}],
                        coords=["x", "t"], fields=["u"], coefficients={"alpha": 0.1})
    bounds = {"x": (0.0, 1.0), "t": (0.0, 0.5)}
    sensors = [1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6]                 # only 0.5 is on the 21-node grid
    plant = PDETwinModel(sys_, bounds, n=81, dt=0.5 / 300, estimate={"alpha": 0.1})   # the fault acts on it
    readings = simulate_sensors(plant, sensors=sensors, t_end=0.5, noise=0.002, coefficients={"alpha": 0.1},
                                fault={"t": 0.25, "alpha": 0.15})
    model = PDETwinModel(sys_, bounds, n=21, dt=0.5 / 300, estimate={"alpha": 0.1})
    W = model.sensor_weights(sensors)
    assert np.allclose(W.sum(axis=1), 1.0) and (W > 0).sum(axis=1).tolist() == [2, 2, 1, 2, 2]
    _rt, res = run_twin(model, readings, sensors=sensors, sigma_obs=0.002, limits={"alpha": (0.07, 0.13)})
    assert not [a for a in res["alerts"] if a["t"] < 0.25]
    assert abs(res["final"]["alpha"] - 0.15) < 0.01
