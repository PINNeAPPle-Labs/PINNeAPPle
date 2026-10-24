"""pinneapple_tools.nvidia (diagrams, OpenUSD stages for Omniverse) and the examples in examples/nvidia (smoke)."""
import importlib.util
import os
import runpy

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_PXR = importlib.util.find_spec("pxr") is not None
HAS_PNEMO = importlib.util.find_spec("physicsnemo") is not None
needs_pxr = pytest.mark.skipif(not HAS_PXR, reason="usd-core (OpenUSD) not installed")
needs_pnemo = pytest.mark.skipif(not HAS_PNEMO, reason="nvidia-physicsnemo not installed")


def _example(name):
    return runpy.run_path(os.path.join(REPO, "examples", "nvidia", name, "run.py"))


# ------------------------------------------------------------------------------------------------ diagram
def test_architecture_diagram(tmp_path):
    from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
    p = architecture_diagram("t", [Block("a", "A"), Block("b", "B", "physicsnemo", 1),
                                   Block("c", "C", "isaac", 1, 1, ran=False)],
                             [("a", "b", "x"), ("b", "c")], str(tmp_path / "a.png"))
    assert os.path.getsize(p) > 1000
    with pytest.raises(KeyError):
        architecture_diagram("t", [Block("a", "A")], [("a", "zz")], str(tmp_path / "b.png"))
    with pytest.raises(ValueError):
        architecture_diagram("t", [Block("a", "A", "acme")], [], str(tmp_path / "c.png"))


# ------------------------------------------------------------------------------------------------ OpenUSD
def test_geometry_helpers():
    from pinneapple_tools.nvidia.omniverse import extrude_profile, grid_surface, tube
    x = np.linspace(0, 1, 4)
    V, F = grid_surface(*np.meshgrid(x, x, indexing="ij"))
    assert V.shape == (16, 3) and F.shape == (18, 3) and F.max() == 15
    V, F = tube(np.c_[x, 0 * x, 0 * x], 0.1, 8)
    assert V.shape == (32, 3) and F.shape == (3 * 8 * 2, 3)
    assert np.allclose(np.linalg.norm(V[:, 1:], axis=1), 0.1)
    V, F = extrude_profile(np.c_[np.cos(np.arange(5)), np.sin(np.arange(5))], 2.0, n=3)
    assert V.shape == (15, 3) and V[:, 1].max() == 2.0 and F.max() == 14


@needs_pxr
def test_field_stage_round_trip_and_validation(tmp_path):
    from pxr import Usd, UsdGeom, Vt

    from pinneapple_tools.nvidia.omniverse import (
        FieldStage,
        grid_surface,
        read_stage_geometry,
        render_stage_png,
        validate_stage,
    )
    x = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, x, indexing="ij")
    V, F = grid_surface(X, Y)
    path = str(tmp_path / "s.usda")
    st = FieldStage(path, "test")
    m = st.add_mesh("Plate", V, F, fields={"T": X.ravel()}, units={"T": "K"}, color_by="T")
    st.add_time_samples(m, "T", [0.0, 1.0], [X.ravel(), 2 * X.ravel()])
    c = st.add_curves("Line", [np.c_[x, x, x]], fields={"p": [x]}, color_by="p")
    st.add_points("Sensors", [[0, 0, 0]], fields={"T": [1.0]})
    st.set_trust(m, "APPROVED", [{"name": "bc", "passed": True}], solver="test")
    st.save()
    r = validate_stage(path, required_primvars={m: ["T"], c: ["p"]})
    assert r["ok"], r["problems"]
    assert (r["meshes"], r["curves"], r["points"]) == (1, 1, 1) and r["time_samples"] == 4
    assert r["up_axis"] == "Z" and r["meters_per_unit"] == 1.0
    stage = Usd.Stage.Open(path)
    prim = stage.GetPrimAtPath(m)
    assert prim.GetAttribute("pinneapple:trust:decision").Get() == "APPROVED"
    assert prim.GetAttribute("pinneapple:unit:T").Get() == "K"
    T1 = np.asarray(UsdGeom.PrimvarsAPI(prim).GetPrimvar("T").Get(24.0))     # 1 s at 24 time codes per second
    assert np.allclose(T1, 2 * X.ravel(), atol=1e-6)
    geo = read_stage_geometry(path)
    assert {g["kind"] for g in geo} == {"mesh", "curves"} and np.allclose(geo[0]["points"], V, atol=1e-6)
    assert os.path.getsize(render_stage_png(path, str(tmp_path / "p.png"))) > 1000
    # a primvar with the wrong length and a missing primvar are reported
    UsdGeom.PrimvarsAPI(prim).GetPrimvar("T").Set(Vt.FloatArray([1.0, 2.0]), 0.0)
    stage.GetRootLayer().Save()
    bad = validate_stage(path, required_primvars={m: ["T", "missing"]})
    assert not bad["ok"]
    assert any("2 values for 25 points" in p for p in bad["problems"])
    assert any("'missing' missing" in p for p in bad["problems"])


@needs_pxr
def test_mesh_rejects_bad_indices(tmp_path):
    from pinneapple_tools.nvidia.omniverse import FieldStage
    st = FieldStage(str(tmp_path / "b.usda"))
    with pytest.raises(ValueError):
        st.add_mesh("m", np.zeros((3, 3)), [[0, 1, 3]])
    with pytest.raises(ValueError):
        st.add_mesh("m2", np.zeros((3, 3)), [[0, 1, 2]], fields={"T": [1.0, 2.0]})


def test_kit_render_script_mentions_the_stage(tmp_path):
    from pinneapple_tools.nvidia.omniverse import kit_render_script
    s = kit_render_script("a.usda", "a.png")
    assert "open_stage_async" in s and "capture_viewport_to_file" in s and "a.usda" in s
    compile(s, "kit.py", "exec")


def test_runinfo(tmp_path):
    from pinneapple_tools.nvidia.runinfo import RunInfo
    info = RunInfo("x")
    with info.step("a"):
        pass
    info.mark("Isaac Sim", ran=False, why="GPU")
    d = info.to_dict()
    assert "a" in d["steps_s"] and d["components"][0]["ran"] is False and d["environment"]["python"]
    assert os.path.exists(info.save(str(tmp_path / "r.json")))


# ------------------------------------------------------------------------------------------------ physics added
def test_water_hammer_matches_joukowsky_and_period():
    from pinneapple_systems.process_components import water_hammer as wh
    line = wh.Pipeline(length=1000.0, diameter=0.4, wave_speed=1000.0)
    pump = wh.PumpCurve(shutoff_head=60.0, k=200.0)
    op = wh.operating_point(line, pump, valve_k=5.0, tank_head=10.0)
    v = op.velocity
    assert pump.head(op.flow) == pytest.approx(10.0 + op.friction_loss + 5.0 * v * v / (2 * wh.G), rel=1e-9)
    r = wh.moc_valve_closure(line, pump, op, closure_time=0.0, t_end=8.0, n_reaches=20, friction=False)
    dh = r.head[:, -1].max() - op.valve_head
    assert dh == pytest.approx(wh.joukowsky_head_rise(1000.0, v), rel=1e-9)
    h = r.head[:, -1] - op.valve_head
    cross = r.t[np.where(np.diff(np.sign(h)) != 0)[0]]
    assert 2 * np.mean(np.diff(cross)) == pytest.approx(wh.wave_period(1000.0, 1000.0), rel=1e-9)
    still = wh.moc_valve_closure(line, pump, op, closure_time=1.0, closure_start=1e9, t_end=5.0, n_reaches=20)
    assert np.abs(still.head - still.head[0]).max() < 1e-8 and np.abs(still.flow - op.flow).max() < 1e-10
    slow = wh.moc_valve_closure(line, pump, op, closure_time=60.0, t_end=60.0, n_reaches=20)
    fast = wh.moc_valve_closure(line, pump, op, closure_time=1.0, t_end=60.0, n_reaches=20)
    assert slow.head.max() < fast.head.max()
    assert abs(fast.flow[-1, -1]) < 1e-12                       # closed valve passes nothing


def test_advection_diffusion_conserves_mass_and_decays():
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import (
        advection_diffusion_2d,
        periodic_grf,
    )
    u0 = np.exp(periodic_grf(2, (16, 16), alpha=2.5, tau=5.0, seed=0))
    tr = advection_diffusion_2d(u0, (1.0, 0.5), 2e-3, 0.05, 20)
    assert tr.shape == (21, 2, 16, 16)
    assert np.allclose(tr.mean((2, 3)), u0.mean((1, 2)), rtol=1e-12)
    var = tr.var(axis=(2, 3))
    assert np.all(np.diff(var, axis=0) <= 1e-14)
    # pure advection by a whole number of cells is an exact shift
    sh = advection_diffusion_2d(u0[0], (1.0, 0.0), 0.0, 1 / 16, 1)[1]
    assert np.allclose(sh, np.roll(u0[0], 1, axis=0), atol=1e-12)


# ------------------------------------------------------------------------------------------------ examples (smoke)
@needs_pxr
def test_example_01_pipeline_twin_smoke(tmp_path):
    s = _example("01_omniverse_pump_pipeline")["main"](True, str(tmp_path))
    assert s["usd"]["ok"] and s["checks"][0]["passed"] and s["checks"][1]["passed"]
    for f in ("architecture.png", "transients.png", "usd_preview.png", "usd_timeline.gif", "summary.json"):
        assert os.path.exists(tmp_path / "results" / f)
    assert os.path.exists(tmp_path / "experiment.json")


@needs_pxr
@needs_pnemo
def test_example_07_coldplate_smoke(tmp_path):
    s = _example("07_omniverse_physicsnemo_coldplate")["main"](True, str(tmp_path))
    assert s["decision"] in ("APPROVED", "REVIEW", "REJECT") and s["usd"]["ok"]
    assert os.path.exists(tmp_path / "results" / "usd_preview.png")


@needs_pnemo
def test_example_02_plume_smoke(tmp_path):
    s = _example("02_physicsnemo_plume_rollout")["main"](True, str(tmp_path))
    assert s["decision"] in ("APPROVED", "REVIEW", "REJECT")
    assert os.path.exists(tmp_path / "results" / "rollout_checks.png")


# ------------------------------------------------------------------------------------------------ robot sim (stack I)
HAS_NEWTON = importlib.util.find_spec("newton") is not None
needs_newton = pytest.mark.skipif(not HAS_NEWTON, reason="NVIDIA Newton not installed (pip install newton)")


def test_two_link_tree_matches_closed_form_and_energy():
    from pinneapple_simulation.multibody import arm as mb
    from pinneapple_simulation.robot_sim import TwoLinkArm
    a = TwoLinkArm()
    tree = a.tree()
    b1, b2 = a.bodies()
    assert b1.mass == pytest.approx(a.l1 * 2700 * a.section().A_m2 + a.motor)
    t = np.linspace(0, 0.8, 161)
    q, qd, qdd = mb.quintic((-0.6, 1.2), (0.9, -0.4), 0.8, t)
    tau = mb.fixed_base_inverse_dynamics(tree, q, qd, qdd)
    ref = mb.planar_two_link_torques(b1.mass, b2.mass, a.l1, b1.com[0], b2.com[0], b1.inertia[1, 1],
                                     b2.inertia[1, 1], a.gravity, q, qd, qdd)
    assert np.abs(tau - ref).max() < 1e-10 * np.abs(ref).max()
    assert mb.energy_balance(tree, t, q, qd, tau)["rel"] < 1e-4
    sig, checks = mb.link_root_checks(tau, a.section(), yield_strength=276e6, fatigue_strength=96.5e6)
    assert sig.shape == tau.shape and len(checks) == 4 and all(c["passed"] for c in checks)


def test_spec_to_urdf_is_valid_xml():
    import xml.etree.ElementTree as ET

    from pinneapple_simulation.robot_sim import TwoLinkArm, spec_to_urdf
    root = ET.fromstring(spec_to_urdf(TwoLinkArm().spec()))
    assert [j.get("name") for j in root.iter("joint")] == ["shoulder", "elbow"]
    masses = [float(m.get("value")) for m in root.iter("mass")]
    assert masses[0] == 20.0 and len(masses) == 3


def test_cem_finds_a_quadratic_minimum():
    from pinneapple_simulation.robot_sim.learn import cem
    r = cem(lambda x: float(np.sum((x - [3.0, -1.0]) ** 2)), [0.0, 0.0], [2.0, 2.0], iterations=12, population=16)
    assert np.allclose(r.best, [3.0, -1.0], atol=0.2) and r.history[-1]["best_cost"] <= r.history[0]["best_cost"]


@needs_pxr
def test_isaac_ready_stage_validates_and_catches_bad_mass(tmp_path):
    from pxr import Usd, UsdPhysics

    from pinneapple_simulation.robot_sim import TwoLinkArm
    from pinneapple_tools.nvidia.isaac import (
        author_articulation,
        isaac_replay_script,
        validate_physics,
    )
    from pinneapple_tools.nvidia.omniverse import FieldStage
    p = str(tmp_path / "arm.usda")
    st = FieldStage(p)
    root = author_articulation(st, TwoLinkArm().spec())
    st.save()
    v = validate_physics(p, expect_bodies=3, expect_joints=2)
    assert v["ok"], v["problems"]
    assert v["physics"]["Articulation"] == 1 and v["total_mass"] == pytest.approx(20 + sum(
        b.mass for b in TwoLinkArm().bodies()), rel=1e-6)
    compile(isaac_replay_script(p, root, "a.npz", "b.npz"), "replay.py", "exec")
    stage = Usd.Stage.Open(p)
    UsdPhysics.MassAPI(stage.GetPrimAtPath(f"{root}/link1")).GetMassAttr().Set(-1.0)
    stage.GetRootLayer().Save()
    assert any("mass" in s for s in validate_physics(p)["problems"])


@needs_newton
def test_robot_sim_energy_order_and_usd_equals_urdf(tmp_path):
    from pinneapple_simulation.robot_sim import RobotSim, TwoLinkArm, convergence_order
    a = TwoLinkArm()
    sim = RobotSim.from_spec(a.spec(), ground=False)
    errs, dts = [], (2e-3, 1e-3)
    for dt in dts:
        r = sim.run(0.2, dt, q0=[0.0, 0.0])
        E = r.energy()["total"]
        errs.append(np.abs(E - E[0]).max() / np.abs(E).max())
    assert errs[-1] < 2e-3 and convergence_order(errs, dts) > 0.8
    # at rest under gravity compensation the IMU reads +g up (specific force)
    if HAS_PXR:
        from pinneapple_tools.nvidia.isaac import author_articulation
        from pinneapple_tools.nvidia.omniverse import FieldStage
        p = str(tmp_path / "arm.usda")
        st = FieldStage(p)
        author_articulation(st, a.spec())
        st.save()
        su = RobotSim.from_usd(p, ground=False)
        assert np.abs(su.run(0.1, 1e-3, q0=[0.1, 0.2]).q - sim.run(0.1, 1e-3, q0=[0.1, 0.2]).q).max() == 0.0


@needs_pnemo
def test_mlp_surrogate_fits_saves_and_checks_signs(tmp_path):
    from pinneapple_analysis.cost.equal_budget import ComputeBudget
    from pinneapple_tools.nvidia.physicsnemo import MLPSurrogate, relative_errors, sign_violations
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (512, 2))
    Y = np.c_[X[:, 0] + 0.5 * X[:, 1], X[:, 0] * X[:, 1]]
    m = MLPSurrogate(2, 2, layer_size=32, num_layers=2)
    m.fit(X, Y, ComputeBudget(steps=300), batch_size=128)
    assert relative_errors(m.predict(X), Y)[0] < 0.2
    q = MLPSurrogate.load(m.save(tmp_path / "m"))
    assert np.allclose(q.predict(X), m.predict(X), atol=1e-6) and q.in_box(X).all() and not q.in_box([[5, 0]])[0]
    ax = [np.linspace(-1, 1, 9)] * 2
    assert sign_violations(lambda Z: np.c_[Z[:, 0], -Z[:, 0]], ax, 0, 0, +1) == 0.0
    assert sign_violations(lambda Z: np.c_[Z[:, 0], -Z[:, 0]], ax, 0, 1, +1) == 1.0


@needs_newton
def test_example_03_robotsim_smoke(tmp_path):
    s = _example("03_robotsim_pick_and_place")["main"](True, str(tmp_path))
    assert s["decision"] in ("APPROVED", "REVIEW", "REJECT") and s["replay_order"] > 0.5
    assert os.path.exists(tmp_path / "results" / "verification.png")


@needs_newton
@needs_pxr
def test_example_12_omniverse_robotsim_smoke(tmp_path):
    s = _example("12_omniverse_robotsim_arm")["main"](True, str(tmp_path))
    assert s["usd_vs_urdf_rad"] == 0.0 and s["physics_stage"]["ok"] and s["timeline_stage"]["ok"]
    assert os.path.exists(tmp_path / "results" / "usd_timeline.gif")


@needs_newton
@needs_pnemo
def test_example_08_learned_dynamics_smoke(tmp_path):
    s = _example("08_robotsim_physicsnemo_learned_dynamics")["main"](True, str(tmp_path))
    assert s["decision"] in ("APPROVED", "REVIEW", "REJECT") and s["samples"] > 0
