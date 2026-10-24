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


@needs_newton
def test_arm_pick_and_place_check_for_flow_node():
    from pinneapple_simulation.robot_sim.checks import arm_pick_and_place
    r = arm_pick_and_place(dt=2e-3)
    assert r["replay_order"] > 0.8 and r["decision"] in ("APPROVED", "REJECT") and len(r["checks"]) == 8


# ------------------------------------------------------------------------------------------------ USD BRep (stack B)
from pinneapple_tools.nvidia.brep import brep_available  # noqa: E402

needs_brep = pytest.mark.skipif(not brep_available(), reason="USD BRep not available (pip install usd-brep on "
                                "CPython 3.12, or set PINNEAPPLE_BREP_PYTHON)")


def test_plane_stress_patch_test_and_uniform_strip():
    from scipy.spatial import Delaunay

    from pinneapple_simulation.numerical_solvers.plane_stress_fem import (
        boundary_edges,
        heywood_kt_net,
        patch_test,
        solve_plane_stress,
    )
    rng = np.random.default_rng(0)
    P = np.r_[rng.uniform(0, 1, (150, 2)), [[0, 0], [1, 0], [0, 1], [1, 1]]]
    assert patch_test(P, Delaunay(P).simplices) < 1e-10
    x, y = np.linspace(0, 2, 11), np.linspace(0, 1, 6)
    X, Y = np.meshgrid(x, y, indexing="ij")
    V = np.c_[X.ravel(), Y.ravel()]
    T = Delaunay(V).simplices
    be = boundary_edges(T)
    right = be[np.all(np.isclose(V[be, 0], 2.0), 1)]
    left = np.where(np.isclose(V[:, 0], 0.0))[0]
    s = solve_plane_stress(V, T, E=70e9, nu=0.33, thickness=0.01, fixed={"x": left, "y": [0]},
                           traction=[(right, (1e6, 0.0))])
    assert np.allclose(s.stress, [1e6, 0, 0], atol=1e-2) and s.reaction[:, 0].sum() == pytest.approx(-1e4)
    assert heywood_kt_net(1 / 3) == pytest.approx(2 + (2 / 3) ** 3)


def test_mesh_thickness_and_heat_sink_closed_form():
    from pinneapple_tools.nvidia.brep import mesh_thickness, vertex_normals
    from pinneapple_tools.nvidia.parts import heat_sink, heat_sink_geometry
    V = np.array([[x, y, z] for x in (0, 2.0) for y in (0, 1.0) for z in (0, 0.25)])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    if np.dot(vertex_normals(V, F)[7], [1, 1, 1]) < 0:            # make the normals point outwards
        F = F[:, ::-1]
    th = mesh_thickness(V, F)
    assert np.all(np.isfinite(th)) and th.min() > 0.2
    g = heat_sink_geometry(n_fins=10, fin_height=0.03, fin_t=0.002)
    assert g["volume"] == pytest.approx(0.1 * 0.1 * 0.005 + 10 * 0.1 * 0.002 * 0.03)
    assert g["gap"] == pytest.approx((0.1 - 10 * 0.002) / 9)
    r = heat_sink(n_fins=4, fin_height=0.02, fin_t=0.002, root_fillet=0.001)
    assert r.result == "hs_filleted" and sum(o["op"] == "box" for o in r.ops) == 5


def test_heat_sink_design_without_brep():
    from pinneapple_tools.nvidia.heatsink import Duty, evaluate_design, sample_designs
    d = sample_designs(6, seed=0)
    assert all((0.1 - n * t) / (n - 1) >= 0.002 for n, _, t in d)
    r = evaluate_design(*d[0], Duty(), brep=False)
    assert r["t_source"] > Duty().t_ambient_c and r["dp"] > 0 and r["mass_closed_form"] > 0


@needs_brep
def test_brep_part_measures_and_checks(tmp_path):
    from pinneapple_tools.nvidia.brep import (
        PartRecipe,
        evaluate_part,
        geometry_checks,
        planar_face_mesh,
    )
    from pinneapple_tools.nvidia.parts import l_bracket, l_bracket_volume
    p = PartRecipe().box("plate", (0, 0, 0), (0.1, 0.06, 0.01)).cylinder("hole", (0.05, 0.03, -0.01), 0.01, 0.03) \
        .difference("b", "plate", "hole")
    r = evaluate_part(p, chord_tolerances=[1e-4, 1e-5], usd_brep_path=str(tmp_path / "b.usda"), vertex_curvature=True)
    assert r.volume == pytest.approx(0.1 * 0.06 * 0.01 - np.pi * 1e-4 * 0.01, rel=1e-9)
    assert r.min_radius == pytest.approx(0.01, rel=1e-6) and os.path.exists(tmp_path / "b.usda")
    assert all(c["passed"] for c in geometry_checks(r, element_size=0.002))
    assert r.tessellations[1]["max_deviation"] < r.tessellations[0]["max_deviation"]
    V2, F2 = planar_face_mesh(*r.mesh(), axis=2, value=0.01)
    assert V2.shape[1] == 2 and len(F2) > 0
    rb = evaluate_part(l_bracket(), chord_tolerances=[])
    assert rb.volume == pytest.approx(l_bracket_volume(), rel=1e-6) and rb.manifold


@needs_brep
@pytest.mark.parametrize("name", ["05_brep_plate_with_hole", "14_brep_omniverse_bracket"])
def test_example_brep_smoke(tmp_path, name):
    if name.startswith("14") and not HAS_PXR:
        pytest.skip("usd-core not installed")
    s = _example(name)["main"](True, str(tmp_path))
    assert s["decision"] in ("APPROVED", "REVIEW", "REJECT")
    assert os.path.exists(tmp_path / "results" / "architecture.png")


# ------------------------------------------------------------------------------------------------ Model Services (N)
def test_model_services_routes_and_trust():
    from pinneapple_tools.model_services import (
        ModelServiceClient,
        ServiceError,
        default_registry,
        serve_in_thread,
    )
    srv, url = serve_in_thread(default_registry())
    try:
        c = ModelServiceClient(url)
        assert c.ready()
        ids = {m["id"] for m in c.models()}
        assert {"pinneapple-intent", "pinneapple-trust-card", "airfoil_cst_rans_mlp"} <= ids
        x = dict(T0=0.169591, T1=0.144959, T2=0.131433, M0=0.024922, M1=0.071088, M2=0.058960)
        ok = c.infer("airfoil_cst_rans_mlp", {**x, "alpha_deg": [4.0]})
        assert ok["trust"]["decision"] == "APPROVED" and abs(ok["outputs"]["Cl"][0] - 0.6625) < 0.02
        out = c.infer("airfoil_cst_rans_mlp", {**x, "alpha_deg": [25.0]})
        assert out["trust"]["decision"] != "APPROVED" and not out["trust"]["in_validity_range"] and out["warnings"]
        with pytest.raises(ServiceError) as e:
            c.infer("nope", {})
        assert e.value.status == 404
        with pytest.raises(ServiceError) as e:
            c.chat("airfoil_cst_rans_mlp", [{"role": "user", "content": "hi"}])
        assert e.value.status == 400
        good = c.chat("pinneapple-intent", [{"role": "user", "content": "An aluminium shelf bracket that holds 1.5 kN "
                                                                         "at 120 mm from the wall, safety factor 2"}],
                      guided_json={"type": "object", "required": ["template", "values"]})
        assert good["object"] == "chat.completion" and good["pinneapple"]["trust"]["decision"] == "APPROVED"
        assert ModelServiceClient.content_json(good)["values"]["arm_length"]["value"] == pytest.approx(0.12)
        bad = c.chat("pinneapple-intent", [{"role": "user", "content": "A bracket holding 1.5 kN on a 120 mm arm, "
                                                                        "50 mm wide"}],
                     guided_json={"type": "object", "required": ["nonexistent"]})
        assert bad["pinneapple"]["trust"]["decision"] != "APPROVED" and bad["pinneapple"]["guided_json_errors"]
    finally:
        srv.shutdown()


def test_model_services_api_key_and_grid_services():
    from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob
    from pinneapple_tools.model_services import (
        ModelServiceClient,
        Registry,
        ServiceError,
        serve_in_thread,
    )
    from pinneapple_tools.model_services.grid import darcy_solver_service, grid_operator_service

    class Exact:                                         # a "surrogate" that returns the solver's answer
        def predict(self, a):
            return ob.darcy_2d(a)

    class Scaled:                                        # one that is 20 % off: breaks the flux balance
        def predict(self, a):
            return 1.2 * ob.darcy_2d(a)

    reg = Registry([grid_operator_service(Exact(), "exact"), grid_operator_service(Scaled(), "scaled"),
                    darcy_solver_service()])
    srv, url = serve_in_thread(reg, api_key="k")
    try:
        with pytest.raises(ServiceError) as e:
            ModelServiceClient(url).models()
        assert e.value.status == 401
        c = ModelServiceClient(url, api_key="k")
        a = ob.threshold_coefficient(ob.periodic_grf(1, (32, 32), alpha=2.0, tau=3.0, seed=0))[0]
        assert c.infer("exact", {"a": a.tolist()})["trust"]["decision"] == "APPROVED"
        assert c.infer("pinneapple-darcy-fv", {"a": a.tolist()})["trust"]["decision"] == "APPROVED"
        assert c.infer("scaled", {"a": a.tolist()})["trust"]["decision"] == "REJECT"
    finally:
        srv.shutdown()


def test_example_06_model_services_smoke(tmp_path):
    s = _example("06_model_services_zoo")["main"](True, str(tmp_path))
    assert s["decision"] == "APPROVED" and os.path.exists(tmp_path / "results" / "sweep.png")


@needs_brep
def test_example_15_request_to_bracket_smoke(tmp_path):
    s = _example("15_model_services_brep_bracket")["main"](True, str(tmp_path))
    assert s["answers"][0]["decision"] == "APPROVED" and s["answers"][1]["decision"] != "APPROVED"


def test_physical_world_ground_truth_passes_and_edits_are_caught():
    from pinneapple_worldmodel.physical_world import (
        bouncing_ball_scene,
        corrupt,
        physics_checks,
        video_check,
    )
    for seed in range(4):
        v = bouncing_ball_scene(seed=seed)
        r = physics_checks(v)
        assert r["passed"], [c for c in r["checks"] if not c["passed"]]
        assert abs(r["g_fit"] - 9.81) / 9.81 < 0.05
        for kind in ("time_warp", "fade", "gravity_flip"):
            assert not physics_checks(corrupt(v, kind, seed=seed))["passed"], kind
    assert video_check(1)["decision"] == "APPROVED" and video_check(1, "fade")["decision"] == "REJECT"


def test_physical_world_state_encoder_reads_positions():
    from pinneapple_worldmodel.physical_world import StateEncoder, dataset
    train, test = dataset(16, seed=1), dataset(2, seed=99)
    enc = StateEncoder().fit(train, steps=300)
    c = enc.predict(test[0].frames)
    err = np.hypot(c[:, 0] - test[0].labels["cx_px"], c[:, 1] - test[0].labels["cy_px"])
    assert np.median(err) < 4.0


@needs_pxr
def test_inspection_label_follows_physics_and_glint_is_rejected(tmp_path):
    from pinneapple_tools.nvidia.omniverse import validate_stage
    from pinneapple_worldmodel.inspection_data import inspection_sample, thermal_plate
    p = thermal_plate((0.4, 0.6, 0.1))
    assert p["rise"] > 0.05 and np.hypot(p["label_xy"][0] - 0.4, p["label_xy"][1] - 0.6) < 0.1
    a, b = thermal_plate((0.4, 0.6, 0.1), 48)["label_xy"], thermal_plate((0.4, 0.6, 0.1), 96)["label_xy"]
    assert np.hypot(a[0] - b[0], a[1] - b[1]) < 1.5 / 48
    ok = inspection_sample(0.4, 0.6, 0.1, seed=3, usd_path=str(tmp_path / "a.usda"))
    assert ok["decision"] == "ACCEPTED"
    assert validate_stage(ok["usd"], required_primvars={"/World/Plate": ["temperature", "radiance"]})["ok"]
    rejected = [inspection_sample(0.4, 0.6, 0.07, seed=s, glint=True, usd_path=str(tmp_path / f"g{s}.usda"))
                ["decision"] == "REJECTED" for s in range(4)]
    assert sum(rejected) >= 2


def test_example_04_world_model_video_smoke(tmp_path):
    s = _example("04_worldmodel_physics_video")["main"](True, str(tmp_path))
    assert os.path.exists(tmp_path / "results" / "architecture.png") and "decision" in s


@needs_pnemo
def test_example_09_state_world_model_smoke(tmp_path):
    s = _example("09_worldmodel_physicsnemo_state")["main"](True, str(tmp_path))
    assert os.path.exists(tmp_path / "results" / "rollouts.png") and "decision" in s


@needs_pxr
def test_example_13_inspection_smoke(tmp_path):
    s = _example("13_omniverse_worldmodel_inspection")["main"](True, str(tmp_path))
    assert all(u["ok"] for u in s["usd"]) and os.path.exists(tmp_path / "results" / "samples.png")


@needs_newton
@needs_pxr
def test_robot_perception_labels_frames_and_payload(tmp_path):
    from pinneapple_simulation.robot_sim import TwoLinkArm
    from pinneapple_tools.nvidia.omniverse import validate_stage
    from pinneapple_worldmodel.robot_perception import (
        CameraVariation,
        arm_motions,
        camera_frames,
        forward_points,
        payload_track_check,
        visibility,
        write_motion_stage,
    )
    arm = TwoLinkArm(payload=2.2)
    m = arm_motions(arm, 1, 0, moves=[((-0.6, 1.2), (0.9, -0.4))])[0]
    assert np.abs(forward_points(arm, m["q"])["elbow"] - m["body_pos"][:, 2][:, [0, 2]]).max() < 1e-6
    cam = CameraVariation(albedo=(0.5, 0.5))
    V = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05)])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    p = write_motion_stage(str(tmp_path / "m.usda"), arm, m, cam, payload=(V, F, 1.0))
    assert validate_stage(p)["ok"]
    fr = camera_frames(p, arm, m, cam, 96, 96)
    assert fr.shape == (len(m["t"]), 96, 96) and visibility(arm, m, cam).all()
    assert payload_track_check(fr, arm, m, cam)["max_px"] <= 2.0
    occluded = CameraVariation(occluder=(*forward_points(arm, m["q"][:1])["elbow"][0], 0.3, 0.3))
    assert not visibility(arm, m, occluded)[0]


@needs_newton
@needs_pxr
def test_example_17_perception_smoke(tmp_path):
    s = _example("17_omniverse_robotsim_worldmodel_perception")["main"](True, str(tmp_path))
    assert s["label_error"] < 1e-6 and os.path.exists(tmp_path / "results" / "frames.png")


@needs_newton
@needs_pxr
@needs_pnemo
@needs_brep
def test_example_18_work_cell_smoke(tmp_path):
    s = _example("18_all_stacks_heatsink_cell")["main"](True, str(tmp_path))
    assert s["part"]["t_source"] <= 75.0 and s["pixel_track_max_px"] <= 2.0 and s["usd"]["ok"]
