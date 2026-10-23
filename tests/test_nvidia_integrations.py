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
def test_example_02_pipeline_twin_smoke(tmp_path):
    s = _example("02_pump_pipeline_twin_omniverse")["main"](True, str(tmp_path))
    assert s["usd"]["ok"] and s["checks"][0]["passed"] and s["checks"][1]["passed"]
    for f in ("architecture.png", "transients.png", "usd_preview.png", "usd_timeline.gif", "summary.json"):
        assert os.path.exists(tmp_path / "results" / f)
    assert os.path.exists(tmp_path / "experiment.json")


@needs_pxr
@needs_pnemo
def test_example_01_coldplate_smoke(tmp_path):
    s = _example("01_coldplate_physicsnemo_omniverse")["main"](True, str(tmp_path))
    assert s["decision"] in ("APPROVED", "REVIEW", "REJECT") and s["usd"]["ok"]
    assert os.path.exists(tmp_path / "results" / "usd_preview.png")


@needs_pnemo
def test_example_03_plume_smoke(tmp_path):
    s = _example("03_plume_rollout_physicsnemo")["main"](True, str(tmp_path))
    assert s["decision"] in ("APPROVED", "REVIEW", "REJECT")
    assert os.path.exists(tmp_path / "results" / "rollout_checks.png")
