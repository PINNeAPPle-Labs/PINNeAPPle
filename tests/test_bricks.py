"""Interlocking bricks (pinneapple_design.bricks): parts, assemblies, LDraw IO, statics against closed forms, FEA,
goals, the self-modifying search and its visuals. Newton and Blender tests skip when those are not installed."""
import importlib.util
import json
import os

import numpy as np
import pytest

from pinneapple_design.bricks import closed_form as cf
from pinneapple_design.bricks import ldraw, mesh, parts
from pinneapple_design.bricks.assembly import Assembly, Ground, stack
from pinneapple_design.bricks.goals import (
    BridgeGoal,
    CantileverGoal,
    TowerGoal,
    reference_cantilever,
)
from pinneapple_design.bricks.parts import G, JointModel
from pinneapple_design.bricks.statics import (
    Load,
    Structure,
    analyse,
    analyse_assembly,
    capacity,
    from_assembly,
    progressive_failure,
    top_load,
)

HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None
HAS_BPY = importlib.util.find_spec("bpy") is not None


# ---------------------------------------------------------------------------------------------- parts
def test_dimensions_follow_the_ldraw_grid():
    assert parts.PITCH_MM == pytest.approx(20 * parts.LDU_MM) and parts.PLATE_MM == pytest.approx(8 * parts.LDU_MM)
    assert parts.BRICK_MM == pytest.approx(3 * parts.PLATE_MM) == pytest.approx(24 * parts.LDU_MM)
    assert np.allclose(parts.part("brick_2x4").size_mm, (31.8, 15.8, 9.6))
    assert parts.part("3001.dat").name == "brick_2x4" and parts.part("3001").nx == 4
    assert len({p.ldraw for p in parts.CATALOG.values()}) == len(parts.CATALOG)


def test_masses_published_and_shell_model():
    assert parts.part("brick_2x4").mass == pytest.approx(2.16e-3)
    assert parts.part("brick_1x8").mass_source == parts.MASS_SOURCE
    res = parts.mass_model_residuals()
    assert max(abs(v) for v in res.values()) < 0.3                  # worst: 1x1 brick, +28 %
    assert abs(res["8x1x3"]) < 0.05 and abs(res["4x2x3"]) < 0.05
    p = parts.part("plate_2x4")
    assert "shell model" in p.mass_source and 0 < p.mass < parts.part("brick_2x4").mass


# ---------------------------------------------------------------------------------------------- assembly
def test_connections_overlaps_and_validity():
    a = Assembly()
    a.add("brick_2x4", 0, 0, 0)
    a.add("brick_2x2", 3, 0, 3)                                     # overlaps one column of studs (x = 3)
    a.add("plate_1x2", 10, 0, 9)                                    # floating
    conns = {(c.lower, c.upper): c for c in a.connections()}
    assert len(conns[(0, 1)].studs) == 2 and not conns[(-1, 0)].clutch
    issues = a.validate()
    assert any("unsupported: brick 2" in s for s in issues)
    a.add("brick_1x1", 0, 0, 0)
    assert (0, 3) in a.overlaps()
    b = Assembly.from_dict(json.loads(json.dumps(a.to_dict())))
    assert b.signature() == a.signature() and len(b) == 4
    c = Assembly(list(reversed(a.bricks)))
    assert c.signature() == a.signature()


def test_baseplate_ground_has_studs_and_components():
    a = Assembly(ground=Ground("baseplate"))
    a.add("brick_2x2", 0, 0, 0)
    a.add("brick_2x2", 4, 0, 0)
    a.add("brick_1x6", 0, 0, 3)
    g = [c for c in a.connections() if c.lower < 0]
    assert all(len(c.studs) == 4 for c in g)
    assert sorted(map(sorted, a.components())) == [[0, 1, 2]]


# ---------------------------------------------------------------------------------------------- ldraw and meshes
def test_ldraw_round_trip_with_rotation_and_mpd(tmp_path):
    a = Assembly(name="t")
    a.add("brick_2x4", 0, 0, 0, colour="blue")
    a.add("brick_2x2", 3, 1, 3, rot=1, colour="yellow")
    a.add("plate_1x4", 0, 0, 6, rot=1, colour="green")
    text = ldraw.to_ldr(a)
    assert "1 1 40 -24 20 1 0 0 0 1 0 0 0 1 3001.dat" in text and text.count("0 STEP") == 3
    back = ldraw.from_ldr(text)
    assert back.skipped == [] and sorted(b.key() for b in back.assembly.bricks) == sorted(b.key() for b in a.bricks)
    assert [b.colour for b in back.assembly.bricks] == ["blue", "yellow", "green"]
    ldraw.write_mpd({"main": a, "sub": a}, str(tmp_path / "m.mpd"))
    assert len(ldraw.read_ldr(str(tmp_path / "m.mpd")).assembly) == 6
    off = ldraw.from_ldr("1 4 3 0 0 1 0 0 0 1 0 0 0 1 3001.dat\n1 4 0 0 0 1 0 0 0 1 0 0 0 1 9999.dat\n")
    assert len(off.assembly) == 0 and len(off.skipped) == 2


def test_parts_library_parser_on_a_tiny_library(tmp_path):
    (tmp_path / "parts" / "s").mkdir(parents=True)
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "quad.dat").write_text("0 unit quad\n4 16 0 0 0 1 0 0 1 0 1 0 0 1\n")
    (tmp_path / "parts" / "s" / "sub.dat").write_text("1 16 0 0 0 1 0 0 0 1 0 0 0 1 quad.dat\n")
    (tmp_path / "parts" / "3001.dat").write_text("0 test\n3 16 0 0 0 20 0 0 0 0 20\n1 16 0 -4 0 2 0 0 0 1 0 0 0 2 s\\sub.dat\n")
    lib = ldraw.PartsLibrary(str(tmp_path))
    V, F = lib.mesh_ldu("3001.dat")
    assert V.shape == (7, 3) and F.shape == (3, 3)
    Vm, _ = lib.part_mesh_mm("3001.dat")
    assert np.allclose(Vm[:, 2].max(), 1.6)                         # LDraw y = -4 is 1.6 mm up
    meshes = lib.assembly_meshes(stack("brick_2x4", 1))
    assert len(meshes) == 1 and meshes[0][0].shape == (7, 3)
    with pytest.raises(RuntimeError):
        ldraw.fetch_parts_library(str(tmp_path / "cache"), urls=("file:///nonexistent/complete.zip",))


def test_mesh_export(tmp_path):
    a = stack("brick_2x4", 2, offsets=[0, 1], colours=["blue", "green"])
    V, F = mesh.brick_mesh(a.bricks[0])
    tri = V[F]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    studs = 8 * np.pi * 2.4 ** 2 * 1.6
    assert vol == pytest.approx(31.8 * 15.8 * 9.6 + studs, rel=0.02)
    glb = mesh.write_gltf(a, str(tmp_path / "a.glb"))
    data = open(glb, "rb").read()
    assert data[:4] == b"glTF" and b"brick_blue" in data and b"brick_green" in data
    stl = open(mesh.write_stl(a, str(tmp_path / "a.stl")), "rb").read()
    assert int.from_bytes(stl[80:84], "little") == 2 * len(F)
    assert "def Mesh" in open(mesh.write_usd(a, str(tmp_path / "a.usda"))).read()


# ---------------------------------------------------------------------------------------------- statics
def _harmonic(n, f, L=63.8):
    right = cf.harmonic_stack(n, L) + L
    right = np.cumsum(np.diff(np.concatenate([[0.0], right])) * f)
    boxes = [((r - L, 0, i * 9.6), (r, 7.8, (i + 1) * 9.6)) for i, r in enumerate(right)]
    return Structure.from_boxes(boxes, [0.003] * n, ground_rects=[(-1e3, 0, -1e3, 1e3)])


def test_harmonic_stack_limit():
    assert cf.harmonic_overhang(4, 2.0) == pytest.approx(1 + 1 / 2 + 1 / 3 + 1 / 4)
    sharp = JointModel(edge_inset_mm=0.0)                           # the closed form has sharp edges
    assert analyse(_harmonic(5, 1.0), sharp).stable
    assert not analyse(_harmonic(5, 1.0)).stable                    # default: 0.5 mm edge inset, no knife edge
    assert analyse(_harmonic(5, 0.97)).stable
    bad = analyse(_harmonic(5, 1.02), sharp)
    assert bad.status == "mechanism" and "outside" in bad.explanation and bad.tipping["distance_mm"] < 0


def test_single_block_tips_at_half_length():
    for over, ok in ((0.49, True), (0.51, False)):
        s = Structure.from_boxes([((-63.8 * (1 - over), 0, 0), (63.8 * over, 7.8, 9.6))], [0.003],
                                 ground_rects=[(-1e3, 0, -1e3, 1e3)])
        assert analyse(s, JointModel(edge_inset_mm=0.0)).stable is ok


def test_stud_cantilever_tension_matches_closed_form():
    for part_name, region, d, studs, x1 in (("brick_1x4", (0, 1, 0, 1), 16.0, [4.0], 8.0),
                                            ("brick_1x8", (0, 2, 0, 1), 32.0, [4.0, 12.0], 16.0)):
        a = Assembly(ground=Ground("baseplate", [region]))
        a.add(part_name, 0, 0, 0)
        r = analyse_assembly(a, joint=JointModel(edge_inset_mm=0.0))
        W = a.bricks[0].part.mass * G
        assert r.utilisation * r.clutch_n == pytest.approx(cf.stud_cantilever_tension(W, d, studs, x1), rel=1e-6)
        assert r.margin == pytest.approx(1 / r.utilisation)


def test_tower_push_capacity_matches_tipping_formula():
    a = stack("brick_2x2", 6)
    s = from_assembly(a, [top_load(a, (1.0, 0.0, 0.0))])
    lam = capacity(s, JointModel(edge_inset_mm=0.0))
    assert lam == pytest.approx(cf.tower_tip_force(a.mass() * G, 8.0, a.height_mm()), rel=1e-6)
    assert capacity(s) == pytest.approx(cf.tower_tip_force(a.mass() * G, 7.5, a.height_mm()), rel=1e-6)


def test_joint_failure_progressive_and_explained():
    a = Assembly(ground=Ground("baseplate", [(0, 1, 0, 1)]))
    a.add("brick_1x2", 0, 0, 0)
    for k in range(1, 4):
        a.add("brick_1x8", 0, 0, 3 * k)
    heavy = [Load(3, (60.0, 4.0, 38.4), (0.0, 0.0, -0.5))]
    r = analyse_assembly(a, heavy)
    assert r.status == "joint_failure" and "joint failure at ground" in r.explanation
    rounds = progressive_failure(a, heavy)
    assert rounds[0].new == [(-1, 0)] and rounds[-1].result.status == "mechanism"
    assert analyse_assembly(a, heavy, JointModel(clutch_n=20.0)).stable


def test_shear_cap_never_increases_capacity():
    g = BridgeGoal()
    a = Assembly(ground=g.ground())
    for x in (-6, 10):
        a.add("brick_1x8", x, 0, 0)
    for x in (-2, 6):
        a.add("brick_1x8", x, 0, 3)
    a.add("brick_1x8", 2, 0, 6)
    caps = [capacity(from_assembly(a, g.loads(a)), JointModel(shear_n=c), lam_max=1e3) for c in (None, 1.0, 0.3, 0.05)]
    assert all(caps[i] >= caps[i + 1] - 1e-9 for i in range(3)) and caps[0] > 1


def test_beam_reactions_sum_to_the_loads():
    a = Assembly(ground=Ground("table", [(0, 1, 0, 1), (15, 16, 0, 1)]))
    for x in (0, 8):
        a.add("plate_1x8", x, 0, 0)
    a.add("plate_1x4", 0, 0, 1)
    a.add("plate_1x8", 4, 0, 1)
    a.add("plate_1x4", 12, 0, 1)
    P = 0.05
    s = from_assembly(a, [top_load(a, (0, 0, -P), at=(64.0, 4.0))])
    r = analyse(s)
    ground = [k for k, i in enumerate(s.interfaces) if i.lower < 0]
    assert r.stable
    assert r.interface_compression[ground].sum() == pytest.approx(a.mass() * G + P, rel=1e-6)
    assert cf.beam_reactions(128.0, 1.0, 2.0) == pytest.approx(65.0)
    assert cf.beam_moment(64.0, 128.0, 0.0, 2.0) == pytest.approx(64.0)


def test_fea_beam_matches_timoshenko():
    from pinneapple_design.bricks import fea
    a = Assembly(ground=Ground("table", [(0, 1, 0, 1), (15, 16, 0, 1)]))
    for k in range(3):
        if k % 2 == 0:
            a.add("plate_1x8", 0, 0, k)
            a.add("plate_1x8", 8, 0, k)
        else:
            a.add("plate_1x4", 0, 0, k)
            a.add("plate_1x8", 4, 0, k)
            a.add("plate_1x4", 12, 0, k)
    m = fea.voxel_mesh(a).mesh
    L, b, h = 128.0, 8.0, 9.6
    ends = [n for n in m.nodes_on(z=0.0) if abs(m.nodes[n, 0]) < 1e-9 or abs(m.nodes[n, 0] - L) < 1e-9]
    fixed = {n: (2,) for n in ends}
    fixed[ends[0]] = (0, 1, 2)
    fixed[[n for n in ends if m.nodes[n, 0] > 1][0]] = (1, 2)
    r = fea.solve(a, [Load(0, (L / 2, 4, h), (0, 0, -1.0))], fixed_nodes=fixed, gravity=False)
    w = -r.fem.u[m.nodes_on(x=L / 2, z=0.0), 2].mean()
    E, nu = fea.E_ABS_MPA, fea.NU_ABS
    tim = cf.beam_midspan_deflection(L, E, b * h ** 3 / 12, P=1.0) + L / (4 * 5 / 6 * E / (2 * (1 + nu)) * b * h)
    assert w == pytest.approx(tim, rel=0.02)
    g = fea.solve(stack("brick_2x2", 3))
    assert g.max_displacement > 0 and g.brick_displacement[-1] >= g.brick_displacement[0]


# ---------------------------------------------------------------------------------------------- goals and search
def test_goals_score_and_explain():
    assert CantileverGoal().evaluate(CantileverGoal().seed()).feasible
    b = BridgeGoal()
    ev = b.evaluate(b.seed())
    assert not ev.feasible and ev.status == "no_deck" and 2 <= ev.violation < 3
    t = TowerGoal(push_n=0.01)
    ev = t.evaluate(t.seed())
    assert ev.status == "mechanism" and "resultant" in ev.explanation       # tips under the push...
    assert 0 < ev.violation < 1 and 0 < ev.metrics["load_factor"] < 1      # ...but carries its own weight
    ref = reference_cantilever(4)
    assert ref["single_block"] < ref["harmonic_unconnected"] < ref["com_bound"]


def test_search_strategies_equal_budget_and_deterministic():
    from pinneapple_design.bricks.search import compare, evolve, surrogate_search
    r1 = evolve(CantileverGoal(4), 40, seed=1)
    r2 = evolve(CantileverGoal(4), 40, seed=1)
    assert len(r1.history) == 40 and [c.asm.signature() for c in r1.history] == [c.asm.signature() for c in r2.history]
    assert r1.best is not None and r1.best.ev.feasible and r1.best.ev.score > 0
    s = surrogate_search(CantileverGoal(4), 40, seed=1, warmup=10, pool=3)
    assert len(s.history) == 40 and s.meta["fits"] > 0
    out = compare(lambda: CantileverGoal(4), budget=30, seeds=(0, 1), surrogate_kw={"warmup": 10, "pool": 2})
    for k in ("evolution", "surrogate"):
        assert out["summary"][k]["budget"] == 30 and len(out["summary"][k]["per_seed"]) == 2


def test_mutation_edits_cover_the_grammar():
    from pinneapple_design.bricks.search import EDITS, mutate, repair
    rng = np.random.default_rng(0)
    g = BridgeGoal()
    a = g.seed()
    seen = set()
    for _ in range(300):
        b, e = mutate(g, a, rng)
        seen.add(e)
        assert len(repair(g, b)) >= 1
    assert {"add", "move", "extend", "mirror", "counterweight", "swap", "recolour"} <= seen
    assert set(EDITS) >= seen - {"widen"}


# ---------------------------------------------------------------------------------------------- visuals
def test_timeline_scene_viewer_and_frames(tmp_path):
    from pinneapple_design.bricks import visuals as V
    from pinneapple_design.bricks.search import evolve
    g = CantileverGoal(4)
    r = evolve(g, 30, seed=0)
    frames = V.evolution_timeline(r, g, generations=[0, 1], hold=2, replay=False, final_hold=2)
    assert all(f.scenery for f in frames) and frames[-1].kind == "final"
    sc = V.timeline_scene(frames)
    man = json.load(open(sc.export(str(tmp_path / "viewer"))))
    assert len(man["step_labels"]) == len(frames) and man["visibility"]
    assert any(f["name"] == V.UTIL_FIELD for f in man["fields"])
    imgs = V.render_frames(frames[:2], size=(160, 90))
    assert imgs[0].shape == (90, 160, 3)
    V.write_gif(str(tmp_path / "a.gif"), imgs, 4)
    assert os.path.getsize(tmp_path / "a.gif") > 0
    steps = V.building_steps(r.best.asm)
    assert sum(len(s) for s in steps) == len(r.best.asm)
    md = V.write_instructions(r.best.asm, str(tmp_path / "instr"), size=(160, 100))
    text = open(md).read()
    assert "## Part list" in text and os.path.exists(tmp_path / "instr" / "build.ldr")


@pytest.mark.skipif(not HAS_NEWTON, reason="needs newton and warp")
def test_newton_replay_and_harmonic_threshold():
    from pinneapple_design.bricks.dynamics import harmonic_threshold, replay
    rows = harmonic_threshold(n=3, factors=(0.95, 1.05), duration=0.5)
    assert all(r["agree"] for r in rows)
    a = Assembly(ground=Ground("table", [(-100, 0, -100, 100)]))
    a.add("brick_1x4", -3, 0, 0)                                     # centre of mass 8 mm inside: stays
    a.add("brick_1x4", -3, 2, 0)
    a.add("brick_1x4", -1, 4, 0)                                     # centre of mass 8 mm past the edge: falls
    rp = replay(a, duration=0.4, frames=6)
    assert rp.moved.tolist() == [False, False, True]


@pytest.mark.skipif(not HAS_BPY, reason="needs bpy (Blender as a module)")
def test_blender_render(tmp_path):
    from pinneapple_design.bricks import visuals as V
    out = V.render_blender(stack("brick_2x4", 2, offsets=[0, 1]), str(tmp_path / "b.png"), samples=4, size=(160, 90))
    assert os.path.getsize(out) > 0


def test_validation_suite_and_report(tmp_path):
    from pinneapple_design.bricks import report, validation
    from pinneapple_design.bricks.search import compare
    from pinneapple_design.bricks.statics import single_stud_torsion
    v = validation.validate(newton=False, fea=False)
    assert all(r["passed"] for r in v["rows"]) and v["report"].decisao == "APPROVED"
    assert any(c.status == "not_run" for c in v["report"].checks)                    # Newton checks not run
    comp = compare(lambda: CantileverGoal(4), budget=20, seeds=(0,), surrogate_kw={"warmup": 10, "pool": 2})
    png = report.plot_comparison({"cantilever": comp["summary"]}, str(tmp_path / "c.png"))
    assert os.path.getsize(png) > 0
    best = comp["runs"]["evolution"][0].best
    sens = report.sensitivity(CantileverGoal(4), best.asm)
    assert len(sens) == 15 and all("feasible" in r for r in sens)
    md = report.write_report(str(tmp_path / "r.md"), validation=v, comparisons={"cantilever": comp["summary"]},
                             best={}, sensitivities={"cantilever": sens}, notes=["a note"])
    text = open(md).read()
    assert "Checks against closed forms" in text and "a note" in text and "0.98 N" in text
    t = Assembly()
    t.add("brick_2x2", 0, 0, 0)
    t.add("brick_1x1", 1, 1, 3)
    t.add("brick_1x4", 0, 1, 6)                                       # held by one stud, pushed off its axis
    push = [Load(2, (28.0, 12.0, 28.8), (0.0, 0.001, 0.0))]
    res = single_stud_torsion(t, push)
    assert res["single_stud_joints"] == 2 and res["status"] == "mechanism" and res["status_with_torsion"] == "stable"


@pytest.mark.skipif(not HAS_NEWTON, reason="needs newton and warp")
def test_cross_check_and_compound_clusters():
    from pinneapple_design.bricks.dynamics import clusters, cross_check, rigid_scene
    a = Assembly(ground=Ground("baseplate", [(0, 2, 0, 2)]))
    a.add("brick_2x2", 0, 0, 0)
    a.add("brick_2x4", 0, 0, 3)
    a.add("brick_1x1", 6, 0, 0)                                       # off the baseplate: its own cluster
    groups, welded = clusters(a)
    assert sorted(map(sorted, groups)) == [[0, 1], [2]] and len(welded) == 1
    scene, layout, lift = rigid_scene(a)
    assert lift == pytest.approx(0.08) and sum(len(g) for _, g, _ in layout) == 3
    g = CantileverGoal(4)
    from pinneapple_design.bricks.search import evolve
    r = evolve(g, 30, seed=0)
    cc = cross_check(g, r.history, max_n=4, duration=0.3, dt=1e-3, iterations=16)
    assert len(cc["rows"]) == 4 and cc["agreement"] == 1.0


def test_summary_median_counts_infeasible_seeds_as_worst(tmp_path):
    from pinneapple_design.bricks import report
    curves = np.array([[np.nan, -5, -3], [np.nan, np.nan, np.nan], [-8, -8, -8], [np.nan, np.nan, np.nan],
                       [np.nan, -3, -3]], float)
    summ = {"surrogate": {"curves": curves, "per_seed": [-3.0, None, -8.0, None, -3.0], "best_median": -8.0,
                          "best_min": -8.0, "best_max": -3.0, "success_rate": 0.6, "seeds": [0, 1, 2, 3, 4],
                          "budget": 3}}
    assert float(np.median(np.where(np.isfinite(curves[:, -1]), curves[:, -1], -np.inf))) == -8.0
    png = report.plot_comparison({"bridge": summ}, str(tmp_path / "b.png"),
                                 display={"bridge": {"scale": -1, "invert": True}})
    assert os.path.getsize(png) > 0
    v = {"rows": [], "harmonic_newton": [], "report": type("R", (), {"decisao": "APPROVED", "score": 1.0,
                                                                      "cobertura": 1.0})()}
    md = open(report.write_report(str(tmp_path / "r.md"), validation=v, comparisons={"bridge": summ}, best={},
                                  sensitivities={}, display={"bridge": {"scale": -1, "unit": " [parts]"}})).read()
    assert "| surrogate | 8.0 | 3.0 to 8.0 | 3/5 | 3.0, -, 8.0, -, 3.0 |" in md
