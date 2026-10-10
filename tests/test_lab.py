"""PINNeAPPle Lab: runs, caching, validation status, failures, sweeps, datasets, export and catalogue."""
import json
import os

import numpy as np
import pytest

from pinneapple_lab import Experiment, LabStore, available, register, run, sweep
from pinneapple_lab.__main__ import main as cli


@register
class _Square(Experiment):
    name = "_test_square"
    description = "y = a x^2 sampled on a grid; fails validation when a < 0 and crashes when a is None."
    params = {"a": 1.0, "n": 5}
    space = {"a": (0.5, 2.0), "n": [3, 5, 7]}

    def run(self, ctx):
        if ctx.params["a"] is None:
            raise ValueError("a is required")
        with ctx.stage("compute"):
            x = np.linspace(0, 1, ctx.params["n"])
            y = ctx.params["a"] * x ** 2
        ctx.input("x", x)
        ctx.output("y", y)
        ctx.metric("y_max", float(y.max()))
        ctx.metric("loss", float(y.sum()), step=0)
        ctx.check("non_negative", bool((y >= 0).all()))
        ctx.check("max_value", value=float(y.max()), reference=ctx.params["a"], rtol=1e-12)
        ds = ctx.dataset("curves", units={"y": "m"})
        for xi, yi in zip(x, y, strict=True):
            ds.add(point=np.array([xi, yi]), a=ctx.params["a"], label="pos" if yi >= 0 else "neg")


def test_run_records_everything_and_is_cached(tmp_path):
    root = str(tmp_path)
    r = run("_test_square", {"a": 2.0}, root=root)
    assert r.status == "completed" and r.metrics["y_max"] == 2.0
    for f in ("run.json", "metrics.json", "validation.json", "log.txt", "progress.json", "inputs/x.npy",
              "outputs/y.npy", "datasets/curves/card.json"):
        assert os.path.exists(os.path.join(r.dir, f)), f
    rec = json.load(open(os.path.join(r.dir, "run.json")))
    assert rec["params"] == {"a": 2.0, "n": 5} and rec["validation"] == {"total": 2, "failed": 0}
    assert rec["stages"][0]["name"] == "compute"
    assert rec["code"]["files"] == ["code/test_lab.py"] and os.path.exists(os.path.join(r.dir, "code/test_lab.py"))
    again = run("_test_square", {"a": 2.0}, root=root)
    assert again.status == "cached:completed" and again.run_id == r.run_id
    card = json.load(open(os.path.join(r.dir, "datasets/curves/card.json")))
    assert card["n_samples"] == 5 and card["schema"]["point"]["shape"] == [2]


def test_failed_validation_and_crash_are_recorded(tmp_path):
    root = str(tmp_path)
    bad = run("_test_square", {"a": -1.0}, root=root)
    assert bad.status == "failed_validation"
    crash = run("_test_square", {"a": None}, root=root)
    assert crash.status == "failed" and "a is required" in crash.error
    st = LabStore(root).status()["_test_square"]
    assert st == {"failed_validation": 1, "failed": 1}
    with pytest.raises(KeyError):
        run("_test_square", {"b": 1}, root=root)


def test_sweep_grid_and_samples_export_and_catalog(tmp_path):
    root = str(tmp_path)
    res = sweep("_test_square", grid={"a": [1.0, 2.0], "n": [3, 4]}, root=root)
    assert len(res) == 4 and all(r.ok for r in res)
    res2 = sweep("_test_square", samples=5, seed=1, root=root)
    assert len({r.run_id for r in res2}) == 5
    store = LabStore(root)
    assert len(store.runs("_test_square")) == 9
    tab = store.table("_test_square")
    assert {"param.a", "param.n", "metric.y_max"} <= set(tab.columns)
    samples = list(store.samples("_test_square", "curves"))
    assert len(samples) == 3 + 4 + 3 + 4 + sum(r_["params"]["n"] for r_ in store.runs("_test_square")[4:])
    card = store.export_dataset("_test_square", "curves", str(tmp_path / "curves.npz"))
    z = np.load(tmp_path / "curves.npz")
    assert z["point"].shape == (card["n_samples"], 2) and z["param__a"].shape == (card["n_samples"],)
    cat = store.catalog()
    text = open(cat).read()
    assert "_test_square" in text and "| run | status |" in text
    # the index can be rebuilt from the folders alone
    os.remove(store.db)
    assert LabStore(root).reindex() == 9


def test_builtin_experiments_run(tmp_path):
    root = str(tmp_path)
    assert {"oscillator", "heat_xtfc", "bondi_accretion", "accretion_flow", "cylinder_lbm"} <= set(available())
    r = run("oscillator", {"method": "rk4", "zeta": 0.2}, root=root)
    assert r.ok and r.metrics["rmse"] < 1e-6
    r = run("heat_xtfc", {"alpha": 0.2}, root=root)
    assert r.ok and r.metrics["rel_l2_error"] < 1e-3
    pytest.importorskip("numba")
    r = run("bondi_accretion", {"nr": 48, "t_end": 100.0}, root=root)
    assert r.ok and r.metrics["mdot_rel_error"] < 0.02
    r = run("accretion_flow", {"nr": 32, "ntheta": 16, "t_end": 60.0, "every": 20.0}, root=root)
    assert r.ok and r.metrics["frames"] == 4
    r = run("cylinder_lbm", {"Re": 30.0, "D": 8, "height": 4, "length": 8, "steps": 600, "save_every": 50}, root=root)
    assert r.ok and r.metrics["regime_shedding"] == 0
    assert LabStore(root).datasets("cylinder_lbm", "vorticity")[0]["n_samples"] == 6


def test_parallel_sweep_and_cli(tmp_path, capsys):
    root = str(tmp_path)
    res = sweep("oscillator", grid={"zeta": [0.1, 0.4], "method": ["rk4", "symplectic"]}, n_jobs=2, root=root)
    assert all(r.ok for r in res)
    assert cli(["--root", root, "status"]) == 0
    assert "oscillator" in capsys.readouterr().out
    assert cli(["--root", root, "run", "oscillator", "-p", "zeta=0.3"]) == 0
    assert cli(["--root", root, "report"]) == 0
    assert cli(["--root", root, "export", "oscillator", "trajectories", str(tmp_path / "osc.npz")]) == 0
    assert np.load(tmp_path / "osc.npz")["x"].shape[0] == 5


def test_law_discovery_experiments(tmp_path):
    root = str(tmp_path)
    r = run("kepler_law", root=root)
    assert r.ok and abs(r.metrics["Sun_exponent"] - 1.5) < 0.002
    assert abs(r.metrics["sun_to_jupiter_mass_ratio"] / 1047.35 - 1) < 0.01
    r = run("lorenz_discovery", root=root)
    assert r.ok and r.metrics["max_coefficient_rel_error"] < 0.05
    r = run("pendulum_video", {"seconds": 8.0, "predict_seconds": 3.0, "size": 120}, root=root)
    assert r.ok and r.metrics["g_rel_error"] < 0.02 and r.metrics["n_terms"] == 2
    sweep("oscillator", grid={"zeta": [0.1, 0.3], "omega": [1.0, 3.0]}, root=root)
    r = run("oscillator_discovery", root=root)
    assert r.ok and r.metrics["n_trajectories"] == 4 and r.metrics["omega_rel_error_median"] < 0.01


def test_repo_results_import_and_dataset_catalog(tmp_path):
    root = str(tmp_path)
    sources = ["burgers_pinn", "fin_inverse_2d", "lbm_strouhal"]
    by = {s: run("repo_results", {"source": s}, root=root) for s in sources}
    assert all(r.ok for r in by.values()), [r.status for r in by.values()]
    assert by["burgers_pinn"].metrics["rel_l2_vs_exact"] < 0.1
    assert abs(by["fin_inverse_2d"].metrics["h_pinn_mean"] / 15.0 - 1) < 0.05
    store = LabStore(root)
    assert store.datasets("repo_results", "temperature_fields_2d")[0]["n_samples"] == 5
    text = open(store.datasets_catalog()).read()
    assert "`repo_results` / `strouhal`" in text and "-g source=" in text


def test_server_routes_auth_and_downloads(tmp_path):
    import base64
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from pinneapple_lab.server import LabServer
    root = str(tmp_path)
    sweep("oscillator", grid={"zeta": [0.1, 0.3]}, root=root)
    lab = LabServer(root, refresh=0, user="u", password="p")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lab.handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    auth = {"Authorization": "Basic " + base64.b64encode(b"u:p").decode()}

    def get(path, headers=auth):
        with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers)) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    try:
        assert get("/health", {})[0] == 200
        with pytest.raises(urllib.error.HTTPError) as e:
            get("/", {})
        assert e.value.code == 401
        code, ctype, body = get("/")
        assert code == 200 and ctype.startswith("text/html") and b"PINNeAPPle Lab" in body
        cat = json.loads(get("/api/catalog")[2])
        assert cat["served"] and len(cat["experiments"][0]["runs"]) == 2
        r = cat["experiments"][0]["runs"][0]
        assert get("/" + r["thumbs"][0])[1] == "image/jpeg"
        assert get("/files/" + r["figures"][0].split("/", 1)[1])[1] == "image/png"
        assert json.loads(get("/api/runs/" + r["id"])[2])["index"]["run_id"] == r["id"]
        with pytest.raises(urllib.error.HTTPError) as e:
            get("/files/../index.sqlite")
        assert e.value.code == 404
        code, _, data = get("/download/oscillator/trajectories.npz")
        assert code == 200 and data[:2] == b"PK"
    finally:
        httpd.shutdown()


def test_cache_recomputes_runs_whose_shards_are_missing(tmp_path):
    root = str(tmp_path)
    r = run("oscillator", {"zeta": 0.2}, root=root)
    shard = os.path.join(r.dir, "datasets", "trajectories", "shard_00000.npz")
    assert run("oscillator", {"zeta": 0.2}, root=root).status.startswith("cached")
    os.remove(shard)                                    # what a fresh git checkout of the lab looks like
    again = run("oscillator", {"zeta": 0.2}, root=root)
    assert again.status == "completed" and os.path.exists(shard)


def test_example_script_collects_outputs_and_leaves_checkout_clean(tmp_path):
    import shutil

    from pinneapple_lab.experiments.examples import REPO, discover_examples
    assert any(x["group"].startswith("use_cases/") for x in discover_examples())
    folder = os.path.join(REPO, "examples", "_lab_test_tmp")
    os.makedirs(folder, exist_ok=True)
    try:
        with open(os.path.join(folder, "demo.py"), "w") as f:
            f.write("import json, os\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
                    "here = os.path.dirname(__file__)\n"
                    "plt.plot([0, 1], [0, 1]); plt.savefig('lab_test_demo_figure.png')\n"
                    "json.dump({'rel_l2': 0.01, 'nested': {'cl': 0.5}}, open(os.path.join(here, 'm.json'), 'w'))\n"
                    "np.save(os.path.join(here, 'u.npy'), np.arange(6.0).reshape(2, 3))\n"
                    "print('relative L2 error: 1.5e-03')\n")
        r = run("example_script", {"script": "examples/_lab_test_tmp/demo.py", "timeout": 120, "isolated": False},
                root=str(tmp_path))
        assert r.ok, r.status
        assert r.metrics["m.rel_l2"] == 0.01 and r.metrics["m.nested.cl"] == 0.5
        assert r.metrics["stdout.relative_l2_error"] == 1.5e-3
        assert any(f.endswith("lab_test_demo_figure.png") for f in os.listdir(os.path.join(r.dir, "figures")))
        assert not os.path.exists(os.path.join(REPO, "lab_test_demo_figure.png"))      # moved out of the checkout
        assert LabStore(str(tmp_path)).datasets("example_script", "artifacts")[0]["n_samples"] == 1
        assert any(c.endswith("demo.py") for c in json.load(open(os.path.join(r.dir, "run.json")))["code"]["files"])
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        if os.path.exists(os.path.join(REPO, "lab_test_demo_figure.png")):
            os.remove(os.path.join(REPO, "lab_test_demo_figure.png"))


def test_car_geometry_and_wind_tunnel_smoke():
    from pinneapple_design.geometry.gen.car2d import DESIGN_SPACE, CarProfile, sample_designs
    from pinneapple_lab.experiments.car import simulate_car
    designs = sample_designs(12, seed=1)
    for d in designs:
        for k, (lo, hi) in DESIGN_SPACE.items():
            assert lo <= d[k] <= hi
    fast, steep = CarProfile(slant_deg=8), CarProfile(slant_deg=38)
    m1, m2 = fast.mask(160, 56, 32, 38), steep.mask(160, 56, 32, 38)
    assert m1.sum() > m2.sum() > 0                       # a steeper slant removes rear volume
    assert not m1[:, 0].any()                            # ground clearance: the road row is fluid
    sim = simulate_car(designs[0], length_cells=16, steps=300, keep_frames=2)
    assert sim["finite"] and np.isfinite(sim["cd_cv"]) and len(sim["frames"]) == 2
    assert sim["mean_ux"].shape == sim["sdf"].shape == (80, 40)


def test_bar_wear_physics():
    from pinneapple_physics.tribology import WEAR_MATERIALS, BarWear, winkler_parabolic_contact
    mat = WEAR_MATERIALS["PTFE"]
    bar = BarWear(mat, load_N=50, nx=201)
    p0, _ = bar.pressure()
    exact = winkler_parabolic_contact(50, bar.width_mm, bar.crown_radius_mm, bar.k_w)
    assert abs(p0.max() / exact["p_max_MPa"] - 1) < 0.02
    r = bar.run(1.5 * bar.running_in_distance_m(), n_save=10)
    assert abs(r["volume"][-1] / (mat.K * 50 * r["s"][-1]) - 1) < 1e-6        # Archard: V = K F s exactly
    assert r["contact"][-1] > 0.999 and abs(r["p_max"][-1] / (50 / bar.area_mm2) - 1) < 0.02
    ks = sorted(WEAR_MATERIALS.values(), key=lambda m: m.K)
    assert ks[0].name == "tungsten carbide" and ks[-1].name == "mild steel"


@register
class _Curated(Experiment):
    name = "_test_curated"
    description = ("A toy experiment with a reference check, a baseline check and three figures, to exercise the "
                   "curation tiers.")
    tags = ["verification"]
    references = ["Toy reference (unit test)"]
    params = {"error": 0.01, "with_checks": True}

    def run(self, ctx):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if ctx.params["with_checks"]:
            ctx.check("rel_l2_vs_exact", value=ctx.params["error"], max=0.05)
            ctx.check("beats_persistence", value=ctx.params["error"], max=0.2)
        for k in range(3):
            plt.figure()
            plt.plot([0, 1], [0, k])
            ctx.figure(f"f{k}")
        ctx.metric("error", ctx.params["error"])


def test_curation_tiers_readiness_and_review(tmp_path):
    from pinneapple_lab.curation import curate, infer_check_kind, save_review
    root = str(tmp_path)
    good = run("_test_curated", root=root)
    bad = run("_test_curated", {"error": 0.5}, root=root)
    thin = run("_test_curated", {"with_checks": False}, root=root)
    out = curate(LabStore(root))
    tiers = {rid: a["tier"] for rid, a in out["runs"].items()}
    assert tiers[good.run_id] == "A" and tiers[bad.run_id] == "D" and tiers[thin.run_id] == "C"
    a = out["runs"][good.run_id]
    assert a["dimensions"]["reference"]["status"] == "PASS" and a["dimensions"]["uncertainty"]["status"] == "NOT_RUN"
    assert not a["readiness"]["marketing"]["ready"]                  # no story yet
    save_review(root, "_test_curated", story="Toy problem solved to 1 %", novelty=4, clarity=4, visual_appeal=4,
                approved_for=["paper"])
    out = curate(LabStore(root))
    a = out["runs"][good.run_id]
    assert a["reviewed"] and a["readiness"]["paper"]["approved"]
    assert out["experiments"]["_test_curated"]["best_run"] == good.run_id
    assert os.path.exists(os.path.join(root, "CURATION.md"))
    assert infer_check_kind({"name": "sun_jupiter_mass_ratio", "reference": 1047.35}) == "reference"
    assert infer_check_kind({"name": "finite_fields"}) == "sanity"
    assert infer_check_kind({"name": "x", "detail": "900 held-out designs"}) == "generalization"


def test_reports_one_item_and_filtered_set(tmp_path):
    from pinneapple_lab.reports import select, write_report
    root = str(tmp_path)
    run("_test_curated", root=root)
    run("_test_curated", {"error": 0.5}, root=root)
    sweep("oscillator", grid={"zeta": [0.1, 0.3]}, root=root)
    store = LabStore(root)
    h = write_report(store, str(tmp_path / "one.html"), items=["_test_curated"])
    text = open(h).read()
    assert "Path forward" in text and "Validation" in text and "data:image/jpeg;base64" in text
    md = write_report(store, str(tmp_path / "set.md"), tier=["A", "B"], use="paper")
    assert "| item | tier |" in open(md).read()
    pdf = write_report(store, str(tmp_path / "set.pdf"), tier=["A", "B", "C", "D"])
    assert open(pdf, "rb").read(4) == b"%PDF"
    from pinneapple_lab.curation import curate
    cur = curate(store)
    assert select(cur, ready="training_data") and not select(cur, tier=["Z"])
    with pytest.raises(ValueError):
        write_report(store, str(tmp_path / "none.html"), tier=["Z"])


def test_internal_flow_geometry_and_references():
    import math

    from pinneapple_simulation.numerical_solvers.internal_flow import (
        Route,
        colebrook,
        ito_bend_loss,
        kenics_elements,
        oblock_mesh_dict,
    )
    rt = Route(D=0.05).straight(5).bend(2.0, 90).straight(5)
    st = rt.stations()
    assert abs(st["s"][-1] - rt.length) < 1e-12 and abs(rt.length / 0.05 - (10 + math.pi)) < 1e-9
    assert np.allclose(st["T"][-1], [0, 1, 0], atol=1e-9)              # turned 90 degrees towards +y
    assert np.allclose(np.einsum("ij,ij->i", st["T"], st["N"]), 0, atol=1e-9)
    d = oblock_mesh_dict(rt)
    n_int = len(st["P"]) - 1
    assert d.count("hex (") == 5 * n_int and "inlet" in d and "outlet" in d
    V, F = kenics_elements(0.05, 2, 1.0)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert (cnt == 2).all()                                             # closed surface for snappyHexMesh
    assert abs(colebrook(1e5) - 0.0180) < 3e-4                          # Moody chart, smooth pipe
    assert 0.15 < ito_bend_loss(5e4, 4.0) < 0.3


def test_solid_fem_cantilever_vs_beam_theory():
    from pinneapple_simulation.numerical_solvers.solid_fem import SolidFEM, box_mesh, von_mises
    L, W, H, P, E, nu = 2.0, 0.1, 0.2, 1e4, 210e9, 0.3
    m = box_mesh(L, W, H, 20, 2, 4)
    f = SolidFEM(m, E, nu)
    f.fix(m.nodes_on(x=0.0))
    f.load_face(m.face_nodes("x+"), (0.0, 0.0, -P))
    r = f.solve()
    Iy, G = W * H ** 3 / 12, E / (2 * (1 + nu))
    ref = P * L ** 3 / (3 * E * Iy) + P * L / (5 / 6 * G * W * H)
    assert abs(-r.u[m.face_nodes("x+"), 2].mean() / ref - 1) < 0.03          # no shear locking with C3D8I
    assert np.allclose(r.reactions.sum(0), [0, 0, P], rtol=1e-8)
    s = np.zeros((1, 6))
    s[0, 0] = 100.0
    assert abs(von_mises(s)[0] - 100.0) < 1e-12
