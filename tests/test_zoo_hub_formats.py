"""How the zoo is published on the Hugging Face Hub: Parquet copies of the datasets (round trip, shard sizes,
configs), the README YAML the Hub and ``datasets`` read, the model cards and their "How to use" code, the
republication revisions, the publishing checks (viewer, Space) and the demo Space."""
from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from pinneapple_hub.zoo import _artifacts as art  # noqa: E402
from pinneapple_hub.zoo import hub_cards  # noqa: E402
from pinneapple_hub.zoo import parquet as zpq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = [n for n, e in art.manifest()["datasets"].items() if all(f.get("bundled") for f in e["files"].values())]
HUB_ONLY = [n for n in art.manifest()["datasets"] if n not in BUNDLED]


def _npz_paths(name):
    return [art.ASSETS / f["bundled"] for f in art.entry("datasets", name)["files"].values()]


def _write(name, data, out, **kw):
    e = art.entry("datasets", name)
    return zpq.write_dataset(name, data, out, split_order=hub_cards.split_order(name),
                             column_order=list(e["columns"]), **kw)


def _assert_same(back, orig):
    assert set(back) == set(orig)
    for k, a in orig.items():
        assert back[k].dtype == a.dtype and back[k].shape == a.shape, k
        assert np.array_equal(back[k], a, equal_nan=a.dtype.kind == "f"), k


_DTYPE = re.compile(r"^([a-z0-9<U]+?)(?:\[([\d, ]+)\])?$")
# columns of the Hub-only datasets that only some splits carry (pinneapple_physics.blackhole.dataset.build_arrays)
_BH_SPLITS = {"grid_r": ["r", "r_face_lo", "r_face_hi"], "grid_theta": ["theta", "theta_face_lo", "theta_face_hi"],
              "convergence": ["nr", "ntheta", "seed", "ell_over_kepler_r10", "ell_over_kepler_r30",
                              "mdot_detrended_rms", "mdot_mean", "mdot_psd_slope", "mdot_rms_over_mean",
                              "persistence_mae_1000M", "persistence_mae_100M", "persistence_mae_20M",
                              "rho_slope_r10_60", "window_mass_mean"]}


def _synthetic(name, n=3, seed=0):
    """Random arrays with the manifest's column dtypes and per-row shapes (``n`` rows per split)."""
    e = art.entry("datasets", name)
    rng = np.random.default_rng(seed)
    out = {}
    for split in e["splits"]:
        cols = e["columns"]
        if name.startswith("blackhole"):
            keep = _BH_SPLITS.get(split, ["run", "alpha", "t", "mdot", "log10_rho_q"])
            cols = {c: cols[c] for c in keep}
        out[split] = {}
        for c, d in cols.items():
            base, shape = _DTYPE.match(d["dtype"].replace(" ", "")).groups()
            shape = tuple(int(s) for s in shape.split(",")) if shape else ()
            dt = np.dtype(base)
            if dt.kind == "U":
                a = np.array([f"{c}{i}"[: dt.itemsize // 4] for i in range(n)], dt)
            elif dt.kind in "iu":
                a = rng.integers(0, 1000, (n, *shape)).astype(dt)
            else:
                a = rng.standard_normal((n, *shape)).astype(dt)
            out[split][c] = a
    return out


# ============================================================================================== round trip
@pytest.mark.parametrize("name", BUNDLED)
def test_bundled_datasets_round_trip_npz_parquet_npz(name, tmp_path):
    data = zpq.read_npz(_npz_paths(name))
    layout = _write(name, data, tmp_path)
    seen = set()
    for cfg in layout["configs"]:
        for s in cfg["splits"]:
            _assert_same(zpq.read_parquet(s["paths"]), data[s["name"]])
            assert s["rows"] == s["samples"] == art.entry("datasets", name)["splits"][s["name"]]["n"]
            seen.add(s["name"])
    assert seen == set(data) == set(art.entry("datasets", name)["splits"])
    assert sum(c["default"] for c in layout["configs"]) == 1 and layout["configs"][0]["name"] == "default"


@pytest.mark.parametrize("name", HUB_ONLY)
def test_field_datasets_round_trip_with_their_real_row_shapes(name, tmp_path):
    data = _synthetic(name)
    layout = _write(name, data, tmp_path)
    for cfg in layout["configs"]:
        for s in cfg["splits"]:
            _assert_same(zpq.read_parquet(s["paths"]), data[s["name"]])
            types = pq.read_schema(s["paths"][0])
            for f in types:                                   # arrays are fixed-size lists of their row shape
                if f.name in data[s["name"]] and data[s["name"]][f.name].ndim > 1 and f.name not in s["exploded"]:
                    assert pa.types.is_fixed_size_list(f.type), (name, f.name)


def test_large_samples_are_stored_one_row_per_step(tmp_path):
    data = _synthetic("thermal_channel_cfd", n=2)
    layout = _write("thermal_channel_cfd", data, tmp_path)
    s = layout["configs"][0]["splits"][0]
    assert s["steps"] == 101 and s["rows"] == 2 * 101 and "fields" in s["exploded"] and "Q" in s["exploded"]
    assert "Q_fine" not in s["exploded"] and "family" not in s["exploded"]          # per sample: repeated
    t = pq.read_table(s["paths"][0])
    assert t.column_names[:2] == ["sample", "step"]
    assert t.schema.field("fields").type == pa.list_(pa.list_(pa.list_(pa.float32(), 32), 128), 4)
    row = t.slice(103, 1).to_pylist()[0]
    assert (row["sample"], row["step"]) == (1, 2)
    assert np.allclose(row["fields"], data[s["name"]]["fields"][1, 2]) and row["Q"] == data[s["name"]]["Q"][1, 2]
    assert row["family"] == data[s["name"]]["family"][1]


def _broadcast(name):
    """Every split at its full size, without memory (broadcast views): enough to plan the layout."""
    e = art.entry("datasets", name)
    out = {}
    for split, arrays in _synthetic(name, n=1).items():
        n = e["splits"][split]["n"]
        out[split] = {c: np.broadcast_to(a, (n, *a.shape[1:])) for c, a in arrays.items()}
    return out


@pytest.mark.parametrize("name", HUB_ONLY)
def test_full_size_layout_keeps_rows_shards_and_row_groups_small(name):
    for split, arrays in _broadcast(name).items():
        p = zpq.plan_split(arrays)
        assert p["row_bytes"] <= zpq.EXPLODE_ROW_BYTES + 2**17, (name, split, p)   # a row the viewer can show
        assert p["rows_per_shard"] * p["row_bytes"] <= zpq.MAX_SHARD_BYTES          # files < 300 MB
        assert p["rows_per_group"] * p["row_bytes"] <= zpq.ROW_GROUP_BYTES
    nav = zpq.plan_split(_broadcast("navier_stokes_2d_li2021_nu1e-3")["train"])
    assert nav["steps"] == 51 and nav["rows"] == 51_000
    air = zpq.plan_split(_broadcast("airfoil_rans_cst_re4e6_fields")["train"])
    assert air["steps"] == 0 and air["rows"] == 340                             # 1-D fields: one row per case


def test_shards_respect_the_size_limit_and_are_deterministic(tmp_path):
    data = _synthetic("darcy_2d_li2021", n=40)                                  # 131 kB per sample
    a = _write("darcy_2d_li2021", data, tmp_path / "a", max_shard_bytes=1_000_000, row_group_bytes=300_000)
    b = _write("darcy_2d_li2021", data, tmp_path / "b", max_shard_bytes=1_000_000, row_group_bytes=300_000)
    s = a["configs"][0]["splits"][0]
    per = 1_000_000 // zpq.plan_split(data[s["name"]])["row_bytes"]
    assert len(s["files"]) == -(-40 // per) > 1
    for f in s["paths"]:
        md = pq.ParquetFile(f).metadata
        assert md.num_rows <= per
        assert all(md.row_group(i).total_byte_size <= 300_000 * 1.05 for i in range(md.num_row_groups))
    for fa, fb in zip(zpq.parquet_files(a), zpq.parquet_files(b), strict=True):
        assert (tmp_path / "a" / fa).read_bytes() == (tmp_path / "b" / fb).read_bytes()
    _assert_same(zpq.read_parquet(s["paths"]), data[s["name"]])


# ============================================================================================== dataset cards
def _card(name, tmp_path, data=None):
    data = data if data is not None else zpq.read_npz(_npz_paths(name))
    layout = _write(name, data, tmp_path)
    readme = hub_cards.dataset_card(name, layout)
    (tmp_path / "README.md").write_text(readme)
    return layout, readme


@pytest.mark.parametrize("name", [*BUNDLED, "thermal_channel_cfd", "blackhole_riaf_density_256"])
def test_dataset_card_yaml_points_at_every_parquet_file(name, tmp_path):
    data = None if name in BUNDLED else _synthetic(name, n=2)
    layout, readme = _card(name, tmp_path, data)
    meta = hub_cards.parse_front_matter(readme)
    e = art.entry("datasets", name)
    assert meta["license"] == e["license"] and meta["pretty_name"] == e["title"]
    assert meta["task_categories"] and meta["size_categories"] and "pinneapple" in meta["tags"]
    files = zpq.parquet_files(layout)
    matched = []
    for cfg, info in zip(meta["configs"], meta["dataset_info"], strict=True):
        assert cfg["config_name"] == info["config_name"]
        lay = next(c for c in layout["configs"] if c["name"] == cfg["config_name"])
        assert info["features"] == lay["features"]
        for df, sp in zip(cfg["data_files"], info["splits"], strict=True):
            assert re.fullmatch(r"\w+(\.\w+)*", df["split"]) and df["split"] == sp["name"]
            hits = [f for f in files if fnmatch.fnmatch(f, df["path"])]
            assert hits == next(s["files"] for s in lay["splits"] if s["name"] == df["split"])
            assert sp["num_examples"] == sum(pq.ParquetFile(tmp_path / f).metadata.num_rows for f in hits)
            matched += hits
    assert sorted(matched) == sorted(files)                                     # every file in exactly one split
    assert [c["config_name"] for c in meta["configs"] if c.get("default")] == ["default"]
    for col in e["columns"]:                                                    # every column is documented
        if any(col in c["columns"] for c in layout["configs"]):
            assert f"| `{col}` |" in readme
    assert f'pp.datasets.load("{name}"' in readme and f'load_dataset("{e["hf_repo"]}"' in readme
    assert all(f["sha256"] in readme for f in e["files"].values())


def test_dataset_card_yaml_is_what_huggingface_hub_and_datasets_read(tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    datasets = pytest.importorskip("datasets")
    layout, readme = _card("airfoil_rans_cst_re4e6", tmp_path)
    card = hub.DatasetCard(readme)
    assert card.data.license == "cc-by-4.0" and card.data.configs and card.data.dataset_info
    for cfg in layout["configs"]:
        feats = datasets.Features._from_yaml_list(cfg["features"])
        assert feats == datasets.Features.from_arrow_schema(pq.read_schema(cfg["splits"][0]["paths"][0]))
        for s in cfg["splits"]:                                                 # the repo layout loads as is
            ds = datasets.load_dataset(str(tmp_path), cfg["name"], split=s["name"])
            assert ds.num_rows == s["rows"] and ds.features == feats


def test_field_dataset_loads_with_datasets_as_arrays(tmp_path):
    datasets = pytest.importorskip("datasets")
    data = _synthetic("darcy_2d_li2021", n=3)
    _card("darcy_2d_li2021", tmp_path, data)
    ds = datasets.load_dataset(str(tmp_path), split="test").with_format("numpy")
    assert ds.num_rows == 3 and np.array_equal(ds[1]["a"], data["test"]["a"][1])
    assert datasets.Features.from_arrow_schema(pq.read_schema(next(tmp_path.glob("data/default/test-*"))))["a"] \
        == datasets.List(datasets.List(datasets.Value("float32"), length=128), length=128)


# ============================================================================================== models
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def model_cards(tmp_path_factory):
    pytest.importorskip("safetensors")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from pinneapple_hub.zoo import publish
    items = publish.plan(work=tmp_path_factory.mktemp("work"), only=list(art.manifest()["models"]))
    return {it["name"]: it for it in items if it["kind"] == "models"}


def test_model_cards_promise_only_what_works(model_cards):
    hub = pytest.importorskip("huggingface_hub")
    assert set(model_cards) == set(art.manifest()["models"])
    for name, it in model_cards.items():
        meta = hub_cards.parse_front_matter(it["readme"])
        e = art.entry("models", name)
        assert meta["library_name"] == "pinneapple" and meta["pipeline_tag"] == "tabular-regression"
        assert meta["inference"] is False and "## Hosted inference" in it["readme"]
        assert meta["datasets"] == [art.entry("datasets", e["dataset"])["hf_repo"]]
        assert hub.ModelCard(it["readme"]).data.pipeline_tag == "tabular-regression"
        assert hub_cards.space_url() in it["readme"] and "## Validity range" in it["readme"]
        assert "## Trust" in it["readme"] and "## Physics validation" in it["readme"]
        assert set(it["derived"]) == {"model.safetensors", "model.onnx", "example.json"}
        assert all(f["sha256"] in it["readme"] for f in e["files"].values())
        assert f'revision="{e["revision"]}"' in it["readme"]                   # the ONNX snippet's pinned tag


def test_safetensors_and_onnx_copies_give_the_same_predictions(model_cards):
    from pinneapple_hub import zoo
    from pinneapple_hub.zoo.export import load_safetensors, onnx_parity, parity_inputs
    for name, it in model_cards.items():
        zm = zoo.load(name)
        X = parity_inputs(zm, n_random=100)
        a, b = zm._forward(X), load_safetensors(zm, it["derived"]["model.safetensors"])._forward(X)
        assert all(np.array_equal(a[k], b[k]) for k in a), name
        assert onnx_parity(zm, it["derived"]["model.onnx"], X) < 1e-5, name
        ex = json.loads(Path(it["derived"]["example.json"]).read_text())
        assert ex["trust_decision"] == "APPROVED" and ex["in_range"]
        assert ex["query_verdict"] in (None, "pass")


def test_how_to_use_code_runs_and_prints_the_card_numbers(model_cards):
    # heatsink_natural_mlp has no query checks (the in_range branch of the snippet); the rover model has them
    for it in (model_cards["heatsink_natural_mlp"], model_cards["rover_contact_grc1_viper"]):
        code = next(b for b in hub_cards.code_blocks(it["readme"]) if "pp.zoo.load" in b)
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, cwd=ROOT)
        assert run.returncode == 0, run.stderr[-2000:]
        lines = run.stdout.splitlines()
        comments = [ln[2:] for ln in code.splitlines() if ln.startswith("# ")]
        assert ast.literal_eval(lines[0]) == ast.literal_eval(comments[0])
        assert lines[1:] == comments[1:]


# ============================================================================================== revisions
def test_republished_revision_falls_back_to_the_version_tag(monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import RevisionNotFoundError

    from pinneapple_hub import zoo
    monkeypatch.setenv("PINNEAPPLE_ZOO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    monkeypatch.delenv("PINNEAPPLE_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    calls = []

    class _R:
        status_code = 404
        headers: dict = {}
        request = None

    def fake(repo_id, filename, *, revision, repo_type, local_dir):
        calls.append(revision)
        if revision.endswith(".post1"):                          # the new tag does not exist yet
            raise RevisionNotFoundError("no such revision", response=_R())
        e = next(e for e in art.manifest()["models"].values() if e["hf_repo"] == repo_id)
        out = Path(local_dir) / filename
        out.write_bytes((art.ASSETS / e["files"][filename]["bundled"]).read_bytes())
        return str(out)

    monkeypatch.setattr(hub, "hf_hub_download", fake)
    m = zoo.load("heatsink_forced_mlp")
    assert m.name == "heatsink_forced_mlp" and "v1.0.0.post1" in calls and "v1.0.0" in calls
    assert art.revisions(art.entry("models", "heatsink_forced_mlp")) == ["v1.0.0.post1", "v1.0.0"]
    assert art.revisions({"version": "2.0.0", "revision": "v2.0.0"}) == ["v2.0.0"]


# ============================================================================================== publish checks
def test_verify_viewer_reads_is_valid_splits_and_first_rows():
    from pinneapple_hub.zoo import publish
    e = art.entry("datasets", "single_wheel_tests")
    answers = {"is-valid": [(200, {"preview": False, "viewer": False}),
                            (200, {"preview": True, "viewer": True, "search": True, "filter": True})],
               "splits": [(200, {"splits": [{"config": "default", "split": "all"}]})],
               "first-rows": [(500, {"error": "not ready"}),
                              (200, {"features": [{}] * 12, "rows": [{"truncated_cells": []}] * 27})]}

    def get(url, timeout=60):
        return answers[url.split("datasets-server.huggingface.co/")[1].split("?")[0]].pop(0)

    assert publish.verify_viewer(["single_wheel_tests"], interval=0, get=get, sleep=lambda s: None) == 0
    st = {"is-valid": [(200, {"preview": False, "viewer": False, "error": "no data files"})] * 3}
    assert publish.verify_viewer(["single_wheel_tests"], timeout=0, interval=0,
                                 get=lambda u, timeout=60: st["is-valid"][0], sleep=lambda s: None) == 1
    assert e["hf_repo"] == "PINNeAPPle-Labs/single-wheel-tests"


def _fake_fetch(pages):
    def fetch(url, timeout=60):
        for suffix, (code, body) in pages.items():
            if url.endswith(suffix):
                return code, body
        return 404, ""
    return fetch


def test_verify_space_waits_for_running_and_checks_the_page_and_data():
    from types import SimpleNamespace

    from pinneapple_hub.zoo import publish
    pages = {"/index.html": (200, "<title>PINNeAPPle zoo demo</title>"),
             "/index.json": (200, json.dumps({"models": [{"name": "m1"}]})),
             "/models/m1.sweeps.json": (200, json.dumps({"sweeps": {"x": {}}}))}
    stages = iter(["BUILDING", "RUNNING"])
    run = lambda: SimpleNamespace(stage=next(stages), hardware=None,                       # noqa: E731
                                  raw={"domains": [{"domain": "org-zoo-demo.static.hf.space"}]})
    assert publish.verify_space(interval=0, sleep=lambda s: None, runtime=run, fetch=_fake_fetch(pages)) == 0
    missing = {k: v for k, v in pages.items() if "sweeps" not in k}
    assert publish.verify_space(interval=0, sleep=lambda s: None, fetch=_fake_fetch(missing),
                                runtime=lambda: SimpleNamespace(stage="RUNNING", hardware=None, raw={})) == 1
    assert publish.verify_space(interval=0, sleep=lambda s: None, fetch=_fake_fetch(pages),
                                runtime=lambda: SimpleNamespace(stage="CONFIG_ERROR", hardware=None)) == 1


# ============================================================================================== demo Space
def test_demo_sweep_marks_extrapolation_and_the_heatsink_natural_checks_follow_in_range():
    from pinneapple_hub import zoo
    from pinneapple_hub.zoo import checks, demo
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("airfoil_cst_rans_mlp", "heatsink_natural_mlp"):
            c = demo.controls(name)
            r = demo.sweep(name, {d["name"]: d["value"] for d in c}, vary=c[-1]["name"], n=31)
            inside = (r["x"] >= r["range"][0]) & (r["x"] <= r["range"][1])
            assert {r["status"][i] for i in np.nonzero(~inside)[0]} == {"fail"}       # beyond the range: fail
            assert r["point_checks"]["verdict"]["status"] == "pass"
        zm = zoo.load("heatsink_natural_mlp")                     # query checks now cover the natural heat sink
        X, _ = zm.reference("test")
        X = np.concatenate([X, X * np.r_[np.ones(10), 3.0]])     # the power tripled: many leave the range
        res = checks.query_checks(zm, X)
        rng_fail = np.array([any(c["status"] == "fail" and c["rule"] == "PC601" for c in r["checks"]) for r in res])
        assert np.array_equal(rng_fail, ~zm.in_range(X)) and rng_fail.any() and not rng_fail.all()


def test_demo_precompute_has_bands_zones_and_the_trust_decision():
    from pinneapple_hub.zoo import demo
    assert demo.bands(["Cl", "Cl_std", "Cd", "Cd_std", "LD", "LD_std"]) == {
        "Cl": {"std": "Cl_std"}, "Cd": {"std": "Cd_std"}, "LD": {"std": "LD_std"}}
    assert demo.bands(["r_base_k_w", "r_base_upper95_k_w"]) == {"r_base_k_w": {"upper": "r_base_upper95_k_w"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = demo.precompute("rover_contact_grc1_viper", n=21)
    assert [d["name"] for d in p["inputs"]] == ["slip", "sinkage_m"] and p["trust"]["decision"] == "APPROVED"
    assert {o["name"] for o in p["outputs"]} == {"Fx_N", "Fz_N", "My_Nm"}
    assert p["default"]["checks"]["verdict"]["status"] == "pass"
    for sw in p["sweeps"].values():
        assert len(sw["x"]) == 21 and set(sw["pred"]) == set(p["all_outputs"])
        assert set(sw["status"]) >= {"pass", "fail"}                            # the zones the page shades
        assert all(len(v) == 21 for v in sw["pred"].values())


@pytest.fixture(scope="module")
def space(tmp_path_factory):
    pytest.importorskip("onnx")
    from pinneapple_hub.zoo import publish
    out = tmp_path_factory.mktemp("space")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        files = publish.build_space(out, ROOT / "apps" / "zoo_space", ROOT / "apps" / "playground" / "site")
    return out, files


def test_static_space_has_its_page_data_and_runtime(space):
    out, files = space
    names = sorted(art.manifest()["models"])
    assert {"index.html", "app.js", "space.css", "README.md", "index.json", "lib/onnx.js", "lib/zoo.js",
            "js/charts.js", "js/ui.js", "css/app.css", "icon.svg"} <= set(files)
    for n in names:
        assert {f"models/{n}.onnx", f"models/{n}.json", f"models/{n}.sweeps.json"} <= set(files)
    assert [m["name"] for m in json.loads((out / "index.json").read_text())["models"]] == names
    meta = hub_cards.parse_front_matter((out / "README.md").read_text())
    assert meta["sdk"] == "static" and meta["app_file"] == "index.html" and "sdk_version" not in meta
    assert sorted(meta["models"]) == sorted(e["hf_repo"] for e in art.manifest()["models"].values())
    html = (out / "index.html").read_text()
    assert "PINNeAPPle zoo demo" in html and "<script" in html and "cdn" not in html.lower()  # no external code
    imports = re.findall(r'from "([^"]+)"', (out / "app.js").read_text())
    assert imports and all((out / i).is_file() for i in imports)              # every module the page loads ships
    assert sum((out / f).stat().st_size for f in files) < 5e6


def test_space_runs_the_same_numbers_and_checks_in_the_browser_runtime(space):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    out, _ = space
    run = subprocess.run([node, "--test", str(ROOT / "tests" / "js" / "zoo_space.test.mjs")], capture_output=True,
                         text=True, timeout=600, env={**os.environ, "SPACE_DIR": str(out)})
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]
    assert f"# pass {len(art.manifest()['models'])}" in run.stdout


# ============================================================================================== publish end to end
class _FakeApi:
    """Records what publish() sends to the Hub."""

    def __init__(self, token=None, remote=()):
        self.commits, self.tags, self.repos, self.remote = [], [], [], list(remote)

    def create_repo(self, repo, repo_type=None, exist_ok=False, **kw):
        self.repos.append((repo, repo_type))

    def list_repo_files(self, repo, repo_type=None):
        return self.remote

    def create_commit(self, repo, operations, commit_message, repo_type=None):
        self.commits.append((repo, repo_type, operations, commit_message))
        from types import SimpleNamespace
        return SimpleNamespace(oid=f"oid{len(self.commits)}")

    def list_repo_refs(self, repo, repo_type=None):
        from types import SimpleNamespace
        return SimpleNamespace(tags=[SimpleNamespace(name="v1.0.0")])

    def create_tag(self, repo, tag, repo_type=None, revision=None):
        self.tags.append((repo, tag, revision))


def test_publish_commits_parquet_card_and_tag_and_converts_hub_copies(tmp_path, monkeypatch, capsys):
    hub = pytest.importorskip("huggingface_hub")
    from pinneapple_hub.zoo import publish
    served = tmp_path / "hub"
    served.mkdir()
    data = _synthetic("darcy_2d_li2021", n=2)
    for split, cols in data.items():
        np.savez(served / f"{split}.npz", **{f"{split}__{c}": a for c, a in cols.items()})
    man = json.loads(art.MANIFEST.read_text())
    toy = {**man["datasets"]["darcy_2d_li2021"], "hf_repo": "PINNeAPPle-Labs/toy-darcy",
           "splits": {"train": {"n": 2, "description": ""}, "test": {"n": 2, "description": ""}},
           "files": {f"{s}.npz": {"sha256": art.sha256_file(served / f"{s}.npz"), "size": 1} for s in data}}
    man["datasets"] = {"toy_darcy": toy, "single_wheel_tests": man["datasets"]["single_wheel_tests"]}
    man["models"] = {}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(man))
    monkeypatch.setattr(art, "MANIFEST", mpath)
    monkeypatch.setenv("PINNEAPPLE_ZOO_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("PINNEAPPLE_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    revs = []

    def fake_download(repo_id, filename, *, revision, repo_type, local_dir):
        revs.append(revision)
        out = Path(local_dir) / filename
        out.write_bytes((served / filename).read_bytes())
        return str(out)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    api = _FakeApi(remote=["README.md", "train.npz", "data/default/train-00000.parquet", "data/old/x.parquet"])
    monkeypatch.setattr(hub, "HfApi", lambda token=None: api)
    items = publish.plan(work=tmp_path / "work", hub_fallback=True)
    assert {it["name"]: it["source"] for it in items} == {"toy_darcy": "hub", "single_wheel_tests": "local"}
    assert publish.publish(items, "token", work=tmp_path / "work") == []
    by_repo = {c[0]: c for c in api.commits}
    ops = by_repo["PINNeAPPle-Labs/toy-darcy"][2]
    added = sorted(o.path_in_repo for o in ops if type(o).__name__ == "CommitOperationAdd")
    deleted = sorted(o.path_in_repo for o in ops if type(o).__name__ == "CommitOperationDelete")
    assert added == ["README.md", "data/default/test-00000.parquet", "data/default/train-00000.parquet"]
    assert deleted == ["data/old/x.parquet"]                     # stale shard of an earlier layout; npz untouched
    readme = next(o for o in ops if o.path_in_repo == "README.md").path_or_fileobj.decode()
    assert hub_cards.parse_front_matter(readme)["configs"][0]["data_files"][0]["path"] == "data/default/train-*.parquet"
    wheel_ops = sorted(o.path_in_repo for o in by_repo["PINNeAPPle-Labs/single-wheel-tests"][2]
                       if type(o).__name__ == "CommitOperationAdd")
    assert wheel_ops == ["README.md", "data.npz", "data/default/all-00000.parquet"]
    assert ("PINNeAPPle-Labs/toy-darcy", toy["revision"], "oid1") in api.tags
    assert set(revs) == {toy["revision"], f"v{toy['version']}"} or set(revs) == {toy["revision"]}
    assert not (tmp_path / "work" / "toy_darcy").exists()       # downloads and derived files cleaned up
    api.create_commit = None                                     # a failing item is reported, the others published
    assert publish.publish(publish.plan(work=tmp_path / "w2"), "token", work=tmp_path / "w2") == ["single_wheel_tests"]
    assert "was not published" in capsys.readouterr().out


def test_verify_model_checks_what_the_card_promises(model_cards, tmp_path, monkeypatch, capsys):
    from pinneapple_hub.zoo import publish
    it = model_cards["airfoil_cst_rans_mlp"]
    local = {**{k: Path(v) for k, v in it["derived"].items()}, "README.md": tmp_path / "README.md"}
    local["README.md"].write_text(it["readme"])
    monkeypatch.setattr(publish, "_download", lambda repo, fn, rev, dest, repo_type="model": local[fn])
    assert publish.verify_model("airfoil_cst_rans_mlp", tmp_path) == []
    assert "airfoil_cst_rans_mlp" in capsys.readouterr().out
    bad = json.loads(local["example.json"].read_text())
    bad["outputs"]["Cl"] += 0.01
    (tmp_path / "example.json").write_text(json.dumps(bad))
    local["example.json"] = tmp_path / "example.json"
    local["README.md"].write_text(it["readme"].replace("inference: false", "inference: true"))
    problems = publish.verify_model("airfoil_cst_rans_mlp", tmp_path)
    assert any("example not reproduced" in p for p in problems) and any("front matter" in p for p in problems)


def test_publish_space_uploads_a_static_space_and_explains_a_refusal(tmp_path, monkeypatch, capsys):
    hub = pytest.importorskip("huggingface_hub")
    from types import SimpleNamespace

    from pinneapple_hub.zoo import publish

    def fake_build(out, app_dir, playground_dir, models=None):
        (out / "models").mkdir(parents=True)
        for f in ("index.html", "README.md", "models/m.json"):
            (out / f).write_text("x")
        return ["README.md", "index.html", "models/m.json"]

    monkeypatch.setattr(publish, "build_space", fake_build)
    api = _FakeApi(remote=[".gitattributes", "app.py", "requirements.txt", "wheels/old.whl", "index.html"])
    created = []
    api.create_repo = lambda repo, repo_type=None, **kw: created.append((repo, repo_type, kw.get("space_sdk")))
    monkeypatch.setattr(hub, "HfApi", lambda token=None: api)
    files = publish.publish_space("token", tmp_path, tmp_path, work=tmp_path / "w", from_hub=False)
    assert files == ["README.md", "index.html", "models/m.json"]
    assert created == [(hub_cards.SPACE_REPO, "space", "static")]
    ops = api.commits[0][2]
    assert sorted(o.path_in_repo for o in ops if type(o).__name__ == "CommitOperationDelete") == [
        "app.py", "requirements.txt", "wheels/old.whl"]                       # the old Gradio Space's files go
    assert "static" in capsys.readouterr().out

    def refuse(repo, repo_type=None, **kw):
        err = RuntimeError("402 Payment Required")
        err.response = SimpleNamespace(status_code=402)
        raise err

    api.create_repo = refuse
    with pytest.raises(SystemExit, match="HTTP 402.*space=false"):
        publish.publish_space("token", tmp_path, tmp_path, work=tmp_path / "w2", from_hub=False)
