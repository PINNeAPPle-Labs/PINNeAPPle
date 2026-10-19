"""Model zoo and dataset hub: round trip load -> predict reproduces the stored metrics, checksums, cache, offline and
not-published behaviour, the fine-tune guard, validation and the publish script."""
from __future__ import annotations

import json
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pinneapple as pp  # noqa: E402
from pinneapple_hub import zoo  # noqa: E402
from pinneapple_hub.zoo import _artifacts as art  # noqa: E402

MODELS = ["airfoil_cst_rans_mlp", "heatsink_forced_mlp", "heatsink_natural_mlp", "rover_contact_grc1_viper"]
BUNDLED_DATASETS = ["airfoil_rans_cst_re4e6", "heatsink_forced_convection", "heatsink_natural_convection",
                    "rover_contact_grc1_viper", "single_wheel_tests"]
HUB_ONLY_DATASETS = ["burgers_1d_li2021", "darcy_2d_li2021", "navier_stokes_2d_li2021_nu1e-3",   # downloaded on first
                     "thermal_channel_cfd", "airfoil_rans_cst_re4e6_fields", "cylinder_wake_lbm"]  # use, never bundled
BLACKHOLE_DATASETS = ["blackhole_riaf_density_256"]                 # Hub-only too, one file per split
DATASETS = BUNDLED_DATASETS + HUB_ONLY_DATASETS + BLACKHOLE_DATASETS


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_ZOO_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("PINNEAPPLE_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return tmp_path / "cache"


# ------------------------------------------------------------------ catalogue
def test_lazy_shortcuts_and_listing():
    assert pp.zoo is zoo and pp.datasets is zoo.datasets
    assert zoo.list() == MODELS
    assert pp.datasets.list() == sorted(DATASETS)
    assert zoo.list(task="terramechanics") == ["rover_contact_grc1_viper"]


def test_manifest_is_small_hashed_and_versioned():
    man = art.manifest()
    for kind in ("models", "datasets"):
        for name, e in man[kind].items():
            # the tag of the version, or a republication of the same files (v1.0.0.post1: new card, Parquet copies)
            assert e["hf_repo"].startswith("PINNeAPPle-Labs/") and re.fullmatch(
                rf"v{re.escape(e['version'])}(\.post\d+)?", e["revision"])
            for fn, spec in e["files"].items():
                assert len(spec["sha256"]) == 64 and spec["size"] > 0, (name, fn)
                if name in HUB_ONLY_DATASETS + BLACKHOLE_DATASETS:     # downloaded on first use, never bundled
                    assert "bundled" not in spec, (name, fn)
                    continue
                p = art.ASSETS / spec["bundled"]
                assert p.stat().st_size == spec["size"] < 2_000_000, (name, fn)
                assert art.sha256_file(p) == spec["sha256"], (name, fn)


@pytest.mark.parametrize("name", MODELS)
def test_model_card_is_complete(name):
    card = zoo.info(name)
    assert card.validate() == []
    assert card.task and card.version and card.license and card.citation and card.reference_source
    assert all(d.get("unit") for d in card.inputs + card.outputs)
    assert card.validity_range and card.data_lineage.get("dataset") in DATASETS
    assert card.trust_decision == "APPROVED"
    md = card.to_markdown()
    assert "Validity range" in md and "Inputs" in md and "Training data" in md


@pytest.mark.parametrize("name", BUNDLED_DATASETS)
def test_dataset_schema_and_splits(name):
    info = pp.datasets.info(name)
    splits = pp.datasets.load(name)
    assert set(splits) == set(info.splits)
    for s, d in splits.items():
        assert d.n_rows == info.splits[s]["n"] > 0
        assert set(d.columns) <= set(info.columns)
    assert info.license and info.source and info.verification
    assert "## Splits" in info.to_markdown()


# ------------------------------------------------------------------ round trip
@pytest.mark.parametrize("name", MODELS)
def test_load_predict_reproduces_card_metrics(name):
    res = zoo.evaluate(name)
    assert res["matches_card"], res                     # within the card tolerance (float32 differs across CPUs)
    assert res["max_rel_diff_vs_card"] <= res["tolerance"] <= 1e-3


def test_airfoil_predicts_known_case_and_compares_with_xfoil():
    m = zoo.load("airfoil_cst_rans_mlp")
    xf = pp.datasets.load("airfoil_rans_cst_re4e6", split="xfoil_naca2412")
    out = m.predict({k: xf[k] for k in m.input_names})
    assert np.all(np.abs(out["Cl"] - xf["Cl"]) < 0.03)                 # independent solver, ~2 % of Cl
    assert np.all(np.abs(out["Cd"] / xf["Cd"] - 1) < 0.10)
    assert set(out) >= {"Cl", "Cd", "Cm", "Cl_std"}


def test_named_inputs_and_out_of_range_warning():
    m = zoo.load("rover_contact_grc1_viper")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = m.predict(slip=0.2, sinkage_m=0.02)
    with pytest.warns(zoo.OutOfRangeWarning):
        b = m.predict(np.array([[0.2, 0.02], [1.5, 0.02]]))
    assert np.allclose(a["Fx_N"], b["Fx_N"][0])
    assert m.in_range([[0.2, 0.02], [1.5, 0.02]]).tolist() == [True, False]
    with pytest.raises(KeyError):
        m.predict(slip=0.2)


def test_evaluate_on_given_cases_reports_out_of_range_fraction():
    m = zoo.load("heatsink_forced_mlp")
    d = pp.datasets.load("heatsink_forced_convection", split="test")
    r = zoo.evaluate(m, d)
    assert r["fraction_out_of_range"] == 0.0
    assert r["metrics"]["R_mean_abs_pct"] == pytest.approx(zoo.info("heatsink_forced_mlp")
                                                            .validation_metrics["test/R_mean_abs_pct"], rel=1e-4)


# ------------------------------------------------------------------ validation
def test_validate_approves_and_flags_out_of_range_queries():
    rep = zoo.validate("rover_contact_grc1_viper")
    assert rep.decisao == "APPROVED"
    with pytest.warns(zoo.OutOfRangeWarning):
        rep2 = zoo.validate("rover_contact_grc1_viper", inputs=[[0.1, 0.01], [0.1, 0.5]])
    assert rep2.details["fraction_out_of_range"] == 0.5
    assert rep2.decisao == "REVIEW"


# ------------------------------------------------------------------ checksums, cache, offline, not published
def _copy_assets(tmp_path, monkeypatch):
    dst = tmp_path / "assets"
    shutil.copytree(art.ASSETS, dst)
    monkeypatch.setattr(art, "ASSETS", dst)
    return dst


def test_bundled_checksum_mismatch_is_detected(tmp_path, monkeypatch, isolated_cache):
    dst = _copy_assets(tmp_path, monkeypatch)
    p = dst / "models" / "rover_contact_grc1_viper" / "weights.pt"
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    p.write_bytes(bytes(b))
    with pytest.raises(zoo.ChecksumMismatchError):
        zoo.load("rover_contact_grc1_viper")


def test_offline_without_bundle_or_cache_says_so(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    monkeypatch.setenv("PINNEAPPLE_OFFLINE", "1")
    with pytest.raises(zoo.ArtifactUnavailableError, match="offline"):
        pp.datasets.load("single_wheel_tests", split="all")


def _fake_hub(monkeypatch, behaviour):
    """Replace huggingface_hub.hf_hub_download: 'serve' copies the bundled file, 'corrupt' serves a damaged copy,
    'missing' raises RepositoryNotFoundError."""
    hub = pytest.importorskip("huggingface_hub")
    calls = []

    def fake(repo_id, filename, *, revision, repo_type, local_dir):
        calls.append((repo_id, filename, revision, repo_type))
        if behaviour == "missing":
            err = RuntimeError("404 Client Error: Repository Not Found")
            err.response = _Resp()
            raise err
        kind = "datasets" if repo_type == "dataset" else "models"
        name = next(n for n, e in art.manifest()[kind].items() if e["hf_repo"] == repo_id)
        src = Path(__file__).resolve().parents[1] / "pinneapple_hub/zoo/assets" / art.manifest()[kind][name][
            "files"][filename]["bundled"]
        out = Path(local_dir) / filename
        data = bytearray(src.read_bytes())
        if behaviour == "corrupt":
            data[-1] ^= 0xFF
        out.write_bytes(bytes(data))
        return str(out)

    monkeypatch.setattr(hub, "hf_hub_download", fake)
    return calls


class _Resp:
    status_code = 404


def test_not_published_via_huggingface_hub(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    _fake_hub(monkeypatch, "missing")
    with pytest.raises(zoo.ArtifactNotPublishedError, match="publish_to_hub"):
        zoo.load("airfoil_cst_rans_mlp")


def test_lazy_download_then_cache_then_offline(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    calls = _fake_hub(monkeypatch, "serve")
    assert pp.datasets.status("single_wheel_tests") == {"data.npz": "remote"}
    d = pp.datasets.load("single_wheel_tests", split="all")
    assert d.n_rows > 20 and len(calls) == 1
    assert calls[0] == ("PINNeAPPle-Labs/single-wheel-tests", "data.npz",
                        art.entry("datasets", "single_wheel_tests")["revision"], "dataset")
    assert pp.datasets.status("single_wheel_tests") == {"data.npz": "cached"}
    monkeypatch.setenv("PINNEAPPLE_OFFLINE", "1")
    pp.datasets.load("single_wheel_tests", split="all")                     # served from the cache
    assert len(calls) == 1


def test_downloaded_checksum_mismatch_is_rejected_and_not_cached(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    _fake_hub(monkeypatch, "corrupt")
    with pytest.raises(zoo.ChecksumMismatchError):
        pp.datasets.load("single_wheel_tests", split="all")
    assert not list(isolated_cache.rglob("data.npz"))


def test_corrupt_cache_entry_offline_is_an_error(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    _fake_hub(monkeypatch, "serve")
    pp.datasets.load("single_wheel_tests", split="all")
    cached = next(isolated_cache.rglob("data.npz"))
    cached.write_bytes(cached.read_bytes()[:-10])
    monkeypatch.setenv("PINNEAPPLE_OFFLINE", "1")
    with pytest.raises(zoo.ChecksumMismatchError):
        pp.datasets.load("single_wheel_tests", split="all")
    monkeypatch.delenv("PINNEAPPLE_OFFLINE")
    assert pp.datasets.load("single_wheel_tests", split="all").n_rows > 20     # re-fetched


def test_not_yet_published_gives_a_clear_message(monkeypatch, isolated_cache):
    monkeypatch.setenv("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "1")
    import urllib.error
    import urllib.request

    def no_hub(*a, **k):
        raise urllib.error.HTTPError(a[0] if a else "", 404, "Not Found", {}, None)

    import builtins
    real_import = builtins.__import__

    def without_hf(name, *args, **kwargs):                       # exercise the plain-HTTPS fallback as well
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_hf)
    monkeypatch.setattr(urllib.request, "urlopen", no_hub)
    with pytest.raises(zoo.ArtifactNotPublishedError, match="not published on the Hugging Face Hub yet"):
        pp.datasets.load("single_wheel_tests", split="all")


@pytest.mark.parametrize("name", HUB_ONLY_DATASETS)
def test_hub_only_dataset_is_documented(name):
    info = pp.datasets.info(name)
    assert info.license == "cc-by-4.0" and "10.5281/zenodo.23343062" in info.citation
    if name.endswith(("li2021", "li2021_nu1e-3")):
        assert set(info.splits) == {"train", "test"} and info.splits["train"]["n"] == 1000
        assert info.splits["test"]["n"] == 200
        assert "Li" in info.citation and "rel. L2" in info.verification
    elif name == "thermal_channel_cfd":                            # heated channel: solver verification and GCI
        assert sum(v["n"] for v in info.splits.values()) == 82 and info.splits["train"]["n"] == 48
        assert "Nusselt" in info.verification and "GCI" in info.verification
    elif name == "cylinder_wake_lbm":                              # cylinder wake: resolution study
        assert {s: v["n"] for s, v in info.splits.items()} == {"train": 480, "test": 240}
        assert "Poiseuille" in info.verification and "D = 40" in info.verification
    else:                                                          # airfoil fields: the 436-run campaign
        assert {s: v["n"] for s, v in info.splits.items()} == {"train": 340, "test": 60, "naca_reference": 36}
        assert "reproduce" in info.verification and "GCI" in info.verification
    assert {f for s in info.splits for f in pp.datasets.split_files(name, s)} == set(info.files)
    assert all(d["dtype"] and d["description"] for d in info.columns.values())
    assert all(not f.get("bundled") for f in info.files.values())


def test_blackhole_dataset_is_documented_with_its_convergence_study():
    info = pp.datasets.info("blackhole_riaf_density_256")
    assert {"train", "val", "test", "ood_alpha0.3", "convergence"} <= set(info.splits)
    assert info.splits["test"]["n"] == info.splits["val"]["n"] == 401 and info.splits["train"]["n"] == 4 * 401
    assert "Resolution study" in info.verification and "Not converged" in info.verification
    assert "trajectory" in info.split_rule
    assert {f for s in info.splits for f in pp.datasets.split_files("blackhole_riaf_density_256", s)} == set(info.files)
    assert all(not f.get("bundled") and f["size"] < 100_000_000 for f in info.files.values())


def test_hub_only_sharded_dataset_downloads_only_the_split(tmp_path, monkeypatch, isolated_cache):
    """A Hub-only dataset stored as <split>.npz / <split>-NN.npz: loading a split fetches only its files, shards are
    concatenated in order, and every file is checked against the manifest."""
    hub = pytest.importorskip("huggingface_hub")
    served = tmp_path / "hub"
    rows = {"train": [np.arange(0, 3), np.arange(3, 5)], "test": [np.arange(5, 7)]}
    files = {}
    for split, parts in rows.items():
        for j, r in enumerate(parts):
            fn = f"{split}-{j:02d}.npz" if len(parts) > 1 else f"{split}.npz"
            served.mkdir(exist_ok=True)
            np.savez(served / fn, **{f"{split}__index": r, f"{split}__u": np.ones((len(r), 4, 4)) * r[:, None, None]})
            files[fn] = {"sha256": art.sha256_file(served / fn), "size": (served / fn).stat().st_size}
    man = json.loads(art.MANIFEST.read_text())
    man["datasets"]["toy_sharded"] = {
        "version": "1.0.0", "revision": "v1.0.0", "hf_repo": "PINNeAPPle-Labs/toy-sharded", "repo_type": "dataset",
        "title": "toy", "description": "toy", "license": "cc-by-4.0", "source": "test", "verification": "test",
        "citation": "", "tags": [], "splits": {"train": {"n": 5, "description": ""}, "test": {"n": 2, "description": ""}},
        "columns": {"index": {"unit": "-", "description": "row", "dtype": "int64"},
                    "u": {"unit": "-", "description": "field", "dtype": "float64[4, 4]"}},
        "files": files}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(man))
    monkeypatch.setattr(art, "MANIFEST", mpath)
    calls = []

    def fake(repo_id, filename, *, revision, repo_type, local_dir):
        calls.append(filename)
        out = Path(local_dir) / filename
        out.write_bytes((served / filename).read_bytes())
        return str(out)

    monkeypatch.setattr(hub, "hf_hub_download", fake)
    test = pp.datasets.load("toy_sharded", split="test")
    assert calls == ["test.npz"] and list(test["index"]) == [5, 6]
    train = pp.datasets.load("toy_sharded", split="train")
    assert calls[1:] == ["train-00.npz", "train-01.npz"] and list(train["index"]) == [0, 1, 2, 3, 4]
    assert train["u"].shape == (5, 4, 4) and train["u"][4, 0, 0] == 4
    assert pp.datasets.status("toy_sharded") == dict.fromkeys(files, "cached")
    with pytest.raises(KeyError, match="no split"):
        pp.datasets.load("toy_sharded", split="val")
    (served / "test.npz").write_bytes((served / "train-01.npz").read_bytes())    # tampered upload
    shutil.rmtree(isolated_cache)
    with pytest.raises(zoo.ChecksumMismatchError):
        pp.datasets.load("toy_sharded", split="test")


# ------------------------------------------------------------------ fine-tuning guard
def test_finetune_guard_refuses_a_worse_model(tmp_path):
    d = pp.datasets.load("rover_contact_grc1_viper", split="test")
    X, Y = d.xy(["slip", "sinkage_m"], ["Fx_N", "Fz_N", "My_Nm"])
    res = zoo.finetune("rover_contact_grc1_viper", (X[:300], 1.3 * Y[:300]), epochs=10, lr=1e-3, replay=False)
    assert not res.accepted and res.after["test/Fx_rmse_N"] > res.before["test/Fx_rmse_N"]
    with pytest.raises(zoo.FinetuneRejectedError):
        res.save(tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    with pytest.raises(zoo.FinetuneRejectedError):
        zoo.finetune("rover_contact_grc1_viper", (X[:300], 1.3 * Y[:300]), epochs=10, lr=1e-3, replay=False,
                     save_to=tmp_path / "refused2")
    assert not (tmp_path / "refused2").exists()
    p = res.save(tmp_path / "forced", force=True)
    card = json.loads((p / "model_card.json").read_text())
    assert card["data_lineage"]["guard"]["accepted"] is False
    # the original is untouched
    assert zoo.evaluate("rover_contact_grc1_viper")["matches_card"]


def test_finetune_accepted_round_trip_and_registry(tmp_path):
    from pinneapple_registry import ArtifactRegistry
    d = pp.datasets.load("heatsink_forced_convection", split="val")
    res = zoo.finetune("heatsink_forced_mlp", d, epochs=1, lr=0.0, save_to=tmp_path / "ft",
                       registry=ArtifactRegistry(str(tmp_path / "reg")))
    assert res.accepted and res.saved_to
    m2 = zoo.load_dir(tmp_path / "ft")
    assert m2.card.version.startswith("1.0.0+ft.")
    assert zoo.evaluate(m2, "heatsink_forced_convection")["metrics"]["R_mean_abs_pct"] == pytest.approx(
        res.before["test/R_mean_abs_pct"], rel=1e-6)
    reg = ArtifactRegistry(str(tmp_path / "reg"))
    assert reg.models.latest("heatsink_forced_mlp-finetuned")


def test_finetune_replay_masks_nan_targets():
    d = pp.datasets.load("airfoil_rans_cst_re4e6", split="naca_reference")
    res = zoo.finetune("airfoil_cst_rans_mlp", d, epochs=1, lr=0.0, batch=64)
    assert res.accepted and np.isfinite(res.history).all()


# ------------------------------------------------------------------ benchmark and publishing
def test_benchmark_via_pp_benchmark(tmp_path):
    pytest.importorskip("pinneapple_analysis.benchmark")
    res = zoo.benchmark("rover_contact_grc1_viper", budget_s=2, out_dir=tmp_path / "b", log=lambda *a: None)
    md = res.to_markdown()
    assert "Zoo model (pre-trained)" in md and "trained from scratch" in md


def test_benchmark_without_pp_benchmark_is_explicit(monkeypatch):
    import importlib.util
    if importlib.util.find_spec("pinneapple_analysis.benchmark") is not None:
        pytest.skip("pp.benchmark is installed")
    with pytest.raises(ImportError, match="pp.benchmark"):
        zoo.benchmark("rover_contact_grc1_viper", budget_s=1)


def test_publish_script_dry_run_and_no_token(monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("publish_to_hub", Path(__file__).resolve().parents[1] /
                                                  "scripts/zoo/publish_to_hub.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    for name in HUB_ONLY_DATASETS:                                  # no staging directory: skipped, not an error
        assert f"skip dataset {name}" in out
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert mod.main([]) == 0
    assert "HF_TOKEN is not set" in capsys.readouterr().out
    items = {it["name"]: it for it in mod.plan()}
    assert "10.5281/zenodo.23343062" in items["single_wheel_tests"]["readme"]


def test_publish_script_uses_the_staging_directory(tmp_path, monkeypatch, capsys):
    """A Hub-only item is uploaded only from the staging directory, whose files must match the manifest."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("publish_to_hub", Path(__file__).resolve().parents[1] /
                                                  "scripts/zoo/publish_to_hub.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    staged = tmp_path / "staging" / "toy_hub_only"
    staged.mkdir(parents=True)
    np.savez(staged / "test.npz", test__x=np.arange(3))
    man = json.loads(art.MANIFEST.read_text())
    man["datasets"]["toy_hub_only"] = {
        **man["datasets"]["single_wheel_tests"], "hf_repo": "PINNeAPPle-Labs/toy-hub-only",
        "splits": {"test": {"n": 3, "description": ""}}, "columns": {},
        "files": {"test.npz": {"sha256": art.sha256_file(staged / "test.npz"), "size": 1}}}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(man))
    monkeypatch.setattr(art, "MANIFEST", mpath)
    assert "toy_hub_only" not in {it["name"] for it in mod.plan()}            # no staging: skipped
    assert "skip dataset toy_hub_only" in capsys.readouterr().out
    items = {it["name"]: it for it in mod.plan(tmp_path / "staging")}
    assert items["toy_hub_only"]["hub_only"] and items["toy_hub_only"]["files"]["test.npz"] == staged / "test.npz"
    (staged / "test.npz").write_bytes(b"not the data")
    with pytest.raises(SystemExit, match="SHA-256"):                # a staged file must match the manifest
        mod.plan(tmp_path / "staging")


def test_publish_keeps_the_files_of_a_dataset_named_like_a_model(tmp_path, monkeypatch):
    """rover_contact_grc1_viper is a model and a dataset: publishing the model first must not delete the dataset's
    Parquet copies before they are committed (one working directory per kind)."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("safetensors")
    import huggingface_hub

    from pinneapple_hub.zoo import publish
    committed = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *a, **k):
            pass

        def list_repo_files(self, *a, **k):
            return []

        def create_commit(self, repo, operations, **k):
            for op in operations:
                src = op.path_or_fileobj
                if isinstance(src, str):
                    assert Path(src).is_file(), (repo, op.path_in_repo)
                committed.append((repo, op.path_in_repo))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    items = publish.plan(work=tmp_path, only=["rover_contact_grc1_viper"])
    assert [it["kind"] for it in items] == ["models", "datasets"]
    assert publish.publish(items, "token", work=tmp_path, tags=False) == []
    assert any(p.endswith(".parquet") for _, p in committed)


def test_catalogue_page_is_up_to_date():
    """docs/guides/zoo_catalogue.md is generated by pp.zoo.catalogue(); regenerate it when the zoo changes:
    python -c "import pinneapple as pp; print(pp.zoo.catalogue())" > docs/guides/zoo_catalogue.md"""
    page = (Path(__file__).resolve().parents[1] / "docs/guides/zoo_catalogue.md").read_text()
    md = pp.zoo.catalogue()
    assert page.strip() == md.strip()
    for name in MODELS + DATASETS:
        assert f"`{name}`" in md

