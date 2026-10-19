"""Publish the zoo to the Hugging Face Hub and check what the Hub serves (``scripts/zoo/publish_to_hub.py`` is the
command line, ``.github/workflows/publish-zoo.yml`` runs it).

One repo per item under ``PINNeAPPle-Labs`` (model repos for models, dataset repos for datasets), tagged with the
item's ``revision``, plus one Space for the demo:

* **datasets**: the npz files of the manifest (the library's format, SHA-256 checked), the Parquet files derived from
  them (:mod:`pinneapple_hub.zoo.parquet`, what the dataset viewer and ``datasets.load_dataset`` read) and the card
  (:func:`pinneapple_hub.zoo.hub_cards.dataset_card`). A Hub-only dataset (npz too large to bundle) is converted from
  its staged files or, with ``hub_fallback``, from the copy already on the Hub (SHA-256 checked again);
* **models**: ``weights.pt``, ``config.json``, ``model_card.json`` of the manifest plus ``model.safetensors``,
  ``model.onnx`` and ``example.json`` derived from them, and the card (:func:`pinneapple_hub.zoo.hub_cards.model_card`);
* **the Space** ``PINNeAPPle-Labs/zoo-demo``: a static Space (no server) built by :func:`build_space`: the page of
  ``apps/zoo_space`` over the sweeps, bands and check verdicts :mod:`pinneapple_hub.zoo.demo` computes from the
  published models, and the models' ONNX graphs with their query checks for the visitor's own inputs.

Checks after an upload, all anonymous: :func:`verify_files` (every manifest file at the tag, SHA-256),
:func:`verify_viewer` (datasets-server ``is-valid``, ``splits`` and ``first-rows`` of every split),
:func:`verify_models` (load from the Hub, card metrics reproduced, the card's example reproduced by the PyTorch,
safetensors and ONNX copies, the card's "How to use" code run) and :func:`verify_space` (the Space runs and serves its
page and data).
"""
from __future__ import annotations

import ast
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import _artifacts as art
from . import hub_cards

__all__ = ["plan", "prepare", "publish", "publish_space", "verify_files", "verify_viewer", "verify_models",
           "verify_space", "build_space", "DATASETS_SERVER"]

DATASETS_SERVER = os.environ.get("HF_DATASETS_SERVER", "https://datasets-server.huggingface.co").rstrip("/")
DERIVED_MODEL_FILES = ("model.safetensors", "model.onnx", "example.json")


def _hub_url(it: dict) -> str:
    return f"https://huggingface.co/{'datasets/' if it['repo_type'] == 'dataset' else ''}{it['repo']}"


# ============================================================================================== plan / prepare
def _local(kind: str, name: str, fn: str, spec: dict, staging: Path | None) -> Path | None:
    """The local copy of a file: bundled, else ``staging/<name>/<fn>``; None when neither exists."""
    if spec.get("bundled"):
        p = art.ASSETS / spec["bundled"]
        if not p.exists():
            raise FileNotFoundError(f"{kind}/{name}/{fn}: bundled file {p} is missing; rebuild first")
        return p
    if staging is not None and (staging / name / fn).exists():
        return staging / name / fn
    return None


def _workdir(work: Path | None) -> Path:
    if work is not None:
        work.mkdir(parents=True, exist_ok=True)
        return work
    tmp = Path(tempfile.mkdtemp(prefix="zoo_publish_"))
    atexit.register(shutil.rmtree, tmp, True)
    return tmp


def _item_dir(work: Path, it: dict) -> Path:
    """Working directory of one item. Per kind, because a model and a dataset may share a name
    (``rover_contact_grc1_viper``): publishing the model must not delete the dataset's prepared files."""
    return work / it["kind"] / it["name"]


def plan(staging: Path | None = None, *, need_files: bool = True, hub_fallback: bool = False,
         work: Path | None = None, only: Sequence[str] = ()) -> list[dict]:
    """Every item to publish: ``kind``, ``name``, ``repo``, ``repo_type``, ``revision``, ``files`` (manifest files
    to upload: name -> local path), ``hub_only`` and ``source`` ("local", or "hub" for a Hub-only dataset converted
    from the Hub copy, prepared when it is published). Items with local files are prepared right away
    (:func:`prepare`: derived files and ``readme``). Local files must match the manifest's SHA-256."""
    items, work = [], (_workdir(work) if need_files else None)
    for kind in ("models", "datasets"):
        for name, e in art.manifest()[kind].items():
            if only and name not in only:
                continue
            hub_only = not all(spec.get("bundled") for spec in e["files"].values())
            it = {"kind": kind, "name": name, "repo": e["hf_repo"], "repo_type": e["repo_type"],
                  "revision": e["revision"], "hub_only": hub_only, "source": "local", "files": {}, "derived": {},
                  "readme": None}
            if not need_files:
                it["files"] = dict.fromkeys(e["files"])
                items.append(it)
                continue
            for fn, spec in e["files"].items():
                p = _local(kind, name, fn, spec, staging)
                if p is None:
                    it["files"] = None
                    break
                got = art.sha256_file(p)
                if got != spec["sha256"]:
                    raise SystemExit(f"{kind}/{name}/{fn}: SHA-256 {got} != manifest {spec['sha256']}; rebuild first")
                it["files"][fn] = p
            if it["files"] is None:
                if hub_fallback and kind == "datasets":
                    it.update(files={}, source="hub")
                    print(f"  {name}: Hub-only, not staged: Parquet and card derived from the copy on the Hub")
                    items.append(it)
                else:
                    print(f"  skip {kind[:-1]} {name}: Hub-only and its files are not in the staging directory"
                          + ("" if kind == "models" else " (--hub-fallback converts the copy on the Hub)"))
                continue
            prepare(it, _item_dir(work, it))
            items.append(it)
    return items


def prepare(it: dict, work: Path) -> dict:
    """Write the item's derived files into ``work`` and its README: Parquet copies and the dataset card, or the
    safetensors and ONNX copies, the example and the model card. Fills ``it["derived"]`` (path in repo -> local
    path) and ``it["readme"]``."""
    work.mkdir(parents=True, exist_ok=True)
    if it["kind"] == "datasets":
        _prepare_dataset(it, work)
    else:
        _prepare_model(it, work)
    return it


def _prepare_dataset(it: dict, work: Path) -> None:
    from . import parquet
    name = it["name"]
    e = art.entry("datasets", name)
    if it["source"] == "hub":                                   # fetched at the tag(s), SHA-256 checked
        paths = [art.fetch("datasets", name, fn, offline=False) for fn in e["files"]]
    else:
        paths = [it["files"][fn] for fn in e["files"]]
    layout = parquet.write_dataset(name, parquet.read_npz(paths), work, split_order=hub_cards.split_order(name),
                                   column_order=list(e["columns"]))
    it["layout"] = layout
    it["derived"] = {f: work / f for f in parquet.parquet_files(layout)}
    it["readme"] = hub_cards.dataset_card(name, layout)


def _prepare_model(it: dict, work: Path) -> None:
    import warnings

    from . import load
    from .export import export_onnx, export_safetensors
    name = it["name"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zm = load(name)
        export_safetensors(zm, work / "model.safetensors")
        try:
            onnx_info = export_onnx(zm, work / "model.onnx")
        except NotImplementedError as exc:                       # no ONNX graph for this adapter
            print(f"  {name}: no ONNX copy ({exc})")
            onnx_info = None
        example = hub_cards.model_example(zm)
    (work / "example.json").write_text(hub_cards.example_json(example))
    it["derived"] = {f: work / f for f in DERIVED_MODEL_FILES if (work / f).exists()}
    it["example"], it["onnx"] = example, onnx_info
    it["readme"] = hub_cards.model_card(name, example=example, onnx_info=onnx_info, files=list(it["derived"]))


# ============================================================================================== publish
def _commit(api, repo: str, repo_type: str, files: dict[str, Path], readme: str, message: str,
            replace_prefixes: Sequence[str] = ()) -> Any:
    """One commit: add ``files`` (path in repo -> local path) and README.md; delete the files under
    ``replace_prefixes`` that the commit does not add (stale shards of an earlier layout)."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode()))
    if replace_prefixes:
        try:
            remote = api.list_repo_files(repo, repo_type=repo_type)
        except Exception:  # noqa: BLE001 - a new repo has no files yet
            remote = []
        ops += [CommitOperationDelete(path_in_repo=f) for f in remote
                if f.startswith(tuple(replace_prefixes)) and f not in files]
    return api.create_commit(repo, operations=ops, commit_message=message, repo_type=repo_type)


def _tag(api, it: dict, commit) -> None:
    refs = api.list_repo_refs(it["repo"], repo_type=it["repo_type"])
    if it["revision"] in {t.name for t in refs.tags}:
        print(f"  {it['repo']}: tag {it['revision']} already exists (kept; bump the revision to republish)")
    else:
        api.create_tag(it["repo"], tag=it["revision"], repo_type=it["repo_type"],
                       revision=getattr(commit, "oid", None))


def publish(items: Iterable[dict], token: str, *, work: Path | None = None, tags: bool = True) -> list[str]:
    """Upload every item (one commit each) and tag it with its revision; returns the items that failed (each failure
    is printed and the others are still published). A Hub-only dataset from the Hub copy is prepared here, one at a
    time, and its downloads deleted afterwards (the npz are already on the Hub, so only the Parquet files and the
    card are committed)."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    work = _workdir(work)
    failed = []
    for it in items:
        cache = Path(tempfile.mkdtemp(prefix="zoo_cache_", dir=work)) if it["readme"] is None else None
        old = os.environ.get("PINNEAPPLE_ZOO_CACHE")
        try:
            api.create_repo(it["repo"], repo_type=it["repo_type"], exist_ok=True)
            if cache is not None:
                os.environ["PINNEAPPLE_ZOO_CACHE"] = str(cache)
                prepare(it, _item_dir(work, it))
            files = {**it["files"], **it["derived"]}
            commit = _commit(api, it["repo"], it["repo_type"], files, it["readme"],
                             f"{it['name']} {it['revision']}: card, "
                             + ("Parquet copies" if it["kind"] == "datasets" else "safetensors/ONNX copies"),
                             replace_prefixes=("data/",) if it["kind"] == "datasets" else ())
            if tags:
                _tag(api, it, commit)
            size = sum(Path(p).stat().st_size for p in it["derived"].values())
            print(f"published {it['kind'][:-1]} {it['name']} -> {_hub_url(it)} @ "
                  f"{it['revision'] if tags else 'main (no tag)'}: {len(files)} files, derived {size / 1e6:.1f} MB")
        except Exception as exc:  # noqa: BLE001 - report it, publish the others, fail at the end
            print(f"::error::{it['kind'][:-1]} {it['name']} was not published: {type(exc).__name__}: {exc}")
            failed.append(it["name"])
        finally:
            if cache is not None:
                os.environ.pop("PINNEAPPLE_ZOO_CACHE") if old is None else os.environ.update(PINNEAPPLE_ZOO_CACHE=old)
                shutil.rmtree(cache, ignore_errors=True)
            shutil.rmtree(_item_dir(work, it), ignore_errors=True)
    return failed


SPACE_UI = ("index.html", "app.js", "space.css")
PLAYGROUND_UI = {"js/charts.js": "js/charts.js", "js/ui.js": "js/ui.js", "css/app.css": "css/app.css",
                 "icon.svg": "icon.svg"}


def build_space(out: Path, app_dir: Path, playground_dir: Path, models: Sequence[str] | None = None) -> list[str]:
    """Write the static demo Space into ``out``: the data and runtime of :func:`pinneapple_hub.zoo.demo.export_space`,
    the page of ``app_dir`` (``apps/zoo_space``), the Playground's charts, widgets, styles and icon
    (``playground_dir`` = ``apps/playground/site``) and the README (Space configuration). Returns the files written,
    relative to ``out``."""
    from . import demo
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    demo.export_space(out, models)
    for f in SPACE_UI:
        shutil.copy2(Path(app_dir) / f, out / f)
    for dst, src in PLAYGROUND_UI.items():
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(playground_dir) / src, out / dst)
    (out / "README.md").write_text(hub_cards.space_readme())
    return sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file())


SPACE_PLAN = ("the Hub refused to create the Space (HTTP 402: a plan is required for this kind of Space); static "
              "Spaces are free for every organisation. Run the workflow with space=false to publish everything else.")


def publish_space(token: str, app_dir: Path, playground_dir: Path, *, work: Path | None = None,
                  from_hub: bool = True) -> list[str]:
    """Build the static demo Space (:func:`build_space`) and upload it in one commit, deleting the files of an earlier
    version that the new one does not have. With ``from_hub`` the sweeps are computed from the models and data
    downloaded from the Hub (the bundled copies ignored), i.e. from what was published. Returns the files uploaded."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    api = HfApi(token=token)
    repo = hub_cards.SPACE_REPO
    work = _workdir(work)
    site = work / "space"
    keep = {k: os.environ.get(k) for k in ("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "PINNEAPPLE_ZOO_CACHE")}
    if from_hub:
        os.environ.update(PINNEAPPLE_ZOO_IGNORE_BUNDLED="1", PINNEAPPLE_ZOO_CACHE=str(work / "space-cache"))
    try:
        files = build_space(site, app_dir, playground_dir)
    finally:
        for k, v in keep.items():
            os.environ.pop(k, None) if v is None else os.environ.update({k: v})
    try:
        api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    except Exception as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) == 402:
            raise SystemExit(f"::error::{repo}: {SPACE_PLAN}") from exc
        raise
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(site / f)) for f in files]
    ops += [CommitOperationDelete(path_in_repo=f) for f in api.list_repo_files(repo, repo_type="space")
            if f not in files and f != ".gitattributes"]
    api.create_commit(repo, operations=ops, repo_type="space", commit_message="zoo demo (static)")
    size = sum((site / f).stat().st_size for f in files)
    print(f"published space -> {hub_cards.space_url()} (static, {len(files)} files, {size / 1e6:.1f} MB)")
    return files


# ============================================================================================== verify: files
def verify_files(items: list[dict], allow_unpublished: bool = False) -> int:
    """Download every manifest file anonymously from the item's revision (or the version tag it republishes) and
    compare its SHA-256 with the manifest; returns the number of failures. With ``allow_unpublished``, a Hub-only
    item that is not on the Hub yet (HTTP 404) is reported but not counted; a wrong checksum always fails."""
    import hashlib
    bad = total = 0
    for it in items:
        prefix = "datasets/" if it["repo_type"] == "dataset" else ""
        ent = art.entry(it["kind"], it["name"])
        for fn in it["files"]:
            total += 1
            want = ent["files"][fn]["sha256"]
            ok, got, url = False, "", ""
            for rev in art.revisions(ent):
                url = f"{art.HF_ENDPOINT}/{prefix}{it['repo']}/resolve/{rev}/{fn}"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "pinneapple-zoo-verify"})
                    h = hashlib.sha256()
                    with urllib.request.urlopen(req, timeout=600) as r:
                        for block in iter(lambda r=r: r.read(1 << 20), b""):
                            h.update(block)
                    got = h.hexdigest()
                    ok = got == want
                    break
                except urllib.error.HTTPError as e:
                    got = f"error: {e}"
                    if e.code != 404:
                        break
                except Exception as e:  # noqa: BLE001 - report every failure, keep going
                    got = f"error: {e}"
                    break
            if not ok and got.startswith("error: HTTP Error 404") and allow_unpublished and it["hub_only"]:
                print(f"  ... {url} not published yet (Hub-only; publish it with its staging files)")
                total -= 1
                continue
            bad += not ok
            print(f"  {'OK ' if ok else 'BAD'} {url} sha256 {got[:16]}{'' if ok else ' != ' + want[:16]}")
    print(f"verify: {total - bad} OK, {bad} failed")
    return bad


# ============================================================================================== verify: viewer
def _get_json(url: str, timeout: float = 60) -> tuple[int, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "pinneapple-zoo-verify"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode() or "null")
        except Exception:  # noqa: BLE001 - not JSON
            body = None
        return e.code, body
    except Exception as e:  # noqa: BLE001 - network error: retried by the caller
        return 0, {"error": str(e)}


def _poll(fn: Callable[[], tuple[bool, Any]], timeout: float, interval: float, sleep=time.sleep) -> tuple[bool, Any]:
    """Call ``fn`` until it returns ``(True, value)`` or ``timeout`` seconds have passed; returns the last result."""
    t0 = time.monotonic()
    while True:
        done, value = fn()
        if done or time.monotonic() - t0 >= timeout:
            return done, value
        sleep(interval)


def viewer_status(repo: str, *, timeout: float = 2400, interval: float = 30, get=_get_json,
                  sleep=time.sleep) -> dict[str, Any]:
    """The dataset viewer's state of one dataset repo: ``is-valid`` (polled until ``preview`` and ``viewer`` are
    true or ``timeout``), the configs and splits it found and, per split, the ``first-rows`` answer (rows,
    truncated cells, error)."""
    q = urllib.parse.quote(repo, safe="")

    def valid():
        code, body = get(f"{DATASETS_SERVER}/is-valid?dataset={q}")
        ok = code == 200 and isinstance(body, dict) and body.get("preview") and body.get("viewer")
        return bool(ok), {"code": code, **(body if isinstance(body, dict) else {"body": body})}

    ok, isvalid = _poll(valid, timeout, interval, sleep)
    out: dict[str, Any] = {"repo": repo, "is_valid": isvalid, "valid": ok, "splits": [], "first_rows": {}}
    if not ok:
        return out

    def splits():
        code, body = get(f"{DATASETS_SERVER}/splits?dataset={q}")
        return code == 200 and bool(body and body.get("splits")), body

    ok_s, body = _poll(splits, min(timeout, 600), interval, sleep)
    out["splits"] = [(s["config"], s["split"]) for s in (body or {}).get("splits", [])] if ok_s else []
    for cfg, split in out["splits"]:
        url = (f"{DATASETS_SERVER}/first-rows?dataset={q}&config={urllib.parse.quote(cfg)}"
               f"&split={urllib.parse.quote(split)}")

        def rows(url=url):
            code, b = get(url, timeout=120)
            return code == 200 and isinstance(b, dict) and "rows" in b, {"code": code, "body": b}

        ok_r, res = _poll(rows, min(timeout, 900), interval, sleep)
        b = res["body"] if isinstance(res["body"], dict) else {}
        out["first_rows"][f"{cfg}/{split}"] = {
            "ok": bool(ok_r and b.get("rows")), "rows": len(b.get("rows", [])),
            "truncated_cells": sum(len(r.get("truncated_cells", [])) for r in b.get("rows", [])),
            "columns": len(b.get("features", [])), "error": None if ok_r else (b.get("error") or res["code"])}
    return out


def verify_viewer(names: Sequence[str] | None = None, *, timeout: float = 2400, interval: float = 30,
                  get=_get_json, sleep=time.sleep) -> int:
    """Check the dataset viewer of every dataset (anonymous datasets-server API): ``is-valid`` must say preview and
    viewer, every split of the manifest must be listed, and ``first-rows`` of every split must return rows. Prints
    one block per dataset; returns the number of datasets that fail."""
    bad = 0
    for name in names or sorted(art.manifest()["datasets"]):
        e = art.entry("datasets", name)
        st = viewer_status(e["hf_repo"], timeout=timeout, interval=interval, get=get, sleep=sleep)
        iv = st["is_valid"]
        print(f"{name} ({e['hf_repo']}): is-valid preview={iv.get('preview')} viewer={iv.get('viewer')} "
              f"search={iv.get('search')} filter={iv.get('filter')}"
              + ("" if st["valid"] else f" -> NOT VALID ({iv.get('error') or iv.get('code')})"))
        listed = {s for _, s in st["splits"]}
        missing = [s for s in e["splits"] if s not in listed] if st["valid"] else list(e["splits"])
        for key, fr in st["first_rows"].items():
            print(f"  first-rows {key}: {'OK ' if fr['ok'] else 'BAD'} {fr['rows']} rows, {fr['columns']} columns, "
                  f"{fr['truncated_cells']} truncated cells" + ("" if fr["ok"] else f", error {fr['error']}"))
        if missing:
            print(f"  splits missing from the viewer: {missing}")
        fail = not st["valid"] or bool(missing) or not all(fr["ok"] for fr in st["first_rows"].values())
        bad += fail
    print(f"viewer: {len(names or art.manifest()['datasets']) - bad} OK, {bad} failed")
    return bad


# ============================================================================================== verify: models
def _download(repo: str, filename: str, revision: str, dest: Path, repo_type: str = "model") -> Path:
    """Anonymous download (no token) of one file of a repo at a revision."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, revision=revision, repo_type=repo_type, token=False,
                                local_dir=str(dest)))


def _close(a: dict[str, float], b: dict[str, float], rtol: float, atol: float = 0.0) -> bool:
    return set(a) == set(b) and all(abs(a[k] - b[k]) <= max(rtol * abs(b[k]), atol) for k in a)


def verify_model(name: str, work: Path) -> list[str]:
    """Problems found with the published model ``name`` (empty when none): it is loaded from the Hub with the zoo's
    loader (bundled copy ignored, fresh cache); its card metrics must be reproduced on the stored reference
    (``pp.zoo.evaluate``); the card's example must be reproduced by the PyTorch weights, by ``model.safetensors``
    and by ``model.onnx`` downloaded from the repo; the README's front matter must say what the card promises and
    its "How to use" code must run and print the example's outputs."""
    import warnings

    from . import evaluate, load
    from .export import (
        PARITY_TOL,
        as_graph_outputs,
        load_safetensors,
        parity_error,
        parity_inputs,
        run_onnx,
    )
    e = art.entry("models", name)
    rev = e["revision"]
    problems = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zm = load(name, offline=False)
        ev = evaluate(zm, offline=False)
    if not ev["matches_card"]:
        problems.append(f"card metrics not reproduced (max relative difference {ev['max_rel_diff_vs_card']:.2e})")
    got = {f: _download(e["hf_repo"], f, rev, work) for f in ("README.md", *DERIVED_MODEL_FILES)}
    ex = json.loads(got["example.json"].read_text())
    pred = {k: float(np.asarray(v).ravel()[0]) for k, v in zm.predict(ex["inputs"], warn=False).items()}
    if not _close(pred, ex["outputs"], 1e-5):
        problems.append(f"example not reproduced: {pred} != {ex['outputs']}")
    X = parity_inputs(zm, n_random=200)
    ref = zm._forward(X)
    st = load_safetensors(zm, got["model.safetensors"])._forward(X)
    if not all(np.array_equal(ref[k], st[k]) for k in ref):
        problems.append("model.safetensors does not give the same predictions as weights.pt")
    onnx_err = parity_error(zm, as_graph_outputs(zm, ref), run_onnx(got["model.onnx"], X))
    if onnx_err > PARITY_TOL:
        problems.append(f"model.onnx differs from PyTorch by {onnx_err:.2e} (> {PARITY_TOL:g})")
    readme = got["README.md"].read_text()
    meta = hub_cards.parse_front_matter(readme)
    want = {"library_name": "pinneapple", "pipeline_tag": "tabular-regression", "inference": False}
    if any(meta.get(k) != v for k, v in want.items()):
        problems.append(f"README front matter {({k: meta.get(k) for k in want})} != {want}")
    if hub_cards.space_url() not in readme:
        problems.append("README does not link the demo Space")
    snippet = next((b for b in hub_cards.code_blocks(readme) if "pp.zoo.load" in b), None)
    if snippet is None:
        problems.append("README has no 'How to use' code")
    else:
        run = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, timeout=900,
                             env={**os.environ, "PYTHONWARNINGS": "ignore"})
        first = run.stdout.splitlines()[0] if run.stdout else ""
        try:
            printed = ast.literal_eval(first)
            shown = {k: round(v, 6) for k, v in ex["outputs"].items()}
            if not _close(printed, shown, 1e-5, 2e-6):                  # printed rounded to 6 decimals
                problems.append(f"'How to use' printed {printed}, the card says {shown}")
        except (ValueError, SyntaxError):
            problems.append(f"'How to use' failed (exit {run.returncode}): {(run.stderr or run.stdout)[-400:]}")
    print(f"{name} ({e['hf_repo']} @ {rev}): {'OK' if not problems else 'BAD'} - loaded from the Hub, card metrics "
          f"{'reproduced' if ev['matches_card'] else 'NOT reproduced'} (max rel diff {ev['max_rel_diff_vs_card']:.1e}),"
          f" example {pred} vs stored reference {ex['reference']}, safetensors identical, ONNX parity {onnx_err:.1e}"
          + "".join(f"\n  problem: {p}" for p in problems))
    return problems


def verify_models(names: Sequence[str] | None = None) -> int:
    """:func:`verify_model` for every model, anonymously (no token), with the bundled copies ignored and an empty
    cache; returns the number of models with problems."""
    keep = {k: os.environ.get(k) for k in ("PINNEAPPLE_ZOO_IGNORE_BUNDLED", "PINNEAPPLE_ZOO_CACHE", "HF_TOKEN",
                                           "HF_HUB_DISABLE_IMPLICIT_TOKEN")}
    bad = 0
    with tempfile.TemporaryDirectory(prefix="zoo_verify_") as tmp:
        os.environ.update(PINNEAPPLE_ZOO_IGNORE_BUNDLED="1", PINNEAPPLE_ZOO_CACHE=str(Path(tmp) / "cache"),
                          HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
        os.environ.pop("HF_TOKEN", None)
        try:
            for name in names or sorted(art.manifest()["models"]):
                try:
                    bad += bool(verify_model(name, Path(tmp) / name))
                except Exception as exc:  # noqa: BLE001 - report every model
                    print(f"{name}: BAD - {type(exc).__name__}: {exc}")
                    bad += 1
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None) if v is None else os.environ.update({k: v})
    print(f"models: {len(names or art.manifest()['models']) - bad} OK, {bad} failed")
    return bad


# ============================================================================================== verify: Space
SPACE_OK = {"RUNNING"}
SPACE_FAILED = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "DELETING"}
SPACE_MARKER = "PINNeAPPle zoo demo"


def _space_host(runtime) -> str:
    """Host that serves the Space's page: the runtime's own domain when it reports one, else the static host."""
    raw = getattr(runtime, "raw", None) or {}
    for d in raw.get("domains", []) or []:
        if d.get("domain"):
            return d["domain"]
    owner, name = hub_cards.SPACE_REPO.split("/")
    return f"{owner}-{name}".lower().replace("_", "-") + ".static.hf.space"


def _fetch(url: str, timeout: float = 60) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "pinneapple-zoo-verify"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:  # noqa: BLE001 - network error: reported
        return 0, str(e)


def verify_space(*, timeout: float = 2400, interval: float = 30, runtime=None, fetch=_fetch, sleep=time.sleep) -> int:
    """Wait for the demo Space to run and serve its page; 0 when the stage is RUNNING and the page, ``index.json``
    and every model's sweeps load (anonymously), 1 otherwise. Prints the stage, hardware and what was fetched."""
    if runtime is None:
        from huggingface_hub import HfApi
        api = HfApi()

        def runtime():
            return api.get_space_runtime(hub_cards.SPACE_REPO)

    last = {}

    def stage():
        try:
            r = runtime()
        except Exception as exc:  # noqa: BLE001 - not created yet / transient
            return False, {"stage": f"unavailable ({exc})", "hardware": None}
        last["runtime"] = r
        s = str(getattr(r, "stage", ""))
        return s in SPACE_OK | SPACE_FAILED, {"stage": s, "hardware": getattr(r, "hardware", None)}

    _, st = _poll(stage, timeout, interval, sleep)
    ok = st["stage"] in SPACE_OK
    print(f"space {hub_cards.SPACE_REPO}: stage {st['stage']}, hardware {st['hardware']} ({hub_cards.space_url()})")
    if ok:
        base = f"https://{_space_host(last.get('runtime'))}"

        def page():
            code, body = fetch(f"{base}/index.html")
            return code == 200 and SPACE_MARKER in body, code

        served, code = _poll(page, min(timeout, 600), interval, sleep)
        print(f"  {base}/index.html: HTTP {code}{'' if served else ' (page not served)'}")
        ok = served
        if served:
            code, body = fetch(f"{base}/index.json")
            names = [m["name"] for m in json.loads(body)["models"]] if code == 200 else []
            print(f"  index.json: HTTP {code}, models {names}")
            for n in names:
                c, b = fetch(f"{base}/models/{n}.sweeps.json")
                good = c == 200 and bool(json.loads(b).get("sweeps")) if c == 200 else False
                print(f"  models/{n}.sweeps.json: HTTP {c}{'' if good else ' BAD'}")
                ok &= good
            ok &= bool(names)
    print(f"space: {'OK' if ok else 'BAD'}")
    return 0 if ok else 1
