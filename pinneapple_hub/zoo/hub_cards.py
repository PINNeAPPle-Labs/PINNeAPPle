"""The README cards of the zoo's Hugging Face repos: YAML front matter the Hub reads, and the text people read.

* :func:`dataset_card`: ``configs`` (``data_files`` per split, pointing at the Parquet files of
  :mod:`pinneapple_hub.zoo.parquet`), ``dataset_info`` (features and split sizes), licence, task categories, size
  category and tags, then what each column means (unit, type), provenance, verification, the npz files with their
  SHA-256 and how to load the data with ``pp.datasets.load`` and with ``datasets.load_dataset``.
* :func:`model_card`: ``library_name: pinneapple``, ``pipeline_tag``, ``inference: false`` (no Hub inference
  provider runs these models, see :data:`NO_HOSTED_INFERENCE`), the dataset, the link to the zoo's demo Space, a
  "How to use" snippet whose numbers come from running the model (:func:`model_example`), the validity range, the
  trust card, the stored validation report and the files (PyTorch, safetensors, ONNX).

``scripts/zoo/publish_to_hub.py`` writes these cards; ``tests/test_zoo_hub_formats.py`` checks that the YAML is what
the Hub and ``datasets`` accept.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import yaml

from . import _artifacts as art

__all__ = ["SPACE_REPO", "space_url", "space_readme", "split_order", "dataset_card", "model_example", "model_card",
           "front_matter", "parse_front_matter", "code_blocks", "CITE"]

SPACE_REPO = "PINNeAPPle-Labs/zoo-demo"
GITHUB = "https://github.com/PINNeAPPle-Labs/PINNeAPPle"
CITE = ("## Cite\n\nIf you use this {what}, please cite PINNeAPPle (software, Zenodo DOI "
        "[10.5281/zenodo.23343062](https://doi.org/10.5281/zenodo.23343062); see "
        f"[CITATION.cff]({GITHUB}/blob/main/CITATION.cff)):\n\n```bibtex\n"
        "@software{{pinneapple,\n  author = {{Barros, Yan and Schiassi, Enrico}},\n  title  = {{PINNeAPPle}},\n"
        "  doi    = {{10.5281/zenodo.23343062}},\n  url    = {{https://github.com/PINNeAPPle-Labs/PINNeAPPle}}\n}}\n```\n")
NO_HOSTED_INFERENCE = (
    "No Hugging Face inference provider can run this model, so the Hub widget is off (`inference: false`). It is "
    "a small PyTorch network, but its prediction is more than a forward pass: the inputs are encoded and "
    "normalised, the outputs denormalised (and, for ensembles, averaged with their spread), and every query is "
    "checked against the validity range and the model's physics checks. The `pinneapple` library does all of "
    "that (below); the ONNX graph carries the pre- and post-processing but not the checks; the "
    f"[zoo demo Space](https://huggingface.co/spaces/{SPACE_REPO}) shows the model along a sweep of each input and "
    "runs it, with its checks, in your browser.")
# the zoo is not in a PyPI release yet (0.7.0 predates it): install from GitHub
INSTALL = 'pip install "pinneapple @ git+https://github.com/PINNeAPPle-Labs/PINNeAPPle"'
_SPLIT_FIRST = ("train", "val", "validation", "cal", "test")


def space_url() -> str:
    return f"https://huggingface.co/spaces/{SPACE_REPO}"


def _hub_url(entry: Mapping[str, Any]) -> str:
    return f"https://huggingface.co/{'datasets/' if entry['repo_type'] == 'dataset' else ''}{entry['hf_repo']}"


def front_matter(meta: Mapping[str, Any]) -> str:
    return "---\n" + yaml.safe_dump(dict(meta), sort_keys=False, allow_unicode=True, width=1000) + "---\n"


def parse_front_matter(readme: str) -> dict[str, Any]:
    """The YAML block at the top of a README (``{}`` when there is none)."""
    if not readme.startswith("---\n"):
        return {}
    end = readme.index("\n---\n", 4)
    return yaml.safe_load(readme[4:end]) or {}


def code_blocks(readme: str, lang: str = "python") -> list[str]:
    """The fenced code blocks of a README in language ``lang``."""
    out, cur = [], None
    for line in readme.splitlines():
        if cur is None and line.strip() == f"```{lang}":
            cur = []
        elif cur is not None and line.strip() == "```":
            out.append("\n".join(cur) + "\n")
            cur = None
        elif cur is not None:
            cur.append(line)
    return out


# ============================================================================================== datasets
def split_order(name: str) -> list[str]:
    """Splits of a dataset with train / val / cal / test first, then the others in manifest order."""
    splits = list(art.entry("datasets", name)["splits"])
    first = [s for s in _SPLIT_FIRST if s in splits]
    return first + [s for s in splits if s not in first]


def _size_category(n: int) -> str:
    for lim, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"), (1_000_000, "100K<n<1M"),
                       (10_000_000, "1M<n<10M")):
        if n < lim:
            return label
    return "10M<n<100M"


def _task_categories(entry: Mapping[str, Any], layout: Mapping[str, Any]) -> list[str]:
    tags = {t.lower() for t in entry.get("tags", [])}
    if "forecasting" in tags:
        return ["time-series-forecasting"]
    if "graph networks" in tags:
        return ["graph-ml"]
    arrays = any("[" in t for c in layout["configs"] for t in c["types"].values())
    return ["other"] if arrays else ["tabular-regression"]


def _tag(t: str) -> str:
    return t.strip().lower().replace(" ", "-")


def _dataset_meta(name: str, layout: Mapping[str, Any]) -> dict[str, Any]:
    e = art.entry("datasets", name)
    n = sum(s["samples"] for c in layout["configs"] for s in c["splits"])
    tags = list(dict.fromkeys(["pinneapple", "physics", "scientific-machine-learning", *map(_tag, e.get("tags", []))]))
    configs, info = [], []
    for c in layout["configs"]:
        configs.append({"config_name": c["name"], **({"default": True} if c["default"] else {}),
                        "data_files": [{"split": s["name"], "path": f"data/{c['name']}/{s['name']}-*.parquet"}
                                       for s in c["splits"]]})
        info.append({"config_name": c["name"], "features": c["features"],
                     "splits": [{"name": s["name"], "num_bytes": s["bytes"], "num_examples": s["rows"]}
                                for s in c["splits"]]})
    return {"pretty_name": e["title"], "license": e["license"], "task_categories": _task_categories(e, layout),
            "size_categories": [_size_category(n)], "tags": tags, "configs": configs, "dataset_info": info}


def _column_rows(name: str, cfg: Mapping[str, Any]) -> list[str]:
    cols = art.entry("datasets", name)["columns"]
    exploded = {c for s in cfg["splits"] for c in s["exploded"]}
    rows = []
    for col in cfg["columns"]:
        if col == "sample":
            d = {"unit": "-", "description": "sample: the row of the split in `pp.datasets.load` this step belongs to"}
        elif col == "step":
            d = {"unit": "-", "description": "step: index along the first axis of the stored arrays"}
        else:
            d = cols.get(col, {"unit": "-", "description": ""})
        typ = cfg["types"][col] + (" (one step)" if col in exploded else "")
        rows.append(f"| `{col}` | {d.get('unit', '-')} | {typ} | {d.get('description', '')} |")
    return rows


def _first_split(cfg: Mapping[str, Any]) -> str:
    names = [s["name"] for s in cfg["splits"]]
    return "test" if "test" in names else names[0]


def _load_snippets(name: str, layout: Mapping[str, Any]) -> list[str]:
    e = art.entry("datasets", name)
    main = layout["configs"][0]
    split = _first_split(main)
    big = next((col for col in main["columns"] if "[" in main["types"][col]), None)
    s0 = next(s for s in main["splits"] if s["name"] == split)
    lines = ["## Load", "", "With the library (the npz files, SHA-256 checked; arrays with their full shape):", "",
             "```bash", INSTALL, "```", "", "```python", "import pinneapple as pp", "",
             f"d = pp.datasets.load(\"{name}\", split=\"{split}\")",
             "print(d.n_rows, d.columns)"]
    if big:
        lines.append(f"print(d[\"{big}\"].shape)")
    lines += ["```", "", "With `datasets` (the Parquet files of this repo):", "", "```python",
              "from datasets import load_dataset", "",
              f"ds = load_dataset(\"{e['hf_repo']}\"{'' if main['name'] == 'default' else ', ' + repr(main['name'])}"
              f", split=\"{split}\", revision=\"{e['revision']}\")",
              "print(ds)"]
    if big:
        lines += ["ds = ds.with_format(\"numpy\")", f"print(ds[0][\"{big}\"].shape)"]
    lines += ["```", ""]
    if s0["steps"]:
        lines += [f"Each sample of this dataset is larger than the viewer can show in one row, so the Parquet files "
                  f"store one row per step of its first axis ({s0['steps']} steps): `sample` is the row of the "
                  f"split in `pp.datasets.load` and `step` the index along that axis; columns "
                  f"{', '.join(f'`{c}`' for c in s0['exploded'])} hold one step, the others are repeated on every "
                  "step. `pp.datasets.load` returns the samples whole; `pinneapple_hub.zoo.parquet.read_parquet` "
                  "puts the Parquet rows back together the same way.", ""]
    if len(layout["configs"]) > 1:
        others = ", ".join(f"`{c['name']}`" for c in layout["configs"][1:])
        lines += [f"Splits with other columns are separate configs ({others}); pass the config name as the second "
                  "argument of `load_dataset`.", ""]
    return lines


def dataset_card(name: str, layout: Mapping[str, Any]) -> str:
    """README of a dataset repo: front matter (configs, dataset_info, licence, task categories, sizes, tags) and the
    card. ``layout`` is what :func:`pinneapple_hub.zoo.parquet.write_dataset` returned for this dataset."""
    from . import datasets as zds
    e = art.entry("datasets", name)
    info = zds.info(name)
    size = sum(f.get("size", 0) for f in e["files"].values())
    lines = [f"# {e['title']}", "", e["description"], "",
             f"`{name}` v{e['version']} (revision `{e['revision']}`) · licence {e['license']} · part of the "
             f"[PINNeAPPle]({GITHUB}) zoo", "",
             "## Configs and splits", "", "| Config | Split | Samples | Parquet rows | Parquet files | Content |",
             "|---|---|---|---|---|---|"]
    for c in layout["configs"]:
        for s in c["splits"]:
            desc = e["splits"].get(s["name"], {}).get("description", "")
            lines.append(f"| `{c['name']}` | `{s['name']}` | {s['samples']} | {s['rows']} | {len(s['files'])} | "
                         f"{desc} |")
    lines.append("")
    if e.get("split_rule"):
        lines += ["Split rule: " + e["split_rule"], ""]
    for c in layout["configs"]:
        title = "Columns" if len(layout["configs"]) == 1 else f"Columns of config `{c['name']}`"
        lines += [f"## {title}", "", "| Column | Unit | Type | Description |", "|---|---|---|---|",
                  *_column_rows(name, c), ""]
    lines += _load_snippets(name, layout)
    lines += ["## Provenance", "", e["source"], "", "## Verification", "", e["verification"], ""]
    card = zds.audit_card(name)
    if card is not None:
        lines += [zds._audit_markdown(card), ""]
    lines += ["## Files", "",
              "The npz files are the library's format and the source of truth: `pp.datasets.load` downloads them "
              "from this repo at the tagged revision and refuses any file whose SHA-256 differs from the manifest of "
              "the package. The Parquet files under `data/` are derived from them by "
              "`pinneapple_hub.zoo.parquet.write_dataset` and hold the same numbers (a test round-trips npz -> "
              "Parquet -> arrays bit for bit); they are what the dataset viewer and `load_dataset` read.", "",
              "| File | Size | SHA-256 |", "|---|---|---|",
              *(f"| `{fn}` | {f['size'] / 1e6:.1f} MB | `{f['sha256']}` |" for fn, f in e["files"].items()),
              "", f"Total {size / 1e6:.1f} MB of npz.", "",
              "## Citation", "", info.citation or "(none)", "", CITE.format(what="dataset")]
    return front_matter(_dataset_meta(name, layout)) + "\n" + "\n".join(lines)


# ============================================================================================== models
def _num(v: float) -> float:
    return float(f"{float(v):.6g}")


def model_example(zm) -> dict[str, Any]:
    """A held-out reference case of the model (first row of its ``test`` split): the inputs (6 significant
    digits), the model's outputs for exactly those inputs, the stored reference outputs, the query-check verdict and
    the model's trust decision. The "How to use" snippet prints these numbers and the publishing workflow checks that
    the model downloaded from the Hub reproduces them."""
    import warnings

    from . import query_checks, validate
    from .checks import has_query_checks
    X, Y = zm.reference("test")
    inputs = {k: _num(v) for k, v in zip(zm.input_names, X[0], strict=True)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = zm.predict(inputs, warn=False)
        decision = validate(zm).decisao
    verdict = query_checks(zm, inputs)[0]["verdict"]["status"] if has_query_checks(zm) else None
    return {"model": zm.name, "version": zm.card.version, "inputs": inputs,
            "in_range": bool(zm.in_range(inputs)[0]),
            "outputs": {k: float(np.asarray(v).ravel()[0]) for k, v in out.items()},
            "reference": {k: float(y) for k, y in zip(zm.reference_outputs, Y[0], strict=True) if np.isfinite(y)},
            "query_verdict": verdict, "trust_decision": decision}


def _how_to_use(name: str, example: Mapping[str, Any]) -> list[str]:
    ins = ", ".join(f"\"{k}\": {v!r}" for k, v in example["inputs"].items())
    shown = {k: round(v, 6) for k, v in example["outputs"].items()}
    ref = ", ".join(f"{k} = {v:.6g}" for k, v in example["reference"].items())
    return ["## How to use", "", "```bash", INSTALL, "```", "", "```python",
            "import pinneapple as pp", "",
            f"model = pp.zoo.load(\"{name}\")          # weights checked against their SHA-256",
            f"inputs = {{{ins}}}",
            "out = model.predict(inputs)                 # warns for inputs outside the validity range",
            "print({k: round(float(v[0]), 6) for k, v in out.items()})",
            f"# {shown}",
            *(["print(pp.zoo.query_checks(model, inputs)[0][\"verdict\"][\"status\"])   # per-query trust checks",
               f"# {example['query_verdict']}"] if example.get("query_verdict") else
              ["print(bool(model.in_range(inputs)[0]))      # inside the validity range",
               f"# {example['in_range']}"]),
            "print(pp.zoo.validate(model).decisao)        # card metrics reproduced + physics checks",
            f"# {example['trust_decision']}",
            "```", "",
            f"The inputs are the first held-out test case of the training dataset; its reference value is {ref}. "
            f"`pp.zoo.evaluate(\"{name}\")` recomputes every metric of the card on the stored reference.", ""]


def _onnx_section(entry: Mapping[str, Any], onnx_info: Mapping[str, Any], example: Mapping[str, Any]) -> list[str]:
    cols = onnx_info["input_columns"]
    row = ", ".join(repr(example["inputs"][c]) for c in cols)
    outs = "; ".join(f"`{k}`: {v}" for k, v in onnx_info["outputs"].items())
    return ["### ONNX", "",
            f"`model.onnx` (opset {onnx_info['opset']}) maps `x`, shape (n, {len(cols)}) float32 with the columns "
            f"{', '.join(f'`{c}`' for c in cols)}, to the outputs {outs}. Pre- and post-processing are in the graph "
            f"(largest difference from PyTorch on the reference inputs: {onnx_info['parity']:.1e}); the "
            "validity-range and physics checks are not, so check the range yourself.", "",
            "```python", "import numpy as np, onnxruntime as ort",
            "from huggingface_hub import hf_hub_download", "",
            f"path = hf_hub_download(\"{entry['hf_repo']}\", \"model.onnx\", revision=\"{entry['revision']}\")",
            "session = ort.InferenceSession(path)",
            f"x = np.array([[{row}]], np.float32)",
            "print(dict(zip([o.name for o in session.get_outputs()], session.run(None, {\"x\": x}))))",
            "```", ""]


def model_card(name: str, *, example: Mapping[str, Any], onnx_info: Mapping[str, Any] | None = None,
               files: Sequence[str] = ()) -> str:
    """README of a model repo. ``example``: :func:`model_example`; ``onnx_info``: what
    :func:`pinneapple_hub.zoo.export.export_onnx` returned (None: no ONNX file); ``files``: the derived files
    published next to the manifest's (``model.safetensors``, ``model.onnx``)."""
    from pinneapple_hub.model_card import ModelCard

    from .validation_reports import card_section
    e = art.entry("models", name)
    card = ModelCard.load(str(art.fetch("models", name, "model_card.json")))
    ds = art.entry("datasets", e["dataset"])
    tags = list(dict.fromkeys(["pinneapple", "physics", "surrogate-model", "pytorch",
                               *(["safetensors"] if "model.safetensors" in files else []),
                               *(["onnx"] if "model.onnx" in files else []), *map(_tag, e.get("tags", []))]))
    meta = {"license": e.get("license", card.license or "apache-2.0"), "library_name": "pinneapple",
            "pipeline_tag": "tabular-regression", "inference": False, "tags": tags, "datasets": [ds["hf_repo"]]}
    body = card.to_markdown().split("\n")
    head = body[:3] + ["", f"`{name}` v{e['version']} (revision `{e['revision']}`) · part of the [PINNeAPPle]({GITHUB}) "
                       f"zoo · training data [{ds['hf_repo']}]({_hub_url(ds)})", "",
                       f"**Try it without installing anything:** [PINNeAPPle zoo demo]({space_url()}) (pick "
                       f"`{name}`: its predictions along each input with their band and where its physics checks warn "
                       "or fail, and your own inputs evaluated in the browser).", ""]
    files_rows = [("weights.pt", "PyTorch `state_dict` (`torch.load(weights_only=True)`), what `pp.zoo.load` reads"),
                  ("config.json", "architecture and normalisation"),
                  ("model_card.json", "the model card: inputs, outputs, validity range, metrics, trust report")]
    if "model.safetensors" in files:
        files_rows.append(("model.safetensors", "the same tensors as `weights.pt` in safetensors format "
                                                "(`pinneapple_hub.zoo.export.load_safetensors` loads them)"))
    if "model.onnx" in files:
        files_rows.append(("model.onnx", "ONNX graph, physical inputs to physical outputs (below)"))
    files_rows.append(("example.json", "the \"How to use\" case: inputs, outputs, reference, verdicts"))
    sha = {fn: f["sha256"] for fn, f in e["files"].items()}
    lines = [*head, *body[3:], ""]
    section = card_section(name)
    if section:
        lines += [section, ""]
    lines += _how_to_use(name, example)
    lines += ["## Hosted inference", "", NO_HOSTED_INFERENCE, ""]
    if onnx_info is not None:
        lines += _onnx_section(e, onnx_info, example)
    lines += ["## Files", "", "| File | Content | SHA-256 |", "|---|---|---|",
              *(f"| `{fn}` | {what} | {('`' + sha[fn] + '`') if fn in sha else 'derived'} |" for fn, what in files_rows),
              "", "The SHA-256 are those of the package's manifest: `pp.zoo.load` refuses any other file.", "",
              CITE.format(what="model")]
    return front_matter(meta) + "\n" + "\n".join(lines)


# ============================================================================================== the demo Space
def space_readme() -> str:
    """README of the demo Space: the configuration of a static Space (``index.html``, no server), the models and
    datasets it shows, and what the page does."""
    man = art.manifest()
    meta = {"title": "PINNeAPPle zoo demo", "emoji": "🍍", "colorFrom": "yellow", "colorTo": "green",
            "sdk": "static", "app_file": "index.html", "pinned": True, "license": "apache-2.0",
            "short_description": "Zoo surrogates with their physics checks",
            "models": sorted(e["hf_repo"] for e in man["models"].values()),
            "datasets": sorted({man["datasets"][e["dataset"]]["hf_repo"] for e in man["models"].values()}),
            "tags": ["pinneapple", "physics", "surrogate-model", "trust", "onnx"]}
    rows = [f"| [`{n}`]({_hub_url(e)}) | {e['task']} | `{e['revision']}` |" for n, e in sorted(man["models"].items())]
    return front_matter(meta) + "\n" + "\n".join([
        "# PINNeAPPle zoo demo", "",
        "Every model of the [PINNeAPPle](" + GITHUB + ") zoo along a sweep of each of its inputs: the prediction, its "
        "ensemble spread or conformal 95 % bound, and where the model's query checks (validity range, physical "
        "outputs, model residuals) warn or fail, with the model's trust decision. The sweeps were computed from the "
        "published models when the zoo was published; **Evaluate in your browser** runs the model's ONNX graph, with "
        "the same query checks, on your own inputs.", "",
        "| Model | Task | Revision |", "|---|---|---|", *rows, "",
        "A static Space: the page is `apps/zoo_space` of the repository (UI only), the data is written by "
        "`pinneapple_hub.zoo.demo.export_space` and the browser runtime is `pinneapple_hub/zoo/web` "
        "(a dependency-free ONNX interpreter and the JS twin of the query checks, tested against Python).", ""])


def example_json(example: Mapping[str, Any]) -> str:
    return json.dumps(example, indent=1, sort_keys=False) + "\n"
