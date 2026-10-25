"""The minimal, runnable code of each Hub item. Every snippet built here is run by
``python -m pinneapple_hub.library run`` (output recorded in ``data/web/hub/snippet_runs/``) and again by
``tests/test_hub_library.py``; keep them short, offline and fast."""
from __future__ import annotations

import re

#: MCP tools whose implementation is cheap and offline enough to call in a snippet
MCP_CHEAP = ("capabilities", "list_methods", "list_presets", "zoo_list")


def _fmt(v: float) -> str:
    return repr(round(float(v), 6))


def model(name: str, example: dict[str, float]) -> str:
    if example:
        args = ", ".join(f"{k}=[{_fmt(v)}]" for k, v in example.items())
        call = f"out = model.predict({args})"
    else:
        call = "out = model.predict(model.reference('test', offline=True)[0][:1])"
    return (f'import numpy as np\nimport pinneapple as pp\n\n'
            f'model = pp.zoo.load("{name}")            # weights checked against their SHA-256\n'
            f'{call}\n'
            f'print({{k: np.round(np.asarray(v, float), 5).tolist() for k, v in out.items()}})\n'
            f'report = pp.zoo.validate("{name}")        # card metrics, physics checks, validity range\n'
            f'print(report.decisao)\n')


def dataset(name: str, split: str | None) -> str:
    if split:
        return (f'import pinneapple as pp\n\n'
                f'data = pp.datasets.load("{name}", split="{split}")\n'
                f'print(data.n_rows, "rows;", list(data.columns)[:8])\n'
                f'print(pp.datasets.info("{name}").license)\n')
    return (f'import pinneapple as pp\n\n'
            f'info = pp.datasets.info("{name}")\n'
            f'print(info.title)\n'
            f'print({{s: d["n"] for s, d in info.splits.items()}}, info.license)\n'
            f'# data = pp.datasets.load("{name}", split="train")   # downloaded from the Hugging Face Hub on first use\n')


def geometry(name: str) -> str:
    return (f'import pinneapple as pp\n\n'
            f'info = pp.geometries.info("{name}")\n'
            f'print(info["title"], "|", info["category"])\n'
            f'stl = pp.geometries.load("{name}", format="stl")      # the committed light STL (or the Hub / cache)\n'
            f'print(stl.name, stl.stat().st_size, "bytes")\n'
            f'# g = pp.geometries.build("{name}"); g.export("{name}.glb")   # regenerate with your own parameters\n')


def flow(name: str) -> str:
    return (f'import pinneapple_flow as pf\n'
            f'from pinneapple_flow.service import FLOWS_DIR\n\n'
            f'flow = pf.Flow.load(FLOWS_DIR / "{name}.json")\n'
            f'print(flow.name, "-", len(flow.nodes), "nodes")\n'
            f'print("issues:", [i.message for i in pf.validate_flow(flow)])\n'
            f'# result = pf.run(flow, workdir="work")      # or: python -m pinneapple_flow run {name}\n')


def node(type_: str) -> str:
    return (f'import pinneapple_flow as pf\n\n'
            f'spec = pf.REGISTRY["{type_}"]\n'
            f'print(spec.title, "->", spec.calls or "flow node")\n'
            f'print("inputs:", [(p.name, p.type) for p in spec.inputs])\n'
            f'print("outputs:", [(p.name, p.type) for p in spec.outputs])\n'
            f'print("defaults:", spec.defaults())\n')


def auto_call(target: str, node_type: str, defaults: dict | None = None) -> str:
    """A generated library call that runs with its defaults (the node's parameter form): call it with them and show
    what it returns."""
    mod, fn = target.split(":")
    args = ", ".join(f"{k}={v!r}" for k, v in (defaults or {}).items())
    call = f"{fn}({args})"
    if len(call) > 100:
        call = f"{fn}(\n    " + ",\n    ".join(f"{k}={v!r}" for k, v in (defaults or {}).items()) + ",\n)"
    return (f'from {mod} import {fn}     # Flow Studio node {node_type}\n\n'
            f'result = {call}\n'
            f'print(type(result).__name__)\n'
            f'print(repr(result)[:800])\n')


def api(dotted: str) -> str:
    head = dotted.split(".")[0]
    imp = "import pinneapple as pp" if head == "pp" else f"import {dotted.rsplit('.', 1)[0]}"
    target = dotted if head == "pp" else dotted
    return (f'import inspect\n{imp}\n\n'
            f'print("{dotted}" + str(inspect.signature({target})))\n'
            f'print(inspect.getdoc({target}).splitlines()[0])\n')


def mcp(name: str, *, call: bool) -> str:
    if call:
        return (f'import json\nfrom pinneapple_mcp import tools as T     # what the MCP tool "{name}" runs\n\n'
                f'out = T.{name}()\n'
                f'print(json.dumps(out, default=str)[:800])\n')
    return (f'import inspect\n'
            f'from pinneapple_mcp import tools as T     # what the MCP tool "{name}" runs\n\n'
            f'print("{name}" + str(inspect.signature(T.{name})))\n'
            f'# start the server for an agent: pinneapple-mcp   (pip install "pinneapple[mcp]")\n')


def example(folder: str) -> str:
    return (f'import os\nfrom pinneapple_lab import manifest\n\n'
            f'm = manifest.load(os.path.join(manifest.REPO, "{folder}", "experiment.json"))\n'
            f'print(m.title, "|", m.domain, "|", m.status)\n'
            f'for x in m.metrics[:5]:\n'
            f'    print(x["name"], x["value"], x.get("unit", ""))\n')


def lab_item(key: str) -> str:
    return (f'import os\nfrom pinneapple_lab import manifest\nfrom pinneapple_lab.seal import seal_for\n\n'
            f's = seal_for("{key}", os.path.join(manifest.REPO, "lab"))\n'
            f'print(s.tier, "-", s.reason)\n')


def symbols(code: str) -> list[str]:
    """``pp.xxx`` names a snippet calls (for tags and links)."""
    return re.findall(r"\b(pp(?:\.\w+)+)\(", code)
