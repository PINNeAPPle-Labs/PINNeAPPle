"""Where the Hub's items come from. Each collector reads one existing source of the repository and returns items in
the Hub format (see :mod:`pinneapple_hub.library`); nothing here is written by hand.

=================  =================================================================================================
``models``         the zoo manifest and the bundled model cards (``pinneapple_hub/zoo``), an example row of each
                   model's own held-out test split, the browser ONNX graphs (``data/web/zoo``)
``datasets``       the zoo manifest's datasets (``pp.datasets``)
``geometries``     the geometry library (``pinneapple_design.geometry_library``: ``web.index()`` and the manifest)
``flows``          the bundled flows of Flow Studio (``pinneapple_flow/flows``)
``nodes``          the curated node types of Flow Studio's palette (``pinneapple_flow.REGISTRY``)
``recipes``        the Python blocks of AGENTS.md and the examples of the module reference
                   (``scripts/reference/content.py``, recorded by ``scripts/reference/run_examples.py``)
``apis``           the "Use these APIs first" table of AGENTS.md, resolved to the public functions it names
``agent_tools``    the tools of the MCP server (``pinneapple_mcp.server``)
``examples``       the experiments page of the Lab (``pinneapple_lab.explorer.collect``): every ``examples/*``
                   manifest and every Lab item, with its A-F seal and six-stage trust card
=================  =================================================================================================
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import snippets as S

ROOT = Path(__file__).resolve().parents[2]
GITHUB = "https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main"
REFERENCE = "docs/org/reference/index.html"


def rel(p: Path | str) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def _json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _first_line(text: str, limit: int = 180) -> str:
    text = " ".join((text or "").strip().split("\n\n")[0].split())
    return text if len(text) <= limit else text[:limit - 1].rsplit(" ", 1)[0] + "…"


def domain(name: str, tags=()) -> str:
    """The physical domain, by the Lab's own rule (``pinneapple_lab.manifest.domain_of``): on the first word of
    ``name`` (the item's own name) when that decides it, else on the whole text and the tags."""
    from pinneapple_lab.manifest import domain_of
    own = domain_of(name.split(" ", 1)[0])
    return own if own != "general" else domain_of(name, tags)


def _file(label: str, path: Path | str | None = None, *, url: str = "", fmt: str = "", size: int | None = None,
          sha256: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"label": label, "format": fmt or Path(str(path or url)).suffix.lstrip(".").lower()}
    if path is not None:
        p = Path(path)
        out["path"] = rel(p)
        out["bytes"] = p.stat().st_size if p.exists() else size
    if url:
        out["url"] = url
        out.setdefault("bytes", size)
    if sha256:
        out["sha256"] = sha256
    return out


@lru_cache(maxsize=1)
def _example_sources() -> dict[str, str]:
    """``examples/<folder>`` -> the text of its Python scripts (to link items an example actually uses)."""
    out = {}
    for d in sorted((ROOT / "examples").iterdir()):
        if d.is_dir() and not d.name.startswith(("_", ".")) and (d / "experiment.json").exists():
            out[f"examples/{d.name}"] = "\n".join(p.read_text(errors="ignore") for p in sorted(d.rglob("*.py"))
                                                  if "__pycache__" not in p.parts)
    return out


def used_by_examples(*needles: str) -> list[str]:
    """Example folders whose scripts contain one of ``needles`` as a quoted string or a call."""
    hits = []
    for folder, text in _example_sources().items():
        if any(re.search(rf"[\"']{re.escape(n)}[\"']", text) for n in needles if n):
            hits.append(f"example:{folder}")
    return hits


# --------------------------------------------------------------------------------------------------------- models
def _zoo_manifest() -> dict[str, Any]:
    return _json(ROOT / "pinneapple_hub" / "zoo" / "manifest.json", {"models": {}, "datasets": {}})


def _assets() -> Path:
    return ROOT / "pinneapple_hub" / "zoo" / "assets"


def _example_row(card: dict[str, Any], dataset: str) -> tuple[dict[str, float], dict[str, float], str]:
    """The first row of the model's own held-out split: its inputs and the stored reference outputs."""
    import numpy as np

    from pinneapple_hub.zoo import datasets as D
    info = D.info(dataset)
    split = next((s for s in ("test", "cal", "val") if s in info.splits), next(iter(info.splits)))
    data = D.load(dataset, split=split, offline=True)
    names = [i["name"] for i in card.get("inputs", [])]
    outs = [o["name"] for o in card.get("outputs", [])]
    cols = set(data.columns)
    x = {n: float(np.asarray(data[n])[0]) for n in names if n in cols}
    y = {n: float(np.asarray(data[n])[0]) for n in outs if n in cols}
    return x, y, split


def models() -> list[dict[str, Any]]:
    man = _zoo_manifest()
    web = ROOT / "data" / "web" / "zoo"
    out = []
    for name, m in man["models"].items():
        files = m.get("files", {})
        card_file = files.get("model_card.json", {}).get("bundled")
        card = _json(_assets() / card_file, {}) if card_file else {}
        try:
            x, y, split = _example_row(card, m.get("dataset", ""))
        except Exception:  # noqa: BLE001 - a dataset that is not bundled gives no example row
            x, y, split = {}, {}, ""
        inputs = [{**i, "range": (card.get("validity_range") or {}).get(i["name"]), "example": x.get(i["name"])}
                  for i in card.get("inputs", [])]
        tr = card.get("trust_report") or {}
        metrics = [{"name": k, "value": v, "unit": "%" if k.endswith("_pct") else "",
                    "how": f"held-out split of {m.get('dataset')} (model card)"}
                   for k, v in (card.get("validation_metrics") or {}).items() if isinstance(v, (int, float))]
        flist = [_file(f"{f} (bundled)", _assets() / e["bundled"], sha256=e.get("sha256", ""))
                 for f, e in files.items() if e.get("bundled") and (_assets() / e["bundled"]).exists()]
        onnx = web / f"{name}.onnx"
        if onnx.exists():
            flist.append(_file("ONNX graph with query checks (browser)", onnx))
        hub = f"https://huggingface.co/{m['hf_repo']}" if m.get("hf_repo") else ""
        out.append({
            "id": f"model:{name}", "kind": "model", "name": name, "title": card.get("task") or m.get("task") or name,
            "summary": _first_line(card.get("description") or m.get("task", "")),
            "description": card.get("description", ""), "domain": domain(name, m.get("tags", [])),
            "tags": sorted(set(m.get("tags", []))),
            "quality": {"decision": tr.get("decisao"), "score": tr.get("score"),
                        "label": f"zoo validation {tr.get('decisao', 'not run')}", "source": "model card trust report",
                        "checks": [{"name": c.get("name"), "status": c.get("status"), "value": c.get("value"),
                                    "limit": c.get("limit")} for c in tr.get("checagens", [])][:8]},
            "code": S.model(name, x),
            "results": {"metrics": metrics,
                        "reference_row": {"split": split, "inputs": x, "outputs": y} if x else None,
                        "limitations": card.get("known_limitations", [])},
            "examples": used_by_examples(name),
            "files": flist + ([{"label": "Hugging Face repository", "url": hub, "format": "hub"}] if hub else []),
            "links": {"guide": "docs/guides/model_zoo.md", "hub": hub, "source": f"{GITHUB}/pinneapple_hub/zoo"},
            "extra": {"inputs": inputs, "outputs": card.get("outputs", []), "dataset": m.get("dataset"),
                      "version": m.get("version"), "onnx": f"{name}" if onnx.exists() else "",
                      "license": card.get("license"), "citation": card.get("citation"),
                      "architecture": card.get("architecture")},
        })
    return out


def datasets() -> list[dict[str, Any]]:
    man = _zoo_manifest()
    users = {m.get("dataset"): n for n, m in man["models"].items()}
    out = []
    for name, d in man["datasets"].items():
        files = d.get("files", {})
        bundled = bool(files) and all(f.get("bundled") for f in files.values())
        splits = d.get("splits", {})
        split = next((s for s in ("test", "val", "train") if s in splits), next(iter(splits), None))
        flist = []
        for f, e in files.items():
            if e.get("bundled") and (_assets() / e["bundled"]).exists():
                flist.append(_file(f"{f} (bundled)", _assets() / e["bundled"], sha256=e.get("sha256", "")))
            elif d.get("hf_repo"):
                flist.append(_file(f, url=f"https://huggingface.co/datasets/{d['hf_repo']}/resolve/"
                                          f"{d.get('revision', 'main')}/{f}", size=e.get("size"),
                                   sha256=e.get("sha256", "")))
        hub = f"https://huggingface.co/datasets/{d['hf_repo']}" if d.get("hf_repo") else ""
        out.append({
            "id": f"dataset:{name}", "kind": "dataset", "name": name, "title": d.get("title", name),
            "summary": _first_line(d.get("description", "")), "description": d.get("description", ""),
            "domain": domain(name + " " + d.get("title", ""), d.get("tags", [])), "tags": sorted(d.get("tags", [])),
            "quality": {"label": "source verified", "verification": d.get("verification", ""),
                        "source": "zoo manifest"} if d.get("verification") else None,
            "code": S.dataset(name, split if bundled else None),
            "results": {"splits": {k: v.get("n") for k, v in splits.items()}, "split_rule": d.get("split_rule", ""),
                        "columns": d.get("columns", {})},
            "examples": used_by_examples(name) + ([f"model:{users[name]}"] if name in users else []),
            "files": flist + ([{"label": "Hugging Face dataset", "url": hub, "format": "hub"}] if hub else []),
            "links": {"guide": "docs/guides/zoo_catalogue.md", "hub": hub},
            "extra": {"license": d.get("license"), "citation": d.get("citation"), "source": d.get("source"),
                      "bundled": bundled, "bytes": sum(f.get("size", 0) for f in files.values())},
        })
    return out


# ---------------------------------------------------------------------------------------------------- geometries
def _geometry_domain(g: dict[str, Any]) -> str:
    """The domain of the physics the entry is built for (its first ``physics`` line, by the Lab's rule); external
    aerodynamics, wind and hull resistance, which that rule has no word for, are fluid dynamics."""
    from pinneapple_lab.manifest import domain_of
    first = (g.get("physics") or [""])[0]
    d = domain_of(first)
    if d == "general" and re.search(r"aerodynam|wind|resistance|hemodynamic|cfd|flow", first, re.I):
        return "fluid dynamics"
    return d if d != "general" else domain(g["name"] + " " + g["category"], g.get("tags", []))


def geometries() -> list[dict[str, Any]]:
    from pinneapple_design.geometry_library import web
    from pinneapple_design.geometry_library.store import hub_url, read_manifest
    man = read_manifest()["geometries"]
    out = []
    for g in web.index()["geometries"]:
        n = g["name"]
        flist = []
        for fname, e in sorted(man.get(n, {}).get("files", {}).items()):
            local = ROOT / "data" / "geometries" / n / fname
            lod = fname.split(".")[-2]
            label = f"{fname.rsplit('.', 1)[-1].upper()} · {lod}"
            if "repo" in e.get("location", "") and local.exists():
                flist.append({**_file(label, local, sha256=e.get("sha256", "")), "url": hub_url(n, fname)})
            else:
                flist.append(_file(label, url=hub_url(n, fname), size=e.get("bytes"), sha256=e.get("sha256", "")))
        thumb = ROOT / "data" / "web" / "geometries" / "thumbs" / f"{n}.jpg"
        out.append({
            "id": f"geometry:{n}", "kind": "geometry", "name": n, "title": g["title"],
            "summary": _first_line(g["description"]), "description": g["description"],
            "domain": _geometry_domain(g),
            "tags": sorted({g["category"], *g.get("tags", [])}),
            "quality": {"label": "sources cited", "source": "geometry library entry",
                        "estimates": sum(1 for s in g.get("sources", []) if s.get("estimate"))}
            if g.get("sources") else None,
            "code": S.geometry(n),
            "results": {"dimensions_m": g["dimensions_m"], "triangles": g["triangles"],
                        "physics": g.get("physics", []), "sources": g.get("sources", [])},
            "examples": used_by_examples(n),
            "files": flist,
            "links": {"guide": "docs/guides/geometry_library.md", "source": g["code"]},
            "thumbnail": rel(thumb) if thumb.exists() else "",
            "extra": {"glb": g["glb_repo"], "category": g["category"], "frame": g["frame"],
                      "parameters": g["parameters"], "regenerate": g["regenerate"]},
        })
    return out


# ---------------------------------------------------------------------------------------------- flows and nodes
def _flows_dir() -> Path:
    from pinneapple_flow.service import FLOWS_DIR
    return Path(FLOWS_DIR)


def flows() -> list[dict[str, Any]]:
    from .flowruns import current, read_flow_runs
    runs = read_flow_runs()
    out = []
    for p in sorted(_flows_dir().glob("*.json")):
        d = _json(p, {})
        types = [n["type"] for n in d.get("nodes", [])]
        text = " ".join([p.stem, d.get("name", ""), d.get("description", "")])
        out.append({
            "id": f"flow:{p.stem}", "kind": "flow", "name": p.stem, "title": d.get("name", p.stem),
            "summary": _first_line(d.get("description", "")), "description": d.get("description", ""),
            "domain": domain(text, types), "tags": sorted({t.split(".")[0] for t in types}),
            "quality": None,
            "code": S.flow(p.stem), "run_cmd": f"python -m pinneapple_flow run {p.stem}",
            "results": {"nodes": len(types), "edges": len(d.get("edges", [])), "node_types": types,
                        "flow_run": current(runs, p.stem)},
            "examples": used_by_examples(p.stem, f"{p.stem}.json"),
            "files": [_file("Flow JSON (pinneapple-flow/1)", p)],
            "links": {"guide": "docs/guides/flow_schema.md", "flow_studio": f"?example={p.stem}",
                      "source": f"{GITHUB}/{rel(p)}"},
            "extra": {"flow": p.stem},
        })
    return out


def _autocalls() -> dict[str, dict[str, Any]]:
    """The generated library-call nodes (``pinneapple_flow/autocalls.json``) by node type, with the check the
    generator ran on each (``runs with its defaults ...`` or ``needs input: ...``)."""
    from pinneapple_flow.autocalls import DATA
    return {c["key"]: c for c in _json(DATA, {}).get("calls", [])}


def _target_source(target: str) -> str:
    try:
        from pinneapple_flow.libcalls import resolve
        return _source_link(resolve(target))
    except Exception:  # noqa: BLE001 - a module that does not import here has no source link
        return ""


def nodes() -> list[dict[str, Any]]:
    """Every node type of Flow Studio's palette (``pinneapple_flow.REGISTRY``: the curated nodes and the generated
    library calls), so the palette and the Hub list the same tools from the same registry."""
    from pinneapple_flow import REGISTRY
    from pinneapple_flow.palette import is_auto
    used: dict[str, list[str]] = {}
    for p in sorted(_flows_dir().glob("*.json")):
        for n in _json(p, {}).get("nodes", []):
            used.setdefault(n["type"], []).append(p.stem)
    auto_info = _autocalls()
    from .flowruns import read_flow_runs, runs_for_node
    flow_runs = read_flow_runs()
    out = []
    for spec in REGISTRY.specs():
        if spec.type.startswith("custom."):
            continue
        d = spec.to_dict()
        pal = d["palette"]
        auto = is_auto(spec)
        info = auto_info.get(spec.type, {})
        check = info.get("check", "")
        runs = check.startswith("runs with its defaults")
        target = info.get("target", "")
        text = info.get("description") or spec.description
        if auto:
            code = S.auto_call(target, spec.type, spec.defaults()) if runs else S.node(spec.type)
            summary = _first_line(info.get("description", "")) or f"{target.replace(':', '.')}: {check}"
            title = f"{pal.get('title') or spec.title} ({target.split(':')[0].removeprefix('pinneapple_')})"
        else:
            code, summary, title = S.node(spec.type), _first_line(spec.description), pal.get("title") or spec.title
        out.append({
            "id": f"node:{spec.type}", "kind": "tool", "subkind": "library call" if auto else "flow node",
            "name": spec.type, "title": title, "summary": summary, "description": text,
            "domain": domain(" ".join([spec.type, spec.title, spec.category, text]), pal.get("tags", [])),
            "tags": sorted({spec.category, pal.get("section", ""), pal.get("group", ""), *pal.get("tags", [])} - {""}),
            "quality": {"label": "runs with its defaults", "check": check, "source": "pinneapple_flow.autocalls"}
            if runs else None,
            "code": code,
            "results": {k: v for k, v in {"check": check, "flow_runs": runs_for_node(flow_runs, spec.type)}.items()
                        if v} or None,
            "examples": [f"flow:{f}" for f in dict.fromkeys(used.get(spec.type, []))] + used_by_examples(spec.type),
            "files": [],
            "links": {"guide": "apps/flow_studio/README.md", "flow_studio": f"?node={spec.type}",
                      "source": _target_source(target) if auto else _source_link(spec.fn)},
            "extra": {"calls": target or spec.calls, "category": spec.category, "available": d["available"],
                      "install": pal.get("install", ""), "auto": auto,
                      "inputs": [{"name": p.name, "type": p.type, "required": p.required} for p in spec.inputs],
                      "outputs": [{"name": p.name, "type": p.type} for p in spec.outputs],
                      "params": [{"name": p.name, "kind": p.kind, "default": p.default, "help": p.help,
                                  "choices": p.choices, "minimum": p.minimum, "maximum": p.maximum,
                                  "advanced": p.advanced} for p in spec.params],
                      "params_schema": d["params_schema"]},
        })
    return out


def node_for_callable() -> dict[str, str]:
    """``module:function`` -> the node type that wraps it (generated library calls by their target, curated nodes by
    the ``module:function`` or ``module.function`` they declare in ``calls``)."""
    from pinneapple_flow import REGISTRY
    from pinneapple_flow.palette import is_auto
    out: dict[str, str] = {}
    for key, c in _autocalls().items():
        out.setdefault(c["target"], key)
    for spec in REGISTRY.specs():
        if is_auto(spec) or not spec.calls:
            continue
        for ref in re.findall(r"(pinneapple\w*(?:\.\w+)*)[:.](\w+)", spec.calls):
            out.setdefault(f"{ref[0]}:{ref[1]}", spec.type)
    return out


def _source_link(obj) -> str:
    try:
        src = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT).as_posix()
        return f"{GITHUB}/{src}#L{inspect.getsourcelines(obj)[1]}"
    except (TypeError, ValueError, OSError):
        return ""


# --------------------------------------------------------------------------------------------- recipes and APIs
def _agents_blocks() -> list[tuple[str, str]]:
    """``(heading, code)`` of each Python block of AGENTS.md (the blocks ``tests/test_agents_md.py`` runs)."""
    text = (ROOT / "AGENTS.md").read_text()
    out = []
    for m in re.finditer(r"```python\n(.*?)```", text, flags=re.S):
        head = re.findall(r"^#{2,4} (.+)$", text[:m.start()], flags=re.M)
        out.append((head[-1].strip() if head else "AGENTS.md", m.group(1)))
    return out


def _reference_content():
    path = ROOT / "scripts" / "reference" / "content.py"
    spec = importlib.util.spec_from_file_location("_pp_reference_content", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def recipes() -> list[dict[str, Any]]:
    out = []
    for k, (head, code) in enumerate(_agents_blocks()):
        out.append({
            "id": f"recipe:agents-{_slug(head)}", "kind": "tool", "subkind": "recipe", "name": f"agents_md_{k}",
            "title": head, "summary": f"From AGENTS.md: {head.lower()}.", "description": "",
            "domain": domain(head + " " + code), "tags": sorted(set(S.symbols(code)))[:12],
            "quality": {"label": "runs in CI", "source": "tests/test_agents_md.py"},
            "code": code, "results": None, "examples": [], "files": [],
            "links": {"guide": "AGENTS.md", "source": f"{GITHUB}/AGENTS.md"}, "extra": {"origin": "AGENTS.md"},
        })
    try:
        content = _reference_content()
    except Exception:  # noqa: BLE001 - outside a checkout the reference content is not there
        return out
    recorded = _json(ROOT / "scripts" / "reference" / "examples_output.json", {})
    for pkg in content.PACKAGES:
        for k, ex in enumerate(pkg.get("examples", [])):
            key = f"{pkg['name']}__{k}"
            rec = recorded.get(key) or {}
            out.append({
                "id": f"recipe:{key}", "kind": "tool", "subkind": "recipe", "name": key, "title": ex["title"],
                "summary": _first_line(pkg.get("tagline", "")), "description": " ".join(pkg.get("about", [])),
                "domain": domain(" ".join([pkg["name"], ex["title"], pkg.get("title", "")])),
                "tags": sorted({pkg["name"], pkg.get("group", "")} - {""}),
                "quality": None, "code": ex["code"], "results": None, "examples": [], "files": [],
                "links": {"reference": f"{REFERENCE}#{pkg['name']}", "source": f"{GITHUB}/scripts/reference/content.py"},
                "extra": {"origin": "module reference", "package": pkg["name"],
                          "reference_run": {"ok": rec.get("ok"), "stdout": rec.get("stdout", ""),
                                            "seconds": rec.get("seconds"), "code_hash": rec.get("code_hash"),
                                            "figures": [f"docs/org/reference/figures/{f}" for f in rec.get("figures", [])
                                                        if (ROOT / "docs/org/reference/figures" / f).exists()]}
                          if rec else None},
            })
    return out


def _resolve(dotted: str):
    parts = dotted.split(".")
    if parts[0] == "pp":
        obj = importlib.import_module("pinneapple")
        rest = parts[1:]
    else:
        obj, rest = None, parts
        for i in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
                rest = parts[i:]
                break
            except Exception:  # noqa: BLE001 - not a module at this depth
                continue
        if obj is None:
            return None
    for p in rest:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def apis() -> list[dict[str, Any]]:
    """One item per row of AGENTS.md's "Use these APIs first" table, for the functions that row names."""
    text = (ROOT / "AGENTS.md").read_text()
    table = text.split("## Use these APIs first", 1)[-1].split("\n\n", 2)[1]
    node_map = node_for_callable()
    out = []
    for line in table.splitlines()[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        task, api = cells[0], cells[1]
        names = list(dict.fromkeys(re.findall(r"`((?:pp|pinneapple_\w+)(?:\.\w+)+)\(", api)))
        resolved = [(n, _resolve(n)) for n in names]
        resolved = [(n, o) for n, o in resolved if callable(o)]
        if not resolved:
            continue
        main, obj = resolved[0]
        doc = inspect.getdoc(obj) or ""
        wraps = [w for w in (node_map.get(f"{getattr(o, '__module__', '')}:{getattr(o, '__name__', '')}")
                             for _, o in resolved) if w]
        links = dict(re.findall(r"\[([^\]]+)\]\(([^)]+)\)", api))
        guide = next((v for v in links.values() if v.endswith(".md")), "")
        out.append({
            "id": f"api:{_slug(main)}", "kind": "tool", "subkind": "function", "name": main, "title": task,
            "summary": _first_line(doc) or task, "description": doc[:3000],
            "domain": domain(task + " " + main), "tags": sorted({n.split("(")[0] for n, _ in resolved})[:10],
            "quality": None, "code": S.api(main), "results": None,
            "examples": [f"node:{w}" for w in dict.fromkeys(wraps)], "files": [],
            "links": {"guide": guide, "source": _source_link(obj),
                      **({"flow_studio": f"?node={wraps[0]}"} if wraps else {})},
            "extra": {"functions": [{"name": n, "signature": _signature(o), "summary": _first_line(inspect.getdoc(o))}
                                    for n, o in resolved], "table_row": api},
        })
    return out


def _signature(obj) -> str:
    try:
        s = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"
    return s if len(s) < 300 else s[:297] + "..."


def agent_tools() -> list[dict[str, Any]]:
    """The MCP server's tools, read from the source (``@server.tool()`` functions)."""
    path = ROOT / "pinneapple_mcp" / "server.py"
    tree = ast.parse(path.read_text())
    from pinneapple_mcp import tools as T
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or not any("tool" in ast.unparse(d) for d in fn.decorator_list):
            continue
        doc = ast.get_docstring(fn) or ""
        impl = getattr(T, fn.name, None)
        no_args = impl is not None and all(p.default is not inspect.Parameter.empty
                                           for p in inspect.signature(impl).parameters.values())
        out.append({
            "id": f"mcp:{fn.name}", "kind": "tool", "subkind": "agent tool (MCP)", "name": fn.name,
            "title": f"MCP tool · {fn.name}", "summary": _first_line(doc), "description": doc,
            "domain": "general", "tags": ["MCP", "agents"], "quality": None,
            "code": S.mcp(fn.name, call=no_args and fn.name in S.MCP_CHEAP), "results": None, "examples": [],
            "files": [], "links": {"guide": "docs/guides/mcp_server.md",
                                   "source": f"{GITHUB}/pinneapple_mcp/server.py#L{fn.lineno}"},
            "extra": {"server": "pinneapple-mcp"},
        })
    return out


# --------------------------------------------------------------------------------------------------- experiments
def examples() -> list[dict[str, Any]]:
    from pinneapple_lab.explorer import collect
    from pinneapple_lab.manifest import scan
    from pinneapple_lab.store import LabStore
    data = collect(LabStore(str(ROOT / "lab")))
    folders = {m.id: rel(Path(p).parent) for p, m in scan([str(ROOT / "examples")], lab="")}
    out = []
    for e in data["entries"]:
        key = e["key"]
        lab_item = e.get("source") == "lab"
        figures, videos = [], []
        for m in e.get("media", []):
            p = (ROOT / "lab" / "runs" / m["run"]) if m.get("run") else (ROOT / m["repo"]) if m.get("repo") else None
            if p is not None and p.exists():
                (videos if m.get("kind") == "video" else figures).append(rel(p))
        tier = e.get("tier")
        script = (e.get("links") or {}).get("script", "")
        out.append({
            "id": f"example:{key}", "kind": "example", "name": key, "title": e["title"],
            "summary": _first_line(e.get("description", "")), "description": e.get("description", ""),
            "domain": e.get("domain") or "general", "tags": sorted(set(e.get("methods", [])))[:10],
            "quality": ({"letter": tier, "label": f"Lab seal {tier} · {e.get('tier_name', '')}",
                         "reason": (e.get("seal") or {}).get("reason", ""), "stages": e.get("stages", {}),
                         "next_steps": e.get("next_steps", [])[:4], "ready": e.get("ready", []),
                         "source": "pinneapple_lab curation"} if tier else
                        {"letter": None, "label": "not curated", "source": "pinneapple_lab curation",
                         "how": e.get("curate") or ""}),
            "code": S.lab_item(key) if lab_item else S.example(folders.get(key, key)),
            "run_cmd": (f"python {script}" if not lab_item and script.endswith(".py") and (ROOT / script).exists()
                        else ""),
            "results": {"metrics": [m for m in e.get("metrics", []) if isinstance(m.get("value"), (int, float))][:16],
                        "figures": figures[:8], "videos": videos[:3]},
            "examples": [], "files": [],
            "links": {k: v for k, v in {"guide": (e.get("links") or {}).get("guide", ""),
                                        "source": f"{GITHUB}/{script}" if script.endswith(".py") else "",
                                        "lab": f"lab/index.html#{key}"}.items() if v},
            "extra": {"origin": e.get("source"), "experiment": e.get("experiment", ""), "run": e.get("run", ""),
                      "items": e.get("items", [])},
        })
    return out


COLLECTORS = {"models": models, "datasets": datasets, "geometries": geometries, "flows": flows, "nodes": nodes,
              "recipes": recipes, "apis": apis, "agent_tools": agent_tools, "examples": examples}
