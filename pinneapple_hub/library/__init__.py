"""The PINNeAPPle Hub index: every ready-made thing in the repository -- tools (public functions, Flow Studio nodes,
recipes, MCP tools), trained models, geometries, datasets, flows and examples -- as one generated list, each item
with a runnable snippet whose recorded run is checked by a test.

    from pinneapple_hub import library as hub
    index = hub.load_index()                              # data/web/hub/index.json (generated, not committed)
    hits = hub.search(index, "airfoil", kind="model")     # ranked items
    item = hub.get(index, "geometry:f1_car")
    item["code"], item["run"]["stdout"], item["files"], item["links"]["guide"]

    python -m pinneapple_hub.library build              # regenerate index.json from the sources
    python -m pinneapple_hub.library run [--stale]      # run every snippet, record its output (snippet_runs/)
    python -m pinneapple_hub.library check              # index is current, every run ok, files and links exist

An item: ``id`` (``<kind-prefix>:<name>``), ``kind`` (tool, model, geometry, dataset, flow, example; tools have a
``subkind`` and one of 20 task ``category`` slugs, :mod:`.categories`), ``title``, ``summary``, ``description``,
``domain`` (``pinneapple_lab.manifest.domain_of``), ``tags``,
``quality`` (a Lab A-F seal with its six-stage trust card, a zoo decision, or None), ``code`` (the snippet),
``run`` (its recorded run: ok, stdout, seconds, figures), ``results`` (measured numbers and figures of the source),
``examples`` (ids of related items), ``files`` (repo paths and download URLs), ``links`` (guide, reference, source,
Flow Studio). Where each kind comes from: :mod:`.sources`. "Create your own": :mod:`.compose`. Saved items:
:mod:`.myhub`. Snippet runner: :mod:`.sandbox`.

``index.json`` is derived and git-ignored: :func:`load_index` builds and caches it on first use, the Hub image
builds it at ``docker build``. The recorded runs it attaches are committed, one file per item under
``snippet_runs/``, so two pull requests that record different snippets never touch the same file.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .categories import table
from .sandbox import code_hash, run_snippet

def _repository_root() -> Path:
    """The checkout this package belongs to.

    ``parents[2]`` is right when the package is imported from the repository. When it is imported
    from an install (an editable install still points here, a plain ``pip install .`` copies the
    files into site-packages), that path is site-packages, which has no ``data/web/hub`` and no
    ``examples``, and every caller that reads the repository's own files fails with a
    ``FileNotFoundError`` several frames away from the cause. So: take the nearest ancestor that
    looks like the repository, else the current directory when that looks like it, else the
    computed path (and let the caller's error name the real problem).
    """
    def looks_like_repo(p: Path) -> bool:
        return (p / "pyproject.toml").is_file() and (p / "pinneapple_hub").is_dir()

    candidate = Path(__file__).resolve().parents[2]
    for p in (candidate, *candidate.parents):
        if looks_like_repo(p):
            return p
    cwd = Path.cwd()
    if looks_like_repo(cwd):
        return cwd
    return candidate


ROOT = _repository_root()
WEB = ROOT / "data" / "web" / "hub"
INDEX = WEB / "index.json"
RUNS = WEB / "snippet_runs"
FIGURES = WEB / "figures"
SCHEMA = "pinneapple-hub/1"
KINDS = {"tool": "Tools", "model": "Trained models", "geometry": "Geometries", "dataset": "Datasets",
         "flow": "Flows", "example": "Examples"}
LETTERS = ("A", "B", "C", "D", "E", "F")
CATEGORY_TITLES = {c["slug"]: c["title"] for c in table()}
SLOW_S = 90.0                     # snippets recorded slower than this run in the test only with PINNEAPPLE_HUB_SLOW=1


def collect(only: list[str] | None = None) -> list[dict[str, Any]]:
    """Items from every source (:data:`.sources.COLLECTORS`), ids unique, sorted by kind then title."""
    from .categories import category_of
    from .sources import COLLECTORS
    items: dict[str, dict[str, Any]] = {}
    for name, fn in COLLECTORS.items():
        if only and name not in only:
            continue
        for it in fn():
            items.setdefault(it["id"], it)
    for it in list(items.values()):                    # related items link both ways
        for rid in it.get("examples", []):
            other = items.get(rid)
            if other is not None and it["id"] not in other["examples"]:
                other["examples"].append(it["id"])
    for it in items.values():
        if it["kind"] == "tool":
            it["category"] = category_of(it)
    order = list(KINDS)
    return sorted(items.values(), key=lambda i: (order.index(i["kind"]), i.get("subkind", ""), i["title"].lower()))


def run_file(item_id: str, runs: Path = RUNS) -> Path:
    """The file holding the recorded run of one item: ``snippet_runs/<id with unsafe characters as _>.json``."""
    return Path(runs) / (re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id) + ".json")


def read_runs(path: Path = RUNS) -> dict[str, Any]:
    """Every recorded run, by item id (each file under ``snippet_runs/`` carries its item's ``id``)."""
    out: dict[str, Any] = {}
    for f in sorted(Path(path).glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        out[rec.pop("id")] = rec
    return out


def write_run(item_id: str, rec: dict[str, Any], runs: Path = RUNS) -> Path:
    f = run_file(item_id, runs)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"id": item_id, **rec}, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return f


def attach_runs(items: list[dict[str, Any]], runs: dict[str, Any]) -> None:
    """Set ``run`` on each item: the recorded run of its current snippet (same code hash), else the module
    reference's own run of a recipe, else ``{"status": "not run"}``."""
    for it in items:
        h = code_hash(it["code"])
        rec = runs.get(it["id"])
        ref = (it.get("extra") or {}).get("reference_run")
        if rec and rec.get("code_hash") == h:
            it["run"] = {**rec, "status": "ok" if rec.get("ok") else "failed"}
        elif ref and ref.get("code_hash") == h:
            it["run"] = {"ok": ref["ok"], "stdout": ref["stdout"], "seconds": ref["seconds"], "code_hash": h,
                         "figures": ref["figures"], "status": "ok" if ref["ok"] else "failed",
                         "source": "scripts/reference/run_examples.py"}
        else:
            it["run"] = {"status": "not run", "code_hash": h}


def facets(items: list[dict[str, Any]]) -> dict[str, Any]:
    def count(key):
        out: dict[str, int] = {}
        for it in items:
            v = key(it)
            if v:
                out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
    return {"kind": {k: sum(i["kind"] == k for i in items) for k in KINDS},
            "subkind": count(lambda i: i.get("subkind")), "domain": count(lambda i: i.get("domain")),
            "category": {c["slug"]: n for c in table() if (n := sum(i.get("category") == c["slug"] for i in items))},
            "quality": count(quality_key)}


def quality_key(it: dict[str, Any]) -> str:
    """The filter value of an item's quality: a Lab letter, a zoo decision, or "not graded"."""
    q = it.get("quality") or {}
    if q.get("letter"):
        return q["letter"]
    if q.get("decision"):
        return q["decision"].lower()
    return "not graded"


def featured(items: list[dict[str, Any]], n: int = 8) -> list[str]:
    """Highlights, by a fixed rule: approved zoo models, A-sealed examples with figures, geometries with a
    thumbnail, then flows; each with a passing recorded run."""
    ok = [i for i in items if i.get("run", {}).get("ok")]
    picks = [i for i in ok if i["kind"] == "model" and (i.get("quality") or {}).get("decision") == "APPROVED"][:2]
    picks += [i for i in ok if i["kind"] == "example" and quality_key(i) == "A" and i["results"].get("figures")][:3]
    picks += [i for i in ok if i["kind"] == "geometry" and i.get("thumbnail")][:2]
    picks += [i for i in ok if i["kind"] == "flow"][:1]
    return [i["id"] for i in picks][:n]


def build(only: list[str] | None = None, runs: dict[str, Any] | None = None) -> dict[str, Any]:
    """The whole index (not written; see :func:`write_index`)."""
    items = collect(only)
    attach_runs(items, read_runs() if runs is None else runs)
    return {"schema": SCHEMA, "generator": "python -m pinneapple_hub.library build",
            "counts": {k: sum(i["kind"] == k for i in items) for k in KINDS}, "kinds": KINDS,
            "categories": table(), "facets": facets(items), "featured": featured(items), "items": items}


def write_index(index: dict[str, Any], path: Path = INDEX) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=1, ensure_ascii=False, sort_keys=False, default=str) + "\n")
    return path


def load_index(path: Path | str = INDEX, *, rebuild: bool = False) -> dict[str, Any]:
    """The index at ``path``. The default index is a cache: built from the sources (and written) when it is missing
    or ``rebuild`` is set; rebuild it after changing a source with ``python -m pinneapple_hub.library build``."""
    path = Path(path)
    if path == INDEX and (rebuild or not path.is_file()):
        write_index(build(), path)
    return json.loads(path.read_text())


def get(index: dict[str, Any], item_id: str) -> dict[str, Any]:
    for it in index["items"]:
        if it["id"] == item_id:
            return it
    raise KeyError(item_id)


def _words(s: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w]


def score(it: dict[str, Any], query: str) -> float:
    """Relevance of an item to a query; 0 = no match. Every query word must match: title weighs most, then name
    and tags, then summary, domain and description."""
    q = _words(query)
    if not q:
        return 1.0
    fields = [(it["title"], 8.0), (it.get("name", ""), 5.0), (" ".join(it.get("tags", [])), 4.0),
              (it.get("summary", ""), 2.0), (it.get("domain", ""), 2.0), (CATEGORY_TITLES.get(it.get("category"), ""), 2.0), (it.get("subkind", "") or it["kind"], 2.0),
              ((it.get("description") or "")[:2000], 1.0)]
    total = 0.0
    for w in q:
        best = 0.0
        for text, weight in fields:
            words = _words(text)
            if w in words:
                best = max(best, weight)
            elif any(x.startswith(w) for x in words):
                best = max(best, 0.7 * weight)
            elif len(w) > 3 and any(w in x for x in words):
                best = max(best, 0.3 * weight)
        if best == 0.0:
            return 0.0
        total += best
    return total


def search(index: dict[str, Any], query: str = "", *, kind: str | None = None, domain: str | None = None,
           quality: str | None = None, subkind: str | None = None, category: str | None = None,
           limit: int | None = None) -> list[dict[str, Any]]:
    """Items matching ``query`` and the filters (``quality``: a letter, a comma list of letters, a zoo decision in
    lower case, or "not graded"), best first; without a query, in index order."""
    quals = {q.strip() for q in (quality or "").split(",") if q.strip()}
    rows = []
    for k, it in enumerate(index["items"]):
        if kind and it["kind"] != kind or subkind and it.get("subkind") != subkind:
            continue
        if category and it.get("category") != category:
            continue
        if domain and it.get("domain") != domain or quals and quality_key(it) not in quals:
            continue
        s = score(it, query)
        if s > 0:
            rows.append((-s, k, it))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows[:limit]]


def record_runs(items: list[dict[str, Any]], *, stale_only: bool = False, timeout: float = 300.0,
                ids: list[str] | None = None, workers: int = 4, log=print) -> dict[str, Any]:
    """Run the snippets (all, the ones not recorded for their current code, or ``ids``) in ``workers`` parallel
    interpreters, save their figures under ``data/web/hub/figures`` and write one ``snippet_runs/<id>.json``
    each. Recipes the module reference already ran are reused."""
    from concurrent.futures import ThreadPoolExecutor
    runs = read_runs()
    FIGURES.mkdir(parents=True, exist_ok=True)
    todo = []
    for it in items:
        h = code_hash(it["code"])
        ref = (it.get("extra") or {}).get("reference_run")
        if ids and it["id"] not in ids:
            continue
        if not ids and ref and ref.get("code_hash") == h and ref.get("ok"):
            continue
        if stale_only and runs.get(it["id"], {}).get("code_hash") == h:
            continue
        todo.append(it)

    def one(it):
        r = run_snippet(it["code"], timeout=timeout)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", it["id"])
        for old in FIGURES.glob(f"{stem}_*.png"):
            old.unlink()
        figs = []
        for k, png in enumerate(r.pop("figures")):
            p = FIGURES / f"{stem}_{k}.png"
            p.write_bytes(png)
            figs.append(p.relative_to(ROOT).as_posix())
        runs[it["id"]] = {**r, "figures": figs, "recorded": time.strftime("%Y-%m-%d", time.gmtime()),
                          "source": "python -m pinneapple_hub.library run"}
        write_run(it["id"], runs[it["id"]])
        log(f"{'ok  ' if r['ok'] else 'FAIL'} {r['seconds']:7.1f}s {it['id']}")
        if not r["ok"]:
            log("     " + (r["stderr"].splitlines() or [""])[-1])

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo))
    return runs


def check(index: dict[str, Any]) -> list[str]:
    """Problems of an index: a snippet without a passing run of its current code, a file or link path that does
    not exist in the repository, an example id that is not an item."""
    ids = {i["id"] for i in index["items"]}
    out = []
    for it in index["items"]:
        r = it.get("run") or {}
        if r.get("code_hash") != code_hash(it["code"]) or not r.get("ok"):
            out.append(f"{it['id']}: snippet has no passing recorded run ({r.get('status', 'not run')})")
        for f in it.get("files", []):
            if f.get("path") and not (ROOT / f["path"]).exists():
                out.append(f"{it['id']}: missing file {f['path']}")
        res = it.get("results") if isinstance(it.get("results"), dict) else {}
        paths = [*res.get("figures", []), *res.get("videos", []), *r.get("figures", []), it.get("thumbnail") or ""]
        for p in paths:
            if p and not (ROOT / p).exists():
                out.append(f"{it['id']}: missing figure {p}")
        for k, v in (it.get("links") or {}).items():
            if v and not v.startswith(("http", "?")) and not (ROOT / v.split("#")[0]).exists():
                out.append(f"{it['id']}: link {k} -> {v} does not exist")
        for e in it.get("examples", []):
            if e not in ids:
                out.append(f"{it['id']}: related item {e} is not in the index")
    return out


__all__ = ["build", "collect", "write_index", "load_index", "get", "search", "score", "facets", "featured",
           "record_runs", "read_runs", "write_run", "run_file", "attach_runs", "check", "quality_key", "run_snippet",
           "code_hash", "KINDS", "INDEX", "RUNS"]
