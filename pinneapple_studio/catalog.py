"""The catalog of tools: every public class and function of the library, with its category, form, status, evidence,
Portuguese prose, navigation tags and the tools related to it in the code.

Status is earned from the repository, with the rules of the Capability Registry of the Engineering API:

* ``verified``: a test that runs by default calls it and compares with a reference (``validate()`` asserted, a
  reference / exact / analytic solution, a relative error);
* ``tested``: a test that runs by default calls it, without a reference comparison;
* ``discovered``: nothing in ``tests/`` that runs by default calls it.

A test calls a symbol when its file imports the symbol's module (or the symbol) and the test function calls the name
(a function, a class, or a method of the class). Opt-in slow tests do not count.

    cat = build()                      # or load(): cached per commit next to the inventory
    cat["tools"]["pinneapple_catalog.methods:method_status"]
"""
from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import classify as C
from . import prose as P
from . import taxonomy as T
from .inventory import EXCLUDE_RE, cache_dir, repo_root, source_key
from .inventory import load as load_inventory

SCHEMA = "pinneapple.studio.catalog/1"
REFERENCE_MARKERS = ("reference", "exact", "analytic", "rel_l2", "rel_err", "all_pass", "epanet", "published")
STATUS_RULES = {
    "discovered": "found in the code; no test that runs by default calls it",
    "tested": "a test that runs by default calls it; no reference comparison found",
    "verified": "a test that runs by default calls it and compares with a reference (validate() asserted, a "
                "reference / exact / analytic solution, a relative error)",
}
ICONS = {"executable": "play", "composite": "puzzle", "reference": "book"}


# ------------------------------------------------------------------------------------------------- tests
@dataclass
class TestFn:
    path: str
    line: int
    name: str
    calls: set[str]
    imports: set[str]          # "module" and "module:name" imported by the file
    reference: bool
    slow: bool


def _imports(tree: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module)
            out |= {f"{n.module}:{a.name}" for a in n.names}
            out |= {f"{n.module}.{a.name}" for a in n.names}          # "from pkg import module"
        elif isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.Call) and isinstance(n.args[0] if n.args else None, ast.Constant) and \
                ast.unparse(n.func) in ("importlib.import_module", "import_module", "pytest.importorskip"):
            out.add(str(n.args[0].value))
    return out


def scan_tests(root: Path) -> list[TestFn]:
    out: list[TestFn] = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError):
            continue
        imps = _imports(tree)
        lines = src.splitlines()
        module_slow = "PINNEAPPLE_SLOW" in src and "pytestmark" in src
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test"):
                continue
            calls = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    calls.add(f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else "")
                elif isinstance(sub, ast.Attribute):
                    calls.add(sub.attr)
            body = "\n".join(lines[node.lineno - 1:node.end_lineno or node.lineno])
            deco = " ".join(ast.unparse(d) for d in node.decorator_list)
            ref = ("validate" in calls and any(m in body for m in ("all_pass", '["pass"]', "rel_err", "tol"))) or \
                any(m in body for m in REFERENCE_MARKERS)
            out.append(TestFn(str(path.relative_to(root)), node.lineno, node.name, calls, imps, ref,
                              module_slow or "SLOW" in deco or "slow" in deco))
    return out


def _status(evidence: list[dict]) -> tuple[str, str]:
    default = [e for e in evidence if e["kind"] in ("test", "reference_test") and e["runs_by_default"]]
    ref = [e for e in default if e["kind"] == "reference_test"]
    if ref:
        return "verified", f"reference test {ref[0]['path']}::{ref[0]['name']}"
    if default:
        return "tested", f"exercised by {default[0]['path']}::{default[0]['name']}; no reference comparison found"
    if evidence:
        return "discovered", "referenced only by examples, snippets or slow (opt-in) tests"
    return "discovered", "found in the code; no test or example references it"


# ------------------------------------------------------------------------------------------------- examples
def scan_examples(root: Path, top_api: dict[str, Any]) -> dict[str, set[str]]:
    """``{"module:name" or "module": {example folders}}`` from the imports (and ``pp.*`` calls) of example scripts."""
    pp = {**top_api.get("lazy", {}), **top_api.get("eager", {})}
    out: dict[str, set[str]] = defaultdict(set)
    base = root / "examples"
    for path in sorted(base.rglob("*.py")) if base.is_dir() else []:
        rel = path.relative_to(base)
        if any(p.startswith(("_", ".")) or p.endswith(("_out", "_runs")) for p in rel.parts[:-1]) \
                or EXCLUDE_RE.search(str(rel)):
            continue
        folder = str(rel.parent) if len(rel.parts) > 1 else rel.stem
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0].startswith("pinneapple"):
                out[n.module].add(folder)
                for a in n.names:
                    out[f"{n.module}:{a.name}"].add(folder)
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "pp" \
                    and n.attr in pp:
                target = pp[n.attr]
                mod, _, name = target.rpartition(".")
                out[f"{mod}:{name}"].add(folder)
                out[target].add(folder)
    return out


# ------------------------------------------------------------------------------------------------- resolver
def make_resolver(inv: dict[str, Any]):
    """``resolver_for(module record)`` -> ``resolve(name)``: a class a form can build (all its required inputs have
    form types), looked up in the module itself, then its imports, then by a unique name in the library."""
    by_name: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for m, r in inv["modules"].items():
        for c in r["classes"]:
            by_name[c["name"]].append((m, c))
    memo: dict[str, dict | None] = {}

    def schema_of(mod: str, cls: dict, depth: int) -> dict | None:
        key = f"{mod}:{cls['name']}"
        if key in memo:
            return memo[key]
        memo[key] = None                                   # cycles resolve to "not a form"
        if depth > 2 or any(C.EXCEPTION_BASES.search(b) or C.BLOCK_BASES.search(b) for b in cls.get("bases") or []):
            return None
        res = resolver(inv["modules"][mod], depth + 1)
        sub = C.form(cls, res)
        if sub["required"] and any(p["default"] is None for p in sub["x-not-in-form"]):
            return None
        if any(p["default"] is None for p in sub["x-not-in-form"]):
            return None
        out = {"type": "object", "properties": sub["properties"], "required": sub["required"], "x-class": key}
        memo[key] = out
        return out

    def resolver(rec: dict, depth: int = 0):
        local = {c["name"]: c for c in rec["classes"]}
        imported = {i.split(":", 1)[1]: i.split(":", 1)[0] for i in rec.get("imports") or [] if ":" in i}

        def resolve(name: str) -> dict | None:
            if name in local:
                return schema_of(rec["module"], local[name], depth)
            if name in imported:
                for m, c in by_name.get(name, []):
                    if m == imported[name] or m.startswith(imported[name] + "."):
                        return schema_of(m, c, depth)
            cands = by_name.get(name, [])
            if len(cands) == 1:
                return schema_of(cands[0][0], cands[0][1], depth)
            return None
        return resolve

    return resolver


# ------------------------------------------------------------------------------------------------- build
def _code_hash(root: Path, rec: dict, sym: dict) -> str:
    try:
        lines = (root / rec["path"]).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    body = "\n".join(lines[sym["line"] - 1:sym.get("end") or sym["line"]])
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def build(root: Path | None = None, inv: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(root or repo_root()).resolve()
    inv = inv or load_inventory(root)
    prose = P.load(root)
    tests = scan_tests(root)
    by_call: dict[str, list[TestFn]] = defaultdict(list)
    for t in tests:
        for c in t.calls:
            by_call[c].append(t)
    examples = scan_examples(root, inv["top_api"])
    resolver = make_resolver(inv)
    pp_names = {v: k for k, v in {**inv["top_api"].get("lazy", {}), **inv["top_api"].get("eager", {})}.items()}
    # who imports what: "module:name" -> modules importing it
    importers: dict[str, set[str]] = defaultdict(set)
    for m, r in inv["modules"].items():
        for i in r.get("imports") or []:
            importers[i].add(m)

    tools: dict[str, dict[str, Any]] = {}
    for m, rec in inv["modules"].items():
        res = resolver(rec)
        pkg = m.split(".")[0]
        for kind, key in (("function", "functions"), ("class", "classes")):
            for sym in rec[key]:
                tid = f"{m}:{sym['name']}"
                cat, why = C.classify(sym, rec, kind, res)
                names = {sym["name"]} | ({*sym.get("methods", [])} if kind == "class" else set())
                ev, seen = [], set()
                for nm in names:
                    for t in by_call.get(nm, []):
                        if (m not in t.imports and tid not in t.imports and not any(
                                i == m or i.startswith(m + ":") for i in t.imports)) or (t.path, t.name) in seen:
                            continue
                        seen.add((t.path, t.name))
                        ev.append({"kind": "reference_test" if t.reference else "test", "path": t.path,
                                   "line": t.line, "name": t.name, "runs_by_default": not t.slow})
                ex = sorted(examples.get(tid, set()))
                for e in ex:
                    ev.append({"kind": "example", "path": f"examples/{e}", "line": None, "name": e,
                               "runs_by_default": True})
                status, status_reason = _status(ev)
                text = " ".join([m, sym["name"], sym.get("doc1", ""), prose.symbol(m, sym["name"])])
                tool = {
                    "id": tid, "kind": kind, "name": sym["name"], "module": m, "package": pkg,
                    "path": rec["path"], "line": sym["line"], "signature": sym["sig"],
                    "params": sym.get("params") or [], "returns": sym.get("returns", ""),
                    "bases": sym.get("bases", []), "methods": sym.get("methods", []),
                    "doc1": sym.get("doc1", ""), "doc": sym.get("doc", ""),
                    "pt": prose.symbol(m, sym["name"]), "pt_module": prose.module(m),
                    "category": cat, "category_reason": why, "icon": ICONS[cat],
                    "status": status, "status_reason": status_reason,
                    "evidence": ev[:40], "n_tests": sum(e["kind"] != "example" for e in ev), "examples": ex,
                    "pp": pp_names.get(f"{m}.{sym['name']}", ""),
                    "tasks": T.tags(text, T.TASKS), "areas": T.tags(text, T.AREAS),
                    "products": T.tags(text, T.PRODUCTS),
                    "used_by_modules": sorted(importers.get(tid, set()))[:30],
                    "flags": rec.get("flags", []),
                    "code_hash": _code_hash(root, rec, sym),
                }
                if cat == "executable":
                    target = sym
                    if kind == "class":
                        target = sym                    # the form builds the object; the run method has its own
                    tool["form"] = C.form(target, res)
                    tool["form_problems"] = C.valid_form(tool["form"])
                tools[tid] = tool
    _related(tools, inv)
    from .extras import preset_tools, science_tools
    for t in preset_tools(root) + science_tools(root):
        text = " ".join([t["name"], t["doc1"], t["doc"]])
        t.update(tasks=T.tags(text, T.TASKS), areas=T.tags(text, T.AREAS), products=T.tags(text, T.PRODUCTS))
        tools[t["id"]] = t
    counts: dict[str, Any] = {"tools": len(tools)}
    for field in ("category", "status", "kind"):
        counts[field] = dict(sorted(_count(tools.values(), field).items()))
    return {"schema": SCHEMA, "commit": inv.get("generated_from_commit", ""), "status_rules": STATUS_RULES,
            "categories": C.CATEGORIES, "tasks": T.labels(T.TASKS), "areas": T.labels(T.AREAS),
            "products": T.labels(T.PRODUCTS), "counts": counts,
            "packages": {k: {**v, "pt": prose.package(k)} for k, v in inv["packages"].items()},
            "modules": {m: {"doc1": r["doc1"], "doc": r["doc"], "path": r["path"], "pt": prose.module(m),
                            "tests": r["tests"], "flags": r["flags"], "imports": r.get("imports", [])}
                        for m, r in inv["modules"].items()},
            "tools": tools,
            "glossary": prose.glossary, "prose_unmatched": P.unmatched(prose, inv)}


def _count(items, field):
    out: dict[str, int] = defaultdict(int)
    for it in items:
        out[it[field]] += 1
    return out


def _related(tools: dict[str, dict[str, Any]], inv: dict[str, Any]) -> None:
    """Related tools from the import graph of the code, never by similarity of text: the library symbols the tool's
    module imports (``uses``) and the modules that import the tool (``used_by``)."""
    for t in tools.values():
        rec = inv["modules"][t["module"]]
        t["related"] = {"uses": [i for i in rec.get("imports") or [] if i in tools and i != t["id"]][:20],
                        "used_by": t["used_by_modules"]}


def load(root: Path | None = None, *, rebuild: bool = False) -> dict[str, Any]:
    """The catalog of ``root``, cached per commit (``cache_dir()``), like the inventory."""
    root = Path(root or repo_root()).resolve()
    key = source_key(root)
    path = cache_dir() / f"catalog-{hashlib.sha256(str(root).encode()).hexdigest()[:8]}-{key or 'nogit'}.json"
    if key and path.is_file() and not rebuild:
        return json.loads(path.read_text(encoding="utf-8"))
    cat = build(root, load_inventory(root, rebuild=rebuild))
    if key:
        path.write_text(json.dumps(cat, ensure_ascii=False, default=str), encoding="utf-8")
    return cat
