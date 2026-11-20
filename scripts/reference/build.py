"""Build the module reference at docs/org/reference/: every package, subpackage and module of the repository with
its docstring and public API (read from the source with ``ast``, nothing is imported), plus the curated
explanations and the examples run by ``run_examples.py`` (code, printed output, figures).

    python scripts/reference/run_examples.py      # run the examples (only needed when they change)
    python scripts/reference/build.py             # write docs/org/reference/{index.html, data/*.json}
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from content import GROUPS, PACKAGES, SUBMODULE_NOTES  # noqa: E402

OUT = os.path.join(ROOT, "docs", "org", "reference")
SKIP_DIRS = {"__pycache__", "node_modules", "frontend", "static", "vendor", "viewer", "tests", "deploy",
             "blender_addon", "templates"}
DOC_MAX = 6000


def _sig(fn: ast.FunctionDef | ast.AsyncFunctionDef, drop_self: bool) -> str:
    a = fn.args
    parts: list[str] = []
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for i, (arg, d) in enumerate(zip(pos, defaults, strict=True)):
        if drop_self and i == 0 and arg.arg in ("self", "cls"):
            continue
        s = arg.arg + (f": {ast.unparse(arg.annotation)}" if arg.annotation else "")
        if d is not None:
            s += f" = {ast.unparse(d)}"
        parts.append(s)
        if a.posonlyargs and arg is a.posonlyargs[-1]:
            parts.append("/")
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, d in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        s = arg.arg + (f": {ast.unparse(arg.annotation)}" if arg.annotation else "")
        if d is not None:
            s += f" = {ast.unparse(d)}"
        parts.append(s)
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    sig = f"({', '.join(parts)}){ret}"
    return sig if len(sig) < 400 else sig[:397] + "..."


def _doc(node) -> str:
    d = ast.get_docstring(node) or ""
    return d if len(d) <= DOC_MAX else d[:DOC_MAX] + "\n..."


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def parse_module(path: str) -> dict | None:
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return None
    classes, functions = [], []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and _is_public(n.name):
            methods = []
            init = None
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if m.name == "__init__":
                        init = _sig(m, True)
                    elif _is_public(m.name):
                        deco = [ast.unparse(d) for d in m.decorator_list]
                        methods.append({"name": m.name, "sig": "" if "property" in deco else _sig(m, True),
                                        "kind": "property" if "property" in deco else
                                        ("classmethod" if "classmethod" in deco else
                                         "staticmethod" if "staticmethod" in deco else "method"),
                                        "doc": (ast.get_docstring(m) or "").strip().split("\n\n")[0][:600]})
            fields = []
            if any("dataclass" in ast.unparse(d) for d in n.decorator_list):
                for m in n.body:
                    if isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name) and _is_public(m.target.id):
                        fields.append(f"{m.target.id}: {ast.unparse(m.annotation)}"
                                      + (f" = {ast.unparse(m.value)}" if m.value is not None else ""))
            classes.append({"name": n.name, "bases": [ast.unparse(b) for b in n.bases], "sig": init or "",
                            "doc": _doc(n), "methods": methods, "fields": fields[:40], "line": n.lineno})
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(n.name):
            functions.append({"name": n.name, "sig": _sig(n, False), "doc": _doc(n), "line": n.lineno})
    exports = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in n.targets):
            try:
                exports = [str(x) for x in ast.literal_eval(n.value)]
            except (ValueError, SyntaxError):
                pass
    return {"doc": _doc(tree), "classes": classes, "functions": functions, "all": exports[:400]}


def walk_package(pkg_dir: str, dotted: str) -> dict:
    """Package node: its __init__ plus children (modules and subpackages), recursively."""
    init = os.path.join(pkg_dir, "__init__.py")
    node = {"name": dotted, "kind": "package", "path": os.path.relpath(pkg_dir, ROOT), "children": []}
    node.update(parse_module(init) or {"doc": "", "classes": [], "functions": [], "all": []}) if os.path.exists(init) \
        else node.update({"doc": "", "classes": [], "functions": [], "all": []})
    for entry in sorted(os.listdir(pkg_dir)):
        p = os.path.join(pkg_dir, entry)
        if entry.startswith((".", "_")) or entry in SKIP_DIRS:
            continue
        if os.path.isdir(p) and any(f.endswith(".py") for f in os.listdir(p)):
            child = walk_package(p, f"{dotted}.{entry}")
            if child["children"] or child["classes"] or child["functions"] or child["doc"]:
                node["children"].append(child)
        elif entry.endswith(".py"):
            m = parse_module(p)
            if m is not None:
                node["children"].append({"name": f"{dotted}.{entry[:-3]}", "kind": "module",
                                         "path": os.path.relpath(p, ROOT), "children": [], **m})
    return node


def count(node: dict) -> tuple[int, int]:
    n_mod, n_sym = 1, len(node["classes"]) + len(node["functions"])
    for c in node["children"]:
        a, b = count(c)
        n_mod, n_sym = n_mod + a, n_sym + b
    return n_mod, n_sym


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).replace("``", "")
    text = re.sub(r":(class|func|mod|meth|attr):", "", text)
    return re.sub(r"`([^`]+)`", r"\1", text).strip()


def first_sentence(doc: str, limit: int = 200, name: str = "") -> str:
    """First sentence of the docstring on one line: reST markup and a leading "module.name --" removed; a heading
    that only introduces what follows ("Physics:") is joined with the next paragraph."""
    paras = [_clean(p) for p in doc.strip().split("\n\n") if p.strip()]
    paras = [p for p in paras if not re.fullmatch(r"[-=~^]{3,}", p)]
    if not paras:
        return ""
    text = paras[0]
    for nm in filter(None, (name, name.split(".")[-1])):
        text = re.sub(r"^" + re.escape(nm) + r"\s*(?:—|--|-|:)\s*", "", text)
    text = re.sub(r"^pinneapple[\w.]*\s*(?:—|--|-|:)\s+", "", text, flags=re.I)        # old package names
    if (len(text) < 20 or text.endswith(":")) and len(paras) > 1:
        text = (text.rstrip(":") + ": " if text else "") + paras[1]
    safe = re.sub(r"\b(e\.g|i\.e|etc|vs|cf)\.", lambda m: m.group(0).replace(".", "\u2024"), text)
    m = re.compile(r"(?<=[a-z0-9)\]])\.(\s|$)").search(safe, 20)
    if m:
        text = text[:m.start() + 1]
    text = text.rstrip(":")
    text = text[0].upper() + text[1:] if text else text
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def describe(node: dict) -> tuple[str, str]:
    """(summary, source): the docstring, else a curated note, else one built from what the module defines."""
    if node["doc"].strip():
        return first_sentence(node["doc"], name=node["name"]), "doc"
    if node["name"] in SUBMODULE_NOTES:
        return SUBMODULE_NOTES[node["name"]], "note"
    syms = node["classes"] + node["functions"]
    documented = [x for x in syms if x["doc"].strip()]
    if documented:
        x = documented[0]
        more = len(syms) - 1
        return (f"{x['name']}: {first_sentence(x['doc'], 160, x['name'])}" + (f" (+{more} more)" if more else "")), "auto"
    if syms:
        names = [x["name"] for x in syms]
        return "Defines " + ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "") + ".", "auto"
    if node["children"]:
        return "Contains " + ", ".join(c["name"].split(".")[-1] for c in node["children"][:5]) + ".", "auto"
    return "No public classes or functions.", "auto"


def annotate(node: dict) -> None:
    node["summary"], node["src"] = describe(node)
    for c in node["children"]:
        annotate(c)


def outline(node: dict) -> dict:
    """The navigation tree without the API bodies."""
    return {"name": node["name"], "kind": node["kind"], "summary": node["summary"], "src": node["src"],
            "children": [outline(c) for c in node["children"]]}


def _build() -> None:
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    runs = json.load(open(os.path.join(HERE, "examples_output.json")))
    from run_examples import hash_code
    packages, totals, symbols = [], [0, 0, 0], []
    for p in PACKAGES:
        pkg_dir = os.path.join(ROOT, p["name"])
        tree = walk_package(pkg_dir, p["name"]) if os.path.isdir(pkg_dir) else None
        if tree:
            annotate(tree)
        examples = []
        for k, ex in enumerate(p.get("examples", [])):
            r = runs.get(f"{p['name']}__{k}")
            stale = r is not None and r.get("code_hash") != hash_code(ex["code"])
            if r is None or not r["ok"] or stale:
                raise SystemExit(f"example {p['name']}__{k} has no fresh successful run: "
                                 f"python scripts/reference/run_examples.py {p['name']}")
            examples.append({"title": ex["title"], "code": ex["code"].strip("\n"), "output": r["stdout"],
                             "figures": [f"figures/{f}" for f in r["figures"]] if ex.get("plot") else [],
                             "seconds": r["seconds"]})
        n_mod, n_sym = count(tree) if tree else (0, 0)
        totals[0] += n_mod
        totals[1] += n_sym
        totals[2] += len(examples)
        entry = {k: p.get(k) for k in ("name", "group", "title", "tagline", "about", "use_when", "shim")}
        entry.update(examples=examples, modules=n_mod, symbols=n_sym,
                     outline=outline(tree) if tree else None)
        packages.append(entry)
        if tree:
            def collect(n):
                symbols.extend([c["name"], n["name"]] for c in n["classes"])
                symbols.extend([f["name"], n["name"]] for f in n["functions"])
                for c in n["children"]:
                    collect(c)
            collect(tree)
            json.dump(tree, open(os.path.join(OUT, "data", f"{p['name']}.json"), "w"), separators=(",", ":"))
    index = {"groups": [{"id": g, "title": t} for g, t in GROUPS], "packages": packages,
             "totals": {"modules": totals[0], "symbols": totals[1], "examples": totals[2]}, "symbols": symbols}
    tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    html = tpl.replace("/*__INDEX__*/null", json.dumps(index, separators=(",", ":")).replace("</", "<\\/"))
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(html)
    used = {f.split("/")[-1] for p in packages for e in p["examples"] for f in e["figures"]}
    for f in os.listdir(os.path.join(OUT, "figures")):
        if f not in used:
            os.remove(os.path.join(OUT, "figures", f))
    print(f"{len(packages)} packages, {totals[0]} modules, {totals[1]} public classes and functions, "
          f"{totals[2]} examples -> {os.path.relpath(OUT, ROOT)}")


def _stale_files(built: str, committed: str) -> list[str]:
    """Files under the built reference that differ from the committed one.

    Only the text artefacts are compared -- ``index.html`` and ``data/*.json``
    -- because those are what ``build.py`` derives from the source. The figures
    belong to ``run_examples.py`` and are not touched by a plain build.
    """
    stale = []
    for name in ["index.html", *[f"data/{f}" for f in sorted(os.listdir(os.path.join(built, "data")))]]:
        new_path = os.path.join(built, name)
        old_path = os.path.join(committed, name)
        new = open(new_path, encoding="utf-8").read() if os.path.isfile(new_path) else None
        old = open(old_path, encoding="utf-8").read() if os.path.isfile(old_path) else None
        if new != old:
            stale.append(name)
    for name in sorted(os.listdir(os.path.join(committed, "data"))):
        rel = f"data/{name}"
        if rel not in stale and not os.path.isfile(os.path.join(built, rel)):
            stale.append(rel)  # a module or package disappeared from the source
    return stale


def main() -> None:
    """``--check`` builds into a temporary directory and compares it with the
    committed reference instead of overwriting it, so a pull request can fail
    when the published API has drifted from the code (issue #451)."""
    global OUT
    check = "--check" in sys.argv[1:]
    committed = OUT
    if not check:
        _build()
        return
    import tempfile

    OUT = tempfile.mkdtemp(prefix="reference-check-")
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    try:
        _build()
        stale = _stale_files(OUT, committed)
    finally:
        import shutil

        shutil.rmtree(OUT, ignore_errors=True)
        OUT = committed
    if stale:
        listing = "\n".join(f"  {name}" for name in stale)
        raise SystemExit(
            f"the committed module reference is out of date ({len(stale)} file(s)):\n{listing}\n\n"
            "Run `python scripts/reference/build.py` and commit docs/org/reference/."
        )
    print("module reference is up to date with the source")


if __name__ == "__main__":
    main()
