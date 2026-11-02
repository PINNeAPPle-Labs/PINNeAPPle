"""The frozen snapshot of the core API (issue #1128): every public name of the core with its signature.

``api/core.json`` (committed) is what the core promises. A test regenerates it and fails when the code no longer
matches, so a change to a core signature is always a visible diff in the pull request, reviewed against the
deprecation policy (``docs/api_stability.md``): additions are free, removals and changes need a deprecation.

The snapshot is **static** (read from the source with :mod:`ast`, nothing is imported): it does not depend on which
optional dependencies are installed, so it is the same on every machine.

    python -m pinneapple_stability.api --write     # after an intended change
    python -m pinneapple_stability.api --check     # what the test does
    python -m pinneapple_stability.api --diff OLD.json   # added / removed / changed against an older snapshot
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

from . import CORE, CORE_PP

__all__ = ["snapshot", "diff", "SNAPSHOT_PATH", "repo_root"]

SNAPSHOT_PATH = Path("api") / "core.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module_files(root: Path, prefix: str) -> list[tuple[str, Path]]:
    base = root / Path(*prefix.split("."))
    out = []
    if (base.with_suffix(".py")).is_file():
        out.append((prefix, base.with_suffix(".py")))
    if base.is_dir():
        for f in sorted(base.rglob("*.py")):
            rel = f.relative_to(root).with_suffix("")
            parts = list(rel.parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if any(p.startswith("_") for p in parts[1:]) or "tests" in parts or "__pycache__" in f.parts:
                continue
            out.append((".".join(parts), f))
    return out


def _sig(fn: ast.FunctionDef | ast.AsyncFunctionDef, drop_self: bool = False) -> str:
    a = fn.args
    args = list(a.posonlyargs) + list(a.args)
    if drop_self and args and args[0].arg in ("self", "cls"):
        args = args[1:]
    defaults = [None] * (len(list(a.posonlyargs) + list(a.args)) - len(a.defaults)) + list(a.defaults)
    if drop_self and len(defaults) > len(args):
        defaults = defaults[1:]
    parts = []
    for arg, d in zip(args, defaults, strict=True):
        parts.append(arg.arg + (f"={ast.unparse(d)}" if d is not None else ""))
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, d in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        parts.append(arg.arg + (f"={ast.unparse(d)}" if d is not None else ""))
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return f"({', '.join(parts)})"


def _public(name: str) -> bool:
    return not name.startswith("_")


def _module_api(path: Path, module: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public(node.name):
            out[node.name] = "def" + _sig(node)
        elif isinstance(node, ast.ClassDef) and _public(node.name):
            methods = {}
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and (_public(b.name) or b.name == "__init__"):
                    methods[b.name] = _sig(b, drop_self=True)
            out[node.name] = {"class": [ast.unparse(x) for x in node.bases], "methods": methods}
        elif isinstance(node, ast.ImportFrom) and path.name == "__init__.py" and node.module is not None or (
                isinstance(node, ast.ImportFrom) and path.name == "__init__.py" and node.level):
            src = "." * node.level + (node.module or "")
            for al in node.names:
                name = al.asname or al.name
                if _public(name) and al.name != "*":
                    out[name] = f"from {src} import {al.name}"
    return out


def _facade(root: Path) -> dict[str, str]:
    """Where each core ``pp.*`` name comes from, read from ``pinneapple/__init__.py`` (imports, the lazy-attribute and
    submodule tables, definitions)."""
    tree = ast.parse((root / "pinneapple" / "__init__.py").read_text(encoding="utf-8"))
    where: dict[str, str] = {}
    top: set[int] = set()                                # module-level statements, also inside if / try blocks

    def collect(stmts):
        for st in stmts:
            top.add(id(st))
            if isinstance(st, ast.If):
                collect(st.body)
                collect(st.orelse)
            elif isinstance(st, ast.Try):
                collect(st.body)
                for h in st.handlers:
                    collect(h.body)
                collect(st.orelse)
                collect(st.finalbody)
    collect(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            src = "." * node.level + (node.module or "")
            for al in node.names:
                where.setdefault(al.asname or al.name, f"from {src} import {al.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and id(node) in top:
            where.setdefault(node.name, "def" + _sig(node))
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            target = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ""
            if target in ("_LAZY_ATTRS", "_SUBMODULES"):
                for k, v in zip(node.value.keys, node.value.values, strict=True):
                    if isinstance(k, ast.Constant):
                        where[k.value] = f"lazy {ast.unparse(v)}"
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            where.setdefault(node.targets[0].id, "value")
    return {n: where.get(n, "MISSING") for n in CORE_PP}


def snapshot(root: Path | None = None) -> dict[str, Any]:
    """``{"pp": {name: origin}, "modules": {module: {name: signature or class}}}`` for the whole core."""
    root = root or repo_root()
    modules: dict[str, Any] = {}
    for prefix in CORE:
        if prefix == "pinneapple" or prefix == "pinneapple_stability":
            continue
        for mod, f in _module_files(root, prefix):
            api = _module_api(f, mod)
            if api:
                modules[mod] = dict(sorted(api.items()))
    return {"schema": "pinneapple-core-api/v1", "pp": _facade(root), "modules": dict(sorted(modules.items()))}


def _flat(snap: dict[str, Any]) -> dict[str, str]:
    out = {f"pp.{k}": v for k, v in snap.get("pp", {}).items()}
    for mod, names in snap.get("modules", {}).items():
        for name, v in names.items():
            if isinstance(v, dict):
                out[f"{mod}.{name}"] = "class(" + ", ".join(v["class"]) + ")"
                for m, s in v["methods"].items():
                    out[f"{mod}.{name}.{m}"] = s
            else:
                out[f"{mod}.{name}"] = v
    return out


def diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, list]:
    """Names added, removed and changed (with the old and new signature) between two snapshots."""
    a, b = _flat(old), _flat(new)
    return {"added": sorted(set(b) - set(a)), "removed": sorted(set(a) - set(b)),
            "changed": sorted([k, a[k], b[k]] for k in set(a) & set(b) if a[k] != b[k])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The frozen snapshot of the core API (api/core.json)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--diff", metavar="OLD_JSON")
    a = ap.parse_args(argv)
    root = repo_root()
    snap = snapshot(root)
    path = root / SNAPSHOT_PATH
    text = json.dumps(snap, indent=1, sort_keys=False) + "\n"
    if a.write:
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path} ({sum(len(v) for v in snap['modules'].values())} names in {len(snap['modules'])} "
              f"modules, {len(snap['pp'])} pp names)")
        return 0
    if a.check:
        old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        d = diff(old, snap)
        if any(d.values()):
            print(json.dumps(d, indent=1))
            print("the core API changed: review it against docs/api_stability.md (removals and changes need a "
                  "deprecation), then run `python -m pinneapple_stability.api --write` and commit api/core.json")
            return 1
        print("core API matches api/core.json")
        return 0
    print(json.dumps(diff(json.loads(Path(a.diff).read_text(encoding="utf-8")), snap), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
