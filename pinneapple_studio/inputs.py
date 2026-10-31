"""Example inputs for executable tools, found in the repository instead of written by hand.

A call of a tool whose arguments are all literals (numbers, text, booleans, lists, dicts, ``None``) in a test, an
example script or a Hub snippet is an example of inputs that the repository already runs. :func:`harvest` maps each
such call to the tool's parameters (positional arguments by the signature's order), keeps it when it fills the
tool's form (every required input, every value of the right kind) and records where it came from. The smoke run
then uses the first one found (``pinneapple_studio/tool_inputs.json``: ``{tool id: {"inputs": {...}, "source":
"tests/test_x.py:12"}}``); inputs written by hand in that file are kept.
"""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .inventory import EXCLUDE_RE, repo_root

INPUTS = Path(__file__).with_name("tool_inputs.json")
SOURCES = ("tests", "examples")


NUMPY_CTORS = {"array", "asarray", "linspace", "arange", "zeros", "ones", "full", "eye"}


def _literal(node: ast.AST, env: dict[str, Any] | None = None) -> tuple[bool, Any]:
    """A literal, a name bound once to a literal in the same function, a small numpy constructor of literals
    (``np.linspace(0, 1, 11)``) as a list, or a class built from keyword literals (``Windkessel(Rp=1e8, C=1e-8)``)
    as the dict of a nested form."""
    env = env or {}
    if isinstance(node, ast.Name) and node.id in env:
        return True, env[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in NUMPY_CTORS \
            and isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy") and not node.keywords:
        vals = [_literal(a, env) for a in node.args]
        if all(ok for ok, _ in vals):
            try:
                import numpy as np
                arr = getattr(np, node.func.attr)(*[v for _, v in vals])
                if getattr(arr, "size", 0) <= 10_000:
                    return True, arr.tolist()
            except Exception:  # noqa: BLE001 - not a constructor call we can evaluate
                return False, None
        return False, None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id[:1].isupper() \
            and not node.args and node.keywords and all(k.arg for k in node.keywords):
        vals = {k.arg: _literal(k.value, env) for k in node.keywords}   # Windkessel(Rp=..., C=...): a nested form
        return (True, {k: v for k, (_, v) in vals.items()}) if all(ok for ok, _ in vals.values()) else (False, None)
    if isinstance(node, (ast.List, ast.Tuple)) and env:
        vals = [_literal(e, env) for e in node.elts]
        return (True, [v for _, v in vals]) if all(ok for ok, _ in vals) else (False, None)
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None


def _bindings(scope: ast.AST) -> dict[str, Any]:
    """Names assigned exactly once in a function (or module) body, to a literal."""
    seen: dict[str, list] = defaultdict(list)
    for n in ast.walk(scope):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            seen[n.targets[0].id].append(n.value)
        elif isinstance(n, (ast.AugAssign, ast.For)) and isinstance(getattr(n, "target", None), ast.Name):
            seen[n.target.id].append(None)
    env = {}
    for name, vals in seen.items():
        if len(vals) == 1 and vals[0] is not None:
            ok, v = _literal(vals[0])
            if ok:
                env[name] = v
    return env


def _imports(tree: ast.AST) -> dict[str, str]:
    """``{local name: "module:name"}`` for ``from pinneapple... import name [as alias]``."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and n.module.split(".")[0].startswith(
                "pinneapple"):
            for a in n.names:
                out[a.asname or a.name] = f"{n.module}:{a.name}"
    return out


def calls_in(src: str) -> list[tuple[str, int, list[Any], dict[str, Any]]]:
    """``(tool id, line, args, kwargs)`` for every call of an imported library name with literal arguments only."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    names = _imports(tree)
    mods = {}
    for n in ast.walk(tree):                                  # import pinneapple_x.y as m  ->  m.f(...)
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0].startswith("pinneapple"):
                    mods[a.asname or a.name] = a.name
    raising = set()                                           # calls expected to fail: inside pytest.raises / warns
    for w in ast.walk(tree):
        if isinstance(w, (ast.With, ast.AsyncWith)) and any(
                isinstance(i.context_expr, ast.Call) and ast.unparse(i.context_expr.func).endswith(("raises", "warns"))
                for i in w.items):
            raising |= {id(c) for b in w.body for c in ast.walk(b) if isinstance(c, ast.Call)}
    scopes = [tree] + [f for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))]
    env_of: dict[int, dict[str, Any]] = {}
    for sc in scopes:                                         # innermost function wins (walked after the module)
        env = _bindings(sc) if not isinstance(sc, ast.Module) else {}
        for c in ast.walk(sc):
            if isinstance(c, ast.Call):
                env_of[id(c)] = env
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or id(n) in raising:
            continue
        env = env_of.get(id(n), {})
        f = n.func
        tid = ""
        if isinstance(f, ast.Name) and f.id in names:
            tid = names[f.id]
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in mods:
            tid = f"{mods[f.value.id]}:{f.attr}"
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in names:
            tid = f"{names[f.value.id].replace(':', '.')}:{f.attr}"   # from pinneapple_x import mod -> mod.f(...)
        if not tid or any(isinstance(a, ast.Starred) for a in n.args) or any(k.arg is None for k in n.keywords):
            continue
        args, kwargs, ok = [], {}, True
        for a in n.args:
            good, v = _literal(a, env)
            ok = ok and good
            args.append(v)
        for k in n.keywords:
            good, v = _literal(k.value, env)
            ok = ok and good
            kwargs[k.arg] = v
        if ok:
            out.append((tid, n.lineno, args, kwargs))
    return out


def _fits(schema: dict[str, Any], v: Any) -> bool:
    if v is None:
        return True
    if "anyOf" in schema:
        return any(_fits(s, v) for s in schema["anyOf"])
    if "enum" in schema:
        return v in schema["enum"]
    t = schema.get("type")
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "array":
        return isinstance(v, (list, tuple)) and (not isinstance(schema.get("items"), dict)
                                                 or all(_fits(schema["items"], x) for x in v))
    if t == "object":
        if not isinstance(v, dict):
            return False
        props = schema.get("properties")
        if props is None:
            return True
        return all(k in props and _fits(props[k], x) for k, x in v.items()) and all(
            k in v for k in schema.get("required", []))
    return True


def to_inputs(tool: dict[str, Any], args: list[Any], kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """The call's arguments as form inputs, or ``None`` when they do not fill the form."""
    form = tool.get("form") or {}
    props = form.get("properties", {})
    names = [p["name"] for p in tool.get("params") or [] if p["kind"] == "positional"]
    if len(args) > len(names):
        return None
    vals = dict(zip(names, args, strict=False))
    if set(vals) & set(kwargs):
        return None
    vals.update(kwargs)
    if any(k not in props for k in vals) or any(vals.get(k) is None for k in form.get("required", [])):
        return None
    if not all(_fits(props[k], v) for k, v in vals.items()):
        return None
    return json.loads(json.dumps(vals, default=list))         # tuples -> lists


def harvest(catalog: dict[str, Any], root: Path | None = None, hub: bool = True) -> dict[str, dict[str, Any]]:
    """``{tool id: {"inputs", "source"}}`` for executable tools with required inputs, from tests, examples and the
    Hub's snippets (the first call that fills the form, tests first)."""
    root = Path(root or repo_root())
    tools = catalog["tools"]
    from .coverage import declared_reason
    want = {i for i, t in tools.items() if t["category"] == "executable" and (t.get("form") or {}).get("required")
            and not declared_reason(t, {})}
    found: dict[str, dict[str, Any]] = {}
    texts: list[tuple[str, str]] = []
    for base in SOURCES:
        for p in sorted((root / base).rglob("*.py")):
            rel = str(p.relative_to(root))
            if EXCLUDE_RE.search(rel) or "/_" in rel.replace(base + "/", "/", 1):
                continue
            try:
                texts.append((rel, p.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    if hub:
        try:
            from pinneapple_hub import library as L
            for it in L.collect():
                if it.get("code"):
                    texts.append((f"hub:{it['id']}", it["code"]))
        except Exception:  # noqa: BLE001 - the Hub is an optional source
            pass
    by_tool: dict[str, list] = defaultdict(list)
    for rel, src in texts:
        for tid, line, args, kwargs in calls_in(src):
            if tid in want:
                by_tool[tid].append((rel, line, args, kwargs))
    for tid, calls in by_tool.items():
        for rel, line, args, kwargs in calls:
            inputs = to_inputs(tools[tid], args, kwargs)
            if inputs is not None:
                found[tid] = {"inputs": inputs, "source": f"{rel}:{line}" if not rel.startswith("hub:") else rel}
                break
    return found


def write(catalog: dict[str, Any], root: Path | None = None) -> dict[str, int]:
    """Merge harvested inputs into ``tool_inputs.json`` (entries already there are kept)."""
    old = json.loads(INPUTS.read_text(encoding="utf-8")) if INPUTS.is_file() else {}
    new = harvest(catalog, root)
    merged = {**new, **old}
    INPUTS.write_text(json.dumps(dict(sorted(merged.items())), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"harvested": len(new), "kept": len(old), "total": len(merged)}
