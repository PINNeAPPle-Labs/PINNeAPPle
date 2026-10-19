"""Library-call nodes generated from the public API (``auto.<module>.<function>``).

:func:`generate` imports every library module that no curated node reaches, takes its public functions (``__all__``
or the public names defined in it) and keeps those a form can call safely:

* every required argument has a simple type (number, text, choice, flag, list, mapping) read from its annotation or
  default, so the parameter form is generated like the curated ``lib.*`` nodes;
* no argument names a path, URL, host, command, expression or credential, and the function body does not evaluate
  text, start processes, open sockets or write files (``eval``, ``sympify``, ``subprocess``, ``open(``, ``.save``...):
  a flow document can only choose values, never code or files;
* the name is not a server, CLI, upload / download or install entry point.

The result is committed as ``autocalls.json`` (callables with their parameter forms, plus a reason for each module that
has none) so the catalog and the coverage table load without importing those modules; a module is imported when one
of its nodes runs. ``python -m pinneapple_flow autocalls`` regenerates it, ``tests/test_flow_library_nodes.py`` fails
when a public module is neither covered by a node nor given a reason.
"""
from __future__ import annotations

import importlib
import inspect
import json
import re
import typing
from pathlib import Path
from typing import Any

from .registry import Param

DATA = Path(__file__).with_name("autocalls.json")

RISKY_PARAM = re.compile(r"(^|_)(path|paths|file|files|filename|dir|dirs|directory|folder|url|uri|host|port|root|"
                         r"cmd|command|shell|exec|code|script|expr|expression|password|token|secret|db|database|"
                         r"out|output|outdir|dest|destination|save|cache|case|workdir|cwd|location|fname)($|_)", re.I)
RISKY_NAME = re.compile(r"(^|_)(main|serve|server|launch|app|cli|upload|download|push|pull|delete|remove|install|"
                        r"login|connect|publish|deploy|write|save|export|dump|open|load|read|fetch|cache|clear|"
                        r"register|set|patch|monkey|daemon|watch|loop)($|_)", re.I)
RISKY_SOURCE = re.compile(r"\b(eval|exec|compile|sympify|parse_expr|lambdify|subprocess|Popen|os\.system|os\.remove|"
                          r"shutil|socket|requests\.|urllib|httpx|open\(|\.write\(|\.save\(|savefig|to_csv|"
                          r"to_parquet|np\.save|torch\.save|pickle|joblib|input\(|__import__|importlib)\b")
MAX_PER_MODULE = 40


def _public(mod) -> list[tuple[str, Any]]:
    names = getattr(mod, "__all__", None)
    out = []
    for name in (names if names else dir(mod)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(mod, name, None)
        except Exception:  # noqa: BLE001 - lazy attributes that need an optional dependency
            continue
        if inspect.isfunction(obj) and (names or (obj.__module__ or "").startswith(mod.__name__)):
            out.append((name, obj))
    return out


def _params(fn) -> tuple[list[dict[str, Any]] | None, str]:
    from .libcalls import _kind_of
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None, "no signature"
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - unresolvable annotations: use the raw ones
        hints = {}
    params = []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if RISKY_PARAM.search(name):
            return None, f"argument {name!r} names a file, address or code"
        default = None if p.default is inspect.Parameter.empty else p.default
        if callable(default) or (default is not None and not isinstance(default, (int, float, str, bool, list,
                                                                                   tuple, dict))):
            if p.default is inspect.Parameter.empty:
                return None, f"argument {name!r} takes an object"
            continue                                   # an object default: keep the default, no form field
        kind, choices = _kind_of(hints.get(name, p.annotation), default)
        if not kind:
            if p.default is inspect.Parameter.empty:
                ann = hints.get(name, p.annotation)
                tname = getattr(ann, "__name__", None) or (str(ann) if ann is not inspect.Parameter.empty else "")
                return None, f"argument {name!r} takes an object" + (f" ({tname})" if tname else "")
            continue
        if isinstance(default, tuple):
            default = list(default)
        if p.default is inspect.Parameter.empty and default is None:
            default = {"int": 1, "float": 1.0, "bool": False, "str": "", "list": [], "dict": {}}.get(kind)
            if kind == "enum":
                default = choices[0] if choices else ""
        Param(name, kind, default, choices=choices)            # the kind must be one the forms know
        try:
            json.dumps(default, allow_nan=False)
        except (TypeError, ValueError):
            if p.default is inspect.Parameter.empty:
                return None, f"argument {name!r} has no JSON value"
            continue
        params.append({"name": name, "kind": kind, "default": default, "choices": choices,
                       "required": p.default is inspect.Parameter.empty})
    return params, ""


def scan(module: str) -> tuple[list[dict[str, Any]], str]:
    """Node entries for one module, or the reason it has none."""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - optional dependencies: the reason says which
        return [], f"not importable here ({type(exc).__name__}: {str(exc)[:80]})"
    entries, why = [], {}
    for name, fn in _public(mod):
        if RISKY_NAME.search(name):
            why[name] = "an I/O, server or setup entry point"
            continue
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            src = ""
        if RISKY_SOURCE.search(src):
            why[name] = "evaluates text, runs processes or touches files"
            continue
        params, reason = _params(fn)
        if params is None:
            why[name] = reason
            continue
        doc = (inspect.getdoc(fn) or "").strip().split("\n\n")[0].replace("\n", " ")
        entries.append({"key": f"auto.{module.removeprefix('pinneapple_').replace('pinneapple', 'pp')}.{name}",
                        "target": f"{module}:{name}", "module": module, "title": name.replace("_", " "),
                        "description": doc[:300], "params": params})
        if len(entries) >= MAX_PER_MODULE:
            break
    if entries:
        return entries, ""
    if not why:
        classes = []
        for n in sorted(getattr(mod, "__all__", None) or dir(mod)):
            try:
                if not n.startswith("_") and inspect.isclass(getattr(mod, n, None)):
                    classes.append(n)
            except Exception:  # noqa: BLE001 - lazy attributes
                pass
        if classes:
            shown = ", ".join(classes[:5]) + (f" and {len(classes) - 5} more" if len(classes) > 5 else "")
            return [], f"no public functions: its API is classes ({shown}) that Python code builds and composes"
        return [], "no public functions: constants and re-exports only"
    counts: dict[str, int] = {}
    for r in why.values():
        r = re.sub(r"argument '[^']+' ", "argument ", r)
        counts[r] = counts.get(r, 0) + 1
    return [], "no form-callable function: " + "; ".join(f"{n} with {r}" for r, n in sorted(counts.items(),
                                                                                         key=lambda kv: -kv[1]))


def generate(modules: list[str] | None = None) -> dict[str, Any]:
    """Scan ``modules`` (default: every module the curated nodes and reasons leave open) and return the data."""
    from .coverage import coverage
    if modules is None:
        modules = [r["module"] for r in coverage(auto=False) if not r["nodes"] and not r["reason"]]
    calls, reasons = [], {}
    for m in modules:
        entries, reason = scan(m)
        calls += entries
        if reason:
            reasons[m] = reason
    return {"calls": calls, "reasons": reasons}


_SMOKE = r"""
import json, sys, importlib, warnings
warnings.filterwarnings("ignore")
c = json.loads(sys.argv[1])
mod, _, name = c["target"].partition(":")
fn = getattr(importlib.import_module(mod), name)
kw = {p["name"]: p["default"] for p in c["params"] if p["required"] or p["default"] is not None}
try:
    r = fn(**kw)
    print("OK", type(r).__name__)
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:120].replace("\n", " "))
"""


def smoke(calls: list[dict[str, Any]], *, timeout: float = 30.0, workers: int = 6) -> None:
    """Call each generated node's function with its form defaults, in a fresh interpreter in an empty folder, and
    record the outcome in ``entry["check"]`` (shown with the node): ``runs with its defaults`` or ``needs input``
    with the error the defaults gave."""
    import concurrent.futures as cf
    import subprocess
    import sys
    import tempfile

    def one(c):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                p = subprocess.run([sys.executable, "-c", _SMOKE, json.dumps(c)], capture_output=True, text=True,
                                   timeout=timeout, cwd=tmp)
                out = (p.stdout.strip().splitlines() or [f"ERR crash {p.stderr.strip()[-120:]}"])[-1]
            except subprocess.TimeoutExpired:
                out = f"ERR timeout after {timeout:g} s"
        return out

    with cf.ThreadPoolExecutor(workers) as ex:
        for c, out in zip(calls, ex.map(one, calls), strict=True):
            c["check"] = ("runs with its defaults (returns " + out.split()[1] + ")") if out.startswith("OK") else \
                "needs input: with its defaults, " + out[4:]


def prune(data: dict[str, Any]) -> None:
    """Drop entries whose check shows they return a function or a class (building blocks for Python code, not a
    result a node can show); a module left with none gets that as its reason."""
    keep, dropped = [], {}
    for c in data["calls"]:
        m = re.match(r"runs with its defaults \(returns (\w+)\)", c.get("check", ""))
        if m and m.group(1) in ("function", "method", "type", "builtin_function_or_method", "partial"):
            dropped.setdefault(c["module"], []).append(c["target"].split(":")[1])
        else:
            keep.append(c)
    data["calls"] = keep
    left = {c["module"] for c in keep}
    for mod, names in dropped.items():
        if mod not in left:
            data["reasons"][mod] = ("its public functions build Python callables for other code (" +
                                    ", ".join(names[:4]) + "), not results")


def load() -> dict[str, Any]:
    if not DATA.is_file():
        return {"calls": [], "reasons": {}}
    return json.loads(DATA.read_text())


def write(data: dict[str, Any]) -> None:
    DATA.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
