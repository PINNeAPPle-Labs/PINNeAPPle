"""Check that every ``pinneapple*`` import in examples/ and templates/ resolves.

Static (AST) check: no example is executed. For ``from pinneapple_x import a, b`` it verifies that the module
imports and that ``a`` and ``b`` exist on it (or are importable submodules); for ``import pinneapple_x`` that the
module imports. Imports guarded by ``try/except ImportError`` are skipped, since they are optional by design.
Run: ``python scripts/check_example_imports.py`` (exit code 1 lists the broken imports).
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("examples", "templates")
PREFIX = "pinneapple"


def _guarded_lines(tree: ast.AST) -> set[int]:
    """Line numbers inside a ``try`` whose handlers catch ImportError/ModuleNotFoundError/Exception."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        names = {getattr(h.type, "id", None) for h in node.handlers} | {None if h.type else "bare" for h in node.handlers}
        for h in node.handlers:
            if isinstance(h.type, ast.Tuple):
                names |= {getattr(e, "id", None) for e in h.type.elts}
        if names & {"ImportError", "ModuleNotFoundError", "Exception", "bare"}:
            for child in node.body:
                guarded.update(range(child.lineno, (child.end_lineno or child.lineno) + 1))
    return guarded


def check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{path.relative_to(ROOT)}: syntax error: {exc.msg} (line {exc.lineno})"]
    guarded = _guarded_lines(tree)
    problems: list[str] = []
    for node in ast.walk(tree):
        if getattr(node, "lineno", 0) in guarded:
            continue
        where = f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.split(".")[0].startswith(PREFIX):
            try:
                mod = importlib.import_module(node.module)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{where}: cannot import {node.module} ({type(exc).__name__}: {exc})")
                continue
            for alias in node.names:
                if alias.name == "*" or hasattr(mod, alias.name):
                    continue
                try:
                    importlib.import_module(f"{node.module}.{alias.name}")
                except Exception:  # noqa: BLE001
                    problems.append(f"{where}: {node.module} has no name '{alias.name}'")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0].startswith(PREFIX):
                    try:
                        importlib.import_module(alias.name)
                    except Exception as exc:  # noqa: BLE001
                        problems.append(f"{where}: cannot import {alias.name} ({type(exc).__name__}: {exc})")
    return problems


def check_all() -> list[str]:
    sys.path.insert(0, str(ROOT))
    problems: list[str] = []
    for d in DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            if "_runs" in path.parts or "_out" in path.parts:
                continue
            problems += check_file(path)
    return problems


if __name__ == "__main__":
    found = check_all()
    for p in found:
        print(p)
    print(f"{len(found)} broken import(s)")
    sys.exit(1 if found else 0)
