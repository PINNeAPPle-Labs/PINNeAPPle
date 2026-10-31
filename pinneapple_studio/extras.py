"""Tools that are not a single public symbol: the industrial presets (``pp.presets``) and the science experiments
(``pp.science``). Each is executable from a form (its documented inputs with units, defaults, sources and ranges) and
returns its own checks, which become verifications of the run.

Status follows the catalog's rules: a preset is ``tested`` by the tests parametrized over every preset name and
``verified`` when its recorded ``validation.json`` shows every reference case passing (the Engineering API's rule); a
science experiment is ``verified`` when a default test runs it and asserts ``passed`` (every comparison against its
cited reference within tolerance), ``tested`` when a test only runs it.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from .inventory import repo_root


def _num(v: Any) -> dict[str, Any]:
    if isinstance(v, bool):
        return {"type": "boolean"}
    if isinstance(v, int):
        return {"type": "integer"}
    if isinstance(v, float):
        return {"type": "number"}
    if isinstance(v, str):
        return {"type": "string"}
    if isinstance(v, list):
        return {"type": "array", "items": _num(v[0]) if v else {}}
    return {}


def _test_strings(path: Path) -> tuple[set[str], str]:
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return set(), ""
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}, src


def _science_runs(path: Path) -> tuple[set[str], str]:
    """Experiment names a test runs: the first argument of a ``.run(...)`` call, or an entry of a list a test is
    parametrized over (``FAST``)."""
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return set(), ""
    lists = {t.id: [e.value for e in n.value.elts if isinstance(e, ast.Constant)]
             for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.List)
             for t in n.targets if isinstance(t, ast.Name)}
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "run" and n.args \
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            out.add(n.args[0].value)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("parametrize") and len(n.args) >= 2 \
                and isinstance(n.args[1], ast.Name):
            out |= set(lists.get(n.args[1].id, []))
    return out, src


def preset_tools(root: Path | None = None) -> list[dict[str, Any]]:
    root = Path(root or repo_root())
    try:
        import pinneapple_presets as P
    except ImportError:
        return []
    strings, src = _test_strings(root / "tests" / "test_presets.py")
    every = 'parametrize("name", NAMES)' in src
    out = []
    for name in P.names():
        card = P.load(name).card()
        props, ev = {}, []
        for i in card["inputs"]:
            s = {**_num(i.get("default")), "default": i.get("default"), "x-unit": i.get("unit", ""),
                 "description": i.get("help") or "", "x-source": i.get("source", "")}
            if i.get("lo") is not None:
                s["minimum"] = i["lo"]
            if i.get("hi") is not None:
                s["maximum"] = i["hi"]
            props[i["name"]] = s
        if every or name in strings:
            ev.append({"kind": "test", "path": "tests/test_presets.py", "line": None,
                       "name": "parametrized over every preset" if every else "calls this preset",
                       "runs_by_default": True})
        rec = root / "examples" / "industrial_presets" / name / "results" / "validation.json"
        status, reason = ("tested", "exercised by tests/test_presets.py; no recorded reference validation") \
            if ev else ("discovered", "no test or recorded validation references it")
        if rec.is_file():
            v = json.loads(rec.read_text(encoding="utf-8"))
            rows = v.get("references", [])
            ok = bool(v.get("all_pass")) and rows and all(r.get("pass") for r in rows)
            ev.append({"kind": "recorded_validation", "path": str(rec.relative_to(root)), "line": None,
                       "name": f"{sum(bool(r.get('pass')) for r in rows)}/{len(rows)} reference cases pass",
                       "runs_by_default": True})
            if ok:
                status, reason = "verified", f"recorded validation {rec.relative_to(root)}: every reference case passes"
        out.append(_tool(f"preset:{name}", "preset", name, "pinneapple_presets", card["title"], card["problem"],
                         {"type": "object", "properties": props, "required": [], "x-not-in-form": []},
                         status, reason, ev, {"outputs": card["outputs"], "validity": card["validity"],
                                               "limitations": card["limitations"], "references": card["references"],
                                               "question": card["question"], "sector": card["sector"],
                                               "usage": card["usage"]}))
    return out


def science_tools(root: Path | None = None) -> list[dict[str, Any]]:
    root = Path(root or repo_root())
    try:
        import pinneapple_science as S
    except ImportError:
        return []
    run_names, src = _science_runs(root / "tests" / "test_science.py")
    asserts_passed = "assert r.passed" in src or "not c.passed" in src
    out = []
    for e in S.experiments():
        props = {k: {**_num(v), "default": v} for k, v in (e.get("params") or {}).items()}
        props["dim"] = {"enum": list(e.get("dims") or []), "default": (e.get("dims") or [None])[0]}
        props["quick"] = {"type": "boolean", "default": True, "description": "smaller sizes"}
        tested = e["name"] in run_names
        ev = [{"kind": "reference_test" if asserts_passed else "test", "path": "tests/test_science.py", "line": None,
               "name": "runs it and asserts every comparison passes" if asserts_passed else "runs it",
               "runs_by_default": True}] if tested else []
        status = ("verified" if asserts_passed else "tested") if tested else "discovered"
        reason = ("tests/test_science.py runs it and asserts passed (every comparison within its tolerance)"
                  if status == "verified" else "exercised by tests/test_science.py" if tested else
                  "no default test runs it")
        out.append(_tool(f"experiment:{e['name']}", "experiment", e["name"], "pinneapple_science", e["title"],
                         e["description"], {"type": "object", "properties": props, "required": [],
                                            "x-not-in-form": []}, status, reason, ev,
                         {"references": e.get("references", []), "area_path": e.get("area"),
                          "topic": e.get("topic"), "level": e.get("level"), "dims": e.get("dims")}))
    return out


def _tool(tid, kind, name, package, title, text, form, status, reason, ev, extra) -> dict[str, Any]:
    return {"id": tid, "kind": kind, "name": name, "module": package, "package": package, "path": "", "line": None,
            "signature": "", "params": [], "returns": "", "bases": [], "methods": [], "doc1": title, "doc": text,
            "pt": "", "pt_module": "", "category": "executable",
            "category_reason": f"a {kind}: runs from its documented inputs and returns its own checks",
            "icon": "play", "status": status, "status_reason": reason, "evidence": ev,
            "n_tests": sum(e["kind"] in ("test", "reference_test") for e in ev), "examples": [], "pp": "",
            "tasks": {}, "areas": {}, "products": {}, "used_by_modules": [], "flags": [], "code_hash": "",
            "form": form, "form_problems": [], "related": {"uses": [], "used_by": []}, **extra}


def run(tool: dict[str, Any], inputs: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Run a preset or an experiment; return its result and the verifications from its own checks."""
    if tool["kind"] == "preset":
        import pinneapple_presets as P
        r = P.load(tool["name"]).run(**inputs)
        ver = [{"name": c.name, "outcome": {"supports": "pass", "contradicts": "fail"}.get(c.status, "not_run"),
                "value": c.value, "reference": c.limit, "source": c.source} for c in r.trust.checks]
        return r.to_dict(), ver
    import pinneapple_science as S
    params = dict(inputs)
    dim, quick = params.pop("dim", None), bool(params.pop("quick", True))
    r = S.run(tool["name"], dim, quick=quick, **params)
    ver = [{"name": c.quantity, "outcome": "pass" if c.passed else "fail", "value": c.computed,
            "reference": c.reference, "rel_error": c.rel_error, "tol": c.tol, "unit": c.unit, "source": c.source}
           for c in r.comparisons]
    return r.to_dict(), ver
