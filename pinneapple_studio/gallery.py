"""The gallery: every example, paper reproduction and app of the repository, with its data sheet and whether it runs.

An entry's data sheet comes from what the repository already records, never from a guess:

* ``experiment.json`` (title, description, domain, methods, modules, parameters, metrics with their measured values,
  checks, references, links);
* the example flows of :mod:`pinneapple_flow.examples` (flows, their quick configurations, the inventory's
  ``needs`` / ``not_convertible`` reasons and the measured runs with seconds and comparisons);
* the README and the Portuguese prose of the map;
* the requirements read from its scripts (:mod:`pinneapple_studio.requirements`), checked on this machine.

``state`` says how the Studio runs it: ``flow`` (an example flow, full or quick), ``script`` (its scripts, no flow),
``app`` (a web app: the command that starts it) or ``blocked`` (a requirement is missing here: ``blocked_by`` lists
each with its fix). A cost is shown only when a run was measured (``measured.seconds``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import prose as P
from . import requirements as R
from .inventory import EXCLUDE_RE, cache_dir, readme_summary, repo_root, source_key


def _experiment(folder: Path) -> dict[str, Any]:
    p = folder / "experiment.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except json.JSONDecodeError:
        return {}


def _scripts(folder: Path, exp: dict[str, Any]) -> list[Path]:
    named = [folder / s for s in exp.get("scripts") or [] if (folder / s).is_file()]
    if named:
        return named
    return sorted(p for p in folder.glob("*.py") if not p.name.startswith("_"))


def _flow_rows() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    try:
        from pinneapple_flow import examples as FX
    except ImportError:                                   # flow extra missing: the gallery still lists scripts
        return {}, [], {}
    return FX.inventory(), FX.flows(), FX.measured()


def _entry(root: Path, folder: Path, kind: str, prose: P.Prose, finv: dict, flows: list, meas: dict,
           key: str) -> dict[str, Any]:
    exp = _experiment(folder)
    rd = readme_summary(folder, root)
    scripts = _scripts(folder, exp)
    reqs = R.check(R.requirements(scripts, root))
    fl = [f for f in flows if f["example"] == key]
    row = (finv.get(key) or {}) if fl or key in finv else {}
    runs = {f["name"]: meas.get(f["name"], {}) for f in fl}
    measured = [{"flow": n, "configuration": c, "seconds": r.get("seconds"), "status": r.get("status"),
                 "date": r.get("date"), "commit": r.get("commit"), "matched": r.get("matched"),
                 "comparisons": r.get("comparisons", [])}
                for n, by in runs.items() for c, r in by.items()]
    missing = [r for r in reqs if not r["ok"]]
    if row.get("needs") and not fl:
        missing.append({"kind": "declared", "name": row["needs"], "fix": row.get("reason", ""), "ok": False})
    if missing:
        state = "blocked"
    elif fl:
        state = "flow"
    elif scripts:
        state = "script"
    else:
        state = "none"
    text_pt = prose.folders.get("examples", {}).get(folder.name, "") if kind == "example" else ""
    if kind == "reproduction":
        text_pt = next((r.get("resultado", "") for r in prose.reproductions
                        if r.get("caminho") == str(folder.relative_to(root))), "")
    return {
        "id": f"{kind}:{key}", "kind": kind, "key": key, "path": str(folder.relative_to(root)),
        "title": exp.get("title") or rd["title"] or folder.name, "description": exp.get("description") or rd["summary"],
        "pt": text_pt, "domain": exp.get("domain") or row.get("domain", ""), "methods": exp.get("methods", []),
        "modules": exp.get("modules", []), "params": exp.get("params", {}), "metrics": exp.get("metrics", []),
        "checks": exp.get("checks", []), "references": exp.get("references", []), "links": exp.get("links", {}),
        "curation": exp.get("status", ""), "readme": rd["readme"],
        "scripts": [str(p.relative_to(root)) for p in scripts],
        "flows": [{"name": f["name"], "title": f["title"], "quick": f["quick"], "nodes": f["nodes"]} for f in fl],
        "flow_status": row.get("status") or ("" if not fl else "has-flow"), "flow_reason": row.get("reason", ""),
        "notes": row.get("notes", ""), "requirements": reqs, "blocked_by": missing, "state": state,
        "measured": measured,
    }


def entries(root: Path | None = None) -> list[dict[str, Any]]:
    """Every example folder, every reproduction (``examples/reproductions/*``) and every app."""
    root = Path(root or repo_root())
    prose = P.load(root)
    finv, flows, meas = _flow_rows()
    out = []
    ex = root / "examples"
    for d in sorted(ex.iterdir()) if ex.is_dir() else []:
        if not d.is_dir() or d.name.startswith((".", "_")) or d.name in ("scratch",) or EXCLUDE_RE.search(d.name) \
                or d.name.endswith(("_out", "_runs")) or d.name.startswith("results"):
            continue
        if d.name == "reproductions":
            for r in sorted(d.iterdir()):
                if r.is_dir() and not r.name.startswith((".", "_")):
                    out.append(_entry(root, r, "reproduction", prose, finv, flows, meas, f"reproductions/{r.name}"))
            continue
        out.append(_entry(root, d, "example", prose, finv, flows, meas, d.name))
    apps = root / "apps"
    for d in sorted(apps.iterdir()) if apps.is_dir() else []:
        if not d.is_dir() or d.name.startswith((".", "_")) or EXCLUDE_RE.search(d.name):
            continue
        rd = readme_summary(d, root)
        out.append({"id": f"app:{d.name}", "kind": "app", "key": d.name, "path": str(d.relative_to(root)),
                    "title": rd["title"] or d.name, "description": rd["summary"], "readme": rd["readme"],
                    "pt": prose.folders.get("apps", {}).get(d.name, ""), "state": "app",
                    "start": _app_command(d, root), "requirements": [], "blocked_by": [], "flows": [],
                    "measured": [], "scripts": []})
    return out


def _app_command(d: Path, root: Path) -> str:
    rel = d.relative_to(root)
    if (d / "docker-compose.yml").is_file():
        return f"docker compose -f {rel}/docker-compose.yml up --build"
    if (d / "Dockerfile").is_file():
        return f"docker build -f {rel}/Dockerfile -t {d.name} . && docker run -p 8000:8000 {d.name}"
    return f"see {rel}/README.md"


def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per kind and state."""
    out: dict[str, Any] = {"total": len(items), "by_kind": {}, "by_state": {}}
    for it in items:
        out["by_kind"][it["kind"]] = out["by_kind"].get(it["kind"], 0) + 1
        out["by_state"][it["state"]] = out["by_state"].get(it["state"], 0) + 1
    return out


def load(root: Path | None = None, *, rebuild: bool = False) -> list[dict[str, Any]]:
    """:func:`entries`, cached per commit like the catalog (requirements are checked again on every load)."""
    root = Path(root or repo_root()).resolve()
    key = source_key(root)
    path = cache_dir() / f"gallery-{hashlib.sha256(str(root).encode()).hexdigest()[:8]}-{key or 'nogit'}.json"
    if key and path.is_file() and not rebuild:
        items = json.loads(path.read_text(encoding="utf-8"))
        for it in items:
            if it["kind"] != "app":
                it["requirements"] = R.check(it["requirements"])
                declared = [r for r in it["blocked_by"] if r["kind"] == "declared"]
                it["blocked_by"] = [r for r in it["requirements"] if not r["ok"]] + declared
                if it["blocked_by"]:
                    it["state"] = "blocked"
                elif it["state"] == "blocked":
                    it["state"] = "flow" if it["flows"] else "script" if it["scripts"] else "none"
        return items
    items = entries(root)
    if key:
        path.write_text(json.dumps(items, ensure_ascii=False, default=str), encoding="utf-8")
    return items
