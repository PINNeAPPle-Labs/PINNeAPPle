"""Coverage of the Studio, as tables and as a CI gate.

**Tools.** Every public symbol has a category. Every *executable* tool has a valid form and, for its current code,
one of the following:

* a smoke run that returned (:mod:`pinneapple_studio.smoke`);
* a declared reason it is not smoke-run: from a rule (its name says it serves, downloads, publishes, ...) or from
  ``pinneapple_studio/tool_reasons.json`` (``{tool id or module prefix: reason}``);
* a line in ``pinneapple_studio/coverage_baseline.txt``: the tools that had none of these when the gate was
  introduced. The baseline only shrinks: a tool that gains a run or a reason must leave it, and a new tool cannot
  enter it.

**Examples.** Every example folder and reproduction runs from the Studio, or says why not:

* it runs when its example flow has a measured run that finished (``pinneapple_flow/flows/examples/measured.json``)
  or one of its scripts has a recorded Studio run that finished (``data/studio/example_runs.json``);
* otherwise its reason is a missing requirement (with the command), the flow inventory's stated reason, or the
  baseline.

    python -m pinneapple_studio coverage              # writes docs/studio/COVERAGE.md, exits 1 on a problem
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .inventory import repo_root
from .smoke import read_runs, recorded_inputs, smoke_inputs

BASELINE = Path(__file__).with_name("coverage_baseline.txt")
REASONS = Path(__file__).with_name("tool_reasons.json")
EXAMPLE_RUNS = Path("data") / "studio" / "example_runs.json"
# names that say a call reaches outside a scratch folder: not smoke-run, by rule
SIDE_EFFECTS = re.compile(r"(^|_)(serve|server|start|launch|open|browser|download|fetch|upload|push|publish|"
                          r"deploy|install|uninstall|delete|remove|rm|kill|login|send|post|email|notify|watch|"
                          r"listen|daemon|run_forever|loop|gui|show|interactive|record|clone|sync|pull)(_|$)", re.I)


def baseline() -> set[str]:
    if not BASELINE.is_file():
        return set()
    return {ln.split("#", 1)[0].strip() for ln in BASELINE.read_text().splitlines() if ln.split("#", 1)[0].strip()}


def reasons() -> dict[str, str]:
    return json.loads(REASONS.read_text(encoding="utf-8")) if REASONS.is_file() else {}


def declared_reason(tool: dict[str, Any], table: dict[str, str]) -> str:
    if tool["id"] in table:
        return table[tool["id"]]
    for prefix, why in table.items():
        if tool["module"] == prefix or tool["module"].startswith(prefix + "."):
            return why
    if SIDE_EFFECTS.search(tool["name"]):
        return f"not smoke-run: its name ({tool['name']}) says it reaches outside a scratch folder"
    return ""


def tool_rows(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per executable tool: ``state`` is ``ran``, ``failed`` (its smoke run raised), ``reason``,
    ``baseline`` or ``missing``."""
    runs, table, base, given = read_runs(), reasons(), baseline(), recorded_inputs()
    rows = []
    for tid, t in sorted(catalog["tools"].items()):
        if t["category"] != "executable":
            continue
        r = runs.get(tid)
        current = r is not None and r.get("code_hash") == t["code_hash"] and r.get("inputs") == smoke_inputs(t, given)
        why = declared_reason(t, table)
        if current and r["ok"]:
            state, detail = "ran", f"{r.get('returned', '')} in {r.get('seconds')} s ({r.get('recorded')})"
        elif why:
            state, detail = "reason", why
        elif tid in base:
            state = "baseline"
            detail = (f"smoke run failed: {r.get('error')}" if current and not r["ok"] else
                      "no inputs recorded for its required fields" if t["form"]["required"] else
                      "not smoke-run yet")
        elif current and not r["ok"]:
            state, detail = "failed", r.get("error", "")
        else:
            state = "missing"
            detail = ("no inputs recorded for its required fields" if t["form"]["required"] and tid not in given
                      else "no smoke run of its current code")
        rows.append({"id": tid, "package": t["package"], "state": state, "detail": detail,
                     "form_problems": t.get("form_problems", [])})
    return rows


def _script_hash(root: Path, scripts: list[str]) -> str:
    h = hashlib.sha256()
    for s in scripts:
        p = root / s
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def example_rows(gallery: list[dict[str, Any]], root: Path | None = None) -> list[dict[str, Any]]:
    """One row per example and reproduction: ``runs`` (with how) or the reason it does not."""
    root = Path(root or repo_root())
    p = root / EXAMPLE_RUNS
    script_runs = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    base = baseline()
    rows = []
    for e in gallery:
        if e["kind"] not in ("example", "reproduction"):
            continue
        done = [m for m in e["measured"] if m["status"] == "done"]
        sr = script_runs.get(e["id"])
        if done:
            m = max(done, key=lambda m: m["configuration"] == "full")
            state, how = "runs", f"flow {m['flow']} ({m['configuration']}, {m['seconds']} s, commit {m['commit']})"
        elif sr and sr.get("ok") and sr.get("hash") == _script_hash(root, sr.get("scripts", [])):
            state, how = "runs", f"script {', '.join(sr['scripts'])} ({sr.get('seconds')} s, {sr.get('recorded')})"
        elif e["blocked_by"]:
            state = "needs"
            how = "; ".join(f"{r['name']}: `{r['fix']}`" for r in e["blocked_by"])
        elif e.get("flow_reason"):
            state, how = "reason", e["flow_reason"]
        elif e["id"] in base:
            state, how = "baseline", "not run from the Studio yet"
        else:
            state, how = "missing", "no finished run and no stated reason"
        rows.append({"id": e["id"], "title": e["title"], "state": state, "how": how})
    return rows


def problems(catalog: dict[str, Any], gallery: list[dict[str, Any]]) -> list[str]:
    out = []
    from .classify import CATEGORIES
    for tid, t in catalog["tools"].items():
        if t["category"] not in CATEGORIES:
            out.append(f"{tid}: no category")
    trows = tool_rows(catalog)
    erows = example_rows(gallery)
    for r in trows:
        if r["form_problems"]:
            out.append(f"{r['id']}: form problems {r['form_problems']}")
        if r["state"] in ("missing", "failed"):
            out.append(f"{r['id']}: {r['detail']} (record inputs in tool_inputs.json and run "
                       f"`python -m pinneapple_studio smoke --stale`, or state a reason in tool_reasons.json)")
    for r in erows:
        if r["state"] == "missing":
            out.append(f"{r['id']}: {r['how']}")
    base = baseline()
    live = {r["id"]: r["state"] for r in trows + erows}
    for b in sorted(base):
        if b not in live:
            out.append(f"{b}: in coverage_baseline.txt but no longer an executable tool or example; remove the line")
        elif live[b] != "baseline":
            out.append(f"{b}: in coverage_baseline.txt but now {live[b]}; remove the line (the baseline only shrinks)")
    return out


def summary(catalog: dict[str, Any], gallery: list[dict[str, Any]]) -> dict[str, Any]:
    trows, erows = tool_rows(catalog), example_rows(gallery)
    cats: dict[str, int] = {}
    for t in catalog["tools"].values():
        cats[t["category"]] = cats.get(t["category"], 0) + 1
    count = lambda rows, k: {s: sum(1 for r in rows if r[k] == s) for s in sorted({r[k] for r in rows})}  # noqa: E731
    return {"commit": catalog.get("commit", ""), "categories": cats, "executable": count(trows, "state"),
            "examples": count(erows, "state"), "examples_total": len(erows),
            "examples_running": sum(r["state"] == "runs" for r in erows)}


def markdown(catalog: dict[str, Any], gallery: list[dict[str, Any]]) -> str:
    s = summary(catalog, gallery)
    trows, erows = tool_rows(catalog), example_rows(gallery)
    c = s["categories"]
    out = ["# PINNeAPPle Studio coverage", "",
           f"Generated by `python -m pinneapple_studio coverage` at commit `{s['commit'][:10]}`. Every number below is "
           "counted from the code and the recorded runs.", "",
           "## Examples", "",
           f"**{s['examples_running']} of {s['examples_total']} examples and reproductions run from the Studio.** "
           "The others say why.", "", "| entry | state | how / why |", "|---|---|---|"]
    out += [f"| `{r['id']}` {r['title']} | {r['state']} | {r['how']} |" for r in erows]
    out += ["", "## Tools", "",
            f"**{c.get('executable', 0)} executable, {c.get('composite', 0)} composite, {c.get('reference', 0)} "
            "reference.**", "",
            "Executable tools by smoke-run state: " + ", ".join(f"{k} {v}" for k, v in s["executable"].items()) + ".",
            "", "`ran`: called from its form and returned (not a physics check: see the status from the tests). "
            "`reason`: not smoke-run, with the reason. `baseline`: not run yet, listed in "
            "`pinneapple_studio/coverage_baseline.txt`, which only shrinks.", "",
            "| tool | state | detail |", "|---|---|---|"]
    out += [f"| `{r['id']}` | {r['state']} | {r['detail']} |" for r in trows]
    return "\n".join(out) + "\n"


def write_baseline(catalog: dict[str, Any], gallery: list[dict[str, Any]]) -> int:
    """Put every tool and example that is ``missing`` or ``failed`` today in the baseline (used once, when the gate
    is introduced; afterwards the baseline only shrinks)."""
    rows = [r["id"] for r in tool_rows(catalog) if r["state"] in ("missing", "failed")]
    rows += [r["id"] for r in example_rows(gallery) if r["state"] == "missing"]
    keep = sorted(baseline() | set(rows))
    BASELINE.write_text("# Executable tools and examples with neither a run nor a reason when the coverage gate was\n"
                        "# introduced. Only remove lines (python -m pinneapple_studio coverage says which).\n"
                        + "\n".join(keep) + "\n")
    return len(keep)


def record_examples(gallery: list[dict[str, Any]], *, ids: list[str] | None = None, timeout: float = 900,
                    log=print) -> dict[str, Any]:
    """Run the scripts of the examples that have no flow (``state == "script"``), each in its own interpreter with
    the job environment, and record the outcome per entry with a hash of its scripts."""
    import subprocess
    import sys
    import time

    from .runner import clean_env
    root = repo_root()
    p = root / EXAMPLE_RUNS
    runs = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    env = clean_env({"PYTHONPATH": str(root), "MPLBACKEND": "Agg"})
    for e in gallery:
        if e["kind"] not in ("example", "reproduction") or e["state"] != "script" or (ids and e["id"] not in ids):
            continue
        t0, ok, tail = time.time(), True, ""
        for s in e["scripts"]:
            try:
                out = subprocess.run([sys.executable, str(root / s)], cwd=(root / s).parent, env=env,
                                     capture_output=True, text=True, timeout=timeout)
                ok = ok and out.returncode == 0
                tail = (out.stderr or out.stdout)[-300:]
            except subprocess.TimeoutExpired:
                ok, tail = False, f"timeout after {timeout:g} s"
            if not ok:
                break
        runs[e["id"]] = {"ok": ok, "scripts": e["scripts"], "hash": _script_hash(root, e["scripts"]),
                         "seconds": round(time.time() - t0, 1), "recorded": time.strftime("%Y-%m-%d"),
                         "error": "" if ok else tail.strip()}
        log(f"{e['id']}: {'ok' if ok else 'failed'} in {runs[e['id']]['seconds']} s")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(runs, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return runs
