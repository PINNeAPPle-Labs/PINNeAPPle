"""The repo's own backlog, generated from GitHub and checked in (#1497 aftermath, ./docs/backlog.md).

Why this exists: the scope of the year was decided once, in a document outside the repository, and a document that
lives outside the repository drifts. This script keeps the list **inside** the repo, derived from the issues
themselves -- a snapshot fetched from the API, and a page rendered from that snapshot -- so the page can be rebuilt by
anyone and checked without the network.

    python scripts/backlog_report.py --fetch   # refresh docs/backlog.json from GitHub (needs gh or GITHUB_TOKEN)
    python scripts/backlog_report.py --write   # regenerate docs/backlog.md from the snapshot
    python scripts/backlog_report.py --check   # the page matches the snapshot and is internally consistent (CI)

Only issues of **this** repository are in it, and only the ones carrying one of the two scope labels (``escopo: 2026``
and ``escopo: futuro``); an issue with no scope label is out of the list on purpose, and the page says how many there
are so the absence is visible rather than silent.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "docs" / "backlog.json"
PAGE = ROOT / "docs" / "backlog.md"
REPO = "PINNeAPPle-Labs/PINNeAPPle"
SCOPES = ("escopo: 2026", "escopo: futuro")
SCHEMA = "pinneapple.backlog/1"
SCOPE_TITLE = {"escopo: 2026": "This year", "escopo: futuro": "Next cycle"}
AREA_ORDER = ("area: core-api", "area: solvers", "area: surrogates-optimization", "area: digital-twin",
              "area: visualization", "area: benchmark-cost", "area: apps", "area: ecosystem",
              "area: release-quality", "area: contributing", "area: futures")


# ------------------------------------------------------------------ fetching
def _gh(args: Sequence[str]) -> str:
    gh = shutil.which("gh")
    if gh is None:
        raise SystemExit("gh is not installed: set GITHUB_TOKEN and use --api, or install the GitHub CLI")
    out = subprocess.run([gh, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"gh {' '.join(args)} failed: {out.stderr.strip()[:400]}")
    return out.stdout


def _api(path: str) -> List[Dict[str, Any]]:
    out = _gh(["api", "--paginate", path, "--jq",
               '.[] | select(.pull_request == null) | {number, title, state, html_url, labels: [.labels[].name], '
               'closed_at}'])
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def fetch() -> Dict[str, Any]:
    """Every issue of this repository that carries a scope label, open or closed, as a dated snapshot."""
    issues: Dict[int, Dict[str, Any]] = {}
    for scope in SCOPES:
        label = scope.replace(" ", "%20").replace(":", "%3A")
        for row in _api(f"repos/{REPO}/issues?state=all&labels={label}&per_page=100"):
            issues[int(row["number"])] = _clean(row)
    everything = _api(f"repos/{REPO}/issues?state=all&per_page=100")
    unscoped = sum(1 for row in everything if not (set(row["labels"]) & set(SCOPES)))
    return {"schema": SCHEMA, "repo": REPO, "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "issues": [issues[k] for k in sorted(issues)], "unscoped_seen": unscoped,
            "note": "one API page of unscoped issues is counted, to show how many exist outside the scope"}


def _clean(row: Mapping[str, Any]) -> Dict[str, Any]:
    labels = [str(x) for x in row.get("labels") or ()]
    scope = next((s for s in SCOPES if s in labels), "")
    area = next((x for x in labels if x.startswith("area: ")), "")
    priority = next((x for x in labels if x.startswith("priority: ")), "")
    effort = next((x for x in labels if x.startswith("effort: ")), "")
    return {"number": int(row["number"]), "title": str(row["title"]).strip(), "state": str(row["state"]),
            "url": row.get("html_url") or f"https://github.com/{REPO}/issues/{row['number']}",
            "scope": scope, "area": area, "priority": priority, "effort": effort,
            "closed_at": (row.get("closed_at") or "")[:10]}


# ------------------------------------------------------------------ rendering
def group(snapshot: Mapping[str, Any]) -> Dict[str, Dict[str, List[Mapping[str, Any]]]]:
    """``{scope: {area: [issues]}}``, areas in the declared order and the unlabelled ones last."""
    out: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {s: defaultdict(list) for s in SCOPES}
    for issue in snapshot["issues"]:
        scope = issue.get("scope") or SCOPES[0]
        out.setdefault(scope, defaultdict(list))[issue.get("area") or "(no area label)"].append(issue)
    for scope in out:
        for area in out[scope]:
            out[scope][area].sort(key=lambda i: i["number"])
    return out


def _area_key(area: str) -> tuple:
    return (AREA_ORDER.index(area) if area in AREA_ORDER else len(AREA_ORDER), area)


def counts(snapshot: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for scope in SCOPES:
        rows = [i for i in snapshot["issues"] if (i.get("scope") or SCOPES[0]) == scope]
        out[scope] = {"total": len(rows), "open": sum(1 for i in rows if i["state"] == "open"),
                      "closed": sum(1 for i in rows if i["state"] == "closed")}
    return out


def render(snapshot: Mapping[str, Any]) -> str:
    """The page. Deterministic: the same snapshot renders the same bytes."""
    c = counts(snapshot)
    total = sum(v["total"] for v in c.values())
    opened = sum(v["open"] for v in c.values())
    grouped = group(snapshot)
    lines = [
        "# Backlog: this repository, by scope",
        "",
        f"Every issue of `{snapshot['repo']}` that carries a scope label, as of **{snapshot['fetched']}**: "
        f"**{total}** in the scope of this year and the next cycle, **{opened}** still open. Issues without a scope "
        f"label are not listed here -- they are the backlog nobody has decided about yet.",
        "",
        "This page is generated. Do not edit it by hand:",
        "",
        "```bash",
        "python scripts/backlog_report.py --fetch   # refresh the snapshot from GitHub",
        "python scripts/backlog_report.py --write   # regenerate this page from the snapshot",
        "python scripts/backlog_report.py --check   # CI: the page matches the snapshot and adds up",
        "```",
        "",
        "| scope | issues | open | closed |",
        "|---|---:|---:|---:|",
    ]
    for scope in SCOPES:
        lines.append(f"| {SCOPE_TITLE[scope]} (`{scope}`) | {c[scope]['total']} | {c[scope]['open']} | "
                     f"{c[scope]['closed']} |")
    lines += ["", "A checked box means the issue is closed. The priority is the label the project gave it, not a "
                  "judgement of this page.", ""]
    for scope in SCOPES:
        lines += [f"## {SCOPE_TITLE[scope]} — {c[scope]['total']} issues, {c[scope]['open']} open", ""]
        areas = sorted(grouped.get(scope, {}), key=_area_key)
        if not areas:
            lines += ["_Nothing in this scope._", ""]
            continue
        for area in areas:
            rows = grouped[scope][area]
            open_rows = sum(1 for r in rows if r["state"] == "open")
            title = area.replace("area: ", "")
            lines += [f"### {title} — {len(rows)} issues ({open_rows} open)", ""]
            for issue in rows:
                box = "x" if issue["state"] == "closed" else " "
                tags = " ".join(t.replace("priority: ", "").replace("effort: ", "")
                                for t in (issue.get("priority"), issue.get("effort")) if t)
                suffix = f" ({tags})" if tags else ""
                lines.append(f"- [{box}] [#{issue['number']}]({issue['url']}) {issue['title']}{suffix}")
            lines.append("")
    lines += ["## What this page is not", "",
              "It is not a plan: it says what is open and what is closed, not what comes next. It is not a promise "
              "either -- an issue closed here was closed by a merged change or by a decision, and the reason lives in "
              "the issue, not in this list.", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ gates
def check(root: Path = ROOT) -> List[str]:
    snapshot_path, page_path = root / SNAPSHOT.relative_to(ROOT), root / PAGE.relative_to(ROOT)
    if not snapshot_path.is_file():
        return [f"{SNAPSHOT.relative_to(ROOT)} is missing: run --fetch"]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    problems: List[str] = []
    if snapshot.get("schema") != SCHEMA:
        problems.append(f"snapshot schema is {snapshot.get('schema')!r}, expected {SCHEMA!r}")
    if snapshot.get("repo") != REPO:
        problems.append(f"snapshot is for {snapshot.get('repo')!r}, this page is for {REPO!r}")
    numbers = [i["number"] for i in snapshot["issues"]]
    dupes = sorted({n for n in numbers if numbers.count(n) > 1})
    if dupes:
        problems.append(f"an issue appears more than once in the snapshot: {dupes[:5]}")
    missing_scope = [i["number"] for i in snapshot["issues"] if i.get("scope") not in SCOPES]
    if missing_scope:
        problems.append(f"issues with no scope label in the snapshot: {missing_scope[:5]}")
    if not page_path.is_file():
        problems.append(f"{PAGE.relative_to(ROOT)} is missing: run --write")
    elif page_path.read_text(encoding="utf-8") != render(snapshot):
        problems.append(f"{PAGE.relative_to(ROOT)} is not what the snapshot renders: run --write")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The repository's own backlog, generated and checked.")
    group_ = parser.add_mutually_exclusive_group(required=True)
    group_.add_argument("--fetch", action="store_true", help="refresh the snapshot from GitHub")
    group_.add_argument("--write", action="store_true", help="regenerate the page from the snapshot")
    group_.add_argument("--check", action="store_true", help="verify the page against the snapshot (offline)")
    args = parser.parse_args(argv)

    if args.fetch:
        snapshot = fetch()
        SNAPSHOT.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        c = counts(snapshot)
        print(f"wrote {SNAPSHOT.relative_to(ROOT)}: " + ", ".join(
            f"{SCOPE_TITLE[s]} {c[s]['total']} ({c[s]['open']} open)" for s in SCOPES))
        return 0
    if args.write:
        if not SNAPSHOT.is_file():
            print(f"problem: {SNAPSHOT.relative_to(ROOT)} is missing; run --fetch")
            return 1
        snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        PAGE.write_text(render(snapshot), encoding="utf-8")
        print(f"wrote {PAGE.relative_to(ROOT)}")
        return 0
    problems = check()
    for problem in problems:
        print(f"problem: {problem}")
    print(f"{len(problems)} problem(s)" if problems else "the backlog page matches its snapshot")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
