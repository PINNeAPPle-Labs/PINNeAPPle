#!/usr/bin/env python3
"""Build manifest.tsv from the backlog files and check the drafted issue bodies.

Usage:
    python3 build_manifest.py          # writes manifest.tsv, prints the parsed entries
    python3 build_manifest.py --check  # validates the bodies that exist and reports problems
"""
from __future__ import annotations

import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKLOGS = ["BACKLOG-A-B.md", "BACKLOG-C.md", "BACKLOG-D.md"]

HEADINGS = [
    "## Summary",
    "## Question",
    "## Why run it",
    "## Physical model",
    "## Geometry and inputs",
    "## Solver and engine",
    "## What already exists in PINNeAPPle (verified)",
    "## What is missing (the work)",
    "## Experiment protocol",
    "## Baselines, references and verification",
    "## Metrics",
    "## Deliverables",
    "## Acceptance criteria",
    "## Realizability and evidence class",
    "## Risks and failure modes",
    "## Out of scope",
    "## References",
]

FORBIDDEN = [
    "Claude", "Anthropic", "Co-Authored-By", "claude.ai", "generated with claude",
    "as an AI", "the assistant",
]


def parse_backlogs():
    rows = []
    for name in BACKLOGS:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("## "):
                    continue
                parts = [p.strip() for p in line[3:].split("|")]
                if len(parts) < 3 or not parts[2].startswith("labels:"):
                    continue
                issue_id, title = parts[0], parts[1]
                labels = [l.strip() for l in parts[2][len("labels:"):].split(",") if l.strip()]
                rows.append({"id": issue_id, "title": title, "labels": labels})
    return rows


def write_manifest(rows):
    path = os.path.join(HERE, "manifest.tsv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "title", "labels"])
        for r in rows:
            w.writerow([r["id"], r["title"], ",".join(r["labels"])])
    return path


def check(rows):
    problems = []
    missing_files = []
    for r in rows:
        path = os.path.join(HERE, r["id"] + ".md")
        if not os.path.exists(path):
            missing_files.append(r["id"])
            continue
        text = open(path, encoding="utf-8").read()
        words = len(text.split())
        if words < 500:
            problems.append(f"{r['id']}: only {words} words")
        if words > 3000:
            problems.append(f"{r['id']}: {words} words (too long)")
        for h in HEADINGS:
            if h not in text:
                problems.append(f"{r['id']}: missing heading {h!r}")
        for bad in FORBIDDEN:
            if re.search(r"\b" + re.escape(bad) + r"\b", text, flags=re.IGNORECASE):
                problems.append(f"{r['id']}: contains forbidden string {bad!r}")
        if re.match(r"^#\s+\S", text.split("\n", 1)[0]):
            problems.append(f"{r['id']}: starts with a level-1 heading; the body must start with '## Summary'")
        if "we measured" in text.lower() or "results show" in text.lower():
            problems.append(f"{r['id']}: reads like a reported result ('we measured'/'results show')")
    return problems, missing_files


if __name__ == "__main__":
    rows = parse_backlogs()
    path = write_manifest(rows)
    print(f"{len(rows)} entries -> {path}")
    print(" ".join(r["id"] for r in rows))
    if "--check" in sys.argv:
        problems, missing = check(rows)
        print(f"\n{len(missing)} bodies missing: {' '.join(missing) if missing else '-'}")
        if problems:
            print(f"{len(problems)} problems:")
            for p in problems:
                print("  -", p)
        else:
            print("no problems in the bodies that exist")
