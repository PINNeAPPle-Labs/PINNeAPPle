#!/usr/bin/env python3
"""Create the campaign issues from manifest.tsv and the drafted bodies.

    python3 create_issues.py labels            # create the labels used by the campaign
    python3 create_issues.py create [ID ...]   # create issues (default: every entry not yet created)

Each body gets `Part of #<index>.` prepended, where <index> is the issue created first (D0).
Progress is written to created.tsv (id, number, url), so a re-run skips what already exists.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys

REPO = "PINNeAPPle-Labs/PINNeAPPle"
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_ID = "D0"

ORDER = ["D0", "A1", "E1"] + [f"B{i}" for i in range(1, 27)] + [f"C{i}" for i in range(1, 13)] + \
        [f"D{i}" for i in range(1, 39)]

LABELS = {
    "experiment": ("Reproducible experiment specification", "ededed"),
}


def run(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def manifest():
    path = os.path.join(HERE, "manifest.tsv")
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return {r["id"]: r for r in rows}


def created():
    path = os.path.join(HERE, "created.tsv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh, delimiter="\t")}


def record(issue_id, number, url):
    path = os.path.join(HERE, "created.tsv")
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if new:
            w.writerow(["id", "number", "url"])
        w.writerow([issue_id, number, url])


def ensure_labels():
    existing = run(["gh", "label", "list", "--repo", REPO, "--limit", "200"]).stdout
    for name, (desc, color) in LABELS.items():
        if f"{name}\t" in existing or existing.startswith(name + "\t") or f"\n{name}\t" in existing:
            continue
        r = run(["gh", "label", "create", name, "--repo", REPO, "--color", color, "--description", desc])
        print(f"label {name}: {'created' if r.returncode == 0 else r.stderr.strip()}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    rows = manifest()

    if cmd == "labels":
        ensure_labels()
        return 0

    if cmd != "create":
        print(__doc__)
        return 1

    done = created()
    index_number = done.get(INDEX_ID, {}).get("number")
    wanted = sys.argv[2:] or ORDER
    ensure_labels()

    for issue_id in wanted:
        if issue_id in done:
            print(f"{issue_id}\tskip (already #{done[issue_id]['number']})")
            continue
        row = rows.get(issue_id)
        if row is None:
            print(f"{issue_id}\tMISSING in manifest")
            continue
        body_path = os.path.join(HERE, issue_id + ".md")
        if not os.path.exists(body_path):
            print(f"{issue_id}\tMISSING body file")
            continue

        body = open(body_path, encoding="utf-8").read().strip()
        if issue_id == INDEX_ID:
            text = body
        else:
            if not index_number:
                print(f"{issue_id}\tno index issue yet — create D0 first")
                return 2
            text = f"Part of #{index_number}.\n\n{body}"

        tmp = os.path.join(HERE, ".tmp-body.md")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")

        args = ["gh", "issue", "create", "--repo", REPO, "--title", row["title"], "--body-file", tmp]
        for label in [l for l in row["labels"].split(",") if l.strip()]:
            args += ["--label", label.strip()]
        r = run(args)
        if r.returncode != 0:
            print(f"{issue_id}\tFAILED: {r.stderr.strip()}")
            return 3
        url = r.stdout.strip().split()[-1]
        number = url.rstrip("/").split("/")[-1]
        if issue_id == INDEX_ID:
            index_number = number
        record(issue_id, number, url)
        print(f"{issue_id}\t#{number}\t{url}")

    os.path.exists(os.path.join(HERE, ".tmp-body.md")) and os.remove(os.path.join(HERE, ".tmp-body.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
