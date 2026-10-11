"""Snapshot the open issues of the repository into the contribute page (docs/org/contribute/index.html).

The page groups and filters the issues in the browser (and refreshes them from the public GitHub API when it can);
this script only embeds a recent snapshot so the page works offline and without API quota.

    python scripts/issues/build.py                 # GITHUB_TOKEN is used when set
    python scripts/issues/build.py --from FILE...  # build from saved API pages instead of fetching
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REPO = "PINNeAPPle-Labs/PINNeAPPle"
OUT = os.path.join(ROOT, "docs", "org", "contribute", "index.html")


def _get(url: str) -> list:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def fetch() -> list[dict]:
    out = []
    for page in range(1, 20):
        path = f"repos/{REPO}/issues?state=open&per_page=100&page={page}"
        try:
            batch = _get("https://api.github.com/" + path)
        except Exception:
            batch = json.loads(subprocess.check_output(["gh", "api", path], text=True))   # authenticated CLI fallback
        out += batch
        if len(batch) < 100:
            break
    return out


def excerpt(body: str | None, n: int = 420) -> str:
    """First paragraphs of the body as plain text (markdown links, images, code fences and HTML removed)."""
    t = body or ""
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"<!--.*?-->|<[^>]+>", " ", t, flags=re.S)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^\s*#+\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*[-*]\s+\[[ xX]\]\s*", "· ", t, flags=re.M)              # task-list boxes
    t = re.sub(r"^\s*[-*]\s+", "· ", t, flags=re.M)
    t = re.sub(r"[*_`>]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


def slim(i: dict) -> dict:
    parent = i.get("parent_issue_url") or ""
    sub = i.get("sub_issues_summary") or {}
    return {"n": i["number"], "t": i["title"], "l": [x["name"] for x in i.get("labels", [])],
            "m": (i.get("milestone") or {}).get("title"), "c": i.get("comments", 0),
            "cr": i["created_at"][:10], "up": i["updated_at"][:10],
            "p": int(parent.rsplit("/", 1)[1]) if parent else None, "s": sub.get("total", 0),
            "a": bool(i.get("assignees")), "x": excerpt(i.get("body"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="files", nargs="*", help="saved API pages (JSON lists)")
    args = ap.parse_args()
    raw = []
    if args.files:
        for f in args.files:
            raw += json.load(open(f))
    else:
        raw = fetch()
    issues = sorted((slim(i) for i in raw if "pull_request" not in i), key=lambda x: -x["n"])
    snap = {"repo": REPO, "taken": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "issues": issues}
    tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    html = tpl.replace("/*__SNAPSHOT__*/null", json.dumps(snap, separators=(",", ":")).replace("</", "<\\/"))
    open(OUT, "w", encoding="utf-8").write(html)
    print(f"{len(issues)} open issues -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
