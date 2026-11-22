#!/usr/bin/env python3
"""Reconcile the mirrored 2026 issue schedule against live GitHub issues.

Read-only: no issue closures, PR creation, test execution, CI or GitHub Actions.
Runs locally with Python stdlib. Optional GH_TOKEN for authenticated requests.
Usage: python scripts/issue_batch_2026.py --date 2026-11-23 --out-dir .local/issue-batches
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPOS = ("PINNeAPPle-Labs/PINNeAPPle", "PINNeAPPle-Labs/pinneapple-engineering-api")
PLAN = Path(__file__).resolve().parents[1] / "docs" / "EXECUTION_PLAN_2026.md"
DAY = re.compile(r"^### Dia \d+ — (\d{2}/\d{2}/\d{4}) \(\d+ issues\)", re.M)
RANGE = re.compile(r"^\s*- \*\*(PINNeAPPle|Engineering API) \(\d+\):\*\* (.+)$", re.M)
PRIORITIES = ("P0", "P1", "P2", "P3")


def expand_ids(value: str) -> list[int]:
    result = []
    for part in value.strip().split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            if lo > hi or hi - lo > 5000:
                raise ValueError(f"invalid range: {part}")
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(part))
    if len(result) != len(set(result)):
        raise ValueError("duplicate issue IDs in the day schedule")
    return result


def parse_day(path: Path, date: str) -> dict[str, list[int]]:
    content = path.read_text(encoding="utf-8")
    headers = list(DAY.finditer(content))
    for index, match in enumerate(headers):
        day = dt.datetime.strptime(match.group(1), "%d/%m/%Y").date().isoformat()
        if day != date:
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(content)
        section = content[match.end():end]
        found = dict(RANGE.findall(section))
        if set(found) != {"PINNeAPPle", "Engineering API"}:
            raise ValueError(f"missing repository in day {date}")
        return {REPOS[0]: expand_ids(found["PINNeAPPle"]),
                REPOS[1]: expand_ids(found["Engineering API"])}
    raise ValueError(f"date not in plan: {date}")


def fetch_issue(repo: str, number: int, token: str | None) -> dict:
    url = f"https://api.github.com/repos/{repo}/issues/{number}"
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "pinneapple-issue-batch/1.0",
               "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub HTTP {exc.code} for {repo}#{number}") from None


def classify(issue: dict) -> dict:
    labels = [x["name"] for x in issue.get("labels", [])]
    body = issue.get("body") or ""
    title = issue.get("title") or ""
    priority = next((p for p in PRIORITIES if any(
        re.search(rf"\b{p}\b", label, re.I) for label in labels)), "unclassified")
    effort = next((size for size in ("S", "M", "L") if any(
        re.search(rf"\b{size}\b", label, re.I) for label in labels)), "unclassified")
    lower = (title + " " + body).lower()
    dependencies = sorted(set(re.findall(r"(?:depends on|blocked by|depende de|bloquead[oa] por)\s+#(\d+)",
                                         lower)))
    return {"priority": priority, "complexity": effort,
            "dependencies_mentioned": dependencies,
            "automation": "needs_review", "acceptance": "needs_manual_review"}


def reconcile(plan: dict[str, list[int]], token: str | None, pause: float) -> dict:
    rows = []
    for repo, numbers in plan.items():
        for number in numbers:
            row = {"repo": repo, "number": number,
                   "url": f"https://github.com/{repo}/issues/{number}"}
            try:
                issue = fetch_issue(repo, number, token)
                if "pull_request" in issue:
                    row.update(state="not_an_issue", title=issue.get("title", ""))
                else:
                    row.update(state=issue["state"], title=issue.get("title", ""),
                               labels=[x["name"] for x in issue.get("labels", [])],
                               **classify(issue))
            except (RuntimeError, urllib.error.URLError, TimeoutError, ValueError) as exc:
                row.update(state="unavailable", error=str(exc))
            rows.append(row)
            if pause:
                time.sleep(pause)
    return {"schema": "pinneapple.issue-batch/1", "snapshot_date": dt.date.today().isoformat(),
            "rows": rows, "counts": dict(Counter(x["state"] for x in rows)),
            "planned": {repo: len(numbers) for repo, numbers in plan.items()},
            "warning": "A plan is not evidence of completion. No issues have been modified."}


def render_md(result: dict, date: str) -> str:
    lines = [f"# Lote {date} — triagem do GitHub", "",
             "Gerado por script local somente leitura. Nenhum teste, GitHub Actions ou fechamento automático.", "",
             f"Planejadas: {sum(result['planned'].values())}; "
             f"estados: {json.dumps(result['counts'], ensure_ascii=False)}", "",
             "| Repositório | Issue | Prioridade | Esforço | Estado | Dependências |",
             "|---|---|---|---|---|---|"]
    for row in result["rows"]:
        repo = "PINNeAPPle" if row["repo"] == REPOS[0] else "Engineering API"
        deps = ", ".join("#" + d for d in row.get("dependencies_mentioned", [])) or "—"
        lines.append(f"| {repo} | [#{row['number']}]({row['url']}) | "
                     f"{row.get('priority', '—')} | {row.get('complexity', '—')} | "
                     f"{row['state']} | {deps} |")
    lines.extend(["", "## Próximos passos", "",
                  "1. Conferir descrição completa e critérios de aceite de cada issue aberta.",
                  "2. Identificar bloqueios e agrupar PRs por componente; não assumir independência.",
                  "3. Registrar implementação e evidência de revisão por issue.",
                  "4. Fechar somente após integração e validação explícita.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-11-23")
    parser.add_argument("--out-dir", default=".local/issue-batches")
    parser.add_argument("--offline", action="store_true", help="only create nominal manifest")
    parser.add_argument("--pause", type=float, default=0.2)
    args = parser.parse_args()
    plan = parse_day(PLAN, args.date)
    if args.offline:
        result = {"schema": "pinneapple.issue-batch/1", "snapshot_date": None,
                  "rows": [{"repo": repo, "number": n, "state": "not_checked",
                            "url": f"https://github.com/{repo}/issues/{n}"}
                           for repo, nums in plan.items() for n in nums],
                  "planned": {repo: len(nums) for repo, nums in plan.items()},
                  "counts": {"not_checked": sum(map(len, plan.values()))},
                  "warning": "Offline nominal manifest only; no live issue states checked."}
    else:
        result = reconcile(plan, os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN"), args.pause)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.date}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (out / f"{args.date}.md").write_text(render_md(result, args.date))
    print(f"{args.date}: {result['planned']} / {result['counts']}")
    return 0 if "unavailable" not in result["counts"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
