"""``python -m pinneapple_analysis.report examples/<x>`` -- write the one-page report of one or more experiments.

    python -m pinneapple_analysis.report examples/condensation_falsification
    python -m pinneapple_analysis.report examples/end_to_end/ahmed_body examples/freecad_openfoam --out run_results/reports
    python -m pinneapple_analysis.report --all --out run_results/reports     # every example with a manifest
    python -m pinneapple_analysis.report examples/freecad_openfoam --theme dark --no-pdf

Without ``--out`` a report lands in ``<folder>/report/``. With ``--out DIR`` and several folders each report gets its
own ``DIR/<name>/``, and ``DIR/index.html`` lists them: that index is what a docs page links to.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys

from .experiment import ReportError, experiment_report


def _folders(args) -> list[str]:
    if args.all:
        from pinneapple_lab.manifest import REPO, scan

        seen, out = set(), []
        for path, _m in scan():
            folder = os.path.dirname(path)
            rel = os.path.relpath(folder, REPO).replace(os.sep, "/")
            if rel in seen or not rel.startswith("examples"):
                continue
            seen.add(rel)
            out.append(folder)
        return out
    return list(args.folders)


def _index(out: str) -> str:
    """``out/index.html``: one row per report in the folder, read from each report's own ``report.json``.

    It scans instead of remembering, so building one more report later does not drop the earlier ones from the
    index -- the page is a view of what is on disk, like the report itself.
    """
    rows = []
    for name in sorted(os.listdir(out)):
        book = os.path.join(out, name, "report.json")
        if not os.path.isfile(book):
            continue
        try:
            with open(book, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        note = (f'{len(data.get("verdicts", {}).get("rows", []))} checks, '
                f'{data.get("verdicts", {}).get("failed", 0)} failed · '
                f'{len(data.get("kpis", []))} headline numbers · {data.get("generated", "")[:10]}')
        rows.append((f"{name}/index.html", data.get("header", {}).get("title") or name, note))
    path = os.path.join(out, "index.html")
    items = "".join(f'<li><a href="{html.escape(href)}">{html.escape(title)}</a> '
                    f'<span>{html.escape(note)}</span></li>' for href, title, note in sorted(rows, key=lambda r: r[1]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                 "<title>PINNeAPPle experiment reports</title><style>body{font:15px/1.6 system-ui,sans-serif;"
                 "max-width:44rem;margin:3rem auto;padding:0 1rem}h1{font-size:1.5rem}"
                 "li{margin:.5rem 0}span{color:#5d6b82;font-size:.9em}</style></head><body>"
                 "<h1>PINNeAPPle experiment reports</h1><p>One page per experiment, generated from its "
                 "<code>experiment.json</code>.</p><ul>" + items + "</ul></body></html>")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_analysis.report", description=__doc__.splitlines()[0])
    ap.add_argument("folders", nargs="*", help="experiment folders (each with its experiment.json)")
    ap.add_argument("--all", action="store_true", help="every example folder with a manifest")
    ap.add_argument("--out", default=None, help="output folder (default: <folder>/report)")
    ap.add_argument("--theme", default="light", choices=("light", "dark"))
    ap.add_argument("--no-pdf", action="store_true", help="write the HTML only")
    ap.add_argument("--figures", type=int, default=4, help="how many of the manifest's figures to show (default 4)")
    ap.add_argument("--cost-per-hour", type=float, default=None,
                    help="price the wall time at this rate; without it the report says the run is not priced")
    args = ap.parse_args(argv)
    folders = _folders(args)
    if not folders:
        ap.error("name at least one experiment folder, or --all")
    multi = len(folders) > 1 or args.out is not None
    built, failed = 0, 0
    for folder in folders:
        out = os.path.join(args.out, os.path.basename(os.path.normpath(folder))) if args.out else None
        try:
            r = experiment_report(folder, out=out, theme=args.theme, pdf=not args.no_pdf,
                                  limit_figures=args.figures, cost_per_hour=args.cost_per_hour)
        except ReportError as exc:
            print(f"skipped {folder}: {exc}", file=sys.stderr)
            failed += 1
            continue
        built += 1
        print(f"{r.html}{'  ' + r.pdf if r.pdf else ''}")
    if multi and built and args.out:
        print(_index(os.path.abspath(args.out)))
    return 1 if failed and not built else 0


if __name__ == "__main__":                       # pragma: no cover
    raise SystemExit(main())
