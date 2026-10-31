"""The evidence report of a study: one self-contained HTML page (sections of text, tables and figures embedded as
data) and, through :mod:`pinneapple_veriphysics.evidence_package`, a zip with every source file hashed.

    html = write_report("Ahmed body", sections, "results/report.html", base="results")
    pkg = evidence_package("Ahmed body", ["results/dataset/ahmed_body.json", "results/benchmark_table.csv"], "results/evidence.zip")

A section is ``{"title", "text"?, "table"?: {"columns": [...], "rows": [[...]]}, "figures"?: [path, ...],
"decision"?: "APPROVED" | "REVIEW" | "REJECT"}``. Numbers come from the study's JSON files; the report adds none.
"""
from __future__ import annotations

import base64
import html as _h
import os
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["write_report", "evidence_package"]

_CSS = """
:root { --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e4e3df; --ok:#0ca30c; --warn:#b45309; --bad:#d03b3b; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --ink:#f0efec; --muted:#c3c2b7; --line:#383835; } }
body { background:var(--bg); color:var(--ink); font:15px/1.55 system-ui, sans-serif; margin:0 auto; max-width:1040px;
       padding:24px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 8px; border-top:1px solid var(--line);
padding-top:16px; } .sub { color:var(--muted); }
table { border-collapse:collapse; width:100%; font-size:13px; margin:8px 0; display:block; overflow-x:auto; }
th, td { border-bottom:1px solid var(--line); padding:4px 8px; text-align:left; font-variant-numeric:tabular-nums; }
th { color:var(--muted); font-weight:600; }
img { max-width:100%; border:1px solid var(--line); border-radius:6px; margin:6px 0; }
.figs { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:8px; }
.badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; font-weight:600; border:1px solid; }
.APPROVED { color:var(--ok); } .REVIEW { color:var(--warn); } .REJECT { color:var(--bad); }
"""


def _cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return _h.escape(str(v))


def _img(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif"}.get(ext, "png")
    data = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/{mime};base64,{data}" alt="{_h.escape(os.path.basename(path))}" loading="lazy">'


def write_report(title: str, sections: Sequence[Mapping[str, Any]], out: str, *, base: str = ".",
                 subtitle: str = "") -> str:
    """Write the page; figure paths are relative to ``base`` (missing ones are listed as missing, not skipped)."""
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
             f"content='width=device-width,initial-scale=1'><title>{_h.escape(title)}</title><style>{_CSS}</style>"
             f"</head><body><h1>{_h.escape(title)}</h1><p class=sub>{_h.escape(subtitle)}</p>"]
    for s in sections:
        dec = s.get("decision")
        badge = f' <span class="badge {dec}">{dec}</span>' if dec else ""
        parts.append(f"<h2>{_h.escape(s['title'])}{badge}</h2>")
        if s.get("text"):
            parts += [f"<p>{_h.escape(t)}</p>" for t in str(s["text"]).split("\n\n")]
        t = s.get("table")
        if t:
            parts.append("<table><thead><tr>" + "".join(f"<th>{_h.escape(c)}</th>" for c in t["columns"])
                         + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_cell(v)}</td>" for v in r)
                                                            + "</tr>" for r in t["rows"]) + "</tbody></table>")
        figs = s.get("figures") or []
        if figs:
            parts.append("<div class=figs>")
            for f in figs:
                p = os.path.join(base, f)
                parts.append(_img(p) if os.path.exists(p) else f"<p class=sub>missing figure: {_h.escape(f)}</p>")
            parts.append("</div>")
    parts.append("</body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("".join(parts))
    return out


def evidence_package(title: str, files: Sequence[str], out_zip: str, *, meta: Mapping[str, Any] | None = None
                     ) -> dict[str, Any]:
    """The VeriPhysics evidence package of ``files`` (contracts, metric tables, ``pinneapple check`` reports, ...):
    report + HTML + zip with SHA-256 of every source; returns the report and the problems
    :func:`verify_package` finds in the zip (empty when intact)."""
    from pinneapple_veriphysics import evidence_package as ep
    srcs = [ep.load_source(os.path.basename(f), open(f, "rb").read()) for f in files]
    rep = ep.build_report({"title": title, **dict(meta or {})}, srcs)
    html = ep.render_html(rep)
    data = ep.build_package(rep, srcs, html=html)
    with open(out_zip, "wb") as fh:
        fh.write(data)
    return {"report": rep, "problems": ep.verify_package(data), "zip": out_zip}
