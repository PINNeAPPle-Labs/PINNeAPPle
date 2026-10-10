"""Reports from the lab: one experiment (or case), or many selected by a filter, as HTML, Markdown or PDF.

A report carries, for each item: what it is, the one-line story, the key figures, the headline metrics, every
validation check (with what it is evidence of), the evidence dimensions and tier, the readiness for product /
publication / marketing / training data, the limitations, the path forward (actions, effort, what they unlock) and
how to reproduce it (run id, commit, code, command). Everything comes from the lab record and the curation
(``pinneapple_lab.curation``); nothing is written by hand here.

    python -m pinneapple_lab brief --item kepler_law -o kepler.html
    python -m pinneapple_lab brief --tier A,B --use paper -o papers.pdf          # everything closest to a paper
    python -m pinneapple_lab brief --ready marketing --format md -o marketing.md
    python -m pinneapple_lab brief --experiment benchmark_case -o landing.html   # every case of one experiment
"""
from __future__ import annotations

import base64
import glob
import html
import io
import json
import os
import time
from typing import Any

from .curation import TIERS, USES, curate

USE_LABEL = {"product": "Product", "paper": "Publication", "marketing": "Marketing", "training_data": "Training data"}


def select(cur: dict[str, Any], *, items: list[str] | None = None, experiment: list[str] | None = None,
           tier: list[str] | None = None, use: str | None = None, ready: str | None = None,
           tag: list[str] | None = None, tags_of: dict[str, list[str]] | None = None, limit: int | None = None) -> list[str]:
    """Keys of the curation items that match every given filter. ``use`` orders by distance to that use;
    ``ready`` keeps only items ready (or approved) for it."""
    keys = []
    for k, it in cur["items"].items():
        if items and k not in items and it["experiment"] not in items:
            continue
        if experiment and it["experiment"] not in experiment:
            continue
        if tier and it["tier"] not in tier:
            continue
        if ready and it["plan"]["uses"][ready]["status"] == "gap":
            continue
        if tag and tags_of is not None and not set(tag) & set(tags_of.get(it["experiment"], [])):
            continue
        keys.append(k)
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    if use:
        keys.sort(key=lambda k: (cur["items"][k]["plan"]["uses"][use]["distance"], order[cur["items"][k]["tier"]]))
    else:
        keys.sort(key=lambda k: (order[cur["items"][k]["tier"]], -(cur["items"][k]["best_score"] or 0)))
    return keys[:limit] if limit else keys


def _load(path: str, default: Any) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _figures(run_dir: str, n: int = 3) -> list[str]:
    from .html import showcase_score
    figs = glob.glob(os.path.join(run_dir, "figures", "*.png")) + glob.glob(os.path.join(run_dir, "figures", "*.jpg")) \
        + glob.glob(os.path.join(run_dir, "figures", "*.gif"))
    return sorted(figs, key=lambda f: (-showcase_score(f), f))[:n]


def _img_data(path: str, width: int = 900) -> tuple[str, bytes] | None:
    try:
        from PIL import Image
        im = Image.open(path)
        im.seek(0)
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, max(1, int(im.height * width / im.width))))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82, optimize=True)
        return "image/jpeg", buf.getvalue()
    except Exception:
        return None


def gather(store, cur: dict[str, Any], key: str, n_figures: int = 3) -> dict[str, Any]:
    """Everything a report says about one item."""
    from .spec import get
    it = cur["items"][key]
    run = cur["runs"][it["best_run"]]
    d = store.run_dir(it["experiment"], it["best_run"])
    rec = _load(os.path.join(d, "run.json"), {})
    try:
        cls = get(it["experiment"])
        desc = (getattr(cls, "case_descriptions", {}) or {}).get(it["case"], cls.description)
        refs, tags = list(getattr(cls, "references", []) or []), list(cls.tags)
    except KeyError:
        desc, refs, tags = "", [], []
    metrics = {k: v for k, v in _load(os.path.join(d, "metrics.json"), {}).items() if isinstance(v, (int, float))}
    src = _load(os.path.join(d, "inputs", "source.json"), {})
    papers = src.get("papers", []) if isinstance(src, dict) else []
    params = rec.get("params", {})
    cmd = f"python -m pinneapple_lab run {it['experiment']} " + " ".join(
        f"-p {k}={json.dumps(v) if not isinstance(v, str) else v}" for k, v in params.items())
    return {"key": key, "title": it["title"], "experiment": it["experiment"], "case": it["case"], "tier": it["tier"],
            "tier_name": TIERS[it["tier"]], "score": it["best_score"], "story": it.get("story", ""), "description": desc,
            "tags": tags, "references": refs + papers, "run": it["best_run"], "runs": it["runs"],
            "usable": it["usable_fraction"], "metrics": metrics, "checks": _load(os.path.join(d, "validation.json"), []),
            "dimensions": run["dimensions"], "coverage": run["coverage"], "plan": it["plan"],
            "figures": _figures(d, n_figures), "git": rec.get("git"), "code": rec.get("code", {}).get("files", []),
            "datasets": rec.get("datasets", {}), "command": cmd, "seconds": rec.get("seconds")}


# ---------------------------------------------------------------------- HTML
_CSS = """
:root { --ink: #142127; --muted: #5a6b72; --line: #cfd9dc; --paper: #ffffff; --chip: #eef2f3; --ok: #2e7d4f;
  --warn: #b26b00; --bad: #b3261e; --accent: #0b6e69 }
body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.55 "IBM Plex Sans", system-ui, sans-serif }
main { max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; display: grid; gap: 28px }
h1 { font: 700 34px/1.1 "IBM Plex Sans Condensed", system-ui, sans-serif; margin: 0 }
h2 { font: 700 24px/1.2 "IBM Plex Sans Condensed", system-ui, sans-serif; margin: 0 0 4px }
h3 { font: 600 12px "IBM Plex Mono", monospace; letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
  margin: 14px 0 6px }
.meta, .muted { color: var(--muted) } code, .mono { font: 12.5px "IBM Plex Mono", monospace; overflow-wrap: anywhere }
table { width: 100%; border-collapse: collapse; font-size: 13.5px }
td, th { border-bottom: 1px solid var(--line); padding: 5px 6px; text-align: left; vertical-align: top }
th { font: 600 11.5px "IBM Plex Mono", monospace; color: var(--muted); text-transform: uppercase; letter-spacing: .05em }
td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: "IBM Plex Mono", monospace }
.tier { display: inline-block; padding: 3px 8px; border-radius: 5px; color: #fff; font: 600 12px "IBM Plex Mono", monospace }
.tA { background: var(--ok) } .tB { background: var(--accent) } .tC { background: var(--warn) } .tD { background: var(--bad) }
.pass { color: var(--ok); font-weight: 600 } .fail { color: var(--bad); font-weight: 600 }
.figs { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px }
.figs img { width: 100%; border: 1px solid var(--line); border-radius: 8px }
.story { font-size: 18px; font-style: italic; margin: 6px 0 }
.item { border-top: 3px solid var(--ink); padding-top: 16px; break-before: page }
.bar { height: 7px; background: var(--chip); border-radius: 4px; overflow: hidden; min-width: 120px }
.bar b { display: block; height: 100%; background: var(--ok) } .bar b.f { background: var(--bad) }
.eff { display: inline-block; padding: 1px 5px; border-radius: 4px; color: #fff; font: 600 11px "IBM Plex Mono", monospace }
.eS { background: var(--ok) } .eM { background: var(--warn) } .eL { background: var(--bad) }
@media print { main { padding: 0 } .item { break-before: page } }
"""


def _e(x: Any) -> str:
    return html.escape(str(x))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        a = abs(v)
        return f"{v:.3e}" if (a and (a >= 1e5 or a < 1e-3)) else f"{v:.4g}"
    return _e(v)


def _readiness_cells(g: dict[str, Any]) -> str:
    out = []
    for u in USES:
        s = g["plan"]["uses"][u]
        out.append("approved" if s["status"] == "approved" else "ready" if s["status"] == "ready" else f"{s['distance']} pts")
    return "".join(f"<td>{_e(x)}</td>" for x in out)


def render_html(store, groups: list[dict[str, Any]], title: str, filters: str = "") -> str:
    L = [f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
         f"content=\"width=device-width, initial-scale=1\"><title>{_e(title)}</title>"
         "<link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&"
         "family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;600&display=swap\">"
         f"<style>{_CSS}</style></head><body><main>",
         f"<header><div class=\"meta mono\">PINNeAPPle Lab report · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}</div>"
         f"<h1>{_e(title)}</h1><p class=\"meta\">{len(groups)} item{'s' if len(groups) != 1 else ''}"
         f"{' · ' + _e(filters) if filters else ''}. Tiers and readiness from <code>pinneapple_lab.curation</code>; "
         "distances in effort points (S = 1, M = 3, L = 8).</p></header>"]
    L.append("<section><h2>Summary</h2><table><tr><th>item</th><th>tier</th><th>score</th>"
             + "".join(f"<th>{USE_LABEL[u]}</th>" for u in USES) + "<th>story / limitation</th></tr>")
    for g in groups:
        note = g["story"] or (g["plan"]["limitations"][0] if g["plan"]["limitations"] else "")
        L.append(f"<tr><td><a href=\"#{_e(g['key'])}\">{_e(g['title'])}</a><br><span class=\"muted mono\">{_e(g['experiment'])}</span></td>"
                 f"<td><span class=\"tier t{g['tier']}\">{g['tier']}</span></td><td class=\"num\">{_fmt(g['score'])}</td>"
                 f"{_readiness_cells(g)}<td class=\"muted\">{_e(note)}</td></tr>")
    L.append("</table></section>")
    for g in groups:
        L.append(f"<section class=\"item\" id=\"{_e(g['key'])}\"><div class=\"meta mono\">{_e(g['experiment'])}"
                 f"{' / ' + _e(g['case']) if g['case'] else ''}</div><h2>{_e(g['title'])} "
                 f"<span class=\"tier t{g['tier']}\">{g['tier']} · {_e(g['tier_name'])}</span></h2>")
        if g["story"]:
            L.append(f"<p class=\"story\">“{_e(g['story'])}”</p>")
        L.append(f"<p>{_e(g['description'])}</p>")
        if g["figures"]:
            imgs = []
            for f in g["figures"]:
                im = _img_data(f)
                if im:
                    imgs.append(f"<img alt=\"{_e(os.path.basename(f))}\" src=\"data:{im[0]};base64,"
                                f"{base64.b64encode(im[1]).decode()}\">")
            L.append(f"<div class=\"figs\">{''.join(imgs)}</div>")
        L.append("<h3>Readiness</h3><table><tr>" + "".join(f"<th>{USE_LABEL[u]}</th>" for u in USES) + "</tr><tr>")
        for u in USES:
            s = g["plan"]["uses"][u]
            L.append(f"<td>{'approved' if s['status'] == 'approved' else 'ready' if s['status'] == 'ready' else 'missing: ' + _e(', '.join(s['missing']))}</td>")
        L.append("</tr></table>")
        if g["metrics"]:
            L.append("<h3>Headline metrics</h3><table>" + "".join(
                f"<tr><td class=\"mono\">{_e(k)}</td><td class=\"num\">{_fmt(v)}</td></tr>"
                for k, v in list(g["metrics"].items())[:14]) + "</table>")
        if g["checks"]:
            L.append("<h3>Validation</h3><table><tr><th>check</th><th>kind</th><th>result</th><th>value</th><th>detail</th></tr>")
            for c in g["checks"]:
                lim = " ≤ " + _fmt(c["max"]) if c.get("max") is not None else ""
                lim += " ≥ " + _fmt(c["min"]) if c.get("min") is not None else ""
                lim += " ref " + _fmt(c["reference"]) if c.get("reference") is not None else ""
                L.append(f"<tr><td class=\"mono\">{_e(c['name'])}</td><td>{_e(c.get('kind', ''))}</td>"
                         f"<td class=\"{'pass' if c['passed'] else 'fail'}\">{'PASS' if c['passed'] else 'FAIL'}</td>"
                         f"<td class=\"num\">{_fmt(c.get('value')) if c.get('value') is not None else ''}{_e(lim)}</td>"
                         f"<td class=\"muted\">{_e(c.get('detail', ''))}</td></tr>")
            L.append("</table>")
        L.append(f"<h3>Evidence (score {_fmt(g['score'])}, coverage {int(100 * g['coverage'])} %)</h3><table>")
        for k, dm in g["dimensions"].items():
            w = 0 if dm["score"] is None else int(100 * dm["score"])
            L.append(f"<tr><td>{_e(k)}</td><td><div class=\"bar\"><b class=\"{'f' if dm['status'] == 'FAIL' else ''}\" "
                     f"style=\"width:{w}%\"></b></div></td><td class=\"mono\">{'not run' if dm['score'] is None else w}</td>"
                     f"<td class=\"muted\">{_e('; '.join(dm['evidence'][:2]))}</td></tr>")
        L.append("</table>")
        L.append("<h3>Limitations</h3>" + ("<ul>" + "".join(f"<li>{_e(x)}</li>" for x in g["plan"]["limitations"]) + "</ul>"
                                         if g["plan"]["limitations"] else "<p class=\"muted\">None declared or detected.</p>"))
        L.append("<h3>Path forward</h3>" + ("<ol>" + "".join(
            f"<li><span class=\"eff e{a['effort']}\">{a['effort']}</span> {_e(a['action'])} "
            f"<span class=\"muted mono\">unlocks {_e(', '.join(USE_LABEL.get(u, u) for u in a['unlocks']))}</span></li>"
            for a in g["plan"]["actions"]) + "</ol>" if g["plan"]["actions"] else "<p class=\"muted\">Nothing missing.</p>"))
        if g["references"]:
            L.append("<h3>References</h3><ul>" + "".join(f"<li>{_e(r)}</li>" for r in g["references"]) + "</ul>")
        L.append(f"<h3>Reproduce</h3><p class=\"mono\">run {_e(g['run'])} · git {_e(g['git'])} · "
                 f"{len(g['code'])} code files saved · {g['runs']} runs, {int(100 * g['usable'])} % usable</p>"
                 f"<pre class=\"mono\">{_e(g['command'])}</pre></section>")
    L.append("</main></body></html>")
    return "\n".join(L)


# ---------------------------------------------------------------------- Markdown
def render_md(groups: list[dict[str, Any]], title: str, filters: str = "") -> str:
    L = [f"# {title}", "", f"{len(groups)} items{' · ' + filters if filters else ''}. "
         f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from the PINNeAPPle Lab.", "",
         "| item | tier | score | " + " | ".join(USE_LABEL[u] for u in USES) + " |",
         "|---|---|---|" + "---|" * len(USES)]
    for g in groups:
        cells = []
        for u in USES:
            s = g["plan"]["uses"][u]
            cells.append(s["status"] if s["status"] != "gap" else f"{s['distance']} pts")
        L.append(f"| {g['title']} (`{g['experiment']}`) | {g['tier']} | {g['score']} | " + " | ".join(cells) + " |")
    for g in groups:
        L += ["", f"## {g['title']} — {g['tier']} {g['tier_name']}", ""]
        if g["story"]:
            L += [f"> {g['story']}", ""]
        L += [g["description"], ""]
        if g["checks"]:
            L += ["**Validation**", ""] + [f"- {'PASS' if c['passed'] else 'FAIL'} `{c['name']}` ({c.get('kind', '')})"
                                           + (f": {c['detail']}" if c.get("detail") else "") for c in g["checks"]] + [""]
        if g["plan"]["limitations"]:
            L += ["**Limitations**", ""] + [f"- {x}" for x in g["plan"]["limitations"]] + [""]
        if g["plan"]["actions"]:
            L += ["**Path forward**", ""] + [f"{i + 1}. [{a['effort']}] {a['action']} (unlocks "
                                             f"{', '.join(USE_LABEL.get(u, u) for u in a['unlocks'])})"
                                             for i, a in enumerate(g["plan"]["actions"])] + [""]
        L += [f"Reproduce: `{g['command']}` (run `{g['run']}`, git {g['git']})"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- PDF
def render_pdf(groups: list[dict[str, Any]], title: str, filters: str = "") -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    st = getSampleStyleSheet()
    body, h1, h2 = st["BodyText"], st["Title"], st["Heading2"]
    small = st["BodyText"].clone("small", fontSize=8, leading=10)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm, title=title)
    W = A4[0] - 3.2 * cm
    S: list[Any] = [Paragraph(_e(title), h1), Paragraph(f"{len(groups)} items{' · ' + _e(filters) if filters else ''}. "
                                                         "Tiers and readiness from pinneapple_lab.curation.", body),
                    Spacer(1, 8)]
    rows = [["item", "tier", "score"] + [USE_LABEL[u] for u in USES]]
    for g in groups:
        rows.append([Paragraph(_e(g["title"]), small), g["tier"], _fmt(g["score"])] +
                    [g["plan"]["uses"][u]["status"] if g["plan"]["uses"][u]["status"] != "gap"
                     else f"{g['plan']['uses'][u]['distance']} pts" for u in USES])
    t = Table(rows, colWidths=[W * 0.3, W * 0.07, W * 0.09] + [W * 0.135] * 4, repeatRows=1)
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    S.append(t)
    for g in groups:
        S += [PageBreak(), Paragraph(f"{_e(g['title'])} — {g['tier']} {_e(g['tier_name'])}", h2)]
        if g["story"]:
            S.append(Paragraph(f"<i>{_e(g['story'])}</i>", body))
        S.append(Paragraph(_e(g["description"]), body))
        for f in g["figures"][:2]:
            im = _img_data(f, 1100)
            if im:
                from PIL import Image as PILImage
                pim = PILImage.open(io.BytesIO(im[1]))
                h = W * pim.height / pim.width
                S.append(Image(io.BytesIO(im[1]), width=W, height=min(h, 9 * cm), kind="proportional"))
        if g["checks"]:
            S.append(Paragraph("Validation", st["Heading4"]))
            rows = [["check", "kind", "result", "detail"]] + [
                [Paragraph(_e(c["name"]), small), c.get("kind", ""), "PASS" if c["passed"] else "FAIL",
                 Paragraph(_e(c.get("detail", "")), small)] for c in g["checks"]]
            t = Table(rows, colWidths=[W * 0.32, W * 0.13, W * 0.1, W * 0.45], repeatRows=1)
            t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.grey),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            S.append(t)
        if g["plan"]["limitations"]:
            S.append(Paragraph("Limitations", st["Heading4"]))
            S += [Paragraph("• " + _e(x), small) for x in g["plan"]["limitations"]]
        if g["plan"]["actions"]:
            S.append(Paragraph("Path forward", st["Heading4"]))
            S += [Paragraph(f"{i + 1}. [{a['effort']}] {_e(a['action'])} — unlocks "
                            f"{_e(', '.join(USE_LABEL.get(u, u) for u in a['unlocks']))}", small)
                  for i, a in enumerate(g["plan"]["actions"])]
        S.append(Paragraph(f"Reproduce: {_e(g['command'])} (run {_e(g['run'])}, git {_e(g['git'])})", small))
    doc.build(S)
    return buf.getvalue()


def write_report(store, out: str, *, keys: list[str] | None = None, title: str | None = None, fmt: str | None = None,
                 filters: str = "", cur: dict[str, Any] | None = None, **select_kw) -> str:
    """Select items (``keys`` or the filters of ``select``) and write the report to ``out``; the format follows the
    extension (.html, .md, .pdf) unless ``fmt`` is given."""
    cur = cur or curate(store)
    if keys is None:
        from .spec import available, get
        tags_of = {}
        for n in available():
            try:
                tags_of[n] = list(get(n).tags)
            except KeyError:
                pass
        keys = select(cur, tags_of=tags_of, **select_kw)
    if not keys:
        raise ValueError("no item matches these filters")
    groups = [gather(store, cur, k) for k in keys]
    fmt = fmt or os.path.splitext(out)[1].lstrip(".").lower() or "html"
    title = title or (groups[0]["title"] if len(groups) == 1 else "PINNeAPPle Lab: selected experiments")
    if fmt == "pdf":
        data = render_pdf(groups, title, filters)
        with open(out, "wb") as f:
            f.write(data)
    else:
        text = render_md(groups, title, filters) if fmt == "md" else render_html(store, groups, title, filters)
        with open(out, "w") as f:
            f.write(text)
    return out
