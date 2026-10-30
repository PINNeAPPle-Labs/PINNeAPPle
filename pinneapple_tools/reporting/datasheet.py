"""A technical datasheet: the engineering record of a study, laid out from data -- no LLM anywhere.

A study produces numbers and figures; this module turns them into one self-contained document in the
shape a hardware datasheet has and a review meeting reads:

* a strip of headline numbers (the question the study answers, in six cells);
* numbered sections, each a table beside its figure, every figure captioned with what it shows and
  what to read from it;
* a cost table (how long each step took, on what), a run plan (done / running / queued, with the reason
  each step exists), the items to confirm, and a limitations block.

Nothing here decides content: every string, table and figure is passed in by the study.  The module
owns only layout, so a report can be regenerated from stored results without re-solving.

    from pinneapple_tools.reporting.datasheet import Datasheet, Figure, KPI, Step, Table

    d = Datasheet("3 GHz compact range", subtitle="quiet zone by physical optics", eyebrow="EM / reflector")
    d.kpis([KPI("Focal length", "1.50", "m"), KPI("Quiet zone", "0.60", "m @ 3 m")])
    s = d.section("Reflector geometry", lede="The section and its rolled edge, as modelled.")
    s.row(Table([["Focal length", "1.50 m"], ["Section", "1.20 m square"]], header=["quantity", "value"],
                caption="As modelled."),
          Figure("side.png", "Side section with the rays."))
    d.plan(done=[Step("Geometry", "built from Problem", "done", "0 s")])
    d.build("cr.html", pdf="cr.pdf")

PDF: written by printing the HTML with a headless Chromium/Chrome (``--headless --print-to-pdf``);
playwright is used when it is installed.  Without either, the HTML is still written and the call
returns it -- no PDF toolchain is required to use the module.
"""
from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = ["KPI", "Table", "Figure", "Note", "Step", "Section", "Datasheet", "find_chrome"]


# --------------------------------------------------------------------------------------------------
# Content blocks
# --------------------------------------------------------------------------------------------------
@dataclass
class KPI:
    """One cell of the headline strip: a label, its value, an optional unit and a note."""
    label: str
    value: Any
    unit: str = ""
    note: str = ""
    tone: str = ""                      # "" | "ok" | "warn" | "fail"


@dataclass
class Table:
    """A two-column (or wider) table.  ``rows`` are sequences of cells; ``header`` names the columns.

    ``emphasis`` marks rows that carry a conclusion (bold, tinted); ``note`` prints under the caption.
    """
    rows: Sequence[Sequence[Any]]
    caption: str = ""
    header: Sequence[str] | None = None
    widths: Sequence[float] | None = None
    align: Sequence[str] | None = None
    emphasis: Sequence[int] = ()
    note: str = ""


@dataclass
class Figure:
    """An image with a caption.  ``wide`` takes the full text width, otherwise it shares a row."""
    path: str
    caption: str = ""
    wide: bool = False
    max_width: str = "100%"


@dataclass
class Note:
    """A callout.  ``kind``: ``info`` (neutral), ``ok``, ``warn``, ``fail`` (a conclusion)."""
    text: str
    kind: str = "info"
    title: str = ""


@dataclass
class Step:
    """One step of the run plan: what it does, why it is in the plan, what it cost, where it stands."""
    title: str
    detail: str = ""
    status: str = "done"                 # done | running | queued | blocked | skipped
    cost: str = ""


# --------------------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------------------
_CSS = """
:root{
  --ink:#111110; --ink2:#42413d; --muted:#7c7b75; --line:#dcdbd5; --line2:#efeee9;
  --bg:#ffffff; --soft:#f7f6f2; --accent:#1c5cab; --ok:#0d7a4a; --warn:#a86a00; --fail:#b3261e;
}
*{box-sizing:border-box}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif;
  font-variant-numeric:tabular-nums}
.page{max-width:960px;margin:0 auto;padding:34px 40px 72px}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
  font-weight:600;margin:0 0 6px}
h1{font-size:30px;line-height:1.15;margin:0 0 6px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:15px;margin:0 0 4px}
.gen{color:var(--muted);font-size:11.5px;margin:12px 0 0}
.verdict{margin:16px 0 0;padding:12px 14px;border-left:3px solid var(--accent);background:var(--soft);
  font-size:13.5px;color:var(--ink2)}
.verdict.ok{border-color:var(--ok)} .verdict.warn{border-color:var(--warn)}
.verdict.fail{border-color:var(--fail)}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:0;
  border:1px solid var(--line);margin:20px 0 8px;border-radius:3px;overflow:hidden}
.strip .c{padding:11px 13px;border-right:1px solid var(--line);background:#fff}
.strip .c:last-child{border-right:0}
.strip .lab{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}
.strip .val{font-size:19px;font-weight:600;margin-top:3px;letter-spacing:-.01em;line-height:1.2}
.strip .val .u{font-size:12px;font-weight:500;color:var(--ink2);margin-left:3px}
.strip .note{font-size:10.5px;color:var(--muted);margin-top:2px;line-height:1.3}
.strip .c.ok .val{color:var(--ok)} .strip .c.warn .val{color:var(--warn)} .strip .c.fail .val{color:var(--fail)}
h2{font-size:19px;font-weight:640;margin:34px 0 4px;padding-top:16px;border-top:1px solid var(--line);
  letter-spacing:-.005em}
h2 .n{color:var(--muted);font-weight:500;margin-right:9px;font-variant-numeric:tabular-nums}
.lede{color:var(--ink2);font-size:13.5px;margin:0 0 14px;max-width:74ch}
p{margin:0 0 10px;max-width:78ch;color:var(--ink2);font-size:13.5px}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px;align-items:start}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;align-items:start}
@media (max-width:760px){.grid2,.grid3{grid-template-columns:minmax(0,1fr)}}
figure{margin:0 0 8px}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line2);border-radius:2px;background:#fff}
figcaption{font-size:11.5px;line-height:1.45;color:var(--ink2);margin-top:6px}
figcaption b{color:var(--ink);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);
  font-weight:650;padding:0 8px 6px;border-bottom:1px solid var(--ink);vertical-align:bottom}
td{padding:6px 8px;border-bottom:1px solid var(--line2);vertical-align:top;color:var(--ink2)}
td:first-child{color:var(--ink)}
tr.em td{background:var(--soft);font-weight:600;color:var(--ink)}
tr.em td:first-child{color:var(--ink)}
caption{caption-side:bottom;font-size:11.5px;color:var(--ink2);text-align:left;padding-top:7px;line-height:1.45}
caption b{color:var(--ink)}
.tnote{font-size:11px;color:var(--muted);margin-top:6px}
.num{font-variant-numeric:tabular-nums}
.callout{padding:11px 14px;border-radius:3px;font-size:12.5px;margin:0 0 12px;line-height:1.5;color:var(--ink2)}
.callout .t{font-weight:650;color:var(--ink);display:block;margin-bottom:2px;font-size:12.5px}
.callout.info{background:var(--soft);border-left:3px solid var(--muted)}
.callout.ok{background:#eef8f2;border-left:3px solid var(--ok)}
.callout.warn{background:#fdf5e7;border-left:3px solid var(--warn)}
.callout.fail{background:#fdeeed;border-left:3px solid var(--fail)}
ol.plan,ul.plain{margin:0;padding:0;list-style:none}
ol.plan li{display:grid;grid-template-columns:74px 1fr auto;gap:12px;padding:8px 0;
  border-bottom:1px solid var(--line2);font-size:12.5px;align-items:baseline}
ul.plain li{padding:6px 0;border-bottom:1px solid var(--line2);font-size:12.5px;color:var(--ink2)}
ul.plain li:last-child,ol.plan li:last-child{border-bottom:0}
.badge{font-size:9.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:2px 7px;
  border-radius:99px;text-align:center;border:1px solid}
.badge.done{color:var(--ok);border-color:#bfe0cd;background:#eff8f3}
.badge.running{color:var(--accent);border-color:#c3d8f1;background:#eef4fc}
.badge.queued{color:var(--muted);border-color:var(--line);background:var(--soft)}
.badge.blocked{color:var(--fail);border-color:#f2c9c6;background:#fdeeed}
.badge.skipped{color:var(--muted);border-color:var(--line);background:#fff}
.step-t{font-weight:600;color:var(--ink)}
.step-d{color:var(--ink2);font-size:12px;margin-top:1px}
.cost{color:var(--muted);font-size:11.5px;white-space:nowrap;font-variant-numeric:tabular-nums}
.math{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;background:var(--soft);
  border:1px solid var(--line2);border-radius:3px;padding:9px 12px;color:var(--ink);overflow-x:auto;
  margin:0 0 12px;white-space:pre-wrap}
.foot{margin-top:44px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);font-size:10.5px;
  display:flex;justify-content:space-between;gap:16px}
@media print{
  .page{max-width:none;padding:0}
  h2{break-after:avoid}
  figure,table,.callout{break-inside:avoid}
  .pagebreak{break-before:page}
}
@page{size:A4;margin:16mm 15mm 16mm}
"""


def _b64_image(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
            "svg": "image/svg+xml", "webp": "image/webp"}.get(ext, "image/png")
    with open(path, "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode()


def _e(x: Any) -> str:
    return _glyph(_html.escape("" if x is None else str(x)))


#: Inline tokens a study can use in any caption or cell without writing HTML.  Applied **after** escaping,
#: so ``$\\pm$`` and ``&deg;`` both render as the glyph instead of leaking markup into the page.
_GLYPHS = (
    ("$\\pm$", "±"), ("$\\lambda^2$", "λ²"), ("$\\lambda$", "λ"), ("$\\mu$", "µ"), ("$\\eta$", "η"),
    ("$\\rho$", "ρ"), ("$\\theta$", "θ"), ("$\\psi$", "ψ"), ("$\\Delta$", "Δ"),
    ("$\\epsilon$", "ε"), ("$\\times$", "×"), ("$\\approx$", "≈"), ("$\\leq$", "≤"),
    ("$\\geq$", "≥"), ("$\\sqrt{2}$", "√2"), ("$^2$", "²"), ("$^3$", "³"),
    ("$^q$", "<sup>q</sup>"), ("$_0$", "<sub>0</sub>"), ("$_x$", "<sub>x</sub>"),
    ("&amp;deg;", "°"), ("&amp;Delta;", "Δ"), ("&amp;middot;", "·"),
    ("&amp;plusmn;", "±"), ("&amp;lambda;", "λ"), ("&amp;micro;", "µ"),
    ("&amp;times;", "×"), ("&amp;le;", "≤"), ("&amp;ge;", "≥"), ("&amp;ndash;", "–"),
)


def _glyph(s: str) -> str:
    for token, glyph in _GLYPHS:
        s = s.replace(token, glyph)
    return s


def _rich(x: Any) -> str:
    """Escape, allow ``**bold**``, and translate the inline tokens of :data:`_GLYPHS`."""
    s = _e(x)
    while "**" in s:
        s = s.replace("**", "<b>", 1)
        if "**" in s:
            s = s.replace("**", "</b>", 1)
    return s


def find_chrome() -> str | None:
    """A Chromium/Chrome executable that can print HTML to PDF, or ``None``.

    Order: ``$PINNEAPPLE_CHROME``, ``google-chrome`` / ``chromium`` / ``chromium-browser`` /
    ``microsoft-edge`` on PATH, the macOS app bundles, then playwright's browser cache.
    """
    env = os.environ.get("PINNEAPPLE_CHROME")
    if env and os.path.exists(env):
        return env
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome",
                 "microsoft-edge", "msedge", "headless_shell"):
        p = shutil.which(name)
        if p:
            return p
    mac = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
           "/Applications/Chromium.app/Contents/MacOS/Chromium",
           "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    for p in mac:
        if os.path.exists(p):
            return p
    cache = os.path.expanduser("~/Library/Caches/ms-playwright")
    if os.path.isdir(cache):
        for d in sorted(os.listdir(cache), reverse=True):
            for rel in ("chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
                        "chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
                        "chrome-linux/chrome", "chrome-headless-shell-mac-arm64/chrome-headless-shell"):
                p = os.path.join(cache, d, rel)
                if os.path.exists(p):
                    return p
    return None


# --------------------------------------------------------------------------------------------------
# Sections and the document
# --------------------------------------------------------------------------------------------------
@dataclass
class Section:
    """One numbered section: a lede, then blocks (paragraphs, callouts, tables, figures, rows)."""
    title: str
    number: int = 0
    lede: str = ""
    blocks: list[dict[str, Any]] = field(default_factory=list)

    # -- blocks -------------------------------------------------------------------------------
    def p(self, text: str) -> Section:
        self.blocks.append({"k": "p", "text": text})
        return self

    def note(self, text: str, kind: str = "info", title: str = "") -> Section:
        self.blocks.append({"k": "note", "text": text, "kind": kind, "title": title})
        return self

    def table(self, t: Table) -> Section:
        self.blocks.append({"k": "table", "t": t})
        return self

    def figure(self, f: Figure | str, caption: str = "", wide: bool = False) -> Section:
        f = Figure(f, caption, wide) if isinstance(f, str) else f
        self.blocks.append({"k": "figure", "f": f})
        return self

    def row(self, *items: Table | Figure) -> Section:
        """Lay blocks side by side (2 or 3 across)."""
        self.blocks.append({"k": "row", "items": list(items)})
        return self

    def math(self, text: str) -> Section:
        self.blocks.append({"k": "math", "text": text})
        return self

    def bullets(self, items: Sequence[str]) -> Section:
        self.blocks.append({"k": "bullets", "items": list(items)})
        return self

    def pagebreak(self) -> Section:
        self.blocks.append({"k": "pagebreak"})
        return self

    # -- rendering ----------------------------------------------------------------------------
    def _render_block(self, b: dict[str, Any]) -> str:
        k = b["k"]
        if k == "p":
            return f"<p>{_rich(b['text'])}</p>\n"
        if k == "note":
            t = f"<span class='t'>{_rich(b['title'])}</span>" if b["title"] else ""
            return f"<div class='callout {b['kind']}'>{t}{_rich(b['text'])}</div>\n"
        if k == "math":
            return f"<pre class='math'>{_e(b['text'])}</pre>\n"
        if k == "bullets":
            li = "".join(f"<li>{_rich(x)}</li>" for x in b["items"])
            return f"<ul class='plain'>{li}</ul>\n"
        if k == "pagebreak":
            return "<div class='pagebreak'></div>\n"
        if k == "table":
            return self._render_table(b["t"])
        if k == "figure":
            return self._render_figure(b["f"])
        if k == "row":
            n = max(2, min(3, len(b["items"])))
            inner = "".join(self._render_table(i) if isinstance(i, Table) else self._render_figure(i)
                            for i in b["items"])
            return f"<div class='grid{n}'>{inner}</div>\n"
        raise ValueError(k)

    @staticmethod
    def _render_table(t: Table) -> str:
        head = ""
        if t.header:
            head = "<thead><tr>" + "".join(
                f"<th{' style=\"text-align:' + (t.align[i] if t.align else 'left') + '\"' if t.align else ''}>"
                f"{_e(h)}</th>" for i, h in enumerate(t.header)) + "</tr></thead>"
        cols = max([len(r) for r in t.rows] + [len(t.header or ())])
        styles = ""
        if t.widths:
            styles = "<colgroup>" + "".join(f"<col style='width:{w * 100:.1f}%'>" for w in t.widths) + "</colgroup>"
        body = []
        for i, r in enumerate(t.rows):
            em = " class='em'" if i in t.emphasis else ""
            cells = "".join(
                f"<td{' style=\"text-align:' + t.align[j] + '\"' if t.align and j < len(t.align) else ''}>"
                f"{_rich(c)}</td>" for j, c in enumerate(r))
            cells += "<td></td>" * max(0, cols - len(r))
            body.append(f"<tr{em}>{cells}</tr>")
        cap = f"<caption>{_rich(t.caption)}</caption>" if t.caption else ""
        note = f"<div class='tnote'>{_rich(t.note)}</div>" if t.note else ""
        return (f"<div><table>{styles}{head}<tbody>{''.join(body)}</tbody>{cap}</table>{note}</div>\n")

    @staticmethod
    def _render_figure(f: Figure) -> str:
        if not os.path.exists(f.path):
            return f"<figure><figcaption><b>missing figure</b> {_e(f.path)}</figcaption></figure>"
        cap = f"<figcaption>{_rich(f.caption)}</figcaption>" if f.caption else ""
        return (f"<figure><img src='{_b64_image(f.path)}' alt='' "
                f"style='max-width:{f.max_width}'>{cap}</figure>\n")

    def render(self) -> str:
        lede = f"<p class='lede'>{_rich(self.lede)}</p>\n" if self.lede else ""
        body = "".join(self._render_block(b) for b in self.blocks)
        return (f"<h2><span class='n'>{self.number:02d}</span>{_e(self.title)}</h2>\n{lede}"
                f"<div class='sec'>{body}</div>\n")


@dataclass
class Datasheet:
    """A technical datasheet: headline strip, numbered sections, run plan, items to confirm, footer."""
    title: str
    subtitle: str = ""
    eyebrow: str = ""
    generated: str = ""
    verified_by: str = ""
    kpis_: list[KPI] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    plan_: list[Step] = field(default_factory=list)
    confirm_: list[str] = field(default_factory=list)
    limitations_: list[str] = field(default_factory=list)
    footer_left: str = "PINNeAPPle"
    verdict: Note | None = None

    def __post_init__(self):
        if not self.generated:
            self.generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # -- content ------------------------------------------------------------------------------
    def kpis(self, items: Sequence[KPI]) -> Datasheet:
        self.kpis_ = list(items)
        return self

    def section(self, title: str, lede: str = "") -> Section:
        s = Section(title, number=len(self.sections) + 1, lede=lede)
        self.sections.append(s)
        return s

    def verdict_box(self, text: str, kind: str = "info", title: str = "") -> Datasheet:
        self.verdict = Note(text, kind, title)
        return self

    def plan(self, *, done: Sequence[Step | tuple] = (), running: Sequence[Step | tuple] = (),
             queued: Sequence[Step | tuple] = (), blocked: Sequence[Step | tuple] = ()) -> Datasheet:
        def mk(items, status):
            out = []
            for it in items:
                if isinstance(it, Step):
                    it.status = status if it.status == "done" and status != "done" else it.status
                    out.append(it)
                else:
                    out.append(Step(title=it[0], detail=it[1] if len(it) > 1 else "",
                                    cost=it[2] if len(it) > 2 else "", status=status))
            return out
        self.plan_ = mk(done, "done") + mk(running, "running") + mk(queued, "queued") + mk(blocked, "blocked")
        return self

    def confirm(self, items: Sequence[str]) -> Datasheet:
        self.confirm_ = list(items)
        return self

    def limitations(self, items: Sequence[str]) -> Datasheet:
        self.limitations_ = list(items)
        return self

    # -- output -------------------------------------------------------------------------------
    def _strip(self) -> str:
        if not self.kpis_:
            return ""
        cells = []
        for k in self.kpis_:
            u = f"<span class='u'>{_e(k.unit)}</span>" if k.unit else ""
            n = f"<div class='note'>{_rich(k.note)}</div>" if k.note else ""
            cells.append(f"<div class='c {_e(k.tone)}'><div class='lab'>{_e(k.label)}</div>"
                         f"<div class='val'>{_rich(k.value)}{u}</div>{n}</div>")
        return f"<div class='strip'>{''.join(cells)}</div>\n"

    def _plan_html(self) -> str:
        if not self.plan_:
            return ""
        li = []
        for s in self.plan_:
            cost = f"<span class='cost'>{_e(s.cost)}</span>" if s.cost else "<span></span>"
            li.append(f"<li><span class='badge {_e(s.status)}'>{_e(s.status)}</span>"
                      f"<span><span class='step-t'>{_e(s.title)}</span>"
                      f"<div class='step-d'>{_rich(s.detail)}</div></span>{cost}</li>")
        return f"<ol class='plan'>{''.join(li)}</ol>\n"

    def to_html(self, path: str | None = None) -> str:
        head = (f"<p class='eyebrow'>{_e(self.eyebrow)}</p>" if self.eyebrow else "")
        sub = f"<p class='sub'>{_rich(self.subtitle)}</p>" if self.subtitle else ""
        ver = (f"<div class='verdict {_e(self.verdict.kind)}'>"
               f"{'<b>' + _rich(self.verdict.title) + '</b> ' if self.verdict.title else ''}"
               f"{_rich(self.verdict.text)}</div>") if self.verdict else ""
        gen = (f"<p class='gen'>Generated {_e(self.generated)}"
               f"{' &middot; ' + _rich(self.verified_by) if self.verified_by else ''}</p>")
        parts = [head, f"<h1>{_e(self.title)}</h1>", sub, ver, gen, self._strip()]
        parts += [s.render() for s in self.sections]
        if self.plan_:
            parts.append("<h2><span class='n'>--</span>Run plan</h2>"
                         "<p class='lede'>Why each step is in the study, what it cost, and where it stands.</p>"
                         + self._plan_html())
        if self.confirm_:
            parts.append("<h2><span class='n'>--</span>Items to confirm</h2><ul class='plain'>"
                         + "".join(f"<li>{_rich(x)}</li>" for x in self.confirm_) + "</ul>")
        if self.limitations_:
            parts.append("<h2><span class='n'>--</span>Limitations</h2><ul class='plain'>"
                         + "".join(f"<li>{_rich(x)}</li>" for x in self.limitations_) + "</ul>")
        parts.append(f"<div class='foot'><span>{_rich(self.footer_left)}</span>"
                     f"<span>{_e(self.generated)}</span></div>")
        doc = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
               "<meta name='viewport' content='width=device-width,initial-scale=1'>"
               f"<title>{_e(self.title)}</title><style>{_CSS}</style></head><body>"
               f"<div class='page'>{''.join(parts)}</div></body></html>")
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(doc)
        return doc

    def to_pdf(self, path: str, *, html_path: str | None = None) -> str | None:
        """Print the datasheet to PDF with a headless Chromium; returns ``path`` or ``None`` if no
        browser is available (the HTML is still written next to it)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        src = html_path or os.path.splitext(path)[0] + ".html"
        self.to_html(src)
        try:                                                    # playwright, when present
            from playwright.sync_api import sync_playwright      # type: ignore
            with sync_playwright() as p:
                b = p.chromium.launch()
                pg = b.new_page()
                pg.goto(os.path.abspath(src), wait_until="networkidle")
                pg.pdf(path=path, format="A4", print_background=True, prefer_css_page_size=True)
                b.close()
            return path
        except Exception:
            pass
        chrome = find_chrome()
        if not chrome:
            return None
        tmp = tempfile.mkdtemp(prefix="ppds-")
        try:
            cmd = [chrome, "--headless", "--disable-gpu", "--no-sandbox", "--no-first-run",
                   "--disable-extensions", "--virtual-time-budget=4000",
                   "--run-all-compositor-stages-before-draw",
                   f"--user-data-dir={tmp}", "--no-pdf-header-footer",
                   f"--print-to-pdf={os.path.abspath(path)}", os.path.abspath(src)]
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        except Exception:
            return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return path if os.path.exists(path) else None

    def to_markdown(self, path: str | None = None) -> str:
        out = [f"# {self.title}", "", self.subtitle, ""]
        if self.kpis_:
            out += ["| " + " | ".join(k.label for k in self.kpis_) + " |",
                    "|" + "---|" * len(self.kpis_),
                    "| " + " | ".join(f"{k.value} {k.unit}".strip() for k in self.kpis_) + " |", ""]
        for s in self.sections:
            out += [f"## {s.number:02d} {s.title}", "", s.lede, ""]
            for b in s.blocks:
                items = b.get("items") if b["k"] == "row" else [b.get("t") or b.get("f")]
                for it in items:
                    if isinstance(it, Table):
                        out.append("| " + " | ".join(str(c) for c in it.rows[0]) + " |")
                        out.append("|" + "---|" * len(it.rows[0]))
                        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in it.rows[1:]]
                        out += ["", it.caption, ""]
                    elif isinstance(it, Figure):
                        out += [f"![{it.caption}]({os.path.basename(it.path)})", "", it.caption, ""]
                if b["k"] == "p":
                    out += [b["text"], ""]
                if b["k"] == "bullets":
                    out += [f"- {x}" for x in b["items"]] + [""]
        doc = "\n".join(out)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(doc)
        return doc

    def build(self, path: str, *, pdf: str | None = None) -> dict[str, str | None]:
        """Write the HTML (and the PDF when a browser is found).  ``path`` is the HTML."""
        html_written = path
        self.to_html(html_written)
        pdf_written = self.to_pdf(pdf, html_path=html_written) if pdf else None
        return {"html": html_written, "pdf": pdf_written}
