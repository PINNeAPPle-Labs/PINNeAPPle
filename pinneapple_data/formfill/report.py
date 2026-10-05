"""The compiled datasheet as a PDF (needs reportlab): every item of the spec with its value and source, the open
items (missing required data, conflicts between documents) first, side-by-side columns where the spec has them
(shell side | tube side), the table rows and the documents read.

This is the tool's own layout, marked as a draft for engineering review. It is the output for templates that have no
fillable official form, and a data compilation report for those that do.
"""
from __future__ import annotations

import io
from datetime import date
from typing import Dict, List, Optional, Tuple

from .extract import Compilation, Decision
from .spec import Field, FormSpec

__all__ = ["render_datasheet"]

INK, MUTED, LINE = "#1f2328", "#57606a", "#d0d7de"
WARN_BG, MISS_BG, HEAD_BG = "#fff4d6", "#fde7e7", "#f3f4f6"


def _source(d: Decision) -> str:
    if d.chosen is not None:
        c = d.chosen
        return (f"{c.doc} p.{c.page}" + (" (local LLM)" if c.method == "llm" else "")
                + (f" (OCR {c.ocr:.0f}%)" if c.ocr is not None else ""))
    return d.note or ""


def _value(d: Decision, f: Field) -> str:
    if d.value is None:
        return "MISSING" if f.required else "—"
    return d.display or str(d.value)


def render_datasheet(comp: Compilation, spec: Optional[FormSpec] = None, title: Optional[str] = None) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:  # pragma: no cover
        raise ImportError("render_datasheet needs reportlab: pip install reportlab") from exc
    from xml.sax.saxutils import escape

    spec = spec or comp.spec
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor(INK))
    small = ParagraphStyle("s", parent=body, fontSize=7, leading=8.5, textColor=colors.HexColor(MUTED))
    bold = ParagraphStyle("bb", parent=body, fontName="Helvetica-Bold")
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=15, leading=18, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10, leading=12, spaceBefore=8, spaceAfter=3,
                        textColor=colors.HexColor("#a14a12"))
    P = lambda t, st=body: Paragraph(escape(str(t)), st)  # noqa: E731

    tag = comp.decisions[spec.filename_key].display if spec.filename_key in comp.decisions else ""
    s = comp.summary()
    width = A4[0] - 30 * mm
    story = [Paragraph(escape(title or spec.title + (f" — {tag}" if tag else "")), h1),
             P(f"{spec.standard} · compiled {date.today():%Y-%m-%d} · DRAFT for engineering review, not the "
               "standard's official form", small), Spacer(1, 6)]

    def grid(rows, widths, styles=()):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEAD_BG)),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2), *styles]))
        return t

    band = (f"{s['required_filled']} of {s['required']} required items found · {s['filled']} of {s['fields']} items · "
            f"{s['conflicts']} conflict(s) · {s['gaps']} missing")
    story += [grid([[P(band, bold)]], [width],
                   [("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(WARN_BG if s["gaps"] or s["conflicts"]
                                                                       else "#e6f4ea"))])]

    open_rows = [[P("Open item", bold), P("Status", bold), P("Detail", bold)]]
    for d in comp.gaps():
        f = spec[d.key]
        open_rows.append([P(f.label), P("missing", bold),
                          P(f"Required; stated in no document" + (f" (usually the {f.source_hint})" if f.source_hint
                                                                  else ""))])
    for d in comp.conflicts():
        alts = "; ".join(f"{c.display()} ({c.doc} p.{c.page})" for c in d.candidates)
        open_rows.append([P(spec[d.key].label), P("conflict", bold), P(f"Using {d.display}. Sources: {alts}")])
    if len(open_rows) > 1:
        story += [Paragraph("Open items", h2), grid(open_rows, [0.3 * width, 0.12 * width, 0.58 * width])]

    sides = list(spec.columns)
    for sec in spec.sections:
        fields = [f for f in spec.fields if f.section == sec]
        if not fields:
            continue
        rows: List[list] = []
        styles = []
        if sides and any(f.column for f in fields):
            groups: Dict[str, Dict[str, Field]] = {}
            for f in fields:
                base = f.key.split("_", 1)[1] if f.column and f.key.startswith(f.column + "_") else f.key
                groups.setdefault(base, {})[f.column or ""] = f
            rows.append([P("Item", bold)] + [P(c.replace("_", " ") + " side", bold) for c in sides])
            for g in groups.values():
                any_f = next(iter(g.values()))
                label = any_f.label.rsplit(",", 1)[0] if any_f.column else any_f.label
                row = [P(label + (" *" if any(f.required for f in g.values()) else ""))]
                for c in sides:
                    f = g.get(c) or g.get("")
                    if f is None:
                        row.append(P(""))
                        continue
                    d = comp.decisions[f.key]
                    src = _source(d)
                    row.append(Paragraph(f"<b>{escape(_value(d, f))}</b><br/><font size=6 color='{MUTED}'>"
                                         f"{escape(src)}</font>", body))
                    col = len(row) - 1
                    if d.status == "conflict":
                        styles.append(("BACKGROUND", (col, len(rows)), (col, len(rows)), colors.HexColor(WARN_BG)))
                    elif d.status == "missing" and f.required:
                        styles.append(("BACKGROUND", (col, len(rows)), (col, len(rows)), colors.HexColor(MISS_BG)))
                rows.append(row)
            widths = [0.34 * width] + [0.66 * width / len(sides)] * len(sides)
        else:
            rows.append([P("Item", bold), P("Value", bold), P("Source", bold)])
            for f in fields:
                d = comp.decisions[f.key]
                rows.append([P(f.label + (" *" if f.required else "")), P(_value(d, f), bold), P(_source(d), small)])
                if d.status == "conflict":
                    styles.append(("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor(WARN_BG)))
                elif d.status == "missing" and f.required:
                    styles.append(("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor(MISS_BG)))
            widths = [0.38 * width, 0.36 * width, 0.26 * width]
        story += [Paragraph(escape(sec), h2), grid(rows, widths, styles)]

    for t in spec.tables:
        trows = comp.tables.get(t.key, [])
        rows = [[P(c.replace("_", " ").capitalize(), bold) for c in t.output_columns] + [P("Source", bold)]]
        for r in trows:
            rows.append([P(r.value.get(c, "")) for c in t.output_columns] + [P(f"{r.doc} p.{r.page}", small)])
        if len(rows) == 1:
            rows.append([P("not found in the documents", small)] + [P("")] * len(t.output_columns))
        n = len(t.output_columns) + 1
        story += [KeepTogether([Paragraph(escape(t.label), h2), grid(rows, [width / n] * n)])]

    docs = [[P("Priority", bold), P("Document", bold), P("Role", bold), P("Pages", bold)]]
    docs += [[P(d["priority"]), P(d["name"]), P(d["role"] or "—"), P(d["pages"])] for d in comp.documents]
    story += [Paragraph("Documents read (higher priority wins conflicts)", h2),
              grid(docs, [0.1 * width, 0.5 * width, 0.28 * width, 0.12 * width]), Spacer(1, 6),
              P("* required item. Values are copied from the documents listed, with their page; the engineer checks "
                "and approves the datasheet. Generated by the PINNeAPPle Form Compiler.", small)]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(15 * mm, 8 * mm, f"{spec.title} · {tag} · DRAFT")
        canvas.drawRightString(A4[0] - 15 * mm, 8 * mm, f"page {doc_.page}")
        canvas.restoreState()

    out = io.BytesIO()
    SimpleDocTemplate(out, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                      bottomMargin=14 * mm, title=f"{spec.title} {tag}".strip()).build(story, onFirstPage=footer,
                                                                                   onLaterPages=footer)
    return out.getvalue()
