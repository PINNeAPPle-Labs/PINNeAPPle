"""Write a compilation into the user's copy of the fillable Form U-DR-1 PDF.

Text fields get the value as written in the source (number and unit, e.g. ``15 barg``); check boxes and radio
groups get the export state from the catalog. Date, User and the signature are left for the engineer: a form that
certifies the user's requirements must be reviewed and signed by a person.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Mapping, Optional, Tuple

from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, TextStringObject

from .catalog import FIELD_BY_KEY, NOZZLE_PDF_ROWS
from .extract import Compilation

__all__ = ["form_values", "fill_form", "read_filled", "LEFT_FOR_ENGINEER"]

LEFT_FOR_ENGINEER = ("date", "user", "registration_id")
NOTES_FIELD = "GENERAL NOTESRow1"
MIN_FONT = 5.0


def _text(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _qualified_name(obj) -> str:
    """Full field name (``parent.child``). Bare child names repeat: the joint table's kids are called "0", "1", ...,
    which would otherwise collide with the form's top-level fields "1" and "2"."""
    parts = []
    while obj is not None:
        t = obj.get("/T")
        if t is not None:
            parts.append(str(t))
        nxt = obj.get("/Parent")
        obj = nxt.get_object() if nxt is not None else None
    return ".".join(reversed(parts))


def _font_size(widget, field_obj, text: str, max_size: float = 9.0) -> Optional[float]:
    """Font size that fits ``text`` on one line of the widget, or ``None`` for multi-line fields (they wrap)."""
    if int(field_obj.get("/Ff", 0) or 0) & 4096:
        return None
    x0, y0, x1, y1 = [float(v) for v in widget["/Rect"]]
    width, height = abs(x1 - x0) - 4, abs(y1 - y0)
    return round(min(max_size, height * 0.75, width / max(1, 0.52 * len(text))), 1)


def form_values(comp: Compilation) -> Tuple[Dict[str, str], Dict[str, str]]:
    """``(text fields, button states)`` for the PDF, from the decisions and the nozzle schedule."""
    texts: Dict[str, str] = {}
    buttons: Dict[str, str] = {}
    for key, d in comp.decisions.items():
        f = FIELD_BY_KEY[key]
        if d.value is None or key in LEFT_FOR_ENGINEER:
            continue
        if f.kind in ("bool", "choice"):
            state = f.states.get(str(d.value))
            if state:
                buttons[state[0]] = state[1]
        elif f.kind == "multi":
            for v in d.value if isinstance(d.value, list) else [d.value]:
                if v in f.states:
                    buttons[f.states[v][0]] = f.states[v][1]
        elif f.pdf:
            texts[f.pdf] = d.display if d.chosen is not None or d.display else _text(d.value)
    for row, cand in zip(NOZZLE_PDF_ROWS, comp.nozzles):
        for col, pdf_name in row.items():
            v = cand.value.get(col, "")
            if v:
                texts[pdf_name] = str(v)
    return texts, buttons


def fill_form(blank_pdf: bytes, comp: Compilation, extra_text: Optional[Mapping[str, str]] = None) -> bytes:
    """The filled form as PDF bytes. ``blank_pdf`` is the user's copy of the fillable Form U-DR-1."""
    texts, buttons = form_values(comp)
    texts.update(extra_text or {})
    reader = PdfReader(io.BytesIO(blank_pdf))
    known = set(reader.get_fields() or {})
    missing = sorted((set(texts) | set(buttons)) - known)
    if missing:
        raise ValueError(f"this PDF is not the fillable Form U-DR-1 this tool maps (missing fields: {missing[:5]}...)")
    writer = PdfWriter(clone_from=reader)
    widgets = []                                   # (page, widget, field object, qualified name)
    for page in writer.pages:
        for annot in page.get("/Annots") or []:
            a = annot.get_object()
            if a.get("/Subtype") != "/Widget":
                continue
            parent = a.get("/Parent")
            field_obj = parent.get_object() if (parent is not None and a.get("/T") is None) else a
            widgets.append((page, a, field_obj, _qualified_name(field_obj)))
    # Values too long for their box even at MIN_FONT go to the general notes, with "see notes" in the box.
    label_by_pdf = {f.pdf: f.label for f in FIELD_BY_KEY.values() if f.pdf}
    overflow = []
    sizes: Dict[str, float] = {}
    for _, a, fo, name in widgets:
        if name in texts and name != NOTES_FIELD:
            size = _font_size(a, fo, texts[name])
            if size is not None and size < MIN_FONT:
                overflow.append(f"{label_by_pdf.get(name, name)}: {texts[name]}")
                texts[name] = "see notes" if _font_size(a, fo, "see notes") >= MIN_FONT else "*"
                size = _font_size(a, fo, texts[name])
            if size is not None:
                sizes[name] = max(MIN_FONT, size)
    if overflow:
        notes = [texts[NOTES_FIELD]] if texts.get(NOTES_FIELD) else []
        texts[NOTES_FIELD] = "; ".join(notes + overflow)
    for page in writer.pages:
        page_texts = {}
        for pg, a, fo, name in widgets:
            if pg is not page:
                continue
            if name in texts:
                page_texts[name] = texts[name]
                if name in sizes:
                    da = TextStringObject(f"/Helv {sizes[name]:g} Tf 0 g")
                    a[NameObject("/DA")] = da
                    fo[NameObject("/DA")] = da
            elif name in buttons:
                want = buttons[name]
                ap = a.get("/AP")
                states = list(ap["/N"].keys()) if ap is not None and "/N" in ap else []
                a[NameObject("/AS")] = NameObject(want if want in states else "/Off")
                fo[NameObject("/V")] = NameObject(want)
        if page_texts:
            writer.update_page_form_field_values(page, page_texts, auto_regenerate=False)
    writer.set_need_appearances_writer(True)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def read_filled(pdf: bytes) -> Dict[str, Any]:
    """Field name -> value of a filled form (for checks and round trips)."""
    fields = PdfReader(io.BytesIO(pdf)).get_fields() or {}
    return {k: v.get("/V") for k, v in fields.items() if v.get("/V") not in (None, "", "/Off")}
