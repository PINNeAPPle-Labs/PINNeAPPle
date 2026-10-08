"""Write a compilation into the user's copy of a fillable PDF form.

Text fields get the value as written in the source (number and unit, e.g. ``15 barg``); check boxes and radio
groups get the export state from the spec. Items in ``spec.left_blank`` (date, signature...) are never filled: a
form that certifies requirements must be reviewed and signed by a person.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Mapping, Optional, Tuple

from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, TextStringObject

from .extract import Compilation
from .spec import FormSpec

__all__ = ["form_values", "fill_pdf", "read_filled", "pdf_widgets"]

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


def pdf_widgets(pdf: bytes) -> List[Dict[str, Any]]:
    """Every form widget of a PDF: qualified name, type (``/Tx``, ``/Btn``...), page, tooltip, button states and
    whether it belongs to a radio group."""
    reader = PdfReader(io.BytesIO(pdf))
    out = []
    for page_no, page in enumerate(reader.pages, start=1):
        for annot in page.get("/Annots") or []:
            a = annot.get_object()
            if a.get("/Subtype") != "/Widget":
                continue
            parent = a.get("/Parent")
            fo = parent.get_object() if (parent is not None and a.get("/T") is None) else a
            ftype = fo.get("/FT") or (fo.get("/Parent").get_object().get("/FT") if fo.get("/Parent") is not None else None)
            ap = a.get("/AP")
            states = [str(k) for k in ap["/N"].keys()] if ap is not None and "/N" in ap else []
            flags = int(fo.get("/Ff", 0) or 0)
            tip = fo.get("/TU")
            out.append({"name": _qualified_name(fo), "type": str(ftype) if ftype else None, "page": page_no,
                        "tooltip": str(tip) if tip else "", "states": states, "radio": bool(flags & 32768)})
    # a radio group's widgets share one name: merge their states
    merged: Dict[str, Dict[str, Any]] = {}
    for w in out:
        if w["name"] in merged:
            merged[w["name"]]["states"] = list(dict.fromkeys(merged[w["name"]]["states"] + w["states"]))
        else:
            merged[w["name"]] = dict(w)
    return list(merged.values())


def form_values(comp: Compilation, spec: Optional[FormSpec] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """``(text fields, button states)`` for the PDF, from the decisions and the table rows."""
    spec = spec or comp.spec
    texts: Dict[str, str] = {}
    buttons: Dict[str, str] = {}
    for key, d in comp.decisions.items():
        f = spec[key]
        if d.value is None or key in spec.left_blank:
            continue
        if f.kind in ("bool", "choice"):
            state = f.states.get(str(d.value))
            if state:
                buttons[state[0]] = state[1]
        elif f.kind == "multi":
            for v in d.value if isinstance(d.value, list) else [d.value]:
                if v in f.states:
                    buttons[f.states[v][0]] = f.states[v][1]
        if f.pdf and f.kind not in ("bool", "choice", "multi"):
            texts[f.pdf] = d.display if d.chosen is not None or d.display else _text(d.value)
        elif f.pdf and not f.states:          # a choice written as text (no check boxes for it)
            texts[f.pdf] = d.display or _text(d.value)
    for t in spec.tables:
        for row, cand in zip(t.pdf_rows, comp.tables.get(t.key, [])):
            for col, pdf_name in row.items():
                v = cand.value.get(col, "")
                if v:
                    texts[pdf_name] = str(v)
    return texts, buttons


def fill_pdf(blank_pdf: bytes, comp: Compilation, spec: Optional[FormSpec] = None,
             extra_text: Optional[Mapping[str, str]] = None) -> bytes:
    """The filled form as PDF bytes. ``blank_pdf`` is the user's copy of the fillable form ``spec`` describes."""
    spec = spec or comp.spec
    texts, buttons = form_values(comp, spec)
    texts.update(extra_text or {})
    reader = PdfReader(io.BytesIO(blank_pdf))
    known = set(reader.get_fields() or {})
    missing = sorted((set(texts) | set(buttons)) - known)
    if missing:
        raise ValueError(f"this PDF is not the fillable form '{spec.title}' (missing fields: {missing[:5]}...)")
    notes_field = spec.notes_field
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
    label_by_pdf = {f.pdf: f.label for f in spec.fields if f.pdf}
    overflow = []
    sizes: Dict[str, float] = {}
    for _, a, fo, name in widgets:
        if name in texts and name != notes_field:
            size = _font_size(a, fo, texts[name])
            if size is not None and size < MIN_FONT and notes_field:
                overflow.append(f"{label_by_pdf.get(name, name)}: {texts[name]}")
                texts[name] = "see notes" if _font_size(a, fo, "see notes") >= MIN_FONT else "*"
                size = _font_size(a, fo, texts[name])
            if size is not None:
                sizes[name] = max(MIN_FONT, size)
    if overflow:
        notes = [texts[notes_field]] if texts.get(notes_field) else []
        texts[notes_field] = "; ".join(notes + overflow)
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
