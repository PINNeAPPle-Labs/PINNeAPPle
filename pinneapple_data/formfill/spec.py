"""What a form asks for, how documents phrase each item, and where each answer goes in the fillable PDF.

A ``FormSpec`` is data, not code: the same engine reads documents, compiles answers and fills the PDF for any spec.
Specs come from three places:

* built-in templates (``pinneapple_data.formfill.specs``, e.g. ASME Form U-DR-1),
* a JSON file (``FormSpec.load``): copy a template or a generated spec, add label patterns, mark required items,
* any fillable PDF (``FormSpec.from_pdf``): one item per AcroForm field, labelled from its tooltip or name. Rules find
  the items whose label is written in the documents; the local LLM finds the rest.

Field kinds: ``text``, ``quantity`` (number + unit, compared in SI), ``number``, ``bool`` (yes/no), ``choice`` (one of
``options``), ``multi`` (several of ``options``).
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Field", "TableSpec", "FormSpec"]

KINDS = ("text", "quantity", "number", "bool", "choice", "multi")


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    section: str = "Form"
    kind: str = "text"
    pdf: Optional[str] = None                # text field (qualified AcroForm name)
    states: Dict[str, Tuple[str, str]] = field(default_factory=dict)   # answer -> (button field, export state)
    patterns: Tuple[str, ...] = ()           # how documents label the item (regex, case-insensitive)
    quantity: Optional[str] = None           # pressure | temperature | length | speed | density | time | ...
    required: bool = False
    source_hint: str = ""                    # which document usually states it
    options: Tuple[str, ...] = ()            # answers for bool/choice/multi (default: the state names)
    aliases: Dict[str, str] = field(default_factory=dict)   # answer -> regex that recognises it in a document
    match: str = "unique"                    # choice: "unique" (exactly one option named) | "first" (first in order)
    other: Optional[str] = None              # choice answer used for any other non-empty text
    description: str = ""                    # extra context for the LLM

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"field {self.key!r}: kind must be one of {KINDS}, got {self.kind!r}")
        if not self.options:
            opts = ("yes", "no") if self.kind == "bool" else tuple(self.states)
            object.__setattr__(self, "options", opts)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["states"] = {k: list(v) for k, v in self.states.items()}
        d["patterns"] = list(self.patterns)
        d["options"] = list(self.options)
        return {k: v for k, v in d.items() if v not in (None, "", (), [], {}) or k in ("key", "label")} | {"kind": self.kind}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Field":
        d = dict(d)
        d["states"] = {k: tuple(v) for k, v in (d.get("states") or {}).items()}
        d["patterns"] = tuple(d.get("patterns") or ())
        d["options"] = tuple(d.get("options") or ())
        return cls(**d)


@dataclass(frozen=True)
class TableSpec:
    """A repeated-row item (nozzle schedule, line list...): found as a document table whose header matches ``header``
    (every regex must match some header cell), filled into ``pdf_rows`` (one dict column -> PDF field per row)."""
    key: str
    label: str
    columns: Dict[str, str]                  # column -> regex matching its header cell
    header: Tuple[str, ...] = ()
    combine: Dict[str, Tuple[str, ...]] = field(default_factory=dict)  # output column <- joined source columns
    defaults: Dict[str, str] = field(default_factory=dict)
    pdf_rows: Tuple[Dict[str, str], ...] = ()
    description: str = ""

    @property
    def output_columns(self) -> Tuple[str, ...]:
        hidden = {c for srcs in self.combine.values() for c in srcs if c not in self.combine}
        return tuple(c for c in self.columns if c not in hidden) + tuple(c for c in self.combine if c not in self.columns)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["header"], d["pdf_rows"] = list(self.header), list(self.pdf_rows)
        d["combine"] = {k: list(v) for k, v in self.combine.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TableSpec":
        d = dict(d)
        d["header"] = tuple(d.get("header") or ())
        d["pdf_rows"] = tuple(d.get("pdf_rows") or ())
        d["combine"] = {k: tuple(v) for k, v in (d.get("combine") or {}).items()}
        return cls(**d)


@dataclass
class FormSpec:
    id: str
    title: str
    fields: List[Field]
    sections: List[str] = field(default_factory=list)
    tables: List[TableSpec] = field(default_factory=list)
    description: str = ""                    # what the form is for (context for the LLM)
    notes_field: Optional[str] = None        # PDF field that receives values too long for their box
    left_blank: Tuple[str, ...] = ()         # keys never filled automatically (date, signature...)
    filename_key: Optional[str] = None       # item used to name the filled PDF (e.g. the tag number)
    aliases: Dict[str, str] = field(default_factory=dict)   # option -> regex, shared by every choice field

    def __post_init__(self):
        keys = [f.key for f in self.fields]
        dup = {k for k in keys if keys.count(k) > 1}
        if dup:
            raise ValueError(f"duplicate field keys: {sorted(dup)}")
        if self.aliases:
            self.fields = [replace(f, aliases={**{o: self.aliases[o] for o in f.options if o in self.aliases},
                                               **f.aliases}) if f.kind in ("choice", "multi") else f
                           for f in self.fields]
        if not self.sections:
            self.sections = list(dict.fromkeys(f.section for f in self.fields))
        self.by_key: Dict[str, Field] = {f.key: f for f in self.fields}

    def __getitem__(self, key: str) -> Field:
        return self.by_key[key]

    def pdf_names(self) -> List[str]:
        names = [f.pdf for f in self.fields if f.pdf]
        names += [s[0] for f in self.fields for s in f.states.values()]
        names += [n for t in self.tables for row in t.pdf_rows for n in row.values()]
        return list(dict.fromkeys(names))

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "description": self.description, "sections": self.sections,
                "notes_field": self.notes_field, "left_blank": list(self.left_blank), "filename_key": self.filename_key,
                "fields": [f.to_dict() for f in self.fields], "tables": [t.to_dict() for t in self.tables]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FormSpec":
        if not isinstance(d, dict) or "fields" not in d:
            raise ValueError("a form spec is a JSON object with at least 'id', 'title' and 'fields'")
        return cls(id=str(d.get("id") or "custom"), title=str(d.get("title") or d.get("id") or "Custom form"),
                   fields=[Field.from_dict(f) for f in d["fields"]], sections=list(d.get("sections") or []),
                   tables=[TableSpec.from_dict(t) for t in d.get("tables") or []],
                   description=d.get("description") or "", notes_field=d.get("notes_field"),
                   left_blank=tuple(d.get("left_blank") or ()), filename_key=d.get("filename_key"),
                   aliases=dict(d.get("aliases") or {}))

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=1, ensure_ascii=False)

    @classmethod
    def loads(cls, text: str) -> "FormSpec":
        return cls.from_dict(json.loads(text))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.dumps())

    @classmethod
    def load(cls, path: str) -> "FormSpec":
        with open(path, encoding="utf-8") as fh:
            return cls.loads(fh.read())

    # ------------------------------------------------------------------ any fillable PDF
    @classmethod
    def from_pdf(cls, blank_pdf: bytes, id: str = "auto", title: str = "") -> "FormSpec":
        """One item per AcroForm field of ``blank_pdf``. Text fields become ``text`` items, a check box ``bool``, a radio
        group ``choice`` (its export states as options). The label is the field's tooltip (``/TU``) or its name made
        readable; items with a readable label also get a label pattern, so the rules can find them."""
        from .fill import pdf_widgets
        widgets = pdf_widgets(blank_pdf)
        if not widgets:
            raise ValueError("this PDF has no fillable form fields")
        seen, fields = set(), []
        for w in widgets:
            name = w["name"]
            if name in seen or w["type"] not in ("/Tx", "/Btn"):
                continue
            seen.add(name)
            label = _clean(w["tooltip"]) if w["tooltip"] else _readable(name)
            key = _key(label if _meaningful(label) else name, {f.key for f in fields})
            pats = (_label_pattern(label),) if _meaningful(label) else ()
            section = f"Page {w['page']}"
            if w["type"] == "/Tx":
                fields.append(Field(key, label, section, pdf=name, patterns=pats))
                continue
            on = [s for s in w["states"] if s != "/Off"]
            if w["radio"] and len(on) > 1:
                states = {_readable(s.lstrip("/")).lower(): (name, s) for s in on}
                fields.append(Field(key, label, section, kind="choice", states=states, patterns=pats))
            elif on:
                fields.append(Field(key, label, section, kind="bool", states={"yes": (name, on[0])}, patterns=pats))
        return cls(id=id, title=title or "Fillable form", fields=fields,
                   description="A fillable PDF form; items are named after its fields.")


_GENERIC = re.compile(r"^(?:text|group|check ?box|undefined|field|radio|button|choice|row|untitled)[\s_.-]*\d*$", re.I)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .:") or text


def _readable(name: str) -> str:
    s = name.split(".")[-1] if "." in name and not name.split(".")[-1].isdigit() else name
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s.replace("_", " "))
    s = re.sub(r"\s+", " ", s).strip(" .:")
    return s or name


def _meaningful(label: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", label)
    return len(letters) >= 3 and not _GENERIC.match(label.strip())


def _key(label: str, taken) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:48] or "field"
    key, i = base, 2
    while key in taken:
        key, i = f"{base}_{i}", i + 1
    return key


def _label_pattern(label: str) -> str:
    words = [re.escape(w) for w in re.findall(r"[A-Za-z0-9]+", label) if not re.fullmatch(r"row\d*|\d+", w, re.I)]
    return r"[\s,./()-]*".join(words) if words else re.escape(label)
