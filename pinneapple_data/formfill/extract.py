"""Read datasheets and specifications, find each answer a form asks for, and keep where every value came from.

Pipeline: ``read_document`` (text per page plus the tables pdfplumber finds) -> ``extract_rules`` (the spec's label
patterns, applied to table cells and to text lines) -> optional ``llm.extract_with_llm`` (a local Ollama model) ->
``compile`` (one decision per item: value from the highest-priority document, conflicts and gaps reported).

Every candidate keeps the document, page and the exact text it was read from, so an engineer can check any value
in one click instead of trusting the tool.
"""
from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..physical_units import try_parse_unit
from .spec import Field, FormSpec, TableSpec

__all__ = ["Document", "Candidate", "Decision", "Compilation", "read_document", "extract_rules", "parse_value",
           "compile", "check_answer"]

ATM_PA = 101325.0


# --------------------------------------------------------------------------- documents
@dataclass
class Document:
    name: str
    pages: List[str]                                     # text per page (1-based page = index + 1)
    tables: List[Tuple[int, List[List[str]]]] = field(default_factory=list)   # (page, rows of cells)
    role: str = ""                                       # e.g. "process datasheet", "client specification"
    has_text: bool = True

    @property
    def n_pages(self) -> int:
        return len(self.pages)


def read_document(source: Union[str, bytes], name: str = "", role: str = "") -> Document:
    """A PDF (path or bytes) or plain text (``.txt``/``.md`` path, or bytes that are not a PDF)."""
    data = open(source, "rb").read() if isinstance(source, str) else bytes(source)
    name = name or (source if isinstance(source, str) else "document")
    if not data.startswith(b"%PDF"):
        text = data.decode("utf-8", errors="replace")
        return Document(name, [text], [], role, bool(text.strip()))
    pages: List[str] = []
    tables: List[Tuple[int, List[List[str]]]] = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                pages.append(page.extract_text() or "")
                for t in page.extract_tables() or []:
                    rows = [[(c or "").strip() for c in row] for row in t if row]
                    if rows:
                        tables.append((i, rows))
    except ImportError:  # pragma: no cover - pypdf fallback, text only
        from pypdf import PdfReader
        pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages]
    has_text = any(p.strip() for p in pages)
    return Document(name, pages, tables, role, has_text)


# --------------------------------------------------------------------------- values
@dataclass
class Candidate:
    key: str
    value: Any                     # str | float | "yes"/"no" | choice name | list of choices
    raw: str                       # value text as found
    doc: str
    page: int
    snippet: str                   # the line or table row it came from
    method: str = "rule"           # rule | llm
    confidence: float = 0.7
    unit: Optional[str] = None
    si: Optional[float] = None     # SI value for quantities (pressures gauge, Pa)
    gauge: Optional[bool] = None

    def display(self) -> str:
        if isinstance(self.value, list):
            return ", ".join(self.value)
        if isinstance(self.value, float):
            if self.unit is None and re.fullmatch(_NUM, self.raw.strip()):
                return self.raw.strip()               # keep "1.0", "0.72" as written
            v = f"{self.value:g}"
            return f"{v} {self.unit}".strip() if self.unit else v
        return str(self.value)


_NUM = r"[-+−–]?\d+(?:[.,]\d+)?(?:\s?[eE][-+]?\d+)?"
_YES = re.compile(r"^\s*(yes|y|required|true|applicable|x)\b", re.I)
_NO = re.compile(r"^\s*(no|n|not required|none|false|n/?a|not applicable|-)\s*\.?$|^\s*(no|not required)\b", re.I)


def _number(s: str) -> Optional[float]:
    m = re.search(_NUM, s)
    if not m:
        return None
    t = m.group(0).replace("−", "-").replace("–", "-").replace(" ", "")
    if t.count(",") == 1 and "." not in t:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _unit_after(s: str, start: int) -> Tuple[Optional[str], Optional[bool], Optional[float]]:
    """Unit text right after the number: returns (unit text, gauge flag, factor-to-SI or None)."""
    rest = s[start:].strip()
    m = re.match(r"(°\s?[CFcf]|deg\s?[CFcf]|[A-Za-zµ°/^0-9²³.]+(?:\s?\(\s?[ga]\s?\))?)", rest)
    if not m:
        return None, None, None
    tok = m.group(1).replace(" ", "").replace("²", "2").replace("³", "3").rstrip(".,;")
    gauge = None
    low = tok.lower()
    gm = re.match(r"^(bar|psi|kpa|mpa|kg/cm2|kgf/cm2)(\(g\)|g|ga)$", low)
    am = re.match(r"^(bar|psi|kpa|mpa)(\(a\)|a)$", low)
    if gm:
        tok, gauge = {"bar": "bar", "psi": "psi", "kpa": "kPa", "mpa": "MPa", "kg/cm2": "kgf/cm2",
                      "kgf/cm2": "kgf/cm2"}[gm.group(1)], True
    elif am:
        tok, gauge = {"bar": "bar", "psi": "psi", "kpa": "kPa", "mpa": "MPa"}[am.group(1)], False
    if low in ("mph",):
        return "mph", None, 0.44704
    u = try_parse_unit(tok)
    if u is None:
        return None, None, None
    if u.gauge:
        gauge = True
    elif gauge is None and u.dim == (-1, 1, -2, 0, 0, 0):
        gauge = False
    return m.group(1).strip(), gauge, u


def parse_value(f: Field, text: str) -> Optional[Dict[str, Any]]:
    """Interpret ``text`` as an answer to ``f``; ``None`` when it does not look like one."""
    t = text.strip().strip(":=").strip()
    if not t:
        return None
    if f.kind == "quantity":
        if re.search(r"full\s*vacuum|\bf\.?v\.?\b", t, re.I):
            return {"value": "Full vacuum", "unit": None, "si": -ATM_PA, "gauge": True, "raw": t}
        m = re.search(_NUM, t)
        if not m:
            return None
        num = _number(m.group(0))
        unit, gauge, u = _unit_after(t, m.end())
        si = None
        if u is not None:
            if isinstance(u, float):
                si = num * u
            else:
                si = float(u.to_si(num))
                if f.quantity == "pressure" and gauge is False:
                    si -= ATM_PA      # compare pressures as gauge
                    gauge = True
        return {"value": num, "unit": unit, "si": si, "gauge": gauge, "raw": t}
    if f.kind == "number":
        num = _number(t)
        return None if num is None else {"value": num, "unit": None, "si": num, "raw": t}
    if f.kind == "bool":
        if _NO.search(t):
            return {"value": "no", "raw": t}
        if _YES.search(t):
            return {"value": "yes", "raw": t}
        return None
    if f.kind in ("choice", "multi"):
        low = t.lower()
        found = [o for o in f.options
                 if re.search(f.aliases.get(o) or (r"\b" + re.escape(o.lower()) + r"s?\b"), low)]
        if f.kind == "multi":
            return {"value": found, "raw": t} if found else None
        if f.match == "first" and found:
            return {"value": found[0], "raw": t}
        if len(found) == 1:
            return {"value": found[0], "raw": t}
        if not found and f.other and len(low) > 1:
            return {"value": f.other, "raw": t}
        return None
    # text
    t = re.sub(r"\s+", " ", t)
    return {"value": t, "raw": t} if len(t) <= 300 else None


# --------------------------------------------------------------------------- rules
_RULES: Dict[int, Tuple[FormSpec, List[Tuple[Field, "re.Pattern"]]]] = {}


def _compiled(spec: FormSpec) -> List[Tuple[Field, "re.Pattern"]]:
    hit = _RULES.get(id(spec))
    if hit is None or hit[0] is not spec:
        rules = [(f, re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(f.patterns) + r")(?![A-Za-z0-9])", re.I))
                 for f in spec.fields if f.patterns]
        hit = _RULES[id(spec)] = (spec, rules)
    return hit[1]


def _label_hits(line: str, rules) -> List[Tuple[int, int, Field]]:
    """Non-overlapping label matches in a line, longest first."""
    hits = []
    for f, rx in rules:
        for m in rx.finditer(line):
            hits.append((m.start(), m.end(), f))
    hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
    kept: List[Tuple[int, int, Field]] = []
    for h in hits:
        if all(h[1] <= k[0] or h[0] >= k[1] for k in kept):
            kept.append(h)
    return sorted(kept, key=lambda h: h[0])


def _label_is_whole_cell(cell: str, start: int, end: int) -> bool:
    rest = (cell[:start] + cell[end:]).strip(" :=-–()[]*.,")
    rest = re.sub(r"\(.*?\)", "", rest).strip()
    return len(rest) <= 12           # allow a unit hint such as "(barg)" or "case 1"


def extract_rules(doc: Document, spec: FormSpec) -> List[Candidate]:
    """Candidates for every item of ``spec`` whose label appears in ``doc`` (tables first, then text lines), plus
    the rows of the spec's tables."""
    rules = _compiled(spec)
    out: List[Candidate] = []
    seen = set()

    def add(f: Field, text: str, page: int, snippet: str, conf: float):
        parsed = parse_value(f, text)
        if parsed is None:
            return
        key = (f.key, page, parsed["raw"])
        if key in seen:
            return
        seen.add(key)
        out.append(Candidate(key=f.key, value=parsed["value"], raw=parsed["raw"], doc=doc.name, page=page,
                             snippet=snippet.strip()[:240], confidence=conf, unit=parsed.get("unit"),
                             si=parsed.get("si"), gauge=parsed.get("gauge")))

    row_tables = {id(rows) for t in spec.tables for _, rows in _find_tables(doc, t)}
    norm = lambda x: re.sub(r"\s+", " ", x).strip().lower()  # noqa: E731
    table_lines = {(page, norm(" ".join(c for c in row if c))) for page, rows in doc.tables for row in rows}
    # 1. tables: label cell followed by value cell (several pairs per row allowed)
    for page, rows in doc.tables:
        if id(rows) in row_tables:
            continue
        for row in rows:
            cells = [c for c in row if c]
            i = 0
            while i < len(cells) - 1:
                hits = _label_hits(cells[i], rules)
                if len(hits) == 1 and _label_is_whole_cell(cells[i], hits[0][0], hits[0][1]):
                    unit_hint = re.search(r"\(([^)]+)\)", cells[i])
                    value = cells[i + 1]
                    if unit_hint and hits[0][2].kind == "quantity" and not re.search(r"[A-Za-z°]", value):
                        value = f"{value} {unit_hint.group(1)}"
                    add(hits[0][2], value, page, " | ".join(row), 0.9)
                    i += 2
                else:
                    i += 1
    # 2. text lines: "label: value" (several labels per line allowed). Lines that are table rows were read above,
    #    and a label without ":" or "=" in running text (a heading, a sentence) is not an answer.
    for page_no, text in enumerate(doc.pages, start=1):
        for line in text.splitlines():
            if (page_no, norm(line)) in table_lines:
                continue
            hits = _label_hits(line, rules)
            # the same field named twice in a row ("overpressure protection: pressure relief valve") is one label
            hits = [h for j, h in enumerate(hits) if j == 0 or h[2] is not hits[j - 1][2]
                    or line[hits[j - 1][1]:h[0]].strip(" :=")]
            for j, (s, e, f) in enumerate(hits):
                nxt = hits[j + 1][0] if j + 1 < len(hits) else len(line)
                tail = line[e:nxt]
                sep = re.match(r"\s*(?:\([^)]*\))?\s*[:=]", tail)
                if not sep:
                    continue
                unit_hint = re.match(r"\s*\(([^)]+)\)", tail)
                value = re.sub(r"^\s*(?:\([^)]*\))?\s*[:=]?\s*", "", tail)
                if unit_hint and f.kind == "quantity" and value and not re.search(r"[A-Za-z°]", value):
                    value = f"{value} {unit_hint.group(1)}"
                if not value.strip() and j + 1 < len(hits) and hits[j + 1][2] is f:
                    value = line[hits[j + 1][0]:(hits[j + 2][0] if j + 2 < len(hits) else len(line))]
                add(f, value, page_no, line, 0.8)
    for t in spec.tables:
        out.extend(_table_rows(doc, t))
    return out


def _is_header(row: List[str], t: TableSpec) -> bool:
    head = [c.lower() for c in row]
    return bool(t.header) and all(any(re.search(rx, h) for h in head) for rx in t.header)


def _find_tables(doc: Document, t: TableSpec) -> List[Tuple[int, List[List[str]]]]:
    """Tables whose header matches ``t``, including a header-less continuation on the next page (same columns)."""
    found, prev = [], None
    for page, rows in doc.tables:
        if rows and _is_header(rows[0], t):
            found.append((page, rows))
            prev = (page, rows[0])
        elif prev and page == prev[0] + 1 and rows and len(rows[0]) == len(prev[1]) and not _is_header(rows[0], t):
            found.append((page, [prev[1]] + rows))
            prev = None
    return found


def _table_rows(doc: Document, t: TableSpec) -> List[Candidate]:
    out = []
    for page, rows in _find_tables(doc, t):
        head = [c.lower() for c in rows[0]]
        idx = {}
        for col, rx in t.columns.items():
            idx[col] = next((i for i, h in enumerate(head) if i not in idx.values() and re.search(rx, h)), None)
        for r in rows[1:]:
            if not any(r):
                continue
            get = lambda k: (r[idx[k]] if idx.get(k) is not None and idx[k] < len(r) else "").strip()  # noqa: E731
            value = {c: get(c) for c in t.columns}
            for dst, srcs in t.combine.items():
                value[dst] = " ".join(x for x in (get(s) for s in srcs) if x)
            value = {c: value.get(c, "") or t.defaults.get(c, "") for c in t.output_columns}
            if sum(bool(get(c)) for c in t.columns) < 2:
                continue
            out.append(Candidate(key=t.key, value=value, raw=" | ".join(r), doc=doc.name, page=page,
                                 snippet=" | ".join(r)[:240], confidence=0.85))
    return out


# --------------------------------------------------------------------------- decisions
@dataclass
class Decision:
    key: str
    label: str
    section: str
    status: str                    # filled | conflict | missing | optional
    value: Any = None
    display: str = ""
    chosen: Optional[Candidate] = None
    candidates: List[Candidate] = field(default_factory=list)
    required: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class Compilation:
    decisions: Dict[str, Decision]
    tables: Dict[str, List[Candidate]]
    documents: List[Dict[str, Any]]
    spec: Optional[FormSpec] = None

    @property
    def nozzles(self) -> List[Candidate]:            # the U-DR-1 nozzle schedule
        return self.tables.get("nozzle", [])

    def gaps(self) -> List[Decision]:
        return [d for d in self.decisions.values() if d.status == "missing" and d.required]

    def conflicts(self) -> List[Decision]:
        return [d for d in self.decisions.values() if d.status == "conflict"]

    def answers(self) -> Dict[str, Any]:
        return {k: d.value for k, d in self.decisions.items() if d.value is not None}

    def summary(self) -> Dict[str, int]:
        req = [d for d in self.decisions.values() if d.required]
        return {"fields": len(self.decisions), "filled": sum(d.value is not None for d in self.decisions.values()),
                "required": len(req), "required_filled": sum(d.value is not None for d in req),
                "conflicts": len(self.conflicts()), "gaps": len(self.gaps()),
                "table_rows": sum(len(v) for v in self.tables.values()), "nozzles": len(self.nozzles)}

    def record(self) -> Dict[str, Any]:
        """The compiled data for other forms and calculations: item -> value, unit, SI value and source."""
        out = {}
        for k, d in self.decisions.items():
            if d.value is None:
                continue
            c = d.chosen
            out[k] = {"value": d.value, "display": d.display, "unit": c.unit if c else None,
                      "si": c.si if c else None, "source": f"{c.doc} p.{c.page}" if c else d.note}
        for k, rows in self.tables.items():
            out[k] = [r.value for r in rows]
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "documents": self.documents,
                "decisions": {k: d.to_dict() for k, d in self.decisions.items()},
                "tables": {k: [asdict(n) for n in v] for k, v in self.tables.items()},
                "nozzles": [asdict(n) for n in self.nozzles]}


def _same(a: Candidate, b: Candidate) -> bool:
    if a.si is not None and b.si is not None:
        return abs(a.si - b.si) <= 1e-3 * max(abs(a.si), abs(b.si), 1e-9) + 1e-9
    va, vb = a.value, b.value
    if isinstance(va, list) and isinstance(vb, list):
        return sorted(va) == sorted(vb)
    norm = lambda v: re.sub(r"[^a-z0-9.]+", "", str(v).lower())  # noqa: E731
    return norm(va) == norm(vb)


def check_answer(f: Field, value: Any) -> Any:
    """An engineer's answer for ``f``, validated (``ValueError`` names the accepted options)."""
    if f.kind in ("bool", "choice") and value not in f.options:
        raise ValueError(f"'{f.key}' must be one of {list(f.options)}")
    if f.kind == "multi":
        vals = value if isinstance(value, list) else [v.strip() for v in str(value).split(",") if v.strip()]
        bad = [v for v in vals if v not in f.options]
        if bad:
            raise ValueError(f"'{f.key}' accepts {list(f.options)}, not {bad}")
        return vals
    return value


def compile(docs: Sequence[Document], candidates: Iterable[Candidate], spec: FormSpec,
            overrides: Optional[Dict[str, Any]] = None) -> Compilation:
    """One decision per item of ``spec``. Documents earlier in ``docs`` win conflicts (order = priority); the conflict is still
    reported with every source. ``overrides`` (an engineer's answers) win over everything and are marked as such."""
    rank = {d.name: i for i, d in enumerate(docs)}
    table_keys = {t.key for t in spec.tables}
    by_key: Dict[str, List[Candidate]] = {}
    rows: Dict[str, List[Candidate]] = {k: [] for k in table_keys}
    for c in candidates:
        if c.key in table_keys:
            rows[c.key].append(c)
        elif c.key in spec.by_key:
            by_key.setdefault(c.key, []).append(c)
    decisions: Dict[str, Decision] = {}
    for f in spec.fields:
        cands = sorted(by_key.get(f.key, []), key=lambda c: (rank.get(c.doc, 99), -c.confidence, c.page))
        d = Decision(f.key, f.label, f.section, "missing", required=f.required, candidates=cands)
        if overrides and f.key in overrides and overrides[f.key] not in (None, ""):
            d.value, d.display, d.status, d.note = overrides[f.key], str(overrides[f.key]), "filled", "entered by engineer"
        elif cands:
            best = cands[0]
            d.chosen, d.value, d.display = best, best.value, best.display()
            distinct = [c for c in cands if not _same(c, best)]
            d.status = "conflict" if distinct else "filled"
            if distinct:
                d.note = (f"{len(distinct)} other value(s) disagree; using {best.doc} (higher priority). "
                          "Confirm with the owner.")
        elif not f.required:
            d.status = "optional"
        decisions[f.key] = d
    # Tables: keep the rows of the highest-priority document that has one (no merging across documents).
    for k, rs in rows.items():
        if rs:
            top = min(rank.get(n.doc, 99) for n in rs)
            rows[k] = [n for n in rs if rank.get(n.doc, 99) == top]
    docs_info = [{"name": d.name, "role": d.role, "pages": d.n_pages, "has_text": d.has_text, "priority": i + 1}
                 for i, d in enumerate(docs)]
    return Compilation(decisions, rows, docs_info, spec)
