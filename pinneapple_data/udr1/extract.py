"""Read datasheets and specifications, find each U-DR-1 answer, and keep where every value came from.

Pipeline: ``read_document`` (text per page plus the tables pdfplumber finds) -> ``extract_rules`` (label patterns
from the catalog, applied to table cells and to text lines) -> optional ``llm.extract_with_claude`` -> ``compile``
(one decision per field: value from the highest-priority document, conflicts and gaps reported).

Every candidate keeps the document, page and the exact text it was read from, so an engineer can check any value
in one click instead of trusting the tool.
"""
from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..physical_units import try_parse_unit
from .catalog import FIELD_BY_KEY, FIELDS, NOZZLE_COLUMNS, Field

__all__ = ["Document", "Candidate", "Decision", "Compilation", "read_document", "extract_rules", "parse_value",
           "compile_requirements"]

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
    method: str = "rule"           # rule | claude
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
        if f.key == "pwht":
            if re.search(r"\bnot required\b|^\s*no\b|^\s*none\b", low):
                return {"value": "none", "raw": t}
            if re.search(r"process|service", low):
                return {"value": "process required", "raw": t}
            if re.search(r"code|required|yes", low):
                return {"value": "per code", "raw": t}
            return None
        if f.key.endswith("_basis") and f.key.startswith("mawp"):
            if re.search(r"calculat|by (?:the )?manufacturer|fabricator", low):
                return {"value": "calculated by manufacturer", "raw": t}
            if re.search(r"same as design|equal to design|= ?design", low):
                return {"value": "same as design pressure", "raw": t}
            return None
        aliases = {"asce 7": r"asce\s*7", "ibc": r"\bibc\b", "ubc": r"\bubc\b", "none": r"\bnone\b|not applicable|n/a",
                   "rupture disk": r"rupture dis[ck]|bursting dis[ck]", "valve": r"\bvalves?\b|\bpsv\b|\bprv\b|\bsrv\b",
                   "system design": r"system design", "ambient temperature": r"ambient",
                   "manufacturer": r"manufacturer|fabricator|vendor", "others": r"\bothers?\b|by client|by owner"}
        found = []
        for answer in f.states:
            pat = aliases.get(answer, r"\b" + re.escape(answer) + r"s?\b")
            if re.search(pat, low):
                found.append(answer)
        if f.kind == "choice":
            if len(found) != 1:
                if "other" in f.states and not found and len(low) > 1 and f.key in ("wind_code", "seismic_code"):
                    return {"value": "other", "raw": t}
                return None
            return {"value": found[0], "raw": t}
        return {"value": found, "raw": t} if found else None
    # text
    t = re.sub(r"\s+", " ", t)
    return {"value": t, "raw": t} if len(t) <= 300 else None


# --------------------------------------------------------------------------- rules
_COMPILED: List[Tuple[Field, re.Pattern]] = [
    (f, re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(f.patterns) + r")(?![A-Za-z0-9])", re.I))
    for f in FIELDS if f.patterns
]


def _label_hits(line: str) -> List[Tuple[int, int, Field]]:
    """Non-overlapping label matches in a line, longest first."""
    hits = []
    for f, rx in _COMPILED:
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


def extract_rules(doc: Document) -> List[Candidate]:
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

    nozzle_tables = {id(rows) for _, rows in _nozzle_tables(doc)}
    norm = lambda x: re.sub(r"\s+", " ", x).strip().lower()  # noqa: E731
    table_lines = {(page, norm(" ".join(c for c in row if c))) for page, rows in doc.tables for row in rows}
    # 1. tables: label cell followed by value cell (several pairs per row allowed)
    for page, rows in doc.tables:
        if id(rows) in nozzle_tables:
            continue
        for row in rows:
            cells = [c for c in row if c]
            i = 0
            while i < len(cells) - 1:
                hits = _label_hits(cells[i])
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
            hits = _label_hits(line)
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
    out.extend(_nozzles(doc))
    return out


def _is_nozzle_header(row: List[str]) -> bool:
    head = [c.lower() for c in row]
    return any("size" in h for h in head) and any(re.search(r"nozzle|mark|service|description", h) for h in head)


def _nozzle_tables(doc: Document) -> List[Tuple[int, List[List[str]]]]:
    """Nozzle schedule tables, including a header-less continuation on the next page (same column count)."""
    found, prev = [], None
    for page, rows in doc.tables:
        if rows and _is_nozzle_header(rows[0]):
            found.append((page, rows))
            prev = (page, rows[0])
        elif prev and page == prev[0] + 1 and rows and len(rows[0]) == len(prev[1]) and not _is_nozzle_header(rows[0]):
            found.append((page, [prev[1]] + rows))
            prev = None
    return found


def _nozzles(doc: Document) -> List[Candidate]:
    """Rows of a nozzle schedule table (header with 'size' and a description/service/mark column)."""
    out = []
    for page, rows in _nozzle_tables(doc):
        head = [c.lower() for c in rows[0]]

        def col(*names):
            for i, h in enumerate(head):
                if any(re.search(n, h) for n in names):
                    return i
            return None

        idx = {"mark": col(r"mark|nozzle\s*(?:no|id)|^no\.?$|^tag"), "description": col(r"description|service|purpose"),
               "number": col(r"qty|quantity|number|no\.? req"), "size": col(r"size"),
               "flange_type": col(r"flange|type|facing"), "class": col(r"class|rating")}
        for r in rows[1:]:
            if not any(r):
                continue
            get = lambda k: (r[idx[k]] if idx[k] is not None and idx[k] < len(r) else "").strip()  # noqa: E731
            desc = " ".join(x for x in (get("mark"), get("description")) if x)
            if not desc and not get("size"):
                continue
            value = {"description": desc, "number": get("number") or "1", "size": get("size"),
                     "flange_type": get("flange_type"), "class": get("class")}
            out.append(Candidate(key="nozzle", value=value, raw=" | ".join(r), doc=doc.name, page=page,
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
    nozzles: List[Candidate]
    documents: List[Dict[str, Any]]

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
                "conflicts": len(self.conflicts()), "gaps": len(self.gaps()), "nozzles": len(self.nozzles)}

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "documents": self.documents,
                "decisions": {k: d.to_dict() for k, d in self.decisions.items()},
                "nozzles": [asdict(n) for n in self.nozzles]}


def _same(a: Candidate, b: Candidate) -> bool:
    if a.si is not None and b.si is not None:
        return abs(a.si - b.si) <= 1e-3 * max(abs(a.si), abs(b.si), 1e-9) + 1e-9
    va, vb = a.value, b.value
    if isinstance(va, list) and isinstance(vb, list):
        return sorted(va) == sorted(vb)
    norm = lambda v: re.sub(r"[^a-z0-9.]+", "", str(v).lower())  # noqa: E731
    return norm(va) == norm(vb)


def compile_requirements(docs: Sequence[Document], candidates: Iterable[Candidate],
                         overrides: Optional[Dict[str, Any]] = None) -> Compilation:
    """One decision per field. Documents earlier in ``docs`` win conflicts (order = priority); the conflict is still
    reported with every source. ``overrides`` (an engineer's answers) win over everything and are marked as such."""
    rank = {d.name: i for i, d in enumerate(docs)}
    by_key: Dict[str, List[Candidate]] = {}
    nozzles: List[Candidate] = []
    for c in candidates:
        if c.key == "nozzle":
            nozzles.append(c)
        else:
            by_key.setdefault(c.key, []).append(c)
    decisions: Dict[str, Decision] = {}
    for f in FIELDS:
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
    # Nozzles: keep rows from the highest-priority document that has a schedule (no merging across documents).
    if nozzles:
        top = min(rank.get(n.doc, 99) for n in nozzles)
        nozzles = [n for n in nozzles if rank.get(n.doc, 99) == top]
    docs_info = [{"name": d.name, "role": d.role, "pages": d.n_pages, "has_text": d.has_text, "priority": i + 1}
                 for i, d in enumerate(docs)]
    return Compilation(decisions, nozzles, docs_info)
