"""Optional extraction with a local LLM served by Ollama, for documents the label rules cannot read (free-form
specifications, odd layouts, forms whose items have no usual label).

Documents never leave the machine: the text of each document (with its tables and page numbers) is sent to an
Ollama server you run (``ollama serve``), by default ``http://localhost:11434``. The model answers in a JSON schema
built from the form spec (Ollama structured outputs) with, for each item it finds, the value, the page and the
**exact text it read it from**. An answer is kept only if

* that quote is found in the document's own text (whitespace and case ignored), and
* for text, numbers and quantities, the value itself is in the quote,

so a value the model made up is rejected and listed, never silently used. Accepted values go through the same parser
as the rule-based ones (units, yes/no, options).

Configuration: ``OllamaClient(url, model)`` or the environment variables ``OLLAMA_HOST`` (server URL) and
``PINNEAPPLE_OLLAMA_MODEL`` (any model you pulled with ``ollama pull`` that supports structured outputs). Only the
Python standard library is used.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .extract import _NUM, Candidate, Document, parse_value
from .spec import FormSpec

__all__ = ["OllamaClient", "OllamaError", "extract_with_llm", "verify_quote", "document_text"]

DEFAULT_URL = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Minimal client for the Ollama HTTP API (``/api/chat``, ``/api/tags``)."""

    def __init__(self, url: Optional[str] = None, model: Optional[str] = None, timeout: float = 600.0,
                 num_ctx: int = 16384):
        url = url or os.environ.get("OLLAMA_HOST") or DEFAULT_URL
        if not re.match(r"^https?://", url):
            url = "http://" + url                  # OLLAMA_HOST is often written as host:port
        self.url = url.rstrip("/")
        self.model = model or os.environ.get("PINNEAPPLE_OLLAMA_MODEL") or None
        self.timeout = timeout
        self.num_ctx = num_ctx

    def _request(self, path: str, payload: Optional[dict] = None, timeout: Optional[float] = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.url + path, data=data, method="POST" if data else "GET",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaError(f"cannot reach Ollama at {self.url}: {exc}") from None

    def models(self, timeout: float = 3.0) -> List[str]:
        return [m.get("name", "") for m in self._request("/api/tags", timeout=timeout).get("models", [])]

    def available(self, timeout: float = 3.0) -> bool:
        """The server answers and the configured model is pulled."""
        if not self.model:
            return False
        try:
            names = self.models(timeout)
        except OllamaError:
            return False
        want = self.model if ":" in self.model else self.model + ":latest"
        return self.model in names or want in names

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        if not self.model:
            raise OllamaError("no model configured: pass model= or set PINNEAPPLE_OLLAMA_MODEL")
        out = self._request("/api/chat", {
            "model": self.model, "stream": False, "format": schema,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        content = (out.get("message") or {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise OllamaError(f"the model did not return valid JSON: {content[:200]!r}") from None


# --------------------------------------------------------------------------- prompt
def _system(spec: FormSpec) -> str:
    return (f"You extract the data needed for this form: {spec.title}. {spec.description} "
            "You read one engineering document (process datasheet, mechanical datasheet, client specification...). "
            "Report only what the document states. For every answer copy, verbatim from the document, the short text "
            "you read it from (normally the label and the value) and give its page number. Never infer a value from "
            "engineering practice or from what is typical; if the document does not state an item, leave it out. "
            "Keep numbers and units exactly as written. For yes/no items answer 'yes' or 'no'. For choice items "
            "answer with one of the listed options.")


def _catalog(spec: FormSpec) -> str:
    lines = []
    for f in spec.fields:
        opts = f" options: {', '.join(f.options)}" if f.kind in ("bool", "choice", "multi") else ""
        extra = f" - {f.description}" if f.description else ""
        lines.append(f"- {f.key}: {f.label} ({f.kind}{', ' + f.quantity if f.quantity else ''}){opts}{extra}")
    for t in spec.tables:
        lines.append(f"- table {t.key}: {t.label}, one entry per row with columns {', '.join(t.output_columns)}"
                     + (f" - {t.description}" if t.description else ""))
    return "\n".join(lines)


def _schema(spec: FormSpec) -> Dict[str, Any]:
    answer = {"type": "object",
              "properties": {"key": {"type": "string", "enum": [f.key for f in spec.fields]}, "value": {"type": "string"},
                             "quote": {"type": "string"}, "page": {"type": "integer"}},
              "required": ["key", "value", "quote", "page"]}
    props: Dict[str, Any] = {"answers": {"type": "array", "items": answer}}
    for t in spec.tables:
        cols = {c: {"type": "string"} for c in t.output_columns}
        props[t.key] = {"type": "array", "items": {
            "type": "object", "properties": {**cols, "quote": {"type": "string"}, "page": {"type": "integer"}},
            "required": [*cols, "quote", "page"]}}
    return {"type": "object", "properties": props, "required": list(props)}


def document_text(doc: Document, pages: Optional[List[int]] = None) -> str:
    """Page-marked text of a document, its tables written as ``cell | cell`` rows."""
    parts = []
    for p in pages or range(1, doc.n_pages + 1):
        parts.append(f"=== Page {p} ===\n{doc.pages[p - 1].strip()}")
        for pg, rows in doc.tables:
            if pg == p:
                parts.append("[table]\n" + "\n".join(" | ".join(r) for r in rows))
    return "\n".join(parts)


def _chunks(doc: Document, max_chars: int) -> List[List[int]]:
    """Consecutive pages grouped so that each request stays under ``max_chars`` of text."""
    groups, cur, size = [], [], 0
    for p in range(1, doc.n_pages + 1):
        n = len(document_text(doc, [p]))
        if cur and size + n > max_chars:
            groups.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += n
    return groups + ([cur] if cur else [])


# --------------------------------------------------------------------------- checks
def _norm(s: str) -> str:
    return re.sub(r"[\s|]+", " ", str(s)).strip().lower()


def verify_quote(doc: Document, quote: str, page: int) -> Optional[int]:
    """Page (1-based) where ``quote`` occurs in the document text, preferring the stated page; ``None`` if absent."""
    q = _norm(quote)
    if len(q) < 3:
        return None
    order = ([page] if 1 <= page <= doc.n_pages else []) + [p for p in range(1, doc.n_pages + 1) if p != page]
    for p in order:
        text = _norm(doc.pages[p - 1])
        table_text = " ".join(_norm(" ".join(r)) for pg, rows in doc.tables if pg == p for r in rows)
        if q in text or q in table_text:
            return p
    return None


def _value_in_quote(kind: str, value: str, quote: str) -> bool:
    q = _norm(quote)
    if kind == "text":
        return _norm(value) in q
    if kind in ("number", "quantity"):
        if re.search(r"full\s*vacuum|\bf\.?v\.?\b", value, re.I):
            return bool(re.search(r"full\s*vacuum|\bf\.?v\.?\b", q))
        m = re.search(_NUM, value)
        return bool(m) and m.group(0).replace(" ", "").lower() in q.replace(" ", "")
    return True                                  # yes/no and options are judgements on the quoted text


# --------------------------------------------------------------------------- extraction
def extract_with_llm(doc: Document, spec: FormSpec, client: Optional[OllamaClient] = None,
                     max_chars: int = 24000) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    """``(accepted candidates, rejected answers with the reason)`` for one document."""
    client = client or OllamaClient()
    if not doc.has_text:
        return [], [{"reason": "the document has no text layer (scanned); quotes cannot be verified, nothing accepted"}]
    accepted: List[Candidate] = []
    rejected: List[Dict[str, Any]] = []
    system, schema = _system(spec), _schema(spec)
    for pages in _chunks(doc, max_chars):
        user = (f"Items to look for (key: label (kind) options):\n{_catalog(spec)}\n\n"
                f"Document '{doc.name}'" + (f" ({doc.role})" if doc.role else "") + f", pages {pages[0]}-{pages[-1]}:\n"
                + document_text(doc, pages))
        data = client.chat_json(system, user, schema)
        for ans in data.get("answers") or []:
            f = spec.by_key.get(ans.get("key"))
            if f is None:
                rejected.append({**ans, "reason": "unknown item"})
                continue
            page = verify_quote(doc, ans.get("quote", ""), int(ans.get("page") or 0))
            if page is None:
                rejected.append({**ans, "reason": "quote not found in the document text"})
                continue
            if not _value_in_quote(f.kind, str(ans.get("value", "")), ans.get("quote", "")):
                rejected.append({**ans, "reason": "value is not in the quoted text"})
                continue
            parsed = parse_value(f, str(ans.get("value", "")))
            if parsed is None:
                rejected.append({**ans, "reason": f"value does not parse as {f.kind}"})
                continue
            accepted.append(Candidate(key=f.key, value=parsed["value"], raw=parsed["raw"], doc=doc.name, page=page,
                                      snippet=str(ans["quote"])[:240], method="llm", confidence=0.6,
                                      unit=parsed.get("unit"), si=parsed.get("si"), gauge=parsed.get("gauge")))
        for t in spec.tables:
            for row in data.get(t.key) or []:
                page = verify_quote(doc, row.get("quote", ""), int(row.get("page") or 0))
                if page is None:
                    rejected.append({**row, "key": t.key, "reason": "table row quote not found in the document text"})
                    continue
                value = {c: str(row.get(c, "")) or t.defaults.get(c, "") for c in t.output_columns}
                accepted.append(Candidate(key=t.key, value=value, raw=str(row["quote"]), doc=doc.name, page=page,
                                          snippet=str(row["quote"])[:240], method="llm", confidence=0.6))
    return accepted, rejected
