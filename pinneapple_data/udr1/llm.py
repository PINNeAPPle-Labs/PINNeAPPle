"""Optional extraction with Claude, for documents the label rules cannot read (free-form specs, odd layouts).

Claude reads the PDF and returns, for each U-DR-1 item it finds, the value, the page and the **exact text it read
it from**. A value is accepted only if that quote is found in the document's own text layer (whitespace and case
ignored); anything else is rejected and listed, never silently used. Accepted values go through the same parser as
the rule-based ones, so units and yes/no answers are handled identically.

Needs ``pip install anthropic``, an API key (``ANTHROPIC_API_KEY``) and the model to use, passed as ``model=`` or set
in ``UDR_CLAUDE_MODEL``. Without them the rule-based extractor works on its own.
"""
from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .catalog import FIELDS, FIELD_BY_KEY
from .extract import Candidate, Document, parse_value

__all__ = ["extract_with_claude", "verify_quote", "configured_model"]

MODEL_ENV = "UDR_CLAUDE_MODEL"


def configured_model() -> Optional[str]:
    """Model ID from ``UDR_CLAUDE_MODEL``, or ``None`` when Claude extraction is not configured."""
    return os.environ.get(MODEL_ENV) or None

_SYSTEM = (
    "You extract the data an ASME Section VIII Division 1 Form U-DR-1 (User's Design Requirements for "
    "Single-Chamber Pressure Vessels) needs, from one engineering document (process datasheet, mechanical "
    "datasheet, client specification...). Report only what the document states. For every answer give the exact "
    "text you read it from, copied verbatim from the document (a short span, normally the label and the value), "
    "and the 1-based page. Never infer a value from engineering practice or from what is typical; if the document "
    "does not state an item, leave it out. Keep units exactly as written. For yes/no items answer 'yes' or 'no'. "
    "For choice items answer with one of the listed options."
)


def _catalog_text() -> str:
    lines = []
    for f in FIELDS:
        opts = f" options: {', '.join(f.states)}" if f.kind in ("choice", "multi") else ""
        if f.kind == "bool":
            opts = " options: yes, no"
        lines.append(f"- {f.key}: {f.label} ({f.kind}{', ' + f.quantity if f.quantity else ''}){opts}")
    return "\n".join(lines)


def _schema() -> Dict[str, Any]:
    span = {"type": "object", "additionalProperties": False,
            "properties": {"key": {"type": "string", "enum": [f.key for f in FIELDS]}, "value": {"type": "string"},
                           "quote": {"type": "string"}, "page": {"type": "integer"}},
            "required": ["key", "value", "quote", "page"]}
    nozzle = {"type": "object", "additionalProperties": False,
              "properties": {k: {"type": "string"} for k in ("description", "number", "size", "flange_type", "class",
                                                             "quote")} | {"page": {"type": "integer"}},
              "required": ["description", "number", "size", "flange_type", "class", "quote", "page"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"answers": {"type": "array", "items": span}, "nozzles": {"type": "array", "items": nozzle}},
            "required": ["answers", "nozzles"]}


def _norm(s: str) -> str:
    return re.sub(r"[\s|]+", " ", s).strip().lower()


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


def extract_with_claude(doc: Document, pdf_bytes: bytes, *, client=None, model: Optional[str] = None,
                        effort: str = "medium") -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    """``(accepted candidates, rejected answers with the reason)`` for one PDF."""
    model = model or configured_model()
    if not model:
        raise ValueError(f"no Claude model given: pass model= or set {MODEL_ENV}")
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    if not doc.has_text:
        return [], [{"reason": "the PDF has no text layer (scanned); quotes cannot be verified, nothing accepted"}]
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",             # a declined request is retried server-side on Anthropic's recommended model
        system=_SYSTEM,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": _schema()}},
        messages=[{"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                            "data": base64.standard_b64encode(pdf_bytes).decode()}},
            {"type": "text", "text": "Items to look for (key: label (kind) options):\n" + _catalog_text()
                                     + "\n\nAlso list every row of a nozzle schedule if the document has one."},
        ]}],
    )
    if response.stop_reason == "refusal":
        return [], [{"reason": "the model declined this document", "details": str(getattr(response, "stop_details", ""))}]
    if response.stop_reason == "max_tokens":
        return [], [{"reason": "the answer was cut off (max_tokens); split the document and retry"}]
    text = next((b.text for b in response.content if b.type == "text"), "")
    data = json.loads(text)
    accepted: List[Candidate] = []
    rejected: List[Dict[str, Any]] = []
    for ans in data.get("answers", []):
        f = FIELD_BY_KEY.get(ans.get("key"))
        if f is None:
            rejected.append({**ans, "reason": "unknown item"})
            continue
        page = verify_quote(doc, ans.get("quote", ""), int(ans.get("page") or 0))
        if page is None:
            rejected.append({**ans, "reason": "quote not found in the document text"})
            continue
        parsed = parse_value(f, ans.get("value", ""))
        if parsed is None:
            rejected.append({**ans, "reason": f"value does not parse as {f.kind}"})
            continue
        accepted.append(Candidate(key=f.key, value=parsed["value"], raw=parsed["raw"], doc=doc.name, page=page,
                                  snippet=ans["quote"][:240], method="claude", confidence=0.75,
                                  unit=parsed.get("unit"), si=parsed.get("si"), gauge=parsed.get("gauge")))
    for noz in data.get("nozzles", []):
        page = verify_quote(doc, noz.get("quote", ""), int(noz.get("page") or 0))
        if page is None:
            rejected.append({**noz, "reason": "nozzle row quote not found in the document text"})
            continue
        value = {k: noz.get(k, "") for k in ("description", "number", "size", "flange_type", "class")}
        accepted.append(Candidate(key="nozzle", value=value, raw=noz["quote"], doc=doc.name, page=page,
                                  snippet=noz["quote"][:240], method="claude", confidence=0.75))
    return accepted, rejected
