"""U-DR-1 Compiler web API + single-page UI: datasheets and specifications in, ASME Form U-DR-1 out.

Run:  uvicorn design_requirements.api:app --port 8085   (from apps/design_requirements)

Environment (all optional):
  UDR_USER / UDR_PASSWORD   HTTP Basic login on everything except /health
  UDR_FORM_PDF              path to your copy of the fillable Form U-DR-1 (otherwise upload it with each fill)
  UDR_MAX_MB                largest upload per request in MB (default 40)
  UDR_MAX_FILES             documents per request (default 8)
  UDR_MAX_HEAVY             concurrent runs per worker (default 2)
  ANTHROPIC_API_KEY         with UDR_CLAUDE_MODEL, enables the optional Claude extraction (every value it returns is
                            checked against the document text before use)
  UDR_CLAUDE_MODEL          Claude model ID used for that extraction
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_data.udr1 import (FIELD_BY_KEY, FIELDS, SECTIONS, compile_requirements, extract_rules, fill_form,
                                  parse_value, read_document)
from pinneapple_data.udr1.fill import LEFT_FOR_ENGINEER, form_values

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.0.0"
STATIC = os.path.join(os.path.dirname(__file__), "static")
EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
EXAMPLE_FILES = [("owner_specification_V-101.pdf", "client specification"),
                 ("process_datasheet_V-101.pdf", "process datasheet"),
                 ("mechanical_datasheet_V-101.pdf", "mechanical datasheet")]
MAX_MB = float(os.environ.get("UDR_MAX_MB", "40"))
MAX_FILES = int(os.environ.get("UDR_MAX_FILES", "8"))
ROLES = ["client specification", "process datasheet", "mechanical datasheet", "project specification",
         "vendor datasheet", "other"]

app = FastAPI(title="U-DR-1 Compiler", version=VERSION,
              description="Reads process datasheets, mechanical datasheets and client specifications, finds every "
                          "item ASME Form U-DR-1 asks for with its source, reports conflicts and gaps, and fills the "
                          "official fillable form.")
install(app, prefix="UDR")
_HEAVY = BusyLimiter("UDR_MAX_HEAVY")

VALIDATION = [
    "Fictitious separator V-101 described by three documents (process datasheet, mechanical datasheet, owner "
    "specification): 27 of the 28 items a manufacturer needs are found with their page, the six-row nozzle schedule "
    "(continued on a second page) is read in full, the one planted conflict (shell corrosion allowance 6 mm in the "
    "owner's specification vs 3 mm in the mechanical datasheet) is reported, and the one planted gap (cyclic "
    "service, stated nowhere) is listed",
    "Every one of the 129 items maps to a field or check-box state that exists in the 07/25 fillable form; the "
    "filled PDF is read back field by field in the tests",
    "Values from Claude are accepted only when the quoted text is found in the document itself; the check is "
    "tested with a fabricated quote, which is rejected",
]


def scope() -> dict:
    return {"validated": VALIDATION, "title": "Scope & validation", "noun": "limitation",
            "items_title": "What to review before signing",
            "band": "Every value shows the document, page and text it came from. The engineer signs the form, not the tool.",
            "tags": {"conservative": "Safe by default", "optimistic": "Review before use", "check": "Assumption to confirm"},
            "items": [
                {"topic": "Label-based reading", "effect": "optimistic",
                 "detail": "The rules find values next to the usual labels (\"Design pressure: 15 barg\", or a table "
                           "row). A value written only in a sentence, or under an unusual label, is missed, and a "
                           "label reused for something else can be misread.",
                 "today": "Check the source text shown for each value; turn on Claude extraction for free-form specs.",
                 "planned": "Per-company label dictionaries learned from corrected forms."},
                {"topic": "Document order decides conflicts", "effect": "check",
                 "detail": "When documents disagree, the one higher in the list wins and the conflict is flagged. "
                           "Specifications often state minimums (\"6 mm minimum\") that should win over a datasheet.",
                 "today": "Order documents by authority (owner specification first) and resolve every flagged conflict.",
                 "planned": "Recognise \"minimum\"/\"maximum\" wording and apply the governing value."},
                {"topic": "Scanned PDFs", "effect": "conservative",
                 "detail": "A PDF without a text layer is reported as unreadable; nothing is guessed from it.",
                 "today": "Run OCR first, or type the missing values in the review table.",
                 "planned": "Built-in OCR."},
                {"topic": "No engineering judgement", "effect": "conservative",
                 "detail": "The tool copies stated requirements; it does not compute MAWP, choose joint types or "
                           "decide impact testing, and it leaves date, user and signature blank.",
                 "today": "Use it to compile; design and certification stay with the engineer.",
                 "planned": "Later phases: reuse the same data for other forms and Code calculations."},
            ]}


# --------------------------------------------------------------------------- helpers
async def _read(files: List[UploadFile]) -> List[Tuple[str, bytes]]:
    out = [(f.filename or f"document_{i}.pdf", await f.read()) for i, f in enumerate(files)]
    if len(out) > MAX_FILES:
        raise HTTPException(413, f"at most {MAX_FILES} documents per request")
    if sum(len(b) for _, b in out) > MAX_MB * 1e6:
        raise HTTPException(413, f"uploads are limited to {MAX_MB:g} MB per request")
    return out


def _example_docs() -> List[Tuple[str, bytes, str]]:
    return [(n, open(os.path.join(EXAMPLES, n), "rb").read(), role) for n, role in EXAMPLE_FILES]


def _claude_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    from pinneapple_data.udr1.llm import configured_model
    return bool((os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")) and configured_model())


def _parse_overrides(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"overrides is not valid JSON: {exc}")
    out = {}
    for k, v in raw.items():
        if k not in FIELD_BY_KEY:
            raise HTTPException(422, f"unknown item '{k}'")
        f = FIELD_BY_KEY[k]
        if v in (None, ""):
            continue
        if f.kind in ("bool", "choice") and v not in f.states and not (f.kind == "bool" and v == "no") \
                and not (f.key == "pwht" and v == "none"):
            raise HTTPException(422, f"'{k}' must be one of {sorted(f.states)}")
        out[k] = v
    return out


def _compile(docs_in: List[Tuple[str, bytes, str]], use_claude: bool, overrides: Dict[str, Any]):
    docs, cands, rejected = [], [], []
    for name, data, role in docs_in:
        d = read_document(data, name=name, role=role)
        docs.append(d)
        cands.extend(extract_rules(d))
        if use_claude and data.startswith(b"%PDF"):
            from pinneapple_data.udr1.llm import extract_with_claude
            try:
                acc, rej = extract_with_claude(d, data)
            except Exception as exc:  # noqa: BLE001 - the rule-based result still stands
                rej = [{"reason": f"Claude extraction failed: {type(exc).__name__}: {exc}"}]
                acc = []
            cands.extend(acc)
            rejected.extend({**r, "doc": name} for r in rej)
    comp = compile_requirements(docs, cands, overrides=overrides)
    return comp, rejected


def _payload(comp, rejected) -> Dict[str, Any]:
    out = comp.to_dict()
    out["gaps"] = [d.key for d in comp.gaps()]
    out["conflicts"] = [d.key for d in comp.conflicts()]
    out["rejected"] = rejected
    out["scope"] = scope()
    texts, buttons = form_values(comp)
    out["form_preview"] = {"text_fields": len(texts), "check_boxes": len(buttons)}
    return out


def _form_bytes(form: Optional[bytes]) -> bytes:
    if form:
        return form
    path = os.environ.get("UDR_FORM_PDF")
    if path and os.path.exists(path):
        return open(path, "rb").read()
    raise HTTPException(422, "upload your copy of the fillable Form U-DR-1 (field 'form'), or set UDR_FORM_PDF")


# --------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "sections": SECTIONS, "roles": ROLES, "max_files": MAX_FILES, "max_mb": MAX_MB,
            "claude_available": _claude_available(),
            "form_on_server": bool(os.environ.get("UDR_FORM_PDF") and os.path.exists(os.environ["UDR_FORM_PDF"])),
            "left_for_engineer": list(LEFT_FOR_ENGINEER),
            "fields": [{"key": f.key, "label": f.label, "section": f.section, "kind": f.kind,
                        "options": list(f.states) + (["none"] if f.key == "pwht" else []) if f.kind != "bool" else ["yes", "no"],
                        "required": f.required, "hint": f.source_hint} for f in FIELDS],
            "examples": [n for n, _ in EXAMPLE_FILES], "scope": scope()}


@app.post("/api/compile")
async def api_compile(files: List[UploadFile] = File(...), roles: str = Form("[]"), use_claude: bool = Form(False),
                      overrides: str = Form("")):
    """Documents in priority order (first wins conflicts) -> every item with value, status and sources."""
    data = await _read(files)
    role_list = json.loads(roles or "[]")
    with _HEAVY:
        comp, rejected = _compile([(n, b, role_list[i] if i < len(role_list) else "") for i, (n, b) in enumerate(data)],
                                  use_claude and _claude_available(), _parse_overrides(overrides))
    return _payload(comp, rejected)


@app.post("/api/fill")
async def api_fill(files: List[UploadFile] = File(...), form: Optional[UploadFile] = File(None),
                   roles: str = Form("[]"), use_claude: bool = Form(False), overrides: str = Form("")):
    """Same inputs as /api/compile plus the blank fillable form -> the filled Form U-DR-1 (PDF)."""
    data = await _read(files)
    form_bytes = _form_bytes(await form.read() if form is not None else None)
    role_list = json.loads(roles or "[]")
    with _HEAVY:
        comp, _ = _compile([(n, b, role_list[i] if i < len(role_list) else "") for i, (n, b) in enumerate(data)],
                           use_claude and _claude_available(), _parse_overrides(overrides))
        try:
            pdf = fill_form(form_bytes, comp)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    name = f"U-DR-1_{(comp.decisions['item_no'].display or 'vessel').replace(' ', '_')}_{date.today():%Y%m%d}.pdf"
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/example/compile")
def api_example_compile():
    with _HEAVY:
        comp, rejected = _compile(_example_docs(), False, {})
    return _payload(comp, rejected)


@app.post("/api/example/fill")
async def api_example_fill(form: Optional[UploadFile] = File(None), overrides: str = Form("")):
    form_bytes = _form_bytes(await form.read() if form is not None else None)
    with _HEAVY:
        comp, _ = _compile(_example_docs(), False, _parse_overrides(overrides))
        try:
            pdf = fill_form(form_bytes, comp)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="U-DR-1_V-101_example.pdf"'})


@app.get("/api/example/files/{name}")
def api_example_file(name: str):
    if name not in [n for n, _ in EXAMPLE_FILES]:
        raise HTTPException(404, "unknown example file")
    return FileResponse(os.path.join(EXAMPLES, name), media_type="application/pdf", filename=name)


@app.get("/api/parse")
def api_parse(key: str, text: str):
    """How a typed answer is interpreted (used by the review table)."""
    if key not in FIELD_BY_KEY:
        raise HTTPException(404, "unknown item")
    return {"parsed": parse_value(FIELD_BY_KEY[key], text)}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
