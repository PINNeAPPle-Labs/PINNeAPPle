"""Form Compiler web API + single-page UI: datasheets and specifications in, filled engineering forms out.

Built-in template: ASME Form U-DR-1. Any other fillable PDF works too: its fields become the items (template "auto"),
or upload a JSON spec (template "custom"; download a template's spec from /api/spec to start one).

Run:  uvicorn design_requirements.api:app --port 8085   (from apps/design_requirements)

Environment (all optional):
  UDR_USER / UDR_PASSWORD   HTTP Basic login on everything except /health
  UDR_FORM_PDF              path to your copy of the fillable Form U-DR-1 (otherwise upload it with each fill)
  UDR_MAX_MB                largest upload per request in MB (default 40)
  UDR_MAX_FILES             documents per request (default 8)
  UDR_MAX_HEAVY             concurrent runs per worker (default 2)
  UDR_OLLAMA_URL            Ollama server for the optional local-LLM extraction (default OLLAMA_HOST, then
                            http://localhost:11434); documents never leave your network
  UDR_OLLAMA_MODEL          model to use (pulled with `ollama pull`); the LLM option is off until this is set
  UDR_OLLAMA_TIMEOUT        seconds per request (default 600)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pinneapple_data.formfill import (FormSpec, OllamaClient, check_answer, compile, extract_rules, extract_with_llm,
                                      fill_pdf, form_values, get_spec, list_specs, parse_value, read_document)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import BusyLimiter, install  # noqa: E402

VERSION = "1.1.0"
STATIC = os.path.join(os.path.dirname(__file__), "static")
EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
EXAMPLE_FILES = [("owner_specification_V-101.pdf", "client specification"),
                 ("process_datasheet_V-101.pdf", "process datasheet"),
                 ("mechanical_datasheet_V-101.pdf", "mechanical datasheet")]
MAX_MB = float(os.environ.get("UDR_MAX_MB", "40"))
MAX_FILES = int(os.environ.get("UDR_MAX_FILES", "8"))
ROLES = ["client specification", "process datasheet", "mechanical datasheet", "project specification",
         "vendor datasheet", "other"]

DEFAULT_TEMPLATE = "asme_u-dr-1"

app = FastAPI(title="Form Compiler", version=VERSION,
              description="Reads process datasheets, mechanical datasheets and client specifications, finds every "
                          "item a form asks for with its source, reports conflicts and gaps, and fills the fillable "
                          "PDF. Built in: ASME Form U-DR-1; any other fillable PDF or a JSON spec works too.")
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
    "Values from the local LLM (Ollama) are accepted only when the quoted text is found in the document and the "
    "value is in that quote; tested with a fabricated quote and a value missing from its quote, both rejected",
    "Any fillable PDF: its fields become the items (label from the tooltip or the field name), filled and read back "
    "in the tests on a form the tool has never seen",
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
                 "today": "Check the source text shown for each value; turn on the local LLM for free-form specs.",
                 "planned": "Per-company label dictionaries learned from corrected forms."},
                {"topic": "Document order decides conflicts", "effect": "check",
                 "detail": "When documents disagree, the one higher in the list wins and the conflict is flagged. "
                           "Specifications often state minimums (\"6 mm minimum\") that should win over a datasheet.",
                 "today": "Order documents by authority (owner specification first) and resolve every flagged conflict.",
                 "planned": "Recognise \"minimum\"/\"maximum\" wording and apply the governing value."},
                {"topic": "Forms other than U-DR-1", "effect": "optimistic",
                 "detail": "An uploaded fillable PDF is read field by field: items are named after the field's tooltip "
                           "or name, so fields called \"Text7\" have no usable label, radio options are named after "
                           "their export values, and nothing is marked required.",
                 "today": "Download the generated spec, add labels, options and required items, and upload it back.",
                 "planned": "More built-in templates (datasheets, other Code forms)."},
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


_LLM_CACHE: Dict[str, Any] = {"t": 0.0, "ok": False}


def _llm() -> OllamaClient:
    return OllamaClient(url=os.environ.get("UDR_OLLAMA_URL") or None, model=os.environ.get("UDR_OLLAMA_MODEL") or None,
                        timeout=float(os.environ.get("UDR_OLLAMA_TIMEOUT", "600")))


def _llm_available() -> bool:
    """Configured, reachable and the model pulled (checked at most every 30 s)."""
    if not os.environ.get("UDR_OLLAMA_MODEL"):
        return False
    if time.time() - _LLM_CACHE["t"] > 30:
        _LLM_CACHE.update(t=time.time(), ok=_llm().available(timeout=2.0))
    return _LLM_CACHE["ok"]


def _spec(template: str, spec_json: Optional[bytes], form: Optional[bytes]) -> FormSpec:
    """The form spec for a request: a built-in template, "auto" (from the fillable PDF) or "custom" (JSON upload)."""
    template = template or DEFAULT_TEMPLATE
    if template == "custom":
        if not spec_json:
            raise HTTPException(422, "template 'custom' needs a JSON spec file (field 'spec')")
        try:
            return FormSpec.loads(spec_json.decode("utf-8"))
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, f"invalid spec: {exc}")
    if template == "auto":
        if not form:
            raise HTTPException(422, "template 'auto' needs the blank fillable PDF (field 'form')")
        try:
            return FormSpec.from_pdf(form, title="Uploaded form")
        except Exception as exc:  # noqa: BLE001 - pypdf raises many types on bad input
            raise HTTPException(422, f"cannot read the form fields of this PDF: {exc}")
    try:
        return get_spec(template)
    except KeyError as exc:
        raise HTTPException(422, str(exc))


def _parse_overrides(text: Optional[str], spec: FormSpec) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"overrides is not valid JSON: {exc}")
    out = {}
    for k, v in raw.items():
        if k not in spec.by_key:
            raise HTTPException(422, f"unknown item '{k}'")
        if v in (None, ""):
            continue
        try:
            out[k] = check_answer(spec[k], v)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    return out


def _compile(docs_in: List[Tuple[str, bytes, str]], spec: FormSpec, use_llm: bool, overrides: Dict[str, Any]):
    docs, cands, rejected = [], [], []
    client = _llm() if use_llm else None
    for name, data, role in docs_in:
        d = read_document(data, name=name, role=role)
        docs.append(d)
        cands.extend(extract_rules(d, spec))
        if client is not None:
            try:
                acc, rej = extract_with_llm(d, spec, client)
            except Exception as exc:  # noqa: BLE001 - the rule-based result still stands
                acc, rej = [], [{"reason": f"local LLM extraction failed: {type(exc).__name__}: {exc}"}]
            cands.extend(acc)
            rejected.extend({**r, "doc": name} for r in rej)
    return compile(docs, cands, spec, overrides=overrides), rejected


def _spec_meta(spec: FormSpec) -> Dict[str, Any]:
    return {"id": spec.id, "title": spec.title, "sections": spec.sections, "left_blank": list(spec.left_blank),
            "notes_field": bool(spec.notes_field),
            "fields": [{"key": f.key, "label": f.label, "section": f.section, "kind": f.kind,
                        "options": list(f.options), "required": f.required, "hint": f.source_hint,
                        "readable": bool(f.patterns)} for f in spec.fields],
            "tables": [{"key": t.key, "label": t.label, "columns": list(t.output_columns), "rows": len(t.pdf_rows)}
                       for t in spec.tables]}


def _payload(comp, rejected) -> Dict[str, Any]:
    out = comp.to_dict()
    out["gaps"] = [d.key for d in comp.gaps()]
    out["conflicts"] = [d.key for d in comp.conflicts()]
    out["rejected"] = rejected
    out["scope"] = scope()
    out["spec"] = _spec_meta(comp.spec)
    out["record"] = comp.record()
    texts, buttons = form_values(comp)
    out["form_preview"] = {"text_fields": len(texts), "check_boxes": len(buttons)}
    return out


def _filename(comp) -> str:
    spec = comp.spec
    tag = comp.decisions[spec.filename_key].display if spec.filename_key in comp.decisions else ""
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{spec.id}_{tag or 'filled'}")
    return f"{base}_{date.today():%Y%m%d}.pdf"


def _form_bytes(form: Optional[bytes], spec: FormSpec) -> bytes:
    if form:
        return form
    path = os.environ.get("UDR_FORM_PDF")
    if spec.id == DEFAULT_TEMPLATE and path and os.path.exists(path):
        return open(path, "rb").read()
    raise HTTPException(422, f"upload the blank fillable PDF of '{spec.title}' (field 'form')"
                        + (", or set UDR_FORM_PDF" if spec.id == DEFAULT_TEMPLATE else ""))


# --------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    spec = get_spec(DEFAULT_TEMPLATE)
    return {"version": VERSION, "roles": ROLES, "max_files": MAX_FILES, "max_mb": MAX_MB,
            "templates": list_specs() + [{"id": "auto", "title": "Any fillable PDF (fields read from the form)"},
                                         {"id": "custom", "title": "Custom spec (JSON)"}],
            "default_template": DEFAULT_TEMPLATE, "llm_available": _llm_available(),
            "llm_model": os.environ.get("UDR_OLLAMA_MODEL") or None,
            "form_on_server": bool(os.environ.get("UDR_FORM_PDF") and os.path.exists(os.environ["UDR_FORM_PDF"])),
            "spec": _spec_meta(spec), "examples": [n for n, _ in EXAMPLE_FILES], "scope": scope()}


@app.get("/api/spec")
def api_spec(template: str = DEFAULT_TEMPLATE):
    """A built-in template as a JSON spec: the starting point for a custom one."""
    spec = _spec(template, None, None)
    return Response(spec.dumps(), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{spec.id}.spec.json"'})


@app.post("/api/spec/from-pdf")
async def api_spec_from_pdf(form: UploadFile = File(...)):
    """The spec generated from a fillable PDF (one item per form field), to review, complete and upload back."""
    data = await form.read()
    spec = _spec("auto", None, data)
    spec.id = re.sub(r"[^a-z0-9_-]+", "_", os.path.splitext(form.filename or "form")[0].lower()) or "form"
    spec.title = form.filename or spec.title
    return Response(spec.dumps(), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{spec.id}.spec.json"'})


async def _inputs(files, roles, template, spec_file, form):
    data = await _read(files)
    form_bytes = await form.read() if form is not None else None
    spec = _spec(template, await spec_file.read() if spec_file is not None else None, form_bytes)
    role_list = json.loads(roles or "[]")
    docs = [(n, b, role_list[i] if i < len(role_list) else "") for i, (n, b) in enumerate(data)]
    return docs, spec, form_bytes


@app.post("/api/compile")
async def api_compile(files: List[UploadFile] = File(...), roles: str = Form("[]"),
                      template: str = Form(DEFAULT_TEMPLATE), spec: Optional[UploadFile] = File(None),
                      form: Optional[UploadFile] = File(None), use_llm: bool = Form(False), overrides: str = Form("")):
    """Documents in priority order (first wins conflicts) -> every item with value, status and sources."""
    docs, form_spec, _ = await _inputs(files, roles, template, spec, form)
    if use_llm and not _llm_available():
        raise HTTPException(503, "the local LLM is not available on this server (UDR_OLLAMA_MODEL / Ollama)")
    with _HEAVY:
        comp, rejected = _compile(docs, form_spec, use_llm, _parse_overrides(overrides, form_spec))
    return _payload(comp, rejected)


@app.post("/api/fill")
async def api_fill(files: List[UploadFile] = File(...), form: Optional[UploadFile] = File(None),
                   roles: str = Form("[]"), template: str = Form(DEFAULT_TEMPLATE),
                   spec: Optional[UploadFile] = File(None), use_llm: bool = Form(False), overrides: str = Form("")):
    """Same inputs as /api/compile plus the blank fillable form -> the filled PDF."""
    docs, form_spec, form_bytes = await _inputs(files, roles, template, spec, form)
    blank = _form_bytes(form_bytes, form_spec)
    if use_llm and not _llm_available():
        raise HTTPException(503, "the local LLM is not available on this server (UDR_OLLAMA_MODEL / Ollama)")
    with _HEAVY:
        comp, _ = _compile(docs, form_spec, use_llm, _parse_overrides(overrides, form_spec))
        try:
            pdf = fill_pdf(blank, comp)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{_filename(comp)}"'})


@app.get("/api/example/compile")
def api_example_compile():
    with _HEAVY:
        comp, rejected = _compile(_example_docs(), get_spec(DEFAULT_TEMPLATE), False, {})
    return _payload(comp, rejected)


@app.post("/api/example/fill")
async def api_example_fill(form: Optional[UploadFile] = File(None), overrides: str = Form("")):
    spec = get_spec(DEFAULT_TEMPLATE)
    blank = _form_bytes(await form.read() if form is not None else None, spec)
    with _HEAVY:
        comp, _ = _compile(_example_docs(), spec, False, _parse_overrides(overrides, spec))
        try:
            pdf = fill_pdf(blank, comp)
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
def api_parse(key: str, text: str, template: str = DEFAULT_TEMPLATE):
    """How a typed answer is interpreted by a built-in template."""
    spec = _spec(template, None, None)
    if key not in spec.by_key:
        raise HTTPException(404, "unknown item")
    return {"parsed": parse_value(spec[key], text)}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
