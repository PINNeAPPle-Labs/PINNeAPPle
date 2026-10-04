"""Fill engineering forms from the documents that hold the data: read datasheets and specifications, find every item a
form asks for with its source, report conflicts between documents and missing data, and fill the fillable PDF.

The engine is generic; a ``FormSpec`` says what one form asks for. Built-in: ASME Form U-DR-1. Any other fillable
PDF works through ``FormSpec.from_pdf`` (or a JSON spec), and a local Ollama model can read what the rules miss.

>>> from pinneapple_data import formfill as ff
>>> spec = ff.get_spec("asme_u-dr-1")                       # or ff.FormSpec.from_pdf(blank) / ff.FormSpec.load(path)
>>> docs = [ff.read_document(p) for p in ("spec.pdf", "process.pdf", "mech.pdf")]      # priority order
>>> comp = ff.compile(docs, [c for d in docs for c in ff.extract_rules(d, spec)], spec)
>>> comp.gaps(), comp.conflicts(), comp.record()            # record(): the data for other forms and calculations
>>> pdf = ff.fill_pdf(open("blank.pdf", "rb").read(), comp)
"""
from .extract import (Candidate, Compilation, Decision, Document, check_answer, compile, extract_rules, parse_value,
                      read_document)
from .fill import fill_pdf, form_values, pdf_widgets, read_filled
from .llm import OllamaClient, OllamaError, extract_with_llm, verify_quote
from .spec import Field, FormSpec, TableSpec
from .specs import get_spec, list_specs

__all__ = ["Field", "FormSpec", "TableSpec", "get_spec", "list_specs", "Candidate", "Compilation", "Decision",
           "Document", "check_answer", "compile", "extract_rules", "parse_value", "read_document", "fill_pdf",
           "form_values", "pdf_widgets", "read_filled", "OllamaClient", "OllamaError", "extract_with_llm",
           "verify_quote"]
